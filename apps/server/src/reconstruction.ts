import { createHash } from "node:crypto";
import { createReadStream, readFileSync } from "node:fs";
import { relative, resolve } from "node:path";
import {
  type Case,
  type CaptureSetPlan,
  type EvidenceAsset,
  type PipelineEvent,
  type PipelineStageName,
  type MultiReconstructionRequest,
  type ReconstructionArtifact,
  type ReconstructionGeometry,
  type ReconstructionQuality,
  type ReconstructionRequest,
  type StageStatus,
} from "@structurefirst/contracts";
import type { AppConfig } from "./config.js";
import { CaseEventHub } from "./events.js";
import { confidence } from "./lib/confidence.js";
import { createId, nowIso } from "./lib/ids.js";
import { planCaptureSet } from "./capture-plan.js";
import { StructureStore } from "./store.js";

type WorkerJob = {
  job_id: string;
  status: "queued" | "running" | "ready" | "failed";
  splat_url?: string;
  structural_model_url?: string;
  manifest_url?: string;
  gaussian_count?: number;
  registration_report_url?: string;
  registration_status?: "connected" | "partial" | "failed";
  connected_frame_count?: number;
  frame_count?: number;
  registration_confidence?: number;
  fallback_used?: boolean;
  fallback_reason?: string;
  error?: string;
};

type WorkerHealth = {
  status?: string;
  gpu_available?: boolean;
  lucidframe_available?: boolean;
  sharp_checkpoint_verified?: boolean;
  runtime_error?: string | null;
};

export class ReconstructionCoordinator {
  private readonly pollers = new Set<string>();

  constructor(
    private readonly store: StructureStore,
    private readonly events: CaseEventHub,
    private readonly config: AppConfig,
    private readonly onReady?: (
      artifact: ReconstructionArtifact,
    ) => void | Promise<void>,
  ) {}

  async health(): Promise<{ reachable: boolean; details?: WorkerHealth }> {
    try {
      const response = await fetch(`${this.config.reconstructionUrl}/health`, {
        signal: AbortSignal.timeout(30_000),
      });
      if (!response.ok) return { reachable: false };
      return {
        reachable: true,
        details: (await response.json()) as WorkerHealth,
      };
    } catch {
      return { reachable: false };
    }
  }

  async detectFrame(input: {
    imageDataUrl: string;
    scoreThreshold?: number | undefined;
  }): Promise<unknown> {
    const response = await fetch(`${this.config.reconstructionUrl}/detect-frame`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        image_data_url: input.imageDataUrl,
        score_threshold: input.scoreThreshold ?? 0.34,
      }),
      signal: AbortSignal.timeout(15_000),
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(
        `Local detector returned ${response.status}: ${detail.slice(0, 300)}`,
      );
    }
    return response.json();
  }

  async queueAvailable(
    caseId: string,
    preferredEvidenceIds?: string[],
  ): Promise<ReconstructionArtifact | undefined> {
    const existing = this.store
      .listArtifacts(caseId)
      .find((item) => ["queued", "running", "ready"].includes(item.status));
    if (existing) return existing;

    const preferred = preferredEvidenceIds
      ? new Set(preferredEvidenceIds)
      : undefined;
    const photos = this.store
      .listEvidence(caseId)
      .filter(
        (item) =>
          (!preferred || preferred.has(item.id)) &&
          Boolean(item.localUrl) &&
          item.mimeType?.startsWith("image/") &&
          isReconstructionEligible(item) &&
          !item.tags.includes("reconstruction-excluded"),
      )
      .sort((left, right) => evidencePriority(right) - evidencePriority(left))
      .slice(0, 12);
    if (photos.length === 0) return undefined;
    if (photos.length >= 2) {
      return this.queueMulti(caseId, {
        evidenceIds: photos.map((item) => item.id),
      });
    }
    return this.queue(caseId, {
      evidenceId: photos[0]!.id,
      mode: "single_image",
    });
  }

  async queueCaptureSet(
    caseId: string,
    evidenceIds: string[],
  ): Promise<{
    artifacts: ReconstructionArtifact[];
    plan: CaptureSetPlan;
  }> {
    const selected = new Set(evidenceIds);
    const evidence = this.store
      .listEvidence(caseId)
      .filter((item) => selected.has(item.id));
    const plan = planCaptureSet(evidence);
    const artifacts: ReconstructionArtifact[] = [];
    for (const batch of plan.batches) {
      if (batch.evidenceIds.length === 1) {
        artifacts.push(
          await this.queue(caseId, {
            evidenceId: batch.evidenceIds[0]!,
            mode: "single_image",
          }),
        );
      } else {
        artifacts.push(
          await this.queueMulti(caseId, { evidenceIds: batch.evidenceIds }),
        );
      }
    }
    return { artifacts, plan };
  }

  async queueFloorPlans(
    caseId: string,
    evidenceIds: string[],
  ): Promise<ReconstructionArtifact> {
    const uniqueIds = [...new Set(evidenceIds)];
    if (uniqueIds.length === 0 || uniqueIds.length > 12)
      throw new Error("Select between one and twelve floorplans.");
    const evidence = uniqueIds
      .map((id) => this.store.getEvidence(id))
      .filter((item): item is EvidenceAsset => Boolean(item));
    if (
      evidence.length !== uniqueIds.length ||
      evidence.some(
        (item) =>
          item.caseId !== caseId || !item.mimeType?.startsWith("image/"),
      )
    ) {
      throw new Error("Every floorplan must be a local image in this case.");
    }
    const ordered = [...evidence].sort((left, right) => {
      const leftFloor = left.visualAnalysis?.floorNumber;
      const rightFloor = right.visualAnalysis?.floorNumber;
      if (leftFloor !== undefined && rightFloor !== undefined)
        return leftFloor - rightFloor;
      if (leftFloor !== undefined) return -1;
      if (rightFloor !== undefined) return 1;
      return (
        (left.capture?.captureOrder ?? 0) - (right.capture?.captureOrder ?? 0)
      );
    });
    const inputPaths = ordered.map((item) => this.localInputPath(item));
    if (inputPaths.some((path) => !path))
      throw new Error("Every floorplan must be locally available.");
    const verifiedPaths = inputPaths as string[];
    const inputSha256s = await Promise.all(
      ordered.map(
        async (item, index) =>
          item.sha256 ?? (await sha256File(verifiedPaths[index]!)),
      ),
    );
    const id = createId("artifact");
    const now = nowIso();
    const artifact: ReconstructionArtifact = {
      id,
      caseId,
      evidenceId: ordered[0]!.id,
      evidenceIds: ordered.map((item) => item.id),
      status: "queued",
      mode: "floorplan",
      modelName: "StructureFirst plan vectorizer",
      modelLicense: "StructureFirst Apache-2.0 code; operator-supplied plans",
      createdAt: now,
      updatedAt: now,
      confidence: confidence(
        0,
        "unknown",
        "unknown",
        "Floorplan lines have not been vectorized yet.",
        ordered.length,
      ),
    };
    this.store.putArtifact(artifact);
    this.updateWorkflow(
      caseId,
      "reconstruction",
      "running",
      `Vectorizing walls and floor plates from ${ordered.length} floorplan ${ordered.length === 1 ? "image" : "images"}.`,
    );
    try {
      const response = await fetch(`${this.config.reconstructionUrl}/jobs`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          job_id: id,
          case_id: caseId,
          evidence_id: ordered[0]!.id,
          evidence_ids: ordered.map((item) => item.id),
          input_path: verifiedPaths[0],
          input_paths: verifiedPaths,
          input_sha256: inputSha256s[0],
          input_sha256s: inputSha256s,
          floor_numbers: ordered.map(
            (item, index) => item.visualAnalysis?.floorNumber ?? index + 1,
          ),
          mode: "floorplan",
        }),
        signal: AbortSignal.timeout(10_000),
      });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(
          `Structural worker returned ${response.status}: ${message.slice(0, 300)}`,
        );
      }
      const updated = this.store.putArtifact({
        ...artifact,
        status: "running",
        updatedAt: nowIso(),
      });
      this.schedulePoll(updated.id, 0);
      return updated;
    } catch (error) {
      const failed = this.store.putArtifact({
        ...artifact,
        status: "failed",
        error: errorMessage(error),
        updatedAt: nowIso(),
      });
      this.updateWorkflow(
        caseId,
        "reconstruction",
        "failed",
        `Floorplan vectorization could not start: ${errorMessage(error)}`,
      );
      return failed;
    }
  }

  reportWorkflow(caseId: string, status: StageStatus, message: string): void {
    this.updateWorkflow(caseId, "reconstruction", status, message);
  }

  async queue(
    caseId: string,
    request: ReconstructionRequest,
  ): Promise<ReconstructionArtifact> {
    const evidence = this.store.getEvidence(request.evidenceId);
    if (!evidence || evidence.caseId !== caseId)
      throw new Error("Evidence was not found in this case.");
    const inputPath = this.localInputPath(evidence);
    if (!inputPath) {
      throw new Error(
        "Reconstruction requires an operator upload or a locally permitted asset.",
      );
    }
    if (!evidence.mimeType?.startsWith("image/")) {
      throw new Error(
        "LucidFrame reconstruction currently accepts image evidence only.",
      );
    }
    const inputSha256 = evidence.sha256 ?? (await sha256File(inputPath));

    const id = createId("artifact");
    const now = nowIso();
    const artifact: ReconstructionArtifact = {
      id,
      caseId,
      evidenceId: evidence.id,
      evidenceIds: [evidence.id],
      status: "queued",
      mode: request.mode,
      modelName:
        request.mode === "panorama"
          ? "LucidFrame SHARP-360"
          : "LucidFrame Apple SHARP",
      modelLicense: "Apple SHARP research-only, noncommercial model license",
      createdAt: now,
      updatedAt: now,
      confidence: confidence(
        0,
        "unknown",
        "unknown",
        "Reconstruction has not completed.",
        1,
      ),
    };
    this.store.putArtifact(artifact);
    this.updateWorkflow(
      caseId,
      "reconstruction",
      "running",
      "LucidFrame reconstruction queued on the local GPU.",
    );

    try {
      const response = await fetch(`${this.config.reconstructionUrl}/jobs`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          job_id: id,
          case_id: caseId,
          evidence_id: evidence.id,
          input_path: inputPath,
          input_sha256: inputSha256,
          mode: request.mode,
        }),
        signal: AbortSignal.timeout(10_000),
      });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(
          `Reconstruction worker returned ${response.status}: ${message.slice(0, 300)}`,
        );
      }
      const updated = this.store.putArtifact({
        ...artifact,
        status: "running",
        updatedAt: nowIso(),
      });
      this.schedulePoll(updated.id, 0);
      return updated;
    } catch (error) {
      const failed = this.store.putArtifact({
        ...artifact,
        status: "failed",
        error: errorMessage(error),
        updatedAt: nowIso(),
      });
      this.updateWorkflow(
        caseId,
        "reconstruction",
        "failed",
        `Reconstruction could not start: ${errorMessage(error)}`,
      );
      return failed;
    }
  }

  async queueMulti(
    caseId: string,
    request: MultiReconstructionRequest,
  ): Promise<ReconstructionArtifact> {
    const evidenceIds = [...new Set(request.evidenceIds)];
    if (evidenceIds.length < 2)
      throw new Error("Select at least two different images to connect.");
    const evidence = evidenceIds.map((id) => this.store.getEvidence(id));
    if (
      evidence.some(
        (item) =>
          !item ||
          item.caseId !== caseId ||
          !item.mimeType?.startsWith("image/"),
      )
    ) {
      throw new Error(
        "Every selected item must be local image evidence in this case.",
      );
    }
    const inputPaths = evidence.map((item) =>
      this.localInputPath(item as EvidenceAsset),
    );
    if (inputPaths.some((path) => !path))
      throw new Error("Every selected image must be locally available.");
    const localEvidence = evidence as EvidenceAsset[];
    const verifiedInputPaths = inputPaths as string[];
    const inputSha256s = await Promise.all(
      localEvidence.map(
        async (item, index) =>
          item.sha256 ?? (await sha256File(verifiedInputPaths[index]!)),
      ),
    );

    const id = createId("artifact");
    const now = nowIso();
    const artifact: ReconstructionArtifact = {
      id,
      caseId,
      evidenceId: evidenceIds[0] as string,
      evidenceIds,
      status: "queued",
      mode: "multi_image",
      modelName: "LucidFrame SHARP smart connect",
      modelLicense: "Apple SHARP research-only, noncommercial model license",
      createdAt: now,
      updatedAt: now,
      confidence: confidence(
        0,
        "unknown",
        "unknown",
        "Photo overlap has not been registered yet.",
        evidenceIds.length,
      ),
    };
    this.store.putArtifact(artifact);
    this.updateWorkflow(
      caseId,
      "reconstruction",
      "running",
      `Registering ${evidenceIds.length} LucidFrame captures by measured visual overlap.`,
    );
    try {
      const response = await fetch(`${this.config.reconstructionUrl}/jobs`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          job_id: id,
          case_id: caseId,
          evidence_id: evidenceIds[0],
          evidence_ids: evidenceIds,
          input_path: verifiedInputPaths[0],
          input_paths: verifiedInputPaths,
          input_sha256: inputSha256s[0],
          input_sha256s: inputSha256s,
          mode: "multi_image",
        }),
        signal: AbortSignal.timeout(10_000),
      });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(
          `Reconstruction worker returned ${response.status}: ${message.slice(0, 300)}`,
        );
      }
      const updated = this.store.putArtifact({
        ...artifact,
        status: "running",
        updatedAt: nowIso(),
      });
      this.schedulePoll(updated.id, 0);
      return updated;
    } catch (error) {
      const failed = this.store.putArtifact({
        ...artifact,
        status: "failed",
        error: errorMessage(error),
        updatedAt: nowIso(),
      });
      this.updateWorkflow(
        caseId,
        "reconstruction",
        "failed",
        `Smart connect could not start: ${errorMessage(error)}`,
      );
      return failed;
    }
  }

  resumePending(): void {
    for (const caseValue of this.store.listCases()) {
      const artifacts = this.store.listArtifacts(caseValue.id);
      const pending = artifacts.filter(
        (artifact) =>
          artifact.status === "queued" || artifact.status === "running",
      );
      for (const artifact of pending) {
        if (artifact.status === "queued" || artifact.status === "running") {
          this.schedulePoll(artifact.id, 0);
        }
      }
      if (pending.length) continue;
      const ready = artifacts.filter((artifact) => artifact.status === "ready");
      if (!ready.length) continue;
      const failed = artifacts.filter(
        (artifact) => artifact.status === "failed",
      );
      const gaussianLayers = ready.filter(
        (artifact) => artifact.mode !== "floorplan",
      ).length;
      const planLayers = ready.length - gaussianLayers;
      this.updateWorkflow(
        caseValue.id,
        "reconstruction",
        failed.length ? "limited" : "complete",
        `${gaussianLayers} verified Gaussian ${
          gaussianLayers === 1 ? "scene is" : "scenes are"
        } ready${planLayers ? ` with ${planLayers} structural plan layer` : ""}${
          planLayers === 1 ? "" : planLayers > 1 ? "s" : ""
        }.${
          failed.length
            ? ` ${failed.length} disconnected capture ${failed.length === 1 ? "group was" : "groups were"} left unplaced.`
            : ""
        }`,
      );
    }
  }

  private schedulePoll(artifactId: string, attempt: number): void {
    if (this.pollers.has(artifactId) && attempt === 0) return;
    this.pollers.add(artifactId);
    const delay = attempt === 0 ? 500 : Math.min(10_000, 2_000 + attempt * 250);
    const timer = setTimeout(() => {
      timer.unref();
      void this.poll(artifactId, attempt);
    }, delay);
    timer.unref();
  }

  private async poll(artifactId: string, attempt: number): Promise<void> {
    const artifact = this.store.getArtifact(artifactId);
    if (
      !artifact ||
      artifact.status === "ready" ||
      artifact.status === "failed"
    ) {
      this.pollers.delete(artifactId);
      return;
    }
    if (attempt > 1_200) {
      await this.failArtifact(
        artifact,
        "Reconstruction exceeded the one-hour monitoring window.",
      );
      return;
    }

    try {
      const response = await fetch(
        `${this.config.reconstructionUrl}/jobs/${encodeURIComponent(artifactId)}`,
        {
          signal: AbortSignal.timeout(5_000),
        },
      );
      if (response.status === 404 && attempt < 5) {
        this.schedulePoll(artifactId, attempt + 1);
        return;
      }
      if (!response.ok)
        throw new Error(`Worker status returned ${response.status}.`);
      const job = (await response.json()) as WorkerJob;
      if (job.status === "failed") {
        await this.failArtifact(
          artifact,
          job.error ?? "LucidFrame reconstruction failed.",
          job,
        );
        return;
      }
      if (job.status !== "ready") {
        if (artifact.status !== "running") {
          this.store.putArtifact({
            ...artifact,
            status: "running",
            updatedAt: nowIso(),
          });
        }
        this.schedulePoll(artifactId, attempt + 1);
        return;
      }
      if (!job.manifest_url)
        throw new Error("Worker completed without a manifest URL.");
      if (artifact.mode === "floorplan" && !job.structural_model_url)
        throw new Error("Worker completed without a structural model URL.");
      if (artifact.mode !== "floorplan" && !job.splat_url)
        throw new Error("Worker completed without a splat URL.");
      const registrationReport = job.registration_report_url
        ? readRegistrationReport(
            job.registration_report_url,
            this.config.casesRoot,
          )
        : undefined;
      const geometry = reconstructionGeometry(
        artifact,
        job,
        registrationReport,
        this.store,
      );
      const quality = reconstructionQuality(artifact, job, registrationReport);

      const ready: ReconstructionArtifact = {
        ...artifact,
        status: "ready",
        ...(job.splat_url ? { splatUrl: job.splat_url } : {}),
        ...(job.structural_model_url
          ? { structuralModelUrl: job.structural_model_url }
          : {}),
        manifestUrl: job.manifest_url,
        ...(job.gaussian_count ? { gaussianCount: job.gaussian_count } : {}),
        ...(artifact.mode === "floorplan"
          ? {
              geometry: {
                backend: "floorplan_vector_extrusion" as const,
                coordinateFrame: "floorplan_metric_y_up" as const,
                jointCameraAccepted: false,
                cameraPoses: [],
              },
            }
          : geometry
            ? { geometry }
            : {}),
        ...(quality ? { quality } : {}),
        ...(artifact.mode === "multi_image" && job.registration_status
          ? {
              registration: {
                status: job.registration_status,
                method: "sift_loftr_sharp_pose_graph" as const,
                frameCount:
                  job.frame_count ?? artifact.evidenceIds?.length ?? 1,
                connectedFrameCount: job.connected_frame_count ?? 1,
                confidenceScore: job.registration_confidence ?? 0,
                ...(job.registration_report_url
                  ? { reportUrl: job.registration_report_url }
                  : {}),
                note:
                  job.registration_status === "connected"
                    ? "All selected photographs joined the measured overlap graph."
                    : "Only photographs with verified overlap were merged; disconnected frames were excluded.",
              },
            }
          : {}),
        ...(job.fallback_used
          ? {
              fallback: {
                mode: "single_image" as const,
                sourceEvidenceId: artifact.evidenceId,
                reason:
                  job.fallback_reason ??
                  "The selected photographs did not form a verified overlap graph.",
              },
            }
          : {}),
        updatedAt: nowIso(),
        confidence: confidence(
          artifact.mode === "floorplan"
            ? 0.64
            : job.fallback_used
              ? 0.52
              : artifact.mode === "multi_image"
                ? Math.min(
                    0.68,
                    0.35 + (job.registration_confidence ?? 0) * 0.33,
                  )
                : 0.58,
          "reconstructed",
          "derived",
          artifact.mode === "floorplan"
            ? "Arbitrary-angle wall candidates and room polygons were extracted from the exact supplied plan ink. Scale, wall height, openings, and stairs remain unverified; unobserved roof geometry is omitted."
            : job.fallback_used
              ? "The photographs did not register, so LucidFrame reconstructed a nearby view from the first exact source image. Occluded space remains unknown."
              : artifact.mode === "multi_image"
                ? "LucidFrame reconstructed and registered only photographs with measured visual and metric overlap. Occluded space remains unknown."
                : "LucidFrame reconstructed nearby appearance from the selected image. Occluded and unseen space remains unknown.",
          artifact.mode === "floorplan"
            ? (artifact.evidenceIds?.length ?? 1)
            : job.fallback_used
              ? 1
              : (job.connected_frame_count ??
                artifact.evidenceIds?.length ??
                1),
        ),
      };
      this.store.putArtifact(ready);
      this.pollers.delete(artifactId);
      await this.applyReadyCoverage(ready);
    } catch (error) {
      if (attempt >= 10) {
        await this.failArtifact(
          artifact,
          `Worker became unreachable: ${errorMessage(error)}`,
        );
      } else {
        this.schedulePoll(artifactId, attempt + 1);
      }
    }
  }

  private async applyReadyCoverage(
    artifact: ReconstructionArtifact,
  ): Promise<void> {
    const evidence = this.store.getEvidence(artifact.evidenceId);
    const current = this.store.getCase(artifact.caseId);
    if (!evidence || !current) return;
    const groupedEvidence = (artifact.evidenceIds ?? [artifact.evidenceId])
      .map((id) => this.store.getEvidence(id))
      .filter((item): item is EvidenceAsset => Boolean(item));
    const isInterior =
      artifact.mode === "floorplan" ||
      groupedEvidence.some(
        (item) =>
          item.tags.includes("interior") ||
          item.tags.includes("scene:interior"),
      );
    const sourceCount = artifact.fallback
      ? 1
      : (artifact.registration?.connectedFrameCount ?? 1);
    const coverage = {
      ...current.coverage,
      ...(isInterior
        ? {
            interior: confidence(
              artifact.mode === "floorplan"
                ? 0.64
                : artifact.mode === "multi_image" && !artifact.fallback
                  ? 0.56
                  : 0.42,
              "reconstructed",
              "derived",
              artifact.mode === "floorplan"
                ? `${groupedEvidence.length} supplied plan sheets were classified and vectorized; multi-level sheets are separated, site plans are references rather than floors, and height, scale, openings, stairs, and inter-floor alignment remain unverified.`
                : artifact.mode === "multi_image" && !artifact.fallback
                  ? `${sourceCount} overlapping interior captures were registered; room topology beyond observed overlap remains unknown.`
                  : "A nearby-view splat exists for one interior image; room topology and occluded space remain unknown.",
              sourceCount,
            ),
          }
        : {
            exterior: confidence(
              artifact.mode === "multi_image" && !artifact.fallback
                ? 0.62
                : 0.58,
              "reconstructed",
              "derived",
              artifact.mode === "multi_image" && !artifact.fallback
                ? `${sourceCount} overlapping exterior captures were registered; unseen elevations remain unknown.`
                : "A nearby-view splat exists for one exterior image; unseen elevations remain unknown.",
              sourceCount,
            ),
          }),
      lastAssessedAt: nowIso(),
    };
    this.store.putCase({
      ...current,
      status: "briefing_ready",
      coverage,
      updatedAt: nowIso(),
    });
    this.updateWorkflow(
      artifact.caseId,
      "reconstruction",
      "complete",
      artifact.mode === "floorplan"
        ? `Built a navigable structural model from ${groupedEvidence.length} supplied plan ${groupedEvidence.length === 1 ? "sheet" : "sheets"}. Floor count comes from labels inside the sheets; site plans remain reference evidence.`
        : artifact.fallback
          ? `The photographs did not register; LucidFrame still produced ${artifact.gaussianCount?.toLocaleString() ?? "a"} Gaussians from the first exact source image.`
          : artifact.mode === "multi_image"
            ? `LucidFrame connected ${sourceCount}/${artifact.evidenceIds?.length ?? sourceCount} captures and produced ${artifact.gaussianCount?.toLocaleString() ?? "a"} Gaussians.`
            : `LucidFrame produced ${artifact.gaussianCount?.toLocaleString() ?? "a"} Gaussian scene. Nearby-view limitations apply.`,
    );
    await this.onReady?.(artifact);
    await this.queueRemainingCoverage(artifact, false);
  }

  private async queueRemainingCoverage(
    artifact: ReconstructionArtifact,
    includeEveryInput: boolean,
  ): Promise<void> {
    if (artifact.mode !== "multi_image" || !artifact.evidenceIds) return;
    const reportUrl = artifact.registration?.reportUrl;
    if (!reportUrl) return;
    const report = readRegistrationReport(reportUrl, this.config.casesRoot);
    const frameGroups = coverageFrameGroups(
      report,
      artifact.evidenceIds.length,
      includeEveryInput,
    );
    const covered = representedEvidenceIds(
      this.store.listArtifacts(artifact.caseId),
    );
    for (const frameGroup of frameGroups) {
      const evidenceIds = frameGroup
        .map((index) => artifact.evidenceIds?.[index])
        .filter((id): id is string => Boolean(id))
        .filter((id) => {
          const evidence = this.store.getEvidence(id);
          return Boolean(
            evidence && !evidence.tags.includes("reconstruction-excluded"),
          );
        })
        .filter((id) => !covered.has(id));
      if (evidenceIds.length === 0) continue;
      for (const id of evidenceIds) covered.add(id);
      if (evidenceIds.length === 1) {
        await this.queue(artifact.caseId, {
          evidenceId: evidenceIds[0]!,
          mode: "single_image",
        });
      } else {
        await this.queueMulti(artifact.caseId, { evidenceIds });
      }
    }
  }

  private async failArtifact(
    artifact: ReconstructionArtifact,
    message: string,
    job?: WorkerJob,
  ): Promise<void> {
    const failed = this.store.putArtifact({
      ...artifact,
      status: "failed",
      error: message,
      ...(artifact.mode === "multi_image" && job?.registration_status
        ? {
            registration: {
              status: job.registration_status,
              method: "sift_loftr_sharp_pose_graph" as const,
              frameCount: job.frame_count ?? artifact.evidenceIds?.length ?? 1,
              connectedFrameCount: job.connected_frame_count ?? 1,
              confidenceScore: job.registration_confidence ?? 0,
              ...(job.registration_report_url
                ? { reportUrl: job.registration_report_url }
                : {}),
              note: "The selected captures did not form a verified overlap graph. Capture adjacent views with 60–80% overlap.",
            },
          }
        : {}),
      updatedAt: nowIso(),
    });
    this.pollers.delete(artifact.id);
    const hasReadyLayer = this.store
      .listArtifacts(artifact.caseId)
      .some((candidate) => candidate.status === "ready");
    this.updateWorkflow(
      artifact.caseId,
      "reconstruction",
      hasReadyLayer ? "limited" : "failed",
      hasReadyLayer
        ? `One capture group was not reconstructed: ${message} Other verified plan or Gaussian layers remain available.`
        : message,
    );
    if (failed.registration?.reportUrl) {
      await this.queueRemainingCoverage(failed, true);
    }
  }

  private localInputPath(evidence: EvidenceAsset): string | undefined {
    if (!evidence.localUrl?.startsWith("/assets/")) return undefined;
    const suffix = decodeURIComponent(
      evidence.localUrl.slice("/assets/".length),
    );
    const candidate = resolve(this.config.casesRoot, suffix);
    const traversal = relative(this.config.casesRoot, candidate);
    if (traversal.startsWith("..") || resolve(traversal) === traversal)
      return undefined;
    const firstSegment = suffix.split(/[\\/]/)[0];
    return firstSegment === evidence.caseId ? candidate : undefined;
  }

  private updateWorkflow(
    caseId: string,
    stageName: PipelineStageName,
    status: StageStatus,
    message: string,
  ): void {
    const current = this.store.getCase(caseId);
    if (!current) return;
    const timestamp = nowIso();
    const updatedCase: Case = {
      ...current,
      status: status === "running" ? "reconstructing" : current.status,
      updatedAt: timestamp,
      stages: current.stages.map((stage) =>
        stage.name === stageName
          ? {
              ...stage,
              status,
              message,
              ...(status === "running"
                ? { startedAt: stage.startedAt ?? timestamp }
                : {}),
              ...(["complete", "limited", "skipped", "failed"].includes(status)
                ? { completedAt: timestamp }
                : {}),
            }
          : stage,
      ),
    };
    this.store.putCase(updatedCase);
    const event: PipelineEvent = {
      id: createId("event"),
      caseId,
      stage: stageName,
      status,
      message,
      createdAt: timestamp,
    };
    this.store.putEvent(event);
    this.events.publish(event);
  }
}

async function sha256File(path: string): Promise<string> {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest("hex");
}

function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "Unknown reconstruction error";
}

function evidencePriority(evidence: EvidenceAsset): number {
  let score = 0;
  if (evidence.tags.includes("operator-upload")) score += 20;
  if (evidence.tags.includes("property-photo")) score += 10;
  if (evidence.visualAnalysis?.propertyRelevance === "likely") score += 8;
  if (evidence.visualAnalysis?.addressMatch === "supported") score += 12;
  if (evidence.tags.includes("address-text-match")) score += 10;
  if (evidence.visualAnalysis?.addressMatch === "contradictory") score -= 50;
  if (evidence.tags.some((tag) => tag.startsWith("overlap-set:"))) score += 3;
  if (evidence.tags.includes("property-proximity-unverified")) score -= 8;
  return score;
}

function isReconstructionEligible(evidence: EvidenceAsset): boolean {
  return (
    evidence.tags.includes("operator-upload") ||
    evidence.tags.includes("address-text-match") ||
    evidence.tags.includes("listing-address-match") ||
    evidence.visualAnalysis?.addressMatch === "supported"
  );
}

type RegistrationReport = {
  disconnectedFrames?: unknown;
  cameraPoses?: unknown;
  poseOptimization?: {
    beforeRmse?: unknown;
    afterRmse?: unknown;
  };
  denseSurfaceRefinement?: {
    denseBeforeRmseMeters?: unknown;
    denseAfterRmseMeters?: unknown;
  };
  jointGeometry?: {
    accepted?: unknown;
    device?: unknown;
    cuda?: unknown;
    peakVramMb?: unknown;
    rotationAgreementMedianDeg?: unknown;
    cameraPoses?: unknown;
  };
  artifactCleanup?: {
    removed?: unknown;
    crossViewSupportedFraction?: unknown;
  };
  sourceCoverageRegularization?: {
    affected?: unknown;
  };
  preflight?: {
    acceptedPairs?: unknown;
  };
};

type FramePair = {
  frameA: number;
  frameB: number;
  confidence: number;
};

function readRegistrationReport(
  reportUrl: string,
  casesRoot: string,
): RegistrationReport | undefined {
  if (!reportUrl.startsWith("/assets/")) return undefined;
  try {
    const suffix = decodeURIComponent(reportUrl.slice("/assets/".length));
    const candidate = resolve(casesRoot, suffix);
    const traversal = relative(casesRoot, candidate);
    if (traversal.startsWith("..") || resolve(traversal) === traversal)
      return undefined;
    return JSON.parse(readFileSync(candidate, "utf8")) as RegistrationReport;
  } catch {
    return undefined;
  }
}

function reconstructionGeometry(
  artifact: ReconstructionArtifact,
  job: WorkerJob,
  report: RegistrationReport | undefined,
  store: StructureStore,
): ReconstructionGeometry | undefined {
  const evidenceIds = artifact.evidenceIds ?? [artifact.evidenceId];
  if (artifact.mode !== "multi_image") {
    const evidence = store.getEvidence(artifact.evidenceId);
    return {
      backend: artifact.mode === "panorama" ? "sharp360" : "sharp_single_view",
      coordinateFrame: "anchor_camera_metric_opencv",
      jointCameraAccepted: false,
      horizontalCoverageDegrees:
        evidence?.capture?.horizontalCoverageDegrees ??
        (artifact.mode === "panorama" ? 360 : 62),
      cameraPoses: [
        {
          evidenceId: artifact.evidenceId,
          sourceIndex: 0,
          position: [0, 0, 0],
          rotationWxyz: [1, 0, 0, 0],
          scale: 1,
          placement: "anchor",
          confidenceScore: artifact.confidence.score,
        },
      ],
    };
  }

  const rawPoses = Array.isArray(report?.cameraPoses) ? report.cameraPoses : [];
  const jointPoses = Array.isArray(report?.jointGeometry?.cameraPoses)
    ? report.jointGeometry.cameraPoses
    : [];
  const jointConfidence = new Map<number, number>();
  for (const value of jointPoses) {
    if (!value || typeof value !== "object") continue;
    const item = value as Record<string, unknown>;
    if (
      Number.isInteger(item.frame) &&
      typeof item.confidence === "number" &&
      Number.isFinite(item.confidence)
    ) {
      jointConfidence.set(item.frame as number, item.confidence);
    }
  }
  const cameraPoses: ReconstructionGeometry["cameraPoses"] = [];
  for (const value of rawPoses) {
    if (!value || typeof value !== "object") continue;
    const item = value as Record<string, unknown>;
    const sourceIndex = integerValue(item.frame);
    const position = triple(item.position);
    const rotationWxyz = quaternion(item.rotationWxyz);
    const evidenceId =
      sourceIndex === undefined ? undefined : evidenceIds[sourceIndex];
    if (
      sourceIndex === undefined ||
      !evidenceId ||
      !position ||
      !rotationWxyz
    ) {
      continue;
    }
    const placement =
      item.placement === "joint_camera_calibrated_by_metric_core"
        ? "joint_camera_calibrated_by_metric_core"
        : item.placement === "measured_feature_and_sharp_metric"
          ? "measured_feature_and_sharp_metric"
          : "anchor";
    cameraPoses.push({
      evidenceId,
      sourceIndex,
      position,
      rotationWxyz,
      scale: positiveNumber(item.scale) ?? 1,
      placement,
      confidenceScore:
        placement === "measured_feature_and_sharp_metric"
          ? Math.max(
              job.registration_confidence ?? 0.5,
              jointConfidence.get(sourceIndex) ?? 0,
            )
          : (jointConfidence.get(sourceIndex) ?? 0.55),
    });
  }
  const peakVramMb = nonnegativeNumber(report?.jointGeometry?.peakVramMb);
  return {
    backend: "vggt_sharp_joint",
    coordinateFrame: "anchor_camera_metric_opencv",
    ...(typeof report?.jointGeometry?.device === "string"
      ? { gpu: report.jointGeometry.device }
      : {}),
    ...(typeof report?.jointGeometry?.cuda === "string"
      ? { cuda: report.jointGeometry.cuda }
      : {}),
    ...(peakVramMb !== undefined ? { peakVramMb } : {}),
    jointCameraAccepted: report?.jointGeometry?.accepted === true,
    cameraPoses,
  };
}

function reconstructionQuality(
  artifact: ReconstructionArtifact,
  job: WorkerJob,
  report: RegistrationReport | undefined,
): ReconstructionQuality | undefined {
  if (artifact.mode !== "multi_image") {
    return {
      registeredRatio: 1,
      missingBridgeEvidenceIds: [],
      cleanupRemovedGaussians: 0,
    };
  }
  const evidenceIds = artifact.evidenceIds ?? [artifact.evidenceId];
  const disconnected = Array.isArray(report?.disconnectedFrames)
    ? report.disconnectedFrames
        .map(integerValue)
        .filter((value): value is number => value !== undefined)
    : [];
  const beforeRmse = nonnegativeNumber(report?.poseOptimization?.beforeRmse);
  const afterRmse = nonnegativeNumber(report?.poseOptimization?.afterRmse);
  const denseBeforeRmse = nonnegativeNumber(
    report?.denseSurfaceRefinement?.denseBeforeRmseMeters,
  );
  const denseAfterRmse = nonnegativeNumber(
    report?.denseSurfaceRefinement?.denseAfterRmseMeters,
  );
  const agreement = nonnegativeNumber(
    report?.jointGeometry?.rotationAgreementMedianDeg,
  );
  const crossViewSupportedRatio = unitNumber(
    report?.artifactCleanup?.crossViewSupportedFraction,
  );
  const sourceCoverageAdjusted = integerValue(
    report?.sourceCoverageRegularization?.affected,
  );
  return {
    registeredRatio:
      (job.connected_frame_count ?? 1) /
      Math.max(1, job.frame_count ?? evidenceIds.length),
    missingBridgeEvidenceIds: disconnected.flatMap((index) =>
      evidenceIds[index] ? [evidenceIds[index]!] : [],
    ),
    ...(beforeRmse !== undefined ? { poseGraphBeforeRmse: beforeRmse } : {}),
    ...(afterRmse !== undefined ? { poseGraphAfterRmse: afterRmse } : {}),
    ...(denseBeforeRmse !== undefined
      ? { denseSurfaceBeforeRmseMeters: denseBeforeRmse }
      : {}),
    ...(denseAfterRmse !== undefined
      ? { denseSurfaceAfterRmseMeters: denseAfterRmse }
      : {}),
    ...(agreement !== undefined
      ? { rotationAgreementMedianDeg: agreement }
      : {}),
    cleanupRemovedGaussians:
      integerValue(report?.artifactCleanup?.removed) ?? 0,
    ...(crossViewSupportedRatio !== undefined
      ? { crossViewSupportedRatio }
      : {}),
    ...(sourceCoverageAdjusted !== undefined
      ? { sourceCoverageAdjustedGaussians: sourceCoverageAdjusted }
      : {}),
  };
}

function integerValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value >= 0
    ? value
    : undefined;
}

function nonnegativeNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : undefined;
}

function unitNumber(value: unknown): number | undefined {
  return typeof value === "number" &&
    Number.isFinite(value) &&
    value >= 0 &&
    value <= 1
    ? value
    : undefined;
}

function positiveNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : undefined;
}

function triple(value: unknown): [number, number, number] | undefined {
  return finiteTuple(value, 3) as [number, number, number] | undefined;
}

function quaternion(
  value: unknown,
): [number, number, number, number] | undefined {
  return finiteTuple(value, 4) as [number, number, number, number] | undefined;
}

function finiteTuple(value: unknown, length: number): number[] | undefined {
  if (
    !Array.isArray(value) ||
    value.length !== length ||
    value.some((item) => typeof item !== "number" || !Number.isFinite(item))
  ) {
    return undefined;
  }
  return value as number[];
}

export function nextVerifiedFrameGroup(
  report: RegistrationReport | undefined,
): number[] | undefined {
  return verifiedFrameGroups(report)[0];
}

export function verifiedFrameGroups(
  report: RegistrationReport | undefined,
): number[][] {
  return coverageFrameGroups(report, 0, false).filter(
    (frames) => frames.length >= 2,
  );
}

export function coverageFrameGroups(
  report: RegistrationReport | undefined,
  frameCount: number,
  includeEveryInput: boolean,
): number[][] {
  if (!report) return [];
  const candidates = includeEveryInput
    ? new Set(Array.from({ length: frameCount }, (_, index) => index))
    : new Set(
        Array.isArray(report.disconnectedFrames)
          ? report.disconnectedFrames.filter(
              (value): value is number => Number.isInteger(value) && value >= 0,
            )
          : [],
      );
  if (candidates.size === 0) return [];
  const pairs = Array.isArray(report.preflight?.acceptedPairs)
    ? report.preflight.acceptedPairs
        .map(parseFramePair)
        .filter((pair): pair is FramePair => Boolean(pair))
        .filter(
          (pair) => candidates.has(pair.frameA) && candidates.has(pair.frameB),
        )
    : [];

  const adjacency = new Map<number, Set<number>>();
  for (const pair of pairs) {
    const left = adjacency.get(pair.frameA) ?? new Set<number>();
    const right = adjacency.get(pair.frameB) ?? new Set<number>();
    left.add(pair.frameB);
    right.add(pair.frameA);
    adjacency.set(pair.frameA, left);
    adjacency.set(pair.frameB, right);
  }
  const groups: Array<{ frames: number[]; confidence: number }> = [];
  const visited = new Set<number>();
  for (const start of candidates) {
    if (visited.has(start)) continue;
    const stack = [start];
    const frames: number[] = [];
    while (stack.length > 0) {
      const frame = stack.pop();
      if (frame === undefined || visited.has(frame)) continue;
      visited.add(frame);
      frames.push(frame);
      for (const neighbor of adjacency.get(frame) ?? []) stack.push(neighbor);
    }
    const groupSet = new Set(frames);
    const confidence = pairs
      .filter((pair) => groupSet.has(pair.frameA) && groupSet.has(pair.frameB))
      .reduce((sum, pair) => sum + pair.confidence, 0);
    groups.push({ frames: frames.sort((a, b) => a - b), confidence });
  }
  groups.sort(
    (left, right) =>
      right.frames.length - left.frames.length ||
      right.confidence - left.confidence ||
      left.frames[0]! - right.frames[0]!,
  );
  return groups.map((group) => group.frames);
}

function representedEvidenceIds(
  artifacts: ReconstructionArtifact[],
): Set<string> {
  const represented = new Set<string>();
  for (const artifact of artifacts) {
    if (artifact.mode === "floorplan" || artifact.status === "failed") continue;
    if (artifact.status === "queued" || artifact.status === "running") {
      for (const id of artifact.evidenceIds ?? [artifact.evidenceId]) {
        represented.add(id);
      }
      continue;
    }
    if (artifact.status !== "ready") continue;
    if (artifact.mode === "multi_image") {
      for (const pose of artifact.geometry?.cameraPoses ?? []) {
        represented.add(pose.evidenceId);
      }
    } else {
      represented.add(artifact.evidenceId);
    }
  }
  return represented;
}

export function evidenceSignature(evidenceIds: string[]): string {
  return [...new Set(evidenceIds)].sort().join("\u0000");
}

function parseFramePair(value: unknown): FramePair | undefined {
  if (!value || typeof value !== "object") return undefined;
  const pair = value as Record<string, unknown>;
  if (!Number.isInteger(pair.frameA) || !Number.isInteger(pair.frameB))
    return undefined;
  return {
    frameA: pair.frameA as number,
    frameB: pair.frameB as number,
    confidence:
      typeof pair.confidence === "number" && Number.isFinite(pair.confidence)
        ? pair.confidence
        : 0,
  };
}
