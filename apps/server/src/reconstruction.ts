import { createHash } from "node:crypto";
import { createReadStream, readFileSync } from "node:fs";
import { relative, resolve } from "node:path";
import {
  type Case,
  type EvidenceAsset,
  type PipelineEvent,
  type PipelineStageName,
  type MultiReconstructionRequest,
  type ReconstructionArtifact,
  type ReconstructionRequest,
  type StageStatus,
} from "@structurefirst/contracts";
import type { AppConfig } from "./config.js";
import { CaseEventHub } from "./events.js";
import { confidence } from "./lib/confidence.js";
import { createId, nowIso } from "./lib/ids.js";
import { StructureStore } from "./store.js";

type WorkerJob = {
  job_id: string;
  status: "queued" | "running" | "ready" | "failed";
  splat_url?: string;
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
      for (const artifact of this.store.listArtifacts(caseValue.id)) {
        if (artifact.status === "queued" || artifact.status === "running") {
          this.schedulePoll(artifact.id, 0);
        }
      }
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
      this.failArtifact(
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
        this.failArtifact(
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
      if (!job.splat_url || !job.manifest_url) {
        throw new Error("Worker completed without a splat and manifest URL.");
      }

      const ready: ReconstructionArtifact = {
        ...artifact,
        status: "ready",
        splatUrl: job.splat_url,
        manifestUrl: job.manifest_url,
        ...(job.gaussian_count ? { gaussianCount: job.gaussian_count } : {}),
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
          job.fallback_used
            ? 0.52
            : artifact.mode === "multi_image"
              ? Math.min(0.68, 0.35 + (job.registration_confidence ?? 0) * 0.33)
              : 0.58,
          "reconstructed",
          "derived",
          job.fallback_used
            ? "The photographs did not register, so LucidFrame reconstructed a nearby view from the first exact source image. Occluded space remains unknown."
            : artifact.mode === "multi_image"
              ? "LucidFrame reconstructed and registered only photographs with measured visual and metric overlap. Occluded space remains unknown."
              : "LucidFrame reconstructed nearby appearance from the selected image. Occluded and unseen space remains unknown.",
          job.fallback_used
            ? 1
            : (job.connected_frame_count ?? artifact.evidenceIds?.length ?? 1),
        ),
      };
      this.store.putArtifact(ready);
      this.pollers.delete(artifactId);
      await this.applyReadyCoverage(ready);
    } catch (error) {
      if (attempt >= 10) {
        this.failArtifact(
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
    const isInterior = groupedEvidence.some((item) =>
      item.tags.includes("interior"),
    );
    const sourceCount = artifact.fallback
      ? 1
      : (artifact.registration?.connectedFrameCount ?? 1);
    const coverage = {
      ...current.coverage,
      ...(isInterior
        ? {
            interior: confidence(
              artifact.mode === "multi_image" && !artifact.fallback
                ? 0.56
                : 0.42,
              "reconstructed",
              "derived",
              artifact.mode === "multi_image" && !artifact.fallback
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
      artifact.fallback
        ? `The photographs did not register; LucidFrame still produced ${artifact.gaussianCount?.toLocaleString() ?? "a"} Gaussians from the first exact source image.`
        : artifact.mode === "multi_image"
          ? `LucidFrame connected ${sourceCount}/${artifact.evidenceIds?.length ?? sourceCount} captures and produced ${artifact.gaussianCount?.toLocaleString() ?? "a"} Gaussians.`
          : `LucidFrame produced ${artifact.gaussianCount?.toLocaleString() ?? "a"} Gaussian scene. Nearby-view limitations apply.`,
    );
    await this.onReady?.(artifact);
    await this.queueNextVerifiedComponent(artifact);
  }

  private async queueNextVerifiedComponent(
    artifact: ReconstructionArtifact,
  ): Promise<void> {
    if (artifact.mode !== "multi_image" || !artifact.evidenceIds) return;
    const reportUrl = artifact.registration?.reportUrl;
    if (!reportUrl) return;
    const report = readRegistrationReport(reportUrl, this.config.casesRoot);
    const frameGroup = nextVerifiedFrameGroup(report);
    if (!frameGroup) return;
    const evidenceIds = frameGroup
      .map((index) => artifact.evidenceIds?.[index])
      .filter((id): id is string => Boolean(id))
      .filter((id) => {
        const evidence = this.store.getEvidence(id);
        return Boolean(
          evidence && !evidence.tags.includes("reconstruction-excluded"),
        );
      });
    if (evidenceIds.length < 2) return;
    const signature = evidenceSignature(evidenceIds);
    const alreadyQueued = this.store
      .listArtifacts(artifact.caseId)
      .some(
        (candidate) =>
          candidate.id !== artifact.id &&
          candidate.evidenceIds &&
          evidenceSignature(candidate.evidenceIds) === signature,
      );
    if (alreadyQueued) return;
    await this.queueMulti(artifact.caseId, { evidenceIds });
  }

  private failArtifact(
    artifact: ReconstructionArtifact,
    message: string,
    job?: WorkerJob,
  ): void {
    this.store.putArtifact({
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
    this.updateWorkflow(artifact.caseId, "reconstruction", "failed", message);
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

export function nextVerifiedFrameGroup(
  report: RegistrationReport | undefined,
): number[] | undefined {
  if (!report || !Array.isArray(report.disconnectedFrames)) return undefined;
  const disconnected = new Set(
    report.disconnectedFrames.filter(
      (value): value is number => Number.isInteger(value) && value >= 0,
    ),
  );
  if (disconnected.size < 2) return undefined;
  const pairs = Array.isArray(report.preflight?.acceptedPairs)
    ? report.preflight.acceptedPairs
        .map(parseFramePair)
        .filter((pair): pair is FramePair => Boolean(pair))
        .filter(
          (pair) =>
            disconnected.has(pair.frameA) && disconnected.has(pair.frameB),
        )
    : [];
  if (pairs.length === 0) return undefined;

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
  for (const start of adjacency.keys()) {
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
    if (frames.length >= 2)
      groups.push({ frames: frames.sort((a, b) => a - b), confidence });
  }
  groups.sort(
    (left, right) =>
      right.frames.length - left.frames.length ||
      right.confidence - left.confidence ||
      left.frames[0]! - right.frames[0]!,
  );
  return groups[0]?.frames;
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

function evidenceSignature(evidenceIds: string[]): string {
  return [...new Set(evidenceIds)].sort().join("\u0000");
}
