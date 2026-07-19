import {
  PIPELINE_STAGE_ORDER,
  type BuildingProfile,
  type Case,
  type CreateCaseInput,
  type EvidenceAsset,
  type HazardCandidate,
  type PipelineEvent,
  type PipelineStageName,
  type StageStatus,
} from "@structurefirst/contracts";
import type { AppConfig } from "./config.js";
import { EvidenceDiscoveryCoordinator } from "./discovery.js";
import { CaseEventHub } from "./events.js";
import { confidence } from "./lib/confidence.js";
import { createId, nowIso } from "./lib/ids.js";
import { classifySource } from "./lib/source-policy.js";
import { discoverBuildingEvidence } from "./providers/brave.js";
import { geocodeAddress } from "./providers/nominatim.js";
import { findNearestBuilding } from "./providers/openstreetmap.js";
import type { ReconstructionCoordinator } from "./reconstruction.js";
import type { SceneIntelligenceService } from "./scene-intelligence.js";
import { StructureStore } from "./store.js";

export type PipelineDependencies = {
  geocode: typeof geocodeAddress;
  findBuilding: typeof findNearestBuilding;
  discover: typeof discoverBuildingEvidence;
};

const defaultDependencies: PipelineDependencies = {
  geocode: geocodeAddress,
  findBuilding: findNearestBuilding,
  discover: discoverBuildingEvidence,
};

export class CasePipeline {
  private readonly active = new Set<string>();

  constructor(
    private readonly store: StructureStore,
    private readonly events: CaseEventHub,
    private readonly config: AppConfig,
    private readonly dependencies: PipelineDependencies = defaultDependencies,
    private readonly discovery?: EvidenceDiscoveryCoordinator,
    private readonly reconstruction?: ReconstructionCoordinator,
    private readonly sceneIntelligence?: SceneIntelligenceService,
  ) {}

  createCase(input: CreateCaseInput): Case {
    const now = nowIso();
    const value: Case = {
      id: createId("case"),
      addressInput: input.address,
      displayAddress: input.address,
      status: "collecting",
      activeRole: input.role,
      createdAt: now,
      updatedAt: now,
      incident: {
        type: input.incidentType,
        blockedAreas: [],
        updatedAt: now,
      },
      coverage: {
        exterior: confidence(
          0,
          "unknown",
          "unknown",
          "No exterior evidence assessed yet.",
          0,
        ),
        interior: confidence(
          0,
          "unknown",
          "unknown",
          "No interior evidence assessed yet.",
          0,
        ),
        records: confidence(
          0,
          "unknown",
          "unknown",
          "No building records assessed yet.",
          0,
        ),
        lastAssessedAt: now,
      },
      stages: PIPELINE_STAGE_ORDER.map((name) => ({
        name,
        status: "pending",
        message: "Waiting",
      })),
      operatorNotes: "",
    };
    return this.store.putCase(value);
  }

  start(caseId: string): void {
    if (this.active.has(caseId)) return;
    this.active.add(caseId);
    void this.run(caseId).finally(() => this.active.delete(caseId));
  }

  isRunning(caseId: string): boolean {
    return this.active.has(caseId);
  }

  private async run(caseId: string): Promise<void> {
    try {
      let caseValue = this.requireCase(caseId);
      this.setStage(
        caseId,
        "address_resolution",
        "running",
        "Resolving the submitted address.",
      );
      const geocoded = await this.dependencies.geocode(
        caseValue.addressInput,
        this.config,
      );
      if (!geocoded) {
        this.setStage(
          caseId,
          "address_resolution",
          "failed",
          "No matching address was found.",
        );
        this.updateCase(caseId, { status: "failed" });
        return;
      }

      const profile: BuildingProfile = {
        displayAddress: geocoded.displayAddress,
        location: {
          latitude: geocoded.latitude,
          longitude: geocoded.longitude,
        },
        tags: {},
        source: {
          id: createId("source"),
          label: "Address resolution",
          provider: geocoded.provider ?? "OpenStreetMap Nominatim",
          url: geocoded.sourceUrl ?? "https://www.openstreetmap.org/copyright",
          retrievedAt: nowIso(),
          license: geocoded.license ?? "ODbL 1.0",
        },
        confidence: confidence(
          geocoded.confidenceScore ?? 0.72,
          "estimated",
          "observed",
          geocoded.matchMethod === "census_exact"
            ? "Matched to the submitted U.S. structure address by the Census Geocoder; the operator has not yet confirmed the location."
            : "Ranked Nominatim candidates against the submitted house number, locality, state, and postal code; operator confirmation is still required.",
          1,
        ),
      };
      caseValue = this.updateCase(caseId, {
        displayAddress: geocoded.displayAddress,
        profile,
      });
      this.setStage(
        caseId,
        "address_resolution",
        "complete",
        `Resolved to ${geocoded.displayAddress}`,
      );

      this.setStage(
        caseId,
        "building_records",
        "running",
        "Checking nearby OpenStreetMap building geometry.",
      );
      let buildingFound = false;
      try {
        const building = await this.dependencies.findBuilding(
          geocoded.latitude,
          geocoded.longitude,
          this.config,
          `${caseValue.addressInput} ${geocoded.displayAddress}`,
        );
        if (building) {
          buildingFound = true;
          const recordSourceId = createId("source");
          const recordProfile: BuildingProfile = {
            ...profile,
            ...(building.footprint ? { footprint: building.footprint } : {}),
            ...(building.levels ? { levels: building.levels } : {}),
            ...(building.buildingType
              ? { buildingType: building.buildingType }
              : {}),
            ...(building.construction
              ? { construction: building.construction }
              : {}),
            ...(building.yearBuilt ? { yearBuilt: building.yearBuilt } : {}),
            tags: building.tags,
            source: {
              id: recordSourceId,
              label: `OpenStreetMap ${building.osmType} ${building.osmId}`,
              provider: "OpenStreetMap",
              url: building.sourceUrl,
              retrievedAt: nowIso(),
              license: "ODbL 1.0",
            },
            confidence: confidence(
              building.footprint ? 0.68 : 0.48,
              "estimated",
              "observed",
              "Crowdsourced building record; not independently verified for this incident.",
              1,
            ),
          };
          caseValue = this.updateCase(caseId, {
            profile: recordProfile,
            coverage: {
              ...caseValue.coverage,
              records: confidence(
                building.footprint ? 0.68 : 0.48,
                "estimated",
                "observed",
                "One OpenStreetMap building record was located; operator verification is pending.",
                1,
              ),
              lastAssessedAt: nowIso(),
            },
          });
          this.store.putEvidence(
            this.createOsmEvidence(caseId, building.sourceUrl),
          );
          this.setStage(
            caseId,
            "building_records",
            "complete",
            building.footprint
              ? "Found a mapped building footprint."
              : "Found a building record without usable geometry.",
            1,
          );
        } else {
          this.setStage(
            caseId,
            "building_records",
            "limited",
            "No nearby mapped building footprint was found.",
            0,
          );
        }
      } catch (error) {
        this.setStage(
          caseId,
          "building_records",
          "limited",
          `Building record lookup was unavailable: ${errorMessage(error)}`,
          0,
        );
      }

      this.setStage(
        caseId,
        "evidence_discovery",
        "running",
        "Searching configured public discovery providers.",
      );
      let discoveredCount = 0;
      if (this.discovery) {
        const result = await this.discovery.discoverCase(caseId, {
          includeOpenverse: true,
          includeBrowser: true,
          includeBrave: true,
        });
        discoveredCount = result.added;
        this.setStage(
          caseId,
          "evidence_discovery",
          result.added > 0 ? "complete" : "limited",
          result.added > 0
            ? `Added ${result.added} review candidates from ${result.providers.join(", ") || "configured discovery"}.`
            : (result.warnings[0] ??
                "No discovery candidates matched this address."),
          result.added,
        );
      } else if (this.config.braveApiKey) {
        try {
          const links = await this.dependencies.discover(
            caseValue.displayAddress,
            this.config.braveApiKey,
          );
          for (const link of links) {
            const policy = classifySource(link.url);
            const evidence: EvidenceAsset = {
              id: createId("evidence"),
              caseId,
              title: link.title,
              kind: link.kind,
              sourceProvider: policy.provider,
              originUrl: link.url,
              ...(!policy.hardBlocked && link.thumbnailUrl
                ? { thumbnailUrl: link.thumbnailUrl }
                : {}),
              discoveredAt: nowIso(),
              rights: policy.rights,
              cachePolicy: policy.cachePolicy,
              redistributable: policy.redistributable,
              validation: "reachable",
              tags: ["automated-discovery"],
              notes: `${link.notes} ${policy.reason}`,
              confidence: confidence(
                0.28,
                "estimated",
                "inferred",
                "Search match only; address relevance and contents have not been visually confirmed.",
                1,
              ),
            };
            this.store.putEvidence(evidence);
            discoveredCount += 1;
          }
          this.setStage(
            caseId,
            "evidence_discovery",
            links.length > 0 ? "complete" : "limited",
            links.length > 0
              ? `Discovered ${links.length} candidate sources for review.`
              : "The configured search returned no candidate sources.",
            links.length,
          );
        } catch (error) {
          this.setStage(
            caseId,
            "evidence_discovery",
            "limited",
            `Public discovery was unavailable: ${errorMessage(error)}`,
            0,
          );
        }
      } else {
        this.setStage(
          caseId,
          "evidence_discovery",
          "skipped",
          "Brave Search is not configured. Building records and operator uploads remain available.",
          0,
        );
      }

      this.setStage(
        caseId,
        "evidence_validation",
        "running",
        "Applying source policy and checking permitted images with the configured vision model.",
      );
      let visuallyAnalyzed = 0;
      let visuallyRejected = 0;
      let visualWarning: string | undefined;
      if (this.sceneIntelligence) {
        try {
          const result = await this.sceneIntelligence.analyzeCase(caseId);
          visuallyAnalyzed = result.analyzed;
          visuallyRejected = result.rejected;
          visualWarning = result.warnings[0];
        } catch (error) {
          visualWarning = errorMessage(error);
        }
      }
      const evidence = this.store.listEvidence(caseId);
      const exteriorCandidates = evidence.filter(
        (item) =>
          item.kind === "image" &&
          item.visualAnalysis?.sceneType === "exterior" &&
          item.visualAnalysis.addressMatch !== "contradictory",
      ).length;
      const linkedOnlyCount = evidence.filter(
        (item) => item.cachePolicy !== "local_allowed",
      ).length;
      caseValue = this.requireCase(caseId);
      caseValue = this.updateCase(caseId, {
        coverage: {
          ...caseValue.coverage,
          exterior:
            exteriorCandidates > 0
              ? confidence(
                  0.24,
                  "estimated",
                  "inferred",
                  `${exteriorCandidates} image candidates were discovered but not visually confirmed.`,
                  exteriorCandidates,
                )
              : confidence(
                  0,
                  "unknown",
                  "unknown",
                  "No exterior image has been validated.",
                  0,
                ),
          lastAssessedAt: nowIso(),
        },
      });
      this.setStage(
        caseId,
        "evidence_validation",
        evidence.length > 0 ? "complete" : "limited",
        `${evidence.length} sources classified; ${linkedOnlyCount} retained as link-only metadata; ${visuallyAnalyzed} permitted images checked by VLM${visuallyRejected ? `, ${visuallyRejected} excluded` : ""}${visualWarning && visuallyAnalyzed === 0 ? ` (${visualWarning})` : ""}.`,
        evidence.length,
      );

      const localPhotos = this.store
        .listEvidence(caseId)
        .filter(
          (item) =>
            Boolean(item.localUrl) &&
            item.mimeType?.startsWith("image/") &&
            !item.tags.includes("reconstruction-excluded"),
        );
      if (this.reconstruction && localPhotos.length > 0) {
        try {
          const artifact = await this.reconstruction.queueAvailable(caseId);
          if (!artifact) {
            this.setStage(
              caseId,
              "reconstruction",
              "limited",
              "No usable local photo was available for reconstruction.",
              0,
            );
          }
        } catch (error) {
          this.setStage(
            caseId,
            "reconstruction",
            "failed",
            `LucidFrame could not start: ${errorMessage(error)}`,
            0,
          );
        }
      } else {
        this.setStage(
          caseId,
          "reconstruction",
          "limited",
          "No downloadable licensed photo was found. Responder photos can be added at any time.",
          0,
        );
      }

      this.setStage(
        caseId,
        "spatial_reasoning",
        "running",
        "Building the evidence-backed spatial graph.",
      );
      const current = this.requireCase(caseId);
      if (this.sceneIntelligence) {
        this.sceneIntelligence.rebuildSpatialGraph(caseId);
      } else if (current.profile) {
        this.store.putNode({
          id: createId("node"),
          caseId,
          label: "Known exterior location",
          kind: "exterior",
          sourceIds: [current.profile.source.id],
          confidence: current.profile.confidence,
        });
      }
      this.setStage(
        caseId,
        "spatial_reasoning",
        "limited",
        "Exterior location retained. Room and floor groups appear only after image classification and geometric registration.",
        this.store.listNodes(caseId).length,
      );

      this.setStage(
        caseId,
        "candidate_generation",
        "running",
        "Generating reviewable operational candidates.",
      );
      const gap = this.createInteriorGap(
        caseId,
        this.store.listEvidence(caseId).map((item) => item.id),
      );
      this.store.putHazard(gap);
      this.setStage(
        caseId,
        "candidate_generation",
        "limited",
        "Created one intelligence-gap advisory. No interior route was generated.",
        1,
      );
      const currentArtifact = this.store
        .listArtifacts(caseId)
        .find((item) => ["queued", "running", "ready"].includes(item.status));
      this.updateCase(caseId, {
        status:
          currentArtifact?.status === "ready"
            ? "briefing_ready"
            : currentArtifact
              ? "reconstructing"
              : "limited_evidence",
        coverage: {
          ...this.requireCase(caseId).coverage,
          interior: confidence(
            0,
            "unknown",
            "unknown",
            "No verified evidence establishes interior rooms, floors, or connections.",
            0,
          ),
          lastAssessedAt: nowIso(),
        },
      });

      void buildingFound;
      void discoveredCount;
    } catch (error) {
      const current = this.store.getCase(caseId);
      if (!current) return;
      const activeStage =
        current.stages.find((stage) => stage.status === "running")?.name ??
        "address_resolution";
      this.setStage(
        caseId,
        activeStage,
        "failed",
        `Pipeline stopped: ${errorMessage(error)}`,
      );
      this.updateCase(caseId, { status: "failed" });
    }
  }

  private createOsmEvidence(caseId: string, sourceUrl: string): EvidenceAsset {
    return {
      id: createId("evidence"),
      caseId,
      title: "OpenStreetMap building record",
      kind: "map",
      sourceProvider: "OpenStreetMap",
      originUrl: sourceUrl,
      discoveredAt: nowIso(),
      rights: "open_license",
      cachePolicy: "local_allowed",
      redistributable: true,
      validation: "reachable",
      tags: ["building-record", "exterior"],
      notes:
        "ODbL data. Attribution and share-alike obligations apply to exported derivative databases.",
      confidence: confidence(
        0.68,
        "estimated",
        "observed",
        "Record exists in OpenStreetMap but is not independently verified.",
        1,
      ),
    };
  }

  private createInteriorGap(
    caseId: string,
    sourceIds: string[],
  ): HazardCandidate {
    return {
      id: createId("hazard"),
      caseId,
      label: "Interior layout unknown",
      category: "intelligence_gap",
      description:
        "No evidence currently establishes interior rooms, floor-to-floor connections, utilities, or obstructions. Do not use the exterior reconstruction as an interior navigation model.",
      locationLabel: "Entire structure",
      severity: "caution",
      roles: ["fire", "law", "ems", "sar"],
      sourceIds,
      confidence: confidence(
        1,
        "verified",
        "derived",
        "Verified as a gap in the current case evidence, not a claim about the building itself.",
        sourceIds.length,
      ),
      review: "pending",
      reviewNote: "",
    };
  }

  private setStage(
    caseId: string,
    stageName: PipelineStageName,
    status: StageStatus,
    message: string,
    itemCount?: number,
  ): void {
    const current = this.requireCase(caseId);
    const now = nowIso();
    const stages = current.stages.map((stage) =>
      stage.name === stageName
        ? {
            ...stage,
            status,
            message,
            ...(status === "running"
              ? { startedAt: stage.startedAt ?? now }
              : {}),
            ...(["complete", "limited", "skipped", "failed"].includes(status)
              ? { completedAt: now }
              : {}),
            ...(itemCount === undefined ? {} : { itemCount }),
          }
        : stage,
    );
    this.store.putCase({ ...current, stages, updatedAt: now });
    this.emit(caseId, stageName, status, message);
  }

  private emit(
    caseId: string,
    stage: PipelineStageName,
    status: StageStatus,
    message: string,
  ): PipelineEvent {
    const event: PipelineEvent = {
      id: createId("event"),
      caseId,
      stage,
      status,
      message,
      createdAt: nowIso(),
    };
    this.store.putEvent(event);
    this.events.publish(event);
    return event;
  }

  private updateCase(caseId: string, patch: Partial<Case>): Case {
    const current = this.requireCase(caseId);
    return this.store.putCase({
      ...current,
      ...patch,
      id: caseId,
      updatedAt: nowIso(),
    });
  }

  private requireCase(caseId: string): Case {
    const value = this.store.getCase(caseId);
    if (!value) throw new Error(`Case ${caseId} does not exist.`);
    return value;
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Unknown provider error";
}
