import { describe, expect, it } from "vitest";
import type {
  EvidenceAsset,
  ReconstructionArtifact,
  RoomType,
} from "@structurefirst/contracts";
import { artifactSceneLabels, maximalReadyArtifacts } from "./scene-labels";

describe("Gaussian scene labels", () => {
  it("numbers repeated room instances and keeps garage distinct", () => {
    const photos = [
      photo("bath-a", "bathroom", 1),
      photo("bath-b", "bathroom", 5),
      photo("garage", "garage", 9),
    ];
    const labels = artifactSceneLabels(
      [
        artifact("a", "bath-a"),
        artifact("b", "bath-b"),
        artifact("g", "garage"),
      ],
      photos,
    );
    expect(labels.get("a")).toBe("Bathroom 1 · 1 view");
    expect(labels.get("b")).toBe("Bathroom 2 · 1 view");
    expect(labels.get("g")).toBe("Garage · 1 view");
  });

  it("treats exterior living-space wording as exterior", () => {
    const exterior = photo("rear", "living_room", 1, "exterior");
    expect(
      artifactSceneLabels([artifact("rear-artifact", "rear")], [exterior]).get(
        "rear-artifact",
      ),
    ).toBe("Exterior · 1 view");
  });

  it("hides single-view scenes after a larger verified scene supersedes them", () => {
    const first = artifact("first", "view-a");
    const second = artifact("second", "view-b");
    const connected: ReconstructionArtifact = {
      ...artifact("connected", "view-a"),
      mode: "multi_image",
      evidenceIds: ["view-a", "view-b"],
      geometry: {
        backend: "vggt_sharp_joint",
        coordinateFrame: "anchor_camera_metric_opencv",
        jointCameraAccepted: true,
        cameraPoses: [
          {
            evidenceId: "view-a",
            sourceIndex: 0,
            position: [0, 0, 0],
            rotationWxyz: [1, 0, 0, 0],
            scale: 1,
            placement: "anchor",
            confidenceScore: 0.8,
          },
          {
            evidenceId: "view-b",
            sourceIndex: 1,
            position: [1, 0, 0],
            rotationWxyz: [1, 0, 0, 0],
            scale: 1,
            placement: "measured_feature_and_sharp_metric",
            confidenceScore: 0.8,
          },
        ],
      },
    };
    expect(
      maximalReadyArtifacts([first, second, connected]).map((item) => item.id),
    ).toEqual(["connected"]);
  });

  it("shows only the best artifact when automatic retries cover the same views", () => {
    const first = {
      ...artifact("older", "view-a"),
      mode: "multi_image" as const,
      evidenceIds: ["view-a", "view-b"],
      gaussianCount: 2_000_000,
    };
    const better = {
      ...first,
      id: "better",
      gaussianCount: 3_000_000,
    };
    expect(
      maximalReadyArtifacts([first, better]).map((item) => item.id),
    ).toEqual(["better"]);
  });

  it("keeps exact source scenes when an old merged overlay lacks shared geometry", () => {
    const first = artifact("first", "view-a");
    const second = artifact("second", "view-b");
    const third = artifact("third", "view-c");
    const broken = {
      ...artifact("broken", "view-a"),
      mode: "multi_image" as const,
      evidenceIds: ["view-a", "view-b", "view-c"],
      quality: {
        registeredRatio: 1,
        missingBridgeEvidenceIds: [],
        cleanupRemovedGaussians: 0,
        crossViewSupportedRatio: 0.26,
      },
      registration: {
        status: "connected" as const,
        method: "sift_loftr_sharp_pose_graph" as const,
        frameCount: 3,
        connectedFrameCount: 3,
        confidenceScore: 0.73,
        note: "test",
      },
    };

    expect(
      maximalReadyArtifacts([first, second, third, broken]).map(
        (item) => item.id,
      ),
    ).toEqual(["first", "second", "third"]);
  });
});

function photo(
  id: string,
  roomType: RoomType,
  captureOrder: number,
  sceneType: "interior" | "exterior" = "interior",
): EvidenceAsset {
  return {
    id,
    caseId: "case_test",
    title: id,
    kind: "image",
    sourceProvider: "test",
    discoveredAt: "2026-01-01T00:00:00.000Z",
    rights: "operator_owned",
    cachePolicy: "local_allowed",
    redistributable: false,
    validation: "operator_uploaded",
    tags: ["operator-upload"],
    notes: "test",
    capture: {
      projection: "perspective",
      width: 100,
      height: 100,
      horizontalCoverageDegrees: 60,
      captureOrder,
      overlapSetId: "test-set",
      projectionSource: "operator",
    },
    visualAnalysis: {
      sceneType,
      roomType,
      floorHint: "unknown",
      roomLabels: [],
      propertyRelevance: "likely",
      addressMatch: "possible",
      connections: [],
      summary: "test",
      provider: "nvidia_nim",
      model: "test",
      confidenceScore: 0.8,
      analyzedAt: "2026-01-01T00:00:00.000Z",
    },
    confidence: {
      score: 0.8,
      band: "verified",
      state: "observed",
      rationale: "test",
      sourceCount: 1,
      updatedAt: "2026-01-01T00:00:00.000Z",
    },
  };
}

function artifact(id: string, evidenceId: string): ReconstructionArtifact {
  return {
    id,
    caseId: "case_test",
    evidenceId,
    evidenceIds: [evidenceId],
    status: "ready",
    mode: "single_image",
    splatUrl: `/assets/${id}.ply`,
    modelName: "test",
    modelLicense: "test",
    createdAt: "2026-01-01T00:00:00.000Z",
    updatedAt: "2026-01-01T00:00:00.000Z",
    confidence: {
      score: 0.5,
      band: "reconstructed",
      state: "derived",
      rationale: "test",
      sourceCount: 1,
      updatedAt: "2026-01-01T00:00:00.000Z",
    },
  };
}
