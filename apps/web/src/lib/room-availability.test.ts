import { describe, expect, it } from "vitest";
import type {
  ReconstructionArtifact,
  SpatialNode,
} from "@structurefirst/contracts";
import {
  artifactForSpace,
  resolvePlanRoom,
  roomTypeFromPlanLabel,
} from "./room-availability";

function artifact(
  id: string,
  calibratedEvidenceIds: string[],
  submittedEvidenceIds = calibratedEvidenceIds,
): ReconstructionArtifact {
  return {
    id,
    evidenceId: submittedEvidenceIds[0] ?? "evidence_anchor",
    evidenceIds: submittedEvidenceIds,
    status: "ready",
    mode: "multi_image",
    splatUrl: `/api/artifacts/${id}.splat`,
    geometry: {
      backend: "vggt_sharp_joint",
      coordinateFrame: "anchor_camera_metric_opencv",
      jointCameraAccepted: true,
      cameraPoses: calibratedEvidenceIds.map((evidenceId, sourceIndex) => ({
        evidenceId,
        sourceIndex,
        position: [sourceIndex, 0, 0],
        rotationWxyz: [1, 0, 0, 0],
        scale: 1,
        placement: "joint_camera_calibrated_by_metric_core",
        confidenceScore: 0.9,
      })),
    },
  } as ReconstructionArtifact;
}

function room(
  id: string,
  roomType: SpatialNode["roomType"],
  sourceIds: string[],
  artifactId?: string,
): SpatialNode {
  return {
    id,
    caseId: "case_test",
    ...(artifactId ? { artifactId } : {}),
    label: roomType ?? "Room",
    kind: "room",
    roomType,
    sourceIds,
    confidence: {
      score: 0.8,
      band: "reconstructed",
      state: "derived",
      rationale: "test",
      sourceCount: sourceIds.length,
      updatedAt: new Date(0).toISOString(),
    },
  };
}

describe("room availability", () => {
  it("classifies Primary Bath as a bathroom rather than a bedroom", () => {
    expect(roomTypeFromPlanLabel("Primary Bath")).toBe("bathroom");
    expect(roomTypeFromPlanLabel("Primary Suite")).toBe("bedroom");
  });

  it("does not treat rejected submitted frames as registered 3D views", () => {
    const scene = artifact(
      "artifact_scene",
      ["evidence_a"],
      ["evidence_a", "evidence_rejected"],
    );
    expect(
      artifactForSpace(
        room("room_rejected", "kitchen", ["evidence_rejected"]),
        [scene],
      ),
    ).toBeUndefined();
  });

  it("links a unique plan room to one calibrated Gaussian scene", () => {
    const scene = artifact("artifact_living", ["evidence_living"]);
    const space = room(
      "room_living",
      "living_room",
      ["evidence_living"],
      scene.id,
    );
    const result = resolvePlanRoom(
      {
        label: "Living Room",
        floorNumber: 1,
        sourceEvidenceId: "plan_floor_1",
        sameTypeLabelCount: 1,
      },
      [space],
      [scene],
    );
    expect(result.available).toBe(true);
    expect(result.space?.id).toBe(space.id);
    expect(result.artifact?.id).toBe(scene.id);
  });

  it("treats several calibrated source nodes in one artifact as one 3D scene", () => {
    const scene = artifact("artifact_kitchen", ["evidence_a", "evidence_b"]);
    const result = resolvePlanRoom(
      {
        label: "Kitchen",
        floorNumber: 1,
        sourceEvidenceId: "plan_floor_1",
        sameTypeLabelCount: 1,
      },
      [
        room("room_kitchen_a", "kitchen", ["evidence_a"], scene.id),
        room("room_kitchen_b", "kitchen", ["evidence_b"], scene.id),
      ],
      [scene],
    );
    expect(result.available).toBe(true);
    expect(result.artifact?.id).toBe(scene.id);
  });

  it("refuses to map one photo scene onto several same-type plan rooms", () => {
    const scene = artifact("artifact_bedroom", ["evidence_bedroom"]);
    const result = resolvePlanRoom(
      {
        label: "Bedroom #3",
        floorNumber: 4,
        sourceEvidenceId: "plan_floor_4",
        sameTypeLabelCount: 3,
      },
      [room("room_bedroom", "bedroom", ["evidence_bedroom"], scene.id)],
      [scene],
    );
    expect(result.available).toBe(false);
    expect(result.detail).toContain("3 bedroom labels");
    expect(result.candidates.map((candidate) => candidate.artifact.id)).toEqual(
      [scene.id],
    );
  });

  it("exposes source previews for ambiguous plan-room candidates", () => {
    const first = artifact("artifact_bedroom_a", ["evidence_a"]);
    const second = artifact("artifact_bedroom_b", ["evidence_b"]);
    const result = resolvePlanRoom(
      {
        label: "Bedroom 2",
        floorNumber: 2,
        sourceEvidenceId: "plan_floor_2",
        sameTypeLabelCount: 2,
      },
      [
        room("room_a", "bedroom", ["evidence_a"], first.id),
        room("room_b", "bedroom", ["evidence_b"], second.id),
      ],
      [first, second],
      [
        {
          id: "evidence_a",
          localUrl: "/uploads/bedroom-a.jpg",
          title: "Bedroom angle A",
        },
        {
          id: "evidence_b",
          localUrl: "/uploads/bedroom-b.jpg",
          title: "Bedroom angle B",
        },
      ] as never,
    );
    expect(result.available).toBe(false);
    expect(result.candidates).toHaveLength(2);
    expect(result.candidates.map((candidate) => candidate.previewUrl)).toEqual([
      "/uploads/bedroom-a.jpg",
      "/uploads/bedroom-b.jpg",
    ]);
  });

  it("leaves a plan label unavailable when no calibrated Gaussian exists", () => {
    const result = resolvePlanRoom(
      {
        label: "Kitchen",
        floorNumber: 1,
        sourceEvidenceId: "plan_floor_1",
        sameTypeLabelCount: 1,
      },
      [room("room_kitchen", "kitchen", ["evidence_kitchen"])],
      [],
    );
    expect(result.available).toBe(false);
    expect(result.detail).toContain("no registered Gaussian");
  });
});
