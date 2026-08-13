import { describe, expect, it } from "vitest";
import type { EvidenceAsset } from "@structurefirst/contracts";
import { planCaptureSet } from "./capture-plan.js";

describe("manual capture set planning", () => {
  it("keeps large capture walks in ordered GPU-safe batches with shared bridges", () => {
    const plan = planCaptureSet(
      Array.from({ length: 27 }, (_, index) => frame(index)),
    );

    expect(plan.batches.map((batch) => batch.evidenceIds.length)).toEqual([
      12, 12, 9,
    ]);
    expect(plan.batches[1]?.bridgeEvidenceIds).toEqual([
      "evidence_09",
      "evidence_10",
      "evidence_11",
    ]);
    expect(plan.batches[2]?.bridgeEvidenceIds).toEqual([
      "evidence_18",
      "evidence_19",
      "evidence_20",
    ]);
    expect(
      new Set(plan.batches.flatMap((batch) => batch.evidenceIds)).size,
    ).toBe(27);
  });

  it("orders frames by capture order and excludes visual outliers", () => {
    const outlier = frame(2, ["reconstruction-excluded"]);
    const plan = planCaptureSet([frame(3), outlier, frame(0), frame(1)]);

    expect(plan.excludedFrames).toBe(1);
    expect(plan.batches[0]?.evidenceIds).toEqual([
      "evidence_00",
      "evidence_01",
      "evidence_03",
    ]);
  });

  it("does not leave a final singleton job", () => {
    const plan = planCaptureSet(
      Array.from({ length: 13 }, (_, index) => frame(index)),
      4,
      1,
    );

    expect(plan.batches.map((batch) => batch.evidenceIds.length)).toEqual([
      4, 4, 4, 4,
    ]);
    expect(plan.batches.at(-1)?.bridgeEvidenceIds).toEqual(["evidence_09"]);
  });

  it("keeps every visible room type separate without inventing hallway bridges", () => {
    const plan = planCaptureSet([
      classifiedFrame(0, "unknown", "exterior"),
      classifiedFrame(1, "living_room"),
      classifiedFrame(2, "kitchen"),
      classifiedFrame(3, "bedroom"),
      classifiedFrame(4, "bathroom"),
      classifiedFrame(5, "corridor"),
      classifiedFrame(6, "utility"),
    ]);

    expect(plan.batches).toHaveLength(7);
    expect(
      plan.batches.find((batch) => batch.evidenceIds.includes("evidence_00"))
        ?.evidenceIds,
    ).toEqual(["evidence_00"]);
    expect(plan.batches.map((batch) => batch.evidenceIds)).toEqual([
      ["evidence_00"],
      ["evidence_01"],
      ["evidence_02"],
      ["evidence_03"],
      ["evidence_04"],
      ["evidence_05"],
      ["evidence_06"],
    ]);
  });

  it("uses a definite indoor room label when the scene label is a window-driven exterior error", () => {
    const plan = planCaptureSet([
      classifiedFrame(0, "unknown", "exterior"),
      classifiedFrame(1, "kitchen", "exterior"),
      classifiedFrame(2, "kitchen", "interior"),
      classifiedFrame(3, "living_room", "exterior"),
    ]);

    expect(plan.batches.map((batch) => batch.evidenceIds)).toEqual([
      ["evidence_00", "evidence_03"],
      ["evidence_01", "evidence_02"],
    ]);
  });
});

function classifiedFrame(
  index: number,
  roomType: NonNullable<EvidenceAsset["visualAnalysis"]>["roomType"],
  sceneType: NonNullable<
    EvidenceAsset["visualAnalysis"]
  >["sceneType"] = "interior",
): EvidenceAsset {
  return {
    ...frame(index),
    visualAnalysis: {
      sceneType,
      roomType,
      floorHint: "unknown",
      roomLabels: [],
      propertyRelevance: "likely",
      addressMatch: "possible",
      connections: roomType === "corridor" ? ["corridor", "door"] : ["door"],
      summary: "Test classification",
      provider: "nvidia_nim",
      model: "test-model",
      confidenceScore: 0.8,
      analyzedAt: new Date(2026, 0, 1).toISOString(),
    },
  };
}

function frame(index: number, extraTags: string[] = []): EvidenceAsset {
  const id = `evidence_${String(index).padStart(2, "0")}`;
  return {
    id,
    caseId: "case_00000001",
    title: id,
    kind: "image",
    sourceProvider: "Responder upload",
    localUrl: `/assets/case_00000001/uploads/${id}.jpg`,
    discoveredAt: new Date(2026, 0, 1, 0, 0, index).toISOString(),
    rights: "operator_owned",
    cachePolicy: "local_allowed",
    redistributable: false,
    validation: "operator_uploaded",
    mimeType: "image/jpeg",
    tags: ["operator-upload", ...extraTags],
    notes: "Test frame",
    capture: {
      projection: "perspective",
      width: 1920,
      height: 1080,
      horizontalCoverageDegrees: 70,
      captureOrder: index,
      overlapSetId: "capture_00000001",
      projectionSource: "operator",
    },
    confidence: {
      score: 0.8,
      verification: "verified",
      geometry: "observed",
      rationale: "Test",
      sourceCount: 1,
    },
  };
}
