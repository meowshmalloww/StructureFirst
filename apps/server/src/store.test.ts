import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type {
  Case,
  HazardCandidate,
  ReconstructionArtifact,
} from "@structurefirst/contracts";
import { confidence } from "./lib/confidence.js";
import { StructureStore } from "./store.js";

const directories: string[] = [];

afterEach(() => {
  for (const directory of directories.splice(0))
    rmSync(directory, { recursive: true, force: true });
});

describe("saved case status migration", () => {
  it("marks a reconstructed case ready when its only finding is the automatic interior gap", () => {
    const directory = mkdtempSync(resolve(tmpdir(), "structurefirst-store-"));
    directories.push(directory);
    const databasePath = resolve(directory, "structurefirst.sqlite");
    const now = new Date().toISOString();
    const caseId = "case_ready_test";
    const store = new StructureStore(databasePath);
    store.putCase(testCase(caseId, now));
    store.putArtifact(testArtifact(caseId, now));
    store.putHazard(testGap(caseId));
    store.close();

    const reopened = new StructureStore(databasePath);
    expect(reopened.getCase(caseId)?.status).toBe("briefing_ready");
    reopened.close();
  });
});

function testCase(caseId: string, now: string): Case {
  const unknown = confidence(0, "unknown", "unknown", "Test fixture.", 0);
  return {
    id: caseId,
    addressInput: "100 Test Avenue",
    displayAddress: "100 Test Avenue",
    status: "review_required",
    activeRole: "fire",
    createdAt: now,
    updatedAt: now,
    incident: { type: "other", blockedAreas: [], updatedAt: now },
    coverage: {
      exterior: unknown,
      interior: unknown,
      records: unknown,
      lastAssessedAt: now,
    },
    stages: [],
    operatorNotes: "",
  };
}

function testArtifact(caseId: string, now: string): ReconstructionArtifact {
  return {
    id: "artifact_ready_test",
    caseId,
    evidenceId: "evidence_ready_test",
    status: "ready",
    mode: "single_image",
    splatUrl: "/assets/test/scene.splat",
    gaussianCount: 100,
    modelName: "LucidFrame Apple SHARP",
    modelLicense: "Test fixture",
    createdAt: now,
    updatedAt: now,
    confidence: confidence(0.5, "reconstructed", "derived", "Test fixture.", 1),
  };
}

function testGap(caseId: string): HazardCandidate {
  return {
    id: "hazard_gap_test",
    caseId,
    label: "Interior layout unknown",
    category: "intelligence_gap",
    description: "No verified interior evidence is available.",
    locationLabel: "Entire structure",
    severity: "caution",
    roles: ["fire", "law", "ems", "sar"],
    sourceIds: [],
    confidence: confidence(1, "verified", "derived", "Test fixture.", 0),
    review: "pending",
    reviewNote: "",
  };
}
