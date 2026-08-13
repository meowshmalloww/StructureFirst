import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { loadConfig } from "./config.js";
import { CaseEventHub } from "./events.js";
import { confidence } from "./lib/confidence.js";
import { createId, nowIso } from "./lib/ids.js";
import { CasePipeline } from "./pipeline.js";
import {
  normalizeGeneratedScene,
  inferObservedLevels,
  SceneIntelligenceService,
  spatialEvidenceGroups,
} from "./scene-intelligence.js";

import type { EvidenceAsset, RoomType } from "@structurefirst/contracts";
import { SettingsService } from "./settings.js";
import { StructureStore } from "./store.js";

const directories: string[] = [];

describe("visual classification normalization", () => {
  it("corrects window-driven exterior and laundry-as-kitchen errors", () => {
    expect(
      normalizeGeneratedScene({
        sceneType: "exterior",
        roomType: "kitchen",
        floorHint: "unknown",
        roomLabels: [],
        propertyRelevance: "likely",
        observedAddress: "",
        connections: ["door"],
        summary: "A washer and dryer are visible beside a utility sink.",
        confidenceScore: 0.6,
      }),
    ).toMatchObject({ sceneType: "interior", roomType: "utility" });
  });

  it("recovers a strongly evidenced room when the model contradicts its own summary", () => {
    expect(
      normalizeGeneratedScene({
        sceneType: "unknown",
        roomType: "unknown",
        floorHint: "unknown",
        roomLabels: [],
        propertyRelevance: "unlikely",
        observedAddress: "",
        connections: [],
        summary:
          "The image shows a kitchen with a stove, sink, refrigerator, cabinets, and a window.",
        confidenceScore: 0.5,
      }),
    ).toMatchObject({
      sceneType: "interior",
      roomType: "kitchen",
      propertyRelevance: "likely",
    });
  });

  it("does not turn unrelated imagery into a room", () => {
    expect(
      normalizeGeneratedScene({
        sceneType: "non_property",
        roomType: "unknown",
        floorHint: "unknown",
        roomLabels: [],
        propertyRelevance: "unlikely",
        observedAddress: "",
        connections: [],
        summary: "A cat resting on a wooden floor.",
        confidenceScore: 0.9,
      }),
    ).toMatchObject({ sceneType: "non_property", roomType: "unknown" });
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  for (const directory of directories.splice(0))
    rmSync(directory, { recursive: true, force: true });
});

describe("scene intelligence", () => {
  it("derives conservative observed levels from calibrated camera elevation", () => {
    const poses = [
      cameraPose("lower_a", [0, 0, 0]),
      cameraPose("lower_b", [1, 0.15, 0.4]),
      cameraPose("upper_a", [0.2, -2.75, 0.3]),
      cameraPose("upper_b", [1.1, -2.9, 0.5]),
    ];
    const levels = inferObservedLevels(poses);

    expect(Object.fromEntries(levels)).toEqual({
      lower_a: 0,
      lower_b: 0,
      upper_a: 1,
      upper_b: 1,
    });

    const evidence = [
      classifiedFrame("lower_a", "bedroom", 0),
      classifiedFrame("lower_b", "bedroom", 1),
      classifiedFrame("upper_a", "bedroom", 2),
      classifiedFrame("upper_b", "bedroom", 3),
    ];
    const groups = spatialEvidenceGroups(
      evidence,
      evidence.map((item) => item.id),
      [
        { frameA: 0, frameB: 1, confidence: 0.9 },
        { frameA: 1, frameB: 2, confidence: 0.9 },
        { frameA: 2, frameB: 3, confidence: 0.9 },
      ],
      new Map(poses.map((pose) => [pose.evidenceId, pose.position])),
      levels,
    );

    expect(groups.map((group) => group.observedLevel)).toEqual([0, 1]);
    expect(
      groups.map((group) => group.evidence.map((item) => item.id)),
    ).toEqual([
      ["lower_a", "lower_b"],
      ["upper_a", "upper_b"],
    ]);
  });

  it("keeps two bedrooms distinct when their verified overlap paths do not join", () => {
    const evidence = [
      classifiedFrame("frame_a", "bedroom", 0),
      classifiedFrame("frame_b", "bedroom", 1),
      classifiedFrame("hallway", "corridor", 2),
      classifiedFrame("frame_c", "bedroom", 3),
      classifiedFrame("frame_d", "bedroom", 4),
    ];
    const groups = spatialEvidenceGroups(
      evidence,
      evidence.map((item) => item.id),
      [
        { frameA: 0, frameB: 1, confidence: 0.9 },
        { frameA: 1, frameB: 2, confidence: 0.8 },
        { frameA: 2, frameB: 3, confidence: 0.8 },
        { frameA: 3, frameB: 4, confidence: 0.9 },
      ],
      new Map(),
    );

    expect(
      groups.map((group) => group.evidence.map((item) => item.id)),
    ).toEqual([["frame_a", "frame_b"], ["hallway"], ["frame_c", "frame_d"]]);
  });

  it("classifies permitted imagery and builds a floor-aware room node", async () => {
    const directory = mkdtempSync(resolve(tmpdir(), "structurefirst-scene-"));
    directories.push(directory);
    const casesRoot = resolve(directory, "cases");
    const config = loadConfig({
      repoRoot: directory,
      dataRoot: directory,
      casesRoot,
      databasePath: ":memory:",
      webDist: resolve(directory, "web"),
      host: "127.0.0.1",
    });
    const store = new StructureStore(":memory:");
    const settings = new SettingsService(store, config);
    settings.saveProvider("nvidia_nim", {
      baseUrl: "https://integrate.api.nvidia.com/v1",
      model: "meta/llama-3.2-11b-vision-instruct",
      enabled: true,
      vision: true,
      apiKey: "test-secret-key",
      clearKey: false,
    });
    const created = new CasePipeline(
      store,
      new CaseEventHub(),
      config,
    ).createCase({
      address: "100 Test Avenue",
      role: "fire",
      incidentType: "other",
    });
    const evidenceId = createId("evidence");
    const uploadDirectory = resolve(casesRoot, created.id, "uploads");
    mkdirSync(uploadDirectory, { recursive: true });
    const imageName = "room.png";
    writeFileSync(
      resolve(uploadDirectory, imageName),
      Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
        "base64",
      ),
    );
    store.putEvidence({
      id: evidenceId,
      caseId: created.id,
      title: "Responder room photo",
      kind: "image",
      sourceProvider: "Responder upload",
      localUrl: `/assets/${created.id}/uploads/${imageName}`,
      discoveredAt: nowIso(),
      rights: "operator_owned",
      cachePolicy: "local_allowed",
      redistributable: false,
      validation: "operator_uploaded",
      mimeType: "image/png",
      tags: ["operator-upload", "property-photo"],
      notes: "Responder supplied this image for the submitted property.",
      confidence: confidence(0.76, "verified", "observed", "Test image", 1),
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        Response.json({
          choices: [
            {
              message: {
                content: `${JSON.stringify({
                  sceneType: "interior",
                  roomType: "bedroom",
                  floorHint: "upper",
                  propertyRelevance: "likely",
                  observedAddress: "possible",
                  connections: ["door", "window"],
                  summary:
                    "An upstairs bedroom with one visible door and window.",
                  confidenceScore: 0.88,
                })}\nClassification complete.`,
              },
            },
          ],
        }),
      ),
    );
    const service = new SceneIntelligenceService(store, settings, config);

    const result = await service.analyzeCase(created.id);

    expect(result).toMatchObject({ analyzed: 1, rejected: 0 });
    expect(store.getEvidence(evidenceId)).toMatchObject({
      visualAnalysis: {
        sceneType: "interior",
        roomType: "bedroom",
        floorHint: "upper",
        addressMatch: "possible",
        observedAddress: "",
      },
    });
    const artifactId = createId("artifact");
    store.putArtifact({
      id: artifactId,
      caseId: created.id,
      evidenceId,
      evidenceIds: [evidenceId],
      status: "ready",
      mode: "single_image",
      splatUrl: `/assets/${created.id}/reconstruction/test/scene.splat`,
      gaussianCount: 1_000,
      modelName: "Test",
      modelLicense: "Test",
      geometry: {
        backend: "sharp_single_view",
        coordinateFrame: "anchor_camera_metric_opencv",
        jointCameraAccepted: false,
        cameraPoses: [
          {
            evidenceId,
            sourceIndex: 0,
            position: [1, 2, 3],
            rotationWxyz: [1, 0, 0, 0],
            scale: 1,
            placement: "anchor",
            confidenceScore: 0.5,
          },
        ],
      },
      createdAt: nowIso(),
      updatedAt: nowIso(),
      confidence: confidence(0.5, "reconstructed", "derived", "Test", 1),
    });
    service.rebuildSpatialGraph(created.id);

    expect(store.listNodes(created.id)).toEqual([
      expect.objectContaining({
        label: "Bedroom · Upper floor",
        kind: "room",
        artifactId,
        coordinateFrameId: artifactId,
        roomType: "bedroom",
        floorLabel: "Upper floor",
        position: [1, 2, 3],
        sourceIds: [evidenceId],
      }),
    ]);
    store.close();
  });
});

function cameraPose(evidenceId: string, position: [number, number, number]) {
  return {
    evidenceId,
    sourceIndex: 0,
    position,
    rotationWxyz: [1, 0, 0, 0] as [number, number, number, number],
    scale: 1,
    placement: "measured_feature_and_sharp_metric" as const,
    confidenceScore: 0.8,
  };
}

function classifiedFrame(
  id: string,
  roomType: RoomType,
  captureOrder: number,
): EvidenceAsset {
  return {
    id,
    caseId: "case_room_graph",
    title: id,
    kind: "image",
    sourceProvider: "Responder upload",
    localUrl: `/assets/case_room_graph/uploads/${id}.jpg`,
    discoveredAt: nowIso(),
    rights: "operator_owned",
    cachePolicy: "local_allowed",
    redistributable: false,
    validation: "operator_uploaded",
    mimeType: "image/jpeg",
    tags: ["operator-upload"],
    notes: "Test capture",
    capture: {
      projection: "perspective",
      width: 1920,
      height: 1080,
      horizontalCoverageDegrees: 70,
      captureOrder,
      overlapSetId: "capture_room_graph",
      projectionSource: "operator",
    },
    visualAnalysis: {
      sceneType: "interior",
      roomType,
      floorHint: "ground",
      propertyRelevance: "likely",
      addressMatch: "possible",
      connections: roomType === "corridor" ? ["corridor", "door"] : ["door"],
      summary: `Visible ${roomType}.`,
      provider: "nvidia_nim",
      model: "test-model",
      confidenceScore: 0.9,
      analyzedAt: nowIso(),
    },
    confidence: confidence(0.8, "verified", "observed", "Test", 1),
  };
}
