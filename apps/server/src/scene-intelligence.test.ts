import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { loadConfig } from "./config.js";
import { CaseEventHub } from "./events.js";
import { confidence } from "./lib/confidence.js";
import { createId, nowIso } from "./lib/ids.js";
import { CasePipeline } from "./pipeline.js";
import { SceneIntelligenceService } from "./scene-intelligence.js";
import { SettingsService } from "./settings.js";
import { StructureStore } from "./store.js";

const directories: string[] = [];

afterEach(() => {
  vi.unstubAllGlobals();
  for (const directory of directories.splice(0))
    rmSync(directory, { recursive: true, force: true });
});

describe("scene intelligence", () => {
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
    store.putArtifact({
      id: createId("artifact"),
      caseId: created.id,
      evidenceId,
      evidenceIds: [evidenceId],
      status: "ready",
      mode: "single_image",
      splatUrl: `/assets/${created.id}/reconstruction/test/scene.splat`,
      gaussianCount: 1_000,
      modelName: "Test",
      modelLicense: "Test",
      createdAt: nowIso(),
      updatedAt: nowIso(),
      confidence: confidence(0.5, "reconstructed", "derived", "Test", 1),
    });
    service.rebuildSpatialGraph(created.id);

    expect(store.listNodes(created.id)).toEqual([
      expect.objectContaining({
        label: "Bedroom · Upper floor",
        kind: "room",
        floorLabel: "Upper floor",
        sourceIds: [evidenceId],
      }),
    ]);
    store.close();
  });
});
