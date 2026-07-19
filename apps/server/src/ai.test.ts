import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import type { AiProviderId } from "@structurefirst/contracts";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AiCaseAnalyzer } from "./ai.js";
import { loadConfig } from "./config.js";
import { CaseEventHub } from "./events.js";
import { confidence } from "./lib/confidence.js";
import { createId, nowIso } from "./lib/ids.js";
import { CasePipeline } from "./pipeline.js";
import { SettingsService } from "./settings.js";
import { StructureStore } from "./store.js";

const directories: string[] = [];

afterEach(() => {
  vi.unstubAllGlobals();
  for (const directory of directories.splice(0))
    rmSync(directory, { recursive: true, force: true });
});

const providers: Array<[AiProviderId, string]> = [
  ["groq", "https://api.groq.com/openai/v1"],
  ["cerebras", "https://api.cerebras.ai/v1"],
  ["openrouter", "https://openrouter.ai/api/v1"],
  ["nvidia_nim", "https://integrate.api.nvidia.com/v1"],
];

describe.each(providers)("%s AI adapter", (provider, baseUrl) => {
  it("stores only source-linked findings as pending review", async () => {
    const directory = mkdtempSync(resolve(tmpdir(), "structurefirst-ai-"));
    directories.push(directory);
    const store = new StructureStore(":memory:");
    const config = loadConfig({
      repoRoot: directory,
      dataRoot: resolve(directory, "data"),
      casesRoot: resolve(directory, "data/cases"),
      databasePath: ":memory:",
      webDist: resolve(directory, "web"),
      host: "127.0.0.1",
    });
    const settings = new SettingsService(store, config);
    settings.saveProvider(provider, {
      baseUrl,
      model: "test-model",
      enabled: true,
      vision: false,
      apiKey: "test-secret-key",
      clearKey: false,
    });
    const pipeline = new CasePipeline(store, new CaseEventHub(), config);
    const caseValue = pipeline.createCase({
      address: "100 Test Avenue",
      role: "fire",
      incidentType: "structure_fire",
    });
    const evidenceId = createId("evidence");
    store.putEvidence({
      id: evidenceId,
      caseId: caseValue.id,
      title: "Responder exterior photo",
      kind: "image",
      sourceProvider: "Operator upload",
      discoveredAt: nowIso(),
      rights: "operator_owned",
      cachePolicy: "local_allowed",
      redistributable: false,
      validation: "operator_uploaded",
      tags: ["exterior"],
      notes: "Smoke is visible near the roof line.",
      confidence: confidence(
        0.76,
        "verified",
        "observed",
        "Operator supplied this file.",
        1,
      ),
    });

    const fetchMock = vi.fn(async () =>
      Response.json({
        choices: [
          {
            message: {
              content: JSON.stringify({
                summary: "One supported candidate and one invalid citation.",
                hazards: [
                  {
                    label: "Possible roof-line smoke",
                    category: "fire_spread",
                    description: "The supplied note reports visible smoke.",
                    locationLabel: "Roof line",
                    severity: "caution",
                    roles: ["fire"],
                    sourceIds: [evidenceId],
                  },
                  {
                    label: "Unsupported room claim",
                    category: "other",
                    description: "This citation does not exist.",
                    locationLabel: "Unknown",
                    severity: "advisory",
                    roles: ["fire"],
                    sourceIds: ["evidence_fabricated"],
                  },
                ],
              }),
            },
          },
        ],
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await new AiCaseAnalyzer(store, settings, config).analyze(
      caseValue.id,
      { provider, includeImage: false },
    );

    expect(result).toMatchObject({
      provider,
      model: "test-model",
      hazardsAdded: 1,
      usedImage: false,
    });
    expect(fetchMock).toHaveBeenCalledWith(
      `${baseUrl}/chat/completions`,
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          authorization: "Bearer test-secret-key",
        }),
      }),
    );
    expect(store.listHazards(caseValue.id)).toEqual([
      expect.objectContaining({
        label: "Possible roof-line smoke",
        sourceIds: [evidenceId],
        review: "pending",
      }),
    ]);
    expect(store.getCase(caseValue.id)?.status).toBe("review_required");
    store.close();
  });
});
