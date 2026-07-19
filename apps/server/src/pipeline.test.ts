import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { loadConfig } from "./config.js";
import { CaseEventHub } from "./events.js";
import { CasePipeline, type PipelineDependencies } from "./pipeline.js";
import { StructureStore } from "./store.js";

const directories: string[] = [];

afterEach(() => {
  for (const directory of directories.splice(0))
    rmSync(directory, { recursive: true, force: true });
});

describe("case pipeline epistemic boundaries", () => {
  it("completes a sparse address without inventing rooms or routes", async () => {
    const directory = mkdtempSync(resolve(tmpdir(), "structurefirst-test-"));
    directories.push(directory);
    const config = loadConfig({
      repoRoot: directory,
      dataRoot: resolve(directory, "data"),
      casesRoot: resolve(directory, "data/cases"),
      databasePath: ":memory:",
      webDist: resolve(directory, "web"),
      lucidFrameRoot: resolve(directory, "missing-lucidframe"),
      host: "127.0.0.1",
    });
    const store = new StructureStore(":memory:");
    const dependencies: PipelineDependencies = {
      geocode: async () => ({
        displayAddress: "100 Test Avenue, Example City",
        latitude: 34.05,
        longitude: -118.25,
      }),
      findBuilding: async () => undefined,
      discover: async () => [],
    };
    const pipeline = new CasePipeline(
      store,
      new CaseEventHub(),
      config,
      dependencies,
    );
    const created = pipeline.createCase({
      address: "100 Test Avenue",
      role: "fire",
      incidentType: "structure_fire",
    });

    pipeline.start(created.id);
    await waitUntil(() => !pipeline.isRunning(created.id));
    const workspace = store.getWorkspace(created.id);

    expect(workspace?.case.status).toBe("limited_evidence");
    expect(workspace?.case.coverage.interior.band).toBe("unknown");
    expect(workspace?.nodes).toHaveLength(1);
    expect(workspace?.nodes[0]?.kind).toBe("exterior");
    expect(workspace?.edges).toHaveLength(0);
    expect(workspace?.routes).toHaveLength(0);
    expect(workspace?.hazards).toEqual([
      expect.objectContaining({
        category: "intelligence_gap",
        review: "pending",
      }),
    ]);
    store.close();
  });
});

async function waitUntil(predicate: () => boolean): Promise<void> {
  const started = Date.now();
  while (!predicate()) {
    if (Date.now() - started > 2_000)
      throw new Error("Timed out waiting for pipeline.");
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 5));
  }
}
