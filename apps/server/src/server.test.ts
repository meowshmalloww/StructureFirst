import { createHash } from "node:crypto";
import {
  existsSync,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import sharp from "sharp";
import { afterEach, describe, expect, it, vi } from "vitest";
import { buildServer } from "./server.js";

const directories: string[] = [];

afterEach(() => {
  vi.unstubAllGlobals();
  for (const directory of directories.splice(0))
    rmSync(directory, { recursive: true, force: true });
});

describe("property routes", () => {
  it("deletes the property record and its local directory", async () => {
    const directory = testDirectory();
    const { app, services } = await testServer(directory);
    const created = services.casePipeline.createCase({
      address: "100 Test Avenue",
      role: "fire",
      incidentType: "other",
    });
    const caseDirectory = resolve(directory, "cases", created.id);
    mkdirSync(caseDirectory, { recursive: true });
    writeFileSync(resolve(caseDirectory, "photo.jpg"), "test");

    const response = await app.inject({
      method: "DELETE",
      url: `/api/cases/${created.id}`,
      headers: { "x-structurefirst-intent": "operator-action" },
    });

    expect(response.statusCode).toBe(200);
    expect(response.json()).toEqual({ deleted: true });
    expect(services.store.getCase(created.id)).toBeUndefined();
    expect(existsSync(caseDirectory)).toBe(false);
    await app.close();
  });

  it("streams multiple photos and starts one grouped reconstruction", async () => {
    const directory = testDirectory();
    const { app, services } = await testServer(directory);
    const created = services.casePipeline.createCase({
      address: "200 Test Avenue",
      role: "fire",
      incidentType: "other",
    });
    const photoBytes = await Promise.all([
      sharp({
        create: {
          width: 8,
          height: 6,
          channels: 3,
          background: { r: 90, g: 110, b: 130 },
        },
      })
        .jpeg()
        .toBuffer(),
      sharp({
        create: {
          width: 8,
          height: 6,
          channels: 3,
          background: { r: 120, g: 100, b: 80 },
        },
      })
        .jpeg()
        .toBuffer(),
    ]);
    const workerRequests: Array<Record<string, unknown>> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: string | URL | Request, init?: RequestInit) => {
        workerRequests.push(
          JSON.parse(String(init?.body)) as Record<string, unknown>,
        );
        return new Response("worker deliberately unavailable in this test", {
          status: 503,
        });
      }),
    );
    const boundary = "structurefirst-test-boundary";
    const response = await app.inject({
      method: "POST",
      url: `/api/cases/${created.id}/photos`,
      headers: {
        "content-type": `multipart/form-data; boundary=${boundary}`,
        "x-structurefirst-intent": "operator-action",
      },
      payload: multipartBody(boundary, [
        { name: "front.jpg", bytes: photoBytes[0]! },
        { name: "side.jpg", bytes: photoBytes[1]! },
      ]),
    });

    expect(response.statusCode).toBe(201);
    const body = response.json() as {
      assets: Array<{
        localUrl: string;
        sha256: string;
        byteSize: number;
        capture: {
          projection: string;
          width: number;
          height: number;
          captureOrder: number;
        };
      }>;
      processing: { status: string; totalFiles: number; planCandidates: number };
    };
    expect(body.assets).toHaveLength(2);
    expect(body.assets.map((asset) => asset.capture)).toEqual([
      expect.objectContaining({
        projection: "perspective",
        width: 8,
        height: 6,
        captureOrder: 0,
      }),
      expect.objectContaining({
        projection: "perspective",
        width: 8,
        height: 6,
        captureOrder: 1,
      }),
    ]);
    expect(body.processing).toEqual({
      status: "organizing",
      totalFiles: 2,
      planCandidates: 0,
    });
    expect(services.store.listEvidence(created.id)).toHaveLength(2);
    await vi.waitFor(() => expect(workerRequests).toHaveLength(1));
    expect(services.store.listArtifacts(created.id)[0]?.mode).toBe("multi_image");
    const expectedHashes = photoBytes.map((bytes) =>
      createHash("sha256").update(bytes).digest("hex"),
    );
    expect(workerRequests[0]?.input_sha256s).toEqual(expectedHashes);
    expect(workerRequests[0]?.input_sha256).toBe(expectedHashes[0]);
    for (const [index, asset] of body.assets.entries()) {
      const suffix = asset.localUrl.slice("/assets/".length).split("/");
      const stored = readFileSync(resolve(directory, "cases", ...suffix));
      expect(stored).toEqual(photoBytes[index]);
      expect(asset.sha256).toBe(expectedHashes[index]);
      expect(asset.byteSize).toBe(photoBytes[index]?.byteLength);
    }
    await app.close();
  });

  it("classifies the complete upload before unrelated media can enter reconstruction", async () => {
    const directory = testDirectory();
    const { app, services } = await testServer(directory);
    const created = services.casePipeline.createCase({
      address: "225 Test Avenue",
      role: "fire",
      incidentType: "other",
    });
    const room = await sharp({
      create: {
        width: 24,
        height: 18,
        channels: 3,
        background: { r: 90, g: 110, b: 130 },
      },
    })
      .jpeg()
      .toBuffer();
    const cat = await sharp({
      create: {
        width: 24,
        height: 18,
        channels: 3,
        background: { r: 190, g: 120, b: 45 },
      },
    })
      .jpeg()
      .toBuffer();
    const analyze = vi
      .spyOn(services.sceneIntelligence, "analyzeCase")
      .mockImplementation(async (_caseId, evidenceIds) => {
        for (const id of evidenceIds ?? []) {
          const evidence = services.store.getEvidence(id);
          if (evidence?.title === "cat.jpg") {
            services.store.putEvidence({
              ...evidence,
              tags: [...evidence.tags, "reconstruction-excluded"],
            });
          }
        }
        return { analyzed: 2, rejected: 1, warnings: [] };
      });
    const queue = vi
      .spyOn(services.reconstruction, "queueCaptureSet")
      .mockResolvedValue(undefined as never);
    const boundary = "structurefirst-classify-before-queue";

    const response = await app.inject({
      method: "POST",
      url: `/api/cases/${created.id}/photos`,
      headers: {
        "content-type": `multipart/form-data; boundary=${boundary}`,
        "x-structurefirst-intent": "operator-action",
      },
      payload: multipartBody(boundary, [
        { name: "room.jpg", bytes: room },
        { name: "cat.jpg", bytes: cat },
      ]),
    });

    expect(response.statusCode).toBe(201);
    await vi.waitFor(() => expect(queue).toHaveBeenCalledOnce());
    expect(analyze.mock.invocationCallOrder[0]).toBeLessThan(
      queue.mock.invocationCallOrder[0]!,
    );
    const queuedIds = queue.mock.calls[0]?.[1] ?? [];
    expect(queuedIds).toHaveLength(1);
    expect(services.store.getEvidence(queuedIds[0]!)?.title).toBe("room.jpg");
    expect(
      services.store
        .listEvidence(created.id)
        .find((item) => item.title === "cat.jpg")?.tags,
    ).toContain("reconstruction-excluded");
    await app.close();
  });

  it("plans large manual capture sets as overlapping reconstruction jobs", async () => {
    const directory = testDirectory();
    const { app, services } = await testServer(directory);
    const created = services.casePipeline.createCase({
      address: "250 Test Avenue",
      role: "fire",
      incidentType: "other",
    });
    const photo = await sharp({
      create: {
        width: 8,
        height: 6,
        channels: 3,
        background: { r: 100, g: 110, b: 120 },
      },
    })
      .jpeg()
      .toBuffer();
    const workerRequests: Array<Record<string, unknown>> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: string | URL | Request, init?: RequestInit) => {
        workerRequests.push(
          JSON.parse(String(init?.body)) as Record<string, unknown>,
        );
        return new Response("worker deliberately unavailable in this test", {
          status: 503,
        });
      }),
    );
    const boundary = "structurefirst-large-capture";
    const response = await app.inject({
      method: "POST",
      url: `/api/cases/${created.id}/photos`,
      headers: {
        "content-type": `multipart/form-data; boundary=${boundary}`,
        "x-structurefirst-intent": "operator-action",
      },
      payload: multipartBody(
        boundary,
        Array.from({ length: 13 }, (_, index) => ({
          name: `room-${String(index).padStart(2, "0")}.jpg`,
          bytes: photo,
        })),
      ),
    });

    expect(response.statusCode).toBe(201);
    const body = response.json() as {
      processing: { totalFiles: number; planCandidates: number };
    };
    expect(body.processing).toEqual({
      status: "organizing",
      totalFiles: 13,
      planCandidates: 0,
    });
    await vi.waitFor(() => expect(workerRequests).toHaveLength(2));
    expect(
      services.store
        .listArtifacts(created.id)
        .map((artifact) => artifact.evidenceIds?.length),
    ).toEqual(expect.arrayContaining([12, 4]));
    await app.close();
  });

  it("routes an explicit full-resolution 360 capture to SHARP-360", async () => {
    const directory = testDirectory();
    const { app, services } = await testServer(directory);
    const created = services.casePipeline.createCase({
      address: "300 Test Avenue",
      role: "fire",
      incidentType: "other",
    });
    const photo = await sharp({
      create: {
        width: 16,
        height: 8,
        channels: 3,
        background: { r: 60, g: 80, b: 100 },
      },
    })
      .jpeg()
      .toBuffer();
    const workerRequests: Array<Record<string, unknown>> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: string | URL | Request, init?: RequestInit) => {
        workerRequests.push(
          JSON.parse(String(init?.body)) as Record<string, unknown>,
        );
        return new Response("worker deliberately unavailable in this test", {
          status: 503,
        });
      }),
    );
    const boundary = "structurefirst-panorama-boundary";

    const response = await app.inject({
      method: "POST",
      url: `/api/cases/${created.id}/photos`,
      headers: {
        "content-type": `multipart/form-data; boundary=${boundary}`,
        "x-structurefirst-intent": "operator-action",
      },
      payload: multipartBody(
        boundary,
        [{ name: "room-360.jpg", bytes: photo }],
        { captureMode: "panorama_360" },
      ),
    });

    expect(response.statusCode).toBe(201);
    const body = response.json() as {
      assets: Array<{
        capture: {
          projection: string;
          horizontalCoverageDegrees: number;
          projectionSource: string;
        };
      }>;
      processing: { status: string; totalFiles: number };
    };
    expect(body.processing).toEqual(
      expect.objectContaining({ status: "organizing", totalFiles: 1 }),
    );
    expect(body.assets[0]?.capture).toEqual(
      expect.objectContaining({
        projection: "panorama_360",
        horizontalCoverageDegrees: 360,
        projectionSource: "operator",
      }),
    );
    await vi.waitFor(() => expect(workerRequests).toHaveLength(1));
    expect(workerRequests[0]?.mode).toBe("panorama");
    await app.close();
  });

  it("routes an explicit 1:1 half-sphere 180 capture to SHARP panorama reconstruction", async () => {
    const directory = testDirectory();
    const { app, services } = await testServer(directory);
    const created = services.casePipeline.createCase({
      address: "400 Test Avenue",
      role: "fire",
      incidentType: "other",
    });
    const photo = await sharp({
      create: {
        width: 8,
        height: 8,
        channels: 3,
        background: { r: 80, g: 100, b: 120 },
      },
    })
      .jpeg()
      .toBuffer();
    const workerRequests: Array<Record<string, unknown>> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: string | URL | Request, init?: RequestInit) => {
        workerRequests.push(
          JSON.parse(String(init?.body)) as Record<string, unknown>,
        );
        return new Response("worker deliberately unavailable in this test", {
          status: 503,
        });
      }),
    );
    const boundary = "structurefirst-half-panorama-boundary";

    const response = await app.inject({
      method: "POST",
      url: `/api/cases/${created.id}/photos`,
      headers: {
        "content-type": `multipart/form-data; boundary=${boundary}`,
        "x-structurefirst-intent": "operator-action",
      },
      payload: multipartBody(
        boundary,
        [{ name: "room-180.jpg", bytes: photo }],
        { captureMode: "panorama_180" },
      ),
    });

    expect(response.statusCode).toBe(201);
    const body = response.json() as {
      assets: Array<{
        capture: {
          projection: string;
          horizontalCoverageDegrees: number;
          projectionSource: string;
        };
      }>;
      processing: { status: string; totalFiles: number };
    };
    expect(body.processing).toEqual(
      expect.objectContaining({ status: "organizing", totalFiles: 1 }),
    );
    expect(body.assets[0]?.capture).toEqual(
      expect.objectContaining({
        projection: "panorama_180",
        horizontalCoverageDegrees: 180,
        projectionSource: "operator",
      }),
    );
    await vi.waitFor(() => expect(workerRequests).toHaveLength(1));
    expect(workerRequests[0]?.mode).toBe("panorama");
    await app.close();
  });

  it("separates a floorplan from photos and starts a structural artifact", async () => {
    const directory = testDirectory();
    const { app, services } = await testServer(directory);
    const created = services.casePipeline.createCase({
      address: "450 Test Avenue",
      role: "fire",
      incidentType: "other",
    });
    const planSvg = Buffer.from(
      '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="white"/><g fill="none" stroke="#111" stroke-width="7"><rect x="70" y="60" width="500" height="350"/><path d="M320 60V410M70 235H570"/></g></svg>',
    );
    const plan = await sharp(planSvg).jpeg().toBuffer();
    const photoPixels = Buffer.alloc(640 * 480 * 3);
    for (let index = 0; index < photoPixels.length; index += 3) {
      const pixel = index / 3;
      photoPixels[index] = pixel % 251;
      photoPixels[index + 1] = Math.floor(pixel / 640) % 241;
      photoPixels[index + 2] = (pixel * 7) % 239;
    }
    const photo = await sharp(photoPixels, {
      raw: { width: 640, height: 480, channels: 3 },
    })
      .jpeg()
      .toBuffer();
    const workerRequests: Array<Record<string, unknown>> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: string | URL | Request, init?: RequestInit) => {
        workerRequests.push(
          JSON.parse(String(init?.body)) as Record<string, unknown>,
        );
        return new Response("worker deliberately unavailable in this test", {
          status: 503,
        });
      }),
    );
    const boundary = "structurefirst-plan-separation";
    const response = await app.inject({
      method: "POST",
      url: `/api/cases/${created.id}/photos`,
      headers: {
        "content-type": `multipart/form-data; boundary=${boundary}`,
        "x-structurefirst-intent": "operator-action",
      },
      payload: multipartBody(boundary, [
        { name: "floor-plan.jpg", bytes: plan },
        { name: "room.jpg", bytes: photo },
      ]),
    });

    expect(response.statusCode).toBe(201);
    expect(response.json()).toEqual(
      expect.objectContaining({
        processing: expect.objectContaining({
          totalFiles: 2,
          planCandidates: 1,
        }),
      }),
    );
    expect(
      services.store.listEvidence(created.id).filter((item) => item.kind === "blueprint"),
    ).toHaveLength(1);
    await vi.waitFor(() => expect(workerRequests).toHaveLength(2));
    expect(workerRequests.map((request) => request.mode)).toEqual([
      "floorplan",
      "single_image",
    ]);
    expect(workerRequests[0]?.input_paths).toHaveLength(1);
    await app.close();
  });

  it("rejects a wide image mislabeled as a 180 half-sphere panorama", async () => {
    const directory = testDirectory();
    const { app, services } = await testServer(directory);
    const created = services.casePipeline.createCase({
      address: "500 Test Avenue",
      role: "fire",
      incidentType: "other",
    });
    const photo = await sharp({
      create: {
        width: 16,
        height: 8,
        channels: 3,
        background: { r: 100, g: 80, b: 60 },
      },
    })
      .jpeg()
      .toBuffer();
    const boundary = "structurefirst-invalid-half-panorama-boundary";

    const response = await app.inject({
      method: "POST",
      url: `/api/cases/${created.id}/photos`,
      headers: {
        "content-type": `multipart/form-data; boundary=${boundary}`,
        "x-structurefirst-intent": "operator-action",
      },
      payload: multipartBody(
        boundary,
        [{ name: "not-a-half-sphere.jpg", bytes: photo }],
        { captureMode: "panorama_180" },
      ),
    });

    expect(response.statusCode).toBe(400);
    expect(response.json()).toEqual(
      expect.objectContaining({
        error:
          "A 180° half-sphere panorama must use a 1:1 equirectangular layout.",
      }),
    );
    expect(services.store.listEvidence(created.id)).toHaveLength(0);
    await app.close();
  });
});

function testDirectory(): string {
  const directory = mkdtempSync(
    resolve(tmpdir(), "structurefirst-server-test-"),
  );
  directories.push(directory);
  return directory;
}

async function testServer(directory: string) {
  return buildServer({
    repoRoot: directory,
    dataRoot: directory,
    casesRoot: resolve(directory, "cases"),
    databasePath: ":memory:",
    webDist: resolve(directory, "web"),
    reconstructionUrl: "http://127.0.0.1:9",
    host: "127.0.0.1",
    cookieSecret: "test-cookie-secret",
  });
}

function multipartBody(
  boundary: string,
  files: Array<{ name: string; bytes: Buffer }>,
  fields: Record<string, string> = {},
): Buffer {
  const chunks: Buffer[] = [];
  for (const [name, value] of Object.entries(fields)) {
    chunks.push(
      Buffer.from(
        `--${boundary}\r\nContent-Disposition: form-data; name="${name}"\r\n\r\n${value}\r\n`,
      ),
    );
  }
  for (const file of files) {
    chunks.push(
      Buffer.from(
        `--${boundary}\r\nContent-Disposition: form-data; name="files"; filename="${file.name}"\r\nContent-Type: image/jpeg\r\n\r\n`,
      ),
      file.bytes,
      Buffer.from("\r\n"),
    );
  }
  chunks.push(Buffer.from(`--${boundary}--\r\n`));
  return Buffer.concat(chunks);
}
