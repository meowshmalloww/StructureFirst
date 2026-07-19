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
    const photoBytes = [
      Buffer.from([0xff, 0xd8, 0x10, 0x20, 0xff, 0xd9]),
      Buffer.from([0xff, 0xd8, 0x30, 0x40, 0xff, 0xd9]),
    ];
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
      }>;
      artifact: { mode: string; status: string };
    };
    expect(body.assets).toHaveLength(2);
    expect(body.artifact.mode).toBe("multi_image");
    expect(["running", "failed"]).toContain(body.artifact.status);
    expect(services.store.listEvidence(created.id)).toHaveLength(2);
    expect(workerRequests).toHaveLength(1);
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
): Buffer {
  const chunks: Buffer[] = [];
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
