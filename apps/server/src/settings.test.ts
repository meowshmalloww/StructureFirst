import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import type { AiProviderId } from "@structurefirst/contracts";
import { afterEach, describe, expect, it, vi } from "vitest";
import { loadConfig } from "./config.js";
import { SettingsService } from "./settings.js";
import { StructureStore } from "./store.js";

const directories: string[] = [];

afterEach(() => {
  vi.unstubAllGlobals();
  for (const directory of directories.splice(0))
    rmSync(directory, { recursive: true, force: true });
});

describe("provider settings", () => {
  it("encrypts API keys and returns only a hint to the browser", () => {
    const directory = mkdtempSync(
      resolve(tmpdir(), "structurefirst-settings-"),
    );
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
    const saved = settings.saveProvider("groq", {
      baseUrl: "https://api.groq.com/openai/v1",
      model: "test-model",
      enabled: true,
      vision: false,
      apiKey: "gsk-secret-example-1234",
      clearKey: false,
    });

    expect(saved).toMatchObject({
      configured: true,
      keyHint: "••••1234",
    });
    const raw = store.database
      .prepare("SELECT payload FROM settings WHERE key = ?")
      .get("provider:groq") as { payload: string };
    expect(raw.payload).not.toContain("gsk-secret-example-1234");
    expect(settings.credential("groq")).toMatchObject({
      apiKey: "gsk-secret-example-1234",
      model: "test-model",
    });
    store.close();
  });

  it("rejects non-official remote provider endpoints", () => {
    const directory = mkdtempSync(
      resolve(tmpdir(), "structurefirst-settings-"),
    );
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
    expect(() =>
      settings.saveProvider("openrouter", {
        baseUrl: "http://127.0.0.1:9999/v1",
        model: "test",
        enabled: true,
        vision: false,
        apiKey: "secret",
        clearKey: false,
      }),
    ).toThrow("official HTTPS endpoint");
    store.close();
  });

  it.each<
    [
      AiProviderId,
      string,
      string,
      "free_plan" | "free_models" | "prototype",
      Array<Record<string, unknown>>,
      boolean,
    ]
  >([
    [
      "groq",
      "https://api.groq.com/openai/v1/models",
      "llama-3.3-70b-versatile",
      "free_plan",
      [
        {
          id: "llama-3.3-70b-versatile",
          owned_by: "Meta",
          active: true,
          context_window: 131072,
        },
        { id: "whisper-large-v3", active: true },
      ],
      false,
    ],
    [
      "cerebras",
      "https://api.cerebras.ai/v1/models",
      "gpt-oss-120b",
      "free_plan",
      [{ id: "gpt-oss-120b", owned_by: "Cerebras" }],
      false,
    ],
    [
      "openrouter",
      "https://openrouter.ai/api/v1/models/user?output_modalities=text",
      "openrouter/free",
      "free_models",
      [
        {
          id: "openrouter/free",
          name: "Free Models Router",
          architecture: {
            input_modalities: ["text", "image"],
            output_modalities: ["text"],
          },
          pricing: { prompt: "0", completion: "0" },
        },
        {
          id: "vendor/paid-chat",
          architecture: { output_modalities: ["text"] },
          pricing: { prompt: "0.001", completion: "0.002" },
        },
      ],
      true,
    ],
    [
      "nvidia_nim",
      "https://integrate.api.nvidia.com/v1/models",
      "meta/llama-3.2-11b-vision-instruct",
      "prototype",
      [
        { id: "meta/llama-3.2-11b-vision-instruct", owned_by: "Meta" },
        { id: "baai/bge-m3", owned_by: "BAAI" },
      ],
      true,
    ],
  ])(
    "loads the filtered %s no-cost chat catalog",
    async (provider, endpoint, expectedId, catalogKind, entries, vision) => {
      const directory = mkdtempSync(
        resolve(tmpdir(), "structurefirst-settings-"),
      );
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
      const fetchMock = vi.fn(
        async (input: string | URL | Request, init?: RequestInit) => {
          expect(String(input)).toBe(endpoint);
          expect(init?.headers).toMatchObject({
            authorization: "Bearer provider-test-key",
          });
          return Response.json({
            data: entries,
          });
        },
      );
      vi.stubGlobal("fetch", fetchMock);

      const result = await new SettingsService(store, config).loadModels(
        provider,
        "provider-test-key",
      );

      expect(result.catalogKind).toBe(catalogKind);
      expect(result.notice).toBeTruthy();
      expect(result.models).toHaveLength(1);
      expect(result.models[0]).toMatchObject({ id: expectedId, vision });
      store.close();
    },
  );

  it("uses the encrypted saved key when refreshing models", async () => {
    const directory = mkdtempSync(
      resolve(tmpdir(), "structurefirst-settings-"),
    );
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
    settings.saveProvider("groq", {
      baseUrl: "https://api.groq.com/openai/v1",
      model: "llama-3.1-8b-instant",
      enabled: true,
      vision: false,
      apiKey: "saved-secret-key",
      clearKey: false,
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: string | URL | Request, init?: RequestInit) => {
        expect(init?.headers).toMatchObject({
          authorization: "Bearer saved-secret-key",
        });
        return Response.json({
          data: [{ id: "llama-3.1-8b-instant", object: "model" }],
        });
      }),
    );

    await expect(settings.loadModels("groq")).resolves.toMatchObject({
      provider: "groq",
      models: [{ id: "llama-3.1-8b-instant" }],
    });
    store.close();
  });

  it.each<[AiProviderId, string]>([
    ["groq", "https://api.groq.com/openai/v1/chat/completions"],
    ["cerebras", "https://api.cerebras.ai/v1/chat/completions"],
    ["openrouter", "https://openrouter.ai/api/v1/chat/completions"],
    ["nvidia_nim", "https://integrate.api.nvidia.com/v1/chat/completions"],
  ])(
    "verifies the selected %s model with a chat request",
    async (provider, endpoint) => {
      const directory = mkdtempSync(
        resolve(tmpdir(), "structurefirst-settings-"),
      );
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
      vi.stubGlobal(
        "fetch",
        vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
          expect(String(input)).toBe(endpoint);
          expect(init?.method).toBe("POST");
          expect(init?.headers).toMatchObject({
            authorization: "Bearer provider-test-key",
          });
          expect(JSON.parse(String(init?.body))).toMatchObject({
            model: "vendor/chat-model",
            max_tokens: 32,
            messages: [
              { role: "system" },
              {
                role: "user",
                content: 'Return exactly {"status":"ok"} and nothing else.',
              },
            ],
            stream: false,
          });
          return Response.json({
            choices: [
              {
                message: {
                  role: "assistant",
                  content: '{"status":"ok"}',
                },
              },
            ],
          });
        }),
      );

      await expect(
        new SettingsService(store, config).test(provider, {
          model: "vendor/chat-model",
          apiKey: "provider-test-key",
        }),
      ).resolves.toMatchObject({
        ok: true,
        provider,
        model: "vendor/chat-model",
      });
      store.close();
    },
  );

  it("does not validate a model that ignores the required JSON format", async () => {
    const directory = mkdtempSync(
      resolve(tmpdir(), "structurefirst-settings-"),
    );
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
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        Response.json({
          choices: [
            {
              message: {
                role: "assistant",
                content: "I can do that for you.",
              },
            },
          ],
        }),
      ),
    );

    const result = await new SettingsService(store, config).test("groq", {
      model: "llama-3.1-8b-instant",
      apiKey: "provider-test-key",
    });

    expect(result.ok).toBe(false);
    expect(result.message).toContain("JSON instruction test");
    store.close();
  });

  it("explains NVIDIA NIM access failures instead of treating a public catalog as validation", async () => {
    const directory = mkdtempSync(
      resolve(tmpdir(), "structurefirst-settings-"),
    );
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
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        Response.json({ error: { message: "Forbidden" } }, { status: 403 }),
      ),
    );

    const result = await new SettingsService(store, config).test("nvidia_nim", {
      model: "meta/llama-model",
      apiKey: "provider-test-key",
    });

    expect(result.ok).toBe(false);
    expect(result.message).toContain("Public API Endpoints");
    store.close();
  });
});
