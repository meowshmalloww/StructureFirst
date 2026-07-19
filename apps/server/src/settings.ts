import { createCipheriv, createDecipheriv, randomBytes } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import {
  AiProviderIdSchema,
  AiProviderModelsResultSchema,
  AppSettingsSchema,
  DiscoverySettingsSchema,
  SaveAiProviderInputSchema,
  SaveDiscoverySettingsInputSchema,
  TestAiProviderInputSchema,
  type AiProviderId,
  type AiProviderCatalogKind,
  type AiProviderModel,
  type AiProviderModelsResult,
  type AiProviderSettings,
  type AiProviderTestResult,
  type AppSettings,
  type SaveAiProviderInput,
  type SaveDiscoverySettingsInput,
  type TestAiProviderInput,
} from "@structurefirst/contracts";
import type { AppConfig } from "./config.js";
import { nowIso } from "./lib/ids.js";
import { StructureStore } from "./store.js";

type StoredProvider = {
  baseUrl: string;
  model: string;
  enabled: boolean;
  vision: boolean;
  encryptedKey?: string;
  updatedAt: string;
};

type StoredDiscovery = {
  openverseEnabled: boolean;
  browserEnabled: boolean;
  encryptedBraveKey?: string;
  updatedAt: string;
};

type ProviderDefinition = {
  id: AiProviderId;
  label: string;
  baseUrl: string;
  envKey: (config: AppConfig) => string | undefined;
};

export type ProviderCredential = {
  id: AiProviderId;
  baseUrl: string;
  model: string;
  apiKey: string;
  vision: boolean;
};

const PROVIDERS: ProviderDefinition[] = [
  {
    id: "groq",
    label: "Groq",
    baseUrl: "https://api.groq.com/openai/v1",
    envKey: (config) => config.groqApiKey,
  },
  {
    id: "cerebras",
    label: "Cerebras",
    baseUrl: "https://api.cerebras.ai/v1",
    envKey: (config) => config.cerebrasApiKey,
  },
  {
    id: "openrouter",
    label: "OpenRouter",
    baseUrl: "https://openrouter.ai/api/v1",
    envKey: (config) => config.openRouterApiKey,
  },
  {
    id: "nvidia_nim",
    label: "NVIDIA NIM",
    baseUrl: "https://integrate.api.nvidia.com/v1",
    envKey: (config) => config.nvidiaApiKey,
  },
];

// StructureFirst intentionally exposes no-cost, general analysis choices only.
// Groq source: https://console.groq.com/docs/rate-limits (Free Plan Limits).
const GROQ_FREE_CHAT_MODELS = new Set([
  "allam-2-7b",
  "groq/compound",
  "groq/compound-mini",
  "llama-3.1-8b-instant",
  "llama-3.3-70b-versatile",
  "meta-llama/llama-4-scout-17b-16e-instruct",
  "openai/gpt-oss-120b",
  "openai/gpt-oss-20b",
  "qwen/qwen3-32b",
  "qwen/qwen3.6-27b",
]);

// NVIDIA's /v1/models response mixes unrelated API families. This set mirrors
// the models documented for chat completions and multimodal responses instead.
// Sources: https://docs.api.nvidia.com/nim/reference/llm-apis and
// https://docs.api.nvidia.com/nim/reference/multimodal-apis.
const NVIDIA_PROTOTYPE_CHAT_MODELS = new Set([
  "abacusai/dracarys-llama-3.1-70b-instruct",
  "bytedance/seed-oss-36b-instruct",
  "deepseek-ai/deepseek-v4-flash",
  "deepseek-ai/deepseek-v4-pro",
  "google/codegemma-7b",
  "google/gemma-2-2b-it",
  "google/gemma-7b",
  "meta/llama2-70b",
  "meta/llama-3.1-8b-instruct",
  "meta/llama-3.1-70b-instruct",
  "meta/llama-3.2-1b-instruct",
  "meta/llama-3.2-3b-instruct",
  "meta/llama-3.3-70b-instruct",
  "microsoft/phi-4-mini-instruct",
  "microsoft/phi-4-mini-flash-reasoning",
  "minimaxai/minimax-m2.5",
  "minimaxai/minimax-m2.7",
  "mistralai/mistral-nemotron",
  "mistralai/mixtral-8x7b-instruct",
  "mistralai/mixtral-8x22b-instruct",
  "moonshotai/kimi-k2-instruct",
  "moonshotai/kimi-k2-thinking",
  "nvidia/llama-3.1-nemotron-nano-8b-v1",
  "nvidia/llama-3.1-nemotron-ultra-253b-v1",
  "nvidia/llama-3.3-nemotron-super-49b-v1",
  "nvidia/llama-3.3-nemotron-super-49b-v1.5",
  "nvidia/nemotron-3-nano-30b-a3b",
  "nvidia/nemotron-3-super-120b-a12b",
  "nvidia/nemotron-3-ultra-550b-a55b",
  "nvidia/nemotron-mini-4b-instruct",
  "nvidia/nvidia-nemotron-nano-9b-v2",
  "openai/gpt-oss-20b",
  "openai/gpt-oss-120b",
  "poolside/laguna-xs-2.1",
  "qwen/qwen2.5-coder-32b-instruct",
  "qwen/qwen3.5-122b-a10b",
  "qwen/qwen3-coder-480b-a35b-instruct",
  "qwen/qwen3-next-80b-a3b-instruct",
  "qwen/qwen3-next-80b-a3b-thinking",
  "qwen/qwq-32b",
  "sarvamai/sarvam-m",
  "stepfun-ai/step-3-5-flash",
  "stockmark/stockmark-2-100b-instruct",
  "upstage/solar-10.7b-instruct",
  "z-ai/glm4.7",
  "z-ai/glm5.1",
  "z-ai/glm-5.2",
  "meta/llama-3.2-11b-vision-instruct",
  "meta/llama-3.2-90b-vision-instruct",
  "meta/llama-4-maverick-17b-128e-instruct",
  "moonshotai/kimi-k2.5",
  "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
  "qwen/qwen3.5-397b-a17b",
]);

const NVIDIA_VISION_MODELS = new Set([
  "meta/llama-3.2-11b-vision-instruct",
  "meta/llama-3.2-90b-vision-instruct",
  "meta/llama-4-maverick-17b-128e-instruct",
  "moonshotai/kimi-k2.5",
  "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
  "qwen/qwen3.5-397b-a17b",
]);

const GROQ_VISION_MODELS = new Set([
  "meta-llama/llama-4-scout-17b-16e-instruct",
]);

const SPECIAL_PURPOSE_MODEL =
  /(?:^|[\/_\-.:])(?:audio|classif(?:y|ier|ication)|content-safety|detect(?:or|ion)?|embed(?:ding)?|guard|lyria|moderation|rerank|retriev(?:al|er)?|reward|safeguard|speech|transcrib(?:e|er|ing|ption)?|translate|tts|whisper)(?:$|[\/_\-.:])/i;

export class SettingsService {
  private readonly vault: SecretVault;

  constructor(
    private readonly store: StructureStore,
    private readonly config: AppConfig,
  ) {
    this.vault = new SecretVault(resolve(config.dataRoot, ".secrets/key.bin"));
  }

  list(): AppSettings {
    const providers = PROVIDERS.map((definition) =>
      this.publicProvider(definition),
    );
    const stored = this.store.getSetting<StoredDiscovery>("discovery");
    const envKey = this.config.braveApiKey;
    const storedKey = stored?.encryptedBraveKey
      ? this.vault.decrypt(stored.encryptedBraveKey)
      : undefined;
    const discovery = DiscoverySettingsSchema.parse({
      openverseEnabled: stored?.openverseEnabled ?? true,
      browserEnabled: stored?.browserEnabled ?? true,
      braveConfigured: Boolean(storedKey ?? envKey),
      ...((storedKey ?? envKey)
        ? { braveKeyHint: keyHint(storedKey ?? envKey ?? "") }
        : {}),
      ...(stored?.updatedAt ? { updatedAt: stored.updatedAt } : {}),
    });
    return AppSettingsSchema.parse({ providers, discovery });
  }

  saveProvider(
    providerId: AiProviderId,
    rawInput: SaveAiProviderInput,
  ): AiProviderSettings {
    const input = SaveAiProviderInputSchema.parse(rawInput);
    const definition = this.definition(providerId);
    const previous = this.store.getSetting<StoredProvider>(
      `provider:${providerId}`,
    );
    const baseUrl = validateProviderBaseUrl(providerId, input.baseUrl);
    const apiKey = input.apiKey?.trim();
    const encryptedKey = input.clearKey
      ? undefined
      : apiKey
        ? this.vault.encrypt(apiKey)
        : previous?.encryptedKey;
    const stored: StoredProvider = {
      baseUrl,
      model: input.model,
      enabled: input.enabled,
      vision: input.vision,
      ...(encryptedKey ? { encryptedKey } : {}),
      updatedAt: nowIso(),
    };
    this.store.putSetting(`provider:${providerId}`, stored);
    return this.publicProvider(definition);
  }

  saveDiscovery(
    rawInput: SaveDiscoverySettingsInput,
  ): AppSettings["discovery"] {
    const input = SaveDiscoverySettingsInputSchema.parse(rawInput);
    const previous = this.store.getSetting<StoredDiscovery>("discovery");
    const braveKey = input.braveApiKey?.trim();
    const encryptedBraveKey = input.clearBraveKey
      ? undefined
      : braveKey
        ? this.vault.encrypt(braveKey)
        : previous?.encryptedBraveKey;
    const stored: StoredDiscovery = {
      openverseEnabled: input.openverseEnabled,
      browserEnabled: input.browserEnabled,
      ...(encryptedBraveKey ? { encryptedBraveKey } : {}),
      updatedAt: nowIso(),
    };
    this.store.putSetting("discovery", stored);
    return this.list().discovery;
  }

  credential(
    requested?: AiProviderId,
    requireEnabled = true,
  ): ProviderCredential | undefined {
    const definitions = requested ? [this.definition(requested)] : PROVIDERS;
    for (const definition of definitions) {
      const stored = this.store.getSetting<StoredProvider>(
        `provider:${definition.id}`,
      );
      const apiKey = stored?.encryptedKey
        ? this.vault.decrypt(stored.encryptedKey)
        : definition.envKey(this.config);
      const enabled = stored?.enabled ?? Boolean(apiKey);
      if (!apiKey || !stored?.model || (requireEnabled && !enabled)) continue;
      return {
        id: definition.id,
        baseUrl: stored.baseUrl,
        model: stored.model,
        apiKey,
        vision: stored.vision,
      };
    }
    return undefined;
  }

  async test(
    providerId: AiProviderId,
    rawInput: TestAiProviderInput,
  ): Promise<AiProviderTestResult> {
    const input = TestAiProviderInputSchema.parse(rawInput);
    const started = Date.now();
    try {
      const definition = this.definition(providerId);
      const stored = this.store.getSetting<StoredProvider>(
        `provider:${providerId}`,
      );
      const storedKey = stored?.encryptedKey
        ? this.vault.decrypt(stored.encryptedKey)
        : undefined;
      const apiKey =
        input.apiKey?.trim() || storedKey || definition.envKey(this.config);
      if (!apiKey) {
        throw new Error("Enter or save an API key before testing a model.");
      }

      const baseUrl = stored?.baseUrl ?? definition.baseUrl;
      let response: Response;
      try {
        response = await fetch(`${baseUrl}/chat/completions`, {
          method: "POST",
          headers: this.providerHeaders(providerId, apiKey),
          body: JSON.stringify({
            model: input.model,
            messages: [
              {
                role: "system",
                content:
                  "You are a JSON API. Output JSON only. Never explain or use Markdown.",
              },
              {
                role: "user",
                content: 'Return exactly {"status":"ok"} and nothing else.',
              },
            ],
            temperature: 0,
            max_tokens: 32,
            stream: false,
          }),
          signal: AbortSignal.timeout(30_000),
        });
      } catch (error) {
        throw new Error(
          error instanceof Error && error.name === "TimeoutError"
            ? `${definition.label} did not complete the test within 30 seconds.`
            : `${definition.label} could not be reached.`,
        );
      }

      if (!response.ok) {
        const detail = await providerErrorDetail(response);
        throw new Error(
          providerTestError(
            providerId,
            definition.label,
            input.model,
            response.status,
            detail,
          ),
        );
      }

      let payload: unknown;
      try {
        payload = await response.json();
      } catch {
        throw new Error(
          `${definition.label} returned an invalid chat response.`,
        );
      }
      const choices =
        payload && typeof payload === "object"
          ? (payload as Record<string, unknown>).choices
          : undefined;
      if (!Array.isArray(choices) || choices.length === 0) {
        throw new Error(
          `${definition.label} returned no chat completion for ${input.model}.`,
        );
      }
      const content = chatChoiceContent(choices[0]);
      if (!content || !isCompatibleChatResponse(content)) {
        throw new Error(
          `${definition.label} answered, but ${input.model} did not pass the JSON instruction test required by StructureFirst.`,
        );
      }

      return {
        ok: true,
        provider: providerId,
        model: input.model,
        latencyMs: Date.now() - started,
        message: `${definition.label} accepted the key and ${input.model} passed the StructureFirst chat-format test.`,
      };
    } catch (error) {
      return {
        ok: false,
        provider: providerId,
        model: input.model,
        latencyMs: Date.now() - started,
        message: error instanceof Error ? error.message : "Connection failed.",
      };
    }
  }

  async loadModels(
    providerId: AiProviderId,
    suppliedApiKey?: string,
  ): Promise<AiProviderModelsResult> {
    const definition = this.definition(providerId);
    const stored = this.store.getSetting<StoredProvider>(
      `provider:${providerId}`,
    );
    const storedKey = stored?.encryptedKey
      ? this.vault.decrypt(stored.encryptedKey)
      : undefined;
    const apiKey =
      suppliedApiKey?.trim() || storedKey || definition.envKey(this.config);
    if (!apiKey) {
      throw new Error("Enter or save an API key before loading models.");
    }

    const baseUrl = stored?.baseUrl ?? definition.baseUrl;
    const endpoint =
      providerId === "openrouter"
        ? `${baseUrl}/models/user?output_modalities=text`
        : `${baseUrl}/models`;
    const started = Date.now();
    let response: Response;
    try {
      response = await fetch(endpoint, {
        headers: this.providerHeaders(providerId, apiKey),
        signal: AbortSignal.timeout(15_000),
      });
    } catch (error) {
      throw new Error(
        error instanceof Error && error.name === "TimeoutError"
          ? `${definition.label} did not respond within 15 seconds.`
          : `${definition.label} could not be reached.`,
      );
    }

    if (!response.ok) {
      if ([401, 403].includes(response.status)) {
        throw new Error(`${definition.label} did not accept that API key.`);
      }
      if (response.status === 429) {
        throw new Error(
          `${definition.label} rate limit reached. Try again shortly.`,
        );
      }
      throw new Error(
        `${definition.label} model list failed (${response.status}).`,
      );
    }

    let payload: { data?: unknown };
    try {
      payload = (await response.json()) as { data?: unknown };
    } catch {
      throw new Error(`${definition.label} returned an invalid model list.`);
    }
    const models = normalizeProviderModels(providerId, payload.data);
    if (models.length === 0) {
      throw new Error(
        `${definition.label} returned no no-cost chat models supported by StructureFirst.`,
      );
    }
    const catalog = providerCatalog(providerId, models.length);
    return AiProviderModelsResultSchema.parse({
      provider: providerId,
      models,
      catalogKind: catalog.kind,
      notice: catalog.notice,
      latencyMs: Date.now() - started,
    });
  }

  discoveryOptions(): {
    openverseEnabled: boolean;
    browserEnabled: boolean;
    braveApiKey?: string;
  } {
    const stored = this.store.getSetting<StoredDiscovery>("discovery");
    const encrypted = stored?.encryptedBraveKey;
    const braveApiKey = encrypted
      ? this.vault.decrypt(encrypted)
      : this.config.braveApiKey;
    return {
      openverseEnabled: stored?.openverseEnabled ?? true,
      browserEnabled: stored?.browserEnabled ?? true,
      ...(braveApiKey ? { braveApiKey } : {}),
    };
  }

  headers(credential: ProviderCredential): Record<string, string> {
    return this.providerHeaders(credential.id, credential.apiKey);
  }

  private providerHeaders(
    providerId: AiProviderId,
    apiKey: string,
  ): Record<string, string> {
    return {
      authorization: `Bearer ${apiKey}`,
      accept: "application/json",
      "content-type": "application/json",
      ...(providerId === "openrouter"
        ? {
            "HTTP-Referer": "http://localhost/structurefirst",
            "X-OpenRouter-Title": "StructureFirst",
          }
        : {}),
    };
  }

  private publicProvider(definition: ProviderDefinition): AiProviderSettings {
    const stored = this.store.getSetting<StoredProvider>(
      `provider:${definition.id}`,
    );
    const savedKey = stored?.encryptedKey
      ? this.vault.decrypt(stored.encryptedKey)
      : undefined;
    const apiKey = savedKey ?? definition.envKey(this.config);
    return {
      id: definition.id,
      label: definition.label,
      baseUrl: stored?.baseUrl ?? definition.baseUrl,
      model: stored?.model ?? "",
      enabled: stored?.enabled ?? Boolean(apiKey),
      vision: stored?.vision ?? false,
      configured: Boolean(apiKey),
      ...(apiKey ? { keyHint: keyHint(apiKey) } : {}),
      ...(stored?.updatedAt ? { updatedAt: stored.updatedAt } : {}),
    };
  }

  private definition(providerId: AiProviderId): ProviderDefinition {
    const parsed = AiProviderIdSchema.parse(providerId);
    const definition = PROVIDERS.find((item) => item.id === parsed);
    if (!definition) throw new Error("Unknown AI provider.");
    return definition;
  }
}

class SecretVault {
  private key?: Buffer;

  constructor(private readonly keyPath: string) {}

  encrypt(plainText: string): string {
    const key = this.loadKey(true);
    const iv = randomBytes(12);
    const cipher = createCipheriv("aes-256-gcm", key, iv);
    const encrypted = Buffer.concat([
      cipher.update(plainText, "utf8"),
      cipher.final(),
    ]);
    return [
      "v1",
      iv.toString("base64url"),
      cipher.getAuthTag().toString("base64url"),
      encrypted.toString("base64url"),
    ].join(":");
  }

  decrypt(payload: string): string {
    const [version, ivText, tagText, cipherText] = payload.split(":");
    if (version !== "v1" || !ivText || !tagText || !cipherText)
      throw new Error("Stored credential is not valid.");
    const decipher = createDecipheriv(
      "aes-256-gcm",
      this.loadKey(false),
      Buffer.from(ivText, "base64url"),
    );
    decipher.setAuthTag(Buffer.from(tagText, "base64url"));
    return Buffer.concat([
      decipher.update(Buffer.from(cipherText, "base64url")),
      decipher.final(),
    ]).toString("utf8");
  }

  private loadKey(create: boolean): Buffer {
    if (this.key) return this.key;
    try {
      const existing = readFileSync(this.keyPath);
      if (existing.byteLength !== 32)
        throw new Error("Credential encryption key has the wrong length.");
      this.key = existing;
      return existing;
    } catch (error) {
      if (!create) throw error;
      mkdirSync(dirname(this.keyPath), { recursive: true });
      const generated = randomBytes(32);
      try {
        writeFileSync(this.keyPath, generated, { flag: "wx", mode: 0o600 });
        this.key = generated;
        return generated;
      } catch {
        const existing = readFileSync(this.keyPath);
        if (existing.byteLength !== 32)
          throw new Error("Credential encryption key has the wrong length.");
        this.key = existing;
        return existing;
      }
    }
  }
}

function normalizeProviderModels(
  providerId: AiProviderId,
  value: unknown,
): AiProviderModel[] {
  if (!Array.isArray(value)) return [];
  const byId = new Map<string, AiProviderModel>();
  for (const entry of value) {
    if (!entry || typeof entry !== "object") continue;
    const item = entry as Record<string, unknown>;
    const id = textValue(item.id)?.trim();
    if (!id || id.length > 300) continue;
    if (!providerModelAllowed(providerId, id, item)) continue;
    const name = textValue(item.name)?.trim() || id;
    const ownedBy = textValue(item.owned_by)?.trim();
    const contextWindow = positiveInteger(
      item.context_length ??
        item.context_window ??
        nestedValue(item, "limits", "max_context_length"),
    );
    const modalities = nestedValue(item, "architecture", "input_modalities");
    const modality = textValue(nestedValue(item, "architecture", "modality"));
    const vision =
      NVIDIA_VISION_MODELS.has(id) ||
      GROQ_VISION_MODELS.has(id) ||
      nestedValue(item, "capabilities", "vision") === true ||
      (Array.isArray(modalities) && modalities.includes("image")) ||
      Boolean(modality?.split("->", 1)[0]?.includes("image"));
    const free = freePricing(item.pricing);
    byId.set(id, {
      id,
      name: name.slice(0, 300),
      ...(ownedBy ? { ownedBy: ownedBy.slice(0, 160) } : {}),
      ...(contextWindow ? { contextWindow } : {}),
      ...(free !== undefined ? { free } : {}),
      vision,
    });
  }
  return [...byId.values()].sort((left, right) =>
    left.id.localeCompare(right.id),
  );
}

function providerModelAllowed(
  providerId: AiProviderId,
  id: string,
  item: Record<string, unknown>,
): boolean {
  if (item.active === false || SPECIAL_PURPOSE_MODEL.test(id)) return false;
  if (providerId === "groq") return GROQ_FREE_CHAT_MODELS.has(id);
  if (providerId === "nvidia_nim") {
    return NVIDIA_PROTOTYPE_CHAT_MODELS.has(id);
  }
  if (providerId === "openrouter") {
    const outputModalities = nestedValue(
      item,
      "architecture",
      "output_modalities",
    );
    const textOnly =
      !Array.isArray(outputModalities) ||
      (outputModalities.length === 1 && outputModalities[0] === "text");
    return textOnly && freePricing(item.pricing) === true;
  }
  // Cerebras' authenticated /v1/models endpoint is its public chat catalog.
  return true;
}

function providerCatalog(
  providerId: AiProviderId,
  count: number,
): { kind: AiProviderCatalogKind; notice: string } {
  if (providerId === "openrouter") {
    return {
      kind: "free_models",
      notice: `${count} zero-price text model${count === 1 ? "" : "s"} matched this key's OpenRouter preferences. Free-model request limits still apply.`,
    };
  }
  if (providerId === "nvidia_nim") {
    return {
      kind: "prototype",
      notice: `${count} documented NVIDIA chat or image-understanding prototype model${count === 1 ? "" : "s"} matched this key's catalog. Verify the selected model before use.`,
    };
  }
  return {
    kind: "free_plan",
    notice:
      providerId === "groq"
        ? `${count} general chat model${count === 1 ? "" : "s"} from Groq's current Free Plan list matched this key.`
        : `${count} public Cerebras chat model${count === 1 ? "" : "s"} matched this key. Free-tier rate limits apply.`,
  };
}

function nestedValue(
  item: Record<string, unknown>,
  parent: string,
  child: string,
): unknown {
  const value = item[parent];
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)[child]
    : undefined;
}

function textValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function positiveInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value > 0
    ? value
    : undefined;
}

function freePricing(value: unknown): boolean | undefined {
  if (!value || typeof value !== "object") return undefined;
  const pricing = value as Record<string, unknown>;
  const prompt = numericPrice(pricing.prompt);
  const completion = numericPrice(pricing.completion);
  if (prompt === undefined || completion === undefined) return undefined;
  const optionalPrices = [
    pricing.request,
    pricing.image,
    pricing.web_search,
    pricing.internal_reasoning,
  ]
    .map(numericPrice)
    .filter((price): price is number => price !== undefined);
  return (
    prompt === 0 &&
    completion === 0 &&
    optionalPrices.every((price) => price === 0)
  );
}

function numericPrice(value: unknown): number | undefined {
  if (typeof value !== "string" && typeof value !== "number") return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function chatChoiceContent(choice: unknown): string | undefined {
  if (!choice || typeof choice !== "object") return undefined;
  const message = (choice as Record<string, unknown>).message;
  if (!message || typeof message !== "object") return undefined;
  return textValue((message as Record<string, unknown>).content)?.trim();
}

function isCompatibleChatResponse(content: string): boolean {
  const unfenced = content
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();
  const start = unfenced.indexOf("{");
  const end = unfenced.lastIndexOf("}");
  if (start < 0 || end <= start) return false;
  try {
    const parsed = JSON.parse(unfenced.slice(start, end + 1)) as unknown;
    return (
      Boolean(parsed) &&
      typeof parsed === "object" &&
      Object.keys(parsed as Record<string, unknown>).length === 1 &&
      (parsed as Record<string, unknown>).status === "ok"
    );
  } catch {
    return false;
  }
}

async function providerErrorDetail(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as Record<string, unknown>;
    const error = payload.error;
    const message =
      error && typeof error === "object"
        ? textValue((error as Record<string, unknown>).message)
        : (textValue(error) ??
          textValue(payload.message) ??
          textValue(payload.detail));
    return message?.replace(/\s+/g, " ").trim().slice(0, 180) ?? "";
  } catch {
    return "";
  }
}

function providerTestError(
  providerId: AiProviderId,
  label: string,
  model: string,
  status: number,
  detail: string,
): string {
  const suffix = detail ? ` ${detail}` : "";
  if (status === 401) return `${label} rejected this API key.${suffix}`;
  if (status === 403) {
    return providerId === "nvidia_nim"
      ? `NVIDIA NIM rejected the request. Confirm the key has Public API Endpoints access and that this model is available.${suffix}`
      : `${label} denied access to this model.${suffix}`;
  }
  if (status === 400 || status === 404) {
    return `${label} could not run ${model} as a chat model.${suffix}`;
  }
  if (status === 429) {
    return `${label} rate limit reached. Try again after the provider limit resets.${suffix}`;
  }
  if (status >= 500) {
    return `${label} is temporarily unavailable (${status}).${suffix}`;
  }
  return `${label} model test failed (${status}).${suffix}`;
}

function validateProviderBaseUrl(
  providerId: AiProviderId,
  rawUrl: string,
): string {
  const url = new URL(rawUrl);
  url.hash = "";
  url.search = "";
  const host = url.hostname.toLowerCase();
  const allowedHosts: Record<Exclude<AiProviderId, "nvidia_nim">, string> = {
    groq: "api.groq.com",
    cerebras: "api.cerebras.ai",
    openrouter: "openrouter.ai",
  };
  if (providerId === "nvidia_nim") {
    const local = ["127.0.0.1", "localhost", "::1"].includes(host);
    if (!(
      (url.protocol === "https:" && host === "integrate.api.nvidia.com") ||
      (local && ["http:", "https:"].includes(url.protocol))
    )) {
      throw new Error(
        "NVIDIA NIM must use NVIDIA's HTTPS endpoint or a localhost NIM server.",
      );
    }
  } else if (url.protocol !== "https:" || host !== allowedHosts[providerId]) {
    throw new Error("Use the official HTTPS endpoint for this provider.");
  }
  return url.toString().replace(/\/$/, "");
}

function keyHint(apiKey: string): string {
  return `••••${apiKey.slice(-4)}`;
}
