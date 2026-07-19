import { existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { config as loadEnvironment } from "dotenv";

const moduleDirectory = dirname(fileURLToPath(import.meta.url));
loadEnvironment({
  path: resolve(moduleDirectory, "../../../.env"),
  quiet: true,
});

export interface AppConfig {
  repoRoot: string;
  dataRoot: string;
  casesRoot: string;
  databasePath: string;
  webDist: string;
  host: string;
  port: number;
  accessKey?: string;
  cookieSecret: string;
  braveApiKey?: string;
  groqApiKey?: string;
  cerebrasApiKey?: string;
  openRouterApiKey?: string;
  nvidiaApiKey?: string;
  browserExecutablePath?: string;
  reconstructionUrl: string;
  lucidFrameRoot?: string;
  nominatimBaseUrl: string;
  overpassBaseUrl: string;
  userAgent: string;
}

function optional(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

export function loadConfig(overrides: Partial<AppConfig> = {}): AppConfig {
  const repoRoot = overrides.repoRoot ?? resolve(moduleDirectory, "../../..");
  const dataRoot =
    overrides.dataRoot ??
    process.env.STRUCTUREFIRST_DATA_DIR ??
    resolve(repoRoot, "data");
  const casesRoot = overrides.casesRoot ?? resolve(dataRoot, "cases");
  const accessKey =
    overrides.accessKey ?? optional(process.env.STRUCTUREFIRST_ACCESS_KEY);
  const siblingLucidFrame = resolve(repoRoot, "../LucidFrame");
  const detectedLucidFrame = existsSync(siblingLucidFrame)
    ? siblingLucidFrame
    : undefined;
  const host = overrides.host ?? process.env.STRUCTUREFIRST_HOST ?? "127.0.0.1";
  const braveApiKey =
    overrides.braveApiKey ?? optional(process.env.BRAVE_SEARCH_API_KEY);
  const groqApiKey = overrides.groqApiKey ?? optional(process.env.GROQ_API_KEY);
  const cerebrasApiKey =
    overrides.cerebrasApiKey ?? optional(process.env.CEREBRAS_API_KEY);
  const openRouterApiKey =
    overrides.openRouterApiKey ?? optional(process.env.OPENROUTER_API_KEY);
  const nvidiaApiKey =
    overrides.nvidiaApiKey ?? optional(process.env.NVIDIA_API_KEY);
  const browserExecutablePath =
    overrides.browserExecutablePath ??
    optional(process.env.STRUCTUREFIRST_BROWSER_EXECUTABLE);
  const lucidFrameRoot =
    overrides.lucidFrameRoot ??
    optional(process.env.LUCIDFRAME_ROOT) ??
    detectedLucidFrame;

  if (!["127.0.0.1", "localhost", "::1"].includes(host) && !accessKey) {
    throw new Error(
      "STRUCTUREFIRST_ACCESS_KEY is required when STRUCTUREFIRST_HOST is exposed beyond localhost.",
    );
  }

  mkdirSync(casesRoot, { recursive: true });

  return {
    repoRoot,
    dataRoot,
    casesRoot,
    databasePath:
      overrides.databasePath ??
      process.env.STRUCTUREFIRST_DATABASE ??
      resolve(dataRoot, "structurefirst.db"),
    webDist: overrides.webDist ?? resolve(repoRoot, "apps/web/dist"),
    host,
    port: overrides.port ?? Number(process.env.STRUCTUREFIRST_PORT ?? 8787),
    ...(accessKey ? { accessKey } : {}),
    cookieSecret:
      overrides.cookieSecret ??
      process.env.STRUCTUREFIRST_COOKIE_SECRET ??
      "local-only-structurefirst-cookie-secret-change-for-lan",
    ...(braveApiKey ? { braveApiKey } : {}),
    ...(groqApiKey ? { groqApiKey } : {}),
    ...(cerebrasApiKey ? { cerebrasApiKey } : {}),
    ...(openRouterApiKey ? { openRouterApiKey } : {}),
    ...(nvidiaApiKey ? { nvidiaApiKey } : {}),
    ...(browserExecutablePath ? { browserExecutablePath } : {}),
    reconstructionUrl:
      overrides.reconstructionUrl ??
      process.env.STRUCTUREFIRST_RECONSTRUCTION_URL ??
      "http://127.0.0.1:8010",
    ...(lucidFrameRoot ? { lucidFrameRoot } : {}),
    nominatimBaseUrl:
      overrides.nominatimBaseUrl ??
      process.env.STRUCTUREFIRST_NOMINATIM_URL ??
      "https://nominatim.openstreetmap.org",
    overpassBaseUrl:
      overrides.overpassBaseUrl ??
      process.env.STRUCTUREFIRST_OVERPASS_URL ??
      "https://overpass-api.de/api/interpreter",
    userAgent:
      overrides.userAgent ??
      "StructureFirst/0.1 (local emergency-structure research; https://github.com/meowshmalloww/StructureFirst)",
  };
}
