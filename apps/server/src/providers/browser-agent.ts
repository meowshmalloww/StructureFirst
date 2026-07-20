import { existsSync } from "node:fs";
import type { AppConfig } from "../config.js";
import type { ProviderCredential, SettingsService } from "../settings.js";

// Local, LLM-driven browser agent. The agent controls a real headed
// Chromium/Edge with mouse and keyboard events. On each turn it sends a
// screenshot plus a compact list of interactive elements to the configured
// vision-capable LLM, receives a single JSON action, executes it, and repeats
// until the model emits `done` or the step budget is exhausted.
//
// The agent NEVER downloads image bytes. It only records candidate image URLs
// and their source pages. The discovery coordinator decides whether to fetch
// bytes and applies the restricted-local rights policy.

export type BrowserAgentCandidate = {
  imageUrl: string;
  sourceUrl: string;
  title: string;
  why: string;
};

export type BrowserAgentEvent =
  | { type: "start"; goal: string; maxSteps: number }
  | { type: "step"; step: number; action: AgentAction; url: string }
  | { type: "note"; step: number; text: string }
  | { type: "collect"; step: number; candidate: BrowserAgentCandidate }
  | { type: "warning"; step: number; message: string }
  | {
      type: "finish";
      steps: number;
      reason: "done" | "budget" | "timeout" | "error";
      candidateCount: number;
    };

export type BrowserAgentResult = {
  candidates: BrowserAgentCandidate[];
  steps: number;
  reason: "done" | "budget" | "timeout" | "error";
  transcript: string[];
};

type AgentAction =
  | { name: "goto"; url: string; reasoning?: string }
  | { name: "type"; ref: string; text: string; submit?: boolean; reasoning?: string }
  | { name: "press"; key: string; reasoning?: string }
  | { name: "click"; ref: string; reasoning?: string }
  | { name: "scroll"; direction: "up" | "down"; amount?: number; reasoning?: string }
  | { name: "wait"; seconds?: number; reasoning?: string }
  | { name: "back"; reasoning?: string }
  | {
      name: "collect_image";
      imageUrl: string;
      sourceUrl?: string;
      title?: string;
      why: string;
      reasoning?: string;
    }
  | { name: "note"; text: string; reasoning?: string }
  | { name: "done"; reasoning?: string };

type InteractiveElement = {
  ref: string;
  tag: string;
  role?: string;
  label: string;
  href?: string;
  imageSrc?: string;
};

export type BrowserAgentOptions = {
  address: string;
  alternatives?: string[];
  maxSteps: number;
  onEvent?: (event: BrowserAgentEvent) => void;
};

const HARD_BLOCK_HOSTS = new Set([
  "accounts.google.com",
  "login.microsoftonline.com",
  "login.live.com",
  "facebook.com",
  "www.facebook.com",
  "twitter.com",
  "x.com",
  "instagram.com",
  "www.instagram.com",
]);

export async function runBrowserAgent(
  options: BrowserAgentOptions,
  settings: SettingsService,
  config: AppConfig,
): Promise<BrowserAgentResult> {
  const credential = settings.credential();
  if (!credential)
    throw new Error(
      "The browser agent needs an enabled AI provider. Configure one in Settings.",
    );
  if (!credential.vision)
    throw new Error(
      "The browser agent needs a vision-capable model. Enable Vision on the active AI provider in Settings.",
    );
  const configuredPath = settings.discoveryOptions().browserExecutablePath;
  const executablePath = findBrowser(configuredPath ?? config.browserExecutablePath);
  if (!executablePath)
    throw new Error(
      "Chrome or Edge was not found. Enter the path under Settings > Local browser agent, or set STRUCTUREFIRST_BROWSER_EXECUTABLE in .env.",
    );

  const { chromium } = await import("playwright-core");
  const browser = await chromium.launch({
    executablePath,
    headless: false,
    slowMo: 120,
    args: ["--start-maximized"],
  });
  const transcript: string[] = [];
  const emit = (event: BrowserAgentEvent) => {
    options.onEvent?.(event);
  };
  emit({ type: "start", goal: options.address, maxSteps: options.maxSteps });

  const started = Date.now();
  const walltimeMs = 4 * 60_000;
  const seenImages = new Set<string>();
  const candidates: BrowserAgentCandidate[] = [];
  const history: string[] = [];

  try {
    const context = await browser.newContext({
      viewport: null,
      javaScriptEnabled: true,
      userAgent:
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    });
    const page = await context.newPage();
    await page.goto("about:blank");

    for (let step = 1; step <= options.maxSteps; step += 1) {
      if (Date.now() - started > walltimeMs) {
        transcript.push(`Step ${step}: wall-time budget exhausted.`);
        emit({
          type: "finish",
          steps: step - 1,
          reason: "timeout",
          candidateCount: candidates.length,
        });
        return { candidates, steps: step - 1, reason: "timeout", transcript };
      }

      const url = page.url();
      const host = safeHost(url);
      if (host && HARD_BLOCK_HOSTS.has(host)) {
        transcript.push(
          `Step ${step}: refused to interact with login domain ${host}; navigating back.`,
        );
        emit({
          type: "warning",
          step,
          message: `Skipped ${host} (login/social).`,
        });
        await page.goBack().catch(() => undefined);
        continue;
      }

      const elements = await extractInteractiveElements(page).catch(() => []);
      const screenshot = await page
        .screenshot({ type: "jpeg", quality: 62, fullPage: false })
        .catch(() => Buffer.alloc(0));

      const action = await askModelForAction(
        credential,
        settings,
        options,
        {
          step,
          url,
          history: history.slice(-6),
          elements,
          screenshot,
        },
      ).catch((error) => {
        transcript.push(
          `Step ${step}: model error - ${error instanceof Error ? error.message : "unknown"}`,
        );
        return null;
      });

      if (!action) {
        emit({
          type: "finish",
          steps: step,
          reason: "error",
          candidateCount: candidates.length,
        });
        return { candidates, steps: step, reason: "error", transcript };
      }

      const summary = describeAction(action);
      history.push(`step ${step}: ${summary}`);
      transcript.push(`Step ${step} @ ${url || "about:blank"}: ${summary}`);
      emit({ type: "step", step, action, url });

      try {
        switch (action.name) {
          case "goto": {
            if (!isAllowedNavigation(action.url)) {
              emit({
                type: "warning",
                step,
                message: `Refused to navigate to ${action.url}.`,
              });
              break;
            }
            await page.goto(action.url, {
              waitUntil: "domcontentloaded",
              timeout: 30_000,
            });
            break;
          }
          case "type": {
            const locator = locatorFromRef(page, action.ref, elements);
            if (!locator) break;
            await locator.click({ delay: 40 }).catch(() => undefined);
            await locator.fill("");
            await locator.type(action.text, { delay: 35 });
            if (action.submit) await page.keyboard.press("Enter");
            break;
          }
          case "press": {
            await page.keyboard.press(action.key);
            break;
          }
          case "click": {
            const locator = locatorFromRef(page, action.ref, elements);
            if (!locator) break;
            await locator
              .click({ timeout: 8_000, delay: 40 })
              .catch(() => undefined);
            break;
          }
          case "scroll": {
            const delta =
              (action.direction === "up" ? -1 : 1) * (action.amount ?? 800);
            await page.mouse.wheel(0, delta);
            break;
          }
          case "wait": {
            await page.waitForTimeout(
              Math.min(6_000, Math.max(250, (action.seconds ?? 1) * 1_000)),
            );
            break;
          }
          case "back": {
            await page.goBack({ waitUntil: "domcontentloaded" }).catch(() => undefined);
            break;
          }
          case "collect_image": {
            const candidate = normalizeCandidate(action, url, elements);
            if (!candidate) break;
            if (seenImages.has(candidate.imageUrl)) break;
            seenImages.add(candidate.imageUrl);
            candidates.push(candidate);
            emit({ type: "collect", step, candidate });
            break;
          }
          case "note": {
            emit({ type: "note", step, text: action.text });
            break;
          }
          case "done": {
            emit({
              type: "finish",
              steps: step,
              reason: "done",
              candidateCount: candidates.length,
            });
            return { candidates, steps: step, reason: "done", transcript };
          }
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : "unknown";
        transcript.push(`Step ${step}: action failed - ${message}`);
        emit({ type: "warning", step, message });
      }

      // brief settle so subsequent screenshots reflect changes
      await page.waitForTimeout(400);
    }

    emit({
      type: "finish",
      steps: options.maxSteps,
      reason: "budget",
      candidateCount: candidates.length,
    });
    return {
      candidates,
      steps: options.maxSteps,
      reason: "budget",
      transcript,
    };
  } finally {
    await browser.close().catch(() => undefined);
  }
}

function describeAction(action: AgentAction): string {
  switch (action.name) {
    case "goto":
      return `goto ${action.url}`;
    case "type":
      return `type "${action.text.slice(0, 40)}" into ${action.ref}${action.submit ? " + Enter" : ""}`;
    case "press":
      return `press ${action.key}`;
    case "click":
      return `click ${action.ref}`;
    case "scroll":
      return `scroll ${action.direction}`;
    case "wait":
      return `wait ${action.seconds ?? 1}s`;
    case "back":
      return "back";
    case "collect_image":
      return `collect_image ${action.imageUrl}`;
    case "note":
      return `note: ${action.text.slice(0, 80)}`;
    case "done":
      return "done";
  }
}

function isAllowedNavigation(rawUrl: string): boolean {
  try {
    const url = new URL(rawUrl);
    if (!["http:", "https:"].includes(url.protocol)) return false;
    if (HARD_BLOCK_HOSTS.has(url.hostname.replace(/^www\./, ""))) return false;
    return true;
  } catch {
    return false;
  }
}

function safeHost(rawUrl: string): string | undefined {
  try {
    return new URL(rawUrl).hostname.replace(/^www\./, "").toLowerCase();
  } catch {
    return undefined;
  }
}

function locatorFromRef(
  page: import("playwright-core").Page,
  ref: string,
  elements: InteractiveElement[],
): import("playwright-core").Locator | undefined {
  const match = elements.find((element) => element.ref === ref);
  if (!match) return undefined;
  return page.locator(`[data-sf-ref="${ref}"]`).first();
}

async function extractInteractiveElements(
  page: import("playwright-core").Page,
): Promise<InteractiveElement[]> {
  return await page.evaluate(() => {
    const out: {
      ref: string;
      tag: string;
      role?: string;
      label: string;
      href?: string;
      imageSrc?: string;
    }[] = [];
    const selector =
      "a[href], button, input:not([type=hidden]), textarea, select, [role='button'], [role='link'], [role='textbox'], [role='searchbox'], img";
    const nodes = Array.from(
      document.querySelectorAll<HTMLElement>(selector),
    ).slice(0, 400);
    let index = 0;
    for (const node of nodes) {
      const rect = node.getBoundingClientRect();
      if (rect.width < 4 || rect.height < 4) continue;
      const style = window.getComputedStyle(node);
      if (style.visibility === "hidden" || style.display === "none") continue;
      const inViewport =
        rect.top < window.innerHeight + 200 &&
        rect.bottom > -200 &&
        rect.left < window.innerWidth + 200 &&
        rect.right > -200;
      if (!inViewport) continue;
      index += 1;
      const ref = `sf${index}`;
      node.setAttribute("data-sf-ref", ref);
      const tag = node.tagName.toLowerCase();
      const role = node.getAttribute("role");
      let label =
        node.getAttribute("aria-label") ??
        (node as HTMLInputElement).placeholder ??
        node.getAttribute("title") ??
        node.textContent?.trim() ??
        "";
      label = label.replace(/\s+/g, " ").slice(0, 160);
      const anchor = tag === "a" ? (node as HTMLAnchorElement).href : undefined;
      const image =
        tag === "img" ? (node as HTMLImageElement).currentSrc || (node as HTMLImageElement).src : undefined;
      out.push({
        ref,
        tag,
        label,
        ...(role ? { role } : {}),
        ...(anchor ? { href: anchor } : {}),
        ...(image ? { imageSrc: image } : {}),
      });
      if (out.length >= 45) break;
    }
    return out;
  });
}

function normalizeCandidate(
  action: Extract<AgentAction, { name: "collect_image" }>,
  currentUrl: string,
  elements: InteractiveElement[],
): BrowserAgentCandidate | undefined {
  let imageUrl = action.imageUrl.trim();
  if (imageUrl.startsWith("sf")) {
    const match = elements.find((element) => element.ref === imageUrl);
    if (match?.imageSrc) imageUrl = match.imageSrc;
    else return undefined;
  }
  if (!/^https?:\/\//i.test(imageUrl)) return undefined;
  const sourceUrl = action.sourceUrl?.trim() || currentUrl || imageUrl;
  return {
    imageUrl,
    sourceUrl,
    title: action.title?.trim().slice(0, 240) || safeHost(sourceUrl) || "captured image",
    why: action.why.trim().slice(0, 400),
  };
}

async function askModelForAction(
  credential: ProviderCredential,
  settings: SettingsService,
  options: BrowserAgentOptions,
  turn: {
    step: number;
    url: string;
    history: string[];
    elements: InteractiveElement[];
    screenshot: Buffer;
  },
): Promise<AgentAction> {
  const alternatives = (options.alternatives ?? [])
    .filter(Boolean)
    .slice(0, 4)
    .join(" | ");
  const system = [
    "You are StructureFirst's local browser agent.",
    "Goal: locate exterior and interior photographs of a specific real property address on the public web.",
    "Prefer real-estate listing pages (zillow.com, redfin.com, realtor.com, trulia.com) and municipal or news photo pages that unambiguously depict the address.",
    "Search engines like bing.com/search or duckduckgo.com are fine for navigation; do NOT try to log in to Google.",
    "Never enter passwords or personal info; if a login/consent wall blocks you, use `back` and try another route.",
    "Return EXACTLY ONE JSON object per turn matching the schema. No prose, no code fences.",
    "Schema: {\"name\":\"goto\"|\"type\"|\"press\"|\"click\"|\"scroll\"|\"wait\"|\"back\"|\"collect_image\"|\"note\"|\"done\", plus fields per action}.",
    "For `type` and `click`, use one of the refs (sfN) from the elements list; do not invent refs.",
    "For `collect_image`, pass either a direct https image URL or a ref that points to an <img>. Include `sourceUrl` (the page URL that displays it) and a short `why`.",
    "Only collect images that appear to show the target property (exterior facade, rooms of that home, floor plan). Skip logos, agents' portraits, map tiles, and stock photos.",
    "Emit `done` once you have 6-15 good candidates or the current site has no more relevant photos.",
  ].join(" ");

  const elementsList = turn.elements
    .map(
      (element) =>
        `${element.ref} <${element.tag}${element.role ? " role=" + element.role : ""}> "${element.label}"${element.href ? " href=" + element.href : ""}${element.imageSrc ? " img=" + element.imageSrc : ""}`,
    )
    .join("\n");
  const userText = [
    `Target address: ${options.address}`,
    alternatives ? `Also known as: ${alternatives}` : undefined,
    `Step ${turn.step} / ${options.maxSteps}. Current URL: ${turn.url || "about:blank"}.`,
    turn.history.length ? `Recent actions:\n${turn.history.join("\n")}` : undefined,
    `Interactive elements (top ${turn.elements.length}):\n${elementsList || "(none visible)"}`,
    "Return only the JSON action.",
  ]
    .filter(Boolean)
    .join("\n\n");

  const content: unknown[] = [{ type: "text", text: userText }];
  if (turn.screenshot.byteLength > 0) {
    content.push({
      type: "image_url",
      image_url: {
        url: `data:image/jpeg;base64,${turn.screenshot.toString("base64")}`,
        detail: "low",
      },
    });
  }

  const response = await fetch(`${credential.baseUrl}/chat/completions`, {
    method: "POST",
    headers: settings.headers(credential),
    body: JSON.stringify({
      model: credential.model,
      temperature: 0.1,
      max_tokens: 400,
      messages: [
        { role: "system", content: system },
        { role: "user", content },
      ],
    }),
    signal: AbortSignal.timeout(45_000),
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(
      `AI provider returned ${response.status}: ${body.slice(0, 240)}`,
    );
  }
  const payload = (await response.json()) as {
    choices?: Array<{ message?: { content?: string | null } }>;
  };
  const text = payload.choices?.[0]?.message?.content ?? "";
  return parseAgentAction(text);
}

export function parseAgentAction(raw: string): AgentAction {
  const trimmed = raw.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  const candidate = fenced?.[1] ?? trimmed;
  const start = candidate.indexOf("{");
  const end = candidate.lastIndexOf("}");
  if (start < 0 || end <= start)
    throw new Error("Model did not return a JSON object.");
  const parsed = JSON.parse(candidate.slice(start, end + 1)) as {
    name?: string;
    [key: string]: unknown;
  };
  const name = parsed.name;
  switch (name) {
    case "goto":
      if (typeof parsed.url !== "string") throw new Error("goto requires url");
      return { name, url: parsed.url };
    case "type": {
      if (typeof parsed.ref !== "string" || typeof parsed.text !== "string")
        throw new Error("type requires ref + text");
      return {
        name,
        ref: parsed.ref,
        text: parsed.text,
        ...(parsed.submit === true ? { submit: true } : {}),
      };
    }
    case "press":
      if (typeof parsed.key !== "string") throw new Error("press requires key");
      return { name, key: parsed.key };
    case "click":
      if (typeof parsed.ref !== "string") throw new Error("click requires ref");
      return { name, ref: parsed.ref };
    case "scroll": {
      const direction = parsed.direction === "up" ? "up" : "down";
      return {
        name,
        direction,
        ...(typeof parsed.amount === "number"
          ? { amount: parsed.amount }
          : {}),
      };
    }
    case "wait":
      return {
        name,
        ...(typeof parsed.seconds === "number"
          ? { seconds: parsed.seconds }
          : {}),
      };
    case "back":
      return { name };
    case "collect_image": {
      const imageUrl =
        typeof parsed.imageUrl === "string"
          ? parsed.imageUrl
          : typeof parsed.url === "string"
            ? (parsed.url as string)
            : "";
      const why = typeof parsed.why === "string" ? parsed.why : "";
      if (!imageUrl || !why)
        throw new Error("collect_image requires imageUrl + why");
      return {
        name,
        imageUrl,
        why,
        ...(typeof parsed.sourceUrl === "string"
          ? { sourceUrl: parsed.sourceUrl }
          : {}),
        ...(typeof parsed.title === "string" ? { title: parsed.title } : {}),
      };
    }
    case "note":
      if (typeof parsed.text !== "string")
        throw new Error("note requires text");
      return { name, text: parsed.text };
    case "done":
      return { name };
    default:
      throw new Error(`Unknown action ${String(name)}`);
  }
}

function findBrowser(preferred: string | undefined): string | undefined {
  const candidates = [
    preferred,
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ];
  return candidates.find((candidate): candidate is string =>
    Boolean(candidate && existsSync(candidate)),
  );
}
