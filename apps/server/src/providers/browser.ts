import { Buffer } from "node:buffer";
import { existsSync } from "node:fs";
import type { AppConfig } from "../config.js";

export type BrowserResult = {
  title: string;
  url: string;
  kind: "image" | "web_page";
  thumbnailUrl?: string;
  notes: string;
  tags: string[];
  confidenceScore: number;
};

type BingImageMetadata = {
  t?: string;
  purl?: string;
  murl?: string;
  turl?: string;
};

export async function discoverWithBrowser(
  address: string,
  config: AppConfig,
  alternatives: string[] = [],
): Promise<BrowserResult[]> {
  const executablePath = findBrowser(config);
  if (!executablePath)
    throw new Error(
      "Chrome or Edge was not found. Set STRUCTUREFIRST_BROWSER_EXECUTABLE in .env.",
    );
  const { chromium } = await import("playwright-core");
  const browser = await chromium.launch({ executablePath, headless: true });
  try {
    const context = await browser.newContext({ javaScriptEnabled: true });
    const searchTerm = (alternatives.find(Boolean) ?? address).trim();
    const outcomes = await Promise.allSettled([
      searchListingLinks(context, address),
      searchBingImages(context, searchTerm, address),
      searchBingWeb(context, address, alternatives),
    ]);
    const results = outcomes.map((outcome) =>
      outcome.status === "fulfilled" ? outcome.value : [],
    );
    const listings = results[0] ?? [];
    const images = results[1] ?? [];
    const pages = results[2] ?? [];
    if (outcomes.every((outcome) => outcome.status === "rejected"))
      throw new Error("Browser search providers did not respond.");
    const seen = new Set<string>();
    return [...listings, ...images, ...pages].filter((item) => {
      if (seen.has(item.url)) return false;
      seen.add(item.url);
      return true;
    });
  } finally {
    await browser.close();
  }
}

async function searchListingLinks(
  context: import("playwright-core").BrowserContext,
  address: string,
): Promise<BrowserResult[]> {
  const page = await context.newPage();
  try {
    const search = new URL("https://www.bing.com/search");
    search.searchParams.set(
      "q",
      `"${address}" (site:zillow.com OR site:redfin.com)`,
    );
    await page.goto(search.toString(), {
      waitUntil: "domcontentloaded",
      timeout: 15_000,
    });
    const raw = await page.locator("li.b_algo h2 a").evaluateAll((anchors) =>
      anchors.slice(0, 12).map((anchor) => ({
        title: anchor.textContent?.trim() ?? "",
        url: (anchor as HTMLAnchorElement).href,
      })),
    );
    return raw
      .map((item) => ({ ...item, url: unwrapBing(item.url) }))
      .filter(
        (item) =>
          isListingUrl(item.url) &&
          Boolean(item.title) &&
          addressTextMatch(address, item.title, item.url),
      )
      .map((item) => ({
        title: item.title,
        url: item.url,
        kind: "web_page" as const,
        tags: ["listing-source", "listing-address-match"],
        confidenceScore: 0.52,
        notes:
          "A listing search result contains the submitted street number and address terms. The listing remains link-only; StructureFirst does not crawl or copy its media.",
      }));
  } finally {
    await page.close();
  }
}

async function searchBingImages(
  context: import("playwright-core").BrowserContext,
  query: string,
  address: string,
): Promise<BrowserResult[]> {
  const page = await context.newPage();
  try {
    const url = new URL("https://www.bing.com/images/search");
    url.searchParams.set("q", `"${query}" building exterior property`);
    await page.goto(url.toString(), {
      waitUntil: "domcontentloaded",
      timeout: 15_000,
    });
    const metadata = await page.locator("a.iusc").evaluateAll((anchors) =>
      anchors.slice(0, 16).flatMap((anchor) => {
        try {
          return [JSON.parse(anchor.getAttribute("m") ?? "{}")];
        } catch {
          return [];
        }
      }),
    );
    return (metadata as BingImageMetadata[]).flatMap((item) => {
      if (
        !item.purl ||
        !item.t ||
        !isExternalResult(item.purl) ||
        !addressTextMatch(address, item.t, item.purl, item.murl ?? "")
      )
        return [];
      return [
        {
          title: item.t.trim(),
          url: item.purl,
          kind: "image" as const,
          ...(item.turl && isRemoteHttpUrl(item.turl)
            ? { thumbnailUrl: item.turl }
            : {}),
          notes: `Bing Images located a possible property image${item.murl ? ` hosted by ${safeHostname(item.murl)}` : ""}. Search thumbnails and source pages remain link-only until reuse rights are established.`,
          tags: ["image-search-candidate", "address-text-match"],
          confidenceScore: 0.46,
        },
      ];
    });
  } finally {
    await page.close();
  }
}

async function searchBingWeb(
  context: import("playwright-core").BrowserContext,
  address: string,
  alternatives: string[],
): Promise<BrowserResult[]> {
  const page = await context.newPage();
  try {
    const url = new URL("https://www.bing.com/search");
    const extra = alternatives.filter(Boolean).slice(0, 1).join(" ");
    url.searchParams.set(
      "q",
      `"${address}" ${extra} building exterior floor plan property record`,
    );
    await page.goto(url.toString(), {
      waitUntil: "domcontentloaded",
      timeout: 15_000,
    });
    const raw = await page.locator("li.b_algo h2 a").evaluateAll((anchors) =>
      anchors.slice(0, 12).map((anchor) => ({
        title: anchor.textContent?.trim() ?? "",
        url: (anchor as HTMLAnchorElement).href,
      })),
    );
    return raw
      .map((item) => ({ ...item, url: unwrapBing(item.url) }))
      .filter(
        (item): item is { title: string; url: string } =>
          Boolean(item.title) &&
          isExternalResult(item.url) &&
          addressTextMatch(address, item.title, item.url),
      )
      .map((item) => ({
        ...item,
        kind: "web_page" as const,
        notes:
          "A browser search found this source page. Its contents and property relevance remain unverified.",
        tags: ["web-search-candidate", "address-text-match"],
        confidenceScore: 0.42,
      }));
  } finally {
    await page.close();
  }
}

function isListingUrl(rawUrl: string): boolean {
  try {
    const hostname = new URL(rawUrl).hostname
      .replace(/^www\./, "")
      .toLowerCase();
    return ["zillow.com", "redfin.com"].some(
      (candidate) =>
        hostname === candidate || hostname.endsWith(`.${candidate}`),
    );
  } catch {
    return false;
  }
}

export function addressTextMatch(
  address: string,
  ...values: string[]
): boolean {
  const expected = normalizedAddressTerms(address);
  const candidate = normalizedAddressTerms(values.join(" "));
  if (!expected.houseNumber || candidate.houseNumber !== expected.houseNumber)
    return false;
  const sharedStreetTerms = expected.streetTerms.filter((term) =>
    candidate.streetTerms.includes(term),
  );
  return sharedStreetTerms.length >= Math.min(2, expected.streetTerms.length);
}

function normalizedAddressTerms(value: string): {
  houseNumber?: string;
  streetTerms: string[];
} {
  const terms = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  const houseNumber = terms.find((term) => /^\d+[a-z]?$/.test(term));
  const ignored = new Set([
    "street",
    "st",
    "avenue",
    "ave",
    "road",
    "rd",
    "drive",
    "dr",
    "lane",
    "ln",
    "boulevard",
    "blvd",
    "court",
    "ct",
    "way",
    "unit",
    "home",
    "homes",
    "for",
    "sale",
    "zillow",
    "redfin",
  ]);
  return {
    ...(houseNumber ? { houseNumber } : {}),
    streetTerms: terms.filter(
      (term) =>
        term !== houseNumber && !ignored.has(term) && !/^\d{5}$/.test(term),
    ),
  };
}

function findBrowser(config: AppConfig): string | undefined {
  const candidates = [
    config.browserExecutablePath,
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

export function unwrapBing(rawUrl: string): string {
  try {
    const url = new URL(rawUrl);
    if (!url.hostname.endsWith("bing.com")) return rawUrl;
    const target = url.searchParams.get("u");
    if (!target) return rawUrl;
    const encoded = target.startsWith("a1") ? target.slice(2) : target;
    const decoded = Buffer.from(encoded, "base64url").toString("utf8");
    return isRemoteHttpUrl(decoded) ? decoded : rawUrl;
  } catch {
    return rawUrl;
  }
}

function isExternalResult(rawUrl: string): boolean {
  try {
    const url = new URL(rawUrl);
    return (
      ["http:", "https:"].includes(url.protocol) &&
      !url.hostname.endsWith("bing.com") &&
      !url.hostname.endsWith("microsoft.com")
    );
  } catch {
    return false;
  }
}

function isRemoteHttpUrl(rawUrl: string): boolean {
  try {
    return ["http:", "https:"].includes(new URL(rawUrl).protocol);
  } catch {
    return false;
  }
}

function safeHostname(rawUrl: string): string {
  try {
    return new URL(rawUrl).hostname.replace(/^www\./, "");
  } catch {
    return "the indexed publisher";
  }
}
