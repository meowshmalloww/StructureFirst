import { createHash } from "node:crypto";
import { lookup } from "node:dns/promises";
import { writeFileSync, mkdirSync } from "node:fs";
import { isIP } from "node:net";
import { resolve } from "node:path";
import {
  DiscoveryRunInputSchema,
  type Case,
  type DiscoveryRunInput,
  type DiscoveryRunResult,
  type EvidenceAsset,
  type EvidenceKind,
} from "@structurefirst/contracts";
import type { AppConfig } from "./config.js";
import { confidence } from "./lib/confidence.js";
import { createId, nowIso } from "./lib/ids.js";
import { classifySource } from "./lib/source-policy.js";
import { addressTextMatch, discoverWithBrowser } from "./providers/browser.js";
import { discoverBuildingEvidence } from "./providers/brave.js";
import { discoverKartaViewImages } from "./providers/kartaview.js";
import { discoverOpenverseImages } from "./providers/openverse.js";
import { discoverWikimediaImages } from "./providers/wikimedia.js";
import { SettingsService } from "./settings.js";
import { StructureStore } from "./store.js";

const MAX_REMOTE_IMAGE_BYTES = 20 * 1024 * 1024;
const MAX_AUTOMATIC_IMPORTS = 6;
const MIME_EXTENSIONS: Record<string, string> = {
  "image/jpeg": ".jpg",
  "image/png": ".png",
  "image/webp": ".webp",
};

export class EvidenceDiscoveryCoordinator {
  constructor(
    private readonly store: StructureStore,
    private readonly settings: SettingsService,
    private readonly config: AppConfig,
  ) {}

  async discoverCase(
    caseId: string,
    rawInput: DiscoveryRunInput,
  ): Promise<DiscoveryRunResult> {
    const input = DiscoveryRunInputSchema.parse(rawInput);
    const caseValue = this.store.getCase(caseId);
    if (!caseValue) throw new Error("Case not found.");
    const options = this.settings.discoveryOptions();
    const queryAddress = caseValue.displayAddress || caseValue.addressInput;
    const queryAlternatives = discoveryQueries(caseValue);
    const existingUrls = new Set(
      this.store
        .listEvidence(caseId)
        .flatMap((item) => (item.originUrl ? [item.originUrl] : [])),
    );
    const providers: string[] = [];
    const warnings: string[] = [];
    let added = 0;
    let imported = 0;

    if (caseValue.profile?.location) {
      try {
        const images = await discoverKartaViewImages(
          caseValue.profile.location.latitude,
          caseValue.profile.location.longitude,
        );
        providers.push("KartaView");
        for (const image of images) {
          if (existingUrls.has(image.landingUrl)) continue;
          const evidence: EvidenceAsset = {
            id: createId("evidence"),
            caseId,
            title: image.title,
            kind: "image",
            sourceProvider: "KartaView",
            originUrl: image.landingUrl,
            downloadUrl: image.downloadUrl,
            ...(image.thumbnailUrl ? { thumbnailUrl: image.thumbnailUrl } : {}),
            creator: "© Grab and KartaView Contributors",
            licenseUrl: "https://creativecommons.org/licenses/by-sa/4.0/",
            discoveredAt: nowIso(),
            ...(image.observedAt ? { observedAt: image.observedAt } : {}),
            rights: "open_license",
            cachePolicy: "local_allowed",
            redistributable: true,
            validation: "reachable",
            tags: [
              "automated-discovery",
              "open-license",
              "street-level",
              "exterior",
              "property-proximity-unverified",
              `kartaview-sequence:${image.sequenceId}`,
              `kartaview-index:${image.sequenceIndex}`,
              `overlap-set:kartaview-${image.sequenceId}`,
            ],
            notes: `KartaView CC BY-SA 4.0 street capture located ${image.distanceMeters} m from the resolved address point; camera-to-address heading difference ${image.headingDifference}°. Proximity and capture sequence are measured, but the target building must still be visually confirmed.`,
            confidence: confidence(
              image.headingDifference <= 65 ? 0.48 : 0.36,
              "estimated",
              "observed",
              "The capture location and heading are measured; whether the target facade is visible remains unverified.",
              1,
            ),
          };
          this.store.putEvidence(evidence);
          existingUrls.add(image.landingUrl);
          added += 1;
          if (imported < MAX_AUTOMATIC_IMPORTS) {
            try {
              await this.importOpenAsset(caseId, evidence.id, "automatic");
              imported += 1;
            } catch (error) {
              warnings.push(
                `Could not import ${image.title}: ${errorMessage(error)}`,
              );
            }
          }
        }
      } catch (error) {
        warnings.push(`KartaView: ${errorMessage(error)}`);
      }
    }

    if (input.includeOpenverse && options.openverseEnabled) {
      try {
        const images = await discoverWikimediaImages([
          ...queryAlternatives,
          queryAddress,
        ]);
        providers.push("Wikimedia Commons");
        for (const image of images) {
          if (existingUrls.has(image.landingUrl)) continue;
          const evidence: EvidenceAsset = {
            id: createId("evidence"),
            caseId,
            title: image.title,
            kind: "image",
            sourceProvider: image.source,
            originUrl: image.landingUrl,
            downloadUrl: image.downloadUrl,
            ...(image.thumbnailUrl ? { thumbnailUrl: image.thumbnailUrl } : {}),
            ...(image.creator ? { creator: image.creator } : {}),
            ...(image.licenseUrl ? { licenseUrl: image.licenseUrl } : {}),
            discoveredAt: nowIso(),
            rights: ["pdm", "cc0"].includes(image.licenseCode)
              ? "public_domain"
              : "open_license",
            cachePolicy: "local_allowed",
            redistributable: true,
            validation: "reachable",
            tags: [
              "automated-discovery",
              "open-license",
              "property-photo",
              `license:${image.licenseCode}`,
              ...(addressTextMatch(queryAddress, image.title, image.landingUrl)
                ? ["address-text-match"]
                : ["address-relevance-unverified"]),
            ],
            notes: `${image.license}${image.creator ? ` · ${image.creator}` : ""}. Wikimedia text match; attribution is retained and address relevance still requires visual confirmation.`,
            confidence: confidence(
              0.34,
              "estimated",
              "inferred",
              "Open-license text search match; visual address relevance is not yet confirmed.",
              1,
            ),
          };
          this.store.putEvidence(evidence);
          existingUrls.add(image.landingUrl);
          added += 1;
          if (imported < MAX_AUTOMATIC_IMPORTS) {
            try {
              await this.importOpenAsset(caseId, evidence.id, "automatic");
              imported += 1;
            } catch (error) {
              warnings.push(
                `Could not import ${image.title}: ${errorMessage(error)}`,
              );
            }
          }
        }
      } catch (error) {
        warnings.push(`Wikimedia Commons: ${errorMessage(error)}`);
      }
    }

    if (input.includeOpenverse && options.openverseEnabled) {
      try {
        const images = await discoverOpenverseImages(
          queryAddress,
          queryAlternatives,
        );
        providers.push("Openverse");
        for (const image of images) {
          if (existingUrls.has(image.landingUrl)) continue;
          const evidence: EvidenceAsset = {
            id: createId("evidence"),
            caseId,
            title: image.title,
            kind: "image",
            sourceProvider: `Openverse / ${image.source}`,
            originUrl: image.landingUrl,
            downloadUrl: image.downloadUrl,
            ...(image.thumbnailUrl ? { thumbnailUrl: image.thumbnailUrl } : {}),
            ...(image.creator ? { creator: image.creator } : {}),
            ...(image.licenseUrl ? { licenseUrl: image.licenseUrl } : {}),
            discoveredAt: nowIso(),
            rights: ["pdm", "cc0"].includes(image.licenseCode)
              ? "public_domain"
              : "open_license",
            cachePolicy: "local_allowed",
            redistributable: true,
            validation: "reachable",
            tags: [
              "automated-discovery",
              "open-license",
              "property-photo",
              `license:${image.licenseCode}`,
              ...(addressTextMatch(queryAddress, image.title, image.landingUrl)
                ? ["address-text-match"]
                : ["address-relevance-unverified"]),
            ],
            notes: `${image.license}${image.creator ? ` · ${image.creator}` : ""}. Openverse search match; the original item page remains attached for attribution and address verification.`,
            confidence: confidence(
              0.24,
              "estimated",
              "inferred",
              "Open-license search match only; visual contents and address relevance require review.",
              1,
            ),
          };
          this.store.putEvidence(evidence);
          existingUrls.add(image.landingUrl);
          added += 1;
          if (imported < MAX_AUTOMATIC_IMPORTS) {
            try {
              await this.importOpenAsset(caseId, evidence.id, "automatic");
              imported += 1;
            } catch (error) {
              warnings.push(
                `Could not import ${image.title}: ${errorMessage(error)}`,
              );
            }
          }
        }
      } catch (error) {
        warnings.push(`Openverse: ${errorMessage(error)}`);
      }
    }

    if (input.includeBrave && options.braveApiKey) {
      try {
        const links = await discoverBuildingEvidence(
          queryAddress,
          options.braveApiKey,
        );
        providers.push("Brave Search");
        for (const link of links) {
          if (existingUrls.has(link.url)) continue;
          const policy = classifySource(link.url);
          this.store.putEvidence(
            discoveredLink(caseId, link.title, link.url, link.kind, policy, {
              ...(link.thumbnailUrl && !policy.hardBlocked
                ? { thumbnailUrl: link.thumbnailUrl }
                : {}),
              notes: link.notes,
            }),
          );
          existingUrls.add(link.url);
          added += 1;
        }
      } catch (error) {
        warnings.push(`Brave Search: ${errorMessage(error)}`);
      }
    }

    if (input.includeBrowser && options.browserEnabled) {
      try {
        const links = await discoverWithBrowser(
          queryAddress,
          this.config,
          queryAlternatives,
        );
        providers.push("Browser search");
        for (const link of links) {
          if (existingUrls.has(link.url)) continue;
          const policy = classifySource(link.url);
          this.store.putEvidence(
            discoveredLink(caseId, link.title, link.url, link.kind, policy, {
              ...(link.thumbnailUrl && !policy.hardBlocked
                ? { thumbnailUrl: link.thumbnailUrl }
                : {}),
              notes: link.notes,
              tags: link.tags,
              confidenceScore: link.confidenceScore,
            }),
          );
          existingUrls.add(link.url);
          added += 1;
        }
      } catch (error) {
        warnings.push(`Browser discovery: ${errorMessage(error)}`);
      }
    }

    return { added, imported, providers, warnings };
  }

  async importOpenAsset(
    caseId: string,
    evidenceId: string,
    origin: "operator" | "automatic" = "operator",
  ): Promise<EvidenceAsset> {
    const evidence = this.store.getEvidence(evidenceId);
    if (!evidence || evidence.caseId !== caseId)
      throw new Error("Evidence was not found in this case.");
    if (
      evidence.cachePolicy !== "local_allowed" ||
      !["open_license", "public_domain"].includes(evidence.rights) ||
      !evidence.downloadUrl
    ) {
      throw new Error(
        "Only an explicitly open-license or public-domain discovery result can be imported.",
      );
    }
    const response = await safeRemoteFetch(evidence.downloadUrl);
    const mimeType = (response.headers.get("content-type") ?? "")
      .split(";", 1)[0]
      ?.trim()
      .toLowerCase();
    const extension = mimeType ? MIME_EXTENSIONS[mimeType] : undefined;
    if (!mimeType || !extension)
      throw new Error(
        "The remote asset is not a supported JPEG, PNG, or WebP image.",
      );
    const bytes = await limitedBytes(response, MAX_REMOTE_IMAGE_BYTES);
    if (!matchesMagic(bytes, mimeType))
      throw new Error(
        "The remote file signature does not match its image type.",
      );

    const uploadId = createId("import");
    const directory = resolve(this.config.casesRoot, caseId, "uploads");
    mkdirSync(directory, { recursive: true });
    const name = `${uploadId}${extension}`;
    const outputPath = resolve(directory, name);
    writeFileSync(outputPath, bytes, { flag: "wx" });
    return this.store.putEvidence({
      ...evidence,
      localUrl: `/assets/${caseId}/uploads/${name}`,
      validation:
        origin === "automatic" ? "automated_imported" : "operator_uploaded",
      mimeType,
      byteSize: bytes.byteLength,
      sha256: createHash("sha256").update(bytes).digest("hex"),
      tags: [
        ...new Set([
          ...evidence.tags,
          origin === "automatic" ? "automatic-import" : "operator-imported",
        ]),
      ],
      notes: `${evidence.notes} ${
        origin === "automatic"
          ? "Automatically downloaded because the indexed license permits modifications."
          : "Imported after an explicit operator action."
      } Attribution remains attached to this record.`,
      confidence: confidence(
        origin === "automatic" ? 0.42 : 0.5,
        "estimated",
        "observed",
        "File bytes and licensing metadata are locally retained; address relevance is still a search-derived estimate.",
        1,
      ),
    });
  }
}

function discoveredLink(
  caseId: string,
  title: string,
  url: string,
  kind: EvidenceKind,
  policy: ReturnType<typeof classifySource>,
  extra: {
    thumbnailUrl?: string;
    notes: string;
    tags?: string[];
    confidenceScore?: number;
  },
): EvidenceAsset {
  return {
    id: createId("evidence"),
    caseId,
    title,
    kind,
    sourceProvider: policy.provider,
    originUrl: url,
    ...(extra.thumbnailUrl ? { thumbnailUrl: extra.thumbnailUrl } : {}),
    discoveredAt: nowIso(),
    rights: policy.rights,
    cachePolicy: policy.cachePolicy,
    redistributable: policy.redistributable,
    validation: "reachable",
    tags: ["automated-discovery", "browser-reviewed", ...(extra.tags ?? [])],
    notes: `${extra.notes} ${policy.reason}`,
    confidence: confidence(
      extra.confidenceScore ?? 0.22,
      "estimated",
      "inferred",
      extra.tags?.includes("listing-address-match")
        ? "The search result text matches key submitted address terms; listing contents and media remain unaccessed."
        : "Search result metadata only; contents and address relevance have not been confirmed.",
      1,
    ),
  };
}

async function safeRemoteFetch(rawUrl: string): Promise<Response> {
  let current = new URL(rawUrl);
  for (let redirects = 0; redirects <= 3; redirects += 1) {
    if (!["http:", "https:"].includes(current.protocol))
      throw new Error("Remote asset URL must use HTTP or HTTPS.");
    await assertPublicHost(current.hostname);
    const response = await fetch(current, {
      redirect: "manual",
      headers: {
        accept: "image/jpeg,image/png,image/webp",
        "user-agent": "StructureFirst/0.2 open-license importer",
      },
      signal: AbortSignal.timeout(30_000),
    });
    if ([301, 302, 303, 307, 308].includes(response.status)) {
      const location = response.headers.get("location");
      if (!location || redirects === 3)
        throw new Error("Remote asset redirected too many times.");
      current = new URL(location, current);
      continue;
    }
    if (!response.ok)
      throw new Error(`Remote asset returned ${response.status}.`);
    return response;
  }
  throw new Error("Remote asset could not be fetched.");
}

async function assertPublicHost(hostname: string): Promise<void> {
  const addresses = await lookup(hostname, { all: true, verbatim: true });
  if (
    addresses.length === 0 ||
    addresses.some((item) => isPrivateIp(item.address))
  )
    throw new Error(
      "Remote asset host resolves to a private or reserved address.",
    );
}

function isPrivateIp(address: string): boolean {
  if (!isIP(address)) return true;
  const lower = address.toLowerCase();
  if (
    lower === "::1" ||
    lower === "::" ||
    lower.startsWith("fc") ||
    lower.startsWith("fd") ||
    lower.startsWith("fe8") ||
    lower.startsWith("fe9") ||
    lower.startsWith("fea") ||
    lower.startsWith("feb")
  )
    return true;
  if (lower.startsWith("::ffff:")) return isPrivateIp(lower.slice(7));
  if (isIP(address) === 6) return false;
  const parts = address.split(".").map(Number);
  const first = parts[0] ?? 0;
  const second = parts[1] ?? 0;
  return (
    first === 0 ||
    first === 10 ||
    first === 127 ||
    first >= 224 ||
    (first === 100 && second >= 64 && second <= 127) ||
    (first === 169 && second === 254) ||
    (first === 172 && second >= 16 && second <= 31) ||
    (first === 192 && second === 168)
  );
}

async function limitedBytes(
  response: Response,
  maximum: number,
): Promise<Buffer> {
  if (!response.body) throw new Error("Remote asset response was empty.");
  const declared = Number(response.headers.get("content-length") ?? 0);
  if (declared > maximum)
    throw new Error("Remote image exceeds the 20 MB import limit.");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maximum) {
      await reader.cancel();
      throw new Error("Remote image exceeds the 20 MB import limit.");
    }
    chunks.push(value);
  }
  return Buffer.concat(
    chunks.map((chunk) => Buffer.from(chunk)),
    total,
  );
}

function matchesMagic(bytes: Buffer, mimeType: string): boolean {
  if (mimeType === "image/jpeg") return bytes[0] === 0xff && bytes[1] === 0xd8;
  if (mimeType === "image/png")
    return bytes
      .subarray(0, 8)
      .equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
  if (mimeType === "image/webp")
    return (
      bytes.subarray(0, 4).toString("ascii") === "RIFF" &&
      bytes.subarray(8, 12).toString("ascii") === "WEBP"
    );
  return false;
}

function discoveryQueries(caseValue: Case): string[] {
  const profile = caseValue.profile;
  const candidates = [
    profile?.tags.name,
    profile?.tags["name:en"],
    profile?.tags["addr:housename"],
    caseValue.addressInput,
    caseValue.displayAddress,
  ];
  const normalized = candidates
    .filter((candidate): candidate is string => Boolean(candidate?.trim()))
    .map((candidate) => candidate.replace(/\s+/g, " ").trim());
  return [...new Set(normalized)];
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Unknown discovery error";
}
