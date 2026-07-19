export type WikimediaImage = {
  title: string;
  landingUrl: string;
  downloadUrl: string;
  thumbnailUrl?: string;
  creator?: string;
  license: string;
  licenseCode: string;
  licenseUrl?: string;
  source: "Wikimedia Commons";
};

type MetadataValue = { value?: string };
type WikimediaResponse = {
  query?: {
    pages?: Array<{
      title?: string;
      imageinfo?: Array<{
        url?: string;
        thumburl?: string;
        descriptionurl?: string;
        mime?: string;
        extmetadata?: Record<string, MetadataValue>;
      }>;
    }>;
  };
};

export async function discoverWikimediaImages(
  queries: string[],
  fetcher: typeof fetch = fetch,
): Promise<WikimediaImage[]> {
  const normalizedQueries = [
    ...new Set(
      queries
        .map((query) => query.replace(/\s+/g, " ").trim())
        .filter((query) => query.length >= 4),
    ),
  ].slice(0, 3);
  const results: WikimediaImage[] = [];
  const seen = new Set<string>();
  for (const query of normalizedQueries) {
    const url = new URL("https://commons.wikimedia.org/w/api.php");
    url.searchParams.set("action", "query");
    url.searchParams.set("generator", "search");
    url.searchParams.set("gsrsearch", query);
    url.searchParams.set("gsrnamespace", "6");
    url.searchParams.set("gsrlimit", "8");
    url.searchParams.set("prop", "imageinfo");
    url.searchParams.set("iiprop", "url|mime|size|extmetadata");
    url.searchParams.set("iiurlwidth", "1600");
    url.searchParams.set("format", "json");
    url.searchParams.set("formatversion", "2");
    url.searchParams.set("origin", "*");
    const response = await fetcher(url, {
      headers: {
        accept: "application/json",
        "user-agent":
          "StructureFirst/0.2 (open-license emergency structure research)",
      },
      signal: AbortSignal.timeout(25_000),
    });
    if (!response.ok)
      throw new Error(`Wikimedia Commons returned ${response.status}.`);
    const payload = (await response.json()) as WikimediaResponse;
    for (const page of payload.query?.pages ?? []) {
      const info = page.imageinfo?.[0];
      if (
        !info?.descriptionurl ||
        !info.url ||
        !info.mime?.startsWith("image/")
      )
        continue;
      if (seen.has(info.descriptionurl)) continue;
      const license = licenseDetails(info.extmetadata);
      if (!license) continue;
      seen.add(info.descriptionurl);
      const downloadUrl = info.thumburl ?? info.url;
      const creator = cleanMetadata(info.extmetadata?.Artist?.value);
      results.push({
        title: cleanTitle(page.title) || "Wikimedia Commons building image",
        landingUrl: info.descriptionurl,
        downloadUrl,
        ...(info.thumburl ? { thumbnailUrl: info.thumburl } : {}),
        ...(creator ? { creator } : {}),
        license: license.label,
        licenseCode: license.code,
        ...(license.url ? { licenseUrl: license.url } : {}),
        source: "Wikimedia Commons",
      });
    }
    if (results.length >= 8) break;
  }
  return results.slice(0, 8);
}

function licenseDetails(
  metadata: Record<string, MetadataValue> | undefined,
): { code: string; label: string; url?: string } | undefined {
  const rawCode = cleanMetadata(metadata?.License?.value)?.toLowerCase() ?? "";
  const rawLabel =
    cleanMetadata(metadata?.LicenseShortName?.value) ??
    cleanMetadata(metadata?.UsageTerms?.value) ??
    rawCode;
  const normalized = `${rawCode} ${rawLabel}`.toLowerCase();
  if (
    normalized.includes("noncommercial") ||
    normalized.includes("no derivatives") ||
    /\bcc[- ]?by[- ]?(?:nc|nd)\b/.test(normalized)
  )
    return undefined;
  let code: string | undefined;
  if (
    normalized.includes("public domain") ||
    normalized.includes("public-domain") ||
    normalized.includes("cc0") ||
    normalized.includes("pd-old")
  )
    code = normalized.includes("cc0") ? "cc0" : "pdm";
  else if (normalized.includes("cc by-sa") || normalized.includes("cc-by-sa"))
    code = "by-sa";
  else if (normalized.includes("cc by") || normalized.includes("cc-by"))
    code = "by";
  if (!code) return undefined;
  const licenseUrl = cleanMetadata(metadata?.LicenseUrl?.value);
  return {
    code,
    label: rawLabel || code.toUpperCase(),
    ...(licenseUrl ? { url: licenseUrl.replace(/^http:/, "https:") } : {}),
  };
}

function cleanTitle(value: string | undefined): string {
  return (value ?? "")
    .replace(/^File:/i, "")
    .replace(/_/g, " ")
    .trim();
}

function cleanMetadata(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const cleaned = value
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#0*39;|&apos;/gi, "'")
    .replace(/\s+/g, " ")
    .trim();
  return cleaned ? cleaned.slice(0, 500) : undefined;
}
