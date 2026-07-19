export type OpenverseImage = {
  title: string;
  landingUrl: string;
  downloadUrl: string;
  thumbnailUrl?: string;
  creator?: string;
  license: string;
  licenseCode: string;
  licenseUrl?: string;
  source: string;
};

type OpenverseResponse = {
  results?: Array<{
    title?: string;
    url?: string;
    thumbnail?: string;
    foreign_landing_url?: string;
    creator?: string;
    license?: string;
    license_version?: string;
    license_url?: string;
    source?: string;
  }>;
};

export async function discoverOpenverseImages(
  address: string,
  alternatives: string[] = [],
): Promise<OpenverseImage[]> {
  const queries = [...new Set([address, ...alternatives].map(searchQuery))]
    .filter((query) => query.length >= 4)
    .slice(0, 3);
  const results: OpenverseImage[] = [];
  const seen = new Set<string>();
  for (const query of queries) {
    const images = await searchOpenverse(query);
    for (const image of images) {
      if (seen.has(image.landingUrl)) continue;
      seen.add(image.landingUrl);
      results.push(image);
    }
    if (results.length >= 12) break;
  }
  return results.slice(0, 12);
}

async function searchOpenverse(query: string): Promise<OpenverseImage[]> {
  const url = new URL("https://api.openverse.org/v1/images/");
  // Openverse treats every extra term as a relevance constraint. The street
  // address is already specific; adding generic words can hide exact matches.
  url.searchParams.set("q", query);
  url.searchParams.set("page_size", "12");
  url.searchParams.set("mature", "false");
  // Only return licenses that permit transformed works. StructureFirst may
  // turn a downloaded photo into a Gaussian representation, so ND variants
  // and unknown licenses must never enter the automatic import path.
  url.searchParams.set("license", "pdm,cc0,by,by-sa");
  url.searchParams.set("license_type", "modification");
  url.searchParams.set("filter_dead", "true");
  const response = await fetchOpenverse(url);
  if (!response.ok) throw new Error(`Openverse returned ${response.status}.`);
  const payload = (await response.json()) as OpenverseResponse;
  const results: OpenverseImage[] = [];
  for (const item of payload.results ?? []) {
    if (!item.url || !item.foreign_landing_url) continue;
    const licenseCode = item.license?.trim().toLowerCase();
    if (!licenseCode || !["pdm", "cc0", "by", "by-sa"].includes(licenseCode))
      continue;
    const license = [item.license?.toUpperCase(), item.license_version]
      .filter(Boolean)
      .join(" ");
    results.push({
      title: item.title?.trim() || "Openly licensed building image",
      landingUrl: item.foreign_landing_url,
      downloadUrl: item.url,
      ...(item.thumbnail ? { thumbnailUrl: item.thumbnail } : {}),
      ...(item.creator ? { creator: item.creator } : {}),
      license: license || "Open license (verify item record)",
      licenseCode,
      ...(item.license_url ? { licenseUrl: item.license_url } : {}),
      source: item.source || "Openverse index",
    });
  }
  return results;
}

function searchQuery(value: string): string {
  return value
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

async function fetchOpenverse(url: URL): Promise<Response> {
  let lastError: unknown;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const response = await fetch(url, {
        headers: {
          accept: "application/json",
          "user-agent":
            "StructureFirst/0.2 (open-license emergency structure research)",
        },
        signal: AbortSignal.timeout(20_000),
      });
      if (
        attempt < 2 &&
        (response.status === 408 ||
          response.status === 429 ||
          response.status >= 500)
      ) {
        await new Promise((resolve) =>
          setTimeout(resolve, 500 * (attempt + 1)),
        );
        continue;
      }
      return response;
    } catch (error) {
      lastError = error;
      if (attempt === 2) throw error;
      await new Promise((resolve) => setTimeout(resolve, 500 * (attempt + 1)));
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Openverse failed.");
}
