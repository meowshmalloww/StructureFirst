import type { EvidenceKind } from "@structurefirst/contracts";

type BraveImageResult = {
  title?: string;
  url?: string;
  source?: string;
  page_url?: string;
  thumbnail?: { src?: string };
};

type BraveWebResult = {
  title?: string;
  url?: string;
  description?: string;
};

type BraveImageResponse = { results?: BraveImageResult[] };
type BraveWebResponse = { web?: { results?: BraveWebResult[] } };

export interface DiscoveredLink {
  title: string;
  url: string;
  kind: EvidenceKind;
  thumbnailUrl?: string;
  notes: string;
}

async function braveRequest<T>(
  endpoint: string,
  query: string,
  apiKey: string,
): Promise<T> {
  const url = new URL(endpoint);
  url.searchParams.set("q", query);
  url.searchParams.set("count", "12");
  url.searchParams.set("safesearch", "strict");
  const response = await fetch(url, {
    headers: {
      accept: "application/json",
      "x-subscription-token": apiKey,
    },
    signal: AbortSignal.timeout(20_000),
  });
  if (!response.ok)
    throw new Error(`Brave Search returned ${response.status}.`);
  return (await response.json()) as T;
}

export async function discoverBuildingEvidence(
  address: string,
  apiKey: string,
): Promise<DiscoveredLink[]> {
  const [images, web] = await Promise.all([
    braveRequest<BraveImageResponse>(
      "https://api.search.brave.com/res/v1/images/search",
      `"${address}" building exterior property`,
      apiKey,
    ),
    braveRequest<BraveWebResponse>(
      "https://api.search.brave.com/res/v1/web/search",
      `"${address}" property building records floor plan`,
      apiKey,
    ),
  ]);

  const found: DiscoveredLink[] = [];
  for (const result of images.results ?? []) {
    const url = result.page_url ?? result.url;
    if (!url || !result.title) continue;
    found.push({
      title: result.title,
      url,
      kind: "image",
      ...(result.thumbnail?.src ? { thumbnailUrl: result.thumbnail.src } : {}),
      notes:
        "Discovered through Brave Image Search; visual relevance and usage rights require review.",
    });
  }
  for (const result of web.web?.results ?? []) {
    if (!result.url || !result.title) continue;
    const lower = `${result.title} ${result.description ?? ""}`.toLowerCase();
    const kind: EvidenceKind =
      lower.includes("floor plan") || lower.includes("blueprint")
        ? "blueprint"
        : lower.includes("record") || lower.includes("assessor")
          ? "record"
          : "web_page";
    found.push({
      title: result.title,
      url: result.url,
      kind,
      notes:
        result.description ??
        "Discovered through Brave Web Search; verify before operational use.",
    });
  }

  return [...new Map(found.map((item) => [item.url, item])).values()].slice(
    0,
    20,
  );
}
