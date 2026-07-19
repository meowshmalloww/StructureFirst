import type { CachePolicy, RightsStatus } from "@structurefirst/contracts";

export type SourcePolicyDecision = {
  provider: string;
  rights: RightsStatus;
  cachePolicy: CachePolicy;
  redistributable: boolean;
  hardBlocked: boolean;
  reason: string;
};

const LINK_ONLY_HOSTS = [
  "youtube.com",
  "youtu.be",
  "google.com",
  "googleusercontent.com",
  "gstatic.com",
];

const RESTRICTED_HOSTS = [
  "facebook.com",
  "instagram.com",
  "tiktok.com",
  "x.com",
  "twitter.com",
];

function hostMatches(hostname: string, candidates: string[]): boolean {
  return candidates.some(
    (candidate) => hostname === candidate || hostname.endsWith(`.${candidate}`),
  );
}

export function classifySource(rawUrl: string): SourcePolicyDecision {
  const url = new URL(rawUrl);
  const hostname = url.hostname.toLowerCase().replace(/^www\./, "");

  if (hostMatches(hostname, ["youtube.com", "youtu.be"])) {
    return {
      provider: "YouTube",
      rights: "link_only",
      cachePolicy: "prohibited",
      redistributable: false,
      hardBlocked: true,
      reason:
        "YouTube audiovisual media is link/embed-only; StructureFirst does not download or cache it.",
    };
  }

  if (
    hostMatches(hostname, [
      "google.com",
      "googleusercontent.com",
      "gstatic.com",
    ])
  ) {
    return {
      provider: "Google",
      rights: "link_only",
      cachePolicy: "prohibited",
      redistributable: false,
      hardBlocked: true,
      reason:
        "Google imagery is retained as a live link only; imagery bytes are never cached.",
    };
  }

  if (hostMatches(hostname, RESTRICTED_HOSTS)) {
    return {
      provider: hostname,
      rights: "restricted",
      cachePolicy: "metadata_only",
      redistributable: false,
      hardBlocked: false,
      reason:
        "The source commonly restricts automated copying; only discovery metadata is retained.",
    };
  }

  if (hostMatches(hostname, ["zillow.com", "redfin.com"])) {
    const provider = hostMatches(hostname, ["zillow.com"])
      ? "Zillow"
      : "Redfin";
    return {
      provider,
      rights: "link_only",
      cachePolicy: "metadata_only",
      redistributable: false,
      hardBlocked: true,
      reason: `${provider} listing results are retained as address-check links only. StructureFirst does not crawl the listing page or copy its photos.`,
    };
  }

  if (hostMatches(hostname, ["openstreetmap.org", "openstreetmap.fr"])) {
    return {
      provider: "OpenStreetMap",
      rights: "open_license",
      cachePolicy: "local_allowed",
      redistributable: true,
      hardBlocked: false,
      reason:
        "OpenStreetMap data is available under ODbL with attribution and share-alike obligations.",
    };
  }

  if (
    hostMatches(hostname, ["loc.gov", "archives.gov", "usgs.gov", "nasa.gov"])
  ) {
    return {
      provider: hostname,
      rights: "public_domain",
      cachePolicy: "local_allowed",
      redistributable: true,
      hardBlocked: false,
      reason:
        "Likely U.S. public-domain source; item-level rights still require operator confirmation.",
    };
  }

  return {
    provider: hostname,
    rights: "research_unknown",
    cachePolicy: "metadata_only",
    redistributable: false,
    hardBlocked: hostMatches(hostname, LINK_ONLY_HOSTS),
    reason:
      "Rights are not established. Keep discovery metadata only and exclude the asset from exports.",
  };
}

export function isRemoteHttpUrl(rawUrl: string): boolean {
  try {
    const url = new URL(rawUrl);
    return url.protocol === "https:" || url.protocol === "http:";
  } catch {
    return false;
  }
}
