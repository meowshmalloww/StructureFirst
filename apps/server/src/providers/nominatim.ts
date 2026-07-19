import type { AppConfig } from "../config.js";

type NominatimResult = {
  lat?: string;
  lon?: string;
  name?: string;
  display_name?: string;
  osm_type?: string;
  osm_id?: number;
  importance?: number;
  address?: {
    house_number?: string;
    road?: string;
    city?: string;
    town?: string;
    village?: string;
    municipality?: string;
    state?: string;
    postcode?: string;
    country_code?: string;
  };
};

type CensusResponse = {
  result?: {
    addressMatches?: Array<{
      matchedAddress?: string;
      coordinates?: { x?: number; y?: number };
      addressComponents?: {
        fromAddress?: string;
        streetName?: string;
        suffixType?: string;
        city?: string;
        state?: string;
        zip?: string;
      };
    }>;
  };
};

export interface GeocodedAddress {
  displayAddress: string;
  latitude: number;
  longitude: number;
  osmType?: string;
  osmId?: number;
  provider?: string;
  sourceUrl?: string;
  license?: string;
  confidenceScore?: number;
  matchMethod?: "census_exact" | "nominatim_ranked";
}

const US_STATES = new Set([
  "AL",
  "AK",
  "AZ",
  "AR",
  "CA",
  "CO",
  "CT",
  "DE",
  "DC",
  "FL",
  "GA",
  "HI",
  "ID",
  "IL",
  "IN",
  "IA",
  "KS",
  "KY",
  "LA",
  "ME",
  "MD",
  "MA",
  "MI",
  "MN",
  "MS",
  "MO",
  "MT",
  "NE",
  "NV",
  "NH",
  "NJ",
  "NM",
  "NY",
  "NC",
  "ND",
  "OH",
  "OK",
  "OR",
  "PA",
  "RI",
  "SC",
  "SD",
  "TN",
  "TX",
  "UT",
  "VT",
  "VA",
  "WA",
  "WV",
  "WI",
  "WY",
  "PR",
]);

const ORDINALS: Record<string, string> = {
  first: "1",
  second: "2",
  third: "3",
  fourth: "4",
  fifth: "5",
  sixth: "6",
  seventh: "7",
  eighth: "8",
  ninth: "9",
  tenth: "10",
};

let nextAllowedRequest = 0;

async function respectPublicServiceRateLimit(): Promise<void> {
  const wait = Math.max(0, nextAllowedRequest - Date.now());
  if (wait > 0) await delay(wait);
  nextAllowedRequest = Date.now() + 1_100;
}

export async function geocodeAddress(
  address: string,
  config: AppConfig,
): Promise<GeocodedAddress | undefined> {
  if (hasLeadingPlaceName(address)) {
    const namedPlace = await geocodeWithNominatim(address, config).catch(
      () => undefined,
    );
    if (namedPlace) return namedPlace;
  }
  if (looksLikeUnitedStatesAddress(address)) {
    const census = await geocodeWithCensus(address, config).catch(
      () => undefined,
    );
    if (census) return census;
  }
  return geocodeWithNominatim(address, config);
}

async function geocodeWithCensus(
  address: string,
  config: AppConfig,
): Promise<GeocodedAddress | undefined> {
  const url = new URL(
    "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress",
  );
  url.searchParams.set("address", addressWithoutPlaceName(address));
  url.searchParams.set("benchmark", "Public_AR_Current");
  url.searchParams.set("format", "json");
  const response = await fetchWithRetries(url, {
    headers: {
      accept: "application/json",
      "user-agent": config.userAgent,
    },
    attempts: 2,
  });
  if (!response.ok)
    throw new Error(
      `U.S. Census address provider returned ${response.status}.`,
    );
  const payload = (await response.json()) as CensusResponse;
  const requested = addressParts(address);
  const candidates = payload.result?.addressMatches ?? [];
  const match = candidates.find((candidate) => {
    const components = candidate.addressComponents;
    if (!candidate.matchedAddress || !components) return false;
    if (
      requested.houseNumber &&
      components.fromAddress &&
      normalizeNumber(components.fromAddress) !== requested.houseNumber
    )
      return false;
    if (
      requested.postcode &&
      components.zip &&
      components.zip.slice(0, 5) !== requested.postcode
    )
      return false;
    if (
      requested.state &&
      components.state &&
      components.state.toUpperCase() !== requested.state
    )
      return false;
    return true;
  });
  const latitude = match?.coordinates?.y;
  const longitude = match?.coordinates?.x;
  if (
    !match?.matchedAddress ||
    !Number.isFinite(latitude) ||
    !Number.isFinite(longitude)
  )
    return undefined;
  return {
    displayAddress: titleCaseAddress(match.matchedAddress),
    latitude: latitude as number,
    longitude: longitude as number,
    provider: "U.S. Census Geocoder",
    sourceUrl: "https://geocoding.geo.census.gov/geocoder/",
    license: "U.S. Government public data",
    confidenceScore: 0.9,
    matchMethod: "census_exact",
  };
}

async function geocodeWithNominatim(
  address: string,
  config: AppConfig,
): Promise<GeocodedAddress | undefined> {
  const url = new URL("/search", config.nominatimBaseUrl);
  url.searchParams.set("q", address);
  url.searchParams.set("format", "jsonv2");
  url.searchParams.set("addressdetails", "1");
  url.searchParams.set("extratags", "1");
  url.searchParams.set("namedetails", "1");
  url.searchParams.set(
    "layer",
    hasLeadingPlaceName(address) ? "address,poi" : "address",
  );
  url.searchParams.set("limit", "10");
  if (looksLikeUnitedStatesAddress(address))
    url.searchParams.set("countrycodes", "us");

  const response = await fetchWithRetries(url, {
    headers: {
      accept: "application/json",
      "accept-language": "en",
      "user-agent": config.userAgent,
    },
    attempts: 3,
    beforeAttempt: respectPublicServiceRateLimit,
  });
  if (!response.ok)
    throw new Error(`Address provider returned ${response.status}.`);

  const results = (await response.json()) as NominatimResult[];
  const ranked = results
    .map((result) => ({ result, score: scoreNominatimResult(address, result) }))
    .filter((candidate) => candidate.score >= 0.58)
    .sort((left, right) => right.score - left.score);
  const selected = ranked[0];
  const latitude = Number(selected?.result.lat);
  const longitude = Number(selected?.result.lon);
  if (
    !selected?.result.display_name ||
    !Number.isFinite(latitude) ||
    !Number.isFinite(longitude)
  )
    return undefined;

  return {
    displayAddress: selected.result.display_name,
    latitude,
    longitude,
    ...(selected.result.osm_type ? { osmType: selected.result.osm_type } : {}),
    ...(selected.result.osm_id ? { osmId: selected.result.osm_id } : {}),
    provider: "OpenStreetMap Nominatim",
    sourceUrl: "https://www.openstreetmap.org/copyright",
    license: "ODbL 1.0",
    confidenceScore: Math.min(0.82, selected.score),
    matchMethod: "nominatim_ranked",
  };
}

export function scoreNominatimResult(
  input: string,
  result: NominatimResult,
): number {
  if (!result.lat || !result.lon || !result.display_name) return 0;
  const requested = addressParts(input);
  const found = result.address ?? {};
  const resultPostcode = found.postcode?.match(/\d{5}/)?.[0];
  const requestedName = leadingPlaceName(input);
  const foundName = result.name ?? result.display_name.split(",", 1)[0] ?? "";
  const nameSimilarity = requestedName
    ? tokenSimilarity(requestedName, foundName)
    : 0;
  if (
    requested.postcode &&
    resultPostcode !== requested.postcode &&
    nameSimilarity < 0.75
  )
    return 0;
  if (
    requested.houseNumber &&
    found.house_number &&
    normalizeNumber(found.house_number) !== requested.houseNumber
  )
    return 0;
  if (
    requested.state &&
    found.state &&
    !stateMatches(requested.state, found.state)
  )
    return 0;

  const locality =
    found.city ?? found.town ?? found.village ?? found.municipality ?? "";
  if (
    requested.city &&
    locality &&
    tokenSimilarity(requested.city, locality) < 0.5
  )
    return 0;

  let score = 0.28;
  if (requested.postcode && resultPostcode === requested.postcode)
    score += 0.24;
  if (requested.houseNumber) {
    score += found.house_number ? 0.2 : 0.04;
  }
  if (found.road) score += 0.16 * tokenSimilarity(requested.street, found.road);
  if (requested.city && locality)
    score += 0.1 * tokenSimilarity(requested.city, locality);
  if (
    requested.state &&
    found.state &&
    stateMatches(requested.state, found.state)
  )
    score += 0.08;
  score += Math.min(0.04, Math.max(0, result.importance ?? 0) * 0.04);
  if (requestedName) score += 0.18 * nameSimilarity;
  return Math.min(1, score);
}

function addressParts(address: string): {
  houseNumber?: string;
  street: string;
  city?: string;
  state?: string;
  postcode?: string;
} {
  const commaParts = address
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  const numberedStreetIndex = commaParts.findIndex((part) =>
    /^\s*\d+[A-Za-z-]*\b/.test(part),
  );
  const streetIndex = numberedStreetIndex >= 0 ? numberedStreetIndex : 0;
  const street = commaParts[streetIndex] ?? address;
  const houseNumber = street.match(/^\s*(\d+[A-Za-z-]*)\b/)?.[1];
  const postcode = address.match(/\b(\d{5})(?:-\d{4})?\b/)?.[1];
  const state = findState(address);
  const city = commaParts[streetIndex + 1];
  return {
    ...(houseNumber ? { houseNumber: normalizeNumber(houseNumber) } : {}),
    street,
    ...(city ? { city } : {}),
    ...(state ? { state } : {}),
    ...(postcode ? { postcode } : {}),
  };
}

function hasLeadingPlaceName(address: string): boolean {
  return Boolean(leadingPlaceName(address));
}

function leadingPlaceName(address: string): string | undefined {
  const commaParts = address
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  return commaParts.length >= 3 &&
    !/^\d+[A-Za-z-]*\b/.test(commaParts[0] ?? "") &&
    commaParts.slice(1).some((part) => /^\d+[A-Za-z-]*\b/.test(part))
    ? commaParts[0]
    : undefined;
}

function addressWithoutPlaceName(address: string): string {
  if (!hasLeadingPlaceName(address)) return address;
  return address
    .split(",")
    .slice(1)
    .map((part) => part.trim())
    .filter(Boolean)
    .join(", ");
}

function looksLikeUnitedStatesAddress(address: string): boolean {
  return Boolean(address.match(/\b\d{5}(?:-\d{4})?\b/) || findState(address));
}

function findState(address: string): string | undefined {
  for (const match of address
    .toUpperCase()
    .matchAll(/(?:^|[,\s])([A-Z]{2})(?=[,\s]|$)/g)) {
    const candidate = match[1];
    if (candidate && US_STATES.has(candidate)) return candidate;
  }
  return undefined;
}

function stateMatches(abbreviation: string, stateName: string): boolean {
  const normalized = normalizeText(stateName);
  if (normalized === abbreviation.toLowerCase()) return true;
  const aliases: Record<string, string> = {
    NY: "new york",
    CA: "california",
    DC: "district of columbia",
    WA: "washington",
    TX: "texas",
    FL: "florida",
  };
  return aliases[abbreviation] === normalized;
}

function tokenSimilarity(left: string, right: string): number {
  const leftTokens = new Set(normalizeText(left).split(" ").filter(Boolean));
  const rightTokens = new Set(normalizeText(right).split(" ").filter(Boolean));
  if (leftTokens.size === 0 || rightTokens.size === 0) return 0;
  let intersection = 0;
  for (const token of leftTokens) if (rightTokens.has(token)) intersection += 1;
  return intersection / Math.max(leftTokens.size, rightTokens.size);
}

function normalizeText(value: string): string {
  return value
    .toLowerCase()
    .replace(/(\d+)(?:st|nd|rd|th)\b/g, "$1")
    .replace(/[\p{P}\p{S}]+/gu, " ")
    .split(/\s+/)
    .map((token) => ORDINALS[token] ?? token)
    .filter(
      (token) =>
        token &&
        ![
          "street",
          "st",
          "avenue",
          "ave",
          "road",
          "rd",
          "boulevard",
          "blvd",
          "drive",
          "dr",
          "lane",
          "ln",
          "the",
          "of",
        ].includes(token),
    )
    .join(" ");
}

function normalizeNumber(value: string): string {
  return value.trim().toLowerCase().replace(/^0+/, "");
}

function titleCaseAddress(value: string): string {
  return value
    .toLowerCase()
    .replace(/\b\p{L}/gu, (letter) => letter.toUpperCase())
    .replace(/\b(Ny|Ca|Dc|Wa|Tx|Fl)\b/g, (state) => state.toUpperCase());
}

async function fetchWithRetries(
  url: URL,
  options: {
    headers: Record<string, string>;
    attempts: number;
    beforeAttempt?: () => Promise<void>;
  },
): Promise<Response> {
  let lastError: unknown;
  for (let attempt = 0; attempt < options.attempts; attempt += 1) {
    await options.beforeAttempt?.();
    try {
      const response = await fetch(url, {
        headers: options.headers,
        signal: AbortSignal.timeout(20_000),
      });
      if (
        attempt < options.attempts - 1 &&
        (response.status === 408 ||
          response.status === 429 ||
          response.status >= 500)
      ) {
        await delay(550 * (attempt + 1));
        continue;
      }
      return response;
    } catch (error) {
      lastError = error;
      if (attempt === options.attempts - 1) throw error;
      await delay(550 * (attempt + 1));
    }
  }
  throw lastError instanceof Error
    ? lastError
    : new Error("Address fetch failed.");
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
