import type { Polygon } from "@structurefirst/contracts";
import type { AppConfig } from "../config.js";

type OverpassElement = {
  type: "way" | "relation" | "node";
  id: number;
  tags?: Record<string, string>;
  center?: { lat: number; lon: number };
  bounds?: {
    minlat: number;
    minlon: number;
    maxlat: number;
    maxlon: number;
  };
  geometry?: Array<{ lat: number; lon: number }>;
};

type OverpassResponse = { elements?: OverpassElement[] };

export interface OpenStreetMapBuilding {
  osmType: "way" | "relation";
  osmId: number;
  tags: Record<string, string>;
  footprint?: Polygon;
  levels?: number;
  buildingType?: string;
  construction?: string;
  yearBuilt?: number;
  sourceUrl: string;
}

function normalizeAddressText(value: string): string {
  const ordinals: Record<string, string> = {
    first: "1st",
    second: "2nd",
    third: "3rd",
    fourth: "4th",
    fifth: "5th",
    sixth: "6th",
    seventh: "7th",
    eighth: "8th",
    ninth: "9th",
    tenth: "10th",
  };
  const suffixes: Record<string, string> = {
    ave: "avenue",
    av: "avenue",
    blvd: "boulevard",
    dr: "drive",
    hwy: "highway",
    ln: "lane",
    pkwy: "parkway",
    pl: "place",
    rd: "road",
    st: "street",
  };
  return value
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .split(" ")
    .filter(Boolean)
    .map((token) => ordinals[token] ?? suffixes[token] ?? token)
    .join(" ");
}

function pointInsideGeometry(
  latitude: number,
  longitude: number,
  element: OverpassElement,
): boolean {
  const geometry = element.geometry;
  if (!geometry || geometry.length < 3) return false;
  let inside = false;
  for (
    let current = 0, previous = geometry.length - 1;
    current < geometry.length;
    previous = current++
  ) {
    const a = geometry[current];
    const b = geometry[previous];
    if (!a || !b) continue;
    const crosses =
      a.lat > latitude !== b.lat > latitude &&
      longitude <
        ((b.lon - a.lon) * (latitude - a.lat)) / (b.lat - a.lat) + a.lon;
    if (crosses) inside = !inside;
  }
  return inside;
}

function pointToSegmentDistanceMeters(
  latitude: number,
  longitude: number,
  start: { lat: number; lon: number },
  end: { lat: number; lon: number },
): number {
  const latitudeScale = 111_320;
  const longitudeScale = latitudeScale * Math.cos((latitude * Math.PI) / 180);
  const ax = (start.lon - longitude) * longitudeScale;
  const ay = (start.lat - latitude) * latitudeScale;
  const bx = (end.lon - longitude) * longitudeScale;
  const by = (end.lat - latitude) * latitudeScale;
  const dx = bx - ax;
  const dy = by - ay;
  const denominator = dx * dx + dy * dy;
  const ratio =
    denominator > 0
      ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / denominator))
      : 0;
  return Math.hypot(ax + ratio * dx, ay + ratio * dy);
}

function geometryDistanceMeters(
  latitude: number,
  longitude: number,
  element: OverpassElement,
): number {
  if (pointInsideGeometry(latitude, longitude, element)) return 0;
  const geometry = element.geometry;
  if (geometry && geometry.length >= 2) {
    let closest = Number.POSITIVE_INFINITY;
    for (let index = 0; index < geometry.length; index += 1) {
      const start = geometry[index];
      const end = geometry[(index + 1) % geometry.length];
      if (!start || !end) continue;
      closest = Math.min(
        closest,
        pointToSegmentDistanceMeters(latitude, longitude, start, end),
      );
    }
    if (Number.isFinite(closest)) return closest;
  }
  const center =
    element.center ??
    (element.bounds
      ? {
          lat: (element.bounds.minlat + element.bounds.maxlat) / 2,
          lon: (element.bounds.minlon + element.bounds.maxlon) / 2,
        }
      : undefined);
  if (!center) return Number.POSITIVE_INFINITY;
  return pointToSegmentDistanceMeters(latitude, longitude, center, center);
}

export function scoreBuildingCandidate(
  latitude: number,
  longitude: number,
  element: OverpassElement,
  requestedAddress = "",
): number {
  const distance = geometryDistanceMeters(latitude, longitude, element);
  if (!Number.isFinite(distance)) return Number.NEGATIVE_INFINITY;
  const normalizedAddress = normalizeAddressText(requestedAddress);
  const requestedHouse = normalizedAddress.match(/^([0-9]+[a-z]?)\b/)?.[1];
  const requestedPostcode = normalizedAddress.match(
    /\b([0-9]{5})(?: [0-9]{4})?\b/,
  )?.[1];
  const tags = element.tags ?? {};
  const candidateHouse = normalizeAddressText(tags["addr:housenumber"] ?? "");
  const candidateStreet = normalizeAddressText(tags["addr:street"] ?? "");
  const candidatePostcode = normalizeAddressText(tags["addr:postcode"] ?? "");
  let score = Math.max(0, 500 - distance * 8);
  if (pointInsideGeometry(latitude, longitude, element)) score += 800;
  if (requestedHouse && candidateHouse)
    score += candidateHouse === requestedHouse ? 1_800 : -800;
  if (candidateStreet)
    score += normalizedAddress.includes(candidateStreet) ? 1_800 : -1_800;
  if (requestedPostcode && candidatePostcode)
    score += candidatePostcode === requestedPostcode ? 500 : -500;
  return score;
}

function parsePositiveInteger(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parsed = Number.parseInt(value, 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}

function parseYear(value: string | undefined): number | undefined {
  const match = value?.match(/(?:16|17|18|19|20|21)\d{2}/);
  return match ? Number(match[0]) : undefined;
}

function toFootprint(element: OverpassElement): Polygon | undefined {
  if (
    element.type !== "way" ||
    !element.geometry ||
    element.geometry.length < 3
  )
    return undefined;
  const ring = element.geometry.map(
    ({ lon, lat }) => [lon, lat] as [number, number],
  );
  const first = ring[0];
  const last = ring.at(-1);
  if (first && last && (first[0] !== last[0] || first[1] !== last[1]))
    ring.push(first);
  return { type: "Polygon", coordinates: [ring] };
}

export async function findNearestBuilding(
  latitude: number,
  longitude: number,
  config: AppConfig,
  requestedAddress = "",
): Promise<OpenStreetMapBuilding | undefined> {
  const query = `[out:json][timeout:20];
    (
      way(around:130,${latitude},${longitude})["building"];
      relation(around:130,${latitude},${longitude})["building"];
    );
    out tags center geom 80;`;
  const response = await fetch(config.overpassBaseUrl, {
    method: "POST",
    headers: {
      accept: "application/json",
      "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
      "user-agent": config.userAgent,
    },
    body: new URLSearchParams({ data: query }),
    signal: AbortSignal.timeout(30_000),
  });
  if (!response.ok)
    throw new Error(`Building record provider returned ${response.status}.`);

  const body = (await response.json()) as OverpassResponse;
  const candidates = (body.elements ?? [])
    .filter(
      (element): element is OverpassElement & { type: "way" | "relation" } =>
        ["way", "relation"].includes(element.type),
    )
    .sort(
      (left, right) =>
        scoreBuildingCandidate(latitude, longitude, right, requestedAddress) -
        scoreBuildingCandidate(latitude, longitude, left, requestedAddress),
    );
  const candidate = candidates[0];
  if (!candidate) return undefined;

  const tags = candidate.tags ?? {};
  const levels = parsePositiveInteger(tags["building:levels"]);
  const buildingType =
    tags.building && tags.building !== "yes" ? tags.building : undefined;
  const construction = tags["building:material"] ?? tags.material;
  const yearBuilt = parseYear(tags.start_date);
  const footprint = toFootprint(candidate);
  return {
    osmType: candidate.type,
    osmId: candidate.id,
    tags,
    ...(footprint ? { footprint } : {}),
    ...(levels ? { levels } : {}),
    ...(buildingType ? { buildingType } : {}),
    ...(construction ? { construction } : {}),
    ...(yearBuilt ? { yearBuilt } : {}),
    sourceUrl: `https://www.openstreetmap.org/${candidate.type}/${candidate.id}`,
  };
}
