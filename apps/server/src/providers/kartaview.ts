export type KartaViewImage = {
  title: string;
  landingUrl: string;
  downloadUrl: string;
  thumbnailUrl?: string;
  observedAt?: string;
  sequenceId: string;
  sequenceIndex: number;
  photoId: string;
  distanceMeters: number;
  headingDifference: number;
};

type KartaViewPhoto = {
  id?: string;
  sequenceId?: string;
  sequenceIndex?: string;
  lat?: string;
  lng?: string;
  heading?: string;
  headers?: string;
  shotDate?: string;
  status?: string;
  visibility?: string;
  imageProcessingStatus?: string;
  autoImgProcessingStatus?: string;
  imageProcUrl?: string;
  imageLthUrl?: string;
  imageThUrl?: string;
};

type KartaViewResponse = {
  status?: { apiCode?: number; httpCode?: number; apiMessage?: string };
  result?: { data?: KartaViewPhoto[] };
};

const API_ROOT = "https://api.openstreetcam.org/2.0/photo/";
const USER_AGENT =
  "StructureFirst/0.2 (open-licensed emergency structure research)";
const PAGE_SIZE = 150;

export async function discoverKartaViewImages(
  latitude: number,
  longitude: number,
  fetcher: typeof fetch = fetch,
): Promise<KartaViewImage[]> {
  const nearbyUrl = new URL(API_ROOT);
  nearbyUrl.searchParams.set("lat", String(latitude));
  nearbyUrl.searchParams.set("lng", String(longitude));
  nearbyUrl.searchParams.set("radius", "180");
  nearbyUrl.searchParams.set("itemsPerPage", "100");
  nearbyUrl.searchParams.set("orderBy", "id");
  nearbyUrl.searchParams.set("orderDirection", "desc");
  const nearby = await fetchPhotos(nearbyUrl, fetcher);
  const ranked = nearby
    .map((photo) => scoredPhoto(photo, latitude, longitude))
    .filter((photo): photo is ScoredPhoto => Boolean(photo))
    .filter((photo) => photo.distanceMeters <= 220)
    .sort((left, right) => right.score - left.score);
  const anchor = ranked[0];
  if (!anchor) return [];

  let sequence = [anchor.photo];
  try {
    sequence = await fetchSequenceWindow(anchor, fetcher);
  } catch {
    // A nearby photo is still useful when the sequence endpoint is unavailable.
  }
  const candidates = sequence
    .map((photo) => scoredPhoto(photo, latitude, longitude))
    .filter((photo): photo is ScoredPhoto => Boolean(photo))
    .filter(
      (photo) =>
        photo.photo.sequenceId === anchor.photo.sequenceId &&
        photo.distanceMeters <= 220 &&
        Math.abs(photo.sequenceIndex - anchor.sequenceIndex) <= 8,
    );
  const selected = selectOverlappingFrames(candidates, anchor.sequenceIndex);
  return (selected.length > 0 ? selected : [anchor]).map((photo, index, all) =>
    toResult(photo, index, all.length),
  );
}

type ScoredPhoto = {
  photo: KartaViewPhoto;
  sequenceIndex: number;
  distanceMeters: number;
  headingDifference: number;
  score: number;
};

function scoredPhoto(
  photo: KartaViewPhoto,
  targetLatitude: number,
  targetLongitude: number,
): ScoredPhoto | undefined {
  const latitude = Number(photo.lat);
  const longitude = Number(photo.lng);
  const sequenceIndex = Number(photo.sequenceIndex);
  const heading = Number(photo.heading ?? photo.headers);
  if (
    !photo.id ||
    !photo.sequenceId ||
    !photo.imageProcUrl ||
    !Number.isFinite(latitude) ||
    !Number.isFinite(longitude) ||
    !Number.isInteger(sequenceIndex) ||
    photo.status === "deleted" ||
    (photo.visibility && photo.visibility !== "public")
  )
    return undefined;
  const distanceMeters = haversineMeters(
    latitude,
    longitude,
    targetLatitude,
    targetLongitude,
  );
  const targetBearing = bearingDegrees(
    latitude,
    longitude,
    targetLatitude,
    targetLongitude,
  );
  const headingDifference = Number.isFinite(heading)
    ? angularDifference(heading, targetBearing)
    : 90;
  const distanceScore = Math.max(0, 1 - Math.abs(distanceMeters - 35) / 220);
  const headingScore = Math.max(0, 1 - headingDifference / 135);
  const dateScore = recencyScore(photo.shotDate);
  return {
    photo,
    sequenceIndex,
    distanceMeters,
    headingDifference,
    score: distanceScore * 0.55 + headingScore * 0.3 + dateScore * 0.15,
  };
}

async function fetchSequenceWindow(
  anchor: ScoredPhoto,
  fetcher: typeof fetch,
): Promise<KartaViewPhoto[]> {
  const page = Math.floor(anchor.sequenceIndex / PAGE_SIZE) + 1;
  const pages = new Set([Math.max(1, page - 1), page, page + 1]);
  const responses = await Promise.all(
    [...pages].map(async (pageNumber) => {
      const url = new URL(API_ROOT);
      url.searchParams.set("sequenceId", anchor.photo.sequenceId as string);
      url.searchParams.set("page", String(pageNumber));
      url.searchParams.set("itemsPerPage", String(PAGE_SIZE));
      return fetchPhotos(url, fetcher);
    }),
  );
  return responses.flat();
}

function selectOverlappingFrames(
  candidates: ScoredPhoto[],
  anchorIndex: number,
): ScoredPhoto[] {
  const byIndex = new Map(
    candidates
      .filter((candidate) => candidate.headingDifference <= 110)
      .map((candidate) => [candidate.sequenceIndex, candidate]),
  );
  const offsets = [0, -2, 2, -4, 4, -1, 1, -3, 3];
  const chosen: ScoredPhoto[] = [];
  for (const offset of offsets) {
    const candidate = byIndex.get(anchorIndex + offset);
    if (!candidate) continue;
    chosen.push(candidate);
    if (chosen.length === 3) break;
  }
  return chosen.sort((left, right) => left.sequenceIndex - right.sequenceIndex);
}

function toResult(
  candidate: ScoredPhoto,
  index: number,
  count: number,
): KartaViewImage {
  const photo = candidate.photo;
  const sequenceId = photo.sequenceId as string;
  const photoId = photo.id as string;
  const observedAt = parseObservedAt(photo.shotDate);
  return {
    title: `KartaView street capture ${index + 1}/${count} (${Math.round(candidate.distanceMeters)} m from address point)`,
    landingUrl: `https://kartaview.org/details/${sequenceId}/${candidate.sequenceIndex}/track-info`,
    downloadUrl: photo.imageProcUrl as string,
    ...((photo.imageThUrl ?? photo.imageLthUrl)
      ? { thumbnailUrl: photo.imageThUrl ?? photo.imageLthUrl }
      : {}),
    ...(observedAt ? { observedAt } : {}),
    sequenceId,
    sequenceIndex: candidate.sequenceIndex,
    photoId,
    distanceMeters: Math.round(candidate.distanceMeters * 10) / 10,
    headingDifference: Math.round(candidate.headingDifference * 10) / 10,
  };
}

async function fetchPhotos(
  url: URL,
  fetcher: typeof fetch,
): Promise<KartaViewPhoto[]> {
  let lastError: unknown;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const response = await fetcher(url, {
        headers: { accept: "application/json", "user-agent": USER_AGENT },
        signal: AbortSignal.timeout(25_000),
      });
      if (
        attempt < 2 &&
        (response.status === 408 ||
          response.status === 429 ||
          response.status >= 500)
      ) {
        await delay(500 * (attempt + 1));
        continue;
      }
      if (!response.ok)
        throw new Error(`KartaView returned ${response.status}.`);
      const payload = (await response.json()) as KartaViewResponse;
      if (payload.status?.httpCode && payload.status.httpCode !== 200)
        throw new Error(
          `KartaView returned ${payload.status.apiMessage ?? payload.status.httpCode}.`,
        );
      return payload.result?.data ?? [];
    } catch (error) {
      lastError = error;
      if (attempt === 2) throw error;
      await delay(500 * (attempt + 1));
    }
  }
  throw lastError instanceof Error ? lastError : new Error("KartaView failed.");
}

function haversineMeters(
  latitudeA: number,
  longitudeA: number,
  latitudeB: number,
  longitudeB: number,
): number {
  const radians = Math.PI / 180;
  const deltaLatitude = (latitudeB - latitudeA) * radians;
  const deltaLongitude = (longitudeB - longitudeA) * radians;
  const a =
    Math.sin(deltaLatitude / 2) ** 2 +
    Math.cos(latitudeA * radians) *
      Math.cos(latitudeB * radians) *
      Math.sin(deltaLongitude / 2) ** 2;
  return 6_371_000 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function bearingDegrees(
  latitudeA: number,
  longitudeA: number,
  latitudeB: number,
  longitudeB: number,
): number {
  const radians = Math.PI / 180;
  const latitudeARadians = latitudeA * radians;
  const latitudeBRadians = latitudeB * radians;
  const deltaLongitude = (longitudeB - longitudeA) * radians;
  const y = Math.sin(deltaLongitude) * Math.cos(latitudeBRadians);
  const x =
    Math.cos(latitudeARadians) * Math.sin(latitudeBRadians) -
    Math.sin(latitudeARadians) *
      Math.cos(latitudeBRadians) *
      Math.cos(deltaLongitude);
  return (Math.atan2(y, x) / radians + 360) % 360;
}

function angularDifference(left: number, right: number): number {
  return Math.abs(((left - right + 540) % 360) - 180);
}

function recencyScore(value: string | undefined): number {
  const observedAt = parseObservedAt(value);
  if (!observedAt) return 0;
  const ageYears =
    (Date.now() - new Date(observedAt).getTime()) /
    (365.25 * 24 * 60 * 60 * 1000);
  return Math.max(0, 1 - ageYears / 15);
}

function parseObservedAt(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const parsed = new Date(
    value.replace(" ", "T") + (value.endsWith("Z") ? "" : "Z"),
  );
  return Number.isNaN(parsed.getTime()) ? undefined : parsed.toISOString();
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
