import type {
  EvidenceAsset,
  ReconstructionArtifact,
  RoomType,
} from "@structurefirst/contracts";
import { registeredEvidenceIds, roomTypeTitle } from "./room-availability";

export function maximalReadyArtifacts(
  artifacts: ReconstructionArtifact[],
): ReconstructionArtifact[] {
  const ready = artifacts.filter(
    (artifact) => artifact.status === "ready" && isDisplayableScene(artifact),
  );
  const gaussian = ready.filter((artifact) => artifact.mode !== "floorplan");
  const selected: ReconstructionArtifact[] = [];
  for (const candidate of [...gaussian].sort(compareSceneCoverage)) {
    const candidateIds = new Set(registeredEvidenceIds(candidate));
    const superseded = selected.some((other) => {
      const otherIds = new Set(registeredEvidenceIds(other));
      return (
        otherIds.size >= candidateIds.size &&
        [...candidateIds].every((id) => otherIds.has(id))
      );
    });
    if (!superseded) selected.push(candidate);
  }
  const selectedIds = new Set(selected.map((artifact) => artifact.id));
  return ready.filter(
    (artifact) => artifact.mode === "floorplan" || selectedIds.has(artifact.id),
  );
}

function isDisplayableScene(artifact: ReconstructionArtifact): boolean {
  if (artifact.mode !== "multi_image") return true;
  const supported = artifact.quality?.crossViewSupportedRatio;
  if (supported === undefined) return true;
  const connected = artifact.registration?.connectedFrameCount ?? 0;

  // Older artifacts predate mergeValidation. Keep their source data on disk,
  // but do not let a weak composite suppress the clear per-view SHARP scenes.
  if (supported < 0.15) return false;
  return connected < 3 || supported >= 0.3;
}

function compareSceneCoverage(
  left: ReconstructionArtifact,
  right: ReconstructionArtifact,
): number {
  return (
    registeredEvidenceIds(right).length - registeredEvidenceIds(left).length ||
    right.confidence.score - left.confidence.score ||
    (right.gaussianCount ?? 0) - (left.gaussianCount ?? 0) ||
    right.updatedAt.localeCompare(left.updatedAt) ||
    left.id.localeCompare(right.id)
  );
}

export function artifactSceneLabels(
  artifacts: ReconstructionArtifact[],
  photos: EvidenceAsset[],
): Map<string, string> {
  const byId = new Map(photos.map((photo) => [photo.id, photo]));
  const entries = artifacts.map((artifact) => {
    const ids = registeredEvidenceIds(artifact);
    const registered = ids.flatMap((id) =>
      byId.get(id) ? [byId.get(id)!] : [],
    );
    return {
      artifact,
      base: artifactSceneBase(artifact, registered),
      views: Math.max(1, ids.length),
      order: Math.min(
        ...registered.map(
          (photo) => photo.capture?.captureOrder ?? Number.MAX_SAFE_INTEGER,
        ),
      ),
    };
  });
  const totals = new Map<string, number>();
  for (const entry of entries) {
    if (entry.artifact.mode === "floorplan") continue;
    totals.set(entry.base, (totals.get(entry.base) ?? 0) + 1);
  }
  const ordinal = new Map<string, number>();
  const labels = new Map<string, string>();
  for (const entry of [...entries].sort(
    (left, right) =>
      left.order - right.order ||
      left.artifact.createdAt.localeCompare(right.artifact.createdAt) ||
      left.artifact.id.localeCompare(right.artifact.id),
  )) {
    if (entry.artifact.mode === "floorplan") {
      labels.set(entry.artifact.id, `${entry.base} · ${entry.views}`);
      continue;
    }
    const next = (ordinal.get(entry.base) ?? 0) + 1;
    ordinal.set(entry.base, next);
    const numbered = (totals.get(entry.base) ?? 0) > 1;
    labels.set(
      entry.artifact.id,
      `${entry.base}${numbered ? ` ${next}` : ""} · ${entry.views} ${entry.views === 1 ? "view" : "views"}`,
    );
  }
  return labels;
}

function artifactSceneBase(
  artifact: ReconstructionArtifact,
  registered: EvidenceAsset[],
): string {
  if (artifact.mode === "floorplan") return "House map";
  const definitiveIndoor = new Set<RoomType>([
    "bedroom",
    "bathroom",
    "kitchen",
    "dining_room",
    "office",
    "garage",
    "basement",
    "attic",
    "corridor",
    "closet",
    "stair",
    "utility",
  ]);
  const allExterior =
    registered.length > 0 &&
    registered.every(
      (photo) =>
        photo.visualAnalysis?.sceneType === "exterior" &&
        !definitiveIndoor.has(photo.visualAnalysis.roomType),
    );
  if (allExterior) return "Exterior";

  const counts = new Map<RoomType, number>();
  for (const photo of registered) {
    const room = photo.visualAnalysis?.roomType;
    if (!room || room === "unknown" || room === "exterior") continue;
    counts.set(room, (counts.get(room) ?? 0) + 1);
  }
  const ranked = [...counts].sort(
    (left, right) => right[1] - left[1] || left[0].localeCompare(right[0]),
  );
  if (ranked[0] && (!ranked[1] || ranked[0][1] > ranked[1][1])) {
    return roomTypeTitle(ranked[0][0]);
  }
  return artifact.fallback || artifact.mode === "single_image"
    ? "Unclassified view"
    : "Observed scene";
}
