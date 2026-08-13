import type {
  EvidenceAsset,
  ReconstructionArtifact,
  RoomType,
  SpatialNode,
} from "@structurefirst/contracts";

export type PlanRoomSelection = {
  label: string;
  floorNumber: number;
  sourceEvidenceId: string;
  sameTypeLabelCount: number;
};

export type PlanRoomCandidate = {
  artifact: ReconstructionArtifact;
  space: SpatialNode;
  previewEvidenceId?: string;
  previewUrl?: string;
  previewTitle?: string;
  registeredViewCount: number;
};

export type PlanRoomResolution = {
  available: boolean;
  detail: string;
  artifact?: ReconstructionArtifact;
  space?: SpatialNode;
  candidates: PlanRoomCandidate[];
};

export function artifactForSpace(
  space: Pick<SpatialNode, "artifactId" | "sourceIds">,
  artifacts: ReconstructionArtifact[],
): ReconstructionArtifact | undefined {
  const readyGaussianArtifacts = artifacts.filter(
    (artifact) =>
      artifact.status === "ready" &&
      artifact.mode !== "floorplan" &&
      Boolean(artifact.splatUrl),
  );
  if (space.artifactId) {
    const direct = readyGaussianArtifacts.find(
      (artifact) => artifact.id === space.artifactId,
    );
    if (direct) return direct;
  }
  const sources = new Set(space.sourceIds);
  return readyGaussianArtifacts.find((artifact) =>
    registeredEvidenceIds(artifact).some((id) => sources.has(id)),
  );
}

export function registeredEvidenceIds(
  artifact: ReconstructionArtifact,
): string[] {
  const calibrated = artifact.geometry?.cameraPoses.map(
    (pose) => pose.evidenceId,
  );
  if (calibrated?.length) return calibrated;
  return artifact.evidenceIds ?? [artifact.evidenceId];
}

export function normalizeRoomLabel(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

export function roomTypeFromPlanLabel(label: string): RoomType | undefined {
  const value = normalizeRoomLabel(label);
  if (/\b(?:bath|bathroom|wc|lavatory|powder)\b/.test(value)) return "bathroom";
  if (/\b(?:bed|bedroom|master|primary)\b/.test(value)) return "bedroom";
  if (/\bkitchen\b/.test(value)) return "kitchen";
  if (/\b(?:living|great room|family room|lounge)\b/.test(value))
    return "living_room";
  if (/\bdining\b/.test(value)) return "dining_room";
  if (/\b(?:office|study)\b/.test(value)) return "office";
  if (/\bgarage\b/.test(value)) return "garage";
  if (/\bbasement\b/.test(value)) return "basement";
  if (/\b(?:attic|loft)\b/.test(value)) return "attic";
  if (/\b(?:hall|hallway|corridor|foyer|entry)\b/.test(value))
    return "corridor";
  if (/\b(?:closet|wardrobe|wic)\b/.test(value)) return "closet";
  if (/\b(?:stair|stairs|stairway)\b/.test(value)) return "stair";
  if (/\b(?:utility|laundry|mechanical|mech)\b/.test(value)) return "utility";
  return undefined;
}

export function resolvePlanRoom(
  selection: PlanRoomSelection,
  nodes: SpatialNode[],
  artifacts: ReconstructionArtifact[],
  evidence: EvidenceAsset[] = [],
): PlanRoomResolution {
  const requestedType = roomTypeFromPlanLabel(selection.label);
  if (!requestedType) {
    return {
      available: false,
      detail: "No photographed 3D scene is linked to this plan label.",
      candidates: [],
    };
  }

  const linked = nodes
    .filter(
      (node) => node.kind !== "exterior" && node.roomType === requestedType,
    )
    .map((space) => ({ space, artifact: artifactForSpace(space, artifacts) }))
    .filter(
      (
        candidate,
      ): candidate is {
        space: SpatialNode;
        artifact: ReconstructionArtifact;
      } => Boolean(candidate.artifact),
    );
  const distinct = linked.filter(
    (candidate, index) =>
      linked.findIndex(
        (other) => other.artifact.id === candidate.artifact.id,
      ) === index,
  );
  const candidates = distinct
    .map(({ artifact, space }): PlanRoomCandidate => {
      const registeredIds = registeredEvidenceIds(artifact);
      const preview = registeredIds
        .map((id) => evidence.find((item) => item.id === id))
        .find((item) => Boolean(item?.localUrl ?? item?.thumbnailUrl));
      const candidate: PlanRoomCandidate = {
        artifact,
        space,
        registeredViewCount: registeredIds.length,
      };
      const previewEvidenceId = preview?.id ?? registeredIds[0];
      const previewUrl = preview?.localUrl ?? preview?.thumbnailUrl;
      if (previewEvidenceId) candidate.previewEvidenceId = previewEvidenceId;
      if (previewUrl) candidate.previewUrl = previewUrl;
      if (preview?.title) candidate.previewTitle = preview.title;
      return candidate;
    })
    .sort(
      (left, right) =>
        right.registeredViewCount - left.registeredViewCount ||
        left.artifact.id.localeCompare(right.artifact.id),
    );

  if (candidates.length === 0) {
    return {
      available: false,
      detail:
        "Photos exist independently, but no registered Gaussian scene is linked to this room.",
      candidates: [],
    };
  }
  if (selection.sameTypeLabelCount > 1) {
    return {
      available: false,
      detail: `${candidates.length} reconstructed ${roomTypeTitle(requestedType).toLowerCase()} ${candidates.length === 1 ? "scene is" : "scenes are"} available, but the plan contains ${selection.sameTypeLabelCount} ${roomTypeTitle(requestedType).toLowerCase()} labels on this floor. Choose a source preview; its exact plan position is not verified.`,
      candidates,
    };
  }
  if (candidates.length > 1) {
    return {
      available: false,
      detail: `${candidates.length} ${roomTypeTitle(requestedType).toLowerCase()} Gaussian scenes exist; choose a source preview because the floor-plan position is not verified.`,
      candidates,
    };
  }
  return {
    available: true,
    detail: "3D available · unique room-label match",
    candidates,
    artifact: candidates[0]!.artifact,
    space: candidates[0]!.space,
  };
}

export function roomTypeTitle(value: RoomType): string {
  if (value === "unknown") return "Unclassified space";
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
