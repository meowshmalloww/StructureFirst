import type { EvidenceAsset } from "@structurefirst/contracts";

export const MAX_RECONSTRUCTION_BATCH_SIZE = 12;
export const RECONSTRUCTION_BRIDGE_SIZE = 3;

export type CaptureBatch = {
  index: number;
  evidenceIds: string[];
  bridgeEvidenceIds: string[];
};

export type CaptureSetPlan = {
  totalFrames: number;
  eligibleFrames: number;
  excludedFrames: number;
  maxFramesPerBatch: number;
  bridgeFramesPerBatch: number;
  batches: CaptureBatch[];
  note: string;
};

/**
 * Divide one ordered capture walk into GPU-safe jobs. Consecutive jobs share
 * exact source frames so their coordinate systems have measured bridge
 * evidence instead of relying on room labels or guessed placement.
 */
export function planCaptureSet(
  evidence: EvidenceAsset[],
  maximumBatchSize = MAX_RECONSTRUCTION_BATCH_SIZE,
  bridgeSize = RECONSTRUCTION_BRIDGE_SIZE,
): CaptureSetPlan {
  if (maximumBatchSize < 2) {
    throw new Error("A reconstruction batch must contain at least two frames.");
  }
  if (bridgeSize < 1 || bridgeSize >= maximumBatchSize) {
    throw new Error(
      "Bridge size must be between one and the batch size minus one.",
    );
  }

  const eligible = evidence.filter(isCaptureEligible).sort((left, right) => {
    const orderDifference =
      (left.capture?.captureOrder ?? Number.MAX_SAFE_INTEGER) -
      (right.capture?.captureOrder ?? Number.MAX_SAFE_INTEGER);
    return (
      orderDifference || left.discoveredAt.localeCompare(right.discoveredAt)
    );
  });
  const batches: CaptureBatch[] = [];
  const groups = semanticCaptureGroups(eligible);
  const stride = maximumBatchSize - bridgeSize;
  for (const group of groups) {
    const groupBatchStart = batches.length;
    for (let start = 0; start < group.length; start += stride) {
      const frames = group.slice(start, start + maximumBatchSize);
      if (frames.length === 1 && batches.length > groupBatchStart) {
        const previous = batches[batches.length - 1]!;
        if (!previous.evidenceIds.includes(frames[0]!.id)) {
          previous.evidenceIds.push(frames[0]!.id);
        }
        break;
      }
      if (frames.length === 0) break;
      const previousIds = new Set(
        batches.length > groupBatchStart
          ? (batches[batches.length - 1]?.evidenceIds ?? [])
          : [],
      );
      const evidenceIds = frames.map((item) => item.id);
      batches.push({
        index: batches.length,
        evidenceIds,
        bridgeEvidenceIds: evidenceIds.filter((id) => previousIds.has(id)),
      });
      if (start + maximumBatchSize >= group.length) break;
    }
  }

  const excludedFrames = evidence.length - eligible.length;
  return {
    totalFrames: evidence.length,
    eligibleFrames: eligible.length,
    excludedFrames,
    maxFramesPerBatch: maximumBatchSize,
    bridgeFramesPerBatch: bridgeSize,
    batches,
    note:
      groups.length > 1
        ? `${groups.length} automatically organized scene groups keep exteriors and each visible room type separate. Geometric verification then divides repeated room types into distinct room instances; only measured overlap enters a Gaussian scene.`
        : batches.length <= 1
          ? "The eligible capture set fits in one joint reconstruction job."
          : `${batches.length} ordered reconstruction jobs share up to ${bridgeSize} exact bridge frames. A room or floor can only align when those bridge frames contain verified visual overlap.`,
  };
}

function semanticCaptureGroups(eligible: EvidenceAsset[]): EvidenceAsset[][] {
  const analyzed = eligible.filter((item) => item.visualAnalysis);
  if (analyzed.length < Math.ceil(eligible.length * 0.75)) return [eligible];
  const byZone = new Map<string, EvidenceAsset[]>();
  for (const item of eligible) {
    const zone = captureZone(item);
    const current = byZone.get(zone) ?? [];
    current.push(item);
    byZone.set(zone, current);
  }
  if (byZone.size <= 1) return [eligible];

  const result = [...byZone.values()].map((items) =>
    items.sort(compareCaptureOrder),
  );
  return result.sort(
    (left, right) =>
      (left[0]?.capture?.captureOrder ?? Number.MAX_SAFE_INTEGER) -
      (right[0]?.capture?.captureOrder ?? Number.MAX_SAFE_INTEGER),
  );
}

function captureZone(item: EvidenceAsset): string {
  const analysis = item.visualAnalysis;
  const roomType = analysis?.roomType;
  const definitiveInteriorRooms = new Set([
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
  // Some VLMs call an indoor room an exterior when a window dominates the
  // frame. A concrete indoor room label is more useful than that weak scene
  // label, while exterior + living_room remains exterior because patios and
  // facades are frequently described as outdoor living space.
  if (
    roomType === "exterior" ||
    (analysis?.sceneType === "exterior" &&
      !definitiveInteriorRooms.has(roomType ?? ""))
  )
    return "exterior";
  if (roomType && roomType !== "unknown") return `room:${roomType}`;
  return analysis?.sceneType === "interior" ? "interior_other" : "unknown";
}

function compareCaptureOrder(left: EvidenceAsset, right: EvidenceAsset) {
  const orderDifference =
    (left.capture?.captureOrder ?? Number.MAX_SAFE_INTEGER) -
    (right.capture?.captureOrder ?? Number.MAX_SAFE_INTEGER);
  return orderDifference || left.discoveredAt.localeCompare(right.discoveredAt);
}

function isCaptureEligible(evidence: EvidenceAsset): boolean {
  return Boolean(
    evidence.localUrl &&
    evidence.mimeType?.startsWith("image/") &&
    !evidence.tags.includes("reconstruction-excluded"),
  );
}
