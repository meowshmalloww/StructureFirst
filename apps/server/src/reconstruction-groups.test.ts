import { describe, expect, it } from "vitest";
import {
  coverageFrameGroups,
  nextVerifiedFrameGroup,
  verifiedFrameGroups,
} from "./reconstruction.js";

describe("disconnected multi-room reconstruction groups", () => {
  it("selects the strongest internally connected group and ignores singletons", () => {
    expect(
      nextVerifiedFrameGroup({
        disconnectedFrames: [3, 4, 5, 6, 7],
        preflight: {
          acceptedPairs: [
            { frameA: 3, frameB: 4, confidence: 0.72 },
            { frameA: 4, frameB: 5, confidence: 0.64 },
            { frameA: 6, frameB: 7, confidence: 0.95 },
          ],
        },
      }),
    ).toEqual([3, 4, 5]);
  });

  it("does not reconstruct unrelated disconnected images", () => {
    expect(
      nextVerifiedFrameGroup({
        disconnectedFrames: [2, 3, 4],
        preflight: { acceptedPairs: [] },
      }),
    ).toBeUndefined();
  });

  it("returns every independently verified room group in stable order", () => {
    expect(
      verifiedFrameGroups({
        disconnectedFrames: [2, 3, 4, 5, 6, 7, 8],
        preflight: {
          acceptedPairs: [
            { frameA: 2, frameB: 3, confidence: 0.6 },
            { frameA: 3, frameB: 4, confidence: 0.7 },
            { frameA: 5, frameB: 6, confidence: 0.95 },
            { frameA: 7, frameB: 8, confidence: 0.8 },
          ],
        },
      }),
    ).toEqual([
      [2, 3, 4],
      [5, 6],
      [7, 8],
    ]);
  });

  it("preserves unconnected photos as automatic singleton scenes", () => {
    expect(
      coverageFrameGroups(
        {
          disconnectedFrames: [2, 3, 4, 5],
          preflight: {
            acceptedPairs: [{ frameA: 2, frameB: 3, confidence: 0.8 }],
          },
        },
        6,
        false,
      ),
    ).toEqual([[2, 3], [4], [5]]);
  });

  it("recovers every input after a failed group instead of dropping its core", () => {
    expect(
      coverageFrameGroups(
        {
          disconnectedFrames: [1, 2, 3],
          preflight: { acceptedPairs: [] },
        },
        4,
        true,
      ),
    ).toEqual([[0], [1], [2], [3]]);
  });
});
