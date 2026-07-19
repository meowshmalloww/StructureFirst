import { describe, expect, it } from "vitest";
import { nextVerifiedFrameGroup } from "./reconstruction.js";

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
});
