import type { Confidence } from "@structurefirst/contracts";
import { nowIso } from "./ids.js";

export function confidence(
  score: number,
  band: Confidence["band"],
  state: Confidence["state"],
  rationale: string,
  sourceCount: number,
): Confidence {
  return {
    score,
    band,
    state,
    rationale,
    sourceCount,
    updatedAt: nowIso(),
  };
}
