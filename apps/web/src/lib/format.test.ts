import { afterEach, describe, expect, it, vi } from "vitest";
import { formatRelative, sentence } from "./format";

afterEach(() => vi.useRealTimers());

describe("operator-facing formatting", () => {
  it("turns contract enum values into plain labels", () => {
    expect(sentence("law_enforcement")).toBe("Law enforcement");
    expect(sentence("search_rescue")).toBe("Search rescue");
  });

  it("formats recent case updates predictably", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-16T18:00:00.000Z"));
    expect(formatRelative("2026-07-16T17:45:00.000Z")).toBe("15 min ago");
    expect(formatRelative("2026-07-15T18:00:00.000Z")).toBe("1 d ago");
  });
});
