import { describe, expect, it } from "vitest";
import { addressTextMatch, unwrapBing } from "./browser.js";

describe("browser discovery URL handling", () => {
  it("unwraps Bing redirect URLs without following them", () => {
    const target = "https://example.com/property/photo";
    const encoded = Buffer.from(target).toString("base64url");
    expect(unwrapBing(`https://www.bing.com/ck/a?u=a1${encoded}`)).toBe(target);
  });

  it("requires the house number and street terms for a listing match", () => {
    expect(
      addressTextMatch(
        "123 Main Street, Seattle, WA 98101",
        "123 Main St, Seattle WA | Redfin",
      ),
    ).toBe(true);
    expect(
      addressTextMatch(
        "123 Main Street, Seattle, WA 98101",
        "125 Main St, Seattle WA | Zillow",
      ),
    ).toBe(false);
  });
});
