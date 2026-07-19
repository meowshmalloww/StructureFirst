import { describe, expect, it } from "vitest";
import { classifySource } from "./lib/source-policy.js";

describe("source policy", () => {
  it("hard-blocks YouTube media caching", () => {
    expect(
      classifySource("https://www.youtube.com/watch?v=example"),
    ).toMatchObject({
      provider: "YouTube",
      rights: "link_only",
      cachePolicy: "prohibited",
      hardBlocked: true,
      redistributable: false,
    });
  });

  it("hard-blocks Google imagery caching", () => {
    expect(classifySource("https://maps.google.com/example")).toMatchObject({
      provider: "Google",
      cachePolicy: "prohibited",
      hardBlocked: true,
    });
  });

  it("retains unknown public sources as metadata-only research links", () => {
    expect(classifySource("https://example.org/property/photo")).toMatchObject({
      rights: "research_unknown",
      cachePolicy: "metadata_only",
      hardBlocked: false,
      redistributable: false,
    });
  });

  it("recognizes OpenStreetMap's open-data policy", () => {
    expect(
      classifySource("https://www.openstreetmap.org/way/123"),
    ).toMatchObject({
      rights: "open_license",
      cachePolicy: "local_allowed",
      redistributable: true,
    });
  });

  it("keeps Zillow and Redfin listing results link-only", () => {
    for (const url of [
      "https://www.zillow.com/homedetails/example",
      "https://www.redfin.com/WA/Seattle/example/home/123",
    ]) {
      expect(classifySource(url)).toMatchObject({
        rights: "link_only",
        cachePolicy: "metadata_only",
        redistributable: false,
        hardBlocked: true,
      });
    }
  });
});
