import { describe, expect, it } from "vitest";
import { isAllowedNavigation, parseAgentAction } from "./browser-agent.js";

describe("browser agent action parser", () => {
  it("parses a plain JSON action", () => {
    expect(
      parseAgentAction(
        '{"name":"click","ref":"sf3","reasoning":"open the listing"}',
      ),
    ).toEqual({ name: "click", ref: "sf3" });
  });

  it("strips code fences", () => {
    const raw =
      '```json\n{"name":"type","ref":"sf1","text":"123 Main St","submit":true}\n```';
    expect(parseAgentAction(raw)).toEqual({
      name: "type",
      ref: "sf1",
      text: "123 Main St",
      submit: true,
    });
  });

  it("accepts type as a provider-compatible action discriminator", () => {
    expect(
      parseAgentAction(
        '{"type":"goto","url":"https://www.bing.com/search?q=address"}',
      ),
    ).toEqual({
      name: "goto",
      url: "https://www.bing.com/search?q=address",
    });
  });

  it("requires an imageUrl on collect_image", () => {
    expect(() =>
      parseAgentAction('{"name":"collect_image","why":"exterior"}'),
    ).toThrow(/imageUrl/);
  });

  it("accepts collect_image with sourceUrl", () => {
    expect(
      parseAgentAction(
        '{"name":"collect_image","imageUrl":"https://ex.com/a.jpg","sourceUrl":"https://ex.com/p","why":"front facade"}',
      ),
    ).toEqual({
      name: "collect_image",
      imageUrl: "https://ex.com/a.jpg",
      sourceUrl: "https://ex.com/p",
      why: "front facade",
    });
  });

  it("rejects unknown action names", () => {
    expect(() => parseAgentAction('{"name":"teleport"}')).toThrow(/Unknown/);
  });

  it("recovers from surrounding prose", () => {
    expect(
      parseAgentAction(
        'Sure, here is the next action: {"name":"scroll","direction":"down"} that should help.',
      ),
    ).toEqual({ name: "scroll", direction: "down" });
  });

  it("blocks automated Zillow, Redfin, Google imagery, and login navigation", () => {
    for (const url of [
      "https://www.zillow.com/homedetails/example",
      "https://www.redfin.com/WA/Seattle/example/home/123",
      "https://maps.google.com/example",
      "https://accounts.google.com/signin",
    ]) {
      expect(isAllowedNavigation(url)).toBe(false);
    }
  });

  it("allows search and non-blocked public source navigation", () => {
    expect(isAllowedNavigation("https://www.bing.com/search?q=address")).toBe(
      true,
    );
    expect(isAllowedNavigation("https://www.loc.gov/pictures/item/123")).toBe(
      true,
    );
  });
});
