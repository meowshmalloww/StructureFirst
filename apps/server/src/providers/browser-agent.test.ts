import { describe, expect, it } from "vitest";
import { parseAgentAction } from "./browser-agent.js";

describe("browser agent action parser", () => {
  it("parses a plain JSON action", () => {
    expect(
      parseAgentAction(
        '{"name":"click","ref":"sf3","reasoning":"open the listing"}',
      ),
    ).toEqual({ name: "click", ref: "sf3" });
  });

  it("strips code fences", () => {
    const raw = "```json\n{\"name\":\"type\",\"ref\":\"sf1\",\"text\":\"123 Main St\",\"submit\":true}\n```";
    expect(parseAgentAction(raw)).toEqual({
      name: "type",
      ref: "sf1",
      text: "123 Main St",
      submit: true,
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
});
