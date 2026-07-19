import { afterEach, describe, expect, it, vi } from "vitest";
import { discoverOpenverseImages } from "./openverse.js";

afterEach(() => vi.unstubAllGlobals());

describe("Openverse property search", () => {
  it("requests modification-safe licenses and rejects ND results defensively", async () => {
    let requestedUrl = "";
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request) => {
        requestedUrl = String(input);
        return new Response(
          JSON.stringify({
            results: [
              {
                title: "Reusable building",
                url: "https://media.example/reusable.jpg",
                foreign_landing_url: "https://source.example/reusable",
                license: "by",
                license_version: "4.0",
                source: "wikimedia",
              },
              {
                title: "No derivatives",
                url: "https://media.example/nd.jpg",
                foreign_landing_url: "https://source.example/nd",
                license: "by-nd",
                license_version: "4.0",
                source: "wikimedia",
              },
            ],
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }),
    );

    const images = await discoverOpenverseImages("100 Test Avenue");
    const url = new URL(requestedUrl);

    expect(url.searchParams.get("license")).toBe("pdm,cc0,by,by-sa");
    expect(url.searchParams.get("license_type")).toBe("modification");
    expect(images).toEqual([
      expect.objectContaining({
        title: "Reusable building",
        licenseCode: "by",
      }),
    ]);
  });
});
