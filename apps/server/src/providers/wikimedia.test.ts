import { describe, expect, it, vi } from "vitest";
import { discoverWikimediaImages } from "./wikimedia.js";

describe("Wikimedia Commons discovery", () => {
  it("keeps modification-safe images and rejects no-derivatives licenses", async () => {
    const fetcher = vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            query: {
              pages: [
                {
                  title: "File:Reusable facade.jpg",
                  imageinfo: [
                    {
                      url: "https://upload.wikimedia.org/reusable.jpg",
                      thumburl:
                        "https://upload.wikimedia.org/reusable-1600.jpg",
                      descriptionurl:
                        "https://commons.wikimedia.org/wiki/File:Reusable_facade.jpg",
                      mime: "image/jpeg",
                      extmetadata: {
                        License: { value: "cc-by-sa-4.0" },
                        LicenseShortName: { value: "CC BY-SA 4.0" },
                        LicenseUrl: {
                          value:
                            "https://creativecommons.org/licenses/by-sa/4.0",
                        },
                        Artist: { value: "<a>Photographer</a>" },
                      },
                    },
                  ],
                },
                {
                  title: "File:No derivatives.jpg",
                  imageinfo: [
                    {
                      url: "https://upload.wikimedia.org/nd.jpg",
                      descriptionurl:
                        "https://commons.wikimedia.org/wiki/File:No_derivatives.jpg",
                      mime: "image/jpeg",
                      extmetadata: {
                        License: { value: "cc-by-nd-4.0" },
                        LicenseShortName: { value: "CC BY-ND 4.0" },
                      },
                    },
                  ],
                },
              ],
            },
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
    );

    const images = await discoverWikimediaImages(["Test Building"], fetcher);

    expect(images).toEqual([
      expect.objectContaining({
        title: "Reusable facade.jpg",
        creator: "Photographer",
        licenseCode: "by-sa",
      }),
    ]);
  });
});
