import { describe, expect, it, vi } from "vitest";
import { discoverKartaViewImages } from "./kartaview.js";

describe("KartaView street imagery discovery", () => {
  it("returns an overlapping three-frame window from the best nearby sequence", async () => {
    const frame = (index: number) => ({
      id: `photo_${index}`,
      sequenceId: "sequence_123",
      sequenceIndex: String(index),
      lat: String(39.9998 + (index - 100) * 0.00001),
      lng: "-74",
      heading: "0",
      shotDate: "2025-01-02 03:04:05.000",
      status: "active",
      visibility: "public",
      imageProcUrl: `https://cdn.kartaview.org/proc/${index}.jpg`,
      imageThUrl: `https://cdn.kartaview.org/thumb/${index}.jpg`,
    });
    const fetcher = vi.fn(async (input: string | URL | Request) => {
      const url = new URL(String(input));
      const data = url.searchParams.has("sequenceId")
        ? [frame(96), frame(98), frame(100), frame(102), frame(104)]
        : [frame(100)];
      return new Response(
        JSON.stringify({
          status: { apiCode: 600, httpCode: 200 },
          result: { data },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });

    const images = await discoverKartaViewImages(40, -74, fetcher);

    expect(images.map((image) => image.sequenceIndex)).toEqual([98, 100, 102]);
    expect(images).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          sequenceId: "sequence_123",
          downloadUrl: "https://cdn.kartaview.org/proc/100.jpg",
          observedAt: "2025-01-02T03:04:05.000Z",
        }),
      ]),
    );
  });
});
