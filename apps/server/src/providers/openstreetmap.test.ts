import { afterEach, describe, expect, it, vi } from "vitest";
import { loadConfig } from "../config.js";
import { findNearestBuilding } from "./openstreetmap.js";

afterEach(() => vi.unstubAllGlobals());

describe("OpenStreetMap building selection", () => {
  it("prefers an address-matched footprint over a nearby unrelated building", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              elements: [
                {
                  type: "way",
                  id: 265260946,
                  tags: { building: "yes", "building:levels": "6" },
                  geometry: [
                    { lat: 40.747567, lon: -73.984981 },
                    { lat: 40.747634, lon: -73.9849325 },
                    { lat: 40.7475404, lon: -73.9847283 },
                    { lat: 40.7474804, lon: -73.9847729 },
                    { lat: 40.747567, lon: -73.984981 },
                  ],
                },
                {
                  type: "way",
                  id: 34633854,
                  tags: {
                    building: "office",
                    name: "Empire State Building",
                    "addr:housenumber": "350",
                    "addr:street": "5th Avenue",
                    "addr:postcode": "10118",
                    "building:levels": "102",
                  },
                  geometry: [
                    { lat: 40.748491, lon: -73.9865012 },
                    { lat: 40.7479255, lon: -73.9851602 },
                    { lat: 40.7483931, lon: -73.9848166 },
                    { lat: 40.7489585, lon: -73.9861574 },
                    { lat: 40.748491, lon: -73.9865012 },
                  ],
                },
              ],
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
      ),
    );

    const building = await findNearestBuilding(
      40.747848600317,
      -73.985077152891,
      loadConfig(),
      "350 Fifth Avenue, New York, NY 10118",
    );

    expect(building).toMatchObject({
      osmId: 34633854,
      levels: 102,
      buildingType: "office",
      tags: { name: "Empire State Building" },
    });
  });
});
