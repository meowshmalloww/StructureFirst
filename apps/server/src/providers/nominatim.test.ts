import { afterEach, describe, expect, it, vi } from "vitest";
import { loadConfig } from "../config.js";
import { geocodeAddress, scoreNominatimResult } from "./nominatim.js";

afterEach(() => vi.unstubAllGlobals());

describe("address resolution", () => {
  it("rejects a fuzzy first result in the wrong postal code and city", () => {
    const score = scoreNominatimResult("350 Fifth Avenue, New York, NY 10118", {
      lat: "40.9161016",
      lon: "-73.8071811",
      display_name:
        "350, 5th Avenue, North Pelham, Westchester County, New York, 10803, United States",
      address: {
        house_number: "350",
        road: "5th Avenue",
        town: "Village of Pelham",
        state: "New York",
        postcode: "10803",
        country_code: "us",
      },
    });

    expect(score).toBe(0);
  });

  it("uses an exact Census structure-address match for U.S. addresses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              result: {
                addressMatches: [
                  {
                    matchedAddress: "350 5TH AVE, NEW YORK, NY, 10118",
                    coordinates: { x: -73.985077, y: 40.747849 },
                    addressComponents: {
                      fromAddress: "350",
                      streetName: "5TH",
                      suffixType: "AVE",
                      city: "NEW YORK",
                      state: "NY",
                      zip: "10118",
                    },
                  },
                ],
              },
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
      ),
    );

    const match = await geocodeAddress(
      "350 Fifth Avenue, New York, NY 10118",
      loadConfig(),
    );

    expect(match).toMatchObject({
      displayAddress: "350 5th Ave, New York, NY, 10118",
      latitude: 40.747849,
      longitude: -73.985077,
      provider: "U.S. Census Geocoder",
      matchMethod: "census_exact",
    });
  });

  it("prefers a mapped POI for a named building followed by its address", async () => {
    const fetchMock = vi.fn(async (input: string | URL | Request) => {
      const url = new URL(String(input));
      expect(url.hostname).toBe("nominatim.openstreetmap.org");
      expect(url.searchParams.get("layer")).toBe("address,poi");
      return new Response(
        JSON.stringify([
          {
            lat: "37.3182932",
            lon: "-121.9509886",
            display_name:
              "Winchester Mystery House, 525, Winchester Boulevard, San Jose, California, 95128, United States",
            osm_type: "way",
            osm_id: 12345,
            importance: 0.52,
            address: {
              house_number: "525",
              road: "Winchester Boulevard",
              city: "San Jose",
              state: "California",
              postcode: "90214",
              country_code: "us",
            },
          },
        ]),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const match = await geocodeAddress(
      "Winchester Mystery House, 525 S Winchester Blvd, San Jose, CA 95128",
      loadConfig(),
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(match).toMatchObject({
      latitude: 37.3182932,
      longitude: -121.9509886,
      provider: "OpenStreetMap Nominatim",
      matchMethod: "nominatim_ranked",
    });
  });

  it("removes the place-name prefix before Census address fallback", async () => {
    const fetchMock = vi.fn(async (input: string | URL | Request) => {
      const url = new URL(String(input));
      if (url.hostname === "nominatim.openstreetmap.org")
        return new Response("[]", {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      expect(url.searchParams.get("address")).toBe(
        "2007 Franklin Street, San Francisco, CA 94109",
      );
      return new Response(
        JSON.stringify({
          result: {
            addressMatches: [
              {
                matchedAddress: "2007 FRANKLIN ST, SAN FRANCISCO, CA, 94109",
                coordinates: { x: -122.4249, y: 37.7932 },
                addressComponents: {
                  fromAddress: "2007",
                  streetName: "FRANKLIN",
                  suffixType: "ST",
                  city: "SAN FRANCISCO",
                  state: "CA",
                  zip: "94109",
                },
              },
            ],
          },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const match = await geocodeAddress(
      "Haas-Lilienthal House, 2007 Franklin Street, San Francisco, CA 94109",
      loadConfig(),
    );

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(match?.provider).toBe("U.S. Census Geocoder");
  });
});
