import { useEffect, useRef } from "react";
import maplibregl, {
  type GeoJSONSource,
  type Map as MapLibreMap,
} from "maplibre-gl";
import type { Case, Polygon } from "@structurefirst/contracts";
import "maplibre-gl/dist/maplibre-gl.css";

type Props = {
  cases: Case[];
  activeCaseId?: string;
  compact?: boolean;
};

const mapStyle: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors",
      maxzoom: 19,
    },
  },
  layers: [
    {
      id: "osm",
      type: "raster",
      source: "osm",
      paint: { "raster-saturation": -0.65 },
    },
  ],
};

function featureCollection(cases: Case[]) {
  return {
    type: "FeatureCollection" as const,
    features: cases.flatMap((item) => {
      if (!item.profile) return [];
      return [
        {
          type: "Feature" as const,
          properties: { id: item.id, label: item.displayAddress },
          geometry: {
            type: "Point" as const,
            coordinates: [
              item.profile.location.longitude,
              item.profile.location.latitude,
            ],
          },
        },
      ];
    }),
  };
}

function footprintCollection(cases: Case[]) {
  return {
    type: "FeatureCollection" as const,
    features: cases.flatMap((item) =>
      item.profile?.footprint
        ? [
            {
              type: "Feature" as const,
              properties: { id: item.id },
              geometry: item.profile.footprint as Polygon,
            },
          ]
        : [],
    ),
  };
}

export function StructureMap({ cases, activeCaseId, compact = false }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const casesRef = useRef(cases);
  const activeCaseIdRef = useRef(activeCaseId);
  casesRef.current = cases;
  activeCaseIdRef.current = activeCaseId;

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const active =
      cases.find((item) => item.id === activeCaseId)?.profile ??
      cases.find((item) => item.profile)?.profile;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: mapStyle,
      center: active
        ? [active.location.longitude, active.location.latitude]
        : [-98.5, 39.8],
      zoom: active ? 18 : 3,
      attributionControl: false,
      maxPitch: 60,
    });
    map.addControl(
      new maplibregl.AttributionControl({ compact: true }),
      "bottom-right",
    );
    if (!compact)
      map.addControl(
        new maplibregl.NavigationControl({ visualizePitch: true }),
        "top-right",
      );
    map.on("load", () => {
      map.addSource("case-points", {
        type: "geojson",
        data: featureCollection(casesRef.current),
      });
      map.addLayer({
        id: "case-points-ring",
        type: "circle",
        source: "case-points",
        paint: {
          "circle-radius": compact ? 7 : 9,
          "circle-color": "#c33c44",
          "circle-stroke-color": "#ffffff",
          "circle-stroke-width": 3,
        },
      });
      map.addSource("footprints", {
        type: "geojson",
        data: footprintCollection(casesRef.current),
      });
      map.addLayer({
        id: "footprint-fill",
        type: "fill",
        source: "footprints",
        paint: { "fill-color": "#18384f", "fill-opacity": 0.2 },
      });
      map.addLayer({
        id: "footprint-line",
        type: "line",
        source: "footprints",
        paint: { "line-color": "#17384f", "line-width": 2 },
      });
      const latestActive = casesRef.current.find(
        (item) => item.id === activeCaseIdRef.current,
      )?.profile;
      if (latestActive) {
        map.jumpTo({
          center: [
            latestActive.location.longitude,
            latestActive.location.latitude,
          ],
          zoom: 18,
        });
      }
    });
    const observer = new ResizeObserver(() => map.resize());
    observer.observe(containerRef.current);
    mapRef.current = map;
    return () => {
      observer.disconnect();
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const update = () => {
      (map.getSource("case-points") as GeoJSONSource | undefined)?.setData(
        featureCollection(casesRef.current),
      );
      (map.getSource("footprints") as GeoJSONSource | undefined)?.setData(
        footprintCollection(casesRef.current),
      );
      const active = casesRef.current.find(
        (item) => item.id === activeCaseIdRef.current,
      )?.profile;
      if (active) {
        map.easeTo({
          center: [active.location.longitude, active.location.latitude],
          zoom: 18,
          duration: 450,
        });
      }
    };
    if (map.isStyleLoaded()) update();
    else map.once("load", update);
    return () => {
      map.off("load", update);
    };
  }, [activeCaseId, cases]);

  return (
    <div
      className="structure-map"
      ref={containerRef}
      aria-label="Building location map"
    />
  );
}
