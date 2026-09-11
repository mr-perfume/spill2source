import React, { useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { PolygonLayer, ScatterplotLayer, PathLayer, TextLayer } from "@deck.gl/layers";
// Aliased deliberately: react-map-gl exports its component as `Map`, which
// would shadow the built-in Map constructor used below to bucket the
// posterior by release time.
import { Map as BaseMap } from "react-map-gl/maplibre";

const CARTO_DARK = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

/**
 * If CARTO is unreachable (offline demo, blocked network), MapLibre would
 * otherwise render nothing and the deck.gl layers would float over white.
 * This fallback style has no tile source at all -- just a flat ocean fill --
 * so the overlays stay readable without any outside request.
 */
const OFFLINE_STYLE = {
  version: 8,
  sources: {},
  layers: [{ id: "sea", type: "background", paint: { "background-color": "#04121c" } }],
};

const INITIAL_VIEW = {
  longitude: -89.12,
  latitude: 28.42,
  zoom: 8.4,
  pitch: 0,
  bearing: 0,
};

/* Colour ramp for the posterior: low mass reads as deep water, high mass
   glows toward the surface teal. Interpolated per point rather than binned,
   so the cloud shows a gradient rather than tiers. */
function posteriorColor(t, alpha) {
  const stops = [
    [11, 60, 92],
    [16, 122, 138],
    [53, 224, 210],
    [188, 255, 244],
  ];
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2);
  const f = x - i;
  const c = stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
  return [...c, alpha];
}

export default function SpillMap({
  detection,
  backtrack,
  vessels,
  forward,
  checkingVesselId,
  selectedVesselId,
  onSelectVessel,
}) {
  const [styleFailed, setStyleFailed] = useState(false);
  const [timeIndex, setTimeIndex] = useState(0);
  const [forwardProgress, setForwardProgress] = useState(1);
  const [viewState, setViewState] = useState(INITIAL_VIEW);
  const settledRef = useRef(false);
  // Remembers what the camera was last fitted to, so the effect below only
  // moves the camera when the thing it is framing actually changed.
  const lastFitRef = useRef("");

  /* ---------------------------------------------------------------------
   * Bucket the posterior by candidate release hour. The grid arrives coarse
   * binned already (config.grid_bin_deg = 0.05), so this is a few dozen
   * points at most and needs no further thinning.
   * ------------------------------------------------------------------ */
  const timeBuckets = useMemo(() => {
    const grid = backtrack?.origin_probability_grid || [];
    if (!grid.length) return [];
    const byTime = new Map();
    for (const cell of grid) {
      const key = cell.time_estimate;
      if (!byTime.has(key)) byTime.set(key, []);
      byTime.get(key).push(cell);
    }
    // Newest release time first: the sweep below reveals this array in
    // order, so it needs to start at the candidate closest to the observed
    // slick (the shortest drift back in time) and grow toward the oldest,
    // furthest candidate. That reads as the trace originating at the spill
    // and unwinding backward -- not a cloud growing in from somewhere else.
    return [...byTime.entries()]
      .sort((a, b) => new Date(b[0]) - new Date(a[0]))
      .map(([time, cells]) => ({ time, cells }));
  }, [backtrack]);

  const maxProbability = useMemo(() => {
    const grid = backtrack?.origin_probability_grid || [];
    return grid.reduce((m, c) => Math.max(m, c.probability), 0) || 1;
  }, [backtrack]);

  /* Sweep oldest to newest once, then hold the full cloud. The sweep is the
     one piece of non-interactive motion on the page: it is showing the
     model's reasoning over time, which is the thing a still image cannot. */
  useEffect(() => {
    if (!timeBuckets.length) return undefined;
    settledRef.current = false;
    setTimeIndex(0);
    let i = 0;
    const id = setInterval(() => {
      i += 1;
      if (i >= timeBuckets.length) {
        settledRef.current = true;
        setTimeIndex(timeBuckets.length);
        clearInterval(id);
      } else {
        setTimeIndex(i);
      }
    }, 260);
    return () => clearInterval(id);
  }, [timeBuckets]);

  /* Draw the forecast track on as it appears rather than snapping it in. */
  useEffect(() => {
    if (!forward?.track?.length) return undefined;
    setForwardProgress(0);
    let p = 0;
    const id = setInterval(() => {
      p += 0.04;
      if (p >= 1) {
        setForwardProgress(1);
        clearInterval(id);
      } else {
        setForwardProgress(p);
      }
    }, 40);
    return () => clearInterval(id);
  }, [forward]);

  /* Frame the scene on whatever the pipeline has produced so far. */
  useEffect(() => {
    const points = [];
    if (detection) points.push([detection.lon, detection.lat]);
    if (backtrack?.most_likely_origin) {
      points.push([backtrack.most_likely_origin.lon, backtrack.most_likely_origin.lat]);
    }
    for (const v of vessels || []) {
      if (v.closest_position) points.push([v.closest_position.lon, v.closest_position.lat]);
    }
    if (points.length < 2) return;

    const key = points.map((p) => p.map((n) => n.toFixed(4)).join()).join("|");
    if (key === lastFitRef.current) return;
    lastFitRef.current = key;

    const lons = points.map((p) => p[0]);
    const lats = points.map((p) => p[1]);
    const spanLon = Math.max(...lons) - Math.min(...lons);
    const spanLat = Math.max(...lats) - Math.min(...lats);
    const span = Math.max(spanLon, spanLat, 0.05);
    const zoom = Math.log2(360 / span) - 0.6;
    setViewState((v) => ({
      ...v,
      longitude: (Math.max(...lons) + Math.min(...lons)) / 2,
      latitude: (Math.max(...lats) + Math.min(...lats)) / 2,
      zoom: Number.isFinite(zoom) ? Math.max(6.5, Math.min(10.5, zoom)) : 8.4,
      transitionDuration: 900,
    }));
  }, [detection, backtrack, vessels]);

  const visibleCells = useMemo(() => {
    if (!timeBuckets.length) return [];
    const upto = settledRef.current || timeIndex >= timeBuckets.length ? timeBuckets.length : timeIndex + 1;
    const out = [];
    for (let i = 0; i < upto; i += 1) {
      const age = upto - i;
      for (const cell of timeBuckets[i].cells) {
        out.push({ ...cell, _age: age, _isLead: i === upto - 1 });
      }
    }
    return out;
  }, [timeBuckets, timeIndex]);

  const layers = [];

  /* -- the observed slick ------------------------------------------------ */
  if (detection?.polygon_geojson) {
    layers.push(
      new PolygonLayer({
        id: "spill-polygon",
        data: [{ polygon: detection.polygon_geojson.coordinates[0] }],
        getPolygon: (d) => d.polygon,
        filled: true,
        stroked: true,
        getFillColor: [255, 164, 61, 46],
        getLineColor: [255, 186, 106, 235],
        getLineWidth: 2,
        lineWidthUnits: "pixels",
        pickable: false,
      })
    );
  }

  /* -- posterior over candidate origins ---------------------------------- */
  if (visibleCells.length) {
    layers.push(
      new ScatterplotLayer({
        id: "origin-cloud",
        data: visibleCells,
        getPosition: (d) => [d.lon, d.lat],
        getRadius: (d) => 900 + 5200 * Math.sqrt(d.probability / maxProbability),
        radiusUnits: "meters",
        radiusMinPixels: 3,
        getFillColor: (d) => {
          const t = d.probability / maxProbability;
          const fade = d._isLead ? 210 : Math.max(70, 190 - d._age * 12);
          return posteriorColor(t, fade);
        },
        stroked: false,
        pickable: true,
        updateTriggers: { getFillColor: [timeIndex], getRadius: [maxProbability] },
      })
    );
  }

  /* -- uncertainty footprint and the point estimate ---------------------- */
  if (backtrack?.most_likely_origin && (settledRef.current || timeIndex >= timeBuckets.length)) {
    const o = backtrack.most_likely_origin;
    layers.push(
      new ScatterplotLayer({
        id: "uncertainty-ring",
        data: [o],
        getPosition: (d) => [d.lon, d.lat],
        getRadius: (backtrack.uncertainty_radius_km || 1) * 1000,
        radiusUnits: "meters",
        filled: true,
        stroked: true,
        getFillColor: [53, 224, 210, 20],
        getLineColor: [53, 224, 210, 150],
        getLineWidth: 1.5,
        lineWidthUnits: "pixels",
      }),
      new ScatterplotLayer({
        id: "origin-point",
        data: [o],
        getPosition: (d) => [d.lon, d.lat],
        getRadius: 5,
        radiusUnits: "pixels",
        getFillColor: [242, 251, 255, 255],
        stroked: true,
        getLineColor: [53, 224, 210, 255],
        lineWidthUnits: "pixels",
        getLineWidth: 2,
      })
    );
  }

  /* -- forecast track ----------------------------------------------------- */
  if (forward?.track?.length > 1) {
    const full = forward.track.map((p) => [p.lon, p.lat]);
    const cut = Math.max(2, Math.round(full.length * forwardProgress));
    const drawn = full.slice(0, cut);
    layers.push(
      new PathLayer({
        id: "forward-track",
        data: [{ path: drawn }],
        getPath: (d) => d.path,
        getColor: [255, 164, 61, 200],
        getWidth: 3,
        widthUnits: "pixels",
        capRounded: true,
        jointRounded: true,
      }),
      new ScatterplotLayer({
        id: "forward-head",
        data: [drawn[drawn.length - 1]],
        getPosition: (d) => d,
        getRadius: 5,
        radiusUnits: "pixels",
        getFillColor: [255, 196, 122, 240],
      })
    );
  }

  /* -- vessel tracks and markers ------------------------------------------ */
  if (vessels?.length) {
    layers.push(
      new PathLayer({
        id: "vessel-tracks",
        data: vessels,
        getPath: (d) => (d.path || []).map((p) => [p.lon, p.lat]),
        getColor: (d) =>
          d.ship_id === selectedVesselId
            ? [255, 90, 95, 190]
            : [78, 122, 140, d.in_release_window ? 120 : 55],
        getWidth: (d) => (d.ship_id === selectedVesselId ? 2.5 : 1.2),
        widthUnits: "pixels",
        pickable: false,
        updateTriggers: { getColor: [selectedVesselId], getWidth: [selectedVesselId] },
      }),
      new ScatterplotLayer({
        id: "vessel-markers",
        data: vessels.filter((v) => v.closest_position),
        getPosition: (d) => [d.closest_position.lon, d.closest_position.lat],
        getRadius: (d) => {
          if (d.ship_id === checkingVesselId) return 13;
          if (d.ship_id === selectedVesselId) return 10;
          return 6 + (d.suspicion_score / 100) * 4;
        },
        radiusUnits: "pixels",
        getFillColor: (d) => {
          if (d.ship_id === checkingVesselId) return [242, 251, 255, 255];
          const s = d.suspicion_score / 100;
          return [
            Math.round(78 + 177 * s),
            Math.round(122 - 32 * s),
            Math.round(140 - 45 * s),
            d.in_release_window ? 235 : 130,
          ];
        },
        stroked: true,
        getLineColor: (d) =>
          d.ship_id === checkingVesselId ? [53, 224, 210, 255] : [4, 18, 28, 200],
        getLineWidth: 1.5,
        lineWidthUnits: "pixels",
        pickable: true,
        onClick: (info) => info.object && onSelectVessel?.(info.object.ship_id),
        updateTriggers: {
          getRadius: [checkingVesselId, selectedVesselId],
          getFillColor: [checkingVesselId],
          getLineColor: [checkingVesselId],
        },
      }),
      new TextLayer({
        id: "vessel-labels",
        data: vessels.filter((v) => v.closest_position && v.suspicion_score >= 50).slice(0, 4),
        getPosition: (d) => [d.closest_position.lon, d.closest_position.lat],
        getText: (d) => d.name,
        getSize: 11,
        getColor: [207, 230, 240, 205],
        getPixelOffset: [0, -18],
        fontFamily: "'IBM Plex Mono', monospace",
        characterSet: "auto",
        background: true,
        getBackgroundColor: [4, 18, 28, 190],
        backgroundPadding: [5, 3, 5, 3],
      })
    );
  }

  return (
    <DeckGL
      viewState={viewState}
      onViewStateChange={({ viewState: vs }) =>
        // transitionDuration is dropped on the way back in. Without this the
        // interpolated state deck.gl reports mid-transition would restart the
        // transition on every frame, and never converge.
        setViewState({ ...vs, transitionDuration: 0 })
      }
      controller={{ dragRotate: false }}
      layers={layers}
      getTooltip={({ object, layer }) => {
        if (!object) return null;
        if (layer.id === "origin-cloud") {
          return {
            html: `<div style="font-family:'IBM Plex Mono',monospace;font-size:11px;line-height:1.5">
                     ${object.lat.toFixed(3)}, ${object.lon.toFixed(3)}<br/>
                     released ${new Date(object.time_estimate).toISOString().slice(11, 16)} UTC<br/>
                     ${(object.probability * 100).toFixed(1)}% of posterior mass
                   </div>`,
            style: {
              background: "#122733",
              color: "#e7f2f6",
              border: "1px solid #20404e",
              borderRadius: "6px",
              padding: "8px 10px",
              boxShadow: "0 8px 20px -10px rgba(0,0,0,0.6)",
            },
          };
        }
        if (layer.id === "vessel-markers") {
          return {
            html: `<div style="font-family:'IBM Plex Mono',monospace;font-size:11px;line-height:1.5">
                     ${object.name}<br/>score ${object.suspicion_score} &middot; ${object.closest_distance_km} km
                   </div>`,
            style: {
              background: "#122733",
              color: "#e7f2f6",
              border: "1px solid #20404e",
              borderRadius: "6px",
              padding: "8px 10px",
              boxShadow: "0 8px 20px -10px rgba(0,0,0,0.6)",
            },
          };
        }
        return null;
      }}
    >
      <BaseMap
        reuseMaps
        mapStyle={styleFailed ? OFFLINE_STYLE : CARTO_DARK}
        onError={() => setStyleFailed(true)}
        attributionControl={{ compact: true }}
      />
    </DeckGL>
  );
}
