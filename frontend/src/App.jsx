import React, { useCallback, useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import SpillMap from "./components/SpillMap.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import { ConfidenceBadge, PipelineRail, ScenePicker, VesselPanel } from "./components/Panels.jsx";
import { fetchHealth, fetchScenes, runForward, runPipeline } from "./api";

const IDLE_STAGES = { detection: "idle", backtrack: "idle", ais: "idle" };

/* Timestamps are shown as "MM-DD HH:MM". Dropping the date was hiding the
   most important fact about a wide release window: that its two ends are on
   different days. A window of 09-02 00:30 to 09-04 00:30 rendered as
   "00:30-00:30", which reads like a bug and undersells a real 48-hour
   uncertainty. */
const fmtStamp = (iso) => (iso ? iso.slice(5, 16).replace("T", " ") : "—");

const hoursBetween = (aIso, bIso) =>
  Math.abs(new Date(bIso) - new Date(aIso)) / 3600000;

const fmtHours = (h) => (h >= 10 ? `${Math.round(h)} h` : `${h.toFixed(1)} h`);

export default function App() {
  const [scenes, setScenes] = useState([]);
  const [selectedScene, setSelectedScene] = useState(null);
  const [health, setHealth] = useState(null);

  const [running, setRunning] = useState(false);
  const [stageState, setStageState] = useState(IDLE_STAGES);
  const [error, setError] = useState(null);

  const [detection, setDetection] = useState(null);
  const [backtrack, setBacktrack] = useState(null);
  const [vessels, setVessels] = useState(null);
  const [forward, setForward] = useState(null);
  const [forwardHours, setForwardHours] = useState(12);
  const [forwardBusy, setForwardBusy] = useState(false);

  const [checkingVesselId, setCheckingVesselId] = useState(null);
  const [selectedVesselId, setSelectedVesselId] = useState(null);
  const sweepTimers = useRef([]);

  /* -- catalog and service health ---------------------------------------- */
  useEffect(() => {
    fetchScenes()
      .then((list) => {
        setScenes(list);
        if (list.length) setSelectedScene(list[0].image_id);
      })
      .catch((e) => setError({ stage: "startup", message: e.message }));
    fetchHealth().then(setHealth);
  }, []);

  /* -- socket: reveal each layer as its stage lands ----------------------- */
  useEffect(() => {
    const socket = io({ path: "/socket.io", transports: ["websocket", "polling"] });

    socket.on("stage:started", ({ stage }) =>
      setStageState((s) => ({ ...s, [stage]: "running" }))
    );
    socket.on("stage:detection_done", ({ detection: d }) => {
      setDetection(d);
      setStageState((s) => ({ ...s, detection: "done" }));
    });
    socket.on("stage:backtrack_done", ({ backtrack: b }) => {
      setBacktrack(b);
      setStageState((s) => ({ ...s, backtrack: "done" }));
    });
    socket.on("stage:ais_done", ({ ranked_vessels }) => {
      setVessels(ranked_vessels);
      setStageState((s) => ({ ...s, ais: "done" }));
      startVesselSweep(ranked_vessels);
    });
    socket.on("stage:forward_done", ({ forward: f }) => setForward(f));
    socket.on("stage:error", ({ stage, message }) => {
      setError({ stage, message });
      setRunning(false);
    });

    return () => socket.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* -- the "checking each vessel" pass -----------------------------------
   * Purely presentational: the scores already exist by the time this runs.
   * It exists to make the correlation step legible rather than having seven
   * markers appear at once.
   * -------------------------------------------------------------------- */
  const startVesselSweep = useCallback((list) => {
    sweepTimers.current.forEach(clearTimeout);
    sweepTimers.current = [];
    if (!list?.length) return;

    const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reduceMotion) return;

    const order = [...list].sort((a, b) => a.suspicion_score - b.suspicion_score);
    order.forEach((v, i) => {
      sweepTimers.current.push(setTimeout(() => setCheckingVesselId(v.ship_id), i * 380));
    });
    sweepTimers.current.push(
      setTimeout(() => {
        setCheckingVesselId(null);
        setSelectedVesselId(list[0]?.ship_id ?? null);
      }, order.length * 380 + 500)
    );
  }, []);

  useEffect(() => () => sweepTimers.current.forEach(clearTimeout), []);

  /* -- run ---------------------------------------------------------------- */
  const handleRun = async () => {
    if (!selectedScene || running) return;
    sweepTimers.current.forEach(clearTimeout);
    setRunning(true);
    setError(null);
    setStageState(IDLE_STAGES);
    setDetection(null);
    setBacktrack(null);
    setVessels(null);
    setForward(null);
    setCheckingVesselId(null);
    setSelectedVesselId(null);

    try {
      const result = await runPipeline(selectedScene);
      // The socket has normally filled these already; assigning again keeps
      // the UI correct if a socket message was missed.
      setDetection(result.detection);
      setBacktrack(result.backtrack);
      setVessels(result.ranked_vessels);
      setStageState({ detection: "done", backtrack: "done", ais: "done" });
    } catch (e) {
      setError((prev) => prev || { stage: "pipeline", message: e.message });
    } finally {
      setRunning(false);
    }
  };

  const handleForecast = async () => {
    if (!backtrack?.most_likely_origin || forwardBusy) return;
    setForwardBusy(true);
    try {
      const f = await runForward({
        detection_id: detection?.detection_id,
        lat: backtrack.most_likely_origin.lat,
        lon: backtrack.most_likely_origin.lon,
        start_time_utc: backtrack.most_likely_origin.time,
        duration_hours: forwardHours,
        area_km2: detection?.area_km2,
      });
      setForward(f);
    } catch (e) {
      setError({ stage: "forward", message: e.message });
    } finally {
      setForwardBusy(false);
    }
  };

  const origin = backtrack?.most_likely_origin;

  return (
    <div className="flex h-full w-full bg-abyss">
      {/* ---------------- left rail ---------------- */}
      <aside className="flex w-[292px] shrink-0 flex-col border-r border-rule bg-hull/40">
        <header className="border-b border-rule px-4 py-3.5">
          <h1 className="text-[15px] font-semibold leading-tight text-chart">
            Spill origin &amp; vessel attribution
          </h1>
          <p className="mt-0.5 text-[11px] leading-snug text-haze">
            Radar detection, reverse drift modelling, and AIS correlation over the Mississippi
            Canyon lease blocks.
          </p>
        </header>

        <div className="scroll-thin flex-1 overflow-y-auto">
          <h2 className="px-4 pb-1.5 pt-3 text-[13px] font-medium text-chart">Sample scenes</h2>
          <ScenePicker
            scenes={scenes}
            selectedId={selectedScene}
            onSelect={setSelectedScene}
            disabled={running}
          />

          <div className="border-t border-rule pt-1">
            <PipelineRail stageState={stageState} error={error} />
          </div>

          {/* forecast control -- live now that forward mode exists */}
          <div className="border-t border-rule px-4 py-3">
            <h2 className="text-[13px] font-medium text-chart">Forecast forward</h2>
            <p className="mt-0.5 text-[11px] leading-snug text-haze">
              Advect the estimated release point forward to see where the oil goes next.
            </p>
            <div className="mt-2 flex gap-2">
              <select
                value={forwardHours}
                onChange={(e) => setForwardHours(Number(e.target.value))}
                disabled={!origin || forwardBusy}
                className="flex-1 rounded-sm border border-rule bg-abyss px-2 py-1.5 font-mono
                  text-[12px] text-chart focus:border-posterior focus:outline-none
                  disabled:opacity-40"
              >
                {[6, 12, 24, 48].map((h) => (
                  <option key={h} value={h}>
                    {h} hours
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={handleForecast}
                disabled={!origin || forwardBusy}
                className="rounded-sm border border-rule px-3 py-1.5 text-[12px] text-chart
                  transition hover:border-slick hover:text-slick focus:outline-none
                  focus-visible:ring-2 focus-visible:ring-posterior disabled:opacity-40
                  disabled:hover:border-rule disabled:hover:text-chart"
              >
                {forwardBusy ? "Running…" : "Forecast"}
              </button>
            </div>
            {forward && (
              <p className="mt-2 font-mono text-[10px] leading-snug text-haze">
                {forward.total_drift_km} km drift over {forward.duration_hours}h · footprint{" "}
                {forward.final_area_km2} km²
              </p>
            )}
          </div>
        </div>

        <div className="border-t border-rule px-4 py-3">
          <button
            type="button"
            onClick={handleRun}
            disabled={!selectedScene || running}
            className="w-full rounded-sm bg-posterior px-4 py-2.5 text-[13px] font-medium
              text-abyss transition hover:bg-posterior/85 focus:outline-none
              focus-visible:ring-2 focus-visible:ring-chart disabled:cursor-not-allowed
              disabled:bg-rule disabled:text-haze"
          >
            {running ? "Analysing scene…" : "Run analysis"}
          </button>
          {health && health.status !== "ok" && (
            <p className="mt-2 text-[10px] leading-snug text-alert">
              {health.services
                ? `Offline: ${health.services
                    .filter((s) => !s.reachable)
                    .map((s) => s.name)
                    .join(", ") || "database"}`
                : health.error}
            </p>
          )}
        </div>
      </aside>

      {/* ---------------- map ---------------- */}
      <main className="relative flex-1">
        <ErrorBoundary label="The map">
          <SpillMap
            detection={detection}
            backtrack={backtrack}
            vessels={vessels}
            forward={forward}
            checkingVesselId={checkingVesselId}
            selectedVesselId={selectedVesselId}
            onSelectVessel={setSelectedVesselId}
          />
        </ErrorBoundary>

        {!detection && !running && (
          <div className="pointer-events-none absolute left-1/2 top-1/2 w-[360px] -translate-x-1/2 -translate-y-1/2 text-center">
            <p className="text-[15px] text-chart/85">Pick a scene and run the analysis.</p>
            <p className="mt-1 text-[12px] leading-snug text-haze">
              The map fills in as each stage finishes: the detected slick, then the cloud of
              possible origins, then the vessels that were nearby.
            </p>
          </div>
        )}

        {error && (
          <div
            role="alert"
            className="absolute left-1/2 top-5 w-[420px] -translate-x-1/2 rounded-sm border
              border-alert/60 bg-hull px-3.5 py-2.5"
          >
            <div className="text-[12px] font-medium text-alert">
              {error.stage === "startup" ? "Cannot reach the gateway" : `The ${error.stage} step failed`}
            </div>
            <p className="mt-0.5 text-[11px] leading-snug text-chart/80">{error.message}</p>
          </div>
        )}
      </main>

      {/* ---------------- right panel ---------------- */}
      <aside className="scroll-thin flex w-[326px] shrink-0 flex-col overflow-y-auto border-l border-rule bg-hull/40">
        {detection && (
          <section className="border-b border-rule px-4 py-3">
            <h2 className="text-[13px] font-medium text-chart">Detected slick</h2>
            <dl className="mt-1.5 grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-[11px]">
              <dt className="text-haze">Centre</dt>
              <dd className="text-right text-chart">
                {detection.lat.toFixed(3)}, {detection.lon.toFixed(3)}
              </dd>
              <dt className="text-haze">Area</dt>
              <dd className="text-right text-chart">{detection.area_km2.toFixed(1)} km²</dd>
              <dt className="text-haze">Model confidence</dt>
              <dd className="text-right text-chart">
                {(detection.confidence * 100).toFixed(0)}%
              </dd>
              <dt className="text-haze">Imaged</dt>
              <dd className="text-right text-chart">{fmtStamp(detection.timestamp_utc)} UTC</dd>
            </dl>
          </section>
        )}

        {backtrack && (
          <section className="border-b border-rule py-3">
            <h2 className="px-4 text-[13px] font-medium text-chart">Estimated release</h2>
            <dl className="mt-1.5 grid grid-cols-2 gap-x-3 gap-y-1 px-4 font-mono text-[11px]">
              <dt className="text-haze">Most likely point</dt>
              <dd className="text-right text-chart">
                {origin.lat.toFixed(4)}, {origin.lon.toFixed(4)}
              </dd>

              <dt className="text-haze">Spread</dt>
              <dd className="text-right text-chart">
                ±{backtrack.uncertainty_radius_km.toFixed(1)} km
              </dd>

              <dt className="text-haze">Most likely time</dt>
              <dd className="text-right text-chart">{fmtStamp(origin.time)} UTC</dd>

              {detection && (
                <>
                  <dt className="text-haze">Before imaging</dt>
                  <dd className="text-right text-chart">
                    {fmtHours(hoursBetween(origin.time, detection.timestamp_utc))}
                  </dd>
                </>
              )}
            </dl>

            {/* The window gets its own block: on a wide posterior its two ends
                fall on different days, which a single cramped row hides. */}
            <div className="mx-4 mt-2 rounded-sm border border-rule px-2.5 py-2">
              <div className="flex items-baseline justify-between">
                <span className="text-[11px] text-haze">Release window</span>
                <span className="font-mono text-[10px] text-haze">
                  {fmtHours(
                    hoursBetween(
                      backtrack.release_time_window[0],
                      backtrack.release_time_window[1]
                    )
                  )}{" "}
                  span
                </span>
              </div>
              <div className="mt-1 font-mono text-[11px] leading-relaxed text-chart">
                <div>{fmtStamp(backtrack.release_time_window[0])} UTC</div>
                <div className="text-haze">to</div>
                <div>{fmtStamp(backtrack.release_time_window[1])} UTC</div>
              </div>
            </div>
            <div className="mt-2.5">
              <ErrorBoundary label="The confidence badge">
                <ConfidenceBadge backtrack={backtrack} detection={detection} />
              </ErrorBoundary>
            </div>
          </section>
        )}

        <ErrorBoundary label="The vessel list">
          <VesselPanel
            vessels={vessels}
            checkingVesselId={checkingVesselId}
            selectedVesselId={selectedVesselId}
            onSelect={setSelectedVesselId}
          />
        </ErrorBoundary>

        {!detection && (
          <div className="px-4 py-5 text-[11px] leading-snug text-haze">
            Results appear here once a scene has been analysed.
          </div>
        )}
      </aside>
    </div>
  );
}
