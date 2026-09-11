import React from "react";
import { sceneImageUrl } from "../api";

/* ---------------------------------------------------------------------------
 * Scene picker
 * ------------------------------------------------------------------------ */
export function ScenePicker({ scenes, selectedId, onSelect, disabled }) {
  if (!scenes.length) {
    return (
      <div className="px-4 py-6 text-sm text-haze">
        No scenes in the catalog yet. Build it with{" "}
        <code className="font-mono text-chart">python scripts/make_demo_catalog.py</code>, then load
        it with <code className="font-mono text-chart">python scripts/seed_db.py</code>.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-2 px-3 pb-3">
      {scenes.map((scene) => {
        const active = scene.image_id === selectedId;
        return (
          <button
            key={scene.image_id}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(scene.image_id)}
            className={`group relative overflow-hidden rounded-sm border text-left transition
              focus:outline-none focus-visible:ring-2 focus-visible:ring-posterior
              ${active ? "border-posterior" : "border-rule hover:border-haze"}
              ${disabled ? "cursor-not-allowed opacity-50" : ""}`}
          >
            <img
              src={sceneImageUrl(scene.image_id)}
              alt={`SAR tile, ${scene.title}`}
              className={`h-20 w-full object-cover transition ${
                active ? "opacity-100" : "opacity-65 group-hover:opacity-90"
              }`}
            />
            <div className="px-2 py-1.5">
              <div className="text-[12px] font-medium leading-tight text-chart">{scene.title}</div>
              <div className="font-mono text-[10px] leading-tight text-haze">
                {scene.acquisition_timestamp?.slice(11, 16)} UTC ·{" "}
                {scene.bbox?.top_left?.lat.toFixed(2)}N
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}

/* ---------------------------------------------------------------------------
 * Pipeline progress
 *
 * Numbered because this genuinely is a sequence: each stage consumes the
 * previous stage's output and cannot start early.
 * ------------------------------------------------------------------------ */
const STAGES = [
  { key: "detection", label: "Detect the slick", detail: "U-Net segmentation" },
  { key: "backtrack", label: "Trace it back", detail: "Drift ensemble" },
  { key: "ais", label: "Match vessels", detail: "AIS correlation" },
];

export function PipelineRail({ stageState, error }) {
  return (
    <ol className="space-y-0">
      {STAGES.map((stage, i) => {
        const state = stageState[stage.key] || "idle";
        const failed = error?.stage === stage.key;
        return (
          <li key={stage.key} className="relative flex gap-3 px-4 py-2.5">
            {i < STAGES.length - 1 && (
              <span
                aria-hidden
                className={`absolute left-[26px] top-8 h-[calc(100%-14px)] w-px ${
                  state === "done" ? "bg-posterior/45" : "bg-rule"
                }`}
              />
            )}
            <span
              className={`relative z-10 mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center
                rounded-full border font-mono text-[10px]
                ${
                  failed
                    ? "border-alert bg-alert/15 text-alert"
                    : state === "done"
                    ? "border-posterior bg-posterior/15 text-posterior"
                    : state === "running"
                    ? "border-posterior/60 bg-abyss text-posterior"
                    : "border-rule bg-abyss text-haze"
                }`}
            >
              {failed ? "!" : state === "done" ? "✓" : i + 1}
            </span>
            <div className="min-w-0">
              <div
                className={`text-[13px] leading-tight ${
                  state === "idle" ? "text-haze" : "text-chart"
                }`}
              >
                {stage.label}
              </div>
              <div className="font-mono text-[10px] leading-tight text-haze">
                {failed ? error.message : state === "running" ? "working…" : stage.detail}
              </div>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

/* ---------------------------------------------------------------------------
 * Confidence badge
 *
 * Driven by the drift model's own diagnostics. The three things that make a
 * posterior untrustworthy each get their own message, because "low
 * confidence" without a reason is not actionable.
 * ------------------------------------------------------------------------ */
export function ConfidenceBadge({ backtrack, detection }) {
  if (!backtrack?._debug) return null;
  const d = backtrack._debug;
  const ess = d.effective_sample_size ?? 0;
  const n = d.n_ensemble_members || 1;
  const ratio = ess / n;

  // How much of the searched time range the credible window still covers. A
  // posterior can pin the location well while leaving the release hour almost
  // wide open, and saying only "well constrained" in that case oversells it.
  const win = backtrack.release_time_window || [];
  const windowHours =
    win.length === 2 ? Math.abs(new Date(win[1]) - new Date(win[0])) / 3600000 : null;
  const searchedHours = d.hour_range ? d.hour_range[1] - d.hour_range[0] : null;
  const timeCoverage =
    windowHours && searchedHours ? windowHours / searchedHours : null;

  let tone = "good";
  let title = "Origin well constrained";
  let detail = `${ess.toFixed(1)} of ${n} drift hypotheses carry the posterior.`;

  if (d.uses_synthetic_placeholder_current) {
    tone = "bad";
    title = "Not physically meaningful";
    detail =
      "The ocean current API was unreachable, so this ran on a synthetic placeholder field. Treat the origin as a demonstration of the pipeline, not a result.";
  } else if (d.degenerate_ensemble) {
    tone = "bad";
    title = "Ensemble collapsed";
    detail =
      "Every candidate scored near zero, so the posterior fell back to uniform. The search window probably does not bracket the real release.";
  } else if (ratio < 0.15) {
    tone = "warn";
    title = "Origin uncertain";
    detail = `Almost all posterior mass sits on ${Math.max(1, Math.round(ess))} hypothesis. Read the uncertainty radius with caution.`;
  } else if (d.current_spatial_speed_range_mps !== undefined && d.current_spatial_speed_range_mps < 0.02) {
    tone = "warn";
    title = "Weak current gradient";
    detail =
      "The current field is nearly uniform across the search area, so release times are hard to tell apart. The origin is better constrained in space than in time.";
  } else if (timeCoverage !== null && timeCoverage > 0.55) {
    tone = "warn";
    title = "Location narrowed, timing open";
    detail =
      `The position estimate holds, but the credible window still covers ` +
      `${Math.round(timeCoverage * 100)}% of the ${Math.round(searchedHours)}-hour search range. ` +
      `Treat the release hour as unresolved rather than estimated.`;
  }

  const styles = {
    good: "border-posterior/45 bg-posterior/10 text-posterior",
    warn: "border-slick/50 bg-slick/10 text-slick",
    bad: "border-alert/50 bg-alert/10 text-alert",
  }[tone];

  return (
    <div className={`mx-4 rounded-sm border px-3 py-2 ${styles}`}>
      <div className="text-[12px] font-medium">{title}</div>
      <p className="mt-0.5 text-[11px] leading-snug text-chart/80">{detail}</p>
      <div className="mt-1.5 font-mono text-[10px] leading-relaxed text-chart/55">
        <div>
          current: {d.current_source}
          {d.current_grid_side ? ` ${d.current_grid_side}×${d.current_grid_side}` : ""}
          {d.runtime_seconds !== undefined ? ` · ${d.runtime_seconds}s` : ""}
        </div>
        {timeCoverage !== null && (
          <div>
            timing: window covers {Math.round(timeCoverage * 100)}% of{" "}
            {Math.round(searchedHours)} h searched
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------------------
 * Ranked vessels
 * ------------------------------------------------------------------------ */
export function VesselPanel({ vessels, checkingVesselId, selectedVesselId, onSelect }) {
  if (!vessels?.length) return null;

  return (
    <div>
      <div className="flex items-baseline justify-between px-4 pb-1.5 pt-3">
        <h2 className="text-[13px] font-medium text-chart">Vessels near the estimated origin</h2>
        <span className="font-mono text-[10px] text-haze">{vessels.length} checked</span>
      </div>
      <p className="px-4 pb-2 text-[11px] leading-snug text-haze">
        Ranked by how much of the drift posterior sits near each vessel's reported position during
        the release window. These are leads to investigate, not findings.
      </p>
      <ul className="px-3 pb-4">
        {vessels.map((v) => {
          const isChecking = v.ship_id === checkingVesselId;
          const isOpen = v.ship_id === selectedVesselId;
          return (
            <li key={v.ship_id}>
              <button
                type="button"
                onClick={() => onSelect(isOpen ? null : v.ship_id)}
                className={`w-full rounded-sm border px-3 py-2 text-left transition
                  focus:outline-none focus-visible:ring-2 focus-visible:ring-posterior
                  ${isChecking ? "border-posterior bg-posterior/10" : ""}
                  ${isOpen && !isChecking ? "border-haze bg-hull" : ""}
                  ${!isChecking && !isOpen ? "border-transparent hover:bg-hull/70" : ""}`}
              >
                <div className="flex items-center gap-3">
                  <span
                    aria-hidden
                    className="h-8 w-1 shrink-0 rounded-full"
                    style={{
                      background: `rgb(${78 + 177 * (v.suspicion_score / 100)}, ${
                        122 - 32 * (v.suspicion_score / 100)
                      }, ${140 - 45 * (v.suspicion_score / 100)})`,
                      opacity: v.in_release_window ? 1 : 0.4,
                    }}
                  />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[13px] leading-tight text-chart">
                      {v.name}
                    </span>
                    <span className="block font-mono text-[10px] leading-tight text-haze">
                      {v.type} · {v.closest_distance_km} km
                      {!v.in_release_window && " · outside window"}
                    </span>
                  </span>
                  <span className="shrink-0 text-right">
                    <span className="block font-mono text-[15px] leading-none text-chart">
                      {v.suspicion_score.toFixed(0)}
                    </span>
                    <span className="block font-mono text-[9px] leading-tight text-haze">score</span>
                  </span>
                </div>

                {isOpen && (
                  <dl className="mt-2.5 grid grid-cols-2 gap-x-3 gap-y-1 border-t border-rule pt-2 font-mono text-[10px]">
                    <dt className="text-haze">Vessel ID</dt>
                    <dd className="text-right text-chart">{v.ship_id}</dd>
                    <dt className="text-haze">Nearest fix</dt>
                    <dd className="text-right text-chart">
                      {v.closest_position.lat.toFixed(3)}, {v.closest_position.lon.toFixed(3)}
                    </dd>
                    <dt className="text-haze">Fix time</dt>
                    <dd className="text-right text-chart">
                      {v.closest_timestamp?.slice(5, 16).replace("T", " ")} UTC
                    </dd>
                    <dt className="text-haze">Fixes in window</dt>
                    <dd className="text-right text-chart">{v.positions_in_window}</dd>
                    <dt className="text-haze">Posterior overlap</dt>
                    <dd className="text-right text-chart">
                      {(v._scoring.normalised_proximity * 100).toFixed(0)}%
                    </dd>
                    <dt className="text-haze">Timing centrality</dt>
                    <dd className="text-right text-chart">
                      {(v._scoring.time_centrality * 100).toFixed(0)}%
                    </dd>
                  </dl>
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
