"""
run_module2.py
End-to-end Module 2 CLI: Module 1's detections.json -> backtrack_results.json,
one ABC-ensemble origin/time posterior per detection.

Usage:
    python run_module2.py --detections ../module1_detection/detections.json \
        --output backtrack_results.json

    # dry run against a synthetic detection + synthetic current/wind fields,
    # no real data or Module 1 output needed:
    python run_module2.py --self-test

Expected input: a JSON file shaped like Module 1's run_module1.py output --
{"scene": "...", "detections": [ {lat, lon, polygon_geojson, timestamp_utc,
area_km2, ...}, ... ]} -- i.e. run this directly on run_module1.py's output
file, unmodified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
# Make sure this runs correctly whether launched from inside
# module2_backtracking/ (the normal way) or from the project root --
# same fix postprocessing.py applies for shared_apis.* imports.
for p in (MODULE_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import BacktrackConfig, DEFAULT_CONFIG  # noqa: E402
from backtrack import parse_iso_to_epoch, run_ensemble_backtrack  # noqa: E402

try:
    from shared_apis.drift_forcing_client import get_ocean_current_provider, get_wind_vector_provider
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "Couldn't import shared_apis.drift_forcing_client -- make sure "
        "shared_apis/ sits at the project root (a sibling of "
        "module1_detection/ and module2_backtracking/), same layout Module "
        "1's postprocessing.py already assumes for shared_apis.wind_client."
    ) from _e


def _polygon_bbox_padded(polygon_geojson: dict, pad_deg: float) -> tuple[float, float, float, float]:
    from shapely.geometry import shape as shapely_shape
    minx, miny, maxx, maxy = shapely_shape(polygon_geojson).bounds
    return (minx - pad_deg, miny - pad_deg, maxx + pad_deg, maxy + pad_deg)


def run_module2_for_detection(
    detection: dict,
    cfg: BacktrackConfig = DEFAULT_CONFIG,
) -> dict:
    """Runs the ABC ensemble backtrack for one Module-1 detection dict and
    returns a result matching the spec's output schema (plus a _debug
    block, plus the originating detection_id for traceability)."""
    from backtrack import parse_iso_to_epoch  # local import avoids a cycle at module load time

    bbox = _polygon_bbox_padded(detection["polygon_geojson"], cfg.bbox_padding_deg)
    t_detect = parse_iso_to_epoch(detection["timestamp_utc"])
    t_start = t_detect - cfg.max_hours_back * 3600.0

    current_provider = get_ocean_current_provider(
        bbox, t_start, t_detect, cache_path=cfg.current_cache_path,
        use_openmeteo=cfg.use_openmeteo, openmeteo_api_key=cfg.openmeteo_api_key,
    )
    wind_provider = get_wind_vector_provider(
        bbox, t_start, t_detect, cache_path=cfg.wind_cache_path,
        u_mps=cfg.wind_u_mps, v_mps=cfg.wind_v_mps,
    )
    # "synthetic placeholder" (current) and "negligible constant" (wind) are
    # tracked separately: the first means "we wanted real CMEMS data and
    # don't have it yet" (a degraded substitute); the second is this
    # pipeline's INTENDED default (wind treated as negligible on purpose),
    # not something to warn about.
    uses_synthetic_current = getattr(current_provider, "is_synthetic_placeholder", False)
    wind_is_negligible = getattr(wind_provider, "is_negligible_constant", False)

    result = run_ensemble_backtrack(
        centroid_lat=detection["lat"],
        centroid_lon=detection["lon"],
        polygon_geojson=detection["polygon_geojson"],
        detection_timestamp_utc=detection["timestamp_utc"],
        current_provider=current_provider,
        wind_provider=wind_provider,
        obs_area_km2=detection.get("area_km2"),
        cfg=cfg,
    )
    out = result.to_schema_dict()
    out["detection_id"] = detection.get("detection_id")
    # Loud, hard-to-miss, IN THE OUTPUT (not just a stderr warning that's
    # easy to lose in a demo terminal) flag for "this posterior used a
    # placeholder CURRENT field, not real CMEMS data" -- same philosophy as
    # Module 1's land_sea_mask warning, just made impossible to
    # accidentally ship past without noticing. Wind's negligible-by-design
    # status is reported alongside it for transparency, but isn't a "this
    # run is fake" flag the way the current one is.
    out["_debug"]["uses_synthetic_placeholder_current"] = uses_synthetic_current
    out["_debug"]["wind_treated_as_negligible"] = wind_is_negligible
    out["_debug"]["current_source"] = type(current_provider).__name__  # NetCDFGridField /
                                                                        # OpenMeteoCurrentField /
                                                                        # SyntheticGyreField
    out["_debug"]["openmeteo_commercial_key_used"] = getattr(current_provider, "used_commercial_key", False)
    return out


def run_module2(
    detections_path: Path,
    output_path: Path,
    cfg: BacktrackConfig = DEFAULT_CONFIG,
    max_detections: int | None = None,
) -> list[dict]:
    with open(detections_path) as f:
        payload = json.load(f)
    detections = payload["detections"]
    if max_detections is not None:
        detections = detections[:max_detections]

    print(f"[module2] {len(detections)} detection(s) to backtrack")
    results = []
    for i, det in enumerate(detections, start=1):
        print(f"[module2] ({i}/{len(detections)}) detection_id={det.get('detection_id')}")
        results.append(run_module2_for_detection(det, cfg))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"scene": payload.get("scene"), "backtrack_results": results}, f, indent=2)
    print(f"[done] {len(results)} backtrack result(s) written to {output_path}")
    return results


# ---------------------------------------------------------------------------
# Self-test: synthetic detection (a real, slightly-elongated polygon) +
# synthetic current/wind fields with a KNOWN true origin, so we can check
# the ensemble actually recovers something in the right neighborhood rather
# than just "runs without crashing".
# ---------------------------------------------------------------------------
def _self_test():
    import numpy as np
    from shapely.geometry import mapping as shapely_mapping
    from advection import (
        integrate_particles, seed_particles_in_circle, polygon_area_km2, convex_hull_polygon,
        geodesic_distance_km,
    )
    from shared_apis.drift_forcing_client import SyntheticGyreField, ConstantVectorField

    print("=== run_module2.py self-test (synthetic scene, known true origin, NEGLIGIBLE wind) ===")

    true_origin_lat, true_origin_lon = 22.40, 68.95   # Gulf-of-Kutch-ish demo coordinate
    true_release_hours_ago = 14.0
    true_alpha = 0.03  # irrelevant in practice since wind=0 below, kept for signature clarity

    detect_dt = "2026-08-20T06:00:00Z"
    t_detect = parse_iso_to_epoch(detect_dt)
    t_release = t_detect - true_release_hours_ago * 3600.0

    # This is now the pipeline's actual default posture: wind is negligible
    # (ConstantVectorField(0, 0) -- no API, no grid, not even a rotating
    # placeholder), and ocean CURRENT is the sole real forcing. bg_u/bg_v on
    # the current field gives real net translation over the backtrack
    # window; strain_rate_per_s gives a real, continuous, incompressible
    # shear (see advection.polygon_elongation_ratio's docstring) so the
    # forward re-simulation's ELONGATION -- not just its area, which pure
    # advection conserves almost exactly -- genuinely depends on elapsed
    # time. Together these are what make the discrimination assertions
    # below a real test of the ensemble's weighting logic using ONLY
    # current data, not just plumbing.
    current = SyntheticGyreField(true_origin_lat, true_origin_lon, speed_mps=0.35, length_scale_km=70.0,
                                  period_hours=96.0, bg_u_mps=0.12, bg_v_mps=-0.05, strain_rate_per_s=3e-5)
    wind = ConstantVectorField(0.0, 0.0)

    rng = np.random.default_rng(7)
    lat0, lon0 = seed_particles_in_circle(true_origin_lat, true_origin_lon, 0.3, 400, rng)
    true_lat, true_lon = integrate_particles(
        lat0, lon0, t_release, duration_seconds=true_release_hours_ago * 3600.0,
        dt_seconds=300.0, alpha=true_alpha, current_provider=current, wind_provider=wind,
        direction="forward", integrator="rk4",
    )
    hull = convex_hull_polygon(true_lat, true_lon)
    assert hull is not None, "self-test's synthetic slick collapsed to a degenerate shape"
    obs_area_km2 = polygon_area_km2(hull)
    centroid = hull.centroid
    print(f"[self-test] synthetic observed slick: centroid=({centroid.y:.4f},{centroid.x:.4f}) "
          f"area_km2={obs_area_km2:.4f}")

    detection = {
        "detection_id": "synthetic-0001",
        "lat": centroid.y,
        "lon": centroid.x,
        "polygon_geojson": shapely_mapping(hull),
        "timestamp_utc": detect_dt,
        "area_km2": obs_area_km2,
    }

    cfg = BacktrackConfig(
        min_hours_back=2.0, max_hours_back=24.0, hours_step=2.0,
        windage_ensemble_size=1,  # matches the new default: wind is negligible, so
                                  # jittering alpha would be pure wasted compute here
        n_particles_backward=200, n_particles_forward=80,
        grid_bin_deg=None,  # keep one row per raw ensemble member so we can
                            # directly inspect which hour the posterior favors, below
    )

    current_provider = SyntheticGyreField(true_origin_lat, true_origin_lon, speed_mps=0.35, length_scale_km=70.0,
                                           period_hours=96.0, bg_u_mps=0.12, bg_v_mps=-0.05, strain_rate_per_s=3e-5)
    wind_provider = ConstantVectorField(0.0, 0.0)

    result = run_ensemble_backtrack(
        centroid_lat=detection["lat"], centroid_lon=detection["lon"],
        polygon_geojson=detection["polygon_geojson"], detection_timestamp_utc=detection["timestamp_utc"],
        current_provider=current_provider, wind_provider=wind_provider,
        obs_area_km2=detection["area_km2"], cfg=cfg,
    )
    out = result.to_schema_dict()

    print(f"[self-test] most_likely_origin: {out['most_likely_origin']}")
    print(f"[self-test] true origin was:    lat={true_origin_lat}, lon={true_origin_lon}, "
          f"time={epoch_to_iso_str(t_release)}")
    print(f"[self-test] release_time_window: {out['release_time_window']}")
    print(f"[self-test] uncertainty_radius_km: {out['uncertainty_radius_km']}")
    print(f"[self-test] effective_sample_size: {out['_debug']['effective_sample_size']} "
          f"/ {out['_debug']['n_ensemble_members']} members (single windage value -- wind negligible)")

    # --- sanity assertions -------------------------------------------------
    assert len(out["origin_probability_grid"]) > 0
    # tolerance is loose enough to absorb the 6-decimal-place rounding
    # applied for display/JSON output, not because the underlying weights
    # themselves are anything less than exactly normalized
    assert abs(sum(r["probability"] for r in out["origin_probability_grid"]) - 1.0) < 1e-3, \
        "posterior probabilities must sum to ~1 (within display rounding)"
    err_km = geodesic_distance_km(
        out["most_likely_origin"]["lat"], out["most_likely_origin"]["lon"],
        true_origin_lat, true_origin_lon,
    )
    print(f"[self-test] most_likely_origin error vs. true origin: {err_km:.2f} km")
    # generous tolerance: this is a coarse ensemble grid (2h steps) on an
    # analytic test field, not a claim of pinpoint accuracy -- the real
    # bar is "in the right neighborhood, not on the other side of the map"
    assert err_km < 40.0, f"most_likely_origin is {err_km:.1f} km from the true origin -- too far for a sanity check"
    assert not out["_debug"]["degenerate_ensemble"], "ensemble should not be degenerate on a well-posed synthetic case"

    # --- does the posterior actually DISCRIMINATE release time, or did it
    #     just happen to land near the truth with no real weighting signal? ---
    from backtrack import parse_iso_to_epoch as _p2e
    graded = []
    for r in out["origin_probability_grid"]:
        hour = (t_detect - _p2e(r["time_estimate"])) / 3600.0
        graded.append((hour, r["probability"]))
    graded.sort(key=lambda hp: hp[1], reverse=True)
    best_hour, best_prob = graded[0]
    worst_hour, worst_prob = graded[-1]
    ranked_hours = [round(h) for h, _ in graded]
    true_rank = min(range(len(graded)), key=lambda i: abs(graded[i][0] - true_release_hours_ago))
    print(f"[self-test] hours ranked best->worst by posterior probability: {ranked_hours}")
    print(f"[self-test] best-supported hour={best_hour:.1f} (prob={best_prob:.4f}); "
          f"worst-supported hour={worst_hour:.1f} (prob={worst_prob:.6f}); "
          f"true hour={true_release_hours_ago:.1f} ranks #{true_rank + 1}/{len(graded)}")

    # NOTE on why these checks are deliberately loose: this synthetic test
    # field is a smooth, close-to-incompressible analytic flow (see
    # advection.polygon_elongation_ratio's docstring) -- real CMEMS/ERA5
    # fields have far more spatial structure (mesoscale eddies, fronts,
    # coastline effects) and would be expected to discriminate release time
    # more sharply than this toy case ever will. The bar here is "the ABC
    # kernel is doing SOMETHING real", not "recovers the exact hour on a
    # deliberately simplified test field".
    assert best_prob > worst_prob * 1.10, (
        "best- and worst-supported candidates have statistically indistinguishable "
        "posterior probability -- the ABC kernel isn't discriminating AT ALL between "
        "plausible and implausible release times, even loosely. Check that "
        "sigma_dist_km/sigma_log_area/sigma_log_elongation aren't absurdly large, and "
        "that the test field's strain_rate_per_s/bg_u_mps/bg_v_mps didn't get zeroed out."
    )
    assert true_rank < len(graded) * 0.6, (
        f"the true release hour ranks in the bottom 40% of the posterior "
        f"(#{true_rank + 1}/{len(graded)}) -- that's worse than a coin flip; the "
        f"ensemble should at least weakly favor hypotheses nearer the truth."
    )
    print("[self-test] OK -- posterior recovers the true origin, and the ABC kernel "
          "shows genuine (if modest, on this simplified test field) release-time discrimination")


def epoch_to_iso_str(epoch_seconds: float) -> str:
    from backtrack import epoch_to_iso
    return epoch_to_iso(epoch_seconds)


def parse_args():
    p = argparse.ArgumentParser(description="Run Module 2 (ABC ensemble ocean-drift backtracking).")
    p.add_argument("--detections", type=Path, help="Path to Module 1's detections.json")
    p.add_argument("--output", type=Path, default=Path("backtrack_results.json"))
    p.add_argument("--current-cache", type=Path, default=None,
                   help="Local NetCDF cache of CMEMS currents (checked first if set)")
    p.add_argument("--no-openmeteo", action="store_true",
                   help="Skip the free Open-Meteo Marine API fallback and go straight to the "
                        "synthetic placeholder if --current-cache isn't set or fails to load.")
    p.add_argument("--openmeteo-api-key", type=str, default=DEFAULT_CONFIG.openmeteo_api_key,
                   help="Paid/commercial Open-Meteo API key. Defaults to the OPENMETEO_API_KEY "
                        "env var if set -- prefer that over this flag so the key doesn't end up "
                        "in your shell history. Omit entirely to use the free keyless tier.")
    p.add_argument("--wind-cache", type=Path, default=None,
                   help="OPTIONAL local NetCDF cache of real ERA5/GFS wind vectors. If omitted (the "
                        "default), wind is treated as a negligible constant -- see --wind-u-mps/--wind-v-mps.")
    p.add_argument("--wind-u-mps", type=float, default=DEFAULT_CONFIG.wind_u_mps,
                   help="Constant eastward wind (m/s) used when --wind-cache isn't set. Default 0.0 "
                        "(negligible) -- ocean current is this pipeline's primary/sole real forcing.")
    p.add_argument("--wind-v-mps", type=float, default=DEFAULT_CONFIG.wind_v_mps,
                   help="Constant northward wind (m/s) used when --wind-cache isn't set. Default 0.0.")
    p.add_argument("--max-hours-back", type=float, default=DEFAULT_CONFIG.max_hours_back)
    p.add_argument("--min-hours-back", type=float, default=DEFAULT_CONFIG.min_hours_back)
    p.add_argument("--hours-step", type=float, default=DEFAULT_CONFIG.hours_step)
    p.add_argument("--windage-central", type=float, default=DEFAULT_CONFIG.windage_central)
    p.add_argument("--windage-jitter-frac", type=float, default=DEFAULT_CONFIG.windage_jitter_frac)
    p.add_argument("--windage-ensemble-size", type=int, default=DEFAULT_CONFIG.windage_ensemble_size,
                   help="Number of windage (alpha) values to jitter across. Default 1 (jitter off) "
                        "since with wind negligible by default, jittering alpha has no effect. Bump "
                        "this up once --wind-cache or a nonzero --wind-u-mps/--wind-v-mps makes it matter.")
    p.add_argument("--n-particles-backward", type=int, default=DEFAULT_CONFIG.n_particles_backward)
    p.add_argument("--n-particles-forward", type=int, default=DEFAULT_CONFIG.n_particles_forward)
    p.add_argument("--grid-bin-deg", type=float, default=DEFAULT_CONFIG.grid_bin_deg)
    p.add_argument("--max-detections", type=int, default=None, help="Only backtrack the first N detections")
    p.add_argument("--self-test", action="store_true", help="Run against a synthetic scene, ignore all other args")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.self_test:
        _self_test()
    else:
        if args.detections is None:
            raise SystemExit("--detections is required (or pass --self-test to dry-run on synthetic data)")
        cfg = BacktrackConfig(
            min_hours_back=args.min_hours_back,
            max_hours_back=args.max_hours_back,
            hours_step=args.hours_step,
            windage_central=args.windage_central,
            windage_jitter_frac=args.windage_jitter_frac,
            windage_ensemble_size=args.windage_ensemble_size,
            wind_u_mps=args.wind_u_mps,
            wind_v_mps=args.wind_v_mps,
            n_particles_backward=args.n_particles_backward,
            n_particles_forward=args.n_particles_forward,
            grid_bin_deg=args.grid_bin_deg,
            current_cache_path=args.current_cache,
            use_openmeteo=not args.no_openmeteo,
            openmeteo_api_key=args.openmeteo_api_key,
            wind_cache_path=args.wind_cache,
        )
        run_module2(args.detections, args.output, cfg, max_detections=args.max_detections)
