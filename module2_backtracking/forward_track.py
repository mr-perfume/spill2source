"""
forward_track.py
----------------------------------------------------------------------------
Module 2's forward/forecast mode -- the gap Section 6b of the build spec
flagged as "not yet implemented".

This is deliberately a NEW file rather than an edit to advection.py or
backtrack.py: it adds nothing to the physics, it only drives the existing
integrator in the other direction. advection.integrate_particles already
takes direction="forward"; all this module does is seed it from a
backtrack's most_likely_origin, run it out for a requested duration, and
snapshot the cloud along the way so the frontend has a track to animate.

Why this is much simpler than the backtrack ensemble: forecasting is a
single deterministic advection. There are no candidate hypotheses to weight
and no ABC kernel, because there is nothing to compare against -- the future
observation does not exist yet. So there is exactly one trajectory, plus an
honest spread measure derived from the particle cloud itself.

WHAT THE UNCERTAINTY HERE DOES AND DOESN'T MEAN
----------------------------------------------------------------------------
The radius reported per step is the RMS spread of the advected particle
cloud about its own centroid. It captures how the flow field stretches and
folds the patch. It does NOT capture current-field forecast error, unmodelled
wind, turbulent diffusion, or weathering -- none of which this pipeline
models. Real forecast uncertainty is larger than what comes out of here, and
grows faster with lead time. Present it as an indicative footprint, not a
confidence interval.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
for _p in (MODULE_ROOT, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from advection import (  # noqa: E402
    convex_hull_polygon,
    geodesic_distance_km,
    integrate_particles,
    polygon_area_km2,
    seed_particles_in_circle,
)
from backtrack import epoch_to_iso, parse_iso_to_epoch  # noqa: E402
from config import DEFAULT_CONFIG, BacktrackConfig  # noqa: E402


def run_forward_track(
    origin_lat: float,
    origin_lon: float,
    start_time_utc: str,
    duration_hours: float,
    current_provider,
    wind_provider,
    cfg: BacktrackConfig = DEFAULT_CONFIG,
    release_radius_km: Optional[float] = None,
    snapshot_step_hours: float = 1.0,
) -> dict:
    """Advects a particle cloud forward from a release point and returns a
    per-hour track.

    Args:
        origin_lat/lon: usually a backtrack's most_likely_origin.
        start_time_utc: release time, ISO-8601 with Z.
        duration_hours: how far ahead to forecast.
        current_provider/wind_provider: the same providers the backtrack
            used, so forward and backward runs share one physical picture.
        release_radius_km: footprint of the release. Defaults to
            cfg.initial_release_radius_km.
        snapshot_step_hours: spacing of reported track points.

    Returns a dict with a `track` list (one entry per snapshot), the final
    footprint polygon, and a `_debug` block.
    """
    if duration_hours <= 0:
        raise ValueError("duration_hours must be > 0")
    if snapshot_step_hours <= 0:
        raise ValueError("snapshot_step_hours must be > 0")

    radius_km = release_radius_km if release_radius_km is not None else cfg.initial_release_radius_km
    rng = np.random.default_rng(cfg.random_seed)
    t0 = parse_iso_to_epoch(start_time_utc)

    lat, lon = seed_particles_in_circle(origin_lat, origin_lon, radius_km, cfg.n_particles_forward, rng)

    n_snaps = int(round(duration_hours / snapshot_step_hours))
    step_seconds = snapshot_step_hours * 3600.0

    track = [_snapshot(lat, lon, t0, origin_lat, origin_lon, 0.0)]
    t = t0
    for i in range(1, n_snaps + 1):
        lat, lon = integrate_particles(
            lat, lon, t, duration_seconds=step_seconds,
            dt_seconds=cfg.integration_dt_seconds, alpha=cfg.windage_central,
            current_provider=current_provider, wind_provider=wind_provider,
            direction="forward", integrator=cfg.integrator,
        )
        t += step_seconds
        track.append(_snapshot(lat, lon, t, origin_lat, origin_lon, i * snapshot_step_hours))

    hull = convex_hull_polygon(lat, lon)
    final_polygon = None
    final_area_km2 = 0.0
    if hull is not None:
        final_area_km2 = polygon_area_km2(hull)
        final_polygon = {
            "type": "Polygon",
            "coordinates": [[[float(x), float(y)] for x, y in hull.exterior.coords]],
        }

    total_drift_km = geodesic_distance_km(
        origin_lat, origin_lon, track[-1]["lat"], track[-1]["lon"]
    )

    return {
        "start_time_utc": epoch_to_iso(t0),
        "end_time_utc": epoch_to_iso(t0 + duration_hours * 3600.0),
        "duration_hours": float(duration_hours),
        "release_point": {"lat": round(float(origin_lat), 5), "lon": round(float(origin_lon), 5)},
        "track": track,
        "final_footprint_polygon": final_polygon,
        "final_area_km2": round(final_area_km2, 4),
        "total_drift_km": round(total_drift_km, 3),
        "_debug": {
            "n_particles": int(cfg.n_particles_forward),
            "release_radius_km": round(float(radius_km), 4),
            "integrator": cfg.integrator,
            "snapshot_step_hours": float(snapshot_step_hours),
            "mean_drift_speed_kmh": round(total_drift_km / max(duration_hours, 1e-9), 3),
            "spread_is_advective_only": True,
        },
    }


def _snapshot(lat: np.ndarray, lon: np.ndarray, epoch: float,
              origin_lat: float, origin_lon: float, hours_elapsed: float) -> dict:
    """One track point: cloud centroid, its advective spread, and how far it
    has travelled from the release point."""
    c_lat = float(np.mean(lat))
    c_lon = float(np.mean(lon))
    offsets = np.array([geodesic_distance_km(c_lat, c_lon, la, lo) for la, lo in zip(lat, lon)])
    return {
        "hours_elapsed": round(float(hours_elapsed), 3),
        "time_utc": epoch_to_iso(epoch),
        "lat": round(c_lat, 5),
        "lon": round(c_lon, 5),
        "spread_radius_km": round(float(np.sqrt(np.mean(offsets ** 2))), 3),
        "distance_from_origin_km": round(geodesic_distance_km(origin_lat, origin_lon, c_lat, c_lon), 3),
    }


if __name__ == "__main__":
    # Self-test against the same synthetic field run_module2.py's self-test
    # uses, so this can be verified with no network and no Module 1 output.
    from shared_apis.drift_forcing_client import ConstantVectorField, SyntheticGyreField

    print("=== forward_track.py self-test (synthetic field) ===")
    field = SyntheticGyreField(28.20, -89.15, speed_mps=0.35, length_scale_km=70.0,
                               period_hours=96.0, bg_u_mps=0.12, bg_v_mps=-0.05,
                               strain_rate_per_s=3e-5)
    out = run_forward_track(
        28.2007, -89.1503, "2026-09-03T19:05:44Z", duration_hours=12.0,
        current_provider=field, wind_provider=ConstantVectorField(0.0, 0.0),
        release_radius_km=1.0, snapshot_step_hours=2.0,
    )
    for p in out["track"]:
        print(f"  +{p['hours_elapsed']:>5.1f}h  ({p['lat']:.4f}, {p['lon']:.4f})  "
              f"spread={p['spread_radius_km']:.2f}km  drift={p['distance_from_origin_km']:.2f}km")
    assert len(out["track"]) == 7, "expected 7 snapshots at 2h steps over 12h"
    assert out["total_drift_km"] > 0.5, "cloud should actually move in a non-zero field"
    assert out["track"][-1]["distance_from_origin_km"] >= out["track"][1]["distance_from_origin_km"], \
        "drift distance should not shrink below the first step in a steady background flow"
    print(f"[self-test] total drift {out['total_drift_km']} km, "
          f"final area {out['final_area_km2']} km2")
    print("[self-test] OK")
