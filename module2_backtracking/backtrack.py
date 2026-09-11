"""
backtrack.py
The Bayesian / Approximate-Bayesian-Computation (ABC) ensemble engine that
turns Module 1's detection (observed polygon + exact timestamp) into a
posterior probability surface over the spill's release location and time.

-----------------------------------------------------------------------------
METHOD, spelled out (matches the spec section by section)
-----------------------------------------------------------------------------
"Represent the slick as particles"
    -> seed_particles_in_polygon() seeds n_particles_backward points
       across Module 1's ACTUAL observed polygon shape, not just its
       centroid (advection.py).

"The physics each particle obeys" / "run this backward"
    -> integrate_backward_with_snapshots() integrates
       dx/dt = u_current(x,t) + alpha*u_wind(x,t)
       backward in time (t decreasing) from the detection time. No
       diffusion term is added, which is exactly what makes running this
       backward mathematically sound (see advection.py's docstring) --
       and it's run ONCE per windage value alpha, snapshotting the particle
       cloud's position at every candidate release hour along the way,
       rather than re-integrating from scratch per candidate (see
       advection.integrate_backward_with_snapshots's docstring for why).

"This is where adjoint and Bayesian actually enter"
    -> For every (candidate release hour, windage alpha) ensemble member:
         1. candidate_origin = mean position of that alpha's backward
            snapshot at that hour.
         2. Re-seed a SMALL compact "point-source release footprint" patch
            at candidate_origin (NOT the large backward-endpoint cloud --
            see seed_particles_in_circle's docstring for why that
            distinction is the whole point of this step) and forward-
            integrate it, with the SAME alpha, for the SAME elapsed
            duration, back up to the detection time.
         3. Compare the resulting simulated particle cloud against what
            Module 1 actually observed, on THREE summary statistics:
            centroid position, convex-hull area, and a PCA-based elongation
            ratio (advection.polygon_elongation_ratio). The elongation term
            matters more than it might look: pure deterministic advection
            (no diffusion, per the spec) is close to area-conserving for a
            small patch -- so area alone is a weak time-discriminator -- but
            it still visibly stretches a patch into a thinner shape over
            time, so elongation carries the discriminating signal that area
            can't. This comparison is the ABC "simulate and compare instead
            of writing down a closed-form likelihood" step: the true
            relationship between (release point, release time, windage) and
            "resulting slick shape at detection time" has no clean formula,
            so we substitute simulation + a distance metric for a
            likelihood.
         4. weight = exp(-0.5 * distance^2)  (Gaussian ABC kernel on that
            distance -- a candidate whose forward-simulated slick nearly
            matches the real one in both size and position gets a weight
            near 1; a candidate that's way off gets a weight near 0.)

    Aggregating every ensemble member's (candidate_origin, weight) pair and
    normalizing the weights to sum to 1 IS a genuine discrete posterior
    over origin location and release time -- the practical, buildable
    stand-in for a full PDE adjoint solve that the spec calls for, without
    writing a separate adjoint model: it's the same "run particles backward"
    idea, just repeated across a weighted ensemble of physical hypotheses
    instead of once, with the weights coming from how well each hypothesis'
    forward re-simulation reproduces the actual observation.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from shapely.geometry import shape as shapely_shape

from advection import (
    VelocityFieldProvider,
    convex_hull_polygon,
    geodesic_distance_km,
    integrate_backward_with_snapshots,
    integrate_particles,
    polygon_area_km2,
    polygon_elongation_ratio,
    polygon_vertex_latlon,
    seed_particles_in_circle,
    seed_particles_in_polygon,
)
from config import BacktrackConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def parse_iso_to_epoch(timestamp_utc: str) -> float:
    ts = timestamp_utc.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def epoch_to_iso(epoch_seconds: float) -> str:
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class EnsembleMember:
    lat: float
    lon: float
    hour: float
    alpha: float
    release_time_utc: str
    weight: float
    probability: float
    sim_area_km2: float
    centroid_offset_km: float


@dataclass
class BacktrackResult:
    origin_probability_grid: list[dict]
    most_likely_origin: dict
    release_time_window: list[str]
    uncertainty_radius_km: float
    n_ensemble_members: int
    effective_sample_size: float
    debug: dict = dc_field(default_factory=dict)

    def to_schema_dict(self) -> dict:
        """Exactly the 4 top-level fields the spec's output schema names,
        plus a _debug block (same pattern as Module 1's
        _debug_features) carrying everything useful for tuning/QA that
        isn't part of the contract."""
        return {
            "origin_probability_grid": self.origin_probability_grid,
            "most_likely_origin": self.most_likely_origin,
            "release_time_window": self.release_time_window,
            "uncertainty_radius_km": round(self.uncertainty_radius_km, 3),
            "_debug": {
                "n_ensemble_members": self.n_ensemble_members,
                "effective_sample_size": round(self.effective_sample_size, 2),
                **self.debug,
            },
        }


# ---------------------------------------------------------------------------
# Weighted-statistics helpers
# ---------------------------------------------------------------------------
def _weighted_circular_mean_origin(lats: np.ndarray, lons: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Weighted mean position. Plain weighted-average lat/lon (not a proper
    spherical/geodesic mean) is fine here: candidate origins from one
    detection's ensemble are always within a few hundred km of each other,
    well inside flat-earth-approximation territory (see advection.py's
    scope note)."""
    return float(np.average(lats, weights=weights)), float(np.average(lons, weights=weights))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum_w = np.cumsum(w)
    cum_w /= cum_w[-1]
    idx = int(np.searchsorted(cum_w, q))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def _effective_sample_size(weights: np.ndarray) -> float:
    """Standard particle-filter ESS = 1/sum(w_i^2) for normalized weights --
    reports how many of the ensemble's members are actually carrying the
    posterior, vs. all the mass sitting on one lucky candidate (ESS close
    to n_ensemble_members = a well-spread, trustworthy posterior; ESS close
    to 1 = the posterior is effectively a single point estimate and the
    uncertainty_radius_km should be read with real caution)."""
    return float(1.0 / np.sum(weights ** 2)) if weights.sum() > 0 else 0.0


def _bin_to_grid(rows: list[dict], bin_deg: float) -> list[dict]:
    """Aggregates raw ensemble rows onto a coarser lat/lon grid by summing
    probability mass per cell (keeping the max-probability hour/alpha as
    that cell's representative time) -- makes for a much cleaner heatmap
    layer on the frontend than one marker per raw ensemble member."""
    cells: dict[tuple[int, int], dict] = {}
    for r in rows:
        key = (round(r["lat"] / bin_deg), round(r["lon"] / bin_deg))
        cell = cells.get(key)
        if cell is None:
            cells[key] = {
                "lat": round(key[0] * bin_deg, 5),
                "lon": round(key[1] * bin_deg, 5),
                "time_estimate": r["release_time_utc"],
                "probability": r["probability"],
                "_max_prob_in_cell": r["probability"],
            }
        else:
            # ALWAYS accumulate probability mass into the cell (this is what
            # keeps sum(probability) == 1 after binning); only the
            # representative time_estimate is swapped, and only when this
            # row individually outweighs whatever set it before.
            cell["probability"] += r["probability"]
            if r["probability"] > cell["_max_prob_in_cell"]:
                cell["_max_prob_in_cell"] = r["probability"]
                cell["time_estimate"] = r["release_time_utc"]
    out = []
    for c in cells.values():
        c.pop("_max_prob_in_cell")
        out.append(c)
    out.sort(key=lambda c: c["probability"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_ensemble_backtrack(
    centroid_lat: float,
    centroid_lon: float,
    polygon_geojson: dict,
    detection_timestamp_utc: str,
    current_provider: VelocityFieldProvider,
    wind_provider: VelocityFieldProvider,
    obs_area_km2: Optional[float] = None,
    cfg: BacktrackConfig = DEFAULT_CONFIG,
) -> BacktrackResult:
    """Runs the full ABC ensemble backtrack for one Module-1 detection.

    Args:
        centroid_lat/lon: Module 1's detection centroid (lat/lon).
        polygon_geojson: Module 1's polygon_geojson field (GeoJSON dict).
        detection_timestamp_utc: Module 1's exact timestamp_utc.
        current_provider/wind_provider: anything satisfying
            advection.VelocityFieldProvider (see
            shared_apis/drift_forcing_client.py).
        obs_area_km2: Module 1 already computed this correctly (equal-area
            reprojection) -- reuse it rather than recomputing, if given.
        cfg: BacktrackConfig; see config.py for every tunable and why its
            default is what it is.
    """
    rng = np.random.default_rng(cfg.random_seed)
    obs_polygon = shapely_shape(polygon_geojson)
    if obs_area_km2 is None:
        obs_area_km2 = polygon_area_km2(obs_polygon)
    if obs_area_km2 <= 0:
        raise ValueError(
            "obs_area_km2 is <= 0 -- Module 1's polygon_geojson looks degenerate; "
            "cannot run a shape-comparison ABC likelihood against a zero-area target."
        )
    obs_lats, obs_lons = polygon_vertex_latlon(obs_polygon)
    obs_elongation = polygon_elongation_ratio(obs_lats, obs_lons)

    t_detect = parse_iso_to_epoch(detection_timestamp_utc)

    if cfg.windage_ensemble_size <= 1:
        # np.linspace(low, high, 1) would return the LOW endpoint, not the
        # center -- special-case this rather than silently jittering off of
        # the wrong value. windage_ensemble_size defaults to 1 (jitter off)
        # precisely because wind is negligible by default (see config.py),
        # so this is the common path, not an edge case to skip past.
        windage_values = np.array([cfg.windage_central], dtype=np.float64)
    else:
        windage_values = np.linspace(
            cfg.windage_central * (1 - cfg.windage_jitter_frac),
            cfg.windage_central * (1 + cfg.windage_jitter_frac),
            cfg.windage_ensemble_size,
        )
    hour_grid = np.arange(cfg.min_hours_back, cfg.max_hours_back + 1e-9, cfg.hours_step)

    # Seed once, reuse the SAME starting positions across every windage
    # value -- otherwise differences between ensemble members would be
    # partly attributable to random seeding noise rather than the physics
    # being varied.
    seed_lats, seed_lons = seed_particles_in_polygon(obs_polygon, cfg.n_particles_backward, rng)

    rows: list[dict] = []

    for alpha in windage_values:
        snapshots = integrate_backward_with_snapshots(
            seed_lats, seed_lons, t_detect, hour_grid, float(alpha),
            current_provider, wind_provider,
            dt_seconds=cfg.integration_dt_seconds, integrator=cfg.integrator,
        )

        for hour in hour_grid:
            snap_lat, snap_lon = snapshots[float(hour)]
            origin_lat = float(np.mean(snap_lat))
            origin_lon = float(np.mean(snap_lon))
            t_release = t_detect - float(hour) * 3600.0

            fwd_lat0, fwd_lon0 = seed_particles_in_circle(
                origin_lat, origin_lon, cfg.initial_release_radius_km, cfg.n_particles_forward, rng
            )
            fwd_lat, fwd_lon = integrate_particles(
                fwd_lat0, fwd_lon0, t_release, duration_seconds=float(hour) * 3600.0,
                dt_seconds=cfg.integration_dt_seconds, alpha=float(alpha),
                current_provider=current_provider, wind_provider=wind_provider,
                direction="forward", integrator=cfg.integrator,
            )

            sim_hull = convex_hull_polygon(fwd_lat, fwd_lon)
            sim_area_km2 = polygon_area_km2(sim_hull) if sim_hull is not None else 0.0
            sim_centroid_lat = float(np.mean(fwd_lat))
            sim_centroid_lon = float(np.mean(fwd_lon))
            sim_elongation = polygon_elongation_ratio(fwd_lat, fwd_lon)

            centroid_offset_km = geodesic_distance_km(
                sim_centroid_lat, sim_centroid_lon, centroid_lat, centroid_lon
            )
            log_area_ratio = math.log(max(sim_area_km2, 1e-6) / max(obs_area_km2, 1e-6))
            log_elong_ratio = math.log(max(sim_elongation, 1e-3) / max(obs_elongation, 1e-3))

            distance = math.sqrt(
                (centroid_offset_km / cfg.sigma_dist_km) ** 2
                + (log_area_ratio / cfg.sigma_log_area) ** 2
                + (log_elong_ratio / cfg.sigma_log_elongation) ** 2
            )
            weight = math.exp(-0.5 * distance ** 2)

            rows.append({
                "lat": origin_lat,
                "lon": origin_lon,
                "hour": float(hour),
                "alpha": float(alpha),
                "release_time_utc": epoch_to_iso(t_release),
                "weight": weight,
                "sim_area_km2": round(sim_area_km2, 6),
                "sim_elongation_ratio": round(sim_elongation, 4),
                "centroid_offset_km": round(centroid_offset_km, 3),
            })

    weights = np.array([r["weight"] for r in rows], dtype=np.float64)
    weight_sum = float(weights.sum())
    degenerate = not np.isfinite(weight_sum) or weight_sum <= 0
    if degenerate:
        # Every candidate scored ~0 -- almost always means the (time, alpha)
        # search range doesn't bracket anything physically plausible for
        # this scene, OR the current/wind fields are too coarse/wrong. Fall
        # back to a uniform posterior rather than dividing by zero or
        # silently returning NaNs; this is a loud, catchable state (flagged
        # in _debug.degenerate_ensemble), not a hidden one.
        weights = np.ones_like(weights)
        weight_sum = float(weights.sum())
    probabilities = weights / weight_sum
    for r, p in zip(rows, probabilities):
        r["probability"] = float(p)

    lats = np.array([r["lat"] for r in rows])
    lons = np.array([r["lon"] for r in rows])
    hours = np.array([r["hour"] for r in rows])

    mean_lat, mean_lon = _weighted_circular_mean_origin(lats, lons, probabilities)
    mean_hour = float(np.average(hours, weights=probabilities))
    mean_release_time = epoch_to_iso(t_detect - mean_hour * 3600.0)

    # Credible interval on release time: mass-weighted quantiles of the
    # MARGINAL over hour (sum probability across alpha for each hour value).
    lower_q = (1 - cfg.credible_mass) / 2
    upper_q = 1 - lower_q
    t_min_hour = _weighted_quantile(hours, probabilities, lower_q)
    t_max_hour = _weighted_quantile(hours, probabilities, upper_q)
    # smaller hour = MORE RECENT release = later wall-clock time
    release_time_window = [
        epoch_to_iso(t_detect - t_max_hour * 3600.0),
        epoch_to_iso(t_detect - t_min_hour * 3600.0),
    ]

    # Uncertainty radius: weighted RMS geodesic distance of every candidate
    # origin from the weighted-mean origin -- a genuine spatial spread
    # measure, not just "biggest minus smallest".
    offsets_km = np.array([
        geodesic_distance_km(mean_lat, mean_lon, lat, lon) for lat, lon in zip(lats, lons)
    ])
    uncertainty_radius_km = float(np.sqrt(np.average(offsets_km ** 2, weights=probabilities)))

    ess = _effective_sample_size(probabilities)

    if cfg.grid_bin_deg:
        grid_rows = _bin_to_grid(rows, cfg.grid_bin_deg)
    else:
        grid_rows = sorted(
            [{"lat": round(r["lat"], 5), "lon": round(r["lon"], 5),
              "time_estimate": r["release_time_utc"], "probability": round(r["probability"], 6)}
             for r in rows],
            key=lambda r: r["probability"], reverse=True,
        )

    best = max(rows, key=lambda r: r["probability"])

    return BacktrackResult(
        origin_probability_grid=grid_rows,
        most_likely_origin={
            "lat": round(mean_lat, 5),
            "lon": round(mean_lon, 5),
            "time": mean_release_time,
        },
        release_time_window=release_time_window,
        uncertainty_radius_km=uncertainty_radius_km,
        n_ensemble_members=len(rows),
        effective_sample_size=ess,
        debug={
            "degenerate_ensemble": degenerate,
            "map_estimate": {  # mode of the posterior, alongside the mean used above
                "lat": round(best["lat"], 5), "lon": round(best["lon"], 5),
                "time": best["release_time_utc"], "probability": round(best["probability"], 6),
            },
            "windage_range": [round(float(windage_values.min()), 4), round(float(windage_values.max()), 4)],
            "hour_range": [float(hour_grid.min()), float(hour_grid.max())],
            "obs_area_km2": round(obs_area_km2, 6),
            "obs_elongation_ratio": round(obs_elongation, 4),
        },
    )
