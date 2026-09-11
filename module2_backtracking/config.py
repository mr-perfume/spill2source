"""
config.py
Central configuration for Module 2 (Bayesian / Approximate-Bayesian-
Computation ensemble ocean-drift backtracking).

Every physical assumption or ensemble knob the pipeline depends on lives
here so advection.py / backtrack.py / run_module2.py never hardcode a
number inline -- same principle Module 1's preprocessing.py used for
db_min/db_max: change it in exactly one place.

-----------------------------------------------------------------------------
WHY THESE SPECIFIC DEFAULTS (read before tuning)
-----------------------------------------------------------------------------
windage_central / windage_jitter_frac / windage_ensemble_size:
    Ocean current (via the CMEMS API path in shared_apis/drift_forcing_client.py)
    is the primary -- practically the sole -- real physical forcing this
    pipeline uses. Wind is treated as NEGLIGIBLE by default: no wind API is
    called, wind_u_mps/wind_v_mps default to 0.0 (a plain constant, not a
    spatiotemporal grid), and windage_ensemble_size defaults to 1 (just the
    central alpha value) because jittering a multiplier on a zero vector
    has no effect on anything -- running 7 identical branches would just
    waste 7x the compute for identical results. If you later want windage
    to matter again (either by setting a nonzero wind_u_mps/wind_v_mps
    constant, or by pointing wind_cache_path at a real ERA5/GFS file), bump
    windage_ensemble_size back up (7 is a reasonable default at that point)
    -- see get_wind_provider() below for how wind is actually resolved.

hours_step / min_hours_back / max_hours_back:
    24 candidate release times (1..48h @ 2h step) by default. With
    windage_ensemble_size=1 (wind negligible), that's 24 ensemble members
    total; it multiplies back up to hours x windage_ensemble_size if you
    turn windage jitter back on. Each member costs one full backward
    integration snapshot (shared across the hour grid per alpha -- see
    advection.integrate_backward_with_snapshots) plus one short forward
    re-simulation. This is deliberately snappy: no ensemble member touches
    a real network API at request time (see shared_apis/drift_forcing_client.py).

integration_dt_seconds:
    300s (5 min) substeps. hours_step*3600 and min_hours_back*3600 are both
    exact multiples of 300, so every candidate release time lands exactly
    on an integration step -- no interpolation-in-time needed for snapshots.

sigma_dist_km / sigma_log_area:
    Bandwidths of the Gaussian ABC kernel (see backtrack.py). These are
    tuning knobs, not physical constants -- tighten them to make the
    posterior more peaked/confident, loosen them if every candidate is
    getting a near-zero weight (a sign the ensemble's time/windage range
    doesn't bracket the truth, or the current/wind fields are too coarse).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent


@dataclass
class BacktrackConfig:
    # ---- Windage (leeway) ensemble ------------------------------------------------
    # windage_central/windage_jitter_frac are the standard oil-leeway range
    # (0.02-0.04 x wind speed; ASCE/API guidance, GNOME/OpenDrift's default
    # "wind_drift_factor") -- kept for whenever real wind data is wired in.
    # windage_ensemble_size defaults to 1 (jitter OFF) because with wind
    # treated as negligible (see wind_u_mps/wind_v_mps below), jittering
    # alpha has no effect to discriminate on. Bump to 7 once wind is real.
    windage_central: float = 0.03
    windage_jitter_frac: float = 0.30
    windage_ensemble_size: int = 1

    # ---- Wind forcing ----------------------------------------------------------------
    # No wind API is called by default -- ocean current (CMEMS, via
    # current_cache_path below) is the primary/sole real forcing. wind_u_mps/
    # wind_v_mps is a plain CONSTANT vector (not a spatiotemporal grid); 0.0
    # means "negligible wind" (the default). Set a rough manual estimate
    # here (e.g. a known prevailing wind for your region/season) if you want
    # windage to contribute WITHOUT standing up a full ERA5/GFS integration
    # -- get_wind_provider() in shared_apis/drift_forcing_client.py only
    # reaches for wind_cache_path (a real ERA5/GFS NetCDF) if you set one.
    wind_u_mps: float = 0.0
    wind_v_mps: float = 0.0

    # ---- Candidate release-time ensemble -------------------------------------------
    min_hours_back: float = 1.0
    max_hours_back: float = 48.0
    hours_step: float = 2.0

    # ---- Integration ----------------------------------------------------------------
    integration_dt_seconds: float = 300.0    # 5 min substeps
    integrator: str = "rk4"                  # "rk4" or "euler"

    # ---- Particle clouds --------------------------------------------------------------
    n_particles_backward: int = 500          # seeded across the OBSERVED polygon
    n_particles_forward: int = 150           # seeded in a small release-footprint circle
    initial_release_radius_km: float = 0.3   # "point source" footprint for re-simulation
    random_seed: int = 42

    # ---- ABC likelihood kernel -----------------------------------------------------
    # distance = sqrt((centroid_offset_km / sigma_dist_km)^2
    #                + (log(sim_area_km2 / obs_area_km2) / sigma_log_area)^2)
    # weight   = exp(-0.5 * distance^2)   (smooth Gaussian ABC kernel -- avoids
    #                                       classic rejection-ABC's "zero samples
    #                                       survived" failure mode)
    sigma_dist_km: float = 8.0
    sigma_log_area: float = 0.6
    # Elongation (shape) term -- see advection.polygon_elongation_ratio's
    # docstring for why this carries real signal even when advection is
    # near-incompressible and area barely changes.
    sigma_log_elongation: float = 0.5

    # ---- Reporting -------------------------------------------------------------------
    credible_mass: float = 0.80              # mass covered by release_time_window
    grid_bin_deg: Optional[float] = 0.05      # bin origin_probability_grid onto a lat/lon
                                              # grid this coarse for the frontend heatmap;
                                              # None -> return one row per raw ensemble member

    # ---- Data sources ------------------------------------------------------------------
    current_cache_path: Optional[Path] = None   # local NetCDF cache of CMEMS currents (checked FIRST)
    use_openmeteo: bool = True                   # if no cache, call the free/keyless Open-Meteo
                                                  # Marine API for real current data (default path
                                                  # for anyone without a CMEMS account -- see
                                                  # shared_apis/drift_forcing_client.py). Only
                                                  # falls through to the synthetic placeholder if
                                                  # this is False or the API call fails.
    openmeteo_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("OPENMETEO_API_KEY"))
                                                  # PAID/commercial Open-Meteo key, if you have one.
                                                  # Defaults from the OPENMETEO_API_KEY env var so the
                                                  # key never has to be typed on the command line or
                                                  # committed to a config file -- set it once with
                                                  # export OPENMETEO_API_KEY=... in your shell.
                                                  # None (the default) uses the free keyless tier.
    wind_cache_path: Optional[Path] = None       # OPTIONAL local NetCDF cache of real ERA5/GFS wind
                                                  # vectors -- only used if you explicitly set this;
                                                  # otherwise wind_u_mps/wind_v_mps (constant, default
                                                  # 0.0) is used and no wind API is ever called
    bbox_padding_deg: float = 2.0                # how far around the polygon to request
                                                  # current data for

    extra: dict = field(default_factory=dict)


DEFAULT_CONFIG = BacktrackConfig()
