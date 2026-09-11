"""
backtrack_service/main.py
----------------------------------------------------------------------------
FastAPI wrapper around Module 2 (ABC ensemble ocean-drift backtracking) plus
its new forward/forecast mode.

Module 2's own physics files -- advection.py, backtrack.py, config.py -- are
imported and called, never edited. Everything below is orchestration: build a
BacktrackConfig, resolve the current/wind providers, call
run_ensemble_backtrack, hand the result back in the spec's schema.

Endpoints:
    POST /drift/backtrack   detection record -> origin posterior
    POST /drift/forward     origin + duration -> forecast track
    GET  /health

The build spec offered two wrapping strategies: shell out to run_module2.py,
or import its entry function. This uses the import path. It skips a
subprocess plus two JSON round-trips per request, it keeps the model loaded
warm in one process, and when something goes wrong the traceback surfaces in
this service's log instead of being flattened into a non-zero exit code.
"""

from __future__ import annotations

import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

SERVICE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SERVICE_ROOT.parent.parent
for _p in (PROJECT_ROOT, PROJECT_ROOT / "module2_backtracking"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

load_dotenv(PROJECT_ROOT / ".env")

from backtrack import run_ensemble_backtrack  # noqa: E402
from config import BacktrackConfig  # noqa: E402
from forward_track import run_forward_track  # noqa: E402
from shared_apis.drift_forcing_client import (  # noqa: E402
    get_ocean_current_provider,
    get_wind_vector_provider,
)

app = FastAPI(title="Drift Backtrack Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Tuning that belongs to the integration, not to Module 2
# ---------------------------------------------------------------------------
# How far around the detection polygon to request ocean current data. Module
# 2's own default is 2.0 deg, which is generous; 0.8 deg still comfortably
# brackets a 48-hour drift at typical Gulf speeds (~0.3 m/s over 48h is about
# 52 km, roughly 0.5 deg) while keeping the Open-Meteo sample grid tight
# enough that adjacent grid points are genuinely different water.
CURRENT_BBOX_PADDING_DEG = float(os.getenv("CURRENT_BBOX_PADDING_DEG", "0.8"))

# Side length of the Open-Meteo sampling grid. 4 means 16 points in one
# batched request. See OpenMeteoGriddedCurrentField for why a grid matters.
OPENMETEO_GRID_SIDE = int(os.getenv("OPENMETEO_GRID_SIDE", "4"))

# The release footprint used for each candidate's forward re-simulation.
#
# This one needs explaining, because leaving it at Module 2's default of
# 0.3 km silently breaks the ABC kernel on real detections.
#
# backtrack.py scores each candidate partly on log(sim_area / obs_area).
# sim_area comes from advecting a circle of initial_release_radius_km
# forward; advection is close to area-preserving, so sim_area stays near the
# area of that seed circle. Our detections come out around 30 km2, whose
# equivalent radius is about 3.1 km. Seeding at 0.3 km gives an area ratio
# near 0.01, so log(ratio) is about -4.6, divided by sigma_log_area 0.6 is
# -7.7, squared is 59 -- and exp(-0.5 * 59) underflows to zero. Every
# candidate scores zero, backtrack.py trips its degenerate_ensemble guard,
# and the posterior falls back to uniform. The run "succeeds" and means
# nothing.
#
# So the seed circle is sized from the observation itself: radius =
# sqrt(obs_area / pi) * RELEASE_FOOTPRINT_FRAC. At frac 1.0 a candidate that
# preserves area scores neutrally on the area term and the discrimination
# comes from centroid offset and elongation, which is the honest position
# given this integrator models no spreading or weathering. Lower the frac if
# you want to encode "the slick grew since release"; the ratio then rewards
# candidates whose flow field stretched the patch by about that factor.
RELEASE_FOOTPRINT_FRAC = float(os.getenv("RELEASE_FOOTPRINT_FRAC", "1.0"))
MIN_RELEASE_RADIUS_KM = 0.3
MAX_RELEASE_RADIUS_KM = 25.0


def _release_radius_km(area_km2: Optional[float], cfg_default: float) -> float:
    if not area_km2 or area_km2 <= 0:
        return cfg_default
    r = math.sqrt(area_km2 / math.pi) * RELEASE_FOOTPRINT_FRAC
    return float(min(max(r, MIN_RELEASE_RADIUS_KM), MAX_RELEASE_RADIUS_KM))


def _build_config(area_km2: Optional[float], overrides: Optional[dict] = None) -> BacktrackConfig:
    cfg = BacktrackConfig()
    cfg.bbox_padding_deg = CURRENT_BBOX_PADDING_DEG
    cfg.initial_release_radius_km = _release_radius_km(area_km2, cfg.initial_release_radius_km)
    key = os.getenv("OPENMETEO_API_KEY") or None
    cfg.openmeteo_api_key = key
    cache = os.getenv("CURRENT_CACHE_PATH")
    if cache:
        cfg.current_cache_path = Path(cache)
    for k, v in (overrides or {}).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _providers(polygon_geojson: dict, t_detect: float, t_start: float, cfg: BacktrackConfig):
    from shapely.geometry import shape as shapely_shape

    minx, miny, maxx, maxy = shapely_shape(polygon_geojson).bounds
    pad = cfg.bbox_padding_deg
    bbox = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        current = get_ocean_current_provider(
            bbox, t_start, t_detect,
            cache_path=cfg.current_cache_path,
            use_openmeteo=cfg.use_openmeteo,
            openmeteo_api_key=cfg.openmeteo_api_key,
            openmeteo_grid_side=OPENMETEO_GRID_SIDE,
        )
        wind = get_wind_vector_provider(
            bbox, t_start, t_detect,
            cache_path=cfg.wind_cache_path,
            u_mps=cfg.wind_u_mps, v_mps=cfg.wind_v_mps,
        )
        messages = [str(w.message) for w in caught]
    return current, wind, bbox, messages


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class BacktrackRequest(BaseModel):
    detection_id: str
    lat: float
    lon: float
    polygon_geojson: dict
    timestamp_utc: str
    area_km2: Optional[float] = None
    min_hours_back: Optional[float] = None
    max_hours_back: Optional[float] = None
    hours_step: Optional[float] = None


class ForwardRequest(BaseModel):
    detection_id: Optional[str] = None
    lat: float
    lon: float
    start_time_utc: str
    duration_hours: float = Field(default=12.0, gt=0, le=120)
    snapshot_step_hours: float = Field(default=1.0, gt=0)
    release_radius_km: Optional[float] = None
    area_km2: Optional[float] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "backtrack",
        "openmeteo_grid_side": OPENMETEO_GRID_SIDE,
        "current_bbox_padding_deg": CURRENT_BBOX_PADDING_DEG,
        "release_footprint_frac": RELEASE_FOOTPRINT_FRAC,
        "commercial_openmeteo_key_configured": bool(os.getenv("OPENMETEO_API_KEY")),
    }


@app.post("/drift/backtrack")
def backtrack(req: BacktrackRequest) -> dict[str, Any]:
    started = time.time()
    cfg = _build_config(req.area_km2, {
        "min_hours_back": req.min_hours_back,
        "max_hours_back": req.max_hours_back,
        "hours_step": req.hours_step,
    })

    from backtrack import parse_iso_to_epoch

    try:
        t_detect = parse_iso_to_epoch(req.timestamp_utc)
    except Exception as e:
        raise HTTPException(400, f"Unparseable timestamp_utc {req.timestamp_utc!r}: {e}") from e

    t_start = t_detect - cfg.max_hours_back * 3600.0
    current, wind, bbox, warn_messages = _providers(req.polygon_geojson, t_detect, t_start, cfg)

    try:
        result = run_ensemble_backtrack(
            centroid_lat=req.lat,
            centroid_lon=req.lon,
            polygon_geojson=req.polygon_geojson,
            detection_timestamp_utc=req.timestamp_utc,
            current_provider=current,
            wind_provider=wind,
            obs_area_km2=req.area_km2,
            cfg=cfg,
        )
    except Exception as e:
        raise HTTPException(500, f"Backtrack ensemble failed: {type(e).__name__}: {e}") from e

    out = result.to_schema_dict()
    out["detection_id"] = req.detection_id

    dbg = out["_debug"]
    dbg["current_source"] = type(current).__name__
    dbg["uses_synthetic_placeholder_current"] = getattr(current, "is_synthetic_placeholder", False)
    dbg["wind_treated_as_negligible"] = getattr(wind, "is_negligible_constant", False)
    dbg["openmeteo_commercial_key_used"] = getattr(current, "used_commercial_key", False)
    dbg["initial_release_radius_km"] = round(cfg.initial_release_radius_km, 4)
    dbg["current_bbox"] = [round(v, 4) for v in bbox]
    dbg["runtime_seconds"] = round(time.time() - started, 3)
    if warn_messages:
        dbg["provider_warnings"] = warn_messages
    # How much spatial structure the current field actually carries. Near
    # zero means the ensemble had almost nothing to discriminate on, no
    # matter what the effective sample size says.
    if hasattr(current, "spatial_speed_range_mps"):
        dbg["current_spatial_speed_range_mps"] = round(current.spatial_speed_range_mps, 4)
        dbg["current_mean_speed_mps"] = round(current.mean_speed_mps, 4)
        dbg["current_grid_side"] = current.n_side
        dbg["current_land_or_null_points"] = current.land_or_null_points

    # Keep the providers around so an immediate /drift/forward call for the
    # same detection reuses the identical physical picture instead of
    # re-querying and possibly getting a differently-cached field.
    _PROVIDER_CACHE[req.detection_id] = (current, wind, time.time())
    _evict_stale_providers()

    return out


@app.post("/drift/forward")
def forward(req: ForwardRequest) -> dict[str, Any]:
    started = time.time()
    cfg = _build_config(req.area_km2)

    cached = _PROVIDER_CACHE.get(req.detection_id or "")
    if cached is not None:
        current, wind, _ = cached
        bbox = None
        warn_messages = []
    else:
        from backtrack import parse_iso_to_epoch

        t0 = parse_iso_to_epoch(req.start_time_utc)
        # A tiny square around the release point is enough to sample a
        # current field for a forecast that starts there.
        pad = cfg.bbox_padding_deg
        polygon = {
            "type": "Polygon",
            "coordinates": [[
                [req.lon - 0.01, req.lat - 0.01], [req.lon + 0.01, req.lat - 0.01],
                [req.lon + 0.01, req.lat + 0.01], [req.lon - 0.01, req.lat + 0.01],
                [req.lon - 0.01, req.lat - 0.01],
            ]],
        }
        current, wind, bbox, warn_messages = _providers(
            polygon, t0 + req.duration_hours * 3600.0, t0, cfg
        )

    radius = req.release_radius_km
    if radius is None:
        radius = _release_radius_km(req.area_km2, cfg.initial_release_radius_km)

    try:
        out = run_forward_track(
            origin_lat=req.lat,
            origin_lon=req.lon,
            start_time_utc=req.start_time_utc,
            duration_hours=req.duration_hours,
            current_provider=current,
            wind_provider=wind,
            cfg=cfg,
            release_radius_km=radius,
            snapshot_step_hours=req.snapshot_step_hours,
        )
    except Exception as e:
        raise HTTPException(500, f"Forward track failed: {type(e).__name__}: {e}") from e

    out["detection_id"] = req.detection_id
    out["_debug"]["current_source"] = type(current).__name__
    out["_debug"]["uses_synthetic_placeholder_current"] = getattr(current, "is_synthetic_placeholder", False)
    out["_debug"]["reused_backtrack_providers"] = cached is not None
    out["_debug"]["runtime_seconds"] = round(time.time() - started, 3)
    if bbox:
        out["_debug"]["current_bbox"] = [round(v, 4) for v in bbox]
    if warn_messages:
        out["_debug"]["provider_warnings"] = warn_messages
    return out


# ---------------------------------------------------------------------------
# Provider reuse between a backtrack and its follow-up forecast
# ---------------------------------------------------------------------------
_PROVIDER_CACHE: dict[str, tuple[Any, Any, float]] = {}
_PROVIDER_TTL_SECONDS = 1800


def _evict_stale_providers() -> None:
    now = time.time()
    for k in [k for k, (_, _, ts) in _PROVIDER_CACHE.items() if now - ts > _PROVIDER_TTL_SECONDS]:
        _PROVIDER_CACHE.pop(k, None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("BACKTRACK_PORT", "8002")))
