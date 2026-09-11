"""
ais_service/main.py
----------------------------------------------------------------------------
Vessel correlation. This is the one piece of the pipeline that is not a
wrapper around a pre-trained model -- it is new logic, and deliberately
simple: no ML, just geometry and probability weighting.

Given a drift posterior over where and when the oil was released, and a set
of vessel tracks, it asks one question per vessel: how much of the posterior's
probability mass sits close to where this vessel actually was, during the
window when the release could have happened?

SCORING
----------------------------------------------------------------------------
For every reported position of a vessel that falls inside
release_time_window:

    grid_proximity = sum over every cell c of origin_probability_grid of
                     p(c) * exp(-distance_km(vessel, c) / SIGMA_DIST_KM)

SIGMA_DIST_KM is imported from Module 2's own config, so "close" means the
same thing here as it does inside the ABC kernel that produced the posterior.
Weighting against the whole grid rather than the single point estimate is the
point: a vessel sitting on a broad ridge of moderate probability is a better
lead than one that happens to be near the mean of a bimodal posterior, which
may be water no candidate ever favoured.

The raw sum is bounded above by 1.0 and in practice much lower, so it is
normalised against a reference: the score an imaginary vessel would get for
sitting exactly on most_likely_origin. That makes the number readable as
"how close to ideal is this vessel's position", and -- unlike normalising by
the best actual vessel -- does not hand out a 100 to the least-bad candidate
in a fleet that was all 200 km away.

    time_centrality = 1 - 2 * |t - window_centre| / window_width,  clamped

    suspicion_score = 100 * (0.7 * normalised_proximity + 0.3 * time_centrality)

Vessels with no position inside the window are still returned, with
time_centrality 0 and in_release_window false, scored on their nearest
in-time position. They belong in the output: "checked, and here is why it is
not them" is a result, and the map animation shows every vessel being
considered.

WHAT THIS IS NOT
----------------------------------------------------------------------------
A high score means a vessel was in the right water at the right time
according to a drift model with real uncertainty. It is an investigative
lead. It is not evidence of discharge, and the API deliberately returns no
field that phrases it as one.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient

SERVICE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SERVICE_ROOT.parent.parent
for _p in (PROJECT_ROOT, PROJECT_ROOT / "module2_backtracking"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

load_dotenv(PROJECT_ROOT / ".env")

from backtrack import parse_iso_to_epoch  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402

# Reuse Module 2's own distance bandwidth so "close" is defined once for the
# whole pipeline. Change it in config.py and both the ABC kernel and this
# scorer move together.
SIGMA_DIST_KM = float(os.getenv("AIS_SIGMA_DIST_KM", str(DEFAULT_CONFIG.sigma_dist_km)))
PROXIMITY_WEIGHT = float(os.getenv("AIS_PROXIMITY_WEIGHT", "0.7"))
TIME_WEIGHT = 1.0 - PROXIMITY_WEIGHT

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "oilspill")

app = FastAPI(title="AIS Correlation Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=4000)
db = _client[MONGODB_DB]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def grid_proximity(lat: float, lon: float, grid: list[dict]) -> float:
    """Probability-weighted proximity of one position to the whole posterior."""
    total = 0.0
    for cell in grid:
        d = haversine_km(lat, lon, cell["lat"], cell["lon"])
        total += float(cell.get("probability", 0.0)) * math.exp(-d / SIGMA_DIST_KM)
    return total


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class CorrelateRequest(BaseModel):
    detection_id: str
    origin_probability_grid: list[dict]
    most_likely_origin: dict
    release_time_window: list[str]
    max_results: Optional[int] = None


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        n_ships = db.ships.count_documents({})
        mongo_ok = True
    except Exception as e:  # pragma: no cover
        n_ships, mongo_ok = 0, False
        return {"status": "degraded", "service": "ais", "mongo": False, "error": str(e)}
    return {
        "status": "ok",
        "service": "ais",
        "mongo": mongo_ok,
        "ships_in_db": n_ships,
        "sigma_dist_km": SIGMA_DIST_KM,
        "proximity_weight": PROXIMITY_WEIGHT,
    }


@app.post("/ais/correlate")
def correlate(req: CorrelateRequest) -> dict[str, Any]:
    if not req.origin_probability_grid:
        raise HTTPException(400, "origin_probability_grid is empty -- nothing to score against")
    if len(req.release_time_window) != 2:
        raise HTTPException(400, "release_time_window must be [start_iso, end_iso]")

    try:
        t_start = parse_iso_to_epoch(req.release_time_window[0])
        t_end = parse_iso_to_epoch(req.release_time_window[1])
    except Exception as e:
        raise HTTPException(400, f"Unparseable release_time_window: {e}") from e
    if t_end < t_start:
        t_start, t_end = t_end, t_start
    t_centre = 0.5 * (t_start + t_end)
    half_width = max((t_end - t_start) / 2.0, 1.0)

    origin = req.most_likely_origin
    # The reference score: what an imaginary vessel sitting exactly on the
    # point estimate would earn. Everything is expressed relative to this.
    reference = grid_proximity(origin["lat"], origin["lon"], req.origin_probability_grid)
    reference = max(reference, 1e-9)

    try:
        ships = list(db.ships.find({}, {"_id": 0}))
    except Exception as e:
        raise HTTPException(503, f"Cannot read the ships collection from MongoDB: {e}") from e
    if not ships:
        raise HTTPException(
            503,
            "The ships collection is empty. Run: python scripts/seed_db.py",
        )

    ranked = []
    for ship in ships:
        path = ship.get("path") or []
        if not path:
            continue

        in_window, all_points = [], []
        for point in path:
            try:
                t = parse_iso_to_epoch(point["timestamp"])
            except Exception:
                continue
            entry = (t, float(point["lat"]), float(point["lon"]), point["timestamp"])
            all_points.append(entry)
            if t_start <= t <= t_end:
                in_window.append(entry)
        if not all_points:
            continue

        candidates = in_window if in_window else [
            min(all_points, key=lambda e: min(abs(e[0] - t_start), abs(e[0] - t_end)))
        ]

        # Best-scoring position wins: a vessel is a lead if it was ever in
        # the right place, not on average.
        best = max(candidates, key=lambda e: grid_proximity(e[1], e[2], req.origin_probability_grid))
        t_best, lat_best, lon_best, ts_best = best

        proximity = grid_proximity(lat_best, lon_best, req.origin_probability_grid)
        normalised_proximity = min(proximity / reference, 1.0)

        if in_window:
            time_centrality = max(0.0, 1.0 - abs(t_best - t_centre) / half_width)
        else:
            time_centrality = 0.0

        score = 100.0 * (PROXIMITY_WEIGHT * normalised_proximity + TIME_WEIGHT * time_centrality)

        closest_km = min(
            haversine_km(origin["lat"], origin["lon"], e[1], e[2]) for e in candidates
        )

        ranked.append({
            "ship_id": ship.get("ship_id"),
            "name": ship.get("name"),
            "type": ship.get("type"),
            "suspicion_score": round(score, 1),
            "closest_distance_km": round(closest_km, 2),
            "closest_position": {"lat": round(lat_best, 5), "lon": round(lon_best, 5)},
            "closest_timestamp": ts_best,
            "in_release_window": bool(in_window),
            "positions_in_window": len(in_window),
            "path": [
                {"lat": p["lat"], "lon": p["lon"], "timestamp": p["timestamp"]}
                for p in path
            ],
            "_scoring": {
                "grid_weighted_proximity": round(proximity, 6),
                "normalised_proximity": round(normalised_proximity, 4),
                "time_centrality": round(time_centrality, 4),
            },
        })

    ranked.sort(key=lambda r: r["suspicion_score"], reverse=True)
    if req.max_results:
        ranked = ranked[: req.max_results]

    return {
        "detection_id": req.detection_id,
        "ranked_vessels": ranked,
        "_debug": {
            "vessels_considered": len(ships),
            "vessels_in_window": sum(1 for r in ranked if r["in_release_window"]),
            "grid_cells_scored": len(req.origin_probability_grid),
            "sigma_dist_km": SIGMA_DIST_KM,
            "reference_proximity_at_point_estimate": round(reference, 6),
            "release_time_window": req.release_time_window,
            "disclaimer": (
                "Scores rank investigative leads by drift-model plausibility. "
                "They are not evidence of discharge."
            ),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("AIS_PORT", "8003")))
