"""
advection.py
The physics core for Module 2: turns a velocity field (ocean current + wind,
each queryable at any lat/lon/time) into particle trajectories.

-----------------------------------------------------------------------------
SCOPE NOTE (same spirit as preprocessing.py's terrain-correction note)
-----------------------------------------------------------------------------
This module treats the advection ODE

    dx/dt = u_current(x, t) + alpha * u_wind(x, t)

in flat local coordinates (degrees lat/lon, converted to/from meters at each
particle's current latitude every substep). That is the standard regional
short-range (hours-to-days, tens-to-few-hundred-km) drift-modeling
approximation used by GNOME/OpenDrift-class tools; it is NOT a full
geodesic/great-circle integration. For a Gulf-of-Kutch/Mumbai/Bay-of-Bengal
scale problem over a 1-48h backtrack window this approximation's error is
negligible next to the current/wind data's own uncertainty. If Module 2 is
ever extended to ocean-basin-scale, multi-week drifts, swap the per-step
degree<->meter conversion below for a proper geodesic stepper.

Per the spec: at MVP, no explicit turbulent-diffusion term is added, and the
SAME deterministic equation is integrated with time running backward for the
backtrack step. That is what keeps the backward run mathematically sound --
reversing a deterministic ODE (just flip the sign of dt) is always valid;
reversing a stochastic diffusion term is not (see backtrack.py's docstring
for how the ensemble/ABC step recovers a genuine uncertainty estimate
without needing a diffusion term at all).
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import pyproj
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry

_GEOD = pyproj.Geod(ellps="WGS84")

# ---------------------------------------------------------------------------
# Geodesy helpers
# ---------------------------------------------------------------------------
METERS_PER_DEG_LAT = 111_320.0  # WGS84 mean; good enough at this scale (see module docstring)
_MIN_COS_LAT = 1e-6             # guards the 1/cos(lat) blowup; spills aren't seeded at the poles


def meters_per_deg_lon(lat_deg: np.ndarray | float) -> np.ndarray | float:
    """East-west meters-per-degree shrinks with cos(latitude)."""
    coslat = np.cos(np.radians(lat_deg))
    coslat = np.where(np.abs(coslat) < _MIN_COS_LAT, _MIN_COS_LAT, coslat) if isinstance(coslat, np.ndarray) \
        else max(abs(coslat), _MIN_COS_LAT) * (1 if coslat >= 0 else -1)
    return METERS_PER_DEG_LAT * coslat


def mps_to_deg_per_s(u_mps: np.ndarray, v_mps: np.ndarray, lat_deg: np.ndarray):
    """Converts an eastward/northward velocity in m/s at the given
    latitude(s) into degrees-of-lon/lat per second."""
    dlat_dt = v_mps / METERS_PER_DEG_LAT
    dlon_dt = u_mps / meters_per_deg_lon(lat_deg)
    return dlat_dt, dlon_dt


def geodesic_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """True geodesic (WGS84 ellipsoid) distance -- used for scoring/reporting,
    not inside the per-substep integration loop (see module docstring)."""
    _, _, dist_m = _GEOD.inv(lon1, lat1, lon2, lat2)
    return float(dist_m) / 1000.0


def polygon_area_km2(poly: BaseGeometry) -> float:
    """Area of a WGS84 lon/lat polygon via a local azimuthal-equal-area
    reprojection centered on the polygon's own centroid -- the same
    technique Module 1's postprocessing.py uses for area_km2, kept as an
    independent, self-contained copy here deliberately: Module 2 shouldn't
    reach into Module 1's internals (incl. its underscore-prefixed helpers)
    just to avoid a ~10-line duplication."""
    if poly.is_empty:
        return 0.0
    c = poly.centroid
    aeqd = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={c.y} +lon_0={c.x} +units=m +ellps=WGS84")
    transformer = pyproj.Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)
    from shapely.ops import transform as shapely_transform
    poly_m = shapely_transform(lambda x, y, z=None: transformer.transform(x, y), poly)
    return float(poly_m.area) / 1e6


def polygon_vertex_latlon(poly: BaseGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Flattens a Polygon or MultiPolygon's exterior ring vertices into
    parallel (lat, lon) arrays -- used to feed polygon_elongation_ratio()
    the same kind of point-cloud input for an observed Module-1 polygon
    that a simulated particle cloud already is."""
    if isinstance(poly, MultiPolygon):
        coords = [c for g in poly.geoms for c in g.exterior.coords]
    elif isinstance(poly, Polygon):
        coords = list(poly.exterior.coords)
    else:
        coords = list(poly.coords)
    lons = np.array([c[0] for c in coords], dtype=np.float64)
    lats = np.array([c[1] for c in coords], dtype=np.float64)
    return lats, lons


def polygon_elongation_ratio(lats: np.ndarray, lons: np.ndarray) -> float:
    """PCA-based elongation: sqrt(largest / smallest eigenvalue) of the
    point set's covariance matrix in local flat-earth meters. 1.0 = round;
    higher = more elongated/filamentary.

    This is the shape metric backtrack.py actually leans on for ABC
    discrimination, deliberately in ADDITION to area: a divergence-free
    (incompressible) advection field conserves a patch's material AREA
    almost exactly while still stretching it into a thinner ellipse/
    filament over time (Liouville's theorem) -- so on a real current
    field, elongation genuinely grows with elapsed advection time even
    when area barely does, and gives the ensemble a real, physically-
    grounded way to discriminate candidate release times beyond "did the
    centroid end up in the right place" (which advection's time-
    reversibility makes only weakly informative on its own -- see
    backtrack.py's module docstring).

    Uses the SAME PCA method for both the observed polygon (fed its
    boundary vertices) and the simulated particle cloud (fed its raw
    particle positions), so the two numbers are computed identically and
    are actually comparable -- deliberately NOT reusing Module 1's
    skimage-ellipse-fit elongation_ratio, which is computed on a raster
    mask by a different method entirely.
    """
    if len(lats) < 3:
        return 1.0
    ref_lat = float(np.mean(lats))
    x_m = (lons - np.mean(lons)) * meters_per_deg_lon(ref_lat)
    y_m = (lats - np.mean(lats)) * METERS_PER_DEG_LAT
    cov = np.cov(np.stack([x_m, y_m]))
    eigvals = np.linalg.eigvalsh(cov)  # ascending
    eigvals = np.clip(eigvals, 1e-9, None)
    return float(math.sqrt(eigvals[-1] / eigvals[0]))


def convex_hull_polygon(lats: np.ndarray, lons: np.ndarray) -> Optional[Polygon]:
    """Convex hull of a particle cloud, used as the 'simulated slick shape'
    for the forward-verification step in backtrack.py. Convex hull (rather
    than e.g. alpha-shape/concave hull) is a deliberate MVP simplification:
    it always produces a single valid polygon from an arbitrary point cloud
    with zero risk of a degenerate/self-intersecting geometry, at the cost
    of slightly overstating area for a genuinely concave slick shape -- an
    acceptable trade for a shape-comparison summary statistic in an ABC
    likelihood, which only needs to discriminate good vs. bad candidates,
    not reproduce the exact slick outline."""
    if len(lats) < 3:
        return None
    from shapely.geometry import MultiPoint
    pts = MultiPoint(list(zip(lons, lats)))
    hull = pts.convex_hull
    if isinstance(hull, Polygon):
        return hull
    return None  # degenerate (collinear points) -> caller treats as zero area


# ---------------------------------------------------------------------------
# Velocity field provider interface
# ---------------------------------------------------------------------------
@runtime_checkable
class VelocityFieldProvider(Protocol):
    """Anything with this method can drive the integrator: a real CMEMS/
    ERA5-backed grid (shared_apis/drift_forcing_client.py), or a synthetic
    field for self-tests."""

    def velocity(self, lat: np.ndarray, lon: np.ndarray, epoch_seconds: float) -> tuple[np.ndarray, np.ndarray]:
        """Returns (u_east_mps, v_north_mps) arrays, same shape as lat/lon,
        evaluated at the given positions and single point in time."""
        ...


class ZeroField:
    """A VelocityFieldProvider that returns zero everywhere. Useful for
    isolating current-only or wind-only behavior in tests."""

    def velocity(self, lat, lon, epoch_seconds):
        z = np.zeros_like(np.asarray(lat, dtype=np.float64))
        return z, z.copy()


# ---------------------------------------------------------------------------
# Particle seeding
# ---------------------------------------------------------------------------
def seed_particles_in_polygon(poly: BaseGeometry, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Rejection-samples n points uniformly inside poly (Polygon or
    MultiPolygon). Seeding across the FULL observed shape (not just the
    centroid) matters for the backward run -- different parts of an
    elongated slick can have travelled through different parts of a
    sheared current field, so the backward-endpoint cloud's spread is
    itself real information, not noise."""
    minx, miny, maxx, maxy = poly.bounds
    lats = np.empty(n, dtype=np.float64)
    lons = np.empty(n, dtype=np.float64)
    filled = 0
    guard = 0
    max_guard = 200  # bail out with a centroid-jittered fallback rather than spin forever
    while filled < n and guard < max_guard:
        batch = max((n - filled) * 4, 64)
        cand_x = rng.uniform(minx, maxx, size=batch)
        cand_y = rng.uniform(miny, maxy, size=batch)
        from shapely import vectorized
        inside = vectorized.contains(poly, cand_x, cand_y)
        take = cand_x[inside][: n - filled]
        take_y = cand_y[inside][: n - filled]
        lons[filled: filled + len(take)] = take
        lats[filled: filled + len(take)] = take_y
        filled += len(take)
        guard += 1
    if filled < n:
        # extremely thin/degenerate polygon: fill the remainder by jittering
        # around the centroid instead of leaving uninitialized values
        c = poly.centroid
        remaining = n - filled
        lats[filled:] = c.y + rng.normal(0, 1e-4, size=remaining)
        lons[filled:] = c.x + rng.normal(0, 1e-4, size=remaining)
    return lats, lons


def seed_particles_in_circle(
    center_lat: float, center_lon: float, radius_km: float, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-area sampling (r = R*sqrt(u)) inside a small disk -- the
    hypothesized initial release footprint used for the forward
    re-simulation step in backtrack.py (deliberately NOT the observed
    polygon's full extent: the whole point of that step is to see whether a
    small point-source release naturally grows into something the observed
    polygon's size, via the velocity field's own shear, over the candidate
    elapsed time)."""
    theta = rng.uniform(0, 2 * math.pi, size=n)
    r_km = radius_km * np.sqrt(rng.uniform(0, 1, size=n))
    dlat = (r_km * np.sin(theta) * 1000.0) / METERS_PER_DEG_LAT
    dlon = (r_km * np.cos(theta) * 1000.0) / meters_per_deg_lon(center_lat)
    return center_lat + dlat, center_lon + dlon


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------
def _derivative(
    lat: np.ndarray, lon: np.ndarray, epoch_seconds: float, alpha: float,
    current_provider: VelocityFieldProvider, wind_provider: VelocityFieldProvider,
):
    u_c, v_c = current_provider.velocity(lat, lon, epoch_seconds)
    u_w, v_w = wind_provider.velocity(lat, lon, epoch_seconds)
    u_total = np.asarray(u_c) + alpha * np.asarray(u_w)
    v_total = np.asarray(v_c) + alpha * np.asarray(v_w)
    return mps_to_deg_per_s(u_total, v_total, lat)


def _rk4_step(lat, lon, t, dt, alpha, current_provider, wind_provider):
    k1_lat, k1_lon = _derivative(lat, lon, t, alpha, current_provider, wind_provider)
    k2_lat, k2_lon = _derivative(lat + 0.5 * dt * k1_lat, lon + 0.5 * dt * k1_lon,
                                  t + 0.5 * dt, alpha, current_provider, wind_provider)
    k3_lat, k3_lon = _derivative(lat + 0.5 * dt * k2_lat, lon + 0.5 * dt * k2_lon,
                                  t + 0.5 * dt, alpha, current_provider, wind_provider)
    k4_lat, k4_lon = _derivative(lat + dt * k3_lat, lon + dt * k3_lon,
                                  t + dt, alpha, current_provider, wind_provider)
    new_lat = lat + (dt / 6.0) * (k1_lat + 2 * k2_lat + 2 * k3_lat + k4_lat)
    new_lon = lon + (dt / 6.0) * (k1_lon + 2 * k2_lon + 2 * k3_lon + k4_lon)
    return new_lat, new_lon


def _euler_step(lat, lon, t, dt, alpha, current_provider, wind_provider):
    dlat_dt, dlon_dt = _derivative(lat, lon, t, alpha, current_provider, wind_provider)
    return lat + dt * dlat_dt, lon + dt * dlon_dt


def integrate_particles(
    lat0: np.ndarray, lon0: np.ndarray, t_start_epoch: float, duration_seconds: float,
    dt_seconds: float, alpha: float,
    current_provider: VelocityFieldProvider, wind_provider: VelocityFieldProvider,
    direction: str = "forward", integrator: str = "rk4",
) -> tuple[np.ndarray, np.ndarray]:
    """Integrates a particle cloud for duration_seconds starting at
    t_start_epoch. direction='backward' just flips the sign of dt and of
    the time coordinate's motion -- the SAME equation, run with t
    decreasing, which is what makes this a valid deterministic backtrack
    (see module docstring)."""
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be 'forward' or 'backward'")
    if duration_seconds < 0:
        raise ValueError("duration_seconds must be >= 0")

    sign = 1.0 if direction == "forward" else -1.0
    step = sign * abs(dt_seconds)
    n_steps = int(round(duration_seconds / abs(dt_seconds)))

    lat = np.array(lat0, dtype=np.float64, copy=True)
    lon = np.array(lon0, dtype=np.float64, copy=True)
    t = t_start_epoch
    stepper = _rk4_step if integrator == "rk4" else _euler_step

    for _ in range(n_steps):
        lat, lon = stepper(lat, lon, t, step, alpha, current_provider, wind_provider)
        t += step

    # final partial step to land exactly on t_start +/- duration if
    # duration_seconds wasn't an exact multiple of dt_seconds
    remainder = duration_seconds - n_steps * abs(dt_seconds)
    if remainder > 1e-6:
        lat, lon = stepper(lat, lon, t, sign * remainder, alpha, current_provider, wind_provider)

    return lat, lon


def integrate_backward_with_snapshots(
    lat0: np.ndarray, lon0: np.ndarray, t_detect_epoch: float, hour_grid: np.ndarray, alpha: float,
    current_provider: VelocityFieldProvider, wind_provider: VelocityFieldProvider,
    dt_seconds: float, integrator: str = "rk4",
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Runs ONE backward integration all the way out to max(hour_grid),
    recording the particle cloud's position at every hour in hour_grid along
    the way. This is an efficiency choice, not just a physics one: without
    it, an ensemble of N candidate release times would re-integrate the
    same [now, T-1h] sub-path N times over. hour_grid values must be exact
    multiples of dt_seconds/3600 (config.py's defaults guarantee this) or
    the nearest integration step is used instead."""
    hour_grid = np.sort(np.asarray(hour_grid, dtype=np.float64))
    max_hours = float(hour_grid[-1])
    step = -abs(dt_seconds)
    total_steps = int(round(max_hours * 3600.0 / abs(dt_seconds)))
    stepper = _rk4_step if integrator == "rk4" else _euler_step

    # map each requested hour -> the step index at which it's reached
    target_steps = {
        int(round(h * 3600.0 / abs(dt_seconds))): float(h) for h in hour_grid
    }

    lat = np.array(lat0, dtype=np.float64, copy=True)
    lon = np.array(lon0, dtype=np.float64, copy=True)
    t = t_detect_epoch

    snapshots: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    if 0 in target_steps:
        snapshots[target_steps[0]] = (lat.copy(), lon.copy())

    for i in range(1, total_steps + 1):
        lat, lon = stepper(lat, lon, t, step, alpha, current_provider, wind_provider)
        t += step
        if i in target_steps:
            snapshots[target_steps[i]] = (lat.copy(), lon.copy())

    missing = [h for h in hour_grid if float(h) not in snapshots]
    if missing:
        raise AssertionError(
            f"hour_grid values {missing} were not exact multiples of "
            f"dt_seconds ({dt_seconds}s) -- adjust config.hours_step / "
            f"min_hours_back or config.integration_dt_seconds so they are."
        )
    return snapshots
