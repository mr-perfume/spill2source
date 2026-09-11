"""
drift_forcing_client.py
Supplies the vector field(s) Module 2's advection physics needs.

Ocean surface current (u, v) is the PRIMARY -- practically the sole real --
physical forcing this pipeline uses. By DEFAULT this comes from the
Open-Meteo Marine API (free, keyless, one HTTP request, no account needed --
see OpenMeteoCurrentField below) rather than CMEMS directly: same
underlying data family, none of the registration friction. A real CMEMS
NetCDF cache (via --current-cache) is still supported and takes priority
if you have one.

Wind is treated as NEGLIGIBLE by default: no wind API is called at all.
get_wind_provider() returns a plain constant vector (0.0, 0.0 m/s unless you
set config.wind_u_mps/wind_v_mps to a rough manual estimate), not a
spatiotemporal grid pulled from ERA5/GFS. Standing up a full second
API/reanalysis integration just for a leeway correction of a few percent
wasn't worth the complexity for this pipeline -- see config.py's docstring
for the reasoning. If you ever do want real wind data, get_wind_provider()
still accepts a wind_cache_path (a real ERA5/GFS NetCDF) and will use it.

-----------------------------------------------------------------------------
LIVE API THAT IS WIRED UP, AND REAL APIS THAT ARE DELIBERATELY NOT
-----------------------------------------------------------------------------
WIRED UP: Open-Meteo Marine API (https://marine-api.open-meteo.com/v1/marine)
  -- OpenMeteoCurrentField below makes a real HTTP GET at request time. No
  credentials needed, marine-api.open-meteo.com just needs to be reachable.

NOT wired up (optional, account-gated alternatives -- CMEMS/ERA5 still both
need credentials this environment doesn't have and hosts outside this
environment's network allowlist; same stance Module 1's preprocessing.py
took on land/sea masking -- fail loudly, never silently fabricate a value
that downstream code would trust as real):
  - CMEMS (a higher-resolution, longer-history alternative to Open-Meteo's
    current data): the copernicusmarine python client (`pip install
    copernicusmarine`), needs a free CMEMS account (`copernicusmarine
    login` once), then something like:

        copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
            variables=["uo", "vo"],
            minimum_longitude=..., maximum_longitude=...,
            minimum_latitude=...,  maximum_latitude=...,
            start_datetime=..., end_datetime=...,
            output_filename="currents.nc",
        )

  - ERA5 (OPTIONAL, only if you want real wind instead of the negligible
    constant default): the cdsapi python client, needs a Climate Data
    Store API key in ~/.cdsapirc, then
    `cdsapi.Client().retrieve("reanalysis-era5-single-levels", {...u10/v10
    request...}, "wind.nc")`.

WHAT IS IMPLEMENTED:
  1. NetCDFGridField -- the real path for ocean current, once you've
     pre-downloaded a CMEMS slice for your demo's region + date range
     (exactly the "cache data locally instead of live calls" pattern the
     spec itself calls for). Point --current-cache at that file. Also
     usable for wind if you ever set --wind-cache.
  2. OpenMeteoCurrentField -- the DEFAULT real path for ocean current: one
     plain HTTP GET to the free, keyless Open-Meteo Marine API, no account
     needed at all. Used automatically whenever --current-cache isn't set.
     This is what makes "just run it" actually work without any signup.
  3. ConstantVectorField -- the default for wind: a plain, constant (u, v)
     vector, 0.0 by default (negligible). Not an API call, not a grid --
     just a number you can optionally set if you know a rough prevailing
     wind for your region/season and want windage to matter a little
     without a full ERA5 integration.
  4. SyntheticGyreField -- an analytic, smooth, mildly time-varying field,
     the LAST-RESORT fallback for current if neither a cache nor Open-Meteo
     is available (e.g. no network). Loudly warns and flags
     is_synthetic_placeholder=True so a demo never silently runs on fake
     current physics without it showing up in the output.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VelocityFieldProvider(Protocol):
    def velocity(self, lat: np.ndarray, lon: np.ndarray, epoch_seconds: float) -> tuple[np.ndarray, np.ndarray]:
        ...


# ---------------------------------------------------------------------------
# Real path: local NetCDF cache (CMEMS/ERA5 slice pre-downloaded before demo)
# ---------------------------------------------------------------------------
class NetCDFGridField:
    """Loads a local NetCDF with dims (time, lat, lon) and u/v variables,
    builds a fast trilinear interpolator over it via scipy's
    RegularGridInterpolator. Expected file layout (matches what
    copernicusmarine.subset()/cdsapi both produce by default):

        dims:  time (datetime64 or hours-since epoch), lat, lon
        vars:  u_var (default "uo" for CMEMS currents / "u10" for ERA5 wind),
               v_var (default "vo" / "v10")

    Pass explicit var names if your cached file uses different ones.
    """

    def __init__(
        self, nc_path: Path,
        u_var: str = "uo", v_var: str = "vo",
        lat_var: str = "latitude", lon_var: str = "longitude", time_var: str = "time",
    ):
        import xarray as xr
        from scipy.interpolate import RegularGridInterpolator

        ds = xr.open_dataset(nc_path)
        if u_var not in ds or v_var not in ds:
            raise KeyError(
                f"{nc_path} has no '{u_var}'/'{v_var}' variables (found: "
                f"{list(ds.data_vars)}). Pass the correct u_var/v_var for "
                f"this file."
            )

        lat = ds[lat_var].values.astype(np.float64)
        lon = ds[lon_var].values.astype(np.float64)
        # normalize time to epoch seconds so callers can pass plain floats
        time_epoch = ds[time_var].values.astype("datetime64[s]").astype(np.int64).astype(np.float64)

        u = ds[u_var].values.astype(np.float64)
        v = ds[v_var].values.astype(np.float64)
        # collapse a possible singleton depth dim (surface currents at depth=0)
        if u.ndim == 4:
            u = u[:, 0, :, :]
            v = v[:, 0, :, :]
        u = np.nan_to_num(u, nan=0.0)
        v = np.nan_to_num(v, nan=0.0)

        # sort ascending on every axis -- RegularGridInterpolator requires it
        order_t = np.argsort(time_epoch)
        order_lat = np.argsort(lat)
        order_lon = np.argsort(lon)
        time_epoch, lat, lon = time_epoch[order_t], lat[order_lat], lon[order_lon]
        u = u[order_t][:, order_lat][:, :, order_lon]
        v = v[order_t][:, order_lat][:, :, order_lon]

        self._t_bounds = (float(time_epoch.min()), float(time_epoch.max()))
        self._lat_bounds = (float(lat.min()), float(lat.max()))
        self._lon_bounds = (float(lon.min()), float(lon.max()))
        self._u_interp = RegularGridInterpolator(
            (time_epoch, lat, lon), u, bounds_error=False, fill_value=None
        )
        self._v_interp = RegularGridInterpolator(
            (time_epoch, lat, lon), v, bounds_error=False, fill_value=None
        )
        self.source_path = str(nc_path)

    def velocity(self, lat: np.ndarray, lon: np.ndarray, epoch_seconds: float):
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        t_clamped = min(max(epoch_seconds, self._t_bounds[0]), self._t_bounds[1])
        pts = np.stack([np.full_like(lat, t_clamped), lat, lon], axis=-1)
        u = self._u_interp(pts)
        v = self._v_interp(pts)
        return u, v


# ---------------------------------------------------------------------------
# Default wind path: a plain constant, not an API/grid at all
# ---------------------------------------------------------------------------
class ConstantVectorField:
    """The default wind provider: a single (u, v) in m/s, the same
    everywhere and at every time -- no API call, no grid, no interpolation.
    (0.0, 0.0) means "negligible wind" (this pipeline's default stance:
    ocean current is the primary/sole real forcing). Set a nonzero
    constant via config.wind_u_mps/wind_v_mps if you know a rough
    prevailing wind for your region/season and want windage to contribute
    a little without standing up a full ERA5/GFS integration -- windage
    jitter (config.windage_ensemble_size > 1) becomes meaningful again as
    soon as this is nonzero, since different alpha values then genuinely
    produce different net drift."""

    is_synthetic_placeholder: bool = False  # this isn't a "stand-in for missing
                                             # data" in the same sense as
                                             # SyntheticGyreField -- it's an
                                             # intentional simplification, so it
                                             # gets its own flag (see below)
    is_negligible_constant: bool = False

    def __init__(self, u_mps: float = 0.0, v_mps: float = 0.0):
        self.u_mps = u_mps
        self.v_mps = v_mps
        self.is_negligible_constant = (u_mps == 0.0 and v_mps == 0.0)

    def velocity(self, lat: np.ndarray, lon: np.ndarray, epoch_seconds: float):
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        return np.full_like(lat, self.u_mps), np.full_like(lat, self.v_mps)


# ---------------------------------------------------------------------------
# Simple real path: Open-Meteo Marine API -- free, keyless, no account
# ---------------------------------------------------------------------------
_OPENMETEO_CLIENT = None  # module-level singleton -- one cache/retry session
                          # reused across every OpenMeteoCurrentField instance
                          # in a process, not re-created per detection


def _get_openmeteo_client():
    """Lazily builds Open-Meteo's official client, wrapped with a disk-
    persisted cache (repeat requests for the same region+dates within the
    cache window are served locally, no network round-trip) and automatic
    retry with backoff on transient failures. Lazy on purpose: importing
    this module shouldn't require these packages installed unless
    OpenMeteoCurrentField is actually used.

    Needs: pip install openmeteo_requests requests_cache retry_requests
    """
    global _OPENMETEO_CLIENT
    if _OPENMETEO_CLIENT is None:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry

        cache_path = Path(__file__).resolve().parent.parent / "data" / "openmeteo_cache"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_session = requests_cache.CachedSession(str(cache_path), expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        _OPENMETEO_CLIENT = openmeteo_requests.Client(session=retry_session)
    return _OPENMETEO_CLIENT


class OpenMeteoCurrentField:
    """Ocean current from the Open-Meteo Marine API
    (https://open-meteo.com/en/docs/marine-weather-api), via Open-Meteo's
    OWN official client library (openmeteo_requests) rather than a hand-
    rolled urllib call -- this gets you two real robustness upgrades for
    free, straight from their SDK:
      - a shared, disk-persisted HTTP cache (requests_cache): if you run
        several detections against the same region/date window in one
        process (or re-run the pipeline while iterating), repeat requests
        are served from local cache instead of re-hitting the network.
      - automatic retry with backoff (retry_requests) on transient
        failures, instead of one urlopen attempt giving up immediately.

    Still works with NO API key/account at all (free, non-commercial use,
    up to 10,000 calls/day). If you have a paid/commercial Open-Meteo API
    key, pass it as api_key -- see below for exactly what that changes.

    Under the hood, Open-Meteo's ocean current variable is itself sourced
    from CMEMS (the "MeteoFrance SMOC Currents" product, ~8km resolution,
    the same GLOBAL_ANALYSISFORECAST_PHY product NetCDFGridField's docstring
    points at) -- so this isn't lower-quality data, just a much simpler way
    to get it.

    LIMITATION (stated plainly, not hidden): this queries ONE representative
    point (the request bbox's center) and returns that point's current,
    UNCHANGED across whatever lat/lon the integrator asks for -- i.e.
    spatially uniform, temporally varying. Fine for a single-scene backtrack
    over a region a few tens of km across (which is this pipeline's whole
    operating range); NOT a substitute for NetCDFGridField's real spatial
    grid if you need current shear across a wider area.

    Also note the API's own accuracy caveat: "Accuracy at coastal areas is
    limited... not suitable for coastal navigation." Fine for this
    pipeline's ABC posterior (which already treats current data as
    uncertain), not fine as ground truth.

    Needs pip install openmeteo_requests requests_cache retry_requests
    (not in this project's original requirements.txt -- add them).
    """

    is_synthetic_placeholder: bool = False

    _FREE_URL = "https://marine-api.open-meteo.com/v1/marine"
    _COMMERCIAL_URL = "https://customer-marine-api.open-meteo.com/v1/marine"

    def __init__(
        self, center_lat: float, center_lon: float,
        start_date: str, end_date: str,
        api_key: Optional[str] = None,
    ):
        client = _get_openmeteo_client()
        params = {
            "latitude": center_lat,
            "longitude": center_lon,
            "hourly": ["ocean_current_velocity", "ocean_current_direction"],
            "length_unit": "metric",       # velocity comes back in km/h
            "start_date": start_date,       # yyyy-mm-dd
            "end_date": end_date,           # yyyy-mm-dd
        }
        # A commercial key isn't just an extra param -- Open-Meteo routes
        # paid-tier traffic through a DIFFERENT host (the "customer-"
        # prefix). Hitting the free host with an apikey param, or the
        # customer host without one, both fail -- so the URL and the param
        # are switched together, never independently.
        if api_key:
            url = self._COMMERCIAL_URL
            params["apikey"] = api_key
        else:
            url = self._FREE_URL
        self.used_commercial_key = bool(api_key)

        responses = client.weather_api(url, params=params)
        if not responses:
            raise ValueError("Open-Meteo returned no responses for this request")
        response = responses[0]

        hourly = response.Hourly()
        if hourly is None or hourly.VariablesLength() < 2:
            raise ValueError(
                f"Open-Meteo response has no usable hourly current data "
                f"(VariablesLength={0 if hourly is None else hourly.VariablesLength()})"
            )
        # order matches the hourly list in params above: [velocity, direction]
        speed_kmh = hourly.Variables(0).ValuesAsNumpy().astype(np.float64)
        direction_deg = hourly.Variables(1).ValuesAsNumpy().astype(np.float64)
        # Time()/TimeEnd() are unix epoch seconds, Interval() is seconds/step
        # -- np.arange with 'end' exclusive matches the SDK's own left-
        # inclusive convention (see Open-Meteo's own pandas example, which
        # uses inclusive="left" for the equivalent pd.date_range call).
        t = np.arange(hourly.Time(), hourly.TimeEnd(), hourly.Interval(), dtype=np.float64)

        n = min(len(t), len(speed_kmh), len(direction_deg))
        t, speed_kmh, direction_deg = t[:n], speed_kmh[:n], direction_deg[:n]
        speed_kmh = np.nan_to_num(speed_kmh, nan=0.0)
        direction_deg = np.nan_to_num(direction_deg, nan=0.0)
        speed_mps = speed_kmh / 3.6

        # Open-Meteo's ocean_current_direction is the direction the current
        # is FLOWING TOWARD (0=north, 90=east) -- NOT the meteorological
        # "direction it's coming from" convention wave_direction uses. So
        # this is a plain vector decomposition, no 180-degree flip needed.
        direction_rad = np.radians(direction_deg)
        self._t = t
        self._u = speed_mps * np.sin(direction_rad)
        self._v = speed_mps * np.cos(direction_rad)
        self.latitude = response.Latitude()
        self.longitude = response.Longitude()

    def velocity(self, lat: np.ndarray, lon: np.ndarray, epoch_seconds: float):
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        t_clamped = min(max(epoch_seconds, self._t[0]), self._t[-1])
        u = float(np.interp(t_clamped, self._t, self._u))
        v = float(np.interp(t_clamped, self._t, self._v))
        return np.full_like(lat, u), np.full_like(lat, v)


# ---------------------------------------------------------------------------
# Preferred real path: Open-Meteo Marine API sampled on a SPATIAL GRID
# ---------------------------------------------------------------------------
class OpenMeteoGriddedCurrentField:
    """Same Open-Meteo Marine data as OpenMeteoCurrentField, but sampled at
    an n x n grid of points across the request bbox in ONE batched call,
    then bilinearly interpolated in space and linearly in time.

    WHY THIS EXISTS
    ---------------------------------------------------------------------
    OpenMeteoCurrentField queries a single representative point and returns
    that velocity unchanged everywhere -- spatially uniform, temporally
    varying. Its own docstring says so plainly. That is fine for a rough
    drift estimate, but it quietly destroys the ABC ensemble's ability to
    discriminate between candidate release times, because in a spatially
    uniform flow:

      - a particle cloud advected backward h hours and then forward h hours
        returns exactly to where it started, for EVERY h, so
        centroid_offset_km is ~0 for every candidate;
      - there is no velocity gradient, so the cloud's area and elongation
        are conserved exactly, so the area and elongation terms of the ABC
        kernel take the same value for every candidate too.

    All three terms of backtrack.py's likelihood then coincide across every
    hypothesis, the posterior comes out flat, and the reported
    most_likely_origin degrades into an unweighted average along the drift
    track. Nothing crashes -- it just stops being Bayesian in any useful
    sense.

    A grid of points restores a real velocity gradient (shear and strain),
    which is exactly the signal backtrack.py's kernel was written to
    exploit. Open-Meteo accepts comma-separated latitude/longitude lists and
    returns one response object per location, so this costs ONE request, not
    n^2 requests, and it shares the same cached/retrying session.

    Honest caveats:
      - Grid spacing is bbox_width / (n_side - 1), typically tens of km.
        This resolves large-scale shear, not mesoscale eddies. A real CMEMS
        NetCDF via NetCDFGridField is still the higher-fidelity path.
      - Open-Meteo's own accuracy note applies: limited near coastlines,
        not suitable for navigation.
      - Some grid points may fall on land, where the API returns nulls.
        Those are zero-filled, which biases the local field toward zero
        rather than raising -- see the land-fraction diagnostic below.
    """

    is_synthetic_placeholder: bool = False

    _FREE_URL = "https://marine-api.open-meteo.com/v1/marine"
    _COMMERCIAL_URL = "https://customer-marine-api.open-meteo.com/v1/marine"

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        n_side: int = 4,
        api_key: Optional[str] = None,
    ):
        min_lon, min_lat, max_lon, max_lat = bbox
        lats = np.linspace(min_lat, max_lat, n_side)
        lons = np.linspace(min_lon, max_lon, n_side)
        mesh_lat, mesh_lon = np.meshgrid(lats, lons, indexing="ij")
        flat_lat = mesh_lat.ravel()
        flat_lon = mesh_lon.ravel()

        client = _get_openmeteo_client()
        params = {
            "latitude": [float(v) for v in flat_lat],
            "longitude": [float(v) for v in flat_lon],
            "hourly": ["ocean_current_velocity", "ocean_current_direction"],
            "length_unit": "metric",
            "start_date": start_date,
            "end_date": end_date,
        }
        if api_key:
            url = self._COMMERCIAL_URL
            params["apikey"] = api_key
        else:
            url = self._FREE_URL
        self.used_commercial_key = bool(api_key)

        responses = client.weather_api(url, params=params)
        if len(responses) != len(flat_lat):
            raise ValueError(
                f"Open-Meteo returned {len(responses)} responses for "
                f"{len(flat_lat)} requested grid points -- cannot build a grid."
            )

        u_stack, v_stack, t_ref = [], [], None
        n_all_zero = 0
        for resp in responses:
            hourly = resp.Hourly()
            if hourly is None or hourly.VariablesLength() < 2:
                raise ValueError("Open-Meteo grid response is missing hourly current variables")
            raw_speed = hourly.Variables(0).ValuesAsNumpy().astype(np.float64)
            raw_dir = hourly.Variables(1).ValuesAsNumpy().astype(np.float64)
            if np.all(~np.isfinite(raw_speed)):
                n_all_zero += 1
            speed_kmh = np.nan_to_num(raw_speed, nan=0.0)
            direction_deg = np.nan_to_num(raw_dir, nan=0.0)
            t = np.arange(hourly.Time(), hourly.TimeEnd(), hourly.Interval(), dtype=np.float64)
            n = min(len(t), len(speed_kmh), len(direction_deg))
            if t_ref is None:
                t_ref = t[:n]
            n = min(len(t_ref), n)
            t_ref = t_ref[:n]
            speed_mps = speed_kmh[:n] / 3.6
            rad = np.radians(direction_deg[:n])
            # Open-Meteo's ocean_current_direction is the direction the flow
            # is heading TOWARD (0=N, 90=E) -- plain vector decomposition,
            # no 180-degree flip, matching OpenMeteoCurrentField above.
            u_stack.append(speed_mps * np.sin(rad))
            v_stack.append(speed_mps * np.cos(rad))

        n_t = len(t_ref)
        self._t = t_ref
        self._lats = lats
        self._lons = lons
        self._u = np.stack([a[:n_t] for a in u_stack]).reshape(n_side, n_side, n_t)
        self._v = np.stack([a[:n_t] for a in v_stack]).reshape(n_side, n_side, n_t)
        self.n_side = n_side
        self.grid_bbox = bbox
        self.land_or_null_points = n_all_zero

        # Honest self-diagnostic: how much spatial variation is actually in
        # this grid? Near zero means the API handed back an effectively
        # uniform field anyway, and the ensemble will have little to
        # discriminate on -- the caller surfaces this rather than assuming.
        speed = np.hypot(self._u, self._v)
        self.spatial_speed_range_mps = float(speed.max() - speed.min())
        self.mean_speed_mps = float(speed.mean())

    def _interp_time(self, arr: np.ndarray, epoch_seconds: float) -> np.ndarray:
        """Linear-in-time slice of a (n_side, n_side, n_t) array."""
        t_clamped = min(max(epoch_seconds, self._t[0]), self._t[-1])
        idx = int(np.searchsorted(self._t, t_clamped))
        if idx <= 0:
            return arr[:, :, 0]
        if idx >= len(self._t):
            return arr[:, :, -1]
        t0, t1 = self._t[idx - 1], self._t[idx]
        w = 0.0 if t1 == t0 else (t_clamped - t0) / (t1 - t0)
        return arr[:, :, idx - 1] * (1 - w) + arr[:, :, idx] * w

    def velocity(self, lat: np.ndarray, lon: np.ndarray, epoch_seconds: float):
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        u_grid = self._interp_time(self._u, epoch_seconds)
        v_grid = self._interp_time(self._v, epoch_seconds)

        # Bilinear in space, clamped at the grid edges so a particle that
        # drifts outside the requested bbox picks up the nearest edge value
        # rather than a runaway extrapolation.
        lat_i = np.interp(lat, self._lats, np.arange(len(self._lats), dtype=np.float64))
        lon_j = np.interp(lon, self._lons, np.arange(len(self._lons), dtype=np.float64))
        i0 = np.clip(np.floor(lat_i).astype(int), 0, len(self._lats) - 2)
        j0 = np.clip(np.floor(lon_j).astype(int), 0, len(self._lons) - 2)
        di = lat_i - i0
        dj = lon_j - j0

        def bilerp(g):
            return (
                g[i0, j0] * (1 - di) * (1 - dj)
                + g[i0 + 1, j0] * di * (1 - dj)
                + g[i0, j0 + 1] * (1 - di) * dj
                + g[i0 + 1, j0 + 1] * di * dj
            )

        return bilerp(u_grid), bilerp(v_grid)


# ---------------------------------------------------------------------------
# Dev/self-test path: analytic synthetic field (current side only)
# ---------------------------------------------------------------------------
class SyntheticGyreField:
    """Smooth, mildly time-varying analytic velocity field for self-tests
    and development ONLY -- not fitted to any real ocean/atmosphere data.
    A single stationary gyre (solid-body-like rotation that decays with
    distance from center, amplitude modulated by a slow sinusoid in time)
    gives the integrator and the ABC ensemble something non-trivial and
    spatially-varying to chew on without needing real data on hand.
    """

    is_synthetic_placeholder: bool = False  # get_*_provider() flips this to True
                                             # when it had to fall back to this class

    def __init__(
        self, center_lat: float, center_lon: float,
        speed_mps: float = 0.3, length_scale_km: float = 60.0, period_hours: float = 72.0,
        bg_u_mps: float = 0.0, bg_v_mps: float = 0.0, strain_rate_per_s: float = 0.0,
    ):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.speed_mps = speed_mps
        self.length_scale_km = length_scale_km
        self.period_seconds = period_hours * 3600.0
        # constant background advection (e.g. a mean current/prevailing wind)
        # superimposed on the rotational gyre -- without this, a field
        # centered exactly on the true test origin has ~zero net velocity
        # right where it matters, which makes for a too-easy self-test.
        self.bg_u_mps = bg_u_mps
        self.bg_v_mps = bg_v_mps
        # a simple linear hyperbolic strain (u=+S*dx, v=-S*dy in local
        # meters) -- unlike the rotational gyre term (whose gradient decays
        # with distance from center, so it barely shears a small patch
        # sitting near the core), a strain field's velocity GRADIENT is
        # constant everywhere, so it stretches a patch continuously
        # regardless of the patch's size or position. This is what gives
        # the self-test's forward re-simulation a genuine, monotonic,
        # elapsed-time-dependent area signal to discriminate on (a real
        # oceanographic strain/deformation zone, e.g. near a front or an
        # eddy edge, does exactly this on real slicks -- see e.g. the
        # classic oil-spill-modeling literature on frontal convergence
        # zones). 0.0 (off) by default; only the self-test turns it on.
        self.strain_rate_per_s = strain_rate_per_s

    def velocity(self, lat: np.ndarray, lon: np.ndarray, epoch_seconds: float):
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        coslat = math.cos(math.radians(self.center_lat))
        dy_km = (lat - self.center_lat) * 111.32
        dx_km = (lon - self.center_lon) * 111.32 * coslat
        r_km = np.hypot(dx_km, dy_km)
        r_km_safe = np.where(r_km < 1e-6, 1e-6, r_km)

        # tangential (rotational) unit vector at each point
        tangent_x = -dy_km / r_km_safe
        tangent_y = dx_km / r_km_safe

        envelope = np.exp(-(r_km ** 2) / (2 * self.length_scale_km ** 2))
        time_mod = 0.6 + 0.4 * math.sin(2 * math.pi * epoch_seconds / self.period_seconds)
        speed = self.speed_mps * envelope * time_mod

        u_strain = self.strain_rate_per_s * (dx_km * 1000.0)
        v_strain = -self.strain_rate_per_s * (dy_km * 1000.0)

        u = speed * tangent_x + self.bg_u_mps + u_strain
        v = speed * tangent_y + self.bg_v_mps + v_strain
        return u, v


# ---------------------------------------------------------------------------
# Factory functions used by Module 2
# ---------------------------------------------------------------------------
def get_ocean_current_provider(
    bbox: tuple[float, float, float, float],
    t_start_epoch: float, t_end_epoch: float,
    cache_path: Optional[Path] = None,
    use_openmeteo: bool = True,
    openmeteo_api_key: Optional[str] = None,
    openmeteo_grid_side: int = 4,
) -> VelocityFieldProvider:
    """bbox = (min_lon, min_lat, max_lon, max_lat).

    Priority order:
      1. Local NetCDF cache (cache_path), if given and it loads -- the real
         CMEMS-grid path, for whenever you have one.
      2. Open-Meteo Marine API (use_openmeteo=True, the default) -- free,
         keyless, one HTTP request, no account. See OpenMeteoCurrentField's
         docstring. This is what actually runs for most people out of the
         box: no CMEMS registration needed at all. If openmeteo_api_key is
         set (a paid/commercial Open-Meteo key), requests go through the
         commercial host instead of the free one -- see
         OpenMeteoCurrentField for exactly what that changes.
      3. Synthetic placeholder -- only if both of the above are unavailable
         or fail (no cache configured AND Open-Meteo unreachable, e.g. no
         network). Loudly warns and flags is_synthetic_placeholder=True.
    """
    if cache_path is not None and Path(cache_path).exists():
        try:
            return NetCDFGridField(cache_path, u_var="uo", v_var="vo")
        except Exception as e:
            warnings.warn(
                f"Failed to load cached current NetCDF at {cache_path} "
                f"({type(e).__name__}: {e}); trying Open-Meteo next.",
                stacklevel=2,
            )

    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2

    if use_openmeteo:
        start_date = datetime.fromtimestamp(t_start_epoch, tz=timezone.utc).strftime("%Y-%m-%d")
        # Open-Meteo's end_date is inclusive by day, not by hour -- pad
        # one extra day so the requested t_end_epoch always falls
        # inside the returned hourly series rather than landing on the
        # last, possibly-truncated hour of that day.
        end_date = (datetime.fromtimestamp(t_end_epoch, tz=timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

        # Preferred: a spatial GRID of Open-Meteo points. Same data, same
        # single request, but with a real velocity gradient -- which is what
        # the ABC ensemble in backtrack.py actually discriminates on. See
        # OpenMeteoGriddedCurrentField's docstring for why a spatially
        # uniform field flattens the posterior.
        if openmeteo_grid_side and openmeteo_grid_side >= 2:
            try:
                return OpenMeteoGriddedCurrentField(
                    bbox, start_date, end_date,
                    n_side=int(openmeteo_grid_side), api_key=openmeteo_api_key,
                )
            except Exception as e:
                warnings.warn(
                    f"Gridded Open-Meteo request failed ({type(e).__name__}: {e}); "
                    f"falling back to a single-point Open-Meteo query. The "
                    f"posterior will be much flatter -- see "
                    f"OpenMeteoGriddedCurrentField's docstring.",
                    stacklevel=2,
                )

        try:
            return OpenMeteoCurrentField(center_lat, center_lon, start_date, end_date, api_key=openmeteo_api_key)
        except Exception as e:
            warnings.warn(
                f"Open-Meteo Marine API call failed ({type(e).__name__}: {e}); "
                f"falling back to a SYNTHETIC analytic current field. Check "
                f"network access to marine-api.open-meteo.com, or pass "
                f"--current-cache pointing at a pre-downloaded CMEMS slice.",
                stacklevel=2,
            )

    warnings.warn(
        "No CMEMS current cache and no working Open-Meteo call -- using a "
        "SYNTHETIC analytic current field. Backtrack origin estimates from "
        "this run are NOT physically meaningful.",
        stacklevel=2,
    )
    # bg_u/bg_v/strain below are ARBITRARY illustrative placeholders (loosely
    # "a mild westward Arabian-Sea-ish surface drift"), not derived from any
    # real product -- chosen only so a demo run without any real data source
    # still shows a spill actually drifting/deforming somewhere, instead of
    # an almost-frozen posterior that (correctly, but unhelpfully for a
    # demo) just reflects a rotational-only field with no net translation.
    # Either way is_synthetic_placeholder on the returned object -- checked
    # by run_module2.py and surfaced in the output's _debug block -- is
    # what actually tells a caller "don't trust this run's physics", not the
    # specific numbers below.
    field = SyntheticGyreField(center_lat, center_lon, speed_mps=0.35, length_scale_km=70.0, period_hours=96.0,
                                bg_u_mps=-0.08, bg_v_mps=-0.04, strain_rate_per_s=1e-5)
    field.is_synthetic_placeholder = True
    return field


def get_wind_vector_provider(
    bbox: tuple[float, float, float, float],
    t_start_epoch: float, t_end_epoch: float,
    cache_path: Optional[Path] = None,
    u_mps: float = 0.0,
    v_mps: float = 0.0,
) -> VelocityFieldProvider:
    """bbox = (min_lon, min_lat, max_lon, max_lat).

    Wind is NOT fetched from any API by default. If cache_path points at a
    real, already-downloaded ERA5/GFS NetCDF, that's used (the real path,
    for whenever you want it). Otherwise this returns a plain CONSTANT
    (u_mps, v_mps) -- 0.0/0.0 unless you've set config.wind_u_mps/wind_v_mps
    to a rough manual estimate. No warning is raised for the default
    negligible-wind case: unlike the current-side fallback (where a missing
    cache means "we wanted real CMEMS data and don't have it yet"), a
    negligible/constant wind here is the INTENDED default, not a degraded
    substitute for something missing.
    """
    if cache_path is not None and Path(cache_path).exists():
        try:
            return NetCDFGridField(cache_path, u_var="u10", v_var="v10")
        except Exception as e:
            warnings.warn(
                f"Failed to load cached wind NetCDF at {cache_path} "
                f"({type(e).__name__}: {e}); falling back to the constant "
                f"wind default (u_mps={u_mps}, v_mps={v_mps}).",
                stacklevel=2,
            )

    return ConstantVectorField(u_mps=u_mps, v_mps=v_mps)
