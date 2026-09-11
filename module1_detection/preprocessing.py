"""
preprocessing.py
Inference-time preprocessing for Module 1 (oil detection).

Turns one Sentinel-1 GRD-derived scene into a stack of 256x256 tiles that
match the format train_unet.py trained on (uint8, VV+VH, dB-scaled), plus
enough georeferencing info for postprocessing.py to turn model output back
into real lat/lon polygons.

-----------------------------------------------------------------------------
IMPORTANT SCOPE NOTE (read this before wiring in a real scene)
-----------------------------------------------------------------------------
Full SAR terrain correction / orthorectification (mapping a raw GRD scene's
rows/columns to lat/lon using orbit state vectors + a DEM) is a serious
geodesy problem on its own - getting it subtly wrong means every downstream
lat/lon in the whole pipeline (Module 2's backward drift, Module 3's vessel
match) is confidently wrong with no obvious symptom. That step is NOT
reimplemented here. The expected contract for this module's input is:

    A GeoTIFF that has ALREADY been:
      1. Radiometrically calibrated to sigma0 (linear power)
      2. Terrain-corrected / orthorectified (has a real CRS + affine transform)
    with two bands, in order: [VV, VH]

The standard, well-tested way to produce that from a raw Sentinel-1 .SAFE
product is ESA SNAP's command-line Graph Processing Tool, e.g.:

    gpt Calibration -Ssource=<scene>.SAFE -PoutputSigmaBand=true \
        -PsourceBands=Intensity_VV,Intensity_VH -t calibrated.dim
    gpt Terrain-Correction -Ssource=calibrated.dim \
        -PdemName='SRTM 3Sec' -PmapProjection=WGS84 -t scene_tc.dim
    gpt Write -Ssource=scene_tc.dim -PformatName=GeoTIFF -PfilePath=scene.tif

What THIS module does with that GeoTIFF (all legitimately implementable in
plain Python and verified below):
  - speckle filtering (Lee filter)
  - land/sea masking (vector polygon rasterization)
  - sigma0 -> dB conversion
  - uint8 normalization matching the training data's scale
  - sliding-window tiling to 256x256 with georeferencing carried per-tile

If you'd rather do calibration in Python instead of SNAP (e.g. you already
have digital numbers + a calibration LUT extracted from the product's
calibration XML), `calibrate_sigma0()` below implements the real Sentinel-1
formula for that case.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter

try:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import Affine, xy
    from rasterio.windows import Window
    from rasterio.warp import transform_bounds
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "preprocessing.py needs rasterio (`pip install rasterio`). "
        "It's the standard library for reading georeferenced SAR GeoTIFFs."
    ) from e


# ---------------------------------------------------------------------------
# Scene metadata
# ---------------------------------------------------------------------------
@dataclass
class SceneMetadata:
    """Fields the spec requires in the final output schema. Deliberately NOT
    guessed/faked if missing -- see read_scene()."""
    timestamp_utc: str                 # exact acquisition time, e.g. "2017-03-12T06:00:00Z"
    satellite_pass_id: str              # e.g. product identifier / orbit number
    incidence_angle: Optional[float] = None
    extra: dict = field(default_factory=dict)


def _read_sidecar_metadata(scene_path: Path) -> dict:
    """Looks for a `<scene>.json` next to the GeoTIFF with acquisition metadata.
    Expected keys: timestamp_utc, satellite_pass_id, incidence_angle (optional).
    This is the recommended way to carry metadata through the SNAP export step,
    since GeoTIFF tags are not a reliable place to stash it."""
    sidecar = scene_path.with_suffix(".json")
    if sidecar.exists():
        with open(sidecar) as f:
            return json.load(f)
    return {}


def resolve_scene_metadata(
    scene_path: Path,
    gdal_tags: dict,
    override_timestamp: Optional[str] = None,
    override_pass_id: Optional[str] = None,
    override_incidence_angle: Optional[float] = None,
) -> SceneMetadata:
    """Combines (in priority order) CLI overrides > sidecar JSON > GDAL tags.
    Raises rather than fabricating a timestamp/pass id, since a silently wrong
    "now()" timestamp is worse than a loud failure for a system whose output
    feeds a drift-backtracking model."""
    sidecar = _read_sidecar_metadata(scene_path)

    timestamp = override_timestamp or sidecar.get("timestamp_utc") or gdal_tags.get("TIFFTAG_DATETIME")
    pass_id = override_pass_id or sidecar.get("satellite_pass_id") or gdal_tags.get("product_id")
    incidence = override_incidence_angle
    if incidence is None:
        incidence = sidecar.get("incidence_angle")
    if incidence is None and "incidence_angle" in gdal_tags:
        incidence = float(gdal_tags["incidence_angle"])

    missing = [name for name, val in (("timestamp_utc", timestamp), ("satellite_pass_id", pass_id)) if not val]
    if missing:
        raise ValueError(
            f"Missing required scene metadata: {missing}. Provide a "
            f"'{scene_path.stem}.json' sidecar file (keys: timestamp_utc, "
            f"satellite_pass_id, incidence_angle) or pass --timestamp/--pass-id "
            f"on the command line. Refusing to guess these -- they end up in "
            f"the detection's timestamp_utc field, and Module 2 depends on it "
            f"being exact."
        )

    return SceneMetadata(timestamp_utc=str(timestamp), satellite_pass_id=str(pass_id),
                          incidence_angle=incidence, extra=gdal_tags)


# ---------------------------------------------------------------------------
# Reading the scene
# ---------------------------------------------------------------------------
def read_scene(scene_path: Path):
    """Reads a 2-band (VV, VH) calibrated + terrain-corrected sigma0 GeoTIFF.

    Returns:
        vv, vh: (H, W) float32 arrays, linear-power sigma0
        transform: rasterio Affine (pixel -> geo coords)
        crs: rasterio CRS
        gdal_tags: dict of any embedded tags (best-effort metadata source)
    """
    scene_path = Path(scene_path)
    with rasterio.open(scene_path) as src:
        if src.count < 2:
            raise ValueError(
                f"Expected a 2-band VV+VH GeoTIFF, got {src.count} band(s) in {scene_path}. "
                f"Make sure the SNAP export kept both polarizations."
            )
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)
        transform = src.transform
        crs = src.crs
        tags = dict(src.tags())
    return vv, vh, transform, crs, tags


# ---------------------------------------------------------------------------
# Radiometric calibration (only needed if you're starting from raw DN
# instead of a SNAP-calibrated product)
# ---------------------------------------------------------------------------
def calibrate_sigma0(digital_numbers: np.ndarray, calibration_lut: np.ndarray) -> np.ndarray:
    """Sentinel-1 radiometric calibration: sigma0 = DN^2 / A^2, where A is the
    calibration constant interpolated onto the full image grid from the
    product's calibration vectors (per ESA's official formula).

    Args:
        digital_numbers: (H, W) raw amplitude/DN array from the product.
        calibration_lut: (H, W) calibration constant A, already resampled to
            the same grid as digital_numbers (bilinear-interpolate the sparse
            calibration vectors from the product's calibration XML onto the
            full grid before calling this -- that resampling step is product-
            format-specific and is intentionally left to the caller).

    Returns:
        (H, W) float32 sigma0 in linear power.
    """
    if digital_numbers.shape != calibration_lut.shape:
        raise ValueError("digital_numbers and calibration_lut must be the same shape")
    dn = digital_numbers.astype(np.float64)
    a = calibration_lut.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma0 = (dn ** 2) / (a ** 2)
    sigma0 = np.nan_to_num(sigma0, nan=0.0, posinf=0.0, neginf=0.0)
    return sigma0.astype(np.float32)


# ---------------------------------------------------------------------------
# Speckle filtering
# ---------------------------------------------------------------------------
def lee_filter(img: np.ndarray, window: int = 7, noise_var: Optional[float] = None) -> np.ndarray:
    """Classic Lee speckle filter, applied per-band in linear power domain
    (call this BEFORE to_db(), not after -- speckle is multiplicative noise
    and the Lee filter's noise model assumes linear power).

    out = mean + W * (pixel - mean),   W = var_local / (var_local + noise_var * mean_local^2)

    Args:
        img: (H, W) linear-power array (e.g. sigma0).
        window: filter window size (odd int).
        noise_var: speckle noise variance (coefficient of variation squared).
            Defaults to a standard single-look-equivalent estimate; override
            if you know the product's equivalent number of looks (ENL) --
            noise_var ~= 1/ENL.
    """
    if window % 2 == 0:
        raise ValueError("window must be odd")
    if noise_var is None:
        noise_var = 0.25  # ~ENL=4, typical for Sentinel-1 IW GRD multi-looked products

    img = img.astype(np.float64)
    local_mean = uniform_filter(img, size=window)
    local_sq_mean = uniform_filter(img * img, size=window)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)

    denom = local_var + noise_var * (local_mean ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(denom > 0, local_var / denom, 0.0)
    weight = np.clip(weight, 0.0, 1.0)

    filtered = local_mean + weight * (img - local_mean)
    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# Land / sea masking
# ---------------------------------------------------------------------------
def land_sea_mask(
    transform: "Affine",
    crs,
    height: int,
    width: int,
    land_shapefile: Optional[Path] = None,
) -> np.ndarray:
    """Returns a boolean array, True = water (keep), False = land (exclude).

    land_shapefile should be a vector layer of land polygons (e.g. Natural
    Earth's ne_10m_land, or a national coastline product) in any CRS -- it
    gets reprojected to the scene's CRS automatically.

    If no shapefile is available, falls back to an all-water mask and prints
    a loud warning, since silently skipping land masking risks land-shadow
    false positives (per the spec) -- better to fail visibly in a log than
    silently ship a possibly-contaminated detection.
    """
    if land_shapefile is None or not Path(land_shapefile).exists():
        warnings.warn(
            "No land_shapefile provided/found -- land/sea masking is DISABLED "
            "for this run. Land-shadow false positives are possible. Pass "
            "--land-shapefile pointing at a coastline polygon layer "
            "(e.g. Natural Earth ne_10m_land) to enable it.",
            stacklevel=2,
        )
        return np.ones((height, width), dtype=bool)

    import geopandas as gpd  # local import: only needed on this path

    land = gpd.read_file(land_shapefile)
    if land.crs is not None and str(land.crs) != str(crs):
        land = land.to_crs(crs)

    land_raster = np.asarray(rasterize(
        [(geom, 1) for geom in land.geometry if geom is not None],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    ), dtype=np.uint8)
    return np.asarray(land_raster == 0, dtype=bool)  # True where NOT land, i.e. water


# ---------------------------------------------------------------------------
# dB conversion + normalization
# ---------------------------------------------------------------------------
def to_db(sigma0_linear: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """10*log10(sigma0). Standard SAR practice: oil/water contrast is closer
    to linearly separable in dB space than in linear power."""
    return (10.0 * np.log10(np.clip(sigma0_linear, eps, None))).astype(np.float32)


def normalize_to_uint8(db_img: np.ndarray, db_min: float = -30.0, db_max: float = 0.0) -> np.ndarray:
    """Rescales a dB image into uint8 [0, 255].

    !! CRITICAL !!: db_min/db_max MUST match whatever range was used when
    merge_sar_oilspill_datasets.py built merged_dataset.npz. If they don't
    match, the trained U-Net sees inference-time inputs on a different scale
    than it was trained on and quality will silently degrade. -30..0 dB is a
    typical Sentinel-1 ocean sigma0 range and is a reasonable default, but
    verify it against your actual training preprocessing before trusting
    results.
    """
    clipped = np.clip(db_img, db_min, db_max)
    scaled = (clipped - db_min) / (db_max - db_min)  # -> [0, 1]
    return (scaled * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------
@dataclass
class Tile:
    image: np.ndarray          # (256, 256, 2) uint8, channel order [VV, VH]
    land_mask: np.ndarray       # (256, 256) bool, True = water
    transform: "Affine"         # pixel->geo transform for THIS tile
    row_off: int
    col_off: int
    valid_h: int                # unpadded height (edge tiles may be padded)
    valid_w: int


def tile_scene(
    vv_u8: np.ndarray,
    vh_u8: np.ndarray,
    land_mask: np.ndarray,
    transform: "Affine",
    tile_size: int = 256,
    stride: int = 224,
) -> list[Tile]:
    """Sliding-window tiling with overlap. Overlap (stride < tile_size) lets
    postprocessing.stitch_tiles() average seams instead of leaving hard tile
    boundaries in the reassembled mask. Edge tiles that run past the scene
    are zero-padded and their valid_h/valid_w record the real extent so
    padding never leaks into a detection polygon."""
    if not (0 < stride <= tile_size):
        raise ValueError("stride must be in (0, tile_size]")

    h, w = vv_u8.shape
    tiles: list[Tile] = []

    row_starts = list(range(0, max(h - tile_size, 0) + 1, stride))
    if not row_starts or row_starts[-1] + tile_size < h:
        row_starts.append(max(h - tile_size, 0))
    col_starts = list(range(0, max(w - tile_size, 0) + 1, stride))
    if not col_starts or col_starts[-1] + tile_size < w:
        col_starts.append(max(w - tile_size, 0))

    for r0 in sorted(set(row_starts)):
        for c0 in sorted(set(col_starts)):
            r1, c1 = min(r0 + tile_size, h), min(c0 + tile_size, w)
            valid_h, valid_w = r1 - r0, c1 - c0

            img = np.zeros((tile_size, tile_size, 2), dtype=np.uint8)
            img[:valid_h, :valid_w, 0] = vv_u8[r0:r1, c0:c1]
            img[:valid_h, :valid_w, 1] = vh_u8[r0:r1, c0:c1]

            lmask = np.zeros((tile_size, tile_size), dtype=bool)
            lmask[:valid_h, :valid_w] = land_mask[r0:r1, c0:c1]

            tile_transform = transform * Affine.translation(c0, r0)

            tiles.append(Tile(
                image=img, land_mask=lmask, transform=tile_transform,
                row_off=r0, col_off=c0, valid_h=valid_h, valid_w=valid_w,
            ))
    return tiles


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
@dataclass
class PreprocessedScene:
    tiles: list[Tile]
    full_shape: tuple[int, int]
    transform: "Affine"
    crs: object
    land_mask: np.ndarray
    vv_db: np.ndarray            # kept for postprocessing (mean_backscatter_db, VH/VV ratio)
    vh_db: np.ndarray
    metadata: SceneMetadata


def preprocess_scene(
    scene_path: Path,
    land_shapefile: Optional[Path] = None,
    db_min: float = -30.0,
    db_max: float = 0.0,
    tile_size: int = 256,
    stride: int = 224,
    apply_speckle_filter: bool = True,
    speckle_window: int = 7,
    override_timestamp: Optional[str] = None,
    override_pass_id: Optional[str] = None,
    override_incidence_angle: Optional[float] = None,
) -> PreprocessedScene:
    """Runs the full inference-time preprocessing chain on one scene."""
    scene_path = Path(scene_path)
    vv, vh, transform, crs, tags = read_scene(scene_path)

    metadata = resolve_scene_metadata(
        scene_path, tags,
        override_timestamp=override_timestamp,
        override_pass_id=override_pass_id,
        override_incidence_angle=override_incidence_angle,
    )

    if apply_speckle_filter:
        vv = lee_filter(vv, window=speckle_window)
        vh = lee_filter(vh, window=speckle_window)

    water_mask = land_sea_mask(transform, crs, vv.shape[0], vv.shape[1], land_shapefile)

    vv_db = to_db(vv)
    vh_db = to_db(vh)

    vv_u8 = normalize_to_uint8(vv_db, db_min, db_max)
    vh_u8 = normalize_to_uint8(vh_db, db_min, db_max)

    tiles = tile_scene(vv_u8, vh_u8, water_mask, transform, tile_size, stride)

    return PreprocessedScene(
        tiles=tiles,
        full_shape=vv.shape,
        transform=transform,
        crs=crs,
        land_mask=water_mask,
        vv_db=vv_db,
        vh_db=vh_db,
        metadata=metadata,
    )


if __name__ == "__main__":
    # Quick self-test on a synthetic scene -- generates a fake calibrated
    # GeoTIFF and runs it through the full chain. Run: python preprocessing.py
    import tempfile

    print("[self-test] building synthetic calibrated scene...")
    h, w = 600, 500
    rng = np.random.default_rng(0)
    vv = np.clip(rng.normal(0.02, 0.01, size=(h, w)), 1e-5, None).astype(np.float32)
    vh = np.clip(rng.normal(0.01, 0.006, size=(h, w)), 1e-5, None).astype(np.float32)
    vv[200:350, 150:400] *= 0.15  # synthetic dark slick

    with tempfile.TemporaryDirectory() as tmp:
        tif_path = Path(tmp) / "synthetic_scene.tif"
        sidecar_path = Path(tmp) / "synthetic_scene.json"
        transform = Affine.translation(-40.5, 43.0) * Affine.scale(0.0001, -0.0001)

        with rasterio.open(
            tif_path, "w", driver="GTiff", height=h, width=w, count=2,
            dtype="float32", crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(vv, 1)
            dst.write(vh, 2)

        with open(sidecar_path, "w") as f:
            json.dump({
                "timestamp_utc": "2017-03-12T06:00:00Z",
                "satellite_pass_id": "S1A_IW_GRDH_TEST",
                "incidence_angle": 34.2,
            }, f)

        scene = preprocess_scene(tif_path)
        print(f"[self-test] scene shape: {scene.full_shape}")
        print(f"[self-test] tiles generated: {len(scene.tiles)}")
        print(f"[self-test] metadata: {scene.metadata}")
        print(f"[self-test] example tile image shape: {scene.tiles[0].image.shape}, "
              f"dtype: {scene.tiles[0].image.dtype}")
        assert scene.tiles[0].image.shape == (256, 256, 2)
        assert scene.tiles[0].image.dtype == np.uint8
        print("[self-test] OK")
