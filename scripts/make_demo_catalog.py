"""
make_demo_catalog.py
----------------------------------------------------------------------------
Builds data/demo_images/catalog.json -- the curated scene catalog the demo
picker reads.

THE PROBLEM THIS SOLVES
----------------------------------------------------------------------------
The Kaggle SAR tiles carry no CRS, no geotransform and no acquisition time.
The build spec's answer is to register each demo scene with a bbox and a
timestamp by hand, and map the U-Net's pixel-space output into real
coordinates by linear interpolation.

Doing that by hand is fiddly: you pick a bbox, run detection, discover the
slick lands 40 km from your vessel dataset, and adjust. So this script
inverts it. You state where the scene's slick is supposed to be -- a target
lat/lon in the Gulf of Mexico near the AIS vessels -- and the script:

  1. runs the trained U-Net on the tile,
  2. finds the primary component's centroid in normalised (u, v) pixel
     coordinates,
  3. solves for the bbox of the requested angular size that puts that
     centroid exactly on the target lat/lon,
  4. writes the catalog entry.

The bbox is demo metadata either way -- these tiles have no true geolocation
to preserve. This just makes the assignment reproducible and guarantees the
detection, the drift posterior and the vessel dataset all live in the same
patch of ocean. Drop your own PNGs into data/demo_images/, add an entry to
SCENES below, and re-run.

Usage:
    python scripts/make_demo_catalog.py
    python scripts/make_demo_catalog.py --rebuild-variants
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "module1_detection"))

DEMO_DIR = PROJECT_ROOT / "data" / "demo_images"
CATALOG_PATH = DEMO_DIR / "catalog.json"
BASE_IMAGE = DEMO_DIR / "scene_01_delta.png"

# Scene span in degrees. At 28.4 N one degree of longitude is ~97.8 km and
# one degree of latitude ~110.9 km, so this is roughly a 15 x 15 km tile --
# a plausible Sentinel-1 sub-scene, and small enough that a detected slick
# comes out at a believable tens-of-km2 rather than hundreds.
SPAN_LON_DEG = 0.153
SPAN_LAT_DEG = 0.135

# Each scene states where its slick is supposed to sit. All four are inside
# the Mississippi Canyon / Mars platform area covered by data/ships/ships_seed.json.
SCENES = [
    {
        "_id": "demo_img_01",
        "filename": "scene_01_delta.png",
        "title": "Mississippi Canyon",
        "subtitle": "Sentinel-1 VV, descending pass",
        "variant": None,
        "target_lat": 28.310,
        "target_lon": -89.020,
        "acquisition_timestamp": "2026-09-04T06:12:00Z",
    },
    {
        "_id": "demo_img_02",
        "filename": "scene_02_marsridge.png",
        "title": "Mars Ridge",
        "subtitle": "Sentinel-1 VV, descending pass",
        "variant": "rot90",
        "target_lat": 28.455,
        "target_lon": -89.145,
        "acquisition_timestamp": "2026-09-04T05:30:00Z",
    },
    {
        "_id": "demo_img_03",
        "filename": "scene_03_shelfedge.png",
        "title": "Shelf Edge",
        "subtitle": "Sentinel-1 VV, ascending pass",
        "variant": "fliplr",
        "target_lat": 28.235,
        "target_lon": -89.190,
        "acquisition_timestamp": "2026-09-04T04:45:00Z",
    },
    {
        "_id": "demo_img_04",
        "filename": "scene_04_westflank.png",
        "title": "West Flank",
        "subtitle": "Sentinel-1 VV, descending pass",
        "variant": "rot270_zoom",
        "target_lat": 28.575,
        "target_lon": -89.255,
        "acquisition_timestamp": "2026-09-04T06:40:00Z",
    },
]


def build_variants(force: bool = False) -> None:
    """Derives the extra demo tiles from the base scene by geometric
    transforms. These are the SAME real SAR texture, re-oriented -- not
    synthetic noise -- so the U-Net is still segmenting genuine slick
    signatures rather than something invented for the demo. Swap in real
    Kaggle tiles whenever you have them; only the filename in SCENES needs
    to change."""
    if not BASE_IMAGE.exists():
        raise FileNotFoundError(f"Base scene missing: {BASE_IMAGE}")
    base = Image.open(BASE_IMAGE).convert("L")

    ops = {
        "rot90": lambda im: im.transpose(Image.ROTATE_90),
        "fliplr": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
        "rot270_zoom": lambda im: im.transpose(Image.ROTATE_270).crop(
            (int(im.width * 0.12), int(im.height * 0.12),
             int(im.width * 0.92), int(im.height * 0.92))
        ).resize(base.size, Image.LANCZOS),
    }

    for scene in SCENES:
        if scene["variant"] is None:
            continue
        out = DEMO_DIR / scene["filename"]
        if out.exists() and not force:
            print(f"  keep   {out.name}")
            continue
        ops[scene["variant"]](base).save(out)
        print(f"  write  {out.name}  ({scene['variant']} of {BASE_IMAGE.name})")


def solve_bbox(u: float, v: float, target_lat: float, target_lon: float) -> dict:
    """Returns the bbox of size SPAN_LON_DEG x SPAN_LAT_DEG whose linear
    pixel mapping sends normalised pixel (u, v) to (target_lat, target_lon).

    pixel_to_latlon uses:  lon = tl.lon + u * span_lon
                           lat = tl.lat - v * span_lat
    so inverting for tl is a one-liner each way.
    """
    tl_lon = target_lon - u * SPAN_LON_DEG
    tl_lat = target_lat + v * SPAN_LAT_DEG
    return {
        "top_left": {"lat": round(tl_lat, 6), "lon": round(tl_lon, 6)},
        "bottom_right": {
            "lat": round(tl_lat - SPAN_LAT_DEG, 6),
            "lon": round(tl_lon + SPAN_LON_DEG, 6),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-variants", action="store_true",
                    help="Regenerate the derived scene tiles even if they already exist")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    print("[1/3] preparing demo tiles")
    build_variants(force=args.rebuild_variants)

    print("[2/3] locating each scene's slick with the trained U-Net")
    from png_inference import PngSpillDetector

    detector = PngSpillDetector()
    entries = []
    for scene in SCENES:
        path = DEMO_DIR / scene["filename"]
        if not path.exists():
            print(f"  SKIP  {scene['filename']} (file not found)")
            continue
        u, v = detector.centroid_uv(path, threshold=args.threshold)
        bbox = solve_bbox(u, v, scene["target_lat"], scene["target_lon"])
        entries.append({
            "_id": scene["_id"],
            "filename": scene["filename"],
            "file_path": f"data/demo_images/{scene['filename']}",
            "title": scene["title"],
            "subtitle": scene["subtitle"],
            "acquisition_timestamp": scene["acquisition_timestamp"],
            "bbox": bbox,
            "_provenance": {
                "derived_from": BASE_IMAGE.name if scene["variant"] else None,
                "transform": scene["variant"],
                "centroid_uv": [round(u, 5), round(v, 5)],
                "bbox_fitted_to": {"lat": scene["target_lat"], "lon": scene["target_lon"]},
                "span_deg": [SPAN_LON_DEG, SPAN_LAT_DEG],
                "note": (
                    "These tiles carry no true geolocation. The bbox is demo "
                    "metadata, fitted so the detected slick lands on the stated "
                    "target coordinate near the AIS vessel dataset."
                ),
            },
        })
        print(f"  {scene['_id']}  centroid_uv=({u:.3f}, {v:.3f})  "
              f"-> slick at ({scene['target_lat']}, {scene['target_lon']})")

    print(f"[3/3] writing {CATALOG_PATH.relative_to(PROJECT_ROOT)}")
    CATALOG_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"[done] {len(entries)} scene(s) in the catalog")


if __name__ == "__main__":
    main()
