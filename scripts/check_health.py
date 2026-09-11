"""
check_health.py
----------------------------------------------------------------------------
One command that tells you what is and is not working. Run it before a demo.

    python scripts/check_health.py

It checks, in the order things tend to break:
  1. Python packages
  2. MongoDB, and whether the collections are seeded
  3. The trained U-Net checkpoint
  4. Live reachability of the Open-Meteo Marine API -- the one dependency
     that fails silently and degrades the drift model to a synthetic field
  5. The four services, if they are running

Nothing here writes to the database.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "
issues: list[str] = []


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label}" + (f"  —  {detail}" if detail else ""))


# ---------------------------------------------------------------------------
print("\n=== 1. Python packages ===")
REQUIRED = [
    ("torch", "pip install torch --index-url https://download.pytorch.org/whl/cpu"),
    ("cv2", "pip install opencv-python"),
    ("numpy", "pip install numpy"),
    ("PIL", "pip install Pillow"),
    ("shapely", "pip install shapely"),
    ("pyproj", "pip install pyproj"),
    ("scipy", "pip install scipy"),
    ("fastapi", "pip install fastapi"),
    ("uvicorn", "pip install 'uvicorn[standard]'"),
    ("pymongo", "pip install pymongo"),
    ("dotenv", "pip install python-dotenv"),
    ("openmeteo_requests", "pip install openmeteo-requests requests-cache retry-requests"),
]
for module, fix in REQUIRED:
    try:
        importlib.import_module(module)
        line(OK, module)
    except ImportError:
        line(BAD, module, fix)
        issues.append(f"missing package: {module}")

# ---------------------------------------------------------------------------
print("\n=== 2. MongoDB ===")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "oilspill")
try:
    from pymongo import MongoClient

    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=4000)
    client.admin.command("ping")
    db = client[MONGODB_DB]
    line(OK, "connection", MONGODB_URI)

    n_scenes = db.demo_images.count_documents({})
    n_ships = db.ships.count_documents({})
    n_events = db.spill_events.count_documents({})

    if n_scenes:
        line(OK, "demo_images", f"{n_scenes} scene(s)")
    else:
        line(BAD, "demo_images", "empty — run: python scripts/seed_db.py")
        issues.append("demo_images collection is empty")

    if n_ships:
        line(OK, "ships", f"{n_ships} vessel(s)")
    else:
        line(BAD, "ships", "empty — run: python scripts/seed_db.py")
        issues.append("ships collection is empty")

    line(OK, "spill_events", f"{n_events} previous run(s)")

    for scene in db.demo_images.find({}):
        tile = PROJECT_ROOT / scene["file_path"]
        if tile.exists():
            line(OK, f"tile {scene['_id']}", scene["filename"])
        else:
            line(BAD, f"tile {scene['_id']}", f"missing on disk: {tile}")
            issues.append(f"missing tile for {scene['_id']}")
except Exception as e:
    line(BAD, "connection", f"{type(e).__name__}: {e}")
    print("       Start MongoDB, then re-run. On Windows:  net start MongoDB")
    issues.append("MongoDB unreachable")

# ---------------------------------------------------------------------------
print("\n=== 3. Trained model ===")
weights = PROJECT_ROOT / "module1_detection" / "weights" / "unet_best.pt"
if not weights.exists():
    line(BAD, "unet_best.pt", f"not found at {weights}")
    issues.append("U-Net checkpoint missing")
else:
    size_mb = weights.stat().st_size / 1e6
    try:
        import torch

        ckpt = torch.load(weights, map_location="cpu")
        val_iou = ckpt.get("val_iou")
        line(
            OK,
            "unet_best.pt",
            f"{size_mb:.0f} MB, epoch {ckpt.get('epoch')}, val IoU "
            f"{val_iou:.4f}" if val_iou is not None else f"{size_mb:.0f} MB",
        )
    except Exception as e:
        line(BAD, "unet_best.pt", f"loads but not readable: {e}")
        issues.append("U-Net checkpoint unreadable")

# ---------------------------------------------------------------------------
print("\n=== 4. Ocean current data (Open-Meteo Marine) ===")
print("      This is the one that quietly ruins a demo: if the API is")
print("      unreachable the drift model falls back to a synthetic field and")
print("      the origin estimate stops being physically meaningful.")
try:
    from shared_apis.drift_forcing_client import get_ocean_current_provider

    bbox = (-89.9, 27.6, -88.4, 29.2)
    import time as _time

    now = _time.time()
    provider = get_ocean_current_provider(
        bbox, now - 48 * 3600, now,
        cache_path=None, use_openmeteo=True,
        openmeteo_api_key=os.getenv("OPENMETEO_API_KEY") or None,
        openmeteo_grid_side=int(os.getenv("OPENMETEO_GRID_SIDE", "4")),
    )
    name = type(provider).__name__
    if getattr(provider, "is_synthetic_placeholder", False):
        line(BAD, name, "the API call failed — check your internet connection")
        issues.append("Open-Meteo unreachable; drift results would be synthetic")
    elif name == "OpenMeteoGriddedCurrentField":
        line(
            OK, name,
            f"{provider.n_side}x{provider.n_side} grid, mean speed "
            f"{provider.mean_speed_mps:.3f} m/s, spatial range "
            f"{provider.spatial_speed_range_mps:.3f} m/s",
        )
        if provider.spatial_speed_range_mps < 0.02:
            line(WARN, "current gradient",
                 "almost uniform — release times will be hard to tell apart")
        if provider.land_or_null_points:
            line(WARN, "grid coverage",
                 f"{provider.land_or_null_points} grid point(s) returned no data (likely land)")
    else:
        line(WARN, name, "single-point field — the posterior will be flatter than it should be")
except Exception as e:
    line(BAD, "Open-Meteo", f"{type(e).__name__}: {e}")
    issues.append("Open-Meteo check failed")

# ---------------------------------------------------------------------------
print("\n=== 5. Services ===")
try:
    import urllib.error
    import urllib.request
    import json as _json

    def probe(name: str, url: str) -> None:
        try:
            with urllib.request.urlopen(url, timeout=4) as r:
                body = _json.loads(r.read())
            line(OK, name, body.get("status", "responding"))
        except urllib.error.HTTPError as e:
            line(WARN, name, f"HTTP {e.code} — running but degraded")
        except Exception:
            line(WARN, name, "not running (start it before the demo)")

    probe("detection :8001", "http://127.0.0.1:8001/health")
    probe("backtrack :8002", "http://127.0.0.1:8002/health")
    probe("ais       :8003", "http://127.0.0.1:8003/health")
    probe("gateway   :4000", "http://127.0.0.1:4000/api/health")
except Exception as e:  # pragma: no cover
    line(WARN, "service probe", str(e))

# ---------------------------------------------------------------------------
print()
if issues:
    print(f"{len(issues)} thing(s) need fixing before the demo:")
    for i in issues:
        print(f"  - {i}")
    sys.exit(1)
print("Everything the pipeline needs is in place.")
