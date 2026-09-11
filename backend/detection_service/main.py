"""
detection_service/main.py
----------------------------------------------------------------------------
FastAPI wrapper around Module 1's trained U-Net.

The model is loaded once at startup and held warm, so a request costs one
forward pass and some OpenCV contour work rather than a 31 MB checkpoint
read. Inference itself runs through module1_detection/png_inference.py --
see that file's docstring for why the curated PNG catalog takes a different
route into the same weights than run_module1.py's GeoTIFF pipeline does.

Endpoints:
    GET  /scenes            the curated catalog, for the scene picker
    GET  /scenes/{id}/image the tile itself, for the thumbnail
    POST /detect            run the U-Net on one catalogued scene
    GET  /health
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pymongo import MongoClient

SERVICE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SERVICE_ROOT.parent.parent
for _p in (PROJECT_ROOT, PROJECT_ROOT / "module1_detection"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

load_dotenv(PROJECT_ROOT / ".env")

from png_inference import PngSpillDetector  # noqa: E402

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "oilspill")
DETECT_THRESHOLD = float(os.getenv("DETECT_THRESHOLD", "0.5"))

app = FastAPI(title="Spill Detection Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=4000)
db = _client[MONGODB_DB]
_detector: Optional[PngSpillDetector] = None


def get_detector() -> PngSpillDetector:
    """Loads the checkpoint on first use and keeps it. The startup hook below
    warms this so the first real request is not the one that pays the load
    cost, but going through here means a request that somehow arrives before
    startup finished still works instead of returning a 503."""
    global _detector
    if _detector is None:
        started = time.time()
        _detector = PngSpillDetector()
        print(f"[detection] U-Net loaded from {_detector.weights_path} "
              f"(epoch {_detector.checkpoint_epoch}, val_iou "
              f"{_detector.checkpoint_val_iou:.4f}) on {_detector.device} "
              f"in {time.time() - started:.1f}s")
    return _detector


@app.on_event("startup")
def _warm_model() -> None:
    get_detector()


def _scene_or_404(image_id: str) -> dict:
    scene = db.demo_images.find_one({"_id": image_id})
    if scene is None:
        known = [d["_id"] for d in db.demo_images.find({}, {"_id": 1})]
        raise HTTPException(
            404,
            f"No scene {image_id!r} in the catalog. Known scenes: {known or 'none — run scripts/seed_db.py'}",
        )
    return scene


class DetectRequest(BaseModel):
    image_id: str
    threshold: Optional[float] = None


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        n_scenes = db.demo_images.count_documents({})
    except Exception as e:
        return {"status": "degraded", "service": "detection", "mongo": False, "error": str(e)}
    return {
        "status": "ok",
        "service": "detection",
        "mongo": True,
        "scenes_in_catalog": n_scenes,
        "model_loaded": _detector is not None,
        "device": str(_detector.device) if _detector else None,
        "checkpoint_val_iou": _detector.checkpoint_val_iou if _detector else None,
        "threshold": DETECT_THRESHOLD,
    }


@app.get("/scenes")
def scenes() -> list[dict]:
    out = []
    for s in db.demo_images.find({}):
        out.append({
            "image_id": s["_id"],
            "title": s.get("title", s["_id"]),
            "subtitle": s.get("subtitle", ""),
            "filename": s.get("filename"),
            "acquisition_timestamp": s.get("acquisition_timestamp"),
            "bbox": s.get("bbox"),
            "thumbnail_url": f"/scenes/{s['_id']}/image",
        })
    return out


@app.get("/scenes/{image_id}/image")
def scene_image(image_id: str):
    scene = _scene_or_404(image_id)
    path = PROJECT_ROOT / scene["file_path"]
    if not path.exists():
        raise HTTPException(404, f"Tile missing on disk: {path}")
    return FileResponse(path, media_type="image/png")


@app.post("/detect")
def detect(req: DetectRequest) -> dict[str, Any]:
    try:
        detector = get_detector()
    except Exception as e:
        raise HTTPException(503, f"Could not load the U-Net checkpoint: {e}") from e

    scene = _scene_or_404(req.image_id)
    path = PROJECT_ROOT / scene["file_path"]
    if not path.exists():
        raise HTTPException(404, f"Tile missing on disk: {path}")

    detection_id = f"spill_evt_{uuid.uuid4().hex[:10]}"
    started = time.time()
    try:
        record = detector.detect(
            image_path=path,
            bbox=scene["bbox"],
            timestamp_utc=scene["acquisition_timestamp"],
            detection_id=detection_id,
            threshold=req.threshold if req.threshold is not None else DETECT_THRESHOLD,
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"Detection failed: {type(e).__name__}: {e}") from e

    record["_debug"]["runtime_seconds"] = round(time.time() - started, 3)
    record["image_id"] = req.image_id
    record["scene_title"] = scene.get("title")
    record["scene_bbox"] = scene["bbox"]

    # One document per pipeline run; the gateway adds the drift and vessel
    # results onto this same document as later stages finish.
    db.spill_events.update_one(
        {"_id": detection_id},
        {"$set": {**record, "_id": detection_id, "created_at": time.time()}},
        upsert=True,
    )
    return record


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("DETECTION_PORT", "8001")))
