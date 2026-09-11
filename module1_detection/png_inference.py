"""
png_inference.py
----------------------------------------------------------------------------
A SECOND, lighter entry point into Module 1's already-trained U-Net, for the
curated-PNG demo catalog.

WHY THIS FILE EXISTS (and why it doesn't touch run_module1.py)
----------------------------------------------------------------------------
run_module1.py is the *real* operational path: it expects a radiometrically
calibrated, terrain-corrected 2-band (VV, VH) GeoTIFF plus a metadata
sidecar, and it runs land masking, Lee speckle filtering, a lookalike
XGBoost gate, and ERA5 wind lookup around the model. That's correct for real
Sentinel-1 scenes and it stays untouched.

The demo catalog is different by construction: the Kaggle SAR tiles are
single-channel 8-bit PNGs with no CRS, no geotransform, no acquisition
metadata, and no second polarisation. Feeding them through preprocess_scene()
would mean faking a GeoTIFF, faking a calibration LUT, and faking a sidecar
just to get back to the same 256x256 uint8 array the U-Net was trained on --
so this module goes straight there instead:

    PNG -> grayscale -> resize to the model's tile size -> duplicate into two
    channels (the training arrays were (N,256,256,2) uint8 scaled by /255,
    see train_unet.py's SARSegDataset) -> U-Net -> sigmoid -> threshold ->
    largest connected component -> cv2 contour -> approxPolyDP -> pixel
    coords mapped into the scene's registered lat/lon bbox by linear
    interpolation.

The weights, the architecture, and the normalisation convention are exactly
run_module1.py's. Only the pre/post-processing around them is simplified to
match what the demo inputs actually are.

Duplicating VV into the VH channel is a real approximation, stated plainly:
the network was trained on genuine dual-pol inputs, so a single-pol demo tile
gives it less to work with than an operational scene would. In practice the
slick geometry still comes out clean (the VV channel carries the dark-patch
signal), which is all the downstream drift model needs -- it consumes the
polygon, not the backscatter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
for _p in (MODULE_ROOT, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from unet_model import UNet  # noqa: E402

DEFAULT_WEIGHTS = MODULE_ROOT / "weights" / "unet_best.pt"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def pixel_to_latlon(px: float, py: float, bbox: dict, width: int, height: int) -> tuple[float, float]:
    """Linear interpolation of a pixel coordinate into the scene's registered
    lat/lon bbox.

    bbox is the demo catalog's shape:
        {"top_left": {"lat":..,"lon":..}, "bottom_right": {"lat":..,"lon":..}}

    Row 0 is the TOP of the image, which is top_left.lat (the larger
    latitude), so the latitude axis runs downward -- hence the subtraction.
    """
    tl, br = bbox["top_left"], bbox["bottom_right"]
    u = px / max(width - 1, 1)
    v = py / max(height - 1, 1)
    lon = tl["lon"] + u * (br["lon"] - tl["lon"])
    lat = tl["lat"] - v * (tl["lat"] - br["lat"])
    return float(lat), float(lon)


def polygon_area_km2(polygon_geojson: dict) -> float:
    """Equal-area area of the lon/lat polygon. Delegates to Module 2's
    advection.polygon_area_km2 (local azimuthal-equal-area reprojection),
    which is the same technique postprocessing.area_km2 uses -- so the
    number the drift model receives is computed the way it expects."""
    from shapely.geometry import shape as shapely_shape

    sys.path.insert(0, str(PROJECT_ROOT / "module2_backtracking"))
    from advection import polygon_area_km2 as _area  # noqa: E402

    return float(_area(shapely_shape(polygon_geojson)))


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class PngSpillDetector:
    """Loads unet_best.pt once and reuses it for every request. The FastAPI
    detection service instantiates exactly one of these at startup, so model
    load cost is paid at boot rather than per detection."""

    def __init__(self, weights_path: Path = DEFAULT_WEIGHTS, device_str: Optional[str] = None):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"No U-Net weights at {weights_path}. Expected the trained "
                f"checkpoint shipped in module1_detection/weights/."
            )
        self.device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(weights_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model = UNet(in_channels=2, out_channels=1)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.checkpoint_epoch = ckpt.get("epoch")
        self.checkpoint_val_iou = ckpt.get("val_iou")
        self.weights_path = str(weights_path)

    # -- inference -----------------------------------------------------------
    @torch.no_grad()
    def probability_map(self, image_path: Path, infer_size: int = 256) -> tuple[np.ndarray, tuple[int, int]]:
        """Returns (prob_map at infer_size x infer_size, original (w, h))."""
        img = Image.open(image_path).convert("L")
        orig_size = img.size  # (w, h)
        arr = np.array(img.resize((infer_size, infer_size), Image.LANCZOS))
        # Same normalisation train_unet.py used: uint8 / 255.0, channels last
        # then permuted to (C, H, W). VV duplicated into VH -- see module docstring.
        x = np.stack([arr, arr], axis=-1).astype(np.float32) / 255.0
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
        prob = torch.sigmoid(self.model(t)).squeeze().cpu().numpy()
        return prob.astype(np.float32), orig_size

    def detect(
        self,
        image_path: Path,
        bbox: dict,
        timestamp_utc: str,
        detection_id: str,
        threshold: float = 0.5,
        min_area_px: int = 60,
        infer_size: int = 256,
        simplify_frac: float = 0.012,
        max_vertices: int = 40,
    ) -> dict:
        """Runs the U-Net on one curated PNG and returns a record shaped
        exactly like Module 2's expected detection input, so it can be passed
        straight through with no field renaming."""
        prob, _ = self.probability_map(image_path, infer_size=infer_size)
        h, w = prob.shape

        mask = (prob > threshold).astype(np.uint8)
        # Small open-then-close: drops isolated speckle-driven pixels and
        # bridges 1-2px gaps inside a slick without moving its outline.
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            raise ValueError(
                f"The U-Net found no slick above threshold={threshold} in "
                f"{Path(image_path).name}. Lower the threshold, or pick a "
                f"different scene from the catalog."
            )

        # "Primary component" = largest by pixel area, per the build spec.
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = int(np.argmax(areas)) + 1
        best_area_px = int(areas.max())
        if best_area_px < min_area_px:
            raise ValueError(
                f"Largest detected component is only {best_area_px}px "
                f"(min_area_px={min_area_px}) -- too small to treat as a slick."
            )

        blob = (labels == best_label).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)

        # Simplify progressively until the ring is small enough to animate
        # cheaply on the frontend but still recognisably the same shape.
        epsilon = simplify_frac * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        while len(approx) > max_vertices:
            epsilon *= 1.4
            approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            approx = contour

        ring = [pixel_to_latlon(float(p[0][0]), float(p[0][1]), bbox, w, h) for p in approx]
        coords = [[lon, lat] for lat, lon in ring]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        polygon_geojson = {"type": "Polygon", "coordinates": [coords]}

        cy, cx = float(centroids[best_label][1]), float(centroids[best_label][0])
        lat, lon = pixel_to_latlon(cx, cy, bbox, w, h)

        # Confidence = mean sigmoid probability inside the detected mask,
        # i.e. how strongly the network committed to the pixels it kept.
        confidence = float(prob[blob > 0].mean())

        area_km2 = polygon_area_km2(polygon_geojson)
        if area_km2 <= 0:
            raise ValueError("Detected polygon has non-positive area after reprojection.")

        return {
            "detection_id": detection_id,
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "polygon_geojson": polygon_geojson,
            "timestamp_utc": timestamp_utc,
            "area_km2": round(area_km2, 4),
            "confidence": round(confidence, 4),
            "_debug": {
                "mask_pixels": best_area_px,
                "mask_fraction": round(best_area_px / float(h * w), 4),
                "n_components": int(n_labels - 1),
                "polygon_vertices": len(coords) - 1,
                "threshold": threshold,
                "infer_size": infer_size,
                "checkpoint_epoch": self.checkpoint_epoch,
                "checkpoint_val_iou": (
                    round(float(self.checkpoint_val_iou), 4) if self.checkpoint_val_iou is not None else None
                ),
                "device": str(self.device),
            },
        }

    # -- helper used by the catalog builder ---------------------------------
    def centroid_uv(self, image_path: Path, threshold: float = 0.5, infer_size: int = 256) -> tuple[float, float]:
        """Normalised (u, v) position of the primary component's centroid,
        u across, v down, both in [0,1]. Used by scripts/make_demo_catalog.py
        to register a bbox that puts the slick where the scene is supposed
        to be."""
        prob, _ = self.probability_map(image_path, infer_size=infer_size)
        h, w = prob.shape
        mask = cv2.morphologyEx((prob > threshold).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            raise ValueError(f"No component found in {image_path}")
        best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        return float(centroids[best][0]) / (w - 1), float(centroids[best][1]) / (h - 1)


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Run the trained U-Net on one curated demo PNG.")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    demo_bbox = {"top_left": {"lat": 28.58, "lon": -89.35}, "bottom_right": {"lat": 28.18, "lon": -88.95}}
    det = PngSpillDetector(args.weights)
    print(json.dumps(
        det.detect(args.image, demo_bbox, "2026-09-04T06:12:00Z", "cli_test_001", threshold=args.threshold),
        indent=2,
    ))
