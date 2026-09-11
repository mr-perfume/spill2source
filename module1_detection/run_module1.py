"""
run_module1.py
End-to-end Module 1 CLI: raw calibrated+terrain-corrected SAR scene ->
detections.json matching the spec's output schema.

Usage:
    python run_module1.py --input scene.tif --weights weights/unet_best.pt --output detections.json

    # dry run against a synthetic scene, no real data needed:
    python run_module1.py --self-test

Expected input: see the docstring at the top of preprocessing.py -- a 2-band
(VV, VH) GeoTIFF that's already radiometrically calibrated to sigma0 and
terrain-corrected, with a `<scene>.json` sidecar carrying timestamp_utc /
satellite_pass_id / incidence_angle (or pass those on the command line).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from unet_model import UNet
from preprocessing import preprocess_scene, PreprocessedScene, Tile
from postprocessing import (
    stitch_tiles, apply_land_mask, binarize, connected_components,
    build_detection_records, LookalikeGate,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "weights" / "unet_best.pt"
DEFAULT_WIND_CACHE = ROOT.parent / "data" / "era5_cache" / "wind_lookup.json"
DEFAULT_LOOKALIKE_MODEL = ROOT / "weights" / "lookalike_xgb.json"


# ---------------------------------------------------------------------------
# Model loading + tiled inference
# ---------------------------------------------------------------------------
def load_model(weights_path: Path, device: torch.device) -> UNet:
    model = UNet(in_channels=2, out_channels=1)
    ckpt = torch.load(weights_path, map_location=device)
    # train_unet.py saves {"model_state_dict": ..., "epoch": ..., "val_iou": ...}
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model: UNet, tiles: list[Tile], device: torch.device, batch_size: int = 16) -> list[np.ndarray]:
    """Runs the U-Net over every tile in batches, returns per-tile sigmoid
    probability maps (256, 256) float32, in the same order as `tiles`."""
    probs: list[np.ndarray] = []
    for i in range(0, len(tiles), batch_size):
        batch = tiles[i:i + batch_size]
        imgs = np.stack([t.image for t in batch], axis=0).astype(np.float32) / 255.0  # (B,256,256,2)
        imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)  # (B,2,256,256)

        logits = model(imgs_t)
        batch_probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (B,256,256)

        for p in batch_probs:
            probs.append(p)
    return probs


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
def run_module1(
    input_path: Path,
    weights_path: Path = DEFAULT_WEIGHTS,
    output_path: Path = Path("detections.json"),
    land_shapefile: Path | None = None,
    wind_cache_path: Path | None = DEFAULT_WIND_CACHE,
    lookalike_model_path: Path | None = DEFAULT_LOOKALIKE_MODEL,
    db_min: float = -30.0,
    db_max: float = 0.0,
    tile_size: int = 256,
    stride: int = 224,
    threshold: float = 0.5,
    min_area_px: int = 30,
    batch_size: int = 16,
    apply_speckle_filter: bool = True,
    override_timestamp: str | None = None,
    override_pass_id: str | None = None,
    override_incidence_angle: float | None = None,
    device_str: str | None = None,
) -> list[dict]:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device}")

    print(f"[1/4] preprocessing {input_path}")
    scene: PreprocessedScene = preprocess_scene(
        input_path,
        land_shapefile=land_shapefile,
        db_min=db_min, db_max=db_max,
        tile_size=tile_size, stride=stride,
        apply_speckle_filter=apply_speckle_filter,
        override_timestamp=override_timestamp,
        override_pass_id=override_pass_id,
        override_incidence_angle=override_incidence_angle,
    )
    print(f"       scene shape={scene.full_shape}  tiles={len(scene.tiles)}")

    print(f"[2/4] loading model from {weights_path}")
    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"No weights found at {weights_path}. Train Module 1 first with "
            f"train_unet.py, or pass --weights pointing at a checkpoint."
        )
    model = load_model(weights_path, device)

    print(f"[3/4] running inference over {len(scene.tiles)} tiles (batch_size={batch_size})")
    tile_probs = run_inference(model, scene.tiles, device, batch_size)

    print("[4/4] postprocessing: stitch -> land mask -> threshold -> connected components -> features")
    prob_map = stitch_tiles(tile_probs, scene.tiles, scene.full_shape)
    prob_map = apply_land_mask(prob_map, scene.land_mask)
    binary = binarize(prob_map, threshold=threshold)
    labeled = connected_components(binary, min_area_px=min_area_px)
    n_blobs = int(labeled.max())
    print(f"       {n_blobs} candidate blob(s) after thresholding + cleanup")

    gate = LookalikeGate(lookalike_model_path)
    records = build_detection_records(
        labeled, scene.vv_db, scene.vh_db, scene.transform, scene.crs,
        scene.metadata.timestamp_utc, scene.metadata.satellite_pass_id,
        scene.metadata.incidence_angle,
        wind_cache_path=wind_cache_path,
        lookalike_gate=gate,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"scene": str(input_path), "detections": records}, f, indent=2)
    print(f"[done] {len(records)} detection(s) written to {output_path}")

    return records


# ---------------------------------------------------------------------------
# Self-test: synthetic scene + an untrained model, so the whole chain (data ->
# tiling -> inference -> stitching -> geometry -> schema) is verified to run
# without needing a real Sentinel-1 scene or trained weights on hand.
# ---------------------------------------------------------------------------
def _self_test():
    import tempfile
    import rasterio as rio
    from rasterio.transform import Affine

    print("=== run_module1.py self-test (synthetic scene, untrained model) ===")
    h, w = 512, 512
    rng = np.random.default_rng(2)
    vv = np.clip(rng.normal(0.02, 0.01, size=(h, w)), 1e-5, None).astype(np.float32)
    vh = np.clip(rng.normal(0.01, 0.006, size=(h, w)), 1e-5, None).astype(np.float32)
    vv[100:220, 80:300] *= 0.15

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tif_path = tmp / "scene.tif"
        transform = Affine.translation(-40.5, 43.0) * Affine.scale(0.0002, -0.0002)
        with rio.open(tif_path, "w", driver="GTiff", height=h, width=w, count=2,
                       dtype="float32", crs="EPSG:4326", transform=transform) as dst:
            dst.write(vv, 1)
            dst.write(vh, 2)
        with open(tmp / "scene.json", "w") as f:
            json.dump({"timestamp_utc": "2017-03-12T06:00:00Z",
                       "satellite_pass_id": "S1A_TEST", "incidence_angle": 34.2}, f)

        # untrained model, just to prove the tensor plumbing (shapes/dtypes/
        # device) is correct end-to-end; obviously won't find the real slick
        weights_path = tmp / "unet_untrained.pt"
        model = UNet(in_channels=2, out_channels=1)
        torch.save({"model_state_dict": model.state_dict(), "epoch": 0, "val_iou": 0.0}, weights_path)

        out_path = tmp / "detections.json"
        records = run_module1(
            tif_path, weights_path=weights_path, output_path=out_path,
            wind_cache_path=None, apply_speckle_filter=False,
            threshold=0.5, min_area_px=5,
        )
        print(f"[self-test] {len(records)} detection(s) (expected: could be 0, model is untrained)")
        assert out_path.exists()
        with open(out_path) as f:
            payload = json.load(f)
        assert "detections" in payload
        print("[self-test] OK -- pipeline runs end-to-end without errors")


def parse_args():
    p = argparse.ArgumentParser(description="Run Module 1 (SAR oil detection) on one scene.")
    p.add_argument("--input", type=Path, help="Path to calibrated + terrain-corrected 2-band GeoTIFF (VV, VH)")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--output", type=Path, default=Path("detections.json"))
    p.add_argument("--land-shapefile", type=Path, default=None,
                   help="Vector land polygon layer (e.g. Natural Earth ne_10m_land) for land/sea masking")
    p.add_argument("--wind-cache", type=Path, default=DEFAULT_WIND_CACHE)
    p.add_argument("--lookalike-model", type=Path, default=DEFAULT_LOOKALIKE_MODEL)
    p.add_argument("--db-min", type=float, default=-30.0)
    p.add_argument("--db-max", type=float, default=0.0)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=224)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-area-px", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-speckle-filter", action="store_true")
    p.add_argument("--timestamp", type=str, default=None, help="Override timestamp_utc if not in a sidecar/tags")
    p.add_argument("--pass-id", type=str, default=None, help="Override satellite_pass_id")
    p.add_argument("--incidence-angle", type=float, default=None)
    p.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'; auto-detects if omitted")
    p.add_argument("--self-test", action="store_true", help="Run against a synthetic scene, ignore all other args")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.self_test:
        _self_test()
    else:
        if args.input is None:
            raise SystemExit("--input is required (or pass --self-test to dry-run on synthetic data)")
        run_module1(
            input_path=args.input,
            weights_path=args.weights,
            output_path=args.output,
            land_shapefile=args.land_shapefile,
            wind_cache_path=args.wind_cache,
            lookalike_model_path=args.lookalike_model,
            db_min=args.db_min, db_max=args.db_max,
            tile_size=args.tile_size, stride=args.stride,
            threshold=args.threshold, min_area_px=args.min_area_px,
            batch_size=args.batch_size,
            apply_speckle_filter=not args.no_speckle_filter,
            override_timestamp=args.timestamp,
            override_pass_id=args.pass_id,
            override_incidence_angle=args.incidence_angle,
            device_str=args.device,
        )
