"""
train_unet.py
Trains the Module 1 U-Net on module1_detection/data/merged_dataset.npz,
using split assignment from metadata.xml (falls back to random split if
a sample_id has no matching split entry).

Assumes:
  - merged_dataset.npz contains: images (N,256,256,2) uint8,
    masks (N,256,256) uint8, sample_id (N,) str
  - metadata.xml contains one <sample> block per sample_id with a
    <split>train|val|test</split> child
  - unet_model.py defines: class UNet(in_channels: int, out_channels: int)
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from unet_model import UNet

# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_NPZ = ROOT / "data" / "merged_dataset.npz"
METADATA_XML = ROOT / "data" / "metadata.xml"
WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Split lookup from metadata.xml
# ---------------------------------------------------------------------------
def load_split_map(xml_path: Path) -> dict:
    """Returns {sample_id: split_str}. Missing entries just aren't in the dict."""
    split_map = {}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for sample in root.findall(".//sample"):
        sid_el = sample.find("sample_id")
        split_el = sample.find("split")
        if sid_el is not None and split_el is not None and sid_el.text:
            split_map[sid_el.text.strip()] = (split_el.text or "").strip().lower()
    return split_map


def assign_splits(sample_ids: np.ndarray, split_map: dict, val_frac: float = 0.1, seed: int = 42):
    """Returns boolean arrays (train_mask, val_mask). Any sample_id missing
    from split_map is randomly assigned to train/val at val_frac."""
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(len(sample_ids), dtype=bool)
    val_mask = np.zeros(len(sample_ids), dtype=bool)
    missing_idx = []

    for i, sid in enumerate(sample_ids):
        split = split_map.get(sid)
        if split == "train":
            train_mask[i] = True
        elif split in ("val", "validation"):
            val_mask[i] = True
        elif split == "test":
            pass  # excluded from this script entirely
        else:
            missing_idx.append(i)

    if missing_idx:
        missing_idx = np.array(missing_idx)
        is_val = rng.random(len(missing_idx)) < val_frac
        val_mask[missing_idx[is_val]] = True
        train_mask[missing_idx[~is_val]] = True
        print(f"[split] {len(missing_idx)} samples had no split in metadata.xml; "
              f"randomly assigned ({is_val.sum()} val / {(~is_val).sum()} train).")

    return train_mask, val_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SARSegDataset(Dataset):
    def __init__(self, images: np.ndarray, masks: np.ndarray, augment: bool = False):
        self.images = images  # (N,256,256,2) uint8
        self.masks = masks    # (N,256,256) uint8
        self.augment = augment

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0  # normalize uint8 -> [0,1]
        mask = (self.masks[idx] > 0).astype(np.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img = img[::-1, :, :].copy()
                mask = mask[::-1, :].copy()

        img = torch.from_numpy(img).permute(2, 0, 1)      # (2,256,256)
        mask = torch.from_numpy(mask).unsqueeze(0)         # (1,256,256)
        return img, mask


# ---------------------------------------------------------------------------
# Loss: BCE + Dice combined (handles class imbalance better than BCE alone)
# ---------------------------------------------------------------------------
class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, smooth: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


@torch.no_grad()
def iou_score(logits, targets, threshold: float = 0.5, smooth: float = 1e-6):
    preds = (torch.sigmoid(logits) > threshold).float()
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
    return ((intersection + smooth) / (union + smooth)).mean().item()


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    print(f"[data] loading {DATA_NPZ}")
    data = np.load(DATA_NPZ, allow_pickle=True)
    images, masks, sample_ids = data["images"], data["masks"], data["sample_id"]

    # class balance check — do this before you trust anything downstream
    unique, counts = np.unique(masks, return_counts=True)
    total = counts.sum()
    print("[class balance]", {int(u): f"{c} ({100*c/total:.2f}%)" for u, c in zip(unique, counts)})

    split_map = load_split_map(METADATA_XML)
    train_mask, val_mask = assign_splits(sample_ids, split_map, val_frac=args.val_frac)
    print(f"[split] train={train_mask.sum()}  val={val_mask.sum()}")

    train_ds = SARSegDataset(images[train_mask], masks[train_mask], augment=True)
    val_ds = SARSegDataset(images[val_mask], masks[val_mask], augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = UNet(in_channels=2, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_iou = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, msks in train_loader:
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, msks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        val_iou_total = 0.0
        val_loss_total = 0.0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(device), msks.to(device)
                logits = model(imgs)
                val_loss_total += criterion(logits, msks).item() * imgs.size(0)
                val_iou_total += iou_score(logits, msks) * imgs.size(0)
        val_loss = val_loss_total / len(val_ds)
        val_iou = val_iou_total / len(val_ds)

        scheduler.step(val_iou)
        print(f"[epoch {epoch:03d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_iou={val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            ckpt_path = WEIGHTS_DIR / "unet_best.pt"
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_iou": val_iou}, ckpt_path)
            print(f"  -> new best (IoU={val_iou:.4f}), saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"[early stop] no val IoU improvement for {args.early_stop_patience} epochs")
                break

    print(f"[done] best val IoU: {best_iou:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--early-stop-patience", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
