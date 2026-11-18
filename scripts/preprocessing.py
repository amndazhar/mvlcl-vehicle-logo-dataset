"""CLAHE image enhancement and image-quality measurement, as used in the MVLCL study.

Contrast Limited Adaptive Histogram Equalisation (CLAHE) is applied to the V (value)
channel in HSV colour space, so brightness/contrast change while hue and saturation
are preserved. Enhancement is measured against the original with MSE and PSNR.

Two modes:

  fixed      apply one CLAHE setting to every image
             (the study's "default" setting: clip limit 2.0, tile grid 8 x 8)

  grid       per image, try clip limits {1.5, 2.0, 2.5} x tile grids {8, 16, 32} and
             keep the setting with the highest PSNR versus the original
             (the study's "best-parameter" variant)

  Caveat (grid mode): the highest PSNR against the original is, by definition, the
  setting that changes the image LEAST. The criterion therefore selects the mildest
  enhancement (in the study almost always clip 1.5, tile 32 x 32) rather than the one
  that helps detection most. It is kept here to document what was done; a
  task-driven criterion (e.g. validation mAP) is recommended for future work.

Usage:
    python scripts/preprocessing.py fixed SRC_DIR DST_DIR [--clip 2.0 --tile 8] [--report r.csv]
    python scripts/preprocessing.py grid  SRC_DIR DST_DIR [--report r.csv]

Dependencies: opencv-python, numpy.
"""
import argparse
import csv
from itertools import product
from pathlib import Path

import cv2
import numpy as np

GRID_CLIP = (1.5, 2.0, 2.5)
GRID_TILE = (8, 16, 32)


def apply_clahe(image_bgr, clip_limit=2.0, tile=8):
    """CLAHE on the HSV value channel; returns a BGR image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    h, s, v = cv2.split(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV))
    return cv2.cvtColor(cv2.merge((h, s, clahe.apply(v))), cv2.COLOR_HSV2BGR)


def mse_psnr(original, enhanced):
    """Mean squared error and peak signal-to-noise ratio (dB) between two 8-bit images."""
    mse = float(np.mean((original.astype(np.float64) - enhanced.astype(np.float64)) ** 2))
    psnr = 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
    return mse, float(psnr)


def best_by_psnr(image_bgr):
    """Grid search used in the study; returns (enhanced, clip, tile, mse, psnr)."""
    best = None
    for clip, tile in product(GRID_CLIP, GRID_TILE):
        out = apply_clahe(image_bgr, clip, tile)
        mse, psnr = mse_psnr(image_bgr, out)
        if best is None or psnr > best[4]:
            best = (out, clip, tile, mse, psnr)
    return best


def main():
    ap = argparse.ArgumentParser(description="CLAHE enhancement (HSV V channel) with MSE/PSNR.")
    ap.add_argument("mode", choices=["fixed", "grid"])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--tile", type=int, default=8)
    ap.add_argument("--report", type=Path, help="optional CSV of per-image parameters/MSE/PSNR")
    a = ap.parse_args()
    if a.src.resolve() == a.dst.resolve():
        ap.error("dst must differ from src (source images are never overwritten)")

    a.dst.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(a.src.glob("*.jpg")):
        img = cv2.imread(str(p))
        if img is None:
            print(f"skipped unreadable file: {p.name}")
            continue
        if a.mode == "fixed":
            out = apply_clahe(img, a.clip, a.tile)
            clip, tile = a.clip, a.tile
            mse, psnr = mse_psnr(img, out)
        else:
            out, clip, tile, mse, psnr = best_by_psnr(img)
        cv2.imwrite(str(a.dst / p.name), out)
        rows.append({"filename": p.name, "clip_limit": clip, "tile": tile,
                     "mse": round(mse, 3), "psnr_db": round(psnr, 3)})

    if a.report and rows:
        with open(a.report, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    if rows:
        print(f"{len(rows)} images; mean MSE {np.mean([r['mse'] for r in rows]):.2f}, "
              f"mean PSNR {np.mean([r['psnr_db'] for r in rows]):.2f} dB")


if __name__ == "__main__":
    main()
