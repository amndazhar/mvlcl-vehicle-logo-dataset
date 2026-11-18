"""Contact sheets for the manual privacy review of anonymised images.

Every anonymised image is shown once, 16 per page, with the regions the pipeline acted
on outlined so a reviewer can scan the whole dataset quickly and spot anything missed:

    red      blurred face or licence plate        magenta  removed (inpainted) text
    yellow   safety blur around the logo          orange   manual box from a reviewer
    green    annotated logo (must remain sharp)

Each tile is captioned with its filename; anything missed is added to manual_blur.csv
(filename,x1,y1,x2,y2 in full-resolution pixels) and anonymization.py is re-run.
Review pages contain anonymised but still-restricted imagery: keep them private.

Usage:
    python scripts/review_contact_sheet.py --images ANONYMISED_IMAGES \
        --detections WORK_DIR/detections.json --out REVIEW_DIR

Dependencies: opencv-python, numpy.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TILE_W, TILE_H, COLS, ROWS = 640, 360, 4, 4
COLOURS = {"faces": (0, 0, 255), "plates": (0, 0, 255), "texts": (255, 0, 255),
           "safety": (0, 255, 255), "manual": (0, 165, 255), "logos": (0, 200, 0)}


def tile(images, key, det):
    img = cv2.imread(str(images / key))
    for kind, colour in COLOURS.items():
        for x1, y1, x2, y2 in det.get(kind, []):
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 6)
    img = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (TILE_W, 26), (0, 0, 0), -1)
    cv2.putText(img, Path(key).name, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description="Build manual-review contact sheets.")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    for old in a.out.glob("page_*.jpg"):
        old.unlink()
    dets = json.loads(a.detections.read_text())
    keys = sorted(dets)
    per_page = COLS * ROWS
    for p in range(0, len(keys), per_page):
        tiles = [tile(a.images, k, dets[k]) for k in keys[p:p + per_page]]
        tiles += [np.zeros((TILE_H, TILE_W, 3), np.uint8)] * (per_page - len(tiles))
        rows = [np.hstack(tiles[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
        cv2.imwrite(str(a.out / f"page_{p // per_page + 1:03d}.jpg"), np.vstack(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{-(-len(keys) // per_page)} review pages -> {a.out}")


if __name__ == "__main__":
    main()
