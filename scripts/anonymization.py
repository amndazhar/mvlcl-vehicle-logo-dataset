"""Privacy anonymisation for CCTV frames in a YOLO-format dataset.

For every image the pipeline:
  1. REMOVES text by inpainting (cv2.inpaint, Telea) - burned-in CCTV timestamps and
     camera labels (located by position, following each frame's known augmentation),
     plus any text found by the EasyOCR/CRAFT text detector (signs, plate characters);
  2. irreversibly BLURS (pixelate + Gaussian) faces (CenterFace via `deface`),
     licence plates (a YOLO licence-plate detector), manual boxes, and a safety window
     around the logo whenever no plate was detected next to it;
  3. never modifies pixels inside the annotated logo box;
  4. re-encodes the JPEG without EXIF / GPS metadata.

Dark and rotated frames are additionally searched as a brightened copy and with their
rotation undone. Thresholds deliberately favour over-removal; every output must still
be reviewed by a person (see review_contact_sheet.py and docs/anonymisation-methodology.md).

Usage:
    python scripts/anonymization.py --images RAW_IMAGES --labels LABELS --out OUT_IMAGES \
        --work WORK_DIR --plate-weights PLATE_MODEL.pt [--manual manual_blur.csv]

    RAW_IMAGES / LABELS / OUT_IMAGES share the same sub-folder layout.
    WORK_DIR receives one JSON of detections per image (makes the run resumable:
    re-running skips images already processed) plus review_log.csv and detections.json.
    manual_blur.csv: columns filename,x1,y1,x2,y2 (pixels) for objects found in review.

Dependencies: opencv-python, numpy, deface (MIT), easyocr (Apache-2.0),
ultralytics (AGPL-3.0). The licence-plate weights are NOT included in this
repository; supply any YOLO licence-plate detector whose licence permits your use.
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import cv2
import easyocr
import numpy as np
from deface.centerface import CenterFace
from ultralytics import YOLO

FACE_CONF = 0.2
PLATE_CONF = 0.1
PLATE_IMGSZ = 1280
PAD = 0.15                 # enlarge detection boxes by 15% per side
MAX_BOX_FRAC = 0.04        # drop face/plate boxes bigger than 4% of the image area
MAX_TEXT_FRAC = 0.10       # drop text polygons covering more than 10% of the image
INPAINT_RADIUS = 7         # neighbourhood used to fill removed text
SAFETY_W, SAFETY_H = 3.5, 3.5   # safety window = logo centre +/- 3.5 logo widths / heights
NEAR_LOGO = 4.0            # a plate within 4 logo-heights of the logo counts as "found"
JPEG_QUALITY = 95


# ---------- geometry helpers ----------
def clip_box(x1, y1, x2, y2, w, h):
    return max(0, int(x1)), max(0, int(y1)), min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))


def pad_box(b, w, h, pad=PAD):
    x1, y1, x2, y2 = b
    dw, dh = (x2 - x1) * pad, (y2 - y1) * pad
    return clip_box(x1 - dw, y1 - dh, x2 + dw, y2 + dh, w, h)


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def rotate(img, angle):
    """Rotate about the centre without cropping; return image and the inverse affine."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh)), cv2.invertAffineTransform(m)


def map_box(b, inv, w, h):
    x1, y1, x2, y2 = b
    pts = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], float) @ inv.T
    return clip_box(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max(), w, h)


def brighten(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(hsv[..., 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def known_angles(name):
    """Angles that undo the filename's rotation (sign convention unknown, so try both)."""
    m = re.search(r"rotate(_neg)?(\d+)", name)
    if not m:
        return []
    a = int(m.group(2))
    return [180] if a == 180 else [a, -a]


# ---------- detectors ----------
class Detectors:
    def __init__(self, plate_weights):
        self.face = CenterFace()
        self.plate = YOLO(str(plate_weights))
        self.text = easyocr.Reader(["en"], gpu=False, recognizer=False, verbose=False)

    def faces(self, img):
        dets, _ = self.face(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), threshold=FACE_CONF)
        return [tuple(d[:4]) for d in dets]

    def plates(self, img):
        r = self.plate.predict(img, conf=PLATE_CONF, imgsz=PLATE_IMGSZ, verbose=False)[0]
        return [tuple(b) for b in r.boxes.xyxy.cpu().numpy()]

    def texts(self, img):
        """Text regions as polygons (Nx2 float arrays) - slanted timestamps stay slanted."""
        horiz, free = self.text.detect(img, text_threshold=0.4, low_text=0.25, add_margin=0.15,
                                       canvas_size=1920)
        polys = [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)
                 for x1, x2, y1, y2 in horiz[0]]
        polys += [np.array(p, float) for p in free[0]]
        return polys


def map_poly(poly, inv):
    return np.hstack([poly, np.ones((len(poly), 1))]) @ inv.T


def grow_poly(poly, factor=1.15, margin=6):
    c = poly.mean(axis=0)
    v = poly - c
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    return c + v * factor + v / norm * margin


def detect_faces_plates(det, img, name):
    h, w = img.shape[:2]
    bright = brighten(img)
    faces = det.faces(img) + det.faces(bright)
    plates = det.plates(img) + det.plates(bright)
    for ang in known_angles(name):
        rot, inv = rotate(img, ang)
        plates += [map_box(b, inv, w, h) for b in det.plates(rot)]
    cap = MAX_BOX_FRAC * w * h            # drops giant false face/plate boxes
    faces = [pad_box(b, w, h) for b in faces if area(b) <= cap]
    plates = [pad_box(b, w, h) for b in plates if area(b) <= cap]
    return faces, plates


def detect_text_polys(det, img, name):
    h, w = img.shape[:2]
    polys = det.texts(img) + det.texts(brighten(img))
    for ang in known_angles(name):
        rot, inv = rotate(img, ang)
        polys += [map_poly(p, inv) for p in det.texts(rot)]   # upright timestamps read better
    # Cap on the polygon's own area (not its bounding box), so long slanted
    # timestamps are kept; only absurdly large regions are dropped.
    cap = MAX_TEXT_FRAC * w * h
    out = []
    for p in polys:
        p = grow_poly(p)
        p[:, 0] = p[:, 0].clip(0, w - 1)
        p[:, 1] = p[:, 1].clip(0, h - 1)
        if cv2.contourArea(p.astype(np.float32)) <= cap:
            out.append(p.round().astype(int))
    return out


# ---------- burned-in CCTV overlay (timestamp + camera label) ----------
# Position in an un-augmented frame, as fractions of width/height (measured, plus margin).
OVERLAY_BOXES = [(0.47, 0.0, 0.93, 0.12),     # "DD-MM-YYYY HH:MM:SS", top right
                 (0.0, 0.85, 0.19, 1.0)]       # "Cam 0X", bottom left


def content_mask(img):
    return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 12).astype(np.uint8)


def rotation_matrix_for(img, angle):
    """The augmentation rotated the frame about its centre, same canvas, no scaling.
    Pick the direction whose predicted black corners best match the image."""
    h, w = img.shape[:2]
    actual = content_mask(img)
    best = None
    for a in (angle, -angle):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        pred = cv2.warpAffine(np.ones((h, w), np.uint8), m, (w, h))
        iou = (pred & actual).sum() / max(1, (pred | actual).sum())
        if best is None or iou > best[0]:
            best = (iou, m)
    return best[1], best[0]


def corner_strokes(img, region):
    """Bright, thin, unsaturated strokes (white/grey overlay text) inside region, dilated."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41)))
    m = ((th > 25) & (gray > 130) & (sat < 60)).astype(np.uint8) * 255
    m &= region
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # drop rain specks
    return cv2.dilate(m, np.ones((7, 7), np.uint8), iterations=3)   # also covers dark outlines


def text_line_mask(img, region, min_chars):
    """Find a row of character-like bright strokes (the overlay text) inside region and
    return a mask over that whole text line; empty if no convincing text line is found."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41)))
    raw = ((th > 35) & (gray > 140) & (sat < 60)).astype(np.uint8) * 255
    raw &= region
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(raw)
    out = np.zeros((h, w), np.uint8)
    # Character-shaped pieces: plausible height, upright digit proportions, thin strokes.
    cands = [i for i in range(1, n)
             if 0.025 * h <= stats[i, 3] <= 0.13 * h
             and 0.3 <= stats[i, 2] / stats[i, 3] <= 0.95
             and 0.15 <= stats[i, 4] / (stats[i, 2] * stats[i, 3]) <= 0.6]
    if len(cands) < min_chars:
        return out
    # Chain pieces that have nearly the same height and sit about one character apart
    # (the overlay's digits); scattered highlights from leaves or glare don't chain up.
    parent = {c: c for c in cands}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for a_i, a in enumerate(cands):
        for b in cands[a_i + 1:]:
            ha, hb = stats[a, 3], stats[b, 3]
            if not 0.85 <= ha / hb <= 1.18:
                continue
            d = np.hypot(*(cent[a] - cent[b]))
            if 0.35 * (ha + hb) / 2 <= d <= 1.3 * (ha + hb) / 2:
                parent[find(a)] = find(b)
    groups = {}
    for c in cands:
        groups.setdefault(find(c), []).append(c)
    keep = max(groups.values(), key=len)
    hs = np.array([stats[c, 3] for c in keep], float)
    if len(keep) < min_chars or hs.std() > 0.12 * hs.mean():
        return out
    med = float(np.median(hs))
    corners = []
    for i in keep:
        x, y, bw, bh, _ = stats[i]
        corners += [[x, y], [x + bw, y], [x, y + bh], [x + bw, y + bh]]
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(np.array(corners, np.float32))
    if rw < rh:   # make "length" follow the text direction
        rw, rh, ang = rh, rw, ang + 90
    # Grow along the text so neighbouring cut-off / unmatched characters are covered too.
    box = cv2.boxPoints(((cx, cy), (rw + 4 * med, rh + 0.9 * med), ang))
    cv2.fillPoly(out, [box.round().astype(np.int32)], 255)
    return out


def overlay_mask(img, name):
    """Mask covering the CCTV timestamp / camera label, following the frame's augmentation."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if "crop" in name:
        # Crop position unknown: the overlay can only appear in the top band or bottom-left.
        top = np.zeros((h, w), np.uint8)
        top[: int(0.25 * h), :] = 255
        bottom_left = np.zeros((h, w), np.uint8)
        bottom_left[int(0.72 * h):, : int(0.40 * w)] = 255
        m = text_line_mask(img, top, 3) | text_line_mask(img, bottom_left, 2)
        # In crops the (often slanted, cut-off) timestamp always ends up in the top-right
        # corner, so bright unsaturated strokes there are always removed.
        corner = np.zeros((h, w), np.uint8)
        corner[: int(0.28 * h), int(0.52 * w):] = 255
        m |= corner_strokes(img, corner)
        return m, "crop-textline+corner"

    polys = [np.array([[x1 * w, y1 * h], [x2 * w, y1 * h], [x2 * w, y2 * h], [x1 * w, y2 * h]], float)
             for x1, y1, x2, y2 in OVERLAY_BOXES]
    how = "fixed"
    if "flipH" in name or "rotate180" in name:
        for p in polys:
            p[:, 0] = w - p[:, 0]
    if "flipV" in name or "rotate180" in name:
        for p in polys:
            p[:, 1] = h - p[:, 1]
    m_ = re.search(r"rotate(_neg)?(\d+)", name)
    if m_ and m_.group(2) != "180":
        angle = int(m_.group(2))
        rot, iou = rotation_matrix_for(img, angle)
        polys = [np.hstack([p, np.ones((4, 1))]) @ rot.T for p in polys]
        how = f"rotated (fit {iou:.2f})"
    cv2.fillPoly(mask, [p.round().astype(np.int32) for p in polys], 255)
    return mask, how


def poly_bbox(p):
    return int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()) + 1, int(p[:, 1].max()) + 1


# ---------- main ----------
def logo_boxes(label_path, w, h):
    boxes = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            cx, cy, bw, bh = (float(v) for v in p[1:])
            boxes.append(clip_box((cx - bw / 2) * w, (cy - bh / 2) * h,
                                  (cx + bw / 2) * w, (cy + bh / 2) * h, w, h))
    return boxes


def safety_boxes(logos, plates, w, h):
    out = []
    for lx1, ly1, lx2, ly2 in logos:
        lw, lh = lx2 - lx1, ly2 - ly1
        cx, cy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
        near = [p for p in plates
                if abs((p[0] + p[2]) / 2 - cx) < NEAR_LOGO * max(lw, lh) * 2
                and abs((p[1] + p[3]) / 2 - cy) < NEAR_LOGO * max(lw, lh)]
        if not near:
            out.append(clip_box(cx - SAFETY_W * lw, cy - SAFETY_H * lh,
                                cx + SAFETY_W * lw, cy + SAFETY_H * lh, w, h))
    return out


def irreversible_blur(region):
    h, w = region.shape[:2]
    if h < 2 or w < 2:
        return region
    small = cv2.resize(region, (max(1, w // 16), max(1, h // 16)), interpolation=cv2.INTER_AREA)
    region = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    k = max(3, (min(w, h) // 4) | 1)
    return cv2.GaussianBlur(region, (k, k), 0)


def load_manual(path):
    manual = {}
    if path and path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                manual.setdefault(r["filename"].strip(), []).append(
                    tuple(int(float(r[k])) for k in ("x1", "y1", "x2", "y2")))
    return manual


def anonymise(img, name, logos, det, prev, manual_boxes):
    """Return (anonymised image, detection record) for one frame."""
    h, w = img.shape[:2]
    if prev and "plates" in prev:                     # reuse detections from an earlier run
        faces = [tuple(b) for b in prev["faces"]]
        plates = [tuple(b) for b in prev["plates"]]
    else:
        faces, plates = detect_faces_plates(det, img, name)
    polys = detect_text_polys(det, img, name)
    safety = safety_boxes(logos, plates, w, h)
    rects = faces + plates + safety + manual_boxes

    # 1) text: removed by inpainting (overlay by position + detected text polygons)
    out = img.copy()
    text_mask, overlay_how = overlay_mask(img, name)
    if polys:
        cv2.fillPoly(text_mask, polys, 255)
    for x1, y1, x2, y2 in logos:                     # logo pixels are never changed
        text_mask[y1:y2, x1:x2] = 0
    if text_mask.any():
        content = content_mask(img)
        out = cv2.inpaint(out, text_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
        out[content == 0] = img[content == 0]       # keep black rotation corners black

    # 2) faces, plates, safety windows, manual boxes: irreversible blur
    mask = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in rects:
        mask[y1:y2, x1:x2] = 255
    for x1, y1, x2, y2 in logos:
        mask[y1:y2, x1:x2] = 0
    for x1, y1, x2, y2 in rects:
        region = out[y1:y2, x1:x2]
        if region.size:
            blurred = irreversible_blur(region.copy())
            m = mask[y1:y2, x1:x2] > 0
            region[m] = blurred[m]

    record = {"faces": faces, "plates": plates, "texts": [poly_bbox(p) for p in polys],
              "safety": safety, "manual": manual_boxes, "logos": logos,
              "overlay": overlay_how, "complete": True}
    return out, record


def main():
    ap = argparse.ArgumentParser(description="Anonymise CCTV frames (see module docstring).")
    ap.add_argument("--images", type=Path, required=True, help="raw image root")
    ap.add_argument("--labels", type=Path, required=True, help="YOLO label root (same layout)")
    ap.add_argument("--out", type=Path, required=True, help="output image root")
    ap.add_argument("--work", type=Path, required=True, help="working dir for detections/logs")
    ap.add_argument("--plate-weights", type=Path, required=True, help="YOLO plate model (.pt)")
    ap.add_argument("--manual", type=Path, help="manual_blur.csv from human review")
    ap.add_argument("--match", default="", help="only images whose name contains this text")
    ap.add_argument("--redo", action="store_true",
                    help="re-process every image (stored face/plate detections are reused)")
    a = ap.parse_args()

    det = Detectors(a.plate_weights)
    manual = load_manual(a.manual)
    dets_dir = a.work / "dets"
    images = sorted(p for p in a.images.rglob("*.jpg") if a.match in p.name)

    def done(rel):
        jp = dets_dir / rel.with_suffix(".json")
        return (not a.redo) and jp.exists() and json.loads(jp.read_text()).get("complete")

    todo = [p for p in images if not done(p.relative_to(a.images))]
    print(f"{len(images) - len(todo)} already done, {len(todo)} to process", flush=True)
    t0 = time.time()
    for i, src in enumerate(todo, 1):
        rel = src.relative_to(a.images)
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        logos = logo_boxes(a.labels / rel.with_suffix(".txt"), w, h)
        jp = dets_dir / rel.with_suffix(".json")
        prev = json.loads(jp.read_text()) if jp.exists() else {}
        out, record = anonymise(img, rel.name, logos, det, prev, manual.get(rel.name, []))

        dst = a.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1]
                        .tobytes())                  # OpenCV writes no EXIF/GPS metadata
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(record, default=int))
        if i % 25 == 0 or i == len(todo):
            rate = (time.time() - t0) / i
            print(f"{i}/{len(todo)} done, {rate:.1f}s/img, ~{rate * (len(todo) - i) / 60:.0f} min left",
                  flush=True)

    log, detections = [], {}
    for p in sorted(a.out.rglob("*.jpg")):
        key = p.relative_to(a.out).as_posix()
        jp = dets_dir / Path(key).with_suffix(".json")
        d = json.loads(jp.read_text()) if jp.exists() else {}
        detections[key] = d
        log.append({"image": key, "faces": len(d.get("faces", [])),
                    "plates": len(d.get("plates", [])), "texts": len(d.get("texts", [])),
                    "safety_window": len(d.get("safety", [])), "manual": len(d.get("manual", [])),
                    "overlay": d.get("overlay", "")})
    if log:
        with open(a.work / "review_log.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(log[0]))
            wr.writeheader(); wr.writerows(log)
        (a.work / "detections.json").write_text(json.dumps(detections, default=int))
    print(f"{len(log)} images in output; review them with scripts/review_contact_sheet.py")


if __name__ == "__main__":
    main()
