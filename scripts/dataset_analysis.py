"""Dataset analysis and quality control for the MVLCL vehicle-logo dataset.

Computes aggregate statistics and quality-control checks for a YOLO-format image
dataset and writes publication-safe charts (aggregate data only, no image content)
plus machine-readable JSON summaries.

Expected layout (one sub-folder per split and lighting condition):

    <images>/{train,test}/{optimal,complex}/<lighting>_<Brand>_<videoID>[_<aug>].jpg
    <labels>/{train,test}/{optimal,complex}/<same stem>.txt     # YOLO: cls cx cy w h

Usage:
    python scripts/dataset_analysis.py --images PATH --labels PATH \
        [--video-info video_info.csv] --out analytics/

Outputs (in --out):
    class-distribution.png, data-split-distribution.png, dataset-statistics.png,
    annotation-statistics.png, lighting-luminance-distribution.png,
    augmentation-distribution.png, clip-duration-distribution.png (if --video-info),
    quality-control-summary.png, dataset_statistics.json, quality_report.json

Dependencies: numpy, opencv-python, pillow, matplotlib.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASSES = ["BMW", "Honda", "Hyundai", "Mazda", "Mercedes",
           "Nissan", "Perodua", "Proton", "Toyota", "Volkswagen"]
SPLITS = ["train", "test"]
CONDITIONS = ["optimal", "complex"]
NAME_RE = re.compile(r"^(optimal|complex)_([A-Za-z]+)_(\d{3})(?:_(.+))?$")

# ---- chart style: one light surface, recessive chrome, validated 2-series palette ----
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = {"optimal": "#2a78d6", "complex": "#eb6834"}          # validated pair
LABEL = {"optimal": "Optimal lighting", "complex": "Complex lighting"}
STATUS = {"pass": "#0ca30c", "fixed": "#2a78d6", "flag": "#fab219", "fail": "#d03b3b"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 12,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def style(ax, grid_axis="y"):
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)
    ax.tick_params(length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)


def save(fig, out, name, note=None):
    if note:
        fig.text(0.01, -0.03, note, fontsize=8, color=MUTED, ha="left", va="top")
    fig.savefig(out / name, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ---------------------------------------------------------------- data loading
def load_items(images, labels):
    items = []
    for split in SPLITS:
        for cond in CONDITIONS:
            for img in sorted((images / split / cond).glob("*.jpg")):
                m = NAME_RE.match(img.stem)
                items.append({
                    "split": split, "condition": cond, "path": img, "stem": img.stem,
                    "label": labels / split / cond / f"{img.stem}.txt",
                    "name_ok": bool(m and m.group(1) == cond),
                    "brand": m.group(2) if m else None,
                    "video_id": m.group(3) if m else None,
                    "aug": (re.sub(r"\d+$", "", m.group(4)) if m and m.group(4) else "none")
                    if m else None,
                })
    return items


HASH_SIZE = 16          # 16 x 16 = 256-bit difference hash
NEAR_DUP_BITS = 20      # <= 20 differing bits (~8 %): candidate near-duplicate
LIKELY_DUP_BITS = 8     # same brand and <= 8 bits: likely the same source frame


def dhash(gray, size=HASH_SIZE):
    """Difference hash of the central 60 % of the frame. The centre is used so that the
    black corners shared by all rotated frames don't make unrelated images look alike."""
    h, w = gray.shape
    centre = gray[int(0.2 * h):int(0.8 * h), int(0.2 * w):int(0.8 * w)]
    small = cv2.resize(centre, (size + 1, size), interpolation=cv2.INTER_AREA)
    return np.packbits((small[:, 1:] > small[:, :-1]).flatten())


def analyse(items):
    """Per-image measurements and per-check findings."""
    qc = defaultdict(list)
    md5s, hashes = defaultdict(list), []
    for it in items:
        # -- image integrity, size, metadata
        try:
            with Image.open(it["path"]) as im:
                im.verify()
            with Image.open(it["path"]) as im:
                it["size"] = im.size
                exif = im.getexif()
                it["exif_tags"] = len(exif)
                it["gps"] = 0x8825 in exif
        except Exception as e:  # noqa: BLE001
            qc["corrupt_image"].append(f"{it['stem']}: {e}")
            continue
        data = it["path"].read_bytes()
        md5s[hashlib.md5(data).hexdigest()].append(it["stem"])
        gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(gray, (480, 270), interpolation=cv2.INTER_AREA)
        content = small[small > 3]                      # ignore black rotation corners
        it["luminance"] = float(content.mean()) if content.size else 0.0
        hashes.append((it["stem"], it["split"], dhash(small)))

        # -- labels
        if not it["name_ok"]:
            qc["unparseable_filename"].append(it["stem"])
        if not it["label"].exists():
            qc["missing_label"].append(it["stem"])
            continue
        rows = [r.split() for r in it["label"].read_text().splitlines() if r.strip()]
        it["boxes"] = []
        if len(rows) != 1:
            qc["not_exactly_one_box"].append(f"{it['stem']}: {len(rows)} boxes")
        for r in rows:
            try:
                c, cx, cy, w, h = int(r[0]), *map(float, r[1:5])
                assert len(r) == 5
            except Exception:  # noqa: BLE001
                qc["malformed_label_row"].append(it["stem"])
                continue
            if not 0 <= c < len(CLASSES):
                qc["invalid_class_id"].append(f"{it['stem']}: {c}")
                continue
            if not (0 < w <= 1 and 0 < h <= 1 and 0 <= cx - w / 2 + 1e-6 and cx + w / 2 <= 1 + 1e-6
                    and 0 <= cy - h / 2 + 1e-6 and cy + h / 2 <= 1 + 1e-6):
                qc["box_outside_image_or_empty"].append(it["stem"])
            if it["brand"] and CLASSES[c] != it["brand"]:
                qc["class_differs_from_source_brand"].append(
                    f"{it['stem']}: label {CLASSES[c]}")
            W, H = it["size"]
            it["boxes"].append({"cls": CLASSES[c], "cx": cx, "cy": cy, "w": w, "h": h,
                                "w_px": w * W, "h_px": h * H})
    # -- duplicates
    for _, stems in md5s.items():
        if len(stems) > 1:
            qc["exact_duplicate_files"].append(", ".join(stems))
    # Near-duplicates, in two confidence tiers. All frames come from the same fixed cameras,
    # so different vehicles in the same view can hash alike; a very small distance between
    # frames of the SAME brand was confirmed by visual spot-checks to mean the same source
    # frame reused under another clip ID. Everything else is only a candidate.
    meta = {it["stem"]: it for it in items}
    hv = np.stack([h for _, _, h in hashes])                      # (n, 32) packed bits
    for i in range(len(hashes) - 1):
        dist = np.unpackbits(np.bitwise_xor(hv[i + 1:], hv[i]), axis=1).sum(axis=1)
        for j in np.nonzero(dist <= NEAR_DUP_BITS)[0]:
            a, b = meta[hashes[i][0]], meta[hashes[i + 1 + j][0]]
            entry = {"a": a["stem"], "b": b["stem"], "distance": int(dist[j]),
                     "spans_train_test": a["split"] != b["split"],
                     "spans_lighting": a["condition"] != b["condition"]}
            likely = a["brand"] == b["brand"] and dist[j] <= LIKELY_DUP_BITS
            qc["likely_duplicate_pairs" if likely else "candidate_duplicate_pairs"].append(entry)
    # -- leakage: same source video in train and test
    vids = defaultdict(set)
    for it in items:
        if it["video_id"]:
            vids[(it["condition"], it["brand"], it["video_id"])].add(it["split"])
    for key, splits in vids.items():
        if len(splits) > 1:
            qc["source_video_in_train_and_test"].append("_".join(key))
    for it in items:
        if it.get("gps"):
            qc["gps_metadata_present"].append(it["stem"])
    return qc


# ---------------------------------------------------------------- charts
def chart_class_distribution(items, out):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = np.arange(len(CLASSES))
    bh = 0.36
    for k, cond in enumerate(CONDITIONS):
        cnt = Counter(b["cls"] for it in items if it["condition"] == cond
                      for b in it.get("boxes", []))
        vals = [cnt.get(c, 0) for c in CLASSES]
        ax.barh(y + (k - 0.5) * (bh + 0.04), vals, height=bh, color=SERIES[cond],
                edgecolor=SURFACE, linewidth=2, label=LABEL[cond])
    ax.set_yticks(y, CLASSES)
    ax.invert_yaxis()
    ax.set_xlabel("Annotated logo instances (train + test)")
    ax.set_title("Class distribution by lighting condition")
    ax.legend(loc="lower right", ncols=2, bbox_to_anchor=(1, 1.0), fontsize=9)
    style(ax, "x")
    save(fig, out, "class-distribution.png",
         "One logo per image. Counts after label validation (see docs/quality-control.md).")


def chart_split_distribution(items, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(SPLITS))
    bw = 0.34
    for k, cond in enumerate(CONDITIONS):
        vals = [sum(1 for it in items if it["split"] == s and it["condition"] == cond)
                for s in SPLITS]
        bars = ax.bar(x + (k - 0.5) * (bw + 0.03), vals, width=bw, color=SERIES[cond],
                      edgecolor=SURFACE, linewidth=2, label=LABEL[cond])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v}", ha="center",
                    va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(x, ["Training", "Testing"])
    ax.set_ylabel("Images")
    ax.set_title("Train / test split by lighting condition")
    ax.legend(loc="upper right", fontsize=9)
    style(ax, "y")
    save(fig, out, "data-split-distribution.png",
         "No separate validation split exists in the original project (see docs/limitations.md).")


def chart_dataset_statistics(stats, out):
    tiles = [
        (f"{stats['source_clips']:,}", "selected source clips"),
        (f"{stats['source_frames']:,}", "frames in selected clips"),
        (f"{stats['annotated_images']:,}", "annotated images"),
        (f"{stats['classes']}", "logo classes"),
        (stats["image_resolution"], "image resolution (px)"),
        ("YOLO txt", "annotation format"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9, 3.6))
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.06, hspace=0.12)
    for ax, (val, lab) in zip(axes.flat, tiles):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.06), 0.96, 0.88, transform=ax.transAxes,
                                   facecolor="#f4f3ef", edgecolor="none"))
        ax.text(0.1, 0.56, val, fontsize=20, fontweight="bold", color=INK,
                transform=ax.transAxes, va="center")
        ax.text(0.1, 0.25, lab, fontsize=9.5, color=INK2, transform=ax.transAxes, va="center")
    fig.suptitle("MVLCL dataset at a glance", x=0.02, ha="left", fontsize=12,
                 fontweight="bold")
    save(fig, out, "dataset-statistics.png",
         "Source clips/frames from video_info.csv; annotated images after de-duplication.")


def chart_annotation_statistics(items, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.1, 1]})
    bins = np.arange(0, 290, 10)
    for cond in CONDITIONS:
        w = [b["w_px"] for it in items if it["condition"] == cond for b in it.get("boxes", [])]
        a1.hist(w, bins=bins, histtype="step", linewidth=2, color=SERIES[cond], label=LABEL[cond])
    a1.set_xlabel("Logo box width (px, at 1920 × 1080)")
    a1.set_ylabel("Images")
    a1.set_title("Logo size")
    a1.legend(fontsize=9)
    style(a1, "y")
    for cond in CONDITIONS:
        pts = [(b["cx"], b["cy"]) for it in items if it["condition"] == cond
               for b in it.get("boxes", [])]
        xs, ys = zip(*pts)
        a2.scatter(xs, ys, s=10, color=SERIES[cond], alpha=0.55, linewidths=0.4,
                   edgecolors=SURFACE, label=LABEL[cond])
    a2.set_xlim(0, 1); a2.set_ylim(1, 0)
    a2.set_aspect(1080 / 1920)
    a2.set_xlabel("Box centre x (fraction of width)")
    a2.set_ylabel("Box centre y (fraction of height)")
    a2.set_title("Logo position in frame")
    style(a2, "both")
    save(fig, out, "annotation-statistics.png",
         "Aggregate box geometry only; positions are normalised YOLO coordinates.")


def chart_luminance(items, out):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bins = np.arange(0, 205, 5)
    for cond in CONDITIONS:
        v = [it["luminance"] for it in items if it["condition"] == cond and "luminance" in it]
        ax.hist(v, bins=bins, histtype="step", linewidth=2, color=SERIES[cond],
                label=f"{LABEL[cond]} (median {np.median(v):.0f})")
    ax.set_xlabel("Mean image luminance (0–255, black rotation borders excluded)")
    ax.set_ylabel("Images")
    ax.set_title("Brightness distribution by lighting condition")
    ax.legend(fontsize=9)
    style(ax, "y")
    save(fig, out, "lighting-luminance-distribution.png")


def chart_augmentation(items, out):
    order = ["none", "crop", "rotate", "rotate_neg", "flipH", "flipV", "rain"]
    nice = {"none": "No augmentation", "crop": "Crop (60/75 %)", "rotate": "Rotate (+20/30/180°)",
            "rotate_neg": "Rotate (−20°)", "flipH": "Horizontal flip", "flipV": "Vertical flip",
            "rain": "Synthetic rain"}
    present = [o for o in order if any(it["aug"] == o for it in items)]
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    y = np.arange(len(present))
    bh = 0.36
    for k, cond in enumerate(CONDITIONS):
        cnt = Counter(it["aug"] for it in items if it["condition"] == cond)
        ax.barh(y + (k - 0.5) * (bh + 0.04), [cnt.get(o, 0) for o in present], height=bh,
                color=SERIES[cond], edgecolor=SURFACE, linewidth=2, label=LABEL[cond])
    ax.set_yticks(y, [nice[o] for o in present])
    ax.invert_yaxis()
    ax.set_xlabel("Annotated images")
    ax.set_title("Clip-level augmentation applied before frame extraction")
    ax.legend(loc="lower right", fontsize=9)
    style(ax, "x")
    save(fig, out, "augmentation-distribution.png",
         "Derived from the augmentation suffix recorded in each source clip's filename.")


def chart_clip_durations(video_info, out):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bins = np.arange(1, 10.5, 0.5)
    # Split by the "Data Type" column: the optimal-lighting Proton rows carry "_UD_" clip
    # names in video_info.csv although the files on disk are named "_CD_" (documented defect).
    for cond, dtype in (("optimal", "Controlled"), ("complex", "Uncontrolled")):
        d = [float(r["Duration (s)"]) for r in video_info if r["Data Type"] == dtype]
        ax.hist(d, bins=bins, histtype="step", linewidth=2, color=SERIES[cond],
                label=f"{LABEL[cond]} ({len(d)} clips)")
    ax.set_xlabel("Clip duration (s)")
    ax.set_ylabel("Clips")
    ax.set_title("Duration of the trimmed source clips")
    ax.legend(fontsize=9)
    style(ax, "y")
    save(fig, out, "clip-duration-distribution.png")


def chart_quality(qc_rows, out):
    fig, ax = plt.subplots(figsize=(11, 0.31 * len(qc_rows) + 0.5))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(len(qc_rows) - 0.4, -0.9)
    ax.text(0.0, -0.45, "Check", fontsize=9, color=MUTED, fontweight="bold")
    ax.text(0.47, -0.45, "Result", fontsize=9, color=MUTED, fontweight="bold")
    ax.text(0.82, -0.45, "Status", fontsize=9, color=MUTED, fontweight="bold")
    for i, (check, result, status) in enumerate(qc_rows):
        ax.axhline(i - 0.5, color=GRID, linewidth=0.6)
        ax.text(0.0, i, check, va="center", fontsize=9.5, color=INK)
        ax.text(0.47, i, result, va="center", fontsize=9.5, color=INK2)
        ax.add_patch(plt.Rectangle((0.82, i - 0.18), 0.012, 0.36, color=STATUS[status]))
        ax.text(0.84, i, {"pass": "Pass", "fixed": "Found & fixed", "flag": "Needs review",
                          "fail": "Fail"}[status], va="center", fontsize=9.5, color=INK)
    ax.set_title("Quality-control summary", pad=4)
    save(fig, out, "quality-control-summary.png",
         "Status is shown as text + colour. Details: docs/quality-control.md.")


# ---------------------------------------------------------------- main
def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--video-info", type=Path, help="video_info.csv of the source clips")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra-qc", type=Path,
                    help="optional JSON of findings from steps run before this script "
                         "(e.g. label corrections), merged into the QC summary")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    items = load_items(a.images, a.labels)
    qc = analyse(items)
    video_info = []
    if a.video_info:
        with open(a.video_info, newline="") as f:
            video_info = list(csv.DictReader(f))

    sizes = Counter(it["size"] for it in items if "size" in it)
    stats = {
        "annotated_images": len(items),
        "classes": len(CLASSES),
        "by_split_condition": {f"{s}/{c}": sum(1 for it in items if it["split"] == s and
                                               it["condition"] == c)
                               for s in SPLITS for c in CONDITIONS},
        "by_class": dict(Counter(b["cls"] for it in items for b in it.get("boxes", []))),
        "image_sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()},
        "image_resolution": "×".join(map(str, sizes.most_common(1)[0][0])) if sizes else "n/a",
        "boxes_per_image": dict(Counter(len(it.get("boxes", [])) for it in items)),
        "logo_width_px": {k: float(f(np.array([b["w_px"] for it in items
                                                  for b in it.get("boxes", [])])))
                          for k, f in (("min", np.min), ("median", np.median),
                                       ("p90", lambda v: np.percentile(v, 90)), ("max", np.max))},
        "median_luminance": {c: float(np.median([it["luminance"] for it in items
                                                 if it["condition"] == c and "luminance" in it]))
                             for c in CONDITIONS},
        "augmentation": {c: dict(Counter(it["aug"] for it in items if it["condition"] == c))
                         for c in CONDITIONS},
        "images_with_exif": sum(1 for it in items if it.get("exif_tags")),
        "source_clips": len(video_info),
        "source_frames": sum(int(r["Num of frames"]) for r in video_info),
        "source_clip_resolution": dict(Counter(r["Width x Height"] for r in video_info)),
        "source_clip_fps": dict(Counter(r["FPS"] for r in video_info)),
    }

    extra = json.loads(a.extra_qc.read_text()) if a.extra_qc else {}
    n = len(items)
    likely, cand = qc["likely_duplicate_pairs"], qc["candidate_duplicate_pairs"]
    stats["duplicates"] = {
        "likely_pairs": len(likely),
        "likely_spanning_train_test": sum(p["spans_train_test"] for p in likely),
        "likely_spanning_lighting_conditions": sum(p["spans_lighting"] for p in likely),
        "candidate_pairs": len(cand),
        "candidate_spanning_train_test": sum(p["spans_train_test"] for p in cand),
    }
    rows = [
        ("Image files open and decode", f"{len(qc['corrupt_image'])} corrupt of {n}",
         "pass" if not qc["corrupt_image"] else "fail"),
        ("Every image has a label file", f"{len(qc['missing_label'])} missing",
         "pass" if not qc["missing_label"] else "fail"),
        ("Exactly one logo box per image", f"{len(qc['not_exactly_one_box'])} exceptions",
         "pass" if not qc["not_exactly_one_box"] else "flag"),
        ("Label rows well-formed, class id 0–9",
         f"{len(qc['malformed_label_row']) + len(qc['invalid_class_id'])} invalid",
         "pass" if not (qc["malformed_label_row"] or qc["invalid_class_id"]) else "fail"),
        ("Boxes inside image, non-zero size", f"{len(qc['box_outside_image_or_empty'])} invalid",
         "pass" if not qc["box_outside_image_or_empty"] else "fail"),
        ("Label class matches source-clip brand",
         extra.get("class_corrections", f"{len(qc['class_differs_from_source_brand'])} mismatches"),
         "fixed" if extra.get("class_corrections") else
         ("pass" if not qc["class_differs_from_source_brand"] else "flag")),
        ("No exact duplicate files", f"{len(qc['exact_duplicate_files'])} groups",
         "pass" if not qc["exact_duplicate_files"] else "flag"),
        ("No source frame reused under another clip ID",
         f"{len(likely)} likely pairs ({sum(p['spans_train_test'] for p in likely)} span train/test)",
         "pass" if not likely else "flag"),
        ("Other visually similar pairs (same camera view)",
         f"{len(cand)} candidates, unverified", "pass" if not cand else "flag"),
        ("No clip ID in both train and test",
         f"{len(qc['source_video_in_train_and_test'])} clips",
         "pass" if not qc["source_video_in_train_and_test"] else "fail"),
        ("No GPS metadata in image files", f"{len(qc['gps_metadata_present'])} files",
         "pass" if not qc["gps_metadata_present"] else "fail"),
    ]
    for r in extra.get("additional_rows", []):
        rows.append(tuple(r))

    chart_class_distribution(items, a.out)
    chart_split_distribution(items, a.out)
    chart_dataset_statistics(stats, a.out)
    chart_annotation_statistics(items, a.out)
    chart_luminance(items, a.out)
    chart_augmentation(items, a.out)
    if video_info:
        chart_clip_durations(video_info, a.out)
    chart_quality(rows, a.out)

    (a.out / "dataset_statistics.json").write_text(json.dumps(stats, indent=2))
    (a.out / "quality_report.json").write_text(json.dumps(
        {"checks": [{"check": c, "result": r, "status": s} for c, r, s in rows],
         "findings": {k: v for k, v in qc.items()}}, indent=2))
    print(json.dumps(stats, indent=2))
    for c, r, s in rows:
        print(f"[{s:5s}] {c}: {r}")


if __name__ == "__main__":
    main()
