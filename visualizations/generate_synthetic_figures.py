"""Generate the illustrative figures in visualizations/ from a SYNTHETIC scene.

No CCTV footage is used anywhere in this file. A cartoon frame is drawn from scratch
(car, generic emblem, fake plate "ABC 1234", fake timestamp/camera label, a simple
face) and processed with the project's real functions:

    preprocessing.apply_clahe / mse_psnr          -> CLAHE demo
    anonymization.overlay_mask / irreversible_blur -> anonymisation steps

Usage:  python visualizations/generate_synthetic_figures.py
Writes: visualizations/annotation-examples/*.png, visualizations/preprocessing-examples/*.png
"""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import preprocessing as P  # noqa: E402

W, H = 1920, 1080
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
                     "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK})
NOTE = "Synthetic illustration - drawn scene, no CCTV footage."

# Logo box of the synthetic scene (pixels) and the matching YOLO label.
LOGO = (930, 585, 1010, 645)


def synthetic_frame(rng=np.random.default_rng(7)):
    """Daylight-style cartoon frame at 1920 x 1080 (BGR)."""
    img = np.zeros((H, W, 3), np.uint8)
    for y in range(H):                                   # sky -> road gradient
        t = y / H
        img[y] = (np.array([200, 190, 170]) * (1 - t) + np.array([90, 92, 95]) * t)
    img = (img + rng.normal(0, 4, img.shape)).clip(0, 255).astype(np.uint8)
    cv2.rectangle(img, (0, 640), (W, H), (85, 88, 92), -1)                    # road
    for x in range(-200, W, 260):                                             # kerb stripes
        cv2.fillPoly(img, [np.array([[x, 1000], [x + 140, 1000], [x + 220, 1080], [x + 80, 1080]])],
                     (40, 200, 230))
    cv2.rectangle(img, (120, 280), (190, 900), (210, 210, 205), -1)           # barrier post
    cv2.rectangle(img, (190, 330), (700, 360), (60, 90, 230), -1)             # barrier arm
    # car body
    cv2.rectangle(img, (640, 360), (1300, 820), (70, 65, 60), -1, lineType=cv2.LINE_AA)
    cv2.fillPoly(img, [np.array([[700, 360], [1240, 360], [1180, 240], [760, 240]])], (120, 110, 100))
    cv2.fillPoly(img, [np.array([[730, 350], [1210, 350], [1160, 260], [780, 260]])], (150, 130, 110))
    for cx in (740, 1200):                                                    # headlights
        cv2.ellipse(img, (cx, 560), (70, 35), 0, 0, 360, (235, 240, 245), -1, cv2.LINE_AA)
    cv2.rectangle(img, (820, 520), (1120, 640), (35, 35, 38), -1)             # grille
    for y in range(530, 640, 18):
        cv2.line(img, (830, y), (1110, y), (60, 60, 65), 3)
    # generic emblem (not any real brand): ring with a diamond
    cx, cy = (LOGO[0] + LOGO[2]) // 2, (LOGO[1] + LOGO[3]) // 2
    cv2.circle(img, (cx, cy), 27, (200, 200, 205), 5, cv2.LINE_AA)
    cv2.fillPoly(img, [np.array([[cx, cy - 18], [cx + 14, cy], [cx, cy + 18], [cx - 14, cy]])],
                 (200, 200, 205), cv2.LINE_AA)
    # fake plate
    cv2.rectangle(img, (850, 690), (1090, 750), (20, 20, 20), -1)
    cv2.putText(img, "ABC 1234", (868, 735), cv2.FONT_HERSHEY_DUPLEX, 1.25, (240, 240, 240), 2, cv2.LINE_AA)
    # simple face (pedestrian on the right)
    cv2.rectangle(img, (1500, 520), (1600, 900), (110, 60, 40), -1)
    cv2.circle(img, (1550, 470), 48, (150, 180, 220), -1, cv2.LINE_AA)
    for ex in (1533, 1567):
        cv2.circle(img, (ex, 462), 6, (40, 40, 40), -1)
    cv2.ellipse(img, (1550, 492), (16, 8), 0, 0, 180, (40, 40, 90), 3)
    # fake burned-in overlay at the same layout as the real cameras
    cv2.putText(img, "DD-MM-YYYY hh:mm:ss", (960, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (235, 235, 235), 4, cv2.LINE_AA)
    cv2.putText(img, "Cam 0X", (10, 1050), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (235, 235, 235), 4, cv2.LINE_AA)
    return img


def low_light(img):
    """Night/complex-lighting version: dark, vignetted, with headlight glare."""
    f = img.astype(np.float32) * 0.28
    yy, xx = np.mgrid[0:H, 0:W]
    f *= (1 - 0.55 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2) * 2)[..., None]
    for cx in (740, 1200):
        glare = np.exp(-(((xx - cx) / 150.0) ** 2 + ((yy - 560) / 110.0) ** 2))
        f += 255 * 0.9 * glare[..., None]
    return f.clip(0, 255).astype(np.uint8)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def finish(fig, path):
    fig.text(0.01, 0.005, NOTE, fontsize=8, color=MUTED, va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def fig_annotation(frame):
    x1, y1, x2, y2 = LOGO
    cx, cy, bw, bh = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    a1.imshow(rgb(frame)); a1.axis("off")
    a1.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#1baf7a", lw=2.5))
    a1.annotate("logo box", (x2, y1), (x2 + 260, y1 - 160), color=INK, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    a1.set_title("One bounding box per image, tight around the logo", loc="left", fontsize=11)
    # zoomed geometry panel
    pad = 70
    a2.imshow(rgb(frame[y1 - pad:y2 + pad, x1 - pad:x2 + pad]),
              extent=(x1 - pad, x2 + pad, y2 + pad, y1 - pad))
    a2.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#1baf7a", lw=2.5))
    a2.plot([(x1 + x2) / 2], [(y1 + y2) / 2], "o", color="#eb6834", ms=6)
    a2.annotate("", (x2, y2 + 22), (x1, y2 + 22), arrowprops=dict(arrowstyle="<->", color="white"))
    a2.text((x1 + x2) / 2, y2 + 32, "w", ha="center", va="top", color="white", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.15", fc="#0b0b0b", ec="none"))
    a2.annotate("", (x2 + 22, y2), (x2 + 22, y1), arrowprops=dict(arrowstyle="<->", color="white"))
    a2.text(x2 + 33, (y1 + y2) / 2, "h", va="center", color="white", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.15", fc="#0b0b0b", ec="none"))
    a2.text((x1 + x2) / 2 + 8, (y1 + y2) / 2 - 8, "(cx, cy)", fontsize=9, color="#eb6834")
    a2.axis("off")
    a2.set_title("YOLO label (normalised to image size)", loc="left", fontsize=11)
    a2.text(0, -0.12, f"<class_id> <cx> <cy> <w> <h>\n4 {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}",
            transform=a2.transAxes, family="monospace", fontsize=10, va="top")
    finish(fig, ROOT / "visualizations/annotation-examples/yolo-annotation-format.png")


def fig_clahe(frame):
    dark = low_light(frame)
    out = P.apply_clahe(dark, 2.0, 8)
    mse, psnr = P.mse_psnr(dark, out)
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 0.8], hspace=0.12, wspace=0.08)
    for k, (im, t) in enumerate(((dark, "Input: low-light frame with glare"),
                                 (out, "CLAHE on HSV value channel (clip 2.0, tile 8 × 8)"))):
        ax = fig.add_subplot(gs[0, k]); ax.imshow(rgb(im)); ax.axis("off")
        ax.set_title(t, loc="left", fontsize=11)
        hx = fig.add_subplot(gs[1, k], facecolor=SURFACE)
        v = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 2].ravel()
        hx.hist(v, bins=64, range=(0, 255), color="#2a78d6" if k == 0 else "#eb6834")
        hx.set_xlim(0, 255); hx.set_yticks([])
        hx.set_xlabel("V-channel intensity")
        for s in ("top", "right", "left"):
            hx.spines[s].set_visible(False)
    fig.suptitle(f"CLAHE redistributes local contrast  ·  MSE {mse:.1f}, PSNR {psnr:.2f} dB vs input",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    finish(fig, ROOT / "visualizations/preprocessing-examples/clahe-synthetic-demo.png")


def rotate_same_canvas(img, angle):
    m = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (W, H))


def crop_zoom(img, scale):
    ch, cw = int(H * scale), int(W * scale)
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    return cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (W, H))


def add_rain(img, rng=np.random.default_rng(3)):
    out = img.copy()
    for _ in range(900):
        x, y = rng.integers(0, W), rng.integers(0, H)
        cv2.line(out, (x, y), (x + 6, y + 40), (210, 210, 215), 2, cv2.LINE_AA)
    return cv2.addWeighted(out, 0.75, cv2.GaussianBlur(img, (0, 0), 2), 0.25, 0)


def fig_augmentation(frame):
    panels = [("No augmentation", frame), ("Crop 75 % (zoom)", crop_zoom(frame, 0.75)),
              ("Crop 60 % (zoom)", crop_zoom(frame, 0.60)), ("Rotate +20°", rotate_same_canvas(frame, 20)),
              ("Rotate −20°", rotate_same_canvas(frame, -20)), ("Rotate +30°", rotate_same_canvas(frame, 30)),
              ("Rotate 180°", cv2.rotate(frame, cv2.ROTATE_180)), ("Horizontal flip", cv2.flip(frame, 1)),
              ("Vertical flip", cv2.flip(frame, 0)), ("Synthetic rain", add_rain(frame))]
    fig, axes = plt.subplots(2, 5, figsize=(14, 4.2))
    for ax, (t, im) in zip(axes.flat, panels):
        ax.imshow(rgb(cv2.resize(im, (480, 270)))); ax.axis("off")
        ax.set_title(t, loc="left", fontsize=10)
    fig.suptitle("Clip-level augmentation types found in MVLCL (geometry illustrated)", x=0.01,
                 ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.06, "Rotation keeps the canvas size (corners become black). Crop position and "
             "the rain effect's exact parameters are undocumented; shown here for illustration.",
             fontsize=8.5, color=INK2)
    finish(fig, ROOT / "visualizations/preprocessing-examples/augmentation-geometry.png")


def fig_anonymisation(frame):
    try:
        import anonymization as A
    except ImportError as e:  # heavy optional deps (deface/easyocr/ultralytics)
        print("skipped anonymisation figure:", e)
        return
    x1, y1, x2, y2 = LOGO
    step1 = frame.copy()
    mask, _ = A.overlay_mask(frame, "synthetic_frame.jpg")          # timestamp + camera label
    mask[y1:y2, x1:x2] = 0
    step1 = cv2.inpaint(step1, mask, A.INPAINT_RADIUS, cv2.INPAINT_TELEA)
    step2 = step1.copy()
    for bx1, by1, bx2, by2 in [(835, 680, 1105, 760), (1490, 410, 1610, 530)]:   # plate, face
        step2[by1:by2, bx1:bx2] = A.irreversible_blur(step2[by1:by2, bx1:bx2].copy())
    panels = [("1  Input (synthetic)", frame, []),
              ("2  Overlay text removed by inpainting", step1, []),
              ("3  Plate and face irreversibly blurred", step2, []),
              ("4  Logo box left untouched", step2, [LOGO])]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))
    fig.subplots_adjust(wspace=0.04, hspace=0.14)
    for ax, (t, im, boxes) in zip(axes.flat, panels):
        ax.imshow(rgb(cv2.resize(im, (640, 360)))); ax.axis("off")
        ax.set_title(t, loc="left", fontsize=10)
        for bx1, by1, bx2, by2 in boxes:
            s = 640 / W
            ax.add_patch(plt.Rectangle((bx1 * s, by1 * s), (bx2 - bx1) * s, (by2 - by1) * s,
                                       fill=False, ec="#1baf7a", lw=2))
    fig.suptitle("Anonymisation steps (real pipeline functions on a synthetic frame)", x=0.01,
                 ha="left", fontsize=12, fontweight="bold", y=0.97)
    finish(fig, ROOT / "visualizations/preprocessing-examples/anonymisation-steps.png")


if __name__ == "__main__":
    f = synthetic_frame()
    fig_annotation(f)
    fig_clahe(f)
    fig_augmentation(f)
    fig_anonymisation(f)
