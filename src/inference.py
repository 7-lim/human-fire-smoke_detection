"""Real-time inference for fire / smoke / person detection.

Supports webcam, video file, or a single image — all routed through one CLI.

Example
-------
    # Webcam (live demo)
    python -m src.inference --source 0

    # Process a video file and save the annotated result
    python -m src.inference --source data/raw/mixed/roomfire41.mp4 --save

    # Annotate a single image
    python -m src.inference --source path/to/photo.jpg --save

Keyboard during video / webcam display:
    q : quit
    s : save the current frame to outputs/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS: Path = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
# Network conf — keep low so we collect candidates, then filter per-class.
DEFAULT_CONF: float = 0.20
DEFAULT_IOU: float = 0.5
DEFAULT_IMGSZ: int = 640
DEFAULT_DEVICE: str = "cuda:0"

# Per-class confidence thresholds applied AFTER YOLO inference. Smoke is the
# unreliable class on this model — large smooth surfaces (walls, ceilings) fire
# spurious smoke detections, so we keep its bar high. Fire stays low because
# the model rarely false-positives fire on indoor scenes.
PER_CLASS_CONF: dict[str, float] = {
    "fire":       0.30,
    "smoke":      0.65,        # walls / ceilings still leak past 0.50
    "person":     0.35,
    "fire_smoke": 0.35,
}
# ALERT must be more conservative than the display threshold — better to miss
# a real fire by a half-second than blink a fake alert at a blank wall.
ALERT_CONF: dict[str, float] = {
    "fire":       0.40,
    "smoke":      0.75,
    "fire_smoke": 0.55,
}
# Sanity gate — a "smoke" box covering more than this fraction of the frame
# is almost always a false positive on a uniform background (wall, ceiling,
# sky). Real smoke has texture and clear extents; YOLO mistaking a flat
# surface for smoke tends to emit a box that fills most of the view.
MAX_BOX_FRAME_FRACTION: dict[str, float] = {
    "smoke":      0.50,
    "fire_smoke": 0.55,
}

CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "fire": (0, 0, 255),         # red
    "smoke": (255, 128, 0),      # blue-ish
    "person": (0, 255, 0),       # green
    "fire_smoke": (0, 0, 255),   # red (merged schema)
}
ALERT_CLASSES: tuple[str, ...] = ("fire", "smoke", "fire_smoke")

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

FPS_EMA: float = 0.9             # smoothing for FPS counter
ALERT_BLINK_PERIOD: float = 0.5  # seconds — alert overlay toggles at this rate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _color_for(cname: str) -> tuple[int, int, int]:
    return CLASS_COLORS_BGR.get(cname, (200, 200, 200))


def draw_detections(frame: np.ndarray, result, names: dict[int, str]) -> tuple[np.ndarray, list[str]]:
    """Draw boxes + labels. Returns ``(annotated_frame, fired_class_names)``.

    Three filters applied per detection:
      * ``PER_CLASS_CONF[cname]`` — minimum score required to *draw* the box.
      * ``MAX_BOX_FRAME_FRACTION[cname]`` — boxes covering more of the frame
        than this fraction get dropped (catches smoke false-positives on
        uniform backgrounds like walls).
      * ``ALERT_CONF[cname]``     — minimum score required to *trigger* the
        red ALERT overlay (stricter than the draw threshold).
    """
    fired: list[str] = []
    if result.boxes is None or len(result.boxes) == 0:
        return frame, fired

    xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)

    frame_h, frame_w = frame.shape[:2]
    frame_area = float(frame_h * frame_w)

    for (x1, y1, x2, y2), s, c in zip(xyxy, confs, cls_ids):
        cname = names.get(int(c), str(c))
        # Per-class confidence gate — silently drop low-confidence detections
        # for unreliable classes like smoke.
        if float(s) < PER_CLASS_CONF.get(cname, 0.0):
            continue
        # Area gate — drop oversized boxes for classes where the model tends
        # to fire on uniform backgrounds. Wall = entire frame = ~95% area.
        max_frac = MAX_BOX_FRAME_FRACTION.get(cname)
        if max_frac is not None:
            box_area = float(max(0, x2 - x1) * max(0, y2 - y1))
            if box_area / max(frame_area, 1.0) > max_frac:
                continue
        color = _color_for(cname)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{cname} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        # Independent ALERT gate — drawing a smoke box at 0.55 is fine, but
        # screaming ALERT at every wall behind the user is not.
        if cname in ALERT_CLASSES and float(s) >= ALERT_CONF.get(cname, 1.0):
            fired.append(cname)
    return frame, fired


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    text = f"FPS: {fps:5.1f}"
    cv2.putText(frame, text, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(frame, text, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
    return frame


def draw_alert(frame: np.ndarray, t: float) -> np.ndarray:
    """Flash a red ⚠ ALERT overlay top-right (toggles every ALERT_BLINK_PERIOD)."""
    on = (int(t / ALERT_BLINK_PERIOD) % 2) == 0
    if not on:
        return frame
    h, w = frame.shape[:2]
    text = "! ALERT !"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 3)
    x1 = w - tw - 30
    y1 = 10
    x2 = w - 10
    y2 = y1 + th + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    cv2.putText(frame, text, (x1 + 10, y2 - 10),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 3)
    return frame


# ---------------------------------------------------------------------------
# Source routing
# ---------------------------------------------------------------------------
def _parse_source(source: str) -> tuple[str, int | str]:
    """Classify the ``--source`` value.

    Returns ``(kind, value)`` where ``kind`` ∈ {'webcam', 'video', 'image'}.
    """
    if source.isdigit():
        return "webcam", int(source)
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")
    if p.suffix.lower() in IMAGE_EXTENSIONS:
        return "image", str(p)
    return "video", str(p)


# ---------------------------------------------------------------------------
# Image / video pipelines
# ---------------------------------------------------------------------------
def run_on_image(
    model, image_path: str, conf: float, iou: float, imgsz: int,
    output_dir: Path, save: bool, show: bool,
) -> Path | None:
    """Annotate one image. Writes to ``output_dir`` when ``save`` is True."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    res = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    annotated, _ = draw_detections(img, res, model.names)

    out_path: Path | None = None
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{Path(image_path).stem}_annotated.jpg"
        cv2.imwrite(str(out_path), annotated)
        logger.info("Wrote %s", out_path)

    if show:
        cv2.imshow("inference", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return out_path


def run_on_stream(
    model, source: int | str, conf: float, iou: float, imgsz: int,
    output_dir: Path, save: bool, show: bool, is_webcam: bool,
) -> Path | None:
    """Run detection on a video file or webcam stream."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Could not open source: {source!r}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer: cv2.VideoWriter | None = None
    out_path: Path | None = None
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = "webcam" if is_webcam else Path(str(source)).stem
        out_path = output_dir / f"{stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (src_w, src_h))

    fps_ema = 0.0
    t_prev = time.time()
    t_start = t_prev

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
            annotated, fired = draw_detections(frame, res, model.names)

            # FPS
            now = time.time()
            dt = max(now - t_prev, 1e-6)
            t_prev = now
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema == 0 else (FPS_EMA * fps_ema + (1 - FPS_EMA) * inst_fps)
            annotated = draw_fps(annotated, fps_ema)

            if fired:
                annotated = draw_alert(annotated, now - t_start)

            if writer is not None:
                writer.write(annotated)

            if show:
                cv2.imshow("inference", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    output_dir.mkdir(parents=True, exist_ok=True)
                    snap = output_dir / f"snapshot_{int(now * 1000)}.jpg"
                    cv2.imwrite(str(snap), annotated)
                    logger.info("Saved snapshot %s", snap)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if out_path is not None:
        logger.info("Wrote %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
def run(
    source: str,
    weights: Path = DEFAULT_WEIGHTS,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
    device: str = DEFAULT_DEVICE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    save: bool = False,
    show: bool | None = None,
):
    """Top-level dispatch — pick image/video/webcam mode and run."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics is required for inference. Install with `pip install ultralytics`."
        ) from e

    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights}. "
            f"Train first (src/train.py) or pass --weights."
        )

    model = YOLO(str(weights))
    try:
        model.to(device)
    except Exception as e:
        logger.warning("Could not move model to %s (%s) — using default device.", device, e)

    kind, value = _parse_source(source)
    if show is None:
        show = (kind == "webcam")   # default: only show window for live webcam

    if kind == "image":
        return run_on_image(model, value, conf, iou, imgsz, output_dir, save, show)
    return run_on_stream(model, value, conf, iou, imgsz, output_dir, save, show,
                        is_webcam=(kind == "webcam"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run YOLOv8 inference (image / video / webcam).")
    p.add_argument("--source", required=True,
                   help="Webcam index (e.g. 0), or path to image / video.")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--iou", type=float, default=DEFAULT_IOU)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--output", dest="output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--save", action="store_true", help="Save annotated output to outputs/.")
    p.add_argument("--show", dest="show", action="store_true",
                   help="Force-display a window (default: only for webcam).")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="Suppress the display window.")
    p.set_defaults(show=None)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)
    run(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        output_dir=args.output_dir,
        save=args.save,
        show=args.show,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
