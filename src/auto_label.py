"""Auto-labeling pipeline for fire / smoke / person detection.

Uses two pretrained YOLOv8 detectors:

* ``yolov8m.pt`` (COCO) — person class (COCO id 0).
* ``betasecond/jimei-fire-smoke-yolo`` (Hugging Face Hub) — fire + smoke.

The HF model id originally used in this project's spec
(``keremberke/yolov8m-fire-detection``) was removed by its author. The
replacement above exposes the same two classes (``Fire``, ``Smoke``) and is
case-insensitively matched by :func:`_resolve_fire_class`.

For every frame, we run both detectors, remap their class ids to our schema
(``fire=0``, ``smoke=1``, ``person=2``) and write a single YOLO ``.txt`` file
to the mirrored output directory.

The schema choice (3-class vs 2-class merged ``fire_smoke``) is *decided*
elsewhere by :func:`src.dataset_builder.decide_class_schema` based on the
fire/smoke frame ratio. This module accepts the schema as input.

Example
-------
    python -m src.auto_label --frames data/frames --labels data/labels
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COCO_PERSON_ID: int = 0
DEFAULT_CONF: float = 0.35
DEFAULT_IOU: float = 0.5
DEFAULT_IMGSZ: int = 640

# Our internal class schemas
SCHEMA_3CLASS: dict[str, int] = {"fire": 0, "smoke": 1, "person": 2}
SCHEMA_2CLASS: dict[str, int] = {"fire_smoke": 0, "person": 1}

# Pretrained model identifiers
# ``COCO_MODEL`` is an ultralytics-hosted name → auto-downloaded by YOLO().
# ``FIRE_MODEL_HF`` is a Hugging Face repo id → fetched via hf_hub_download
# because ultralytics' YOLO() does not natively understand HF repo ids.
COCO_MODEL: str = "yolov8m.pt"
FIRE_MODEL_HF: str = "betasecond/jimei-fire-smoke-yolo"
FIRE_MODEL_HF_FILENAME: str = "yolov8n_fire_smoke.pt"

IMG_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class LabelingReport:
    """Aggregated stats over an auto-labeling run."""

    schema: dict[str, int] = field(default_factory=dict)
    frames_seen: int = 0
    frames_labeled: int = 0
    frames_empty: int = 0          # no detections from either model
    empty_frame_paths: list[str] = field(default_factory=list)
    boxes_per_class: dict[str, int] = field(default_factory=dict)
    confidences_per_class: dict[str, list[float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "frames_seen": self.frames_seen,
            "frames_labeled": self.frames_labeled,
            "frames_empty": self.frames_empty,
            "empty_frame_paths_count": len(self.empty_frame_paths),
            "boxes_per_class": self.boxes_per_class,
            "confidences_per_class": {
                k: {
                    "n": len(v),
                    "mean": float(np.mean(v)) if v else 0.0,
                    "min": float(np.min(v)) if v else 0.0,
                    "max": float(np.max(v)) if v else 0.0,
                }
                for k, v in self.confidences_per_class.items()
            },
        }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _looks_like_hf_repo(model_id: str) -> bool:
    """Heuristic — `org/name` (no extension, no URL scheme, not a local path)."""
    if "/" not in model_id:
        return False
    if model_id.startswith(("http://", "https://", "file://")):
        return False
    if Path(model_id).suffix.lower() in (".pt", ".pth", ".onnx", ".engine"):
        return False
    if Path(model_id).exists():
        return False
    return True


def _resolve_hf(repo_id: str, filename: str | None = None) -> str:
    """Download a YOLO weight file from HF Hub and return its local path.

    If ``filename`` is not given, tries a list of common defaults.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to load HF repo models. "
            "Install with `pip install huggingface_hub`."
        ) from e

    candidates = [filename] if filename else [
        "best.pt", "weights/best.pt", "model.pt", "yolov8m.pt", "yolov8n.pt",
    ]
    last_err: Exception | None = None
    for fn in candidates:
        if fn is None:
            continue
        try:
            local = hf_hub_download(repo_id=repo_id, filename=fn)
            logger.info("Downloaded %s/%s -> %s", repo_id, fn, local)
            return local
        except Exception as e:
            last_err = e
            logger.debug("hf_hub_download(%s, %s) failed: %s", repo_id, fn, e)
    raise FileNotFoundError(
        f"Could not download any YOLO weights from HF repo '{repo_id}' "
        f"(tried {candidates}). Last error: {last_err}"
    )


def _load_yolo(model_id: str, hf_filename: str | None = None):
    """Lazily import ultralytics and return a YOLO model.

    Accepts:
      * local path (e.g. ``"models/best.pt"``)
      * ultralytics-hosted name (e.g. ``"yolov8m.pt"`` — auto-downloaded)
      * Hugging Face repo id (e.g. ``"org/name"``) — fetched via
        :func:`huggingface_hub.hf_hub_download`; pass ``hf_filename`` to
        override the default file lookup.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics is required for auto-labeling. "
            "Install with `pip install ultralytics`."
        ) from e

    if _looks_like_hf_repo(model_id):
        local = _resolve_hf(model_id, hf_filename)
        return YOLO(local)

    logger.info("Loading model: %s", model_id)
    return YOLO(model_id)


def load_detectors(
    coco_model: str = COCO_MODEL,
    fire_model: str = FIRE_MODEL_HF,
    fire_filename: str | None = FIRE_MODEL_HF_FILENAME,
    device: str | int = "cuda:0",
):
    """Load both YOLOv8 detectors. Returns ``(coco, fire)`` tuple."""
    coco = _load_yolo(coco_model)
    fire = _load_yolo(fire_model, hf_filename=fire_filename)
    try:
        coco.to(device)
        fire.to(device)
    except Exception as e:
        logger.warning("Could not move models to %s (%s) — falling back to CPU.", device, e)
    return coco, fire


# ---------------------------------------------------------------------------
# Box helpers (YOLO format = normalized [x_center, y_center, w, h])
# ---------------------------------------------------------------------------
def _xyxy_to_yolo(xyxy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Convert ``(x1, y1, x2, y2)`` pixel boxes → normalized ``(cx, cy, w, h)``."""
    x1, y1, x2, y2 = xyxy.T
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    out = np.stack([cx, cy, w, h], axis=1)
    return np.clip(out, 0.0, 1.0)


def _resolve_fire_class(model, name_lower: str) -> int | None:
    """Return the model-side class id matching ``name_lower`` (fire/smoke), or None.

    Fire detector class names vary across HF mirrors — some use 'fire'/'smoke',
    others 'Fire'/'Smoke', others a single 'fire' class. We match case-insensitively.
    """
    for cid, cname in model.names.items():
        if str(cname).strip().lower() == name_lower:
            return int(cid)
    return None


# ---------------------------------------------------------------------------
# Per-frame labeling
# ---------------------------------------------------------------------------
def _detect(model, img_path: Path, conf: float, iou: float, imgsz: int):
    """Run a YOLOv8 model on one image. Returns ``(boxes_xyxy, scores, cls_ids)``."""
    res = model.predict(
        source=str(img_path), conf=conf, iou=iou, imgsz=imgsz,
        verbose=False,
    )
    if not res:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    r = res[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    xyxy = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    return xyxy, scores, cls_ids


def label_image(
    img_path: Path,
    coco_model,
    fire_model,
    schema: dict[str, int],
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
) -> tuple[list[str], dict[str, list[float]]]:
    """Run both detectors on one frame, return YOLO-format label lines + per-class confidences.

    The schema decides how fire vs. smoke are mapped:
      * 3-class: fire→0, smoke→1, person→2
      * 2-class: fire and smoke both → fire_smoke→0, person→1
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return [], {}
    h, w = img.shape[:2]

    lines: list[str] = []
    conf_by_class: dict[str, list[float]] = {k: [] for k in schema.keys()}

    # ---- Person from COCO ------------------------------------------------
    xyxy, scores, cls_ids = _detect(coco_model, img_path, conf, iou, imgsz)
    mask = cls_ids == COCO_PERSON_ID
    if mask.any():
        person_cls = schema["person"]
        yolo = _xyxy_to_yolo(xyxy[mask], w, h)
        for (cx, cy, bw, bh), s in zip(yolo, scores[mask]):
            lines.append(f"{person_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            conf_by_class["person"].append(float(s))

    # ---- Fire + smoke from HF model -------------------------------------
    fire_cid = _resolve_fire_class(fire_model, "fire")
    smoke_cid = _resolve_fire_class(fire_model, "smoke")

    if fire_cid is not None or smoke_cid is not None:
        xyxy, scores, cls_ids = _detect(fire_model, img_path, conf, iou, imgsz)

        if "fire" in schema:
            # 3-class
            if fire_cid is not None:
                m = cls_ids == fire_cid
                if m.any():
                    yolo = _xyxy_to_yolo(xyxy[m], w, h)
                    for (cx, cy, bw, bh), s in zip(yolo, scores[m]):
                        lines.append(f"{schema['fire']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        conf_by_class["fire"].append(float(s))
            if smoke_cid is not None:
                m = cls_ids == smoke_cid
                if m.any():
                    yolo = _xyxy_to_yolo(xyxy[m], w, h)
                    for (cx, cy, bw, bh), s in zip(yolo, scores[m]):
                        lines.append(f"{schema['smoke']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        conf_by_class["smoke"].append(float(s))
        else:
            # 2-class merged
            merged_cls = schema["fire_smoke"]
            valid_ids = {cid for cid in (fire_cid, smoke_cid) if cid is not None}
            m = np.isin(cls_ids, list(valid_ids))
            if m.any():
                yolo = _xyxy_to_yolo(xyxy[m], w, h)
                for (cx, cy, bw, bh), s in zip(yolo, scores[m]):
                    lines.append(f"{merged_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    conf_by_class["fire_smoke"].append(float(s))

    return lines, conf_by_class


# ---------------------------------------------------------------------------
# Walk the frames directory and label everything
# ---------------------------------------------------------------------------
def _iter_images(root: Path) -> Iterable[Path]:
    for ext in IMG_EXTENSIONS:
        yield from root.rglob(f"*{ext}")


def auto_label_directory(
    frames_root: Path,
    labels_root: Path,
    schema: dict[str, int],
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
    device: str | int = "cuda:0",
    coco_model_id: str = COCO_MODEL,
    fire_model_id: str = FIRE_MODEL_HF,
    fire_filename: str | None = FIRE_MODEL_HF_FILENAME,
    report_path: Path | None = None,
) -> LabelingReport:
    """Label every image under ``frames_root`` and mirror outputs to ``labels_root``.

    Frames with no detections still get logged to ``empty_frame_paths`` so the
    caller can inspect / discard them; no label file is written for those (a
    missing file is interpreted by :mod:`dataset_builder` as 'unlabeled').
    """
    frames_root = Path(frames_root)
    labels_root = Path(labels_root)
    labels_root.mkdir(parents=True, exist_ok=True)

    coco_model, fire_model = load_detectors(
        coco_model_id, fire_model_id, fire_filename=fire_filename, device=device,
    )

    report = LabelingReport(schema=dict(schema))
    report.boxes_per_class = {k: 0 for k in schema.keys()}
    report.confidences_per_class = {k: [] for k in schema.keys()}

    images = sorted(_iter_images(frames_root))
    if not images:
        logger.warning("No images found under %s", frames_root)
        return report

    for img_path in tqdm(images, desc="auto-labeling"):
        report.frames_seen += 1
        lines, conf_map = label_image(
            img_path, coco_model, fire_model, schema, conf=conf, iou=iou, imgsz=imgsz,
        )

        if not lines:
            report.frames_empty += 1
            report.empty_frame_paths.append(str(img_path))
            continue

        rel = img_path.relative_to(frames_root).with_suffix(".txt")
        out = labels_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines), encoding="utf-8")
        report.frames_labeled += 1

        for cname, scores in conf_map.items():
            report.confidences_per_class[cname].extend(scores)
            report.boxes_per_class[cname] += len(scores)

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        logger.info("Wrote labeling report to %s", report_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auto-label frames with two YOLOv8 models.")
    p.add_argument("--frames", required=True, type=Path, help="Root of extracted frames.")
    p.add_argument("--labels", required=True, type=Path, help="Output root for label .txt files.")
    p.add_argument("--schema", choices=["3", "2", "auto"], default="auto",
                   help="Class schema. 'auto' delegates to dataset_builder.decide_class_schema.")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--iou", type=float, default=DEFAULT_IOU)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--report", type=Path, default=None,
                   help="Optional path for JSON labeling report.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)

    if args.schema == "3":
        schema = SCHEMA_3CLASS
    elif args.schema == "2":
        schema = SCHEMA_2CLASS
    else:
        from src.dataset_builder import decide_class_schema
        schema, _info = decide_class_schema(args.frames)

    report = auto_label_directory(
        frames_root=args.frames,
        labels_root=args.labels,
        schema=schema,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        report_path=args.report,
    )
    print(json.dumps(report.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
