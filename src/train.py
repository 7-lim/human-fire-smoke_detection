"""YOLOv8 training entry point.

Reads ``configs/dataset.yaml``, fine-tunes a ``yolov8m.pt`` checkpoint with the
augmentation recipe from the project spec, and copies the best weights to
``models/best.pt`` when training finishes.

Example
-------
    python -m src.train                       # use defaults
    python -m src.train --epochs 200 --batch 8

Augmentation recipe (per project spec)
--------------------------------------
* mosaic=1.0, mixup=0.1
* hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
* flipud=0.0, fliplr=0.5
* degrees=5.0
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_YAML: Path = PROJECT_ROOT / "configs" / "dataset.yaml"
# yolov8s instead of yolov8m — fits comfortably in 6 GB VRAM (GTX 1060 etc.).
# Bump back to yolov8m if you have a 10 GB+ card.
DEFAULT_BASE_MODEL: str = "yolov8s.pt"
DEFAULT_EPOCHS: int = 100
DEFAULT_PATIENCE: int = 20
DEFAULT_IMGSZ: int = 640
# batch=8 leaves ~2 GB free on a 6 GB card with yolov8s @ 640 + AMP.
# _adjust_batch_on_oom halves on CUDA OOM so this is a ceiling, not a floor.
DEFAULT_BATCH: int = 8
# Low worker count keeps the laptop responsive (data loaders run on CPU).
DEFAULT_WORKERS: int = 2
DEFAULT_DEVICE: str = "cuda:0"
# Save a checkpoint every N epochs into runs/<name>/weights/epoch{N}.pt so a
# crash / power loss / Ctrl+C doesn't waste the whole training run.
DEFAULT_SAVE_PERIOD: int = 10
# Stream the dataset from disk — caching everything to RAM eats memory and
# makes a laptop unresponsive. Set to 'ram' only on a dedicated training box.
DEFAULT_CACHE: bool | str = False
# Mixed-precision (fp16) on CUDA — halves VRAM usage and speeds up forward
# pass with no accuracy hit on detection tasks.
DEFAULT_AMP: bool = True
DEFAULT_PROJECT_DIR: Path = PROJECT_ROOT / "runs"
DEFAULT_RUN_NAME: str = "fire_smoke_person"
BEST_WEIGHTS_TARGET: Path = PROJECT_ROOT / "models" / "best.pt"

AUGMENT_KWARGS: dict = dict(
    mosaic=1.0,
    mixup=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.0,
    fliplr=0.5,
    degrees=5.0,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _try_wandb() -> bool:
    """Return True iff wandb is importable — ultralytics enables it auto."""
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


def _adjust_batch_on_oom(train_fn, batch: int, min_batch: int = 2):
    """Run ``train_fn(batch)`` halving on torch OOM until it fits or batch<min."""
    import torch
    while True:
        try:
            return train_fn(batch)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:  # type: ignore[attr-defined]
            msg = str(e).lower()
            if "out of memory" not in msg and "cuda" not in msg:
                raise
            torch.cuda.empty_cache()
            new_batch = max(batch // 2, min_batch)
            if new_batch == batch:
                raise
            logger.warning("CUDA OOM with batch=%d → retrying with batch=%d", batch, new_batch)
            batch = new_batch


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(
    data: Path = DEFAULT_DATASET_YAML,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    workers: int = DEFAULT_WORKERS,
    device: str = DEFAULT_DEVICE,
    save_period: int = DEFAULT_SAVE_PERIOD,
    cache: bool | str = DEFAULT_CACHE,
    amp: bool = DEFAULT_AMP,
    project: Path = DEFAULT_PROJECT_DIR,
    name: str = DEFAULT_RUN_NAME,
    resume: bool = False,
    copy_best_to: Path | None = BEST_WEIGHTS_TARGET,
):
    """Train YOLOv8 on the prepared dataset.

    Returns the ultralytics ``results`` object from ``model.train(...)``.

    Notes
    -----
    * ``save_period=N`` writes ``runs/<name>/weights/epoch{N}.pt`` every N
      epochs, on top of ``best.pt`` / ``last.pt``. Use this so a Ctrl+C / OOM
      / power loss part-way through training doesn't waste everything.
    * Pass ``resume=True`` to pick up from ``runs/<name>/weights/last.pt``.
    * ``cache=False`` keeps the dataset on disk — important on a laptop where
      caching to RAM would freeze the system.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics is required for training. Install with `pip install ultralytics`."
        ) from e

    data = Path(data)
    if not data.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {data}. "
            "Run src.dataset_builder first to generate it."
        )

    if resume:
        resume_pt = Path(project) / name / "weights" / "last.pt"
        if not resume_pt.exists():
            raise FileNotFoundError(
                f"resume=True but no last.pt at {resume_pt}. "
                "Either run a first training pass or set resume=False."
            )
        logger.info("Resuming from: %s", resume_pt)
        model = YOLO(str(resume_pt))
    else:
        logger.info("Loading base model: %s", base_model)
        model = YOLO(base_model)

    if _try_wandb():
        logger.info("wandb is installed — ultralytics will log there automatically.")
    else:
        logger.info("wandb not installed — falling back to TensorBoard logs in runs/.")

    def _do(b: int):
        return model.train(
            data=str(data),
            epochs=epochs,
            patience=patience,
            imgsz=imgsz,
            batch=b,
            workers=workers,
            device=device,
            save_period=save_period,
            cache=cache,
            amp=amp,
            project=str(project),
            name=name,
            exist_ok=True,
            resume=resume,
            **AUGMENT_KWARGS,
        )

    results = _adjust_batch_on_oom(_do, batch)

    # ---- Promote best.pt --------------------------------------------------
    if copy_best_to is not None:
        run_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") else (project / name)
        best = run_dir / "weights" / "best.pt"
        if best.exists():
            copy_best_to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, copy_best_to)
            logger.info("Copied %s → %s", best, copy_best_to)
        else:
            logger.warning("best.pt not found at %s — skipping copy.", best)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train YOLOv8 on the prepared dataset.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATASET_YAML)
    p.add_argument("--model", dest="base_model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--save-period", type=int, default=DEFAULT_SAVE_PERIOD,
                   help="Save checkpoint every N epochs (0 = off).")
    p.add_argument("--cache", default=str(DEFAULT_CACHE),
                   help="False (default), 'ram', or 'disk'. RAM caching eats memory on laptops.")
    p.add_argument("--no-amp", dest="amp", action="store_false",
                   help="Disable mixed-precision training.")
    p.set_defaults(amp=DEFAULT_AMP)
    p.add_argument("--project", type=Path, default=DEFAULT_PROJECT_DIR)
    p.add_argument("--name", default=DEFAULT_RUN_NAME)
    p.add_argument("--resume", action="store_true",
                   help="Resume from runs/<name>/weights/last.pt.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)
    # Allow --cache False / True / 'ram' / 'disk'
    cache_arg: bool | str
    if args.cache.lower() in ("false", "0", "none"):
        cache_arg = False
    elif args.cache.lower() in ("true", "1"):
        cache_arg = True
    else:
        cache_arg = args.cache
    train(
        data=args.data,
        base_model=args.base_model,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        save_period=args.save_period,
        cache=cache_arg,
        amp=args.amp,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
