"""Bullet-proof launcher for labelImg on this project's data.

labelImg's ``classes.txt`` write logic rewrites the file on every save from its
in-memory ``label_hist``. If you launch ``labelImg`` without the predefined
classes file as the 2nd CLI argument, ``label_hist`` starts empty and the very
next save truncates ``classes.txt`` to whatever subset of classes you happen to
have annotated this session — silently corrupting the dataset schema and
guaranteeing an ``IndexError: list index out of range`` on the next image
load.

Always launch labelImg through this wrapper. It pre-loads the project's full
3-class schema (``fire``, ``smoke``, ``person``) before any annotation work.

Examples
--------
    python scripts/labelimg.py             # defaults to the train split
    python scripts/labelimg.py train
    python scripts/labelimg.py val
    python scripts/labelimg.py test
    python scripts/labelimg.py --raw       # raw frames in data/frames/
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
PREDEFINED_CLASSES: Path = ROOT / "configs" / "predefined_classes.txt"
DATASET_ROOT: Path = ROOT / "data" / "dataset"
RAW_FRAMES_ROOT: Path = ROOT / "data" / "frames"
RAW_LABELS_ROOT: Path = ROOT / "data" / "labels"
VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")


def _resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return ``(image_dir, save_dir)`` based on CLI args."""
    if args.raw:
        return RAW_FRAMES_ROOT, RAW_LABELS_ROOT
    img = DATASET_ROOT / "images" / args.split
    lbl = DATASET_ROOT / "labels" / args.split
    return img, lbl


def _refresh_classes_txt(label_dir: Path) -> None:
    """Copy the canonical predefined_classes.txt to ``label_dir/classes.txt``.

    Restores the file in case a previous labelImg session truncated it.
    Without this, a corrupted classes.txt causes ``IndexError`` on load.
    """
    label_dir.mkdir(parents=True, exist_ok=True)
    target = label_dir / "classes.txt"
    if not PREDEFINED_CLASSES.exists():
        print(
            f"ERROR: {PREDEFINED_CLASSES} is missing — run "
            "src.dataset_builder first.",
            file=sys.stderr,
        )
        sys.exit(2)
    shutil.copy2(PREDEFINED_CLASSES, target)


def _find_labelimg_executable() -> str:
    """Return the labelImg executable path (works inside or outside a venv)."""
    candidate = Path(sys.executable).parent / (
        "labelImg.exe" if os.name == "nt" else "labelImg"
    )
    if candidate.exists():
        return str(candidate)
    found = shutil.which("labelImg") or shutil.which("labelImg.exe")
    if found:
        return found
    print(
        "ERROR: labelImg executable not found. Install with `pip install labelImg`.",
        file=sys.stderr,
    )
    sys.exit(127)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "split", nargs="?", default="train", choices=VALID_SPLITS,
        help="Which dataset split to edit (default: train).",
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Edit raw frames in data/frames/ instead of the split dataset.",
    )
    args = parser.parse_args(argv)

    img_dir, lbl_dir = _resolve_dirs(args)
    if not img_dir.exists():
        print(f"ERROR: image directory does not exist: {img_dir}", file=sys.stderr)
        return 1

    _refresh_classes_txt(lbl_dir)

    exe = _find_labelimg_executable()
    cmd = [exe, str(img_dir), str(PREDEFINED_CLASSES), str(lbl_dir)]
    print("Launching:", " ".join(cmd))
    # Run in-process so the user's terminal stays attached and Ctrl+C works.
    os.execv(exe, cmd)
    return 0  # unreachable


if __name__ == "__main__":
    sys.exit(main())
