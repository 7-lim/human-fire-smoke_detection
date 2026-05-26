"""Dataset assembly: decide class schema, split train/val/test, emit dataset.yaml.

Pipeline overview
-----------------
1. :func:`decide_class_schema` counts extracted frames in ``fire/`` and ``smoke/``.
   If their ratio is between 0.5 and 2.0 → 3-class schema. Otherwise → merged
   ``fire_smoke`` + ``person`` 2-class schema (with a warning).
2. :func:`collect_pairs` walks ``data/frames`` and pairs every image with its
   ``data/labels`` ``.txt`` (frames without a label file are dropped — those are
   the unlabeled negatives).
3. :func:`split_pairs` splits *by source video* (frames from the same video
   never span splits — avoids leakage) with a stratified 75 / 15 / 10 plan.
4. :func:`copy_split` mirrors the chosen pairs into
   ``data/dataset/images/<split>/`` and ``data/dataset/labels/<split>/``.
5. :func:`write_dataset_yaml` emits ``configs/dataset.yaml`` so YOLO can find
   the splits.

Example
-------
    python -m src.dataset_builder \\
        --frames data/frames --labels data/labels \\
        --dataset data/dataset --config configs/dataset.yaml
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_FRAC: float = 0.75
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.10            # 1 - TRAIN_FRAC - VAL_FRAC

CLASS_RATIO_MIN: float = 0.5       # if fire/smoke ratio < this → merge
CLASS_RATIO_MAX: float = 2.0       # if fire/smoke ratio > this → merge

SCHEMA_3CLASS: dict[str, int] = {"fire": 0, "smoke": 1, "person": 2}
SCHEMA_2CLASS: dict[str, int] = {"fire_smoke": 0, "person": 1}

IMG_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

RANDOM_SEED: int = 1337


# ---------------------------------------------------------------------------
# Class schema decision
# ---------------------------------------------------------------------------
@dataclass
class SchemaDecision:
    """Result of inspecting fire/smoke frame counts."""

    fire_frames: int
    smoke_frames: int
    ratio: float
    merged: bool
    schema: dict[str, int]
    reason: str


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for ext in IMG_EXTENSIONS for _ in folder.rglob(f"*{ext}"))


def decide_class_schema(frames_root: Path) -> tuple[dict[str, int], SchemaDecision]:
    """Pick 3-class vs 2-class merged schema by inspecting frame counts.

    Returns
    -------
    (schema, decision)
        ``schema`` is the chosen mapping name → class id.
        ``decision`` carries the counts and the human-readable rationale.
    """
    frames_root = Path(frames_root)
    fire_n = _count_images(frames_root / "fire")
    smoke_n = _count_images(frames_root / "smoke")

    if fire_n == 0 or smoke_n == 0:
        ratio = float("inf") if smoke_n == 0 else 0.0
        reason = (
            f"One source has zero frames (fire={fire_n}, smoke={smoke_n}). "
            "Falling back to merged 2-class schema."
        )
        logger.warning(reason)
        decision = SchemaDecision(fire_n, smoke_n, ratio, True, SCHEMA_2CLASS, reason)
        print(f"\n[schema] {reason}")
        print(f"[schema] Using 2-class: {SCHEMA_2CLASS}\n")
        return SCHEMA_2CLASS, decision

    ratio = fire_n / smoke_n
    if CLASS_RATIO_MIN <= ratio <= CLASS_RATIO_MAX:
        reason = (
            f"Balanced (fire={fire_n}, smoke={smoke_n}, ratio={ratio:.2f} ∈ "
            f"[{CLASS_RATIO_MIN}, {CLASS_RATIO_MAX}]). Using 3-class schema."
        )
        logger.info(reason)
        decision = SchemaDecision(fire_n, smoke_n, ratio, False, SCHEMA_3CLASS, reason)
        print(f"\n[schema] {reason}")
        print(f"[schema] Using 3-class: {SCHEMA_3CLASS}\n")
        return SCHEMA_3CLASS, decision

    reason = (
        f"WARNING — imbalanced (fire={fire_n}, smoke={smoke_n}, ratio={ratio:.2f} "
        f"outside [{CLASS_RATIO_MIN}, {CLASS_RATIO_MAX}]). Merging fire+smoke "
        f"into a single 'fire_smoke' class to avoid training instability."
    )
    logger.warning(reason)
    decision = SchemaDecision(fire_n, smoke_n, ratio, True, SCHEMA_2CLASS, reason)
    print(f"\n[schema] {reason}")
    print(f"[schema] Using 2-class: {SCHEMA_2CLASS}\n")
    return SCHEMA_2CLASS, decision


# ---------------------------------------------------------------------------
# Pair collection
# ---------------------------------------------------------------------------
@dataclass
class FramePair:
    """One (image, label) pair plus the source-video stem for split grouping."""

    image: Path
    label: Path
    source: str          # video stem the frame came from


def _source_from_image(img_path: Path) -> str:
    """Derive the source-video identifier from the image filename.

    Our naming convention from :mod:`extract_frames` is
    ``{video_stem}_{frame_idx:06d}.jpg``. We strip the trailing ``_NNNNNN`` and
    keep the parent class folder (``fire``, ``smoke``, ``mixed``) as a prefix —
    that way two videos called ``posVideo1`` under different class folders are
    treated as distinct sources.
    """
    stem = img_path.stem
    if "_" in stem:
        parts = stem.rsplit("_", 1)
        if parts[1].isdigit():
            stem = parts[0]
    parent = img_path.parent.name
    return f"{parent}/{stem}"


def collect_pairs(frames_root: Path, labels_root: Path) -> list[FramePair]:
    """Walk frames + labels and return every pair that has both files."""
    frames_root = Path(frames_root)
    labels_root = Path(labels_root)

    pairs: list[FramePair] = []
    skipped_no_label = 0
    for ext in IMG_EXTENSIONS:
        for img in frames_root.rglob(f"*{ext}"):
            rel = img.relative_to(frames_root).with_suffix(".txt")
            lbl = labels_root / rel
            if not lbl.exists():
                skipped_no_label += 1
                continue
            pairs.append(FramePair(image=img, label=lbl, source=_source_from_image(img)))
    logger.info("Collected %d labeled pairs (%d images had no label and were dropped).",
                len(pairs), skipped_no_label)
    return pairs


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def split_pairs(
    pairs: list[FramePair],
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = RANDOM_SEED,
) -> dict[str, list[FramePair]]:
    """Split pairs into train/val/test by source-video (no leakage).

    Strategy:
      * group pairs by ``source``
      * shuffle source list with a fixed seed
      * walk in order, assigning whole sources to the split that is most
        below its quota (computed by current image count, not source count —
        so videos with more frames are balanced naturally).
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train + val + test fractions must sum to 1.")

    by_source: dict[str, list[FramePair]] = defaultdict(list)
    for p in pairs:
        by_source[p.source].append(p)

    rng = random.Random(seed)
    sources = list(by_source.keys())
    rng.shuffle(sources)

    total = len(pairs)
    targets = {
        "train": int(round(total * train_frac)),
        "val": int(round(total * val_frac)),
        "test": total - int(round(total * train_frac)) - int(round(total * val_frac)),
    }
    splits: dict[str, list[FramePair]] = {"train": [], "val": [], "test": []}

    for src in sources:
        chunk = by_source[src]
        # assign whole source to the split with the largest remaining deficit
        deficits = {k: targets[k] - len(splits[k]) for k in splits}
        chosen = max(deficits, key=lambda k: deficits[k])
        splits[chosen].extend(chunk)

    logger.info("Split counts → train=%d val=%d test=%d (targets %s)",
                len(splits["train"]), len(splits["val"]), len(splits["test"]),
                targets)
    return splits


# ---------------------------------------------------------------------------
# Copy + yaml
# ---------------------------------------------------------------------------
def copy_split(splits: dict[str, list[FramePair]], dataset_root: Path) -> None:
    """Copy images + labels to ``dataset_root/{images,labels}/{split}/``."""
    dataset_root = Path(dataset_root)
    for split, pairs in splits.items():
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for p in pairs:
            shutil.copy2(p.image, img_dir / p.image.name)
            shutil.copy2(p.label, lbl_dir / p.label.name)


def write_classes_txt(
    dataset_root: Path,
    schema: dict[str, int],
    predefined_path: Path | None = Path("configs/predefined_classes.txt"),
) -> None:
    """Emit ``classes.txt`` next to every split's labels folder.

    labelImg requires this file when editing YOLO-format annotations — it
    lists one class name per line, ordered by class id, so the integer ids in
    the ``.txt`` annotations resolve back to human-readable names in the UI.

    Also writes ``configs/predefined_classes.txt`` (same content) when
    ``predefined_path`` is given. Pass that file to labelImg's CLI so the
    in-memory ``label_hist`` starts populated with every class — otherwise
    labelImg overwrites ``classes.txt`` on each save with whatever subset of
    classes you've already drawn this session, silently corrupting the
    project's class schema.
    """
    dataset_root = Path(dataset_root)
    names = [name for name, _ in sorted(schema.items(), key=lambda kv: kv[1])]
    payload = "\n".join(names) + "\n"
    for split in ("train", "val", "test"):
        lbl_dir = dataset_root / "labels" / split
        if not lbl_dir.exists():
            continue
        (lbl_dir / "classes.txt").write_text(payload, encoding="utf-8")
    if predefined_path is not None:
        predefined_path = Path(predefined_path)
        predefined_path.parent.mkdir(parents=True, exist_ok=True)
        predefined_path.write_text(payload, encoding="utf-8")
    logger.info(
        "Wrote classes.txt to every labels/<split>/ (and %s) — names=%s",
        predefined_path, names,
    )


def class_distribution(
    splits: dict[str, list[FramePair]], schema: dict[str, int],
) -> dict[str, dict[str, int]]:
    """Count boxes per class per split — useful sanity check."""
    inv = {v: k for k, v in schema.items()}
    out: dict[str, dict[str, int]] = {}
    for split, pairs in splits.items():
        counter: Counter = Counter()
        for p in pairs:
            for line in p.label.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    cls_id = int(line.split()[0])
                except (IndexError, ValueError):
                    continue
                counter[inv.get(cls_id, f"unknown_{cls_id}")] += 1
        out[split] = dict(counter)
    return out


def write_dataset_yaml(
    config_path: Path, dataset_root: Path, schema: dict[str, int],
) -> None:
    """Emit a YOLOv8 ``dataset.yaml`` for the assembled splits."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    names = [name for name, _ in sorted(schema.items(), key=lambda kv: kv[1])]
    config = {
        "path": str(Path(dataset_root).as_posix()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    logger.info("Wrote dataset config to %s (nc=%d, names=%s)",
                config_path, config["nc"], names)


# ---------------------------------------------------------------------------
# End-to-end builder
# ---------------------------------------------------------------------------
def build_dataset(
    frames_root: Path,
    labels_root: Path,
    dataset_root: Path,
    config_path: Path,
    schema: dict[str, int] | None = None,
    seed: int = RANDOM_SEED,
) -> dict:
    """Run the full pipeline: decide schema → collect → split → copy → yaml.

    Returns a summary dict. If ``schema`` is provided, the decision step is
    skipped (use this if you already labeled with a chosen schema).
    """
    frames_root = Path(frames_root)
    labels_root = Path(labels_root)
    dataset_root = Path(dataset_root)

    if schema is None:
        schema, decision = decide_class_schema(frames_root)
    else:
        decision = None

    pairs = collect_pairs(frames_root, labels_root)
    if not pairs:
        raise RuntimeError(
            f"No labeled (image, label) pairs found. "
            f"Ran on frames={frames_root}, labels={labels_root}. "
            f"Did you run auto_label first?"
        )

    splits = split_pairs(pairs, seed=seed)
    copy_split(splits, dataset_root)
    write_dataset_yaml(config_path, dataset_root, schema)
    write_classes_txt(dataset_root, schema)

    dist = class_distribution(splits, schema)
    summary = {
        "schema": schema,
        "decision": decision.__dict__ if decision else None,
        "splits": {k: len(v) for k, v in splits.items()},
        "class_distribution": dist,
        "dataset_root": str(dataset_root),
        "config_path": str(config_path),
    }

    print("\n=== Dataset build summary ===")
    print(f"  schema     : {schema}")
    print(f"  splits     : {summary['splits']}")
    print(f"  by class   :")
    for split, counts in dist.items():
        print(f"    {split:6s}: {counts}")
    print(f"  yaml       : {config_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Assemble the YOLO dataset.")
    p.add_argument("--frames", required=True, type=Path, default=Path("data/frames"))
    p.add_argument("--labels", required=True, type=Path, default=Path("data/labels"))
    p.add_argument("--dataset", required=True, type=Path, default=Path("data/dataset"))
    p.add_argument("--config", required=True, type=Path, default=Path("configs/dataset.yaml"))
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)
    build_dataset(
        frames_root=args.frames,
        labels_root=args.labels,
        dataset_root=args.dataset,
        config_path=args.config,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
