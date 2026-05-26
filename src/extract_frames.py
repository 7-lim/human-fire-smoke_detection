"""Frame extraction utility for video → image dataset preparation.

Reads a video (or every video in a folder), keeps every N-th frame, drops blurry
and near-duplicate frames, letterboxes them to a square target size, and writes
them out as high-quality JPEGs.

Designed to be called either from a notebook (``extract_frames_from_video``,
``extract_frames_from_dir``) or directly from the command line.

Example
-------
    python -m src.extract_frames \\
        --input data/raw --output data/frames --sample-rate 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

try:
    import imagehash
    from PIL import Image
    _HAS_IMAGEHASH = True
except Exception:
    _HAS_IMAGEHASH = False


# ---------------------------------------------------------------------------
# Constants (all tunables live here)
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_RATE: int = 10           # keep every N-th frame
DEFAULT_TARGET_SIZE: int = 640          # YOLO standard input size
DEFAULT_BLUR_THRESHOLD: float = 100.0   # Laplacian variance threshold
DEFAULT_HASH_DISTANCE: int = 4          # perceptual-hash Hamming distance
DEFAULT_JPEG_QUALITY: int = 95
LETTERBOX_PAD_COLOR: tuple[int, int, int] = (114, 114, 114)  # YOLO grey
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stats container
# ---------------------------------------------------------------------------
@dataclass
class ExtractionStats:
    """Per-video extraction statistics."""

    video: str
    total_frames: int = 0
    extracted: int = 0
    skipped_blur: int = 0
    skipped_duplicate: int = 0
    skipped_unreadable: int = 0
    output_dir: str = ""

    def as_dict(self) -> dict:
        return {
            "video": self.video,
            "total_frames": self.total_frames,
            "extracted": self.extracted,
            "skipped_blur": self.skipped_blur,
            "skipped_duplicate": self.skipped_duplicate,
            "skipped_unreadable": self.skipped_unreadable,
            "output_dir": self.output_dir,
        }


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def laplacian_variance(gray: np.ndarray) -> float:
    """Return the Laplacian variance — a cheap focus / sharpness metric.

    Low variance ⇒ image is smooth ⇒ likely blurry. Values < 100 are typical
    for noticeably blurred frames on 640×480 footage.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def letterbox(image: np.ndarray, target: int = DEFAULT_TARGET_SIZE) -> np.ndarray:
    """Resize the longest side to ``target``, pad the short side with grey.

    Preserves aspect ratio (no stretching) so subsequent detections train on
    undistorted geometry. Padding uses YOLO's standard 114-grey.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Cannot letterbox an empty image.")
    scale = target / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (target - new_h) // 2
    pad_bottom = target - new_h - pad_top
    pad_left = (target - new_w) // 2
    pad_right = target - new_w - pad_left
    return cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=LETTERBOX_PAD_COLOR,
    )


def _phash_or_none(bgr: np.ndarray):
    """Return a perceptual hash of a BGR frame, or ``None`` if imagehash absent."""
    if not _HAS_IMAGEHASH:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------
def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    target_size: int = DEFAULT_TARGET_SIZE,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    hash_distance: int = DEFAULT_HASH_DISTANCE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    show_progress: bool = True,
) -> ExtractionStats:
    """Extract frames from one video file.

    Parameters
    ----------
    video_path : Path
        Source video.
    output_dir : Path
        Destination folder. Created if missing. Frames are written as
        ``{video_stem}_{frame_idx:06d}.jpg``.
    sample_rate : int
        Keep one frame every ``sample_rate`` decoded frames.
    target_size : int
        Output square side length (letterboxed).
    blur_threshold : float
        Drop frames whose Laplacian variance is below this.
    hash_distance : int
        Drop frames whose phash Hamming distance from the previous kept frame
        is < this. Set to 0 to disable deduplication. No-op if ``imagehash``
        is not installed.
    jpeg_quality : int
        OpenCV JPEG quality (0–100).
    show_progress : bool
        Display a tqdm bar.

    Returns
    -------
    ExtractionStats
        Counters describing what happened.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = ExtractionStats(video=str(video_path), output_dir=str(output_dir))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video: %s", video_path)
        return stats

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stats.total_frames = total

    last_hash = None
    idx = -1
    bar = tqdm(total=total, desc=video_path.name, disable=not show_progress, leave=False)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            bar.update(1)

            if idx % sample_rate != 0:
                continue

            if frame is None or frame.size == 0:
                stats.skipped_unreadable += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if laplacian_variance(gray) < blur_threshold:
                stats.skipped_blur += 1
                continue

            if hash_distance > 0 and _HAS_IMAGEHASH:
                h = _phash_or_none(frame)
                if last_hash is not None and h is not None:
                    if (h - last_hash) < hash_distance:
                        stats.skipped_duplicate += 1
                        continue
                last_hash = h

            letterboxed = letterbox(frame, target=target_size)
            out_name = f"{video_path.stem}_{idx:06d}.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(
                str(out_path),
                letterboxed,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            stats.extracted += 1
    finally:
        bar.close()
        cap.release()

    return stats


def _find_videos(input_dir: Path) -> list[Path]:
    """Recursively collect all video files under ``input_dir``."""
    files: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(input_dir.rglob(f"*{ext}"))
    return sorted(files)


def _worker(args: tuple) -> dict:
    """Multiprocessing worker — must be picklable, so accepts a tuple."""
    video_path, output_dir, kwargs = args
    s = extract_frames_from_video(
        Path(video_path), Path(output_dir), show_progress=False, **kwargs
    )
    return s.as_dict()


def extract_frames_from_dir(
    input_dir: Path,
    output_dir: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    target_size: int = DEFAULT_TARGET_SIZE,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    hash_distance: int = DEFAULT_HASH_DISTANCE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    workers: int = 1,
    preserve_subdirs: bool = True,
) -> list[ExtractionStats]:
    """Extract frames from every video under ``input_dir``.

    If ``preserve_subdirs`` is True, the immediate-parent folder of each video
    becomes the matching output subfolder (so ``raw/fire/x.avi`` lands in
    ``frames/fire/...``).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = _find_videos(input_dir)
    if not videos:
        logger.warning("No videos found under %s", input_dir)
        return []

    kwargs = dict(
        sample_rate=sample_rate,
        target_size=target_size,
        blur_threshold=blur_threshold,
        hash_distance=hash_distance,
        jpeg_quality=jpeg_quality,
    )

    tasks: list[tuple] = []
    for v in videos:
        if preserve_subdirs:
            rel = v.relative_to(input_dir).parent
            out = output_dir / rel
        else:
            out = output_dir
        tasks.append((v, out, kwargs))

    results: list[ExtractionStats] = []

    if workers <= 1:
        for v, out, kw in tqdm(tasks, desc="videos"):
            results.append(extract_frames_from_video(v, out, **kw))
        return results

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="videos"):
            d = fut.result()
            results.append(ExtractionStats(**d))
    return results


def summarize(results: Iterable[ExtractionStats]) -> dict:
    """Aggregate per-video stats into a flat summary dict."""
    total = extracted = blur = dup = unreadable = 0
    for r in results:
        total += r.total_frames
        extracted += r.extracted
        blur += r.skipped_blur
        dup += r.skipped_duplicate
        unreadable += r.skipped_unreadable
    return {
        "videos": sum(1 for _ in results) if not isinstance(results, list) else len(results),
        "total_frames": total,
        "extracted": extracted,
        "skipped_blur": blur,
        "skipped_duplicate": dup,
        "skipped_unreadable": unreadable,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract frames from videos.")
    p.add_argument("--input", required=True, type=Path,
                   help="Source video file or folder.")
    p.add_argument("--output", required=True, type=Path,
                   help="Destination folder for extracted frames.")
    p.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
                   help="Keep every N-th frame.")
    p.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE,
                   help="Letterbox square side length.")
    p.add_argument("--blur-threshold", type=float, default=DEFAULT_BLUR_THRESHOLD,
                   help="Drop frames with Laplacian variance below this.")
    p.add_argument("--hash-distance", type=int, default=DEFAULT_HASH_DISTANCE,
                   help="Min phash Hamming distance from previous kept frame.")
    p.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    p.add_argument("--workers", type=int, default=1, help="Parallel video workers.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)

    if not args.input.exists():
        logger.error("Input path does not exist: %s", args.input)
        return 1

    if args.input.is_file():
        stats = extract_frames_from_video(
            args.input, args.output,
            sample_rate=args.sample_rate,
            target_size=args.target_size,
            blur_threshold=args.blur_threshold,
            hash_distance=args.hash_distance,
            jpeg_quality=args.jpeg_quality,
        )
        print(stats.as_dict())
        return 0

    results = extract_frames_from_dir(
        args.input, args.output,
        sample_rate=args.sample_rate,
        target_size=args.target_size,
        blur_threshold=args.blur_threshold,
        hash_distance=args.hash_distance,
        jpeg_quality=args.jpeg_quality,
        workers=args.workers,
    )
    summary = summarize(results)
    print("\n=== Extraction summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
