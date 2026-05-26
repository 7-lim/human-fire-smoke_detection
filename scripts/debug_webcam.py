"""One-shot webcam capture + raw YOLO inspection.

Captures a single frame from the webcam, runs best.pt with NO filters, and
prints every raw detection (class, confidence, box-area-fraction). Use this
to find out what the model is actually emitting for whatever scene is in
front of you — so you know whether the filter thresholds in inference.py
need to move.

Usage:
    python scripts/debug_webcam.py
    python scripts/debug_webcam.py --source 1            # second webcam
    python scripts/debug_webcam.py --source path/to.jpg  # also accepts an image
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import (
    DEFAULT_WEIGHTS, PER_CLASS_CONF, ALERT_CONF, MAX_BOX_FRAME_FRACTION,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="Webcam index or image path.")
    p.add_argument("--conf", type=float, default=0.05,
                   help="Raw YOLO threshold — keep low to see everything.")
    args = p.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(str(DEFAULT_WEIGHTS))

    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():
            print(f"ERROR: could not open webcam {args.source}", file=sys.stderr)
            return 1
        # discard a couple of warm-up frames (webcam auto-exposure)
        for _ in range(5):
            cap.read()
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("ERROR: failed to read a frame from webcam", file=sys.stderr)
            return 1
    else:
        frame = cv2.imread(args.source)
        if frame is None:
            print(f"ERROR: could not read image: {args.source}", file=sys.stderr)
            return 1

    h, w = frame.shape[:2]
    frame_area = float(h * w)

    res = model.predict(frame, conf=args.conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        print("\n(no detections — try lower --conf, or model truly sees nothing here)")
        return 0

    print(f"\nFrame: {w}x{h}   raw detections @ conf >= {args.conf}:\n")
    print(f"  {'class':<12} {'conf':>6}  {'box_area_%':>10}  passes_draw  triggers_alert  reason")
    print(f"  {'-'*12} {'-'*6}  {'-'*10}  {'-'*11}  {'-'*14}  {'-'*30}")

    xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), s, c in zip(xyxy, confs, cls_ids):
        cname = model.names[int(c)]
        area_pct = (max(0, x2 - x1) * max(0, y2 - y1)) / frame_area * 100.0
        min_conf = PER_CLASS_CONF.get(cname, 0.0)
        max_frac = MAX_BOX_FRAME_FRACTION.get(cname)
        alert_thresh = ALERT_CONF.get(cname)

        reasons = []
        if float(s) < min_conf:
            reasons.append(f"conf<{min_conf}")
        if max_frac is not None and area_pct / 100.0 > max_frac:
            reasons.append(f"area>{int(max_frac*100)}%")
        passes_draw = "YES" if not reasons else "no"
        if alert_thresh is None:
            triggers_alert = "n/a"
        elif passes_draw == "YES" and float(s) >= alert_thresh:
            triggers_alert = "YES"
        elif passes_draw == "YES":
            triggers_alert = f"no (need>={alert_thresh})"
        else:
            triggers_alert = "no"
        why = ", ".join(reasons) if reasons else "passes all filters"

        print(f"  {cname:<12} {s:>6.3f}  {area_pct:>9.1f}%  {passes_draw:<11}  {triggers_alert:<14}  {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
