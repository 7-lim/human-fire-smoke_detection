"""Streamlit UI for the fire / smoke / person YOLOv8 detector.

Run with:
    streamlit run app.py

This is a thin web front-end over ``src.inference`` — it reuses the exact same
model, per-class confidence gates, area filters and ALERT logic as the CLI, so
what you see here matches ``python -m src.inference``.
"""
from __future__ import annotations

import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.inference import (
    ALERT_CLASSES,
    ALERT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_WEIGHTS,
    MAX_BOX_FRAME_FRACTION,
    PER_CLASS_CONF,
    draw_detections,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "dataset" / "images" / "test"
# Collect network candidates low, then filter per-class (mirrors the CLI).
NETWORK_CONF = 0.20

st.set_page_config(page_title="Fire / Smoke / Person Detector", page_icon="🔥", layout="wide")


# ---------------------------------------------------------------------------
# Model loading (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_model(weights: str, device: str):
    from ultralytics import YOLO

    model = YOLO(weights)
    try:
        model.to(device)
    except Exception:
        pass
    return model


def pick_device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def detect(model, bgr: np.ndarray, iou: float, imgsz: int):
    """Run YOLO + the project's per-class filters on one BGR frame.

    Returns ``(annotated_bgr, fired_classes, counts)`` where ``counts`` maps
    drawn class name -> number of boxes that survived the gates.
    """
    res = model.predict(bgr, conf=NETWORK_CONF, iou=iou, imgsz=imgsz, verbose=False)[0]
    counts = _count_drawn(res, model.names, bgr.shape[:2])
    annotated, fired = draw_detections(bgr, res, model.names)
    return annotated, fired, counts


def _count_drawn(result, names, frame_hw) -> Counter:
    """Re-apply the draw-time gates to tally which boxes are actually shown."""
    counts: Counter = Counter()
    if result.boxes is None or len(result.boxes) == 0:
        return counts
    xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    frame_area = float(frame_hw[0] * frame_hw[1])
    for (x1, y1, x2, y2), s, c in zip(xyxy, confs, cls_ids):
        cname = names.get(int(c), str(c))
        if float(s) < PER_CLASS_CONF.get(cname, 0.0):
            continue
        max_frac = MAX_BOX_FRAME_FRACTION.get(cname)
        if max_frac is not None:
            box_area = float(max(0, x2 - x1) * max(0, y2 - y1))
            if box_area / max(frame_area, 1.0) > max_frac:
                continue
        counts[cname] += 1
    return counts


def render_summary(counts: Counter, fired: list[str]) -> None:
    if fired:
        alert_names = ", ".join(sorted(set(fired))).upper()
        st.error(f"🚨  ALERT — {alert_names} detected", icon="🔥")
    elif counts:
        st.success("No fire/smoke alert — scene looks clear.", icon="✅")

    if counts:
        cols = st.columns(len(counts))
        for col, (cname, n) in zip(cols, sorted(counts.items())):
            col.metric(cname, n)
    else:
        st.info("No detections above the per-class thresholds.")


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

device = st.sidebar.selectbox(
    "Device", ["auto", "cuda:0", "cpu"], index=0,
    help="auto = CUDA if available, else CPU.",
)
device = pick_device() if device == "auto" else device
st.sidebar.caption(f"Using device: `{device}`")

weights = st.sidebar.text_input("Weights", value=str(DEFAULT_WEIGHTS))
iou = st.sidebar.slider("NMS IoU", 0.1, 0.9, DEFAULT_IOU, 0.05)
imgsz = st.sidebar.select_slider("Image size", [320, 416, 512, 640, 768], value=DEFAULT_IMGSZ)

with st.sidebar.expander("Per-class confidence gates"):
    st.caption("Applied after inference (from `src.inference`).")
    st.json({k: PER_CLASS_CONF[k] for k in ("fire", "smoke", "person")})
    st.caption("ALERT thresholds (stricter):")
    st.json(ALERT_CONF)

if not Path(weights).exists():
    st.error(f"Weights not found at `{weights}`. Train first or fix the path in the sidebar.")
    st.stop()

model = load_model(weights, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🔥 Fire / Smoke / Person Detection")
st.caption("YOLOv8m fine-tuned detector — upload media or pick a test image.")

tab_image, tab_video, tab_live, tab_samples = st.tabs(
    ["🖼️ Image", "🎞️ Video", "📷 Live webcam", "🗂️ Sample images"]
)

# Read last-known checkbox state so the other tabs can skip their (heavy)
# re-processing while the live loop is spinning reruns.
live_active = st.session_state.get("live_run", False)


def run_image(bgr: np.ndarray, caption: str = "", key_prefix: str = "img") -> None:
    t0 = time.time()
    annotated, fired, counts = detect(model, bgr.copy(), iou, imgsz)
    dt = (time.time() - t0) * 1000
    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption=f"Input{' — ' + caption if caption else ''}")
    c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Detections ({dt:.0f} ms)")
    render_summary(counts, fired)
    ok, buf = cv2.imencode(".jpg", annotated)
    if ok:
        # Unique key per call site — otherwise two identical download_buttons
        # (Image tab + Samples tab) collide on their auto-generated ID.
        st.download_button("⬇️ Download annotated image", buf.tobytes(),
                           file_name=f"annotated_{Path(caption).stem or 'image'}.jpg",
                           mime="image/jpeg", key=f"dl_{key_prefix}")


with tab_image:
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    if up is not None and not live_active:
        data = np.frombuffer(up.read(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not decode that image.")
        else:
            run_image(bgr, up.name, key_prefix="upload")


with tab_video:
    upv = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    max_frames = st.slider("Max frames to process", 30, 900, 300, 30,
                           help="Caps processing time. Longer videos are truncated.")
    if upv is not None and st.button("▶️ Run detection on video", type="primary"):
        tmp_in = Path(tempfile.gettempdir()) / f"st_in_{int(time.time())}{Path(upv.name).suffix}"
        tmp_in.write_bytes(upv.read())
        cap = cv2.VideoCapture(str(tmp_in))
        if not cap.isOpened():
            st.error("Could not open that video.")
        else:
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
            n_target = min(total, max_frames)

            out_path = Path(tempfile.gettempdir()) / f"st_out_{int(time.time())}.mp4"
            # avc1 (H.264) plays in-browser; fall back to mp4v if unavailable.
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"),
                                     src_fps, (src_w, src_h))
            if not writer.isOpened():
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         src_fps, (src_w, src_h))

            progress = st.progress(0.0, text="Processing…")
            agg: Counter = Counter()
            alert_frames = 0
            i = 0
            preview = st.empty()
            while i < n_target:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated, fired, counts = detect(model, frame, iou, imgsz)
                writer.write(annotated)
                agg.update(counts)
                if fired:
                    alert_frames += 1
                if i % 15 == 0:
                    preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                  caption=f"Frame {i + 1}/{n_target}", channels="RGB")
                i += 1
                progress.progress(i / n_target, text=f"Processing frame {i}/{n_target}")
            cap.release()
            writer.release()
            progress.empty()

            st.subheader("Result")
            if alert_frames:
                st.error(f"🚨 ALERT fired on {alert_frames}/{i} frames.", icon="🔥")
            else:
                st.success("No fire/smoke alert across processed frames.", icon="✅")
            if agg:
                cols = st.columns(len(agg))
                for col, (cname, n) in zip(cols, sorted(agg.items())):
                    col.metric(f"{cname} (total boxes)", n)

            video_bytes = out_path.read_bytes()
            st.video(video_bytes)
            st.download_button("⬇️ Download annotated video", video_bytes,
                               file_name="annotated.mp4", mime="video/mp4")


with tab_live:
    st.caption("Streams the local machine's webcam through the detector continuously. "
               "Untick the box to stop — the stream halts at the next frame.")
    cam_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
    run = st.checkbox("🔴 Run live detection", key="live_run")
    status_slot = st.empty()
    frame_slot = st.empty()

    if run:
        # CAP_DSHOW is the fast DirectShow backend on Windows.
        cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)
        if not cap.isOpened():
            status_slot.error(
                f"Could not open webcam at index {cam_index}. "
                "Try another index or close other apps using the camera."
            )
        else:
            # One continuous loop — NOT a per-frame st.rerun(). Unticking the
            # checkbox makes Streamlit raise a rerun at the next frame_slot.image()
            # call, which breaks out cleanly; the finally releases the camera.
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        status_slot.warning("Dropped a frame — retrying…")
                        continue
                    annotated, fired, counts = detect(model, frame, iou, imgsz)
                    if fired:
                        names = ", ".join(sorted(set(fired))).upper()
                        status_slot.error(f"🚨 ALERT — {names} detected", icon="🔥")
                    elif counts:
                        summary = " · ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
                        status_slot.success(f"Clear — {summary}", icon="✅")
                    else:
                        status_slot.info("No detections above the per-class thresholds.")
                    frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                     channels="RGB", use_container_width=True)
            finally:
                cap.release()
    else:
        frame_slot.info("Tick **Run live detection** to start the webcam.")


with tab_samples:
    if not TEST_IMAGES_DIR.exists():
        st.info(f"No sample images found at `{TEST_IMAGES_DIR}`.")
    else:
        samples = sorted(TEST_IMAGES_DIR.glob("*.jpg"))[:200]
        if not samples:
            st.info("Test image folder is empty.")
        else:
            names = [p.name for p in samples]
            choice = st.selectbox(f"Pick a test image ({len(samples)} available)", names)
            if live_active:
                st.info("Live webcam is running — stop it to run sample detection.")
            else:
                bgr = cv2.imread(str(TEST_IMAGES_DIR / choice))
                if bgr is not None:
                    run_image(bgr, choice, key_prefix="sample")
