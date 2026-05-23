# Fire & Smoke Detection with Human Motion Tracking

Classical Computer Vision project — **no deep learning, no pre-trained models**.
Detects fire, smoke, and moving humans in surveillance-style video using only
OpenCV primitives: color spaces, background subtraction, morphology, contours.

> Computer Vision module — university evaluation project.

---

## What it does

For every frame of a video, the pipeline produces an annotated frame with:

- 🟥 **Red boxes** around fire regions
- 🟦 **Blue boxes** around smoke regions
- 🟩 **Green boxes** around moving humans
- A HUD showing per-frame counts (`H:1 F:2 S:0`)

The detectors are **motion-correlated** — color alone isn't enough to distinguish
flame from warm-lit wood, or smoke from gray walls. Real fire flickers, real
smoke billows, real people move. Static warm/gray objects don't.

## How it works (pipeline)

```
raw frame ──► resize ──► Gaussian blur ──► motion mask
                                          (MOG2 + 1-frame diff + 8-frame diff)
                                                │
                ┌─────────────────────┬─────────┴─────────┐
                ▼                     ▼                   ▼
         detect_humans         detect_fire          detect_smoke
         (area/aspect/         (HSV + RGB +        (HSV + motion +
          solidity)            motion + peak-S)    texture filter)
                │                     │                   │
                └─────────────────────┴─────────┬─────────┘
                                                ▼
                                      annotated .mp4 output
```

See [`PROJECT_EXPLANATION.txt`](PROJECT_EXPLANATION.txt) for an extensive cell-by-cell breakdown.

## Dataset

[Fire and Smoke Dataset](https://www.kaggle.com/datasets/unidpro/fire-and-smoke-dataset/data) (Kaggle, by unidpro).

3 videos:

| Folder | File | Content |
|---|---|---|
| `data/part1/` | `bucket11.mp4` | Industrial bucket fire (white-hot flame) |
| `data/part2/` | `roomfire41.mp4` | Indoor room — calm scene then sudden flame + heavy smoke |
| `data/part3/` | `printer31.mp4` | Office printer fire — dense smoke, people reacting |

Download the dataset from Kaggle and place the three `part*/` folders under `data/` in the project root.

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-user>/fire_smoke_withoutDL.git
cd fire_smoke_withoutDL

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Register the kernel for Jupyter
python -m ipykernel install --user --name=fire-smoke-cv --display-name="Python (Fire-Smoke-CV)"

# 5. Download the dataset from Kaggle and put it under data/ (see above)
```

## Run

Open `Fire_Smoke_Detection.ipynb` in Jupyter / VS Code, select the
**Python (Fire-Smoke-CV)** kernel, and Run All.

The notebook will:

1. Walk through every step of the image-processing pipeline on a reference frame.
2. Demonstrate each detector individually.
3. Run the combined pipeline on all 3 videos.
4. Save annotated `.mp4`s to `outputs/`.
5. Plot per-frame detection counts.

Total runtime: ~3–5 minutes on a laptop CPU.

> The committed `Fire_Smoke_Detection.ipynb` is **pre-executed** — all figures
> and outputs are saved inside the notebook, so you can view results directly
> on GitHub without running anything.

## Results

| Video | Detection |
|---|---|
| `bucket11` | Bucket fire + thin smoke continuously detected (white-hot HSV rule) |
| `roomfire41` | Table flame correctly caught at the moment of fire eruption; smoke region detected; residual false positives on warm-lit wood (known limitation) |
| `printer31` | Smoke around the printer + reacting humans detected; hidden flame not visible |

Annotated `.mp4` outputs are in `outputs/`.

## Tech stack

- Python 3.13
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- tqdm
- Jupyter

## Project structure

```
fire_smoke_withoutDL/
├── Fire_Smoke_Detection.ipynb              # main notebook (pre-executed, with outputs)
├── PROJECT_EXPLANATION.txt                 # cell-by-cell walkthrough
├── README.md
├── requirements.txt
├── .gitignore
├── data/                                   # dataset (excluded from git)
│   ├── part1/bucket11.mp4
│   ├── part2/roomfire41.mp4
│   └── part3/printer31.mp4
└── outputs/                                # annotated videos (committed)
    ├── bucket11_annotated.mp4
    ├── roomfire41_annotated.mp4
    └── printer31_annotated.mp4
```

## Limitations

- **Color-only fire detection** can be fooled when fire is large enough to bathe
  surrounding wood in orange firelight — the lit wood then flickers with the
  flame and motion correlation no longer disambiguates.
- **Smoke vs light walls** — without optical-flow signatures (smoke expands
  upward) we cannot fully separate thin smoke from a uniform gray wall.
- **MOG2 absorbs stationary objects** — a person standing perfectly still
  eventually disappears from the foreground mask. The long-window frame
  difference helps but does not solve it.
- **No identity tracking** — each frame is processed independently.

## Possible improvements

- Temporal flicker analysis (Fourier on the red channel — flames flicker at
  5–15 Hz).
- Optical flow for smoke (upward dispersive flow signature).
- Kalman / centroid tracker for stable identities over time.
- Adaptive HSV thresholds learned online from the first seconds of each clip.

## License

Educational project — feel free to read, fork, and learn from it.
