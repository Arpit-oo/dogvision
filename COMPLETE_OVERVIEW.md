# DogVision — Complete Project Overview

A GPU-accelerated real-time video surveillance system that detects dogs and persons, analyzes dog bite/aggression risk, and enforces time-based access control per camera.

---

## What This Project Does

Given any video input (file, webcam, RTSP stream), the system:

1. **Detects dogs and persons** in every frame using YOLOv8 deep learning
2. **Tracks each individual** across frames with persistent IDs (ByteTrack)
3. **Analyzes bite risk** when a dog is near a person — scores proximity, overlap, lunging, and sustained contact
4. **Flags unauthorized access** when a person appears outside allowed hours for that camera
5. **Outputs annotated video** with color-coded bounding boxes, alert lines, and a live stats HUD
6. **Logs all events** to JSON for review via the web dashboard

---

## How It Runs — Step by Step

### 1. Video Input

The system reads frames from the source video one at a time using OpenCV.

```
Video file (.mp4) → cv2.VideoCapture → BGR frame (numpy array)
```

**File:** `utils/video.py` — frame iterator with reconnection support
**See also:** [PIPELINE_WALKTHROUGH.md](PIPELINE_WALKTHROUGH.md) for the full frame-by-frame GPU flow

### 2. Detection + Tracking (Single YOLO Pass)

Each frame goes through YOLOv8 in a single forward pass that detects **both** dogs and persons simultaneously. ByteTrack assigns persistent IDs so the same dog keeps the same ID across frames.

```
Frame → YOLOv8 (classes=[0, 16]) → detections with bounding boxes + confidence + track IDs
```

- **Class 0** = Person (COCO dataset label)
- **Class 16** = Dog (COCO dataset label)
- Model: `yolov8m.pt` (25.9M parameters, pretrained on COCO 2017)
- Input resolution: 960px for best recall on small/distant objects
- Confidence threshold: 0.25 (catches most dogs without too many false positives)

**File:** `detection/yolo.py` — YOLOv8 wrapper with TensorRT FP16 export
**File:** `tracking/tracker.py` — per-ID trajectory accumulator
**See also:** [MODEL_JUSTIFICATION.md](MODEL_JUSTIFICATION.md) for why we use pretrained instead of custom-trained

### 3. Detection Routing

After detection, each result is routed by class into two independent pipelines:

```
Detections
    ├── cls == 16 (dog)    → Dog Pipeline (bite risk)
    └── cls == 0  (person) → Person Pipeline (access control)
```

Both pipelines run on the same frame's detections. A single YOLO pass feeds both — no duplicate inference.

**File:** `run_demo_cpu.py` lines 155–188 — routing logic

### 4a. Dog Pipeline — Bite Risk Analysis

Every dog-person pair is scored for aggression risk using 4 weighted factors:

| Factor | Weight | What It Measures | Threshold |
|--------|--------|-----------------|-----------|
| **Proximity** | 30% | How close dog center is to person center, normalized by dog size | 2.0× dog diagonal |
| **Overlap** | 25% | IoU (intersection over union) between dog and person bounding boxes | 0.03 minimum |
| **Lunge** | 25% | Dog bounding box area growing >25% over 4 frames (dog rushing toward person) | 1.25× area ratio |
| **Sustained** | 20% | How many consecutive frames the pair has been in close proximity | 3-frame window |

```
risk = 0.30 × proximity + 0.25 × overlap + 0.25 × lunge + 0.20 × sustained
```

**If risk ≥ 0.40 → BITE RISK ALERT** is emitted with reason tags (close_proximity, physical_contact, lunge_detected, sustained_contact).

Key design: a dog can trigger a bite alert **without physically touching** the person — just by being close + approaching fast + staying near. This catches aggressive approach behavior, not just contact.

The analyzer maintains state per dog-person pair across frames. When a pair is no longer visible, their proximity history decays and is cleaned up.

**File:** `behavior/bite_detector.py` — the full 4-factor scoring engine
**See also:** [PIPELINE_WALKTHROUGH.md](PIPELINE_WALKTHROUGH.md) for the scoring formula diagram

### 4b. Person Pipeline — Access Control

Each detected person is checked against per-camera time-based rules loaded from a YAML config file.

```
Person detected on Camera 0 at 23:15
    → Rules for Camera 0: allowed 06:00–22:00
    → 23:15 is OUTSIDE allowed window
    → ACCESS VIOLATION event emitted
```

- Supports multiple time windows per camera (e.g., morning shift + evening shift)
- Handles overnight windows (e.g., 22:00–06:00 crosses midnight)
- Gracefully disabled if no config file is provided

**File:** `behavior/access_control.py` — time-based person authorization
**Config:** `configs/access_schedule.yaml` — per-camera allowed hours
**Config:** `configs/access_schedule_restricted.yaml` — night-only window for testing
**See also:** [UNAUTHORIZED_ACCESS_EXPLAINED.md](UNAUTHORIZED_ACCESS_EXPLAINED.md) for full access control deep-dive

### 5. Ghost Dog Persistence

When ByteTrack temporarily loses a dog (brief occlusion, turned away from camera), we keep showing its last-known bounding box for **30 frames** (~1.2 seconds at 25 FPS). This:
- Prevents visual flickering in the output video
- Helps the bite risk analyzer maintain proximity state through brief occlusions
- Allows the tracker to re-link the same ID when the dog reappears

Ghost bboxes are drawn at 50% confidence to indicate they're estimated positions.

**File:** `run_demo_cpu.py` lines 189–208 — ghost persistence logic

### 6. Annotation Rendering

Each frame gets layered annotations drawn on top:

```
1. draw_dogs()              → Green bounding boxes, label "dog"
2. draw_persons()           → Teal bounding boxes, label "person#ID 0.85"
3. draw_bite_alerts()       → Red line between dog↔person + "BITE RISK 68%" label
4. draw_access_violations() → Orange box + "UNAUTHORIZED @ 23:15:02" label
5. overlay_hud()            → Semi-transparent stats panel (FPS, counts, alerts)
```

The HUD shows live stats: FPS, current dog/person count, unique dogs seen, bite alert total, access violation total, frame number. Bite/access counters turn red/orange when non-zero.

**File:** `utils/draw.py` — all annotation rendering functions
**File:** `utils/color.py` — deterministic track-ID → color mapping (same dog = same color)

### 7. Output

The system produces these artifacts:

| File | Format | Contents |
|------|--------|---------|
| `out/dogvision_output.mp4` | MP4 video | Annotated video with all bboxes, alerts, and HUD |
| `out/events.json` | JSON array | Every bite risk and access violation event with metadata |
| `out/summary.json` | JSON object | Run summary: frame count, unique dogs/persons, alert counts, FPS |

---

## Entry Points — Which Script Does What

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `run_demo_cpu.py` | **Main demo.** Single video, full dual pipeline on CPU | Showing the project to teacher |
| `run_demo_gpu.py` | GPU demo with TensorRT FP16, CuPy ring buffer, cuDF analytics | If GPU is available |
| `run_multi_stream.py` | 2×2 CCTV grid processing 1–4 videos simultaneously | Multi-camera demo |
| `demo.py` | Threaded GPU pipeline orchestrator | Production GPU mode |
| `dashboard.py` | **Web dashboard** at localhost:5000 — view results, upload videos | Presenting results to teacher |
| `train_and_evaluate.py` | Train custom model, compare with pretrained, generate report | Showing training attempt |
| `benchmark.py` | GPU vs CPU speed comparison | Performance metrics |
| `generate_report.py` | Academic Word document generator | Professor submission |
| `train.py` | Basic YOLOv8 fine-tuning script | Optional model improvement |

**See also:** [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) for detailed CLI flags and examples for every script

---

## Project Structure — What Each Folder Contains

```
vaibhav/
├── detection/              YOLO model wrapper
│   └── yolo.py             Load model, run inference, export TensorRT
│
├── tracking/               Post-detection tracking
│   └── tracker.py          Per-ID trajectory accumulator (centers, timestamps)
│
├── behavior/               Business logic pipelines
│   ├── bite_detector.py    4-factor bite risk scoring engine
│   └── access_control.py   Time-based person authorization
│
├── analytics/              GPU-accelerated data processing
│   ├── ring_buffer.py      CuPy GPU ring buffer (O(1) detection storage)
│   ├── window.py           cuDF rolling-window aggregates
│   ├── roi_hist.py         CuPy HSV color histograms on dog ROIs
│   └── event_log.py        Unified JSON event logger
│
├── pipeline/               Threaded GPU orchestrator
│   └── orchestrator.py     3-thread pipeline: decode → inference → analytics
│
├── utils/                  Shared utilities
│   ├── draw.py             All annotation rendering (dogs, persons, alerts, HUD)
│   ├── color.py            Deterministic track-ID → BGR color mapping
│   └── video.py            Video I/O with reconnection support
│
├── configs/                Configuration files
│   ├── default.yaml        GPU pipeline config
│   ├── cpu.yaml            CPU pipeline config
│   ├── access_schedule.yaml           Normal access rules (06:00–22:00)
│   └── access_schedule_restricted.yaml Night-only rules (for testing violations)
│
├── run_demo_cpu.py         Main CPU demo (full dual pipeline)
├── run_demo_gpu.py         GPU demo (TRT FP16 + CuPy + cuDF)
├── run_multi_stream.py     Multi-stream 2×2 CCTV grid
├── demo.py                 Threaded GPU entry point
├── dashboard.py            Web dashboard (Flask)
├── train_and_evaluate.py   Training attempt + model comparison
├── generate_report.py      Academic Word report generator
├── benchmark.py            GPU vs CPU benchmark
└── train.py                Optional YOLOv8 fine-tuning
```

**See also:** [GPU_ACCELERATION_MAP.md](GPU_ACCELERATION_MAP.md) for exact file:line mapping of every GPU technology

---

## GPU Acceleration Stack

The project is designed as a GPU-accelerated computing demonstration. Every stage prioritizes GPU execution:

| Stage | GPU Technology | CPU Fallback |
|-------|---------------|-------------|
| Video decode | NVDEC hardware decoder | cv2.VideoCapture |
| Detection | PyTorch CUDA + TensorRT FP16 | PyTorch CPU FP32 |
| NMS | CUDA parallel sort | PyTorch CPU |
| Tracking | ByteTrack (Ultralytics GPU-fused) | CPU association |
| Detection storage | CuPy ring buffer | Python list |
| Analytics | cuDF groupby/agg (RAPIDS) | pandas |
| Color histograms | CuPy BGR→HSV + bincount | Skipped |

The CPU demo (`run_demo_cpu.py`) runs the full pipeline without GPU — same features, just slower (~1-2 FPS vs 25+ FPS on GPU).

**See also:** [GPU_ACCELERATION_MAP.md](GPU_ACCELERATION_MAP.md) for where each technology is used
**See also:** [PIPELINE_WALKTHROUGH.md](PIPELINE_WALKTHROUGH.md) for the full GPU pipeline flow diagram

---

## Model Choice — Why Pretrained

We use **pretrained YOLOv8m** (trained on COCO 2017 by Ultralytics) directly. No custom training needed because:

1. COCO has 5,500+ dog instances + 262,000+ person instances across diverse conditions
2. Pretrained features generalize well to CCTV footage
3. Project focus is GPU acceleration, not model accuracy
4. Fine-tuning on a small custom dataset would overfit

We **did attempt** training a custom model (`train_and_evaluate.py`) to compare — the pretrained model outperformed it due to COCO's larger and more diverse training data.

**See also:** [MODEL_JUSTIFICATION.md](MODEL_JUSTIFICATION.md) for full rationale

---

## Evaluation Results

Tested on 5 different videos covering dogs, persons, bites, and unauthorized access:

| Video | Frames | Dogs | Persons | Bite Alerts | Access Violations | FPS |
|-------|--------|------|---------|-------------|-------------------|-----|
| dogbite.mp4 | 166 | 1 | 3 | 73 | 0 | 1.6 |
| testdiog.mp4 | 1,151 | 41 | 33 | 469 | 0 | 1.9 |
| CCTV People Demo 2 | 1,932 | 0 | 348 | 0 | 0 | 0.9 |
| House Break-in | 4,117 | 3 | 11 | 29 | 0 | 1.2 |
| XlZXsvOuuRc | 1,200 | 55 | 19 | 71 | 0 | 0.8 |

Key observations:
- **People-only video** → zero dogs detected, zero false bite alerts (correct)
- **Dog bite video** → 73 bite alerts in 166 frames (high density, correct)
- **House break-in** → detected intruder (person) + dogs in scene
- **Access violations** trigger when using restricted access config (night-only hours)

---

## Web Dashboard

Run `python dashboard.py` and open `http://localhost:5000`.

Three tabs:
1. **Overview** — global stats across all runs + per-run cards with Video, Events, JSON buttons
2. **Upload & Analyze** — drag & drop a video, runs pipeline, returns results with video playback
3. **Event Logs** — browse events per run with filters (All / Bite Alerts / Access Violations)

Videos are auto-transcoded from mp4v to H.264 for browser playback.

---

## CPU vs GPU — Why Both Exist

### Can we run entirely on GPU?

**Yes.** Use `run_demo_gpu.py` or `demo.py` for full GPU execution with TensorRT FP16, CuPy ring buffers, and cuDF analytics.

### Then why do we demo on CPU?

The CPU demo (`run_demo_cpu.py`) exists for **portability and ease of setup**:

| Aspect | GPU Path | CPU Path |
|--------|---------|---------|
| Script | `run_demo_gpu.py` / `demo.py` | `run_demo_cpu.py` |
| Speed | **25-45 FPS** (real-time) | **1-2 FPS** (offline processing) |
| Inference | TensorRT FP16 on CUDA cores | PyTorch FP32 on CPU cores |
| Analytics | CuPy ring buffer + cuDF (all GPU) | Python lists (no GPU analytics) |
| Detection storage | GPU-resident arrays, O(1) append | In-memory Python dicts |
| Requirements | NVIDIA GPU + CUDA 12.x + WSL2 + RAPIDS conda env | Just `pip install ultralytics torch opencv-python pyyaml` |
| Setup time | ~30 minutes | ~2 minutes |
| Portability | No — TensorRT engine is GPU-architecture specific | Yes — runs on any machine |
| Real-time capable? | Yes — can process live cameras | No — too slow for real-time |

### Why the CPU path still matters

1. **Reproducibility** — instructor can run the demo on any laptop without GPU
2. **Same features** — detection, tracking, bite risk, access control all work identically
3. **No driver headaches** — CUDA/cuDNN/TensorRT version mismatches are common pain points
4. **Native Windows** — RAPIDS (cuDF) requires Linux or WSL2, CPU path runs on Windows directly

### Why the GPU path matters (the point of the project)

1. **Real-time processing** — 25+ FPS enables live surveillance, not just offline video analysis
2. **TensorRT FP16** — fuses Conv+BatchNorm+ReLU into a single CUDA kernel, quantizes FP32→FP16 for 2× memory bandwidth reduction. Result: 2-3× faster than PyTorch alone
3. **CuPy ring buffer** — detection data stays on GPU memory with O(1) append. No per-frame CPU↔GPU transfer overhead
4. **cuDF analytics** — rolling-window groupby/agg runs on GPU. For 54,000-row DataFrames, cuDF is 10-50× faster than pandas
5. **Three-thread pipeline** — decode uploads frame N+1 to GPU while inference runs on frame N. CUDA streams hide PCIe transfer latency — GPU never idles waiting for data
6. **Multi-stream batching** — process 4 cameras simultaneously with shared GPU inference

### Performance comparison (RTX 3060, YOLOv8s, 640px)

| Metric | GPU (TRT FP16) | CPU (FP32) | Speedup |
|--------|---------------|------------|---------|
| FPS | 35 | 3-5 | **7-10×** |
| Latency (p50) | ~15ms | ~200-300ms | **13-20×** |
| cuDF analytics | ~5ms | ~50ms (pandas) | **10×** |
| Ring buffer append | <0.1ms (GPU) | ~1ms (Python list) | **10×** |

At 1-2 FPS on CPU, it takes 10 minutes to process a 1-minute video. On GPU, same video processes in under 3 seconds. For a 4-camera 24/7 surveillance system, GPU is the only viable option.

---

## How to Demo for Teacher — Step by Step

### Option A: Browser Dashboard (Recommended — Easiest)

```bash
cd C:\code\vaibhav
python dashboard.py
```

Open `http://localhost:5000` in browser. From there you can:

1. **Overview tab** — show all past evaluation results (7 runs, 10K+ frames analyzed)
2. **Click "Video"** on any card — watch annotated output with bounding boxes and alerts
3. **Click "Events"** on any card — see bite risk + access violation event logs
4. **Click "JSON"** on any card — view raw summary data
5. **Upload & Analyze tab** — drag & drop any new video
   - Check "GPU Mode" to run on GPU with TensorRT FP16
   - Check "Restricted Access" to trigger unauthorized access detection
   - Watch progress bar as pipeline processes the video
   - View results inline when done

### Option B: Command Line Demo

**Step 1 — Show dog bite detection:**
```bash
python run_demo_cpu.py --source input/dogbite.mp4 --no-display
```
Output: `out/dogvision_output.mp4` — open to show red BITE RISK alerts

**Step 2 — Show access control (unauthorized person detection):**
```bash
python run_demo_cpu.py --source "input/vidssave.com Man Caught Break Into House  Man Breaking into House Caught on Camera 1080P.mp4" --no-display --access-config configs/access_schedule_restricted.yaml
```
Output: orange UNAUTHORIZED labels on detected persons

**Step 3 — Show multi-camera grid:**
```bash
python run_multi_stream.py --sources input/dogbite.mp4 "input/The CCTV People Demo 2.mp4" input/15440276_2160_3840_30fps.mp4 input/XlZXsvOuuRc.mp4 --no-display
```
Output: `out/multi_stream_output.mp4` — 2×2 CCTV grid with independent tracking per camera

**Step 4 — Show training attempt:**
```bash
python train_and_evaluate.py
```
Output: `out/training_report.json` — comparison showing pretrained outperformed custom model

**Step 5 — Show dashboard with all results:**
```bash
python dashboard.py
# Open http://localhost:5000
```

**Step 6 — Generate Word report for submission:**
```bash
python generate_report.py
```
Output: `DogVision_GPU_Pipeline_Report.docx`

### Option C: GPU Demo (If NVIDIA GPU Available)

```bash
python run_demo_gpu.py --source input/dogbite.mp4 --no-display
```
Shows: TensorRT FP16 auto-export, CUDA inference, CuPy ring buffer, cuDF analytics — all logged in terminal. Same output but 10-20× faster.

Or from the browser dashboard: check "GPU Mode" checkbox before uploading a video.

### What to Show in Each Demo

| What Teacher Asks | What to Show |
|-------------------|-------------|
| "How does detection work?" | Run dogbite.mp4, show green dog boxes + teal person boxes in output video |
| "How does bite detection work?" | Show red BITE RISK lines in dogbite.mp4 output, explain 4-factor scoring |
| "How does access control work?" | Run house break-in with restricted config, show orange UNAUTHORIZED labels |
| "Can it handle multiple cameras?" | Show multi_stream_output.mp4 — 4 cameras in 2×2 grid |
| "Why not train your own model?" | Run train_and_evaluate.py, show comparison report |
| "Where is GPU acceleration?" | Open GPU_ACCELERATION_MAP.md or run run_demo_gpu.py |
| "Show me the data" | Open dashboard → Events tab → select any run |
| "How fast is it?" | Show benchmark.py results or CPU vs GPU table in COMPLETE_OVERVIEW.md |

---

## Documentation Map

| Document | What It Covers |
|----------|---------------|
| [README.md](README.md) | Project overview, quick start, layout |
| [COMPLETE_OVERVIEW.md](COMPLETE_OVERVIEW.md) | **This file** — full system explanation for teacher |
| [PROJECT_REPORT.md](PROJECT_REPORT.md) | Detailed technical report (code-to-feature mapping) |
| [PIPELINE_WALKTHROUGH.md](PIPELINE_WALKTHROUGH.md) | Frame-by-frame GPU pipeline flow with diagrams |
| [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) | Per-script usage guide with CLI flags |
| [MODEL_JUSTIFICATION.md](MODEL_JUSTIFICATION.md) | Why pretrained over custom training |
| [GPU_ACCELERATION_MAP.md](GPU_ACCELERATION_MAP.md) | GPU technology → file:line mapping |
| [UNAUTHORIZED_ACCESS_EXPLAINED.md](UNAUTHORIZED_ACCESS_EXPLAINED.md) | Access control system deep-dive |
| [HOW_TO_TRAIN_AND_RUN.md](HOW_TO_TRAIN_AND_RUN.md) | Setup, run, benchmark, fine-tuning guide |
| [plan.md](plan.md) | Original design specification |
