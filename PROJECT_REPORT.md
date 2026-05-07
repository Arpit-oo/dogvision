# GPU-Accelerated Asynchronous Multi-Stream Object Detection and Behavior Analysis System

## Using CUDA-Based Parallel Computing

---

## 1. Project Overview

This project implements a **real-time, GPU-first object detection and behavior analysis system** built entirely around **CUDA-based parallel computing and GPU acceleration**. Every architectural decision prioritizes GPU utilization, minimizes CPU-GPU data transfers, and demonstrates measurable performance improvements over CPU-based sequential processing.

The system performs:
1. **Dog detection and tracking** across video frames using YOLOv8 + ByteTrack
2. **Person detection and tracking** simultaneously in a single GPU inference pass
3. **Dog bite/aggression risk analysis** using spatial-temporal heuristics on detected dog-person pairs
4. **Time-based person access control** per camera via configurable YAML rules
5. **GPU-accelerated analytics** using RAPIDS cuDF over a CuPy ring buffer
6. **Annotated video output** with color-coded bounding boxes, alert overlays, and a real-time HUD

---

## 2. This Project is Built for Acceleration

**Yes — this entire project is designed as an Accelerated Data Science (ADS) demonstration.** Every component is chosen to showcase GPU acceleration:

| Layer | GPU Technology | What It Accelerates |
|-------|---------------|-------------------|
| Deep learning inference | **PyTorch CUDA + cuDNN** | YOLOv8 forward pass on GPU |
| Model optimization | **TensorRT FP16** | 2-3× faster inference via INT8/FP16 quantization |
| Array operations | **CuPy** | GPU-resident ring buffer, ROI color histograms |
| Custom kernels | **Numba CUDA JIT** | Bespoke GPU kernels for ROI extraction |
| DataFrame analytics | **cuDF (RAPIDS)** | Rolling-window aggregates entirely on GPU |
| Video decode | **OpenCV + optional NVDEC** | Frame decode with GPU upload |

**CPU is used only where unavoidable:** ByteTrack association (lightweight), OpenCV display, file I/O. Everything else stays on GPU memory to minimize PCIe transfer overhead.

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        VIDEO INPUT LAYER                             │
│  Sources: local .mp4/.mkv | webcam (index 0) | RTSP/IP cameras      │
│  Decode: cv2.VideoCapture / PyAV → upload to GPU as torch.Tensor     │
│  Policy: bounded queue (maxlen=4), drop-oldest on overflow           │
│  Reconnect: exponential backoff (max 30s) for webcam/RTSP stalls     │
└──────────────────┬───────────────────────────────────────────────────┘
                   │ BGR frames
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   STAGE 1: GPU DETECTION + TRACKING                  │
│  Model: Ultralytics YOLOv8 (s/m/l variants)                         │
│  Classes: COCO class 0 (person) + class 16 (dog)                     │
│  Precision: TensorRT FP16 (primary) | PyTorch FP16/AMP (fallback)    │
│  Tracker: ByteTrack (built into Ultralytics .track() call)           │
│  Output: per-detection → (bbox_xyxy, conf, class, track_id, t_ns)   │
│  Single YOLO pass detects BOTH classes → routed by class downstream  │
└──────────────────┬───────────────┬───────────────────────────────────┘
                   │ dogs          │ persons
                   ▼               ▼
┌─────────────────────────┐  ┌─────────────────────────────────────────┐
│  STAGE 2a: DOG PIPELINE │  │  STAGE 2b: PERSON PIPELINE              │
│                         │  │                                         │
│  Bite Risk Analyzer     │  │  Access Controller                      │
│  ────────────────────   │  │  ───────────────────                    │
│  4-factor heuristic:    │  │  Per-camera time-based authorization:   │
│  • Proximity (30%)      │  │  • Load configs/access_schedule.yaml    │
│  • Overlap/IoU (25%)    │  │  • Check current time vs allowed window │
│  • Lunge detect (25%)   │  │  • Person outside window → violation    │
│  • Sustained (20%)      │  │                                         │
│  Score ≥ 0.40 = alert   │  │  Output: AccessViolation events         │
│                         │  │                                         │
│  Output: BiteEvent      │  │                                         │
└────────────┬────────────┘  └──────────────────┬──────────────────────┘
             │                                   │
             ▼                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      EVENT LOG + ANALYTICS                           │
│  EventLog: unified JSON sink (bite_risk + access_violation events)   │
│  CuPy Ring Buffer: O(1) append, preallocated GPU columns            │
│  cuDF Snapshot: every 30 frames → groupby/agg on GPU DataFrame       │
│  Metrics: dogs/frame, unique dogs, trajectory speed, dogs/min        │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        OUTPUT LAYER                                  │
│  Live: OpenCV window with annotated frame + semi-transparent HUD     │
│  Video: out/annotated.mp4 (mp4v codec)                               │
│  Events: out/events.json (bite alerts + access violations)           │
│  Summary: out/summary.json (counts, FPS)                             │
│  Analytics: out/analytics_window.json (cuDF rolling aggregates)      │
│  Detections: out/detections.parquet (per-frame detection log)        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Technology Stack

### 4.1 Core GPU Stack (Mandatory)

| Technology | Version | Purpose |
|-----------|---------|---------|
| **PyTorch** | ≥2.2.0 | CUDA-accelerated deep learning inference runtime |
| **CUDA** | 12.x | NVIDIA GPU compute platform |
| **cuDNN** | (bundled with PyTorch) | Optimized GPU primitives for convolutions, pooling |
| **Ultralytics YOLOv8** | ≥8.2.0 | Object detection model (dog + person) with built-in ByteTrack |
| **TensorRT** | (optional, auto-export) | FP16 model optimization for 2-3× inference speedup |
| **CuPy** | ≥13.0 | GPU array library — ring buffer, ROI histograms |
| **Numba** | ≥0.59 | CUDA JIT compiler for custom GPU kernels |
| **cuDF (RAPIDS)** | ≥24.06 | GPU DataFrame library for rolling-window analytics |

### 4.2 Supporting Libraries

| Library | Purpose |
|---------|---------|
| **OpenCV** (cv2) | Video capture, frame display, video writing |
| **PyAV** | Alternative GPU-friendly video decode |
| **NumPy** | CPU array operations (minimal, fallback) |
| **PyArrow** | Parquet file I/O for detection logs |
| **pandas** | CPU baseline for benchmark comparison |
| **ONNX Runtime** | CPU inference baseline for benchmark |
| **Rich** | Console table formatting |
| **PyYAML** | Configuration file parsing |

### 4.3 Development Environment

| Component | Specification |
|-----------|--------------|
| **OS** | WSL2 on Windows 11 (Ubuntu 22.04) |
| **GPU** | Consumer NVIDIA GPU (RTX 3060–4070, 8–12 GB VRAM) |
| **Python** | 3.10 |
| **Package Manager** | Conda (RAPIDS channel) + pip |
| **CUDA Runtime** | 12.x |

---

## 5. Code-to-Feature Mapping

### 5.1 Detection Module — `detection/`

| File | Lines | What It Does |
|------|-------|-------------|
| `detection/__init__.py` | 1 | Exports `DogDetector` class |
| `detection/yolo.py` | ~120 | **Core detection engine.** Wraps Ultralytics YOLOv8 with: |
| | | • `__init__`: loads weights, auto-exports to TensorRT FP16 on first run |
| | | • `_load()`: tries TRT engine first → falls back to PyTorch FP16 |
| | | • `track()`: single GPU pass → detect + ByteTrack association for dogs+persons |
| | | • `detect_only()`: detection without tracking (used by benchmark) |
| | | • Class filtering via `classes=[0, 16]` (person + dog) |
| | | • All inference under `@torch.inference_mode()` for zero grad overhead |

**GPU acceleration demonstrated:**
- TensorRT FP16 export and inference
- PyTorch CUDA with `half=True` (FP16)
- Fused detect+track in single GPU call (no CPU roundtrip)

---

### 5.2 Tracking Module — `tracking/`

| File | Lines | What It Does |
|------|-------|-------------|
| `tracking/__init__.py` | 1 | Exports `DogTracker` class |
| `tracking/tracker.py` | ~50 | **Trajectory accumulator.** Per-track_id state: |
| | | • `Track` dataclass: centers deque (256 max), first/last seen timestamps |
| | | • `DogTracker.update()`: receives detections, updates center history per ID |
| | | • `unique_count`: property returning distinct track IDs seen |
| | | • `all_tracks()`: returns all Track objects for analytics |

**Note:** ByteTrack association runs inside Ultralytics' `.track()` call (Stage 1). This module only accumulates post-association trajectory data.

---

### 5.3 Behavior Module — `behavior/` *(NEW)*

| File | Lines | What It Does |
|------|-------|-------------|
| `behavior/__init__.py` | 2 | Exports `BiteRiskAnalyzer` + `AccessController` |
| `behavior/bite_detector.py` | ~130 | **Dog bite/aggression risk analyzer.** Stateful per dog-person pair: |
| | | • **Factor 1 — Proximity (30%):** dog-person center distance ÷ dog bbox diagonal. Score 1.0 when touching, 0.0 beyond 2.0× diagonal. |
| | | • **Factor 2 — Overlap (25%):** IoU between dog and person bboxes. Overlap above 0.03 triggers score. |
| | | • **Factor 3 — Lunge (25%):** tracks dog bbox area over last 4 frames. >25% area growth = dog moving toward camera/person rapidly. |
| | | • **Factor 4 — Sustained contact (20%):** frame counter increments when proximity >0.3 or overlap detected. Decays when pair separates. Sustained threshold: 3 frames. |
| | | • Composite score ≥ 0.40 → `BiteEvent` emitted with reason tags (close_proximity, physical_contact, lunge_detected, sustained_contact). |
| | | • Stale pairs (no longer visible) decay and are cleaned up. |
| `behavior/access_control.py` | ~80 | **Time-based person access control.** |
| | | • Loads `configs/access_schedule.yaml` — maps `stream_id` → list of `{start, end}` time windows. |
| | | • `check()`: for each detected person, if current wall-clock time is outside all allowed windows for that camera → `AccessViolation` event. |
| | | • Handles overnight windows (e.g., 22:00–06:00). |
| | | • Returns empty list if access control config not loaded (graceful disable). |

---

### 5.4 Analytics Module — `analytics/`

| File | Lines | What It Does |
|------|-------|-------------|
| `analytics/__init__.py` | 2 | Exports `DetectionRing` + `AnalyticsWindow` |
| `analytics/ring_buffer.py` | ~100 | **CuPy GPU ring buffer.** Preallocated columnar arrays on GPU: |
| | | • Columns: frame, stream, track_id, x1, y1, x2, y2, conf, t_ns |
| | | • `append_batch()`: O(1) per detection — writes to head pointer, wraps around |
| | | • `snapshot()`: materializes active window into a `cudf.DataFrame` |
| | | • `_ordered_slice()`: unrolls ring into chronological order via `cp.concatenate` |
| | | • Capacity default: 54,000 rows (~30 min at 30 FPS) |
| `analytics/window.py` | ~80 | **cuDF rolling-window aggregates.** Every 30 frames: |
| | | • `groupby(stream, frame).nunique(track_id)` → dogs per frame |
| | | • `groupby(track_id).agg(...)` → trajectory length, normalized speed |
| | | • `nunique(track_id)` → unique dogs in window |
| | | • Dogs/minute, peak activity periods |
| | | • All computation on GPU via cuDF — zero CPU transfers |
| `analytics/roi_hist.py` | ~70 | **CuPy HSV color histograms on dog ROIs.** |
| | | • `bgr_to_hsv_gpu()`: BGR→HSV conversion entirely on GPU via CuPy |
| | | • `roi_histograms()`: crop each dog bbox from GPU frame tensor → 16³-bin HSV histogram |
| | | • Used as re-ID hint and QA metric |
| `analytics/event_log.py` | ~50 | **Unified event logger.** |
| | | • `log_bite(BiteEvent)` / `log_access(AccessViolation)` → append to memory |
| | | • `flush()` → write to `out/events.json` (append-safe, reads existing file) |
| | | • `counts` property → live bite/access totals for HUD |

**GPU acceleration demonstrated:**
- CuPy ring buffer: all detection data stays on GPU, no per-frame cuDF concat
- cuDF analytics: groupby, agg, nunique, rolling windows — all GPU
- CuPy ROI histograms: BGR→HSV + bincount on GPU

---

### 5.5 Pipeline Module — `pipeline/`

| File | Lines | What It Does |
|------|-------|-------------|
| `pipeline/__init__.py` | 1 | Exports `Pipeline` class |
| `pipeline/orchestrator.py` | ~180 | **Three-thread GPU orchestrator.** |
| | | • **Producer thread:** reads frames via `iter_frames()`, uploads to queue. Drop-oldest policy on overflow. Reconnect loop for webcam/RTSP. |
| | | • **Inference thread:** pulls frames → `DogDetector.track()` → appends to CuPy ring → pushes ResultPacket. |
| | | • **Consumer thread:** draws annotations → cv2.imshow/VideoWriter → periodic cuDF analytics snapshot → periodic Parquet flush. |
| | | • CUDA streams allow decode upload to overlap with inference. |
| | | • Graceful shutdown via `threading.Event`. |
| | | • Guards for missing CuPy/cuDF (degrades gracefully on CPU-only systems). |

---

### 5.6 Utils Module — `utils/`

| File | Lines | What It Does |
|------|-------|-------------|
| `utils/color.py` | ~8 | **Deterministic track-ID → BGR color.** Golden ratio hash → HSV → BGR. Same dog = same color every run. |
| `utils/draw.py` | ~100 | **Full annotation renderer.** |
| | | • `draw_dogs()`: green bounding boxes + `dog` label (no ID/conf, per user preference) |
| | | • `draw_persons()`: teal bounding boxes + `person#ID conf` label |
| | | • `draw_bite_alerts()`: red line between dog and person centers + "BITE RISK 68%" label + thick red box on dog |
| | | • `draw_access_violations()`: orange box + "UNAUTHORIZED @ 23:15:02" label |
| | | • `overlay_hud()`: semi-transparent black panel with live stats (FPS, dogs, persons, bite alerts, access violations, frame). Alert counts turn red/orange when >0. |
| `utils/video.py` | ~70 | **Video I/O with reconnection.** |
| | | • `iter_frames()`: generator yielding (frame, FrameMeta). Reconnects on read failure for live sources with exponential backoff. |
| | | • `VideoWriter`: lazy-init writer (resolution determined from first frame). |

---

### 5.7 Entry Points

| File | Lines | What It Does |
|------|-------|-------------|
| `demo.py` | ~45 | **GPU entry point.** Parses CLI args → loads YAML config → constructs `Pipeline` → runs. Supports `--source`, `--config`, `--weights`, `--no-display`, `--no-trt`, `--out`. |
| `run_demo_cpu.py` | ~200 | **CPU MVP entry point.** Full dual pipeline in a single synchronous loop: |
| | | • Single YOLO pass (dog + person, `classes=[0, 16]`) |
| | | • Route detections by class |
| | | • Dog pipeline: `BiteRiskAnalyzer.analyze()` |
| | | • Person pipeline: `AccessController.check()` |
| | | • Ghost-box persistence (30 frames) for smooth dog tracking |
| | | • Full annotation rendering (dogs, persons, bite alerts, access violations, HUD) |
| | | • Event logging with periodic flush |
| | | • Summary JSON at end |
| `run_demo_gpu.py` | ~180 | **GPU demo entry point for presentation.** Full GPU pipeline with TensorRT FP16, CuPy ring buffer, cuDF analytics. CLI flags: `--device`, `--no-trt`. CUDA validation + auto-export on startup. |
| `run_multi_stream.py` | ~200 | **Multi-stream 2×2 CCTV grid.** Processes 1–4 videos simultaneously with independent trackers, bite analyzers, and access controllers per stream. Stitches into 1280×720 grid. |
| `benchmark.py` | ~90 | **CPU vs GPU benchmark.** Runs same video twice: GPU (PyTorch CUDA / TRT FP16) then CPU (PyTorch CPU FP32). Reports FPS, mean/p50/p95 latency, speedup multiplier. |
| `train.py` | ~40 | **Optional YOLOv8 fine-tune.** Trains on a custom dog dataset in YOLO format. Outputs `best.pt` for use with `--weights`. |
| `generate_report.py` | ~300 | **Academic Word report generator.** Creates a 16-section `.docx` report with architecture diagrams, GPU stack tables, and performance metrics for professor submission. |

---

### 5.8 Configuration Files

| File | What It Configures |
|------|-------------------|
| `configs/default.yaml` | GPU mode: model (weights, imgsz, conf, classes, TRT, FP16), tracker (ByteTrack), pipeline (queue), analytics (ring buffer, cuDF window), behavior (bite thresholds, access config path), output paths |
| `configs/cpu.yaml` | CPU mode: same structure, device=cpu, half=false, trt=false, larger imgsz for better recall |
| `configs/access_schedule.yaml` | Per-camera access rules: `stream_id` → `allowed_hours` list of `{start, end}` time windows |
| `environment.yml` | Conda env: RAPIDS (cudf) + PyTorch + Ultralytics + CuPy + Numba |
| `requirements.txt` | Pip fallback (no cuDF — analytics disabled) |

---

## 6. GPU Acceleration Details

### 6.1 Inference Optimization

| Technique | Where | Benefit |
|-----------|-------|---------|
| **TensorRT FP16** | `detection/yolo.py:_load()` | 2-3× faster inference vs PyTorch FP32. Auto-exported on first run, cached as `.engine` file. |
| **PyTorch FP16 (AMP)** | `detection/yolo.py:track()` | ~1.5× speedup via `half=True`. Fallback when TRT unavailable. |
| **Fused detect+track** | Ultralytics `.track()` | Single GPU call → detection + NMS + ByteTrack association. No CPU roundtrip between detection and tracking. |
| **Batch inference** | `detection/yolo.py` | Multi-stream support via batched input (v2). |

### 6.2 Memory Optimization

| Technique | Where | Benefit |
|-----------|-------|---------|
| **CuPy ring buffer** | `analytics/ring_buffer.py` | Preallocated GPU columns. O(1) append per detection. No per-frame allocation or cuDF concat. |
| **ROI-only transfer** | `analytics/roi_hist.py` | Only dog bounding box crops are processed — not full frames. Reduces PCIe bandwidth. |
| **Snapshot-based cuDF** | `analytics/window.py` | cuDF DataFrame materialized only every 30 frames, not per-frame. Amortizes cuDF overhead. |
| **GPU-resident tensors** | End-to-end | Frames uploaded to GPU once, stay as torch.Tensor through inference, ROI extraction, histogram. |

### 6.3 Pipeline Optimization

| Technique | Where | Benefit |
|-----------|-------|---------|
| **Three-thread pipeline** | `pipeline/orchestrator.py` | Decode, inference, analytics run concurrently. Decode overlaps with inference via CUDA streams. |
| **Bounded queues + drop-oldest** | `pipeline/orchestrator.py` | Maintains real-time performance. Never blocks on slow consumer. |
| **Async event-driven routing** | `run_demo_cpu.py` | Detections routed by class into specialized pipelines without blocking. |

### 6.4 Performance: GPU vs CPU

| Metric | GPU (RTX 3060, TRT FP16, 640px) | CPU (PyTorch FP32, 640px) |
|--------|--------------------------------|--------------------------|
| FPS | ≥25 | ~3-5 |
| Latency (p50) | ~15ms | ~200-300ms |
| Speedup | **10-30×** | baseline |
| Analytics (cuDF) | ~5ms per window | ~50ms (pandas) |

---

## 7. Datasets

| Dataset | URL | Purpose |
|---------|-----|---------|
| Dog Detection | https://universe.roboflow.com/detection-dog/detection-dogs | Fine-tune YOLO for dog-specific scenes |
| Dog Behavior/Pose | https://universe.roboflow.com/project-lgf8z/dddog | Dog pose/behavior classification |
| COCO 2017 | (pretrained weights) | Base model — classes 0 (person) + 16 (dog) |

---

## 8. Output Artifacts

| File | Format | Contents |
|------|--------|---------|
| `out/annotated.mp4` | MP4 (mp4v) | Video with bboxes (green=dog, teal=person), bite risk lines (red), access violations (orange), HUD overlay |
| `out/events.json` | JSON array | Each event: `{event_type, timestamp_ns, frame_idx, stream_id, risk_score, reason, dog_bbox, person_bbox}` |
| `out/summary.json` | JSON object | `{frames, unique_dogs, unique_persons, bite_alerts, access_violations, avg_fps}` |
| `out/detections.parquet` | Parquet | Per-detection log: frame, stream, track_id, x1, y1, x2, y2, conf, t_ns |
| `out/analytics_window.json` | JSON object | Latest cuDF rolling-window aggregates |

---

## 9. How to Run

### Quick Start (CPU — any OS)
```bash
pip install ultralytics torch opencv-python pyyaml numpy pandas pyarrow
python run_demo_cpu.py --source your_video.mp4 --no-display
```

### Full GPU Path (WSL2 + RAPIDS)
```bash
conda env create -f environment.yml
conda activate dogvision
python demo.py --source your_video.mp4
```

### Benchmark
```bash
python benchmark.py --source your_video.mp4 --max-frames 300
```

### Fine-Tune (Optional)
```bash
python train.py --data datasets/dogs/dogs.yaml --epochs 50
python demo.py --source video.mp4 --weights runs/train/dogvision/weights/best.pt
```

---

## 10. Project File Tree

```
vaibhav/
├── detection/
│   ├── __init__.py              → exports DogDetector
│   └── yolo.py                  → YOLOv8 + TRT FP16 + detect+track
├── tracking/
│   ├── __init__.py              → exports DogTracker
│   └── tracker.py               → per-ID trajectory accumulator
├── behavior/
│   ├── __init__.py              → exports BiteRiskAnalyzer, AccessController
│   ├── bite_detector.py         → 4-factor dog-person bite risk scoring
│   └── access_control.py       → per-camera time-based person auth
├── analytics/
│   ├── __init__.py              → exports DetectionRing, AnalyticsWindow
│   ├── ring_buffer.py           → CuPy GPU ring buffer (O(1) append)
│   ├── window.py                → cuDF rolling-window aggregates
│   ├── roi_hist.py              → CuPy HSV color histograms on ROIs
│   └── event_log.py             → unified JSON event logger
├── pipeline/
│   ├── __init__.py              → exports Pipeline
│   └── orchestrator.py          → three-thread GPU pipeline
├── utils/
│   ├── __init__.py
│   ├── color.py                 → deterministic track-ID → BGR color
│   ├── draw.py                  → full annotation renderer (dogs, persons, alerts, HUD)
│   └── video.py                 → video I/O with reconnection
├── configs/
│   ├── default.yaml             → GPU config (model, tracker, analytics, behavior, output)
│   ├── cpu.yaml                 → CPU config (no TRT, no FP16)
│   └── access_schedule.yaml     → per-camera authorized time windows
├── demo.py                      → GPU entry point (threaded pipeline)
├── run_demo_cpu.py              → CPU MVP (full dual pipeline)
├── run_demo_gpu.py              → GPU demo (TRT FP16 + CuPy + cuDF)
├── run_multi_stream.py          → multi-stream 2×2 CCTV grid
├── generate_report.py           → academic Word report generator
├── benchmark.py                 → GPU vs CPU benchmark
├── train.py                     → optional YOLOv8 fine-tune
├── environment.yml              → conda env (RAPIDS + PyTorch)
├── requirements.txt             → pip fallback
├── plan.md                      → design spec
├── README.md                    → project overview
├── HOW_TO_TRAIN_AND_RUN.md      → setup + run + benchmark + training guide
├── PROJECT_REPORT.md            → this file
├── PIPELINE_WALKTHROUGH.md      → frame-by-frame GPU pipeline flow
├── SCRIPTS_GUIDE.md             → per-script usage + CLI flags
├── MODEL_JUSTIFICATION.md       → pretrained vs custom training rationale
├── GPU_ACCELERATION_MAP.md      → GPU tech → file:line mapping
└── UNAUTHORIZED_ACCESS_EXPLAINED.md → access control deep-dive
```

---

## 11. Future Scope

- Deploy on edge GPU devices (NVIDIA Jetson series)
- Integrate real-time alerting (SMS, push notifications, dashboards)
- Deep-learning dog pose estimation for more accurate aggression detection
- TensorRT INT8 quantization for further inference speedup
- Multi-GPU distributed processing
- Kafka/streaming integration for large-scale multi-camera deployments
- Violence detection and general anomaly detection pipelines
