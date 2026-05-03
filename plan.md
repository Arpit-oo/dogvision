# GPU-Accelerated Object Detection & Behavior Analysis — Spec

## 1. Objective
Real-time system that (1) detects **dogs and persons** in video, (2) tracks each across
frames with unique IDs, (3) analyzes **dog bite/aggression risk** via proximity and motion
heuristics, (4) enforces **time-based person access control** per camera, (5) runs
GPU-accelerated analytics over detections with RAPIDS cuDF, and (6) outputs annotated
video + event log + structured insights. Everything feasible runs on GPU; CPU work is minimized.

## 2. Hard Constraints
- Two classes: **dog** (COCO class 16) + **person** (COCO class 0).
- GPU-first: decode → inference → ROI features → analytics stay on GPU where practical.
- No person/behavior/multi-class detection, no distributed systems, no cloud deploy.

## 3. Target Environment
- **Hardware:** consumer GPU (RTX 3060–4070, 8–12 GB VRAM).
- **OS:** WSL2 on Windows 11 (Ubuntu 22.04), CUDA 12.x, cuDNN.
- **Language:** Python 3.10.
- **Streams:** v1 = single source; v2 = 2–4 streams via batched inference.

## 4. GPU Stack
- PyTorch (CUDA, FP16/AMP) — inference runtime.
- Ultralytics YOLOv8 (pretrained COCO, class-filtered to `dog` + `person`).
- **TensorRT FP16** export as the primary inference path; PyTorch FP16 as fallback.
- CuPy — ring buffer for detection metadata; color-histogram kernels on ROIs.
- Numba CUDA — any bespoke kernels (e.g., bbox-area batch ops).
- cuDF (RAPIDS) — rolling-window analytics DataFrames.
- OpenCV (cv2) — display window + file/webcam capture.
- PyAV or decord — video file decode; cv2 VideoCapture for webcam.

## 5. Architecture

```
[Decoder thread] --frames--> [Inference thread] --det+crops--> [Pipeline Router]
   (PyAV/cv2)                 (YOLOv8 TRT FP16)                      |
                              [ByteTrack per-class]           ┌───────┴────────┐
                                                              │                │
                                                     [Dog Pipeline]    [Person Pipeline]
                                                      bite risk        access control
                                                      proximity+       time-based
                                                      lunge+overlap    per-camera
                                                              │                │
                                                     [Event Log + cuDF Analytics + HUD]
```

Three threads communicate via bounded queues; CUDA streams let decode upload overlap with
inference. Detections are routed by class into two async pipelines:
- **Dog Pipeline:** bite risk analysis (proximity, lunge, overlap, sustained contact).
- **Person Pipeline:** access control (per-camera time windows via YAML config).

### Stage 1 — Ingestion
- Sources: local `.mp4/.mkv` and live webcam (v1). Multi-stream (v2) via repeated `--source`.
- Decode on CPU (PyAV/cv2), upload to GPU as `torch.Tensor` on the inference stream.
- Dropped-frame policy: queue maxlen=4; drop oldest on overflow to preserve real-time.
- Reconnect/backoff loop for webcam disconnects and stream timeouts.

### Stage 2 — Detection (YOLOv8)
- Model: `yolov8s.pt` default (configurable to n/m). Exported to TensorRT FP16 on first run.
- Class filter: keep only `cls == 16` (dog).
- Conf threshold **0.35**, IoU-NMS **0.5**.
- Batch inference across active streams when N>1.
- Output per detection: `bbox_xyxy`, `conf`, `frame_idx`, `stream_id`, `t_ns`.

### Stage 3 — GPU ROI features
- CuPy kernel crops each dog bbox from the GPU frame tensor.
- Compute per-crop color histogram (HSV, 16×16×16 bins) for future re-ID hints and QA.
- Store histogram vector alongside detection row; not used for tracking in v1.

### Stage 4 — Tracking (ByteTrack)
- ByteTrack over bbox+conf, per stream.
- Track state: `track_id`, `trajectory` (list of centers), `first_seen`, `last_seen`.
- Unique-dog count = distinct `track_id`s (known to overcount on long occlusion — documented).

### Stage 5 — cuDF Analytics (rolling window)
- Hot storage: preallocated **CuPy ring buffer** (columns: `frame`, `stream`, `track_id`,
  `x1,y1,x2,y2`, `conf`, `t_ns`). O(1) append, no per-frame cuDF concat.
- Every **30 frames** (~1 s @ 30 FPS) snapshot the active window into a `cudf.DataFrame` and run:
  - `groupby(stream, frame).size()` → dogs per frame / per stream.
  - `groupby(track_id).agg(first/last t_ns, count, bbox deltas)` → trajectory length +
    normalized speed (pixels·sec⁻¹ ÷ bbox diagonal).
  - Rolling aggregates: dogs/min, peak activity periods.
  - `nunique(track_id)` → unique dogs so far.
- Ring buffer capped (e.g. 60 s of detections); oldest rows flushed to Parquet on disk.

### Stage 6 — Output
- Live: OpenCV window with color-coded bboxes (dogs=green, persons=teal), track IDs,
  bite risk alert lines (red), access violation labels (orange), semi-transparent HUD.
- Box color: deterministic `hash(track_id) → HSV` for dogs; fixed teal for persons.
- Files:
  - `out/annotated.mp4` — rendered video with full HUD.
  - `out/detections.parquet` — per-detection log (frame, stream, track_id, bbox, conf, ts).
  - `out/events.json` — bite risk events + access violation events.
  - `out/summary.json` — run summary (counts, FPS).
  - `out/analytics_window.json` — latest rolling aggregates snapshot.

## 6. Repo Layout
```
vaibhav/
├── detection/       # YOLOv8 wrapper + TRT export
├── tracking/        # ByteTrack trajectory accumulator
├── behavior/        # bite risk analyzer + access control
├── analytics/       # CuPy ring buffer, cuDF window ops, event log, ROI hist
├── pipeline/        # threaded GPU orchestrator
├── utils/           # video IO, drawing, color hash
├── demo.py          # GPU entry point
├── run_demo_cpu.py  # CPU MVP entry point (full dual pipeline)
├── benchmark.py     # CPU-vs-GPU baseline
├── train.py         # optional YOLOv8 fine-tune
├── configs/         # model config + access schedule YAML
└── README.md
```

## 7. Demo Script
`python demo.py --source path.mp4|0 [--model yolov8s] [--trt] [--no-display] [--out out/]`

v2 will accept multiple `--source` flags for 2–4 stream batched inference.

## 8. CPU-vs-GPU Benchmark (deliverable)
- Same YOLOv8s weights on CPU (ONNX Runtime) + pandas analytics.
- Report: FPS, end-to-end latency, per-stage time, analytics-window latency.
- Numbers go in the report; script reproduces them.

## 9. Error / Edge Cases (must handle)
- Empty-frame stretches → window aggregates produce 0s, not NaN.
- Decoder stall / frame drop → queue drop-oldest; log drop count.
- GPU OOM risk on long runs → ring buffer cap + Parquet flush of old rows.
- Webcam disconnect / RTSP timeout → reconnect w/ exponential backoff (max 30 s).

## 10. Performance Targets
- ≥ 25 FPS sustained on RTX 3060 for 1080p, YOLOv8s TRT FP16, single stream.
- ≥ 20 FPS aggregate on 4× 720p streams (RTX 4070), batched inference.
- Analytics window recompute ≤ 20 ms on window size ≤ 1800 rows.

## 11. Behavior Analysis

### Dog Bite Risk (behavior/bite_detector.py)
Four-factor heuristic scoring (0–1):
- **Proximity** (30%): dog-person center distance normalized by dog diagonal.
- **Overlap** (25%): IoU between dog and person bboxes.
- **Lunge** (25%): rapid bbox area growth (>35% in 4 frames).
- **Sustained contact** (20%): frames of continuous proximity.
Score ≥ 0.55 → bite risk event logged.

### Person Access Control (behavior/access_control.py)
- YAML config (`configs/access_schedule.yaml`) maps stream IDs to allowed time windows.
- Persons detected outside allowed hours → unauthorized access event logged.

### Event Log (analytics/event_log.py)
- Unified JSON log for bite + access events.
- Periodic flush to `out/events.json`.
- Counts surfaced in HUD overlay.

## 12. Out of Scope (v1)
Breed classification, deep-learning pose estimation, zone counting,
trajectory heatmaps, dwell-time analytics, distributed ingestion, cloud deploy,
real-time SMS/push alerts, multi-GPU.
These may be revisited per future needs.

## 12. Deliverables
1. Python package with the 5-dir layout above.
2. `demo.py` working on a sample mp4 and on webcam.
3. `benchmark.py` producing a CPU-vs-GPU comparison table.
4. `README.md` covering: architecture, GPU optimizations used, memory/ring-buffer strategy,
   cuDF analytics layer, reproducible setup (WSL2 + RAPIDS conda env).
