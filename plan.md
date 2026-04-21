# GPU-First Dog Detection, Tracking & Analytics — Spec

## 1. Objective
Real-time system that (1) detects dogs in video, (2) tracks them across frames with unique IDs,
(3) runs GPU-accelerated analytics over detections with RAPIDS cuDF, and (4) outputs annotated
video + structured insights. Everything feasible runs on GPU; CPU work is minimized.

## 2. Hard Constraints
- Single class: **dog only** (COCO class 16).
- GPU-first: decode → inference → ROI features → analytics stay on GPU where practical.
- No person/behavior/multi-class detection, no distributed systems, no cloud deploy.

## 3. Target Environment
- **Hardware:** consumer GPU (RTX 3060–4070, 8–12 GB VRAM).
- **OS:** WSL2 on Windows 11 (Ubuntu 22.04), CUDA 12.x, cuDNN.
- **Language:** Python 3.10.
- **Streams:** v1 = single source; v2 = 2–4 streams via batched inference.

## 4. GPU Stack
- PyTorch (CUDA, FP16/AMP) — inference runtime.
- Ultralytics YOLOv8 (pretrained COCO, class-filtered to `dog`).
- **TensorRT FP16** export as the primary inference path; PyTorch FP16 as fallback.
- CuPy — ring buffer for detection metadata; color-histogram kernels on ROIs.
- Numba CUDA — any bespoke kernels (e.g., bbox-area batch ops).
- cuDF (RAPIDS) — rolling-window analytics DataFrames.
- OpenCV (cv2) — display window + file/webcam capture.
- PyAV or decord — video file decode; cv2 VideoCapture for webcam.

## 5. Architecture

```
[Decoder thread] --frames--> [Inference thread] --det+crops--> [Analytics thread]
   (PyAV/cv2)                 (YOLOv8 TRT FP16)                 (CuPy ring + cuDF)
                                     |
                              [ByteTrack (CPU-light)]
                                     |
                               annotated frames
                                     |
                            [Display + file writer]
```

Three threads communicate via bounded queues; CUDA streams let decode upload overlap with
inference. Tensors stay on GPU from decode → inference → ROI features.

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
- Live: OpenCV window with bbox, `track_id`, conf; console table of live analytics
  (refreshed every window).
- Box color: deterministic `hash(track_id) → HSV` so a given dog keeps its color.
- Files:
  - `out/annotated_<source>.mp4` — rendered video.
  - `out/detections.parquet` — full per-detection log (frame, stream, track_id, bbox, conf, ts).
  - `out/analytics_window.json` — latest rolling aggregates snapshot.

## 6. Repo Layout
```
vaibhav/
├── detection/       # yolov8 wrapper + TRT export
├── tracking/        # ByteTrack integration
├── analytics/       # CuPy ring buffer, cuDF window ops
├── pipeline/        # threaded orchestrator, queues, CUDA streams
├── utils/           # video io, drawing, color hash, benchmarking
├── demo.py          # entry point: `python demo.py --source x.mp4`
├── benchmark.py     # CPU-vs-GPU baseline
├── configs/         # yaml hyperparams
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

## 11. Out of Scope (v1)
Breed classification, behavior/pose, person detection, multi-class, zone counting,
trajectory heatmaps, dwell-time analytics, distributed ingestion, cloud deploy.
These may be revisited per future needs.

## 12. Deliverables
1. Python package with the 5-dir layout above.
2. `demo.py` working on a sample mp4 and on webcam.
3. `benchmark.py` producing a CPU-vs-GPU comparison table.
4. `README.md` covering: architecture, GPU optimizations used, memory/ring-buffer strategy,
   cuDF analytics layer, reproducible setup (WSL2 + RAPIDS conda env).
