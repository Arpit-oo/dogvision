# Graph Report - C:/code/vaibhav  (2026-04-22)

## Corpus Check
- Corpus is ~5,787 words - fits in a single context window. You may not need a graph.

## Summary
- 144 nodes · 209 edges · 21 communities detected
- Extraction: 75% EXTRACTED · 25% INFERRED · 0% AMBIGUOUS · INFERRED: 52 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Pipeline Orchestration Core|Pipeline Orchestration Core]]
- [[_COMMUNITY_Demo Entry & Video IO|Demo Entry & Video IO]]
- [[_COMMUNITY_Detection & Benchmark|Detection & Benchmark]]
- [[_COMMUNITY_Ring Buffer & Window Snapshot|Ring Buffer & Window Snapshot]]
- [[_COMMUNITY_Architecture & Pipeline Stages|Architecture & Pipeline Stages]]
- [[_COMMUNITY_GPU Stack Dependencies|GPU Stack Dependencies]]
- [[_COMMUNITY_Project Overview & Environment|Project Overview & Environment]]
- [[_COMMUNITY_Visualization Utils (drawcolor)|Visualization Utils (draw/color)]]
- [[_COMMUNITY_ROI Color Histogram (CuPy)|ROI Color Histogram (CuPy)]]
- [[_COMMUNITY_GPU Data Flow (CuPy+cuDF)|GPU Data Flow (CuPy+cuDF)]]
- [[_COMMUNITY_Fine-Tune Datasets & Workflow|Fine-Tune Datasets & Workflow]]
- [[_COMMUNITY_Tracking Module|Tracking Module]]
- [[_COMMUNITY_CPU Baseline Tools|CPU Baseline Tools]]
- [[_COMMUNITY_Train Script|Train Script]]
- [[_COMMUNITY_Tracker Tuning (ByteTrackBoTSORT)|Tracker Tuning (ByteTrack/BoTSORT)]]
- [[_COMMUNITY_Run Demo Docs|Run Demo Docs]]
- [[_COMMUNITY_YOLO detect_only Rationale|YOLO detect_only Rationale]]
- [[_COMMUNITY_YOLO track Rationale|YOLO track Rationale]]
- [[_COMMUNITY_README Quick Start|README Quick Start]]
- [[_COMMUNITY_Target Environment Note|Target Environment Note]]
- [[_COMMUNITY_Out of Scope Note|Out of Scope Note]]

## God Nodes (most connected - your core abstractions)
1. `Pipeline` - 13 edges
2. `DetectionRing` - 11 edges
3. `DogDetector` - 11 edges
4. `VideoWriter` - 11 edges
5. `pip requirements.txt` - 10 edges
6. `DogTracker` - 9 edges
7. `GPU Stack` - 8 edges
8. `Three-Thread Pipeline Architecture` - 8 edges
9. `main()` - 7 edges
10. `AnalyticsWindow` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Pipeline Diagram (decode -> YOLOv8+ByteTrack -> CuPy/cuDF)` --semantically_similar_to--> `Three-Thread Pipeline Architecture`  [INFERRED] [semantically similar]
  HOW_TO_TRAIN_AND_RUN.md → plan.md
- `BoTSORT alternative tracker` --semantically_similar_to--> `Stage 4 - ByteTrack Tracking`  [INFERRED] [semantically similar]
  HOW_TO_TRAIN_AND_RUN.md → plan.md
- `Benchmark GPU vs CPU` --semantically_similar_to--> `CPU-vs-GPU Benchmark Deliverable`  [INFERRED] [semantically similar]
  HOW_TO_TRAIN_AND_RUN.md → plan.md
- `CPU-vs-GPU benchmark for the report.  Runs the detector on the same video twice:` --uses--> `DogDetector`  [INFERRED]
  C:\code\vaibhav\benchmark.py → C:\code\vaibhav\detection\yolo.py
- `main()` --calls--> `Pipeline`  [INFERRED]
  C:\code\vaibhav\demo.py → C:\code\vaibhav\pipeline\orchestrator.py

## Hyperedges (group relationships)
- **Three-Thread GPU Pipeline Flow** — plan_stage1_ingestion, plan_stage2_detection, plan_stage4_bytetrack, plan_stage5_cudf_analytics, plan_stage6_output [EXTRACTED 1.00]
- **GPU-First Stack (PyTorch+YOLOv8+TRT+CuPy+cuDF)** — plan_pytorch, plan_ultralytics_yolov8, plan_tensorrt_fp16, plan_cupy, plan_cudf [EXTRACTED 1.00]
- **YOLOv8 Fine-Tune Workflow (dataset -> YOLO format -> train -> TRT export)** — howto_stanford_dogs, howto_yolo_format, howto_finetune, plan_tensorrt_fp16 [EXTRACTED 0.90]

## Communities

### Community 0 - "Pipeline Orchestration Core"
Cohesion: 0.16
Nodes (10): FramePacket, Pipeline, Threaded pipeline: decode → inference+track → analytics → display/write.  v1 run, ResultPacket, DetectionRing, Column-oriented ring on GPU. O(1) append per row., DogTracker, Track (+2 more)

### Community 1 - "Demo Entry & Video IO"
Cohesion: 0.14
Nodes (12): main(), _parse_source(), Entry point.  Examples:     python demo.py --source video.mp4     python demo.py, main(), CPU demo with high-recall settings + bbox persistence for smooth output., FrameMeta, iter_frames(), _open() (+4 more)

### Community 2 - "Detection & Benchmark"
Cohesion: 0.18
Nodes (12): _bench_cpu(), _bench_gpu(), main(), CPU-vs-GPU benchmark for the report.  Runs the detector on the same video twice:, _read_video(), detect_only(), Detection, DogDetector (+4 more)

### Community 3 - "Ring Buffer & Window Snapshot"
Cohesion: 0.2
Nodes (4): Preallocated CuPy ring buffer for per-detection rows.  Avoids per-frame cuDF con, Return the ring unrolled into chronological order., RingSnapshot, cuDF rolling-window aggregates over the DetectionRing snapshot.

### Community 4 - "Architecture & Pipeline Stages"
Cohesion: 0.22
Nodes (9): Pipeline Diagram (decode -> YOLOv8+ByteTrack -> CuPy/cuDF), Three-Thread Pipeline Architecture, Rationale: Drop-Oldest Queue Policy for Real-Time, Error / Edge Cases, Performance Targets (>=25 FPS 1080p RTX 3060), Repo Layout (detection/tracking/analytics/pipeline/utils), Stage 1 - Ingestion, Stage 2 - Detection (YOLOv8) (+1 more)

### Community 5 - "GPU Stack Dependencies"
Cohesion: 0.33
Nodes (9): GPU Stack, Numba CUDA, OpenCV (cv2), PyAV / decord decode, PyTorch (CUDA, FP16/AMP), Ultralytics YOLOv8, PyArrow (Parquet), pip requirements.txt (+1 more)

### Community 6 - "Project Overview & Environment"
Cohesion: 0.25
Nodes (8): Environment Setup (WSL2 + conda), Rationale: WSL2 Required Because cuDF is Linux-only, COCO Class 16 (dog), Hard Constraints (dog-only, GPU-first), Real-time Dog Detection Objective, TensorRT FP16 Export, Rationale: TensorRT FP16 Primary, PyTorch FP16 Fallback, Dogvision Project

### Community 7 - "Visualization Utils (draw/color)"
Cohesion: 0.33
Nodes (5): id_to_bgr(), Deterministic color for a tracker ID. Same ID → same color every run., draw_detections(), overlay_hud(), Draw bbox + track_id + conf on a BGR frame. Returns the same frame.

### Community 8 - "ROI Color Histogram (CuPy)"
Cohesion: 0.4
Nodes (5): bgr_to_hsv_gpu(), GPU color-histogram per ROI using CuPy.  Crops each detection from the frame on-, Minimal BGR→HSV on GPU. Matches cv2.COLOR_BGR2HSV semantics approximately., Return an (N, bins**3) float32 array of normalized HSV histograms on CPU.      K, roi_histograms()

### Community 9 - "GPU Data Flow (CuPy+cuDF)"
Cohesion: 0.4
Nodes (6): Key GPU Optimizations, cuDF (RAPIDS), CuPy Ring Buffer / Kernels, Rationale: CuPy Ring Buffer for O(1) Append, Stage 3 - GPU ROI Features (HSV histogram), Stage 5 - cuDF Rolling Window Analytics

### Community 10 - "Fine-Tune Datasets & Workflow"
Cohesion: 0.33
Nodes (6): Fine-Tune YOLOv8 on Dogs, Open Images V7, Oxford-IIIT Pet Dataset, Roboflow / CVAT Labeling, Stanford Dogs Dataset, YOLO Dataset Format

### Community 11 - "Tracking Module"
Cohesion: 0.5
Nodes (1): Thin wrapper that keeps trajectories per track_id.  Ultralytics handles ByteTrac

### Community 12 - "CPU Baseline Tools"
Cohesion: 0.5
Nodes (4): Benchmark GPU vs CPU, CPU-vs-GPU Benchmark Deliverable, ONNX Runtime (CPU baseline), pandas (CPU baseline)

### Community 13 - "Train Script"
Cohesion: 0.67
Nodes (1): Optional: fine-tune YOLOv8 on dog-only datasets.  Not required for the default p

### Community 14 - "Tracker Tuning (ByteTrack/BoTSORT)"
Cohesion: 0.67
Nodes (3): BoTSORT alternative tracker, Troubleshooting (TRT, cuDF, FPS, OOM), Stage 4 - ByteTrack Tracking

### Community 15 - "Run Demo Docs"
Cohesion: 0.67
Nodes (3): CLI Flags (--source, --weights, --no-trt), configs/default.yaml, Run the Demo (demo.py)

### Community 16 - "YOLO detect_only Rationale"
Cohesion: 1.0
Nodes (1): Run detect+track on a single BGR frame. Returns dog-only detections.

### Community 17 - "YOLO track Rationale"
Cohesion: 1.0
Nodes (1): Detection without tracking — used by the CPU/GPU benchmark.

### Community 19 - "README Quick Start"
Cohesion: 1.0
Nodes (1): Quick Start

### Community 20 - "Target Environment Note"
Cohesion: 1.0
Nodes (1): Target Environment (WSL2, CUDA 12.x, Python 3.10)

### Community 21 - "Out of Scope Note"
Cohesion: 1.0
Nodes (1): Out of Scope (v1) - breed, pose, multi-class

## Knowledge Gaps
- **44 isolated node(s):** `Entry point.  Examples:     python demo.py --source video.mp4     python demo.py`, `CPU demo with high-recall settings + bbox persistence for smooth output.`, `Optional: fine-tune YOLOv8 on dog-only datasets.  Not required for the default p`, `Preallocated CuPy ring buffer for per-detection rows.  Avoids per-frame cuDF con`, `Column-oriented ring on GPU. O(1) append per row.` (+39 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Tracking Module`** (4 nodes): `Thin wrapper that keeps trajectories per track_id.  Ultralytics handles ByteTrac`, `unique_count()`, `__init__.py`, `tracker.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Train Script`** (3 nodes): `main()`, `train.py`, `Optional: fine-tune YOLOv8 on dog-only datasets.  Not required for the default p`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `YOLO detect_only Rationale`** (1 nodes): `Run detect+track on a single BGR frame. Returns dog-only detections.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `YOLO track Rationale`** (1 nodes): `Detection without tracking — used by the CPU/GPU benchmark.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `README Quick Start`** (1 nodes): `Quick Start`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Target Environment Note`** (1 nodes): `Target Environment (WSL2, CUDA 12.x, Python 3.10)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Out of Scope Note`** (1 nodes): `Out of Scope (v1) - breed, pose, multi-class`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Pipeline` connect `Pipeline Orchestration Core` to `Demo Entry & Video IO`, `Detection & Benchmark`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `DogDetector` connect `Detection & Benchmark` to `Pipeline Orchestration Core`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Why does `DetectionRing` connect `Pipeline Orchestration Core` to `Ring Buffer & Window Snapshot`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `Pipeline` (e.g. with `DetectionRing` and `AnalyticsWindow`) actually correct?**
  _`Pipeline` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `DetectionRing` (e.g. with `FramePacket` and `ResultPacket`) actually correct?**
  _`DetectionRing` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `DogDetector` (e.g. with `CPU-vs-GPU benchmark for the report.  Runs the detector on the same video twice:` and `FramePacket`) actually correct?**
  _`DogDetector` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `VideoWriter` (e.g. with `FramePacket` and `ResultPacket`) actually correct?**
  _`VideoWriter` has 6 INFERRED edges - model-reasoned connections that need verification._