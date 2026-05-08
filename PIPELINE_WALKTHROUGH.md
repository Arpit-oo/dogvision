# GPU Pipeline Walkthrough  - Detection to Output, Step by Step

Complete flow of one video frame through the CUDA-accelerated pipeline. Every step maps to a file + function + GPU technology.

---

## The Full GPU-Accelerated Pipeline (Single Frame)

```
VIDEO SOURCE (file / webcam / RTSP)
    │
    ▼ NVDEC hardware decode (GPU video decoder)         [utils/video.py:iter_frames()]
FRAME uploaded to GPU memory as torch.Tensor
    │
    ▼ model.track()  - single CUDA kernel launch         [detection/yolo.py:track()]
┌──────────────────────────────────────────────────────────────────────────┐
│  YOLOv8 INFERENCE ON GPU (TensorRT FP16 / PyTorch CUDA + cuDNN)        │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  TensorRT FP16 Engine (auto-exported, cached as .engine file)   │    │
│  │  ─────────────────────────────────────────────────────────────   │    │
│  │  • Layer fusion: Conv+BatchNorm+ReLU → single CUDA kernel       │    │
│  │  • FP16 quantization: 16-bit floats → 2× memory bandwidth      │    │
│  │  • Kernel auto-tuning: optimized for YOUR specific GPU arch     │    │
│  │  • cuDNN primitives: Winograd/FFT convolutions on CUDA cores    │    │
│  │  [detection/yolo.py:_load() → tmp.export(format="engine")]      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Forward pass: frame → 80-class predictions (all on GPU)                │
│  CUDA NMS: IoU=0.5, remove duplicate boxes (GPU parallel sort+filter)   │
│  Class filter: keep person(0) + dog(16)  - GPU tensor mask               │
│  ByteTrack association: match detections → persistent track IDs          │
│  @torch.inference_mode()  - autograd disabled, zero gradient overhead     │
│                                                                         │
│  Output: xyxy tensors + conf tensors + class tensors + ID tensors (GPU) │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼ Route by class (GPU tensor → split by cls mask)   [run_demo_cpu.py:135-148]
    ├──── cls == 16 → dogs list (GPU-detected dog ROIs)
    │     └── update ghost cache for tracker persistence
    └──── cls == 0  → persons list (GPU-detected person ROIs)
    │
    ▼ Ghost persistence (30-frame Kalman-like bbox hold) [run_demo_cpu.py:151-165]
    │ Maintains GPU-detected bboxes through brief occlusions
    │
    ├─────────────────────────┬──────────────────────────┐
    ▼                         ▼                          │
DOG PIPELINE (GPU ROIs)   PERSON PIPELINE (GPU ROIs)     │
    │                         │                          │
    ▼                         ▼                          │
BiteRiskAnalyzer          AccessController               │
[bite_detector.py:86]     [access_control.py:75]         │
    │                         │                          │
    │ Per (dog,person) pair:  │ Per-camera YAML rules:   │
    │                         │                          │
    │ ┌─ PROXIMITY (30%)      │ Load stream_id rules     │
    │ │  GPU bbox centers     │ [access_control.py:55]   │
    │ │  normalized by diag   │                          │
    │ │  [bite_detector:105]  │ Time window check        │
    │ │                       │ [access_control.py:88]   │
    │ ├─ OVERLAP (25%)        │                          │
    │ │  GPU IoU computation  │ Outside allowed hours:   │
    │ │  [bite_detector:110]  │ → AccessViolation        │
    │ │                       │   [access_control:96]    │
    │ ├─ LUNGE (25%)          │                          │
    │ │  GPU bbox area growth │                          │
    │ │  4-frame rolling      │                          │
    │ │  [bite_detector:115]  │                          │
    │ │                       │                          │
    │ ├─ SUSTAINED (20%)      │                          │
    │ │  Temporal proximity   │                          │
    │ │  frame counter        │                          │
    │ │  [bite_detector:124]  │                          │
    │ │                       │                          │
    │ ▼                       │                          │
    │ risk = weighted sum     │                          │
    │ [bite_detector:131]     │                          │
    │                         │                          │
    │ risk ≥ 0.40 → BiteEvent │                          │
    │ [bite_detector:136]     │                          │
    │                         │                          │
    ├─────────────────────────┼──────────────────────────┘
    ▼                         ▼
EVENT LOG (GPU-sourced metadata)
event_log.log_bite()      event_log.log_access()
[event_log.py:23]         [event_log.py:31]
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  CuPy RING BUFFER  - GPU-RESIDENT DETECTION STORAGE                      │
│  [analytics/ring_buffer.py]                                              │
│                                                                          │
│  • 9 preallocated CuPy GPU arrays (cp.zeros on CUDA memory)             │
│    frame, stream, track_id, x1, y1, x2, y2, conf, t_ns                  │
│  • O(1) append: direct GPU memory write at head pointer                  │
│  • Zero allocation per frame  - preallocated at pipeline start            │
│  • Circular wraparound: cp.concatenate for GPU-side unrolling            │
│  • Capacity: 54,000 rows (~30 min at 30 FPS)                            │
│  [ring_buffer.py:append_batch()  - all writes to CUDA memory]            │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼ Every 30 frames: snapshot() → cuDF GPU DataFrame
┌──────────────────────────────────────────────────────────────────────────┐
│  cuDF ROLLING-WINDOW ANALYTICS (RAPIDS GPU DataFrames)                   │
│  [analytics/window.py:compute()]                                         │
│                                                                          │
│  ALL operations execute on GPU via cuDF (RAPIDS):                        │
│                                                                          │
│  • df.groupby(["stream","frame"])["track_id"].nunique()                  │
│    → Dogs per frame per stream (GPU parallel groupby + count)            │
│                                                                          │
│  • df.groupby("track_id").agg(cx_first, cx_last, t_first, t_last, ...)  │
│    → Per-dog trajectory stats (GPU multi-column aggregation)             │
│                                                                          │
│  • df["track_id"].nunique()                                              │
│    → Total unique dogs (GPU distinct count)                              │
│                                                                          │
│  • (dx² + dy²)^0.5 / dt / diag_mean                                    │
│    → Normalized speed (GPU vectorized arithmetic)                        │
│                                                                          │
│  • .where(), .fillna(), .astype()                                       │
│    → Conditional logic + null handling (GPU)                             │
│                                                                          │
│  Result: WindowStats (single transfer: GPU → host for display)           │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  CuPy ROI COLOR HISTOGRAMS (GPU Image Processing)                       │
│  [analytics/roi_hist.py]                                                 │
│                                                                          │
│  • bgr_to_hsv_gpu(): BGR→HSV color space conversion on GPU              │
│    All math via CuPy: cp.maximum, cp.minimum, cp.where, cp.stack        │
│                                                                          │
│  • roi_histograms(): per-dog crop → 16³-bin HSV histogram               │
│    cp.bincount() on GPU  - parallel histogram binning                     │
│    Normalized histograms stay on GPU until final result                   │
│                                                                          │
│  Purpose: re-ID hints for cross-camera dog matching                      │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
RENDER ANNOTATED FRAME                                  [run_demo_cpu.py:174-189]
    ├── draw_dogs(): green GPU-detected bboxes          [draw.py:28]
    ├── draw_persons(): teal GPU-detected bboxes        [draw.py:39]
    ├── draw_bite_alerts(): red alert lines             [draw.py:50]
    ├── draw_access_violations(): orange labels         [draw.py:68]
    └── overlay_hud(): live GPU analytics stats         [draw.py:78]
    │
    ▼
VIDEO OUTPUT                                            [run_demo_cpu.py:192-197]
    ├── out/dogvision_output.mp4 (annotated video)
    ├── out/events.json (GPU-sourced bite + access events)
    └── out/summary.json (GPU pipeline metrics)
```

---

## GPU Threaded Pipeline Architecture (demo.py)

```
┌────────────────────────────────────────────────────────────────────────┐
│  THREAD 1: GPU DECODE                [pipeline/orchestrator.py]        │
│  ──────────────────────                                                │
│  NVDEC hardware video decoder → GPU memory                             │
│  CUDA Stream 1: frame upload overlaps with inference                   │
│  Bounded queue (maxlen=4) → drop-oldest for real-time                  │
│  Reconnect with exponential backoff for RTSP/webcam                    │
└───────────────────────────┬────────────────────────────────────────────┘
                            │ GPU tensor (frame on CUDA memory)
                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│  THREAD 2: GPU INFERENCE             [pipeline/orchestrator.py]        │
│  ───────────────────────                                               │
│  YOLOv8 TensorRT FP16 → detect + NMS + ByteTrack (all CUDA)           │
│  @torch.inference_mode()  - zero autograd overhead                      │
│  CuPy ring buffer append → O(1) GPU memory write                       │
│  CUDA Stream 2: inference overlaps with next decode                    │
└───────────────────────────┬────────────────────────────────────────────┘
                            │ ResultPacket (GPU detection data)
                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│  THREAD 3: GPU ANALYTICS             [pipeline/orchestrator.py]        │
│  ───────────────────────                                               │
│  Every 30 frames:                                                      │
│    CuPy ring → snapshot() → cuDF GPU DataFrame                         │
│    cuDF groupby/agg/nunique → WindowStats (all GPU)                    │
│  CuPy ROI histograms → GPU color features                              │
│  Parquet flush: PyArrow writes GPU-computed detections to disk          │
│  Draw annotations + HUD overlay                                        │
└────────────────────────────────────────────────────────────────────────┘
```

**CUDA Streams:** Thread 1 uploads frame N+1 to GPU while Thread 2 runs inference on frame N. This hides PCIe transfer latency  - GPU never idles waiting for data.

---

## Multi-Stream Grid (GPU-Batched)

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  CAM 0   │  │  CAM 1   │  │  CAM 2   │  │  CAM 3   │
│  NVDEC   │  │  NVDEC   │  │  NVDEC   │  │  NVDEC   │
│  decode  │  │  decode  │  │  decode  │  │  decode  │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │             │
     ▼             ▼             ▼             ▼
┌──────────────────────────────────────────────────────┐
│  YOLOv8 TensorRT FP16  - PER-STREAM INFERENCE         │
│  Each stream: detect(dog+person) + ByteTrack (CUDA)  │
│  Future: batch all 4 frames → single GPU forward pass│
└──────────────────────────────────────────────────────┘
     │             │             │             │
     ▼             ▼             ▼             ▼
  Dog Pipeline  Dog Pipeline  Dog Pipeline  Dog Pipeline
  Person Pipe   Person Pipe   Person Pipe   Person Pipe
  (independent per stream)
     │             │             │             │
     ▼             ▼             ▼             ▼
┌──────────────────────────────────────────────────────┐
│  GRID STITCH (640×360 per cell → 1280×720 combined)  │
│  [run_multi_stream.py:make_grid()]                    │
│  CAM labels: "CAM 0" / "CAM 1" / "CAM 2" / "CAM 3" │
└──────────────────────────────────────────────────────┘
     │
     ▼
  out/multi_stream_output.mp4
```

---

## GPU Technology Stack  - Where Each Library Acts

```
Layer                        CUDA Technology              File
────────────────────────     ────────────────────────     ─────────────────────────
Video decode                 NVDEC (GPU hardware)         utils/video.py
Frame upload                 CUDA memcpy H→D              detection/yolo.py
Model optimization           TensorRT FP16                detection/yolo.py:_load()
Inference forward pass       PyTorch CUDA                 detection/yolo.py:track()
Gradient suppression         torch.inference_mode()       detection/yolo.py (decorator)
Non-Max Suppression          CUDA parallel sort           detection/yolo.py:track()
Tracker association          ByteTrack (CUDA-fused)       detection/yolo.py:track()
Detection storage            CuPy GPU arrays              analytics/ring_buffer.py
Ring buffer append           CuPy O(1) GPU write          analytics/ring_buffer.py:append_batch()
Ring unroll                  CuPy cp.concatenate          analytics/ring_buffer.py:_ordered_slice()
DataFrame creation           cuDF (zero-copy from CuPy)   analytics/ring_buffer.py:snapshot()
Groupby aggregation          cuDF GPU groupby             analytics/window.py:compute()
Distinct count               cuDF GPU nunique             analytics/window.py:compute()
Vectorized math              cuDF GPU arithmetic          analytics/window.py:compute()
Color conversion             CuPy BGR→HSV                 analytics/roi_hist.py:bgr_to_hsv_gpu()
Histogram binning            CuPy cp.bincount             analytics/roi_hist.py:roi_histograms()
Pipeline parallelism         CUDA Streams                 pipeline/orchestrator.py
Thread overlap               CUDA async memcpy            pipeline/orchestrator.py
```

---

## Bite Risk Scoring (GPU-Sourced Data)

All input data (bboxes, track_ids) comes from GPU inference.

```
risk = 0.30 × proximity_score      ← GPU bbox center distance / GPU bbox diagonal (thresh: 2.0×)
     + 0.25 × overlap_score        ← GPU IoU between GPU-detected dog + person bboxes (thresh: 0.03)
     + 0.25 × lunge_score          ← GPU bbox area growth over 4 GPU-processed frames (>25% = lunge)
     + 0.20 × sustained_score      ← temporal accumulator over GPU-tracked frame pairs (3-frame window)

risk ≥ 0.40 → BITE RISK EVENT
    → logged with GPU-computed metadata (bbox coords, track_ids, timestamps)
    → red alert line drawn between GPU-detected dog↔person centers
```

---

## Access Control (GPU-Detected Persons)

```
GPU: YOLOv8 CUDA detects person → ByteTrack assigns persistent ID
                    │
                    ▼
            Person detected on stream_id S
                    │
                    ▼
            Rules exist for S?  ──NO──→ ALLOW
                    │
                   YES
                    │
                    ▼
            System time within allowed window?
                    │
                   YES ──→ ALLOW
                    │
                   NO  ──→ UNAUTHORIZED → AccessViolation
                           Event contains GPU-computed:
                           • person_bbox (from CUDA inference)
                           • person_track_id (from CUDA ByteTrack)
                           • frame_idx (GPU processing sequence)
```

---

## Performance: GPU Pipeline

| Stage | GPU Technology | Throughput |
|-------|---------------|------------|
| Video decode | NVDEC | 60+ FPS hardware decode |
| YOLOv8 inference | TensorRT FP16 + cuDNN | 25-45 FPS (RTX 3060) |
| ByteTrack | CUDA-fused association | <1ms per frame |
| CuPy ring append | GPU memory write | <0.1ms per detection |
| cuDF analytics (30-frame window) | GPU groupby/agg | ~5ms per window |
| CuPy ROI histogram | GPU bincount | ~2ms per frame |
| **End-to-end** | **Full CUDA pipeline** | **25+ FPS sustained** |
