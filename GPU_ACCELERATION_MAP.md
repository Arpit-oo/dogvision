# GPU Acceleration Map — Where Each Technology Is Used

This document maps **every GPU acceleration technology** to the **exact file, function, and line** where it's applied in the codebase.

---

## 1. PyTorch CUDA (Deep Learning Inference)

| File | Function/Line | What It Does |
|------|--------------|-------------|
| `detection/yolo.py` | `DogDetector.__init__()` → `device=0` | Loads YOLOv8 model onto GPU device 0 |
| `detection/yolo.py` | `DogDetector.track()` → `device=self.device` | Runs inference on GPU — forward pass through YOLOv8 conv layers |
| `detection/yolo.py` | `DogDetector.detect_only()` → `device=self.device` | Same GPU inference for benchmark mode |
| `detection/yolo.py` | `@torch.inference_mode()` decorator | Disables autograd gradient tracking — zero overhead for inference |
| `run_demo_cpu.py` | `model.track(device="cpu")` | CPU fallback path (still uses PyTorch, just CPU device) |
| `benchmark.py` | `_bench_gpu()` vs `_bench_cpu()` | Compares GPU device=0 vs device="cpu" for the same model |

**What PyTorch CUDA accelerates:** Matrix multiplications in convolutional layers, batch normalization, activation functions — all run on GPU CUDA cores instead of CPU.

---

## 2. cuDNN (Optimized GPU Primitives)

| File | How It's Used |
|------|--------------|
| `detection/yolo.py` | **Implicit** — PyTorch automatically uses cuDNN for convolutions when CUDA is available. No explicit code needed. |
| `environment.yml` | `cudatoolkit=12.2` installs cuDNN as a dependency |

**What cuDNN accelerates:** Winograd/FFT convolution algorithms, pooling, normalization — cuDNN selects the fastest algorithm for each layer shape on your specific GPU.

---

## 3. TensorRT FP16 (Model Optimization)

| File | Function/Line | What It Does |
|------|--------------|-------------|
| `detection/yolo.py` | `_load()` line ~60 | Checks if `.engine` file exists on disk |
| `detection/yolo.py` | `tmp.export(format="engine", half=True)` | Exports PyTorch model → TensorRT FP16 engine (one-time, cached) |
| `detection/yolo.py` | `YOLO(str(engine_path))` | Loads cached TensorRT engine for inference |
| `configs/default.yaml` | `model.trt: true` | Config flag to enable/disable TRT export |
| `configs/default.yaml` | `model.half: true` | Config flag for FP16 precision |

**What TensorRT accelerates:** Fuses adjacent layers (Conv+BN+ReLU → single kernel), quantizes FP32→FP16 (halves memory bandwidth), optimizes for specific GPU architecture. Result: 2-3× faster inference.

---

## 4. FP16 Half Precision

| File | Function/Line | What It Does |
|------|--------------|-------------|
| `detection/yolo.py` | `track(half=self.half)` | Tells Ultralytics to cast input tensor to FP16 before inference |
| `detection/yolo.py` | `detect_only(half=self.half)` | Same FP16 casting for benchmark mode |
| `configs/default.yaml` | `model.half: true` | Enable FP16 |
| `configs/cpu.yaml` | `model.half: false` | Disabled on CPU (CPU doesn't benefit from FP16) |

**What FP16 accelerates:** Halves memory bandwidth (16 bits vs 32 bits per value). GPU Tensor Cores process FP16 ~2× faster than FP32. Accuracy loss is negligible for detection.

---

## 5. CuPy (GPU Array Operations)

| File | Function/Line | What It Does |
|------|--------------|-------------|
| `analytics/ring_buffer.py` | `__init__()` → `cp.zeros(capacity, ...)` | Preallocates 9 GPU arrays (one per column) on GPU memory |
| `analytics/ring_buffer.py` | `append_batch()` → `self.frame[i] = ...` | Writes detection data directly to GPU array at O(1) cost |
| `analytics/ring_buffer.py` | `_ordered_slice()` → `cp.concatenate(...)` | Unrolls circular buffer on GPU (no CPU transfer) |
| `analytics/ring_buffer.py` | `snapshot()` → `cudf.DataFrame(data)` | Converts CuPy GPU arrays → cuDF GPU DataFrame (zero-copy) |
| `analytics/roi_hist.py` | `bgr_to_hsv_gpu()` | BGR→HSV color conversion entirely on GPU via CuPy math |
| `analytics/roi_hist.py` | `roi_histograms()` → `cp.bincount()` | Computes 16³-bin HSV histograms per dog ROI on GPU |
| `analytics/roi_hist.py` | `cp.asnumpy(out)` | Final transfer GPU→CPU only for the small histogram result |

**What CuPy accelerates:** Array allocation, indexing, concatenation, math operations — all run on GPU CUDA cores. Replaces NumPy for GPU-resident data.

---

## 6. Numba CUDA JIT (Custom GPU Kernels)

| File | Where Referenced |
|------|-----------------|
| `configs/default.yaml` | Listed in GPU stack |
| `plan.md` | Architecture spec — "Numba CUDA for bespoke GPU kernels" |
| `analytics/roi_hist.py` | ROI extraction kernels (CuPy used instead for MVP simplicity) |
| `requirements.txt` | `numba>=0.59.0` |

**What Numba accelerates:** Compiles Python functions to CUDA kernels at runtime via `@cuda.jit`. Used for custom per-pixel operations that don't map to standard CuPy/NumPy functions. In this project, CuPy handles most GPU array work; Numba is available for extensions.

---

## 7. cuDF / RAPIDS (GPU DataFrame Analytics)

| File | Function/Line | What It Does |
|------|--------------|-------------|
| `analytics/window.py` | `compute()` → `df.groupby(["stream","frame"])["track_id"].nunique()` | Dogs per frame per stream — groupby+count on GPU |
| `analytics/window.py` | `compute()` → `df.groupby("track_id").agg(...)` | Per-dog trajectory stats (first/last seen, bbox deltas) on GPU |
| `analytics/window.py` | `compute()` → `df["track_id"].nunique()` | Unique dog count — distinct count on GPU |
| `analytics/window.py` | `compute()` → arithmetic on cuDF Series | Speed calculation (dx²+dy²)^0.5 / dt — all GPU vectorized |
| `analytics/window.py` | `compute()` → `.where()`, `.fillna()` | Conditional logic and null handling on GPU |
| `analytics/ring_buffer.py` | `snapshot()` → `cudf.DataFrame(data)` | Materialize GPU arrays into cuDF DataFrame |
| `pipeline/orchestrator.py` | `_consumer()` → `self.window.compute(snap)` | Triggers cuDF analytics every 30 frames |

**What cuDF accelerates:** groupby, aggregation, joins, filtering, sorting, nunique — all operations that pandas does on CPU, cuDF does on GPU. For 54,000-row DataFrames, cuDF is 10-50× faster than pandas.

---

## 8. OpenCV + NVDEC (Video Decode)

| File | Function/Line | What It Does |
|------|--------------|-------------|
| `utils/video.py` | `cv2.VideoCapture(source)` | Decodes video frames (uses NVDEC when available via OpenCV CUDA build) |
| `utils/video.py` | `iter_frames()` | Generator yielding decoded BGR frames |
| `run_demo_cpu.py` | `cap = cv2.VideoCapture(source)` | Same decode in CPU demo |

**What NVDEC accelerates:** Hardware video decoding on NVIDIA GPU — frees CPU cores for other work. Requires OpenCV built with CUDA support (optional).

---

## 9. CUDA Streams (Pipeline Parallelism)

| File | Function/Line | What It Does |
|------|--------------|-------------|
| `pipeline/orchestrator.py` | Three-thread architecture | Decode thread, inference thread, analytics thread run concurrently |
| `pipeline/orchestrator.py` | `threading.Thread(target=self._producer)` | Decode thread feeds frames via bounded queue |
| `pipeline/orchestrator.py` | `threading.Thread(target=self._inference)` | Inference thread runs YOLO on GPU while decode reads next frame |
| `pipeline/orchestrator.py` | `threading.Thread(target=self._consumer)` | Analytics thread processes results while inference runs on next frame |

**What CUDA streams accelerate:** GPU operations from different threads can overlap — decode uploads frame N+1 to GPU while inference runs on frame N. This hides PCIe transfer latency.

---

## 10. Batch Inference (Multi-Stream)

| File | Function/Line | What It Does |
|------|--------------|-------------|
| `run_multi_stream.py` | `process_frame()` | Processes each stream's frame through YOLO independently |
| `detection/yolo.py` | `model.track(source=frame)` | Single-frame inference (batch=1 per stream) |
| `configs/default.yaml` | `model.classes: [0, 16]` | Multi-class detection in single pass (dogs + persons) |

**Future acceleration:** Batch multiple streams' frames into a single YOLO forward pass (batch=N). Currently each stream is processed sequentially; batching would fully utilize GPU parallelism.

---

## Summary: GPU vs CPU at Each Stage

```
Stage                    GPU Technology              CPU Fallback
─────────────────────    ─────────────────────       ─────────────────
Video decode             NVDEC (optional)            cv2.VideoCapture
Detection                PyTorch CUDA + TRT FP16     PyTorch CPU FP32
NMS                      PyTorch CUDA                PyTorch CPU
Tracking (ByteTrack)     Ultralytics GPU-fused       CPU association
ROI extraction           CuPy GPU kernels            NumPy (skip)
Color histograms         CuPy (bgr_to_hsv_gpu)      Skip
Detection storage        CuPy ring buffer            Python list
Analytics aggregation    cuDF groupby/agg            pandas (benchmark)
Rendering                OpenCV (CPU)                OpenCV (CPU)
Event logging            CPU (lightweight)            CPU (lightweight)
Access control           CPU (time comparison)        CPU (time comparison)
```

**Key insight:** The GPU handles the computationally expensive work (detection, tracking, analytics). CPU handles only lightweight logic (access control = time comparison, event logging = JSON append, rendering = cv2 draw calls).
