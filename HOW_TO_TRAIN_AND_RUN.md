# How to Train and Run — Dogvision

A GPU-first dog detection, tracking, and analytics pipeline using YOLOv8 +
ByteTrack + RAPIDS cuDF. This document walks through setup, running the demo,
benchmarking, and (optionally) fine-tuning YOLOv8 on your own dog dataset.

---

## 1. Environment

RAPIDS cuDF only supports Linux. On Windows, use **WSL2** (Ubuntu 22.04).

### 1.1 Prerequisites
- NVIDIA GPU (RTX 3060+ recommended), 8GB+ VRAM
- NVIDIA driver ≥ 535
- CUDA 12.x runtime
- Miniconda/Mambaforge

### 1.2 Create the conda env (recommended)
```bash
cd /path/to/vaibhav
conda env create -f environment.yml
conda activate dogvision
```

### 1.3 Or pip-only (no cuDF — analytics stage disabled)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1.4 Verify GPU
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import cupy; print(cupy.__version__)"
python -c "import cudf; print(cudf.__version__)"
```

---

## 2. Run the Demo

### 2.1 On a video file
```bash
python demo.py --source path/to/video.mp4
```
Press **q** to quit. Outputs land in `out/`:
- `out/annotated.mp4` — bounding boxes + track IDs
- `out/detections.parquet` — per-detection log
- `out/analytics_window.json` — latest cuDF rolling-window stats

### 2.2 On your webcam
```bash
python demo.py --source 0
```

### 2.3 On an RTSP stream
```bash
python demo.py --source rtsp://user:pass@camera.local/stream
```

### 2.4 Headless / server mode
```bash
python demo.py --source video.mp4 --no-display
```

### 2.5 Disable TensorRT export (if driver mismatch)
```bash
python demo.py --source video.mp4 --no-trt
```

### 2.6 CLI flags
| Flag | Meaning |
|------|---------|
| `--source` | path, webcam index (`0`), or RTSP URL |
| `--config` | yaml config path (default `configs/default.yaml`) |
| `--weights` | override model weights (e.g. your fine-tuned `best.pt`) |
| `--no-display` | skip the cv2 window |
| `--no-trt` | skip TensorRT engine export / use |
| `--out` | output directory (default `out/`) |

---

## 3. What the Pipeline Does

```
┌──────────┐    ┌──────────────────┐    ┌──────────────────────┐
│ decode   │───▶│ YOLOv8 + ByteTrack│───▶│ CuPy ring → cuDF     │
│ (cv2/PyAV)│    │ (TRT FP16 on GPU)│    │ rolling window stats │
└──────────┘    └──────────────────┘    └──────────────────────┘
                         │
                         ▼
                 annotated frames ─▶ cv2.imshow + out/*.mp4
```

Key GPU optimizations:
- **YOLOv8 exported to TensorRT FP16** on first run (engine cached next to the `.pt`).
- **ByteTrack** runs inside Ultralytics' fused detect-and-track path; no extra CPU hop.
- **CuPy ring buffer** stores detection rows in preallocated GPU columns — O(1) append,
  no per-frame cuDF concat.
- **cuDF rolling window** recomputes aggregates every 30 frames (≈1 s at 30 FPS).
- **Three-thread pipeline** (decode / inference / analytics+display) with bounded
  queues so decode overlaps with inference.

---

## 4. Benchmark GPU vs CPU

```bash
python benchmark.py --source path/to/video.mp4 --max-frames 300
```
Outputs FPS / p50 / p95 for GPU (PyTorch CUDA or TRT FP16) vs CPU
(PyTorch CPU FP32) paths, plus a speedup multiplier. Expect 10×–30× on an
RTX 3060 at 640px input.

---

## 5. (Optional) Fine-Tune YOLOv8 on Dogs

You do **not** need to train to use the pipeline — COCO-pretrained weights
already detect `dog` (class 16). Fine-tune only if you want improved recall
on your specific scenes or camera angles.

### 5.1 Pick a dataset
Public options you can use directly or subset:
- **Stanford Dogs** — 120 breeds, 20k images (bbox labels available).
- **Oxford-IIIT Pet** — 37 breeds incl. dogs, bbox labels.
- **Open Images V7** — filter `Dog` class.
- **Your own recordings** — label with [Roboflow](https://roboflow.com) or CVAT.

### 5.2 Convert to YOLO format
Ultralytics expects:
```
datasets/dogs/
├── images/
│   ├── train/*.jpg
│   └── val/*.jpg
├── labels/
│   ├── train/*.txt        # each line: "0 cx cy w h" (normalized, class 0 = dog)
│   └── val/*.txt
└── dogs.yaml
```
`dogs.yaml`:
```yaml
path: datasets/dogs
train: images/train
val: images/val
names:
  0: dog
```

### 5.3 Train
```bash
python train.py --data datasets/dogs/dogs.yaml --epochs 50 --batch 16
```
GPU memory planner: start `--batch 16` on 12GB, drop to 8 on 8GB. The run
lands in `runs/train/dogvision/`.

### 5.4 Use the new weights
```bash
python demo.py --source video.mp4 \
    --weights runs/train/dogvision/weights/best.pt
```
Because the fine-tuned model has a single class (`dog` at index 0), also set
`model.dog_class_id: 0` in `configs/default.yaml` before running.

### 5.5 Export to TensorRT (optional, speeds up inference)
Ultralytics auto-exports on first `DogDetector` load when `trt: true`. To
export manually:
```bash
yolo export model=runs/train/dogvision/weights/best.pt format=engine half=True imgsz=640 device=0
```

---

## 6. Configuration (configs/default.yaml)

| Section | Key | Notes |
|---------|-----|-------|
| model | weights, imgsz, conf, iou, half, trt, device, dog_class_id | Adjust `dog_class_id` to `0` after fine-tune |
| tracker | cfg | `bytetrack.yaml` (shipped with Ultralytics) |
| pipeline | queue_maxlen, drop_policy | Keep `oldest` for real-time |
| analytics | window_frames, ring_capacity, hist_bins | Lower `window_frames` for faster dashboard updates |
| output | display, save_video, paths | Toggle file vs live outputs |
| reconnect | max_backoff_s | For webcam/RTSP stall recovery |

---

## 7. Troubleshooting

**`TensorRT export failed`** — proceed with PyTorch FP16 (the fallback runs
automatically). Or re-install a matching TensorRT version for your CUDA.

**`cuDF not available`** — you're probably in the pip-only env. Install via
`conda env create -f environment.yml` instead.

**Low FPS on webcam** — your camera driver may cap capture FPS. Check with
`v4l2-ctl --list-formats-ext` (Linux) or try `--source 0` vs `1`.

**Tracker IDs jump around** — occlusion will always reassign IDs with
ByteTrack. For heavier re-ID, swap `tracker.cfg: botsort.yaml` in the config.

**GPU OOM on long videos** — lower `analytics.ring_capacity` in the config
or reduce `imgsz` to 512.

---

## 8. Project Layout

```
vaibhav/
├── detection/       YOLOv8 wrapper + TRT export
├── tracking/        per-ID trajectory accumulator
├── analytics/       CuPy ring buffer, cuDF window ops, ROI histograms
├── pipeline/        threaded orchestrator
├── utils/           video IO, drawing, deterministic color
├── configs/         yaml configs
├── demo.py          main entry point
├── benchmark.py     GPU-vs-CPU benchmark
├── train.py         optional YOLOv8 fine-tune
├── environment.yml  conda env (RAPIDS + PyTorch + Ultralytics)
├── requirements.txt pip fallback
├── plan.md          design spec
└── HOW_TO_TRAIN_AND_RUN.md (this file)
```
