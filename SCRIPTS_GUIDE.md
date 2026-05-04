# Scripts Guide — What Each Script Does

---

## 1. `run_demo_cpu.py` — Single-Stream Full Pipeline (CPU)

**Purpose:** Main demo script. Processes one video through the complete dual pipeline on CPU.

**What it does:**
1. Opens a video file, webcam, or RTSP stream
2. Runs YOLOv8 detect+track on every frame (detects dogs + persons simultaneously)
3. Routes detections by class into two pipelines:
   - **Dog Pipeline:** runs `BiteRiskAnalyzer` — scores every dog-person pair for bite risk using proximity, overlap, lunge detection, and sustained contact duration
   - **Person Pipeline:** runs `AccessController` — checks each detected person against per-camera time-based access rules from YAML config
4. Draws annotated frame: green boxes (dogs), teal boxes (persons), red alert lines (bite risk), orange labels (unauthorized access), semi-transparent HUD
5. Writes output video + events JSON + summary JSON
6. Ghost persistence: keeps showing last-known dog bbox for 30 frames after tracker loses it

**How to run:**
```bash
python run_demo_cpu.py --source video.mp4 --no-display
python run_demo_cpu.py --source 0                        # webcam
python run_demo_cpu.py --source video.mp4 --model yolov8m.pt --imgsz 960 --conf 0.15
```

**Outputs:**
- `out/dogvision_output.mp4` — annotated video
- `out/events.json` — bite risk + access violation events
- `out/summary.json` — run stats (frames, unique counts, alert counts, FPS)

---

## 2. `run_multi_stream.py` — Multi-Stream 2×2 CCTV Grid

**Purpose:** Process 1–4 videos simultaneously, displayed in a 2×2 CCTV-style grid.

**What it does:**
1. Opens up to 4 video sources (files, webcams, RTSP)
2. Each stream gets its own independent:
   - ByteTrack tracker (separate ID space per camera)
   - BiteRiskAnalyzer (per-camera dog-person proximity tracking)
   - AccessController (per-camera time rules from same YAML config)
3. Reads one frame from each stream per iteration
4. Runs YOLOv8 detect+track on each frame independently
5. Annotates each frame with detections + alerts
6. Adds "CAM 0" / "CAM 1" / "CAM 2" / "CAM 3" label to each cell
7. Stitches 4 annotated frames into a 2×2 grid (1280×720)
8. Adds global FPS counter to top center of grid
9. When a stream ends, its cell freezes on the last frame
10. Writes combined grid video + per-stream summary

**How to run:**
```bash
# 4 different videos
python run_multi_stream.py --sources vid1.mp4 vid2.mp4 vid3.mp4 vid4.mp4

# 2 videos (bottom row stays black)
python run_multi_stream.py --sources cam1.mp4 cam2.mp4

# Same video 4 times
python run_multi_stream.py --sources test.mp4 test.mp4 test.mp4 test.mp4

# With settings
python run_multi_stream.py --sources a.mp4 b.mp4 c.mp4 d.mp4 --model yolov8s.pt --imgsz 640 --no-display
```

**Outputs:**
- `out/multi_stream_output.mp4` — 2×2 grid video (1280×720)
- `out/events.json` — events from ALL streams combined
- `out/multi_summary.json` — per-stream breakdown (dogs, persons, bites, access per camera)

---

## 3. `demo.py` — GPU Pipeline Entry Point

**Purpose:** Production GPU entry point using the threaded pipeline orchestrator.

**What it does:**
1. Loads YAML config (`configs/default.yaml`)
2. Constructs the threaded `Pipeline` orchestrator:
   - Thread 1 (Producer): decodes video frames, uploads to GPU
   - Thread 2 (Inference): runs YOLOv8 TensorRT FP16 + ByteTrack on GPU
   - Thread 3 (Consumer): cuDF analytics + display + file writing
3. Bounded queues between threads with drop-oldest policy for real-time
4. CuPy ring buffer stores detections on GPU; cuDF analytics every 30 frames
5. Auto-exports YOLOv8 to TensorRT FP16 engine on first run (cached)

**How to run:**
```bash
python demo.py --source video.mp4                # GPU with TensorRT
python demo.py --source 0                        # webcam
python demo.py --source video.mp4 --no-trt       # GPU without TensorRT
python demo.py --source video.mp4 --no-display   # headless
```

**Requires:** NVIDIA GPU + CUDA + WSL2 + RAPIDS conda env

---

## 4. `benchmark.py` — GPU vs CPU Performance Comparison

**Purpose:** Measures and compares inference speed between GPU and CPU paths.

**What it does:**
1. Loads N frames from a video into memory (default 300)
2. **GPU benchmark:** runs YOLOv8 with PyTorch CUDA / TensorRT FP16, measures per-frame latency
3. **CPU benchmark:** runs same YOLOv8 with `device="cpu"`, FP32, measures per-frame latency
4. Includes 3-frame warmup for each path (excludes from timing)
5. Reports: FPS, mean latency, p50 (median), p95 latency
6. Calculates speedup multiplier (GPU FPS / CPU FPS)

**How to run:**
```bash
python benchmark.py --source video.mp4 --max-frames 300
```

**Output:** Console table with GPU vs CPU comparison. Expect 10–30× speedup on RTX 3060+.

---

## 5. `train.py` — Optional YOLOv8 Fine-Tuning

**Purpose:** Fine-tune YOLOv8 on a custom dog dataset for better recall on specific scenes.

**What it does:**
1. Loads pretrained YOLOv8 weights (default yolov8s.pt)
2. Trains on a user-provided dataset in YOLO format (images/ + labels/ + dataset.yaml)
3. Uses mixed precision (AMP) for faster training on GPU
4. Saves best weights to `runs/train/dogvision/weights/best.pt`

**How to run:**
```bash
# Prepare dataset first (see HOW_TO_TRAIN_AND_RUN.md section 5)
python train.py --data datasets/dogs/dogs.yaml --epochs 50

# Then use fine-tuned weights
python run_demo_cpu.py --source video.mp4 --model runs/train/dogvision/weights/best.pt
```

**When to use:** Only if COCO-pretrained model misses dogs in your specific camera angle/environment. Not needed for the default demo.

---

## Quick Reference Table

| Script | GPU Needed? | Input | Output | Speed (CPU) |
|--------|------------|-------|--------|-------------|
| `run_demo_cpu.py` | No | 1 video | `dogvision_output.mp4` + events + summary | ~2-5 FPS |
| `run_multi_stream.py` | No | 1-4 videos | `multi_stream_output.mp4` + events + summary | ~1-3 FPS |
| `demo.py` | Yes (CUDA) | 1 video | `annotated.mp4` + parquet + analytics | 25+ FPS |
| `benchmark.py` | Yes (CUDA) | 1 video | Console table (GPU vs CPU) | N/A |
| `train.py` | Yes (CUDA) | Dataset YAML | `best.pt` weights | N/A |

---

## CLI Flags Reference (all scripts)

| Flag | Scripts | Default | Description |
|------|---------|---------|-------------|
| `--source` | demo, run_demo_cpu | required | Single video path, webcam index, or RTSP URL |
| `--sources` | run_multi_stream | required | 1-4 video paths (space-separated) |
| `--model` | all demo scripts | `yolov8m.pt` / `yolov8s.pt` | YOLOv8 weights (n=fastest, x=most accurate) |
| `--imgsz` | all demo scripts | 640-960 | YOLO input resolution |
| `--conf` | all demo scripts | 0.25 | Detection confidence threshold |
| `--access-config` | demo scripts | `configs/access_schedule.yaml` | Per-camera access rules |
| `--no-display` | all demo scripts | off | Disable OpenCV live window |
| `--out` | all demo scripts | `out` | Output directory |
| `--stream-id` | run_demo_cpu | 0 | Camera ID for access control |
| `--no-trt` | demo.py | off | Skip TensorRT export |
| `--weights` | demo.py | from config | Override model weights |
| `--max-frames` | benchmark.py | 300 | Frames to benchmark |
| `--data` | train.py | required | Dataset YAML path |
| `--epochs` | train.py | 50 | Training epochs |
| `--batch` | train.py | 16 | Training batch size |
