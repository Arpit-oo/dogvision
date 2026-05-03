# Dogvision — GPU-Accelerated Object Detection & Behavior Analysis

Real-time multi-class pipeline: detect dogs and persons, analyze dog bite risk,
enforce time-based access control — all GPU-first with CUDA acceleration.

## Architecture

```
┌──────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│ Decode   │───▶│  YOLOv8 + ByteTrack  │───▶│  Pipeline Router    │
│ (cv2/PyAV)│    │  (TRT FP16 on GPU)   │    │  dogs → Dog Pipeline│
└──────────┘    │  detect: dog+person   │    │  persons → Person   │
                └──────────────────────┘    └─────────────────────┘
                                                     │
                    ┌────────────────────┐    ┌──────┴───────────┐
                    │  Dog Pipeline      │    │  Person Pipeline  │
                    │  bite risk analyzer│    │  access control   │
                    │  (proximity+lunge) │    │  (time-based)     │
                    └────────────────────┘    └──────────────────┘
                                │                      │
                         ┌──────┴──────────────────────┴──────┐
                         │  Event Log + cuDF Analytics + HUD  │
                         └────────────────────────────────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| **Dog detection** | YOLOv8 + ByteTrack, COCO class 16 |
| **Person detection** | COCO class 0, tracked separately |
| **Bite risk analysis** | Proximity + overlap + lunge + sustained contact → 0-1 risk score |
| **Access control** | Per-camera time-based person authorization |
| **GPU analytics** | CuPy ring buffer → cuDF rolling window (30-frame aggregates) |
| **Event logging** | Bite alerts + access violations → `out/events.json` |
| **Annotated output** | Color-coded bboxes, alert lines, HUD overlay → `out/annotated.mp4` |
| **CPU/GPU benchmark** | Side-by-side FPS/latency comparison |

## Quick Start

```bash
# GPU path (WSL2 + RAPIDS)
conda env create -f environment.yml
conda activate dogvision
python demo.py --source your_video.mp4

# CPU path (any OS)
pip install ultralytics torch opencv-python pyyaml
python run_demo_cpu.py --source your_video.mp4
```

## Docs

- [`plan.md`](plan.md) — design spec
- [`HOW_TO_TRAIN_AND_RUN.md`](HOW_TO_TRAIN_AND_RUN.md) — setup, demo, benchmark, fine-tuning
- [`configs/access_schedule.yaml`](configs/access_schedule.yaml) — per-camera access rules

## Project Layout

```
detection/       YOLOv8 wrapper + TRT export
tracking/        per-ID trajectory accumulator
behavior/        bite risk analyzer + access control
analytics/       CuPy ring buffer, cuDF window, event log, ROI histograms
pipeline/        threaded GPU orchestrator
utils/           video IO, drawing, color
configs/         YAML configs (model, access schedule)
demo.py          GPU entry point
run_demo_cpu.py  CPU MVP entry point (full pipeline)
benchmark.py     GPU-vs-CPU benchmark
train.py         optional YOLOv8 fine-tune
```
