# DogVision - GPU-Accelerated Object Detection and Behavior Analysis System

**Course:** Accelerated Data Science
**Instructor:** Dr. Manisha Malik

**Submitted By:**

| Name | Roll Number |
|------|-------------|
| Priya Sharma | 102316008 |
| Vaibhav Sundriyal | 102316077 |
| Shikhar Saxena | 102316078 |
| Arpit Walia | 102316109 |

---

Real-time multi-class pipeline that detects dogs and persons, analyzes dog bite risk,
and enforces time-based access control using CUDA-based GPU acceleration.

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
| **Annotated output** | Color-coded bboxes, alert lines, HUD overlay → `out/dogvision_output.mp4` |
| **Multi-stream grid** | 2×2 CCTV grid processing 1–4 video streams simultaneously |
| **CPU/GPU benchmark** | Side-by-side FPS/latency comparison |
| **Academic report** | Auto-generated 16-section Word document for professor submission |

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

- [`plan.md`](plan.md)  - design spec
- [`HOW_TO_TRAIN_AND_RUN.md`](HOW_TO_TRAIN_AND_RUN.md)  - setup, demo, benchmark, fine-tuning
- [`PROJECT_REPORT.md`](PROJECT_REPORT.md)  - full technical report
- [`PIPELINE_WALKTHROUGH.md`](PIPELINE_WALKTHROUGH.md)  - frame-by-frame GPU pipeline flow
- [`SCRIPTS_GUIDE.md`](SCRIPTS_GUIDE.md)  - per-script usage guide
- [`MODEL_JUSTIFICATION.md`](MODEL_JUSTIFICATION.md)  - why pretrained over custom training
- [`GPU_ACCELERATION_MAP.md`](GPU_ACCELERATION_MAP.md)  - GPU tech → file:line mapping
- [`UNAUTHORIZED_ACCESS_EXPLAINED.md`](UNAUTHORIZED_ACCESS_EXPLAINED.md)  - access control deep-dive
- [`configs/access_schedule.yaml`](configs/access_schedule.yaml)  - per-camera access rules

## Project Layout

```
detection/          YOLOv8 wrapper + TRT export
tracking/           per-ID trajectory accumulator
behavior/           bite risk analyzer + access control
analytics/          CuPy ring buffer, cuDF window, event log, ROI histograms
pipeline/           threaded GPU orchestrator
utils/              video IO, drawing, color
configs/            YAML configs (model, access schedule)
demo.py             GPU entry point (threaded pipeline)
run_demo_cpu.py     CPU MVP entry point (full dual pipeline)
run_demo_gpu.py     GPU demo (TRT FP16 + CuPy + cuDF)
run_multi_stream.py multi-stream 2×2 CCTV grid
generate_report.py  academic Word report generator
benchmark.py        GPU-vs-CPU benchmark
train.py            optional YOLOv8 fine-tune
```
