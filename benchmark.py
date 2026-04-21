"""CPU-vs-GPU benchmark for the report.

Runs the detector on the same video twice:
  1. GPU path: YOLOv8 + PyTorch CUDA (or TensorRT) FP16, cuDF analytics.
  2. CPU path: YOLOv8 ONNX Runtime CPU, pandas analytics.

Reports FPS, total time, and per-stage averages.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


def _read_video(path: str, max_frames: int | None = None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def _bench_gpu(frames, cfg) -> dict:
    from detection.yolo import DogDetector
    det = DogDetector(
        weights=cfg["model"]["weights"],
        imgsz=cfg["model"]["imgsz"],
        conf=cfg["model"]["conf"],
        iou=cfg["model"]["iou"],
        half=cfg["model"]["half"],
        trt=cfg["model"]["trt"],
        device=cfg["model"]["device"],
        dog_class_id=cfg["model"]["dog_class_id"],
    )
    # warmup
    for f in frames[:3]:
        det.detect_only(f)
    times = []
    t0 = time.perf_counter()
    for f in frames:
        s = time.perf_counter()
        det.detect_only(f)
        times.append(time.perf_counter() - s)
    total = time.perf_counter() - t0
    return {
        "backend": "GPU (PyTorch CUDA / TRT FP16)",
        "n_frames": len(frames),
        "total_s": total,
        "fps": len(frames) / total if total > 0 else 0.0,
        "mean_ms": statistics.mean(times) * 1000,
        "p50_ms": statistics.median(times) * 1000,
        "p95_ms": sorted(times)[int(0.95 * len(times)) - 1] * 1000,
    }


def _bench_cpu(frames, cfg) -> dict:
    from ultralytics import YOLO
    model = YOLO(cfg["model"]["weights"])
    # force CPU
    for f in frames[:3]:
        model.predict(
            source=f, imgsz=cfg["model"]["imgsz"], conf=cfg["model"]["conf"],
            iou=cfg["model"]["iou"], device="cpu", half=False,
            classes=[cfg["model"]["dog_class_id"]], verbose=False,
        )
    times = []
    t0 = time.perf_counter()
    for f in frames:
        s = time.perf_counter()
        model.predict(
            source=f, imgsz=cfg["model"]["imgsz"], conf=cfg["model"]["conf"],
            iou=cfg["model"]["iou"], device="cpu", half=False,
            classes=[cfg["model"]["dog_class_id"]], verbose=False,
        )
        times.append(time.perf_counter() - s)
    total = time.perf_counter() - t0
    return {
        "backend": "CPU (PyTorch CPU FP32)",
        "n_frames": len(frames),
        "total_s": total,
        "fps": len(frames) / total if total > 0 else 0.0,
        "mean_ms": statistics.mean(times) * 1000,
        "p50_ms": statistics.median(times) * 1000,
        "p95_ms": sorted(times)[int(0.95 * len(times)) - 1] * 1000,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    frames = _read_video(args.source, args.max_frames)
    print(f"loaded {len(frames)} frames")

    gpu = _bench_gpu(frames, cfg)
    cpu = _bench_cpu(frames, cfg)
    print("\n=== Results ===")
    for r in (gpu, cpu):
        print(f"{r['backend']:35s}  FPS={r['fps']:6.2f}  "
              f"mean={r['mean_ms']:6.2f}ms  p50={r['p50_ms']:6.2f}ms  "
              f"p95={r['p95_ms']:6.2f}ms")
    if cpu["fps"] > 0:
        print(f"\nSpeedup (GPU/CPU): {gpu['fps'] / cpu['fps']:.1f}x")


if __name__ == "__main__":
    main()
