# Dogvision — GPU-First Dog Detection, Tracking & Analytics

Real-time pipeline that detects dogs, tracks them across frames, and runs
GPU analytics over the detection stream.

- **Detection:** YOLOv8 (Ultralytics), TensorRT FP16, filtered to COCO class `dog`.
- **Tracking:** ByteTrack via Ultralytics' fused detect+track path.
- **Analytics:** CuPy ring buffer → cuDF rolling-window aggregates every 30 frames.
- **Output:** annotated video, Parquet detection log, live JSON analytics, cv2 HUD.

See:
- [`plan.md`](plan.md) — design spec.
- [`HOW_TO_TRAIN_AND_RUN.md`](HOW_TO_TRAIN_AND_RUN.md) — setup, demo, benchmark, training.

## Quick start
```bash
conda env create -f environment.yml
conda activate dogvision
python demo.py --source your_video.mp4
```

## Benchmark
```bash
python benchmark.py --source your_video.mp4 --max-frames 300
```

## Fine-tune (optional)
```bash
python train.py --data datasets/dogs/dogs.yaml --epochs 50
```
