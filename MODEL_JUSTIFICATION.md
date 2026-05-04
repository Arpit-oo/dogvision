# Model Training & Dataset Justification

---

## Did We Train Our Own Model?

**No.** We use **pretrained YOLOv8 weights** (trained on COCO 2017 dataset by Ultralytics) directly out of the box. We filter inference to only two COCO classes: `person` (class 0) and `dog` (class 16).

A fine-tuning script (`train.py`) is provided for optional domain-specific training, but the default pipeline runs entirely on pretrained weights with zero training.

---

## Why Pretrained Instead of Training From Scratch?

### 1. COCO Already Covers Our Target Classes

The COCO 2017 dataset contains:
- **Person:** ~262,000 annotated instances across 64,115 images
- **Dog:** ~5,500 annotated instances across 4,385 images
- **Total dataset:** 330,000+ images, 80 classes, 1.5 million object instances

Both `person` and `dog` are well-represented native COCO classes. The pretrained model has already learned robust features for detecting these objects across diverse:
- Lighting conditions (indoor, outdoor, dawn, dusk, night)
- Camera angles (eye-level, elevated CCTV, fisheye)
- Scales (close-up to far-away)
- Occlusion levels (partial, overlapping, behind objects)
- Poses (standing, sitting, running, lying down)

**Training from scratch would require 100,000+ labeled images to match COCO's coverage.** Using pretrained weights gives us this for free.

### 2. Transfer Learning Advantage

YOLOv8 pretrained on COCO has learned:
- **Low-level features** (edges, textures, gradients) in early layers
- **Mid-level features** (body parts, fur patterns, limbs) in middle layers
- **High-level features** (dog silhouettes, human poses, object shapes) in final layers

These features generalize across domains. Even on CCTV footage the model has never seen, the pretrained features activate correctly because the visual concepts (four-legged animal, upright biped) are universal.

### 3. GPU Acceleration Focus

This project's core objective is demonstrating **GPU-accelerated computing**, not achieving state-of-the-art detection accuracy. Training a custom model would:
- Consume 4-8 hours of GPU time (not contributing to the acceleration demo)
- Require dataset collection, labeling, and validation infrastructure
- Risk overfitting to a small custom dataset
- Produce marginal accuracy gains over COCO-pretrained weights

Using pretrained weights lets us focus engineering effort on the GPU pipeline: TensorRT FP16 export, CuPy ring buffers, cuDF analytics, multi-stream batching — the actual deliverables.

### 4. Reproducibility

Pretrained weights are:
- **Deterministic:** same `yolov8s.pt` produces identical results on any machine
- **Downloadable:** auto-fetched from Ultralytics GitHub releases on first run
- **Version-locked:** tied to a specific Ultralytics release for consistency
- **No data dependency:** reviewer can run the demo immediately without sourcing training data

---

## Datasets Referenced

### Used Directly (Pretrained Weights)

| Dataset | Role | Size | Classes Used |
|---------|------|------|-------------|
| **COCO 2017** | Pretrained YOLOv8 weights | 330K images, 80 classes | `person` (0), `dog` (16) |

The pretrained `yolov8s.pt` / `yolov8m.pt` weights are trained by Ultralytics on COCO 2017 train split (118,287 images) and validated on val split (5,000 images).

### Available for Optional Fine-Tuning

These datasets are referenced in the project documentation and `train.py` for users who want to improve detection on specific scenes:

| Dataset | URL | Size | Use Case |
|---------|-----|------|----------|
| **Dog Detection Dataset** | https://universe.roboflow.com/detection-dog/detection-dogs | ~2,000 images | Fine-tune for better dog recall in CCTV angles |
| **Dog Behavior / Pose Dataset** | https://universe.roboflow.com/project-lgf8z/dddog | ~1,500 images | Train dog pose/behavior classifier (future scope) |
| **Stanford Dogs** | stanford.edu | 20,580 images, 120 breeds | Breed-specific fine-tuning |
| **Oxford-IIIT Pet** | robots.ox.ac.uk | 7,349 images, 37 breeds | Pet detection fine-tuning |
| **Open Images V7** | storage.googleapis.com | 9M images | Large-scale dog class subset |

### Input Videos (Test Data)

| Video | Source | Content | Used For |
|-------|--------|---------|----------|
| `testdiog.mp4` | CCTV footage | Dog + child in front yard | Dog detection + bite risk demo |
| `Shopping Mall` | YouTube stock | Crowded mall walkway | Person detection + access control |
| `CCTV People Demo 2` | Demo footage | People walking in corridor | Person tracking demo |
| `Man Breaking Into House` | YouTube | Intruder on CCTV | Unauthorized access demo |
| `XlZXsvOuuRc.mp4` | YouTube | Outdoor scene | Multi-stream grid demo |

---

## Pretrained Model Variants Used

| Model | Parameters | Size | Speed (GPU) | Speed (CPU) | When to Use |
|-------|-----------|------|-------------|-------------|-------------|
| `yolov8n.pt` | 3.2M | 6 MB | ~45 FPS | ~15 FPS | Fastest, lowest accuracy |
| `yolov8s.pt` | 11.2M | 22 MB | ~35 FPS | ~5 FPS | **Multi-stream default** — good speed/accuracy balance |
| `yolov8m.pt` | 25.9M | 50 MB | ~25 FPS | ~2 FPS | **Single-stream default** — best recall for small/distant objects |
| `yolov8l.pt` | 43.7M | 84 MB | ~18 FPS | ~1 FPS | Highest accuracy, needs more VRAM |

We default to `yolov8m.pt` for single-stream (best recall for small dogs in CCTV) and `yolov8s.pt` for multi-stream (need 4× throughput).

---

## When Would Fine-Tuning Help?

Fine-tuning is recommended **only** when:

1. **Camera-specific angle:** your CCTV has an extreme top-down view not common in COCO
2. **Specific dog breeds:** need to detect very small dogs (Chihuahua) or unusual breeds that COCO underrepresents
3. **Night/IR footage:** infrared cameras produce greyscale images that differ from COCO's RGB training data
4. **Production deployment:** moving from demo to 24/7 system where every percentage point of recall matters

For this academic project, pretrained weights provide sufficient accuracy to demonstrate all GPU acceleration features.

---

## How to Fine-Tune (If Needed)

```bash
# 1. Prepare dataset in YOLO format
#    datasets/dogs/images/{train,val}/*.jpg
#    datasets/dogs/labels/{train,val}/*.txt  (class_id cx cy w h)
#    datasets/dogs/dogs.yaml

# 2. Train
python train.py --data datasets/dogs/dogs.yaml --epochs 50 --batch 16

# 3. Use fine-tuned weights
python run_demo_cpu.py --source video.mp4 --model runs/train/dogvision/weights/best.pt
```

See `HOW_TO_TRAIN_AND_RUN.md` Section 5 for detailed instructions.

---

## Summary

| Decision | Justification |
|----------|--------------|
| **Pretrained over custom** | COCO covers dog+person with 267K+ instances. Training from scratch would require comparable data for no accuracy gain. |
| **YOLOv8 over other architectures** | Best speed/accuracy tradeoff. Native TensorRT FP16 export. Built-in ByteTrack. Single `pip install`. |
| **COCO over domain-specific dataset** | Universal features generalize to CCTV. Fine-tuning available as optional enhancement. |
| **No training in default pipeline** | Project focus is GPU acceleration, not model training. Zero-setup demo experience. |
| **Fine-tune script provided** | `train.py` ready for users who need domain-specific improvement. |
