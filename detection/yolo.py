"""YOLOv8 dog+person detector with optional TensorRT FP16 engine.

Uses Ultralytics' built-in tracking (ByteTrack) via `model.track(...)` so we get
detection + tracking in one GPU-resident call. Class filtering via `classes=` arg.

GPU ACCELERATION USED HERE:
  - PyTorch CUDA: model runs on GPU via device=0
  - TensorRT FP16: auto-exported .engine file for 2-3× inference speedup
  - cuDNN: optimized convolution primitives (bundled with PyTorch CUDA)
  - torch.inference_mode(): disables autograd for zero gradient overhead
  - FP16 (half precision): halves memory bandwidth, ~1.5× speedup on GPU
"""
from __future__ import annotations  # Allow modern type hints on Python 3.8+

import os                      # OS utilities
from dataclasses import dataclass  # Clean data classes
from pathlib import Path       # Cross-platform file paths

import numpy as np             # Array operations for post-processing
import torch                   # PyTorch deep learning framework
from ultralytics import YOLO   # YOLOv8 object detection model


@dataclass
class Detection:
    """Single detection result from YOLO inference."""
    bbox: tuple[float, float, float, float]  # Bounding box in (x1, y1, x2, y2) pixel coords
    conf: float                               # Detection confidence score (0.0 to 1.0)
    track_id: int | None                      # ByteTrack persistent ID (None if untracked)


class DogDetector:
    """YOLOv8 detector + ByteTrack tracker wrapper for the GPU pipeline.

    On init, attempts to export model to TensorRT FP16 for maximum inference speed.
    Falls back to PyTorch FP16 if TensorRT export fails (driver mismatch, etc).
    """

    def __init__(
        self,
        weights: str = "yolov8s.pt",     # Path to YOLOv8 pretrained weights
        imgsz: int = 640,                 # Input image size (pixels, square)
        conf: float = 0.35,               # Minimum confidence threshold
        iou: float = 0.5,                 # IoU threshold for Non-Maximum Suppression
        half: bool = True,                # Enable FP16 half-precision on GPU
        trt: bool = True,                 # Enable TensorRT engine export
        device: int | str = 0,            # GPU device index (0) or "cpu"
        dog_class_id: int = 16,           # COCO class ID for "dog"
        tracker_cfg: str = "bytetrack.yaml",  # Tracker config file (shipped with Ultralytics)
    ):
        # Store all config for use in track() and detect_only()
        self.conf = conf                   # Confidence threshold
        self.iou = iou                     # NMS IoU threshold
        self.imgsz = imgsz                 # Input resolution
        self.half = half                   # FP16 flag
        self.device = device               # GPU device or "cpu"
        self.dog_class_id = dog_class_id   # COCO class to filter
        self.tracker_cfg = tracker_cfg     # ByteTrack YAML config

        # Load model — tries TensorRT first, falls back to PyTorch
        self.model = self._load(weights, trt=trt)

    def _load(self, weights: str, trt: bool) -> YOLO:
        """Load PyTorch weights; export to TensorRT FP16 and reload if requested.

        TensorRT export converts the PyTorch model to an optimized .engine file
        that runs 2-3× faster by fusing layers, quantizing to FP16, and optimizing
        for the specific GPU architecture. The engine is cached on disk.

        If TRT export fails (driver/version mismatch), fall back to PyTorch FP16.
        """
        w_path = Path(weights)             # Path to .pt weights file
        engine_path = w_path.with_suffix(".engine")  # Corresponding .engine path

        # Only attempt TensorRT if CUDA is available
        if trt and torch.cuda.is_available():
            if not engine_path.exists():
                # Engine doesn't exist yet — export it
                try:
                    tmp = YOLO(weights)     # Load PyTorch model temporarily
                    tmp.export(             # Export to TensorRT engine
                        format="engine",    # TensorRT format
                        imgsz=self.imgsz,   # Must match inference resolution
                        half=self.half,     # FP16 quantization
                        device=self.device, # Target GPU
                    )
                except Exception as e:  # noqa: BLE001
                    # TRT export failed — fall back gracefully
                    print(f"[yolo] TRT export failed ({e}); falling back to PyTorch FP16.")
                    return YOLO(weights)    # Load PyTorch model instead
            if engine_path.exists():
                return YOLO(str(engine_path))  # Load cached TensorRT engine

        return YOLO(weights)  # No TRT available — use PyTorch model directly

    @torch.inference_mode()  # Disable autograd — no gradients needed for inference
    def track(self, frame: np.ndarray, persist: bool = True) -> list[Detection]:
        """Run detect+track on a single BGR frame. Returns filtered detections.

        The entire detect→NMS→track pipeline runs on GPU in a single call.
        `persist=True` maintains ByteTrack state across frames for consistent IDs.
        """
        results = self.model.track(
            source=frame,              # Input BGR frame from OpenCV
            imgsz=self.imgsz,          # Resize to this resolution before inference
            conf=self.conf,            # Minimum detection confidence
            iou=self.iou,              # NMS IoU threshold
            half=self.half,            # FP16 inference on GPU
            device=self.device,        # GPU device index or "cpu"
            classes=[self.dog_class_id],  # Only detect this COCO class
            tracker=self.tracker_cfg,  # ByteTrack association config
            persist=persist,           # Maintain tracker state between calls
            verbose=False,             # Suppress per-frame YOLO output
        )
        if not results:                # No results (shouldn't happen normally)
            return []
        r = results[0]                 # First (only) result for single frame
        if r.boxes is None or len(r.boxes) == 0:
            return []                  # No detections this frame

        # Transfer detection tensors from GPU to CPU for Python processing
        xyxy = r.boxes.xyxy.detach().cpu().numpy()    # Bounding boxes: [[x1,y1,x2,y2], ...]
        confs = r.boxes.conf.detach().cpu().numpy()   # Confidence scores
        if r.boxes.id is not None:
            ids = r.boxes.id.detach().cpu().numpy().astype(int)  # ByteTrack IDs
        else:
            ids = np.full(len(xyxy), -1, dtype=int)   # No IDs assigned yet

        # Build Detection objects for downstream processing
        out: list[Detection] = []
        for (x1, y1, x2, y2), c, tid in zip(xyxy, confs, ids):
            out.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    conf=float(c),
                    track_id=int(tid) if tid >= 0 else None,
                )
            )
        return out

    @torch.inference_mode()  # Disable autograd for inference
    def detect_only(self, frame: np.ndarray) -> list[Detection]:
        """Detection without tracking — used by the CPU/GPU benchmark.

        Runs model.predict() instead of model.track() to measure pure
        detection speed without ByteTrack overhead.
        """
        results = self.model.predict(
            source=frame,              # Input BGR frame
            imgsz=self.imgsz,          # Input resolution
            conf=self.conf,            # Confidence threshold
            iou=self.iou,              # NMS threshold
            half=self.half,            # FP16 on GPU
            device=self.device,        # GPU or CPU
            classes=[self.dog_class_id],  # Class filter
            verbose=False,             # Suppress output
        )
        r = results[0]                 # Single frame result
        if r.boxes is None or len(r.boxes) == 0:
            return []                  # No detections
        # Transfer to CPU and build Detection objects (no track IDs)
        xyxy = r.boxes.xyxy.detach().cpu().numpy()
        confs = r.boxes.conf.detach().cpu().numpy()
        return [
            Detection(bbox=tuple(map(float, xy)), conf=float(c), track_id=None)
            for xy, c in zip(xyxy, confs)
        ]
