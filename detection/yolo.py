"""YOLOv8 dog detector with optional TensorRT FP16 engine.

Uses Ultralytics' built-in tracking (ByteTrack) via `model.track(...)` so we get
detection + tracking in one GPU-resident call. Class filtering is done with
the `classes=` arg to keep only COCO `dog` (id 16).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    conf: float
    track_id: int | None


class DogDetector:
    """Single-class (dog) YOLOv8 detector + tracker wrapper."""

    def __init__(
        self,
        weights: str = "yolov8s.pt",
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.5,
        half: bool = True,
        trt: bool = True,
        device: int | str = 0,
        dog_class_id: int = 16,
        tracker_cfg: str = "bytetrack.yaml",
    ):
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.half = half
        self.device = device
        self.dog_class_id = dog_class_id
        self.tracker_cfg = tracker_cfg

        self.model = self._load(weights, trt=trt)

    def _load(self, weights: str, trt: bool) -> YOLO:
        """Load PyTorch weights; export to TensorRT FP16 and reload if requested.

        If TRT export fails (driver/version mismatch), fall back to PyTorch FP16.
        """
        w_path = Path(weights)
        engine_path = w_path.with_suffix(".engine")

        if trt and torch.cuda.is_available():
            if not engine_path.exists():
                try:
                    tmp = YOLO(weights)
                    tmp.export(
                        format="engine",
                        imgsz=self.imgsz,
                        half=self.half,
                        device=self.device,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[yolo] TRT export failed ({e}); falling back to PyTorch FP16.")
                    return YOLO(weights)
            if engine_path.exists():
                return YOLO(str(engine_path))

        return YOLO(weights)

    @torch.inference_mode()
    def track(self, frame: np.ndarray, persist: bool = True) -> list[Detection]:
        """Run detect+track on a single BGR frame. Returns dog-only detections."""
        results = self.model.track(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            half=self.half,
            device=self.device,
            classes=[self.dog_class_id],
            tracker=self.tracker_cfg,
            persist=persist,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        xyxy = r.boxes.xyxy.detach().cpu().numpy()
        confs = r.boxes.conf.detach().cpu().numpy()
        if r.boxes.id is not None:
            ids = r.boxes.id.detach().cpu().numpy().astype(int)
        else:
            ids = np.full(len(xyxy), -1, dtype=int)

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

    @torch.inference_mode()
    def detect_only(self, frame: np.ndarray) -> list[Detection]:
        """Detection without tracking — used by the CPU/GPU benchmark."""
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            half=self.half,
            device=self.device,
            classes=[self.dog_class_id],
            verbose=False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.detach().cpu().numpy()
        confs = r.boxes.conf.detach().cpu().numpy()
        return [
            Detection(bbox=tuple(map(float, xy)), conf=float(c), track_id=None)
            for xy, c in zip(xyxy, confs)
        ]
