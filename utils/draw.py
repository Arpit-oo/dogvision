from __future__ import annotations

import cv2
import numpy as np

from .color import id_to_bgr


def draw_detections(frame: np.ndarray, tracks) -> np.ndarray:
    """Draw bbox + track_id + conf on a BGR frame. Returns the same frame."""
    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        tid = int(t["track_id"])
        conf = float(t["conf"])
        color = id_to_bgr(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"dog#{tid} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return frame


def overlay_hud(frame: np.ndarray, stats: dict) -> np.ndarray:
    lines = [
        f"FPS: {stats.get('fps', 0):.1f}",
        f"Dogs now: {stats.get('dogs_now', 0)}",
        f"Unique: {stats.get('unique', 0)}",
        f"Frame: {stats.get('frame', 0)}",
    ]
    y = 20
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
        y += 22
    return frame
