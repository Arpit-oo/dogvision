from __future__ import annotations

import cv2
import numpy as np

from .color import id_to_bgr

DOG_COLOR_DEFAULT = (0, 200, 0)      # green
PERSON_COLOR_DEFAULT = (200, 150, 0) # teal
ALERT_COLOR = (0, 0, 255)            # red
WARNING_COLOR = (0, 140, 255)        # orange


def _label_box(frame, x1, y1, label, color, text_color=(0, 0, 0)):
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)


def draw_dogs(frame: np.ndarray, dogs: list[dict]) -> np.ndarray:
    for d in dogs:
        x1, y1, x2, y2 = map(int, d["bbox"])
        tid = int(d.get("track_id", -1))
        conf = float(d.get("conf", 0))
        color = id_to_bgr(tid) if tid >= 0 else DOG_COLOR_DEFAULT
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"dog#{tid} {conf:.2f}" if tid >= 0 else f"dog {conf:.2f}"
        _label_box(frame, x1, y1, label, color)
    return frame


def draw_persons(frame: np.ndarray, persons: list[dict]) -> np.ndarray:
    for p in persons:
        x1, y1, x2, y2 = map(int, p["bbox"])
        tid = int(p.get("track_id", -1))
        conf = float(p.get("conf", 0))
        color = PERSON_COLOR_DEFAULT
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"person#{tid} {conf:.2f}" if tid >= 0 else f"person {conf:.2f}"
        _label_box(frame, x1, y1, label, color)
    return frame


def draw_bite_alerts(frame: np.ndarray, events) -> np.ndarray:
    for e in events:
        # Draw red line between dog and person centers
        dx = int((e.dog_bbox[0] + e.dog_bbox[2]) / 2)
        dy = int((e.dog_bbox[1] + e.dog_bbox[3]) / 2)
        px = int((e.person_bbox[0] + e.person_bbox[2]) / 2)
        py = int((e.person_bbox[1] + e.person_bbox[3]) / 2)
        cv2.line(frame, (dx, dy), (px, py), ALERT_COLOR, 3)
        # Alert label at midpoint
        mx, my = (dx + px) // 2, (dy + py) // 2
        label = f"BITE RISK {e.risk_score:.0%}"
        _label_box(frame, mx - 40, my, label, ALERT_COLOR, (255, 255, 255))
        # Thicker red box around dog
        x1, y1, x2, y2 = map(int, e.dog_bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), ALERT_COLOR, 3)
    return frame


def draw_access_violations(frame: np.ndarray, violations) -> np.ndarray:
    for v in violations:
        x1, y1, x2, y2 = map(int, v.person_bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), WARNING_COLOR, 3)
        label = f"UNAUTHORIZED @ {v.current_time}"
        _label_box(frame, x1, y2 + 2, label, WARNING_COLOR, (0, 0, 0))
    return frame


def draw_detections(frame: np.ndarray, tracks) -> np.ndarray:
    """Legacy: draw dog-only tracks. Kept for backward compat."""
    return draw_dogs(frame, tracks)


def overlay_hud(frame: np.ndarray, stats: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {stats.get('fps', 0):.1f}",
        f"Dogs: {stats.get('dogs_now', 0)}  Persons: {stats.get('persons_now', 0)}",
        f"Unique dogs: {stats.get('unique_dogs', 0)}",
        f"Bite alerts: {stats.get('bite_alerts', 0)}",
        f"Access violations: {stats.get('access_violations', 0)}",
        f"Frame: {stats.get('frame', 0)}",
    ]
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 2), (280, 8 + 22 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    y = 20
    for line in lines:
        color = (0, 255, 0)
        if "Bite" in line and stats.get("bite_alerts", 0) > 0:
            color = ALERT_COLOR
        if "Access" in line and stats.get("access_violations", 0) > 0:
            color = WARNING_COLOR
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)
        y += 22
    return frame
