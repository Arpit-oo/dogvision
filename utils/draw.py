"""Annotation drawing functions for the dogvision pipeline.

Renders bounding boxes, labels, alert lines, and HUD overlay onto video frames.
Each detection class gets a distinct visual treatment:
  - Dogs:              green bounding boxes, label "dog"
  - Persons:           teal bounding boxes, label "person#ID"
  - Bite risk alerts:  red line between dog-person pair + "BITE RISK XX%" label
  - Access violations: orange box + "UNAUTHORIZED @ HH:MM:SS" label
  - HUD overlay:       semi-transparent panel with live stats
"""
from __future__ import annotations  # Allow modern type hints on Python 3.8+

import cv2        # OpenCV — drawing primitives (rectangle, putText, line)
import numpy as np  # Array operations for overlay blending

from .color import id_to_bgr  # Deterministic track-ID → color mapping

# Default colors in BGR format (OpenCV uses BGR, not RGB)
DOG_COLOR_DEFAULT = (0, 200, 0)      # Green — for dogs without a track ID
PERSON_COLOR_DEFAULT = (200, 150, 0) # Teal — for all persons
ALERT_COLOR = (0, 0, 255)            # Red — for bite risk alerts
WARNING_COLOR = (0, 140, 255)        # Orange — for access violations


def _label_box(frame, x1, y1, label, color, text_color=(0, 0, 0)):
    """Draw a filled rectangle behind text label for readability.

    The box sits just above the bounding box top edge (y1).
    """
    # Measure text dimensions to size the background box
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # Draw filled rectangle as text background (4px padding)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    # Draw text on top of the background
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)


def draw_dogs(frame: np.ndarray, dogs: list[dict]) -> np.ndarray:
    """Draw dog bounding boxes. Each dog gets a unique color based on track ID.

    Labels just say "dog" — no ID numbers, per user preference.
    Different colored boxes distinguish individual dogs visually.
    """
    for d in dogs:
        x1, y1, x2, y2 = map(int, d["bbox"])  # Convert float bbox to int pixels
        tid = int(d.get("track_id", -1))       # ByteTrack ID (-1 if untracked)
        # Use deterministic color from track ID so same dog = same color across frames
        color = id_to_bgr(tid) if tid >= 0 else DOG_COLOR_DEFAULT
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw bbox (2px thick)
        _label_box(frame, x1, y1, "dog", color)  # Label says just "dog"
    return frame


def draw_persons(frame: np.ndarray, persons: list[dict]) -> np.ndarray:
    """Draw person bounding boxes in teal with person#ID labels."""
    for p in persons:
        x1, y1, x2, y2 = map(int, p["bbox"])  # Convert float bbox to int pixels
        tid = int(p.get("track_id", -1))       # ByteTrack person ID
        conf = float(p.get("conf", 0))         # Detection confidence
        color = PERSON_COLOR_DEFAULT            # All persons use teal
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw bbox
        # Person label includes ID and confidence for identification
        label = f"person#{tid} {conf:.2f}" if tid >= 0 else f"person {conf:.2f}"
        _label_box(frame, x1, y1, label, color)
    return frame


def draw_bite_alerts(frame: np.ndarray, events) -> np.ndarray:
    """Draw bite risk alerts: red line connecting dog↔person + risk percentage.

    Visual treatment:
    - Red line from dog center to person center
    - "BITE RISK XX%" label at the midpoint of the line
    - Thick red border around the dog's bounding box
    """
    for e in events:
        # Calculate center points of dog and person bboxes
        dx = int((e.dog_bbox[0] + e.dog_bbox[2]) / 2)    # Dog center X
        dy = int((e.dog_bbox[1] + e.dog_bbox[3]) / 2)    # Dog center Y
        px = int((e.person_bbox[0] + e.person_bbox[2]) / 2)  # Person center X
        py = int((e.person_bbox[1] + e.person_bbox[3]) / 2)  # Person center Y
        # Draw red connecting line between dog and person
        cv2.line(frame, (dx, dy), (px, py), ALERT_COLOR, 3)
        # Place "BITE RISK XX%" label at midpoint of the line
        mx, my = (dx + px) // 2, (dy + py) // 2  # Midpoint coordinates
        label = f"BITE RISK {e.risk_score:.0%}"   # e.g. "BITE RISK 68%"
        _label_box(frame, mx - 40, my, label, ALERT_COLOR, (255, 255, 255))
        # Draw thick red border around the aggressive dog
        x1, y1, x2, y2 = map(int, e.dog_bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), ALERT_COLOR, 3)  # 3px thick red
    return frame


def draw_access_violations(frame: np.ndarray, violations) -> np.ndarray:
    """Draw orange box + 'UNAUTHORIZED' label for persons outside allowed hours."""
    for v in violations:
        x1, y1, x2, y2 = map(int, v.person_bbox)  # Person bbox
        # Draw thick orange border around unauthorized person
        cv2.rectangle(frame, (x1, y1), (x2, y2), WARNING_COLOR, 3)
        # Place label BELOW the bbox (y2 + 2) with current time
        label = f"UNAUTHORIZED @ {v.current_time}"  # e.g. "UNAUTHORIZED @ 23:15:02"
        _label_box(frame, x1, y2 + 2, label, WARNING_COLOR, (0, 0, 0))
    return frame


def draw_detections(frame: np.ndarray, tracks) -> np.ndarray:
    """Legacy: draw dog-only tracks. Kept for backward compatibility with old pipeline."""
    return draw_dogs(frame, tracks)


def overlay_hud(frame: np.ndarray, stats: dict) -> np.ndarray:
    """Draw a semi-transparent heads-up display with live statistics.

    Shows: FPS, dog count, person count, unique dogs, bite alerts, access violations.
    Alert counters turn red/orange when non-zero to draw attention.
    """
    h, w = frame.shape[:2]  # Frame dimensions (not used currently, available for future)
    # Stats lines to display
    lines = [
        f"FPS: {stats.get('fps', 0):.1f}",
        f"Dogs: {stats.get('dogs_now', 0)}  Persons: {stats.get('persons_now', 0)}",
        f"Unique dogs: {stats.get('unique_dogs', 0)}",
        f"Bite alerts: {stats.get('bite_alerts', 0)}",
        f"Access violations: {stats.get('access_violations', 0)}",
        f"Frame: {stats.get('frame', 0)}",
    ]
    # Draw semi-transparent black background panel behind the text
    overlay = frame.copy()  # Copy frame for alpha blending
    cv2.rectangle(overlay, (5, 2), (280, 8 + 22 * len(lines)), (0, 0, 0), -1)  # Black fill
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)  # 55% opacity blend
    # Draw each stats line
    y = 20  # Starting Y position
    for line in lines:
        color = (0, 255, 0)  # Default: green text
        # Bite alert count turns red when non-zero
        if "Bite" in line and stats.get("bite_alerts", 0) > 0:
            color = ALERT_COLOR
        # Access violation count turns orange when non-zero
        if "Access" in line and stats.get("access_violations", 0) > 0:
            color = WARNING_COLOR
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)  # Anti-aliased text
        y += 22  # Advance to next line
    return frame
