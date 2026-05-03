"""Dog bite / aggression risk detector.

Heuristics-based approach using spatial + temporal cues:
  1. Dog-person proximity (IoU + center distance normalized by bbox size)
  2. Dog lunge detection (rapid bbox expansion = dog moving toward camera/person)
  3. Dog-person overlap (dog bbox overlapping person bbox = close contact)
  4. Sustained close contact duration (frames in proximity)

Each factor contributes to a 0-1 risk score. Threshold triggers alert.
Works on CPU — no pose model needed for MVP.
"""
from __future__ import annotations  # Allow modern type hints on Python 3.8+

import math  # For sqrt in distance calculations
import time  # For timestamps
from collections import defaultdict, deque  # Efficient data structures
from dataclasses import dataclass, field    # Clean data classes
from typing import Deque  # Type hint for deque


@dataclass
class BiteEvent:
    """Represents a single bite risk alert emitted when risk score exceeds threshold."""
    timestamp_ns: int       # Nanosecond timestamp when event occurred
    frame_idx: int          # Video frame number
    stream_id: int          # Camera/stream identifier
    dog_track_id: int       # ByteTrack ID of the dog involved
    person_track_id: int    # ByteTrack ID of the person involved
    risk_score: float       # Composite risk score (0.0 to 1.0)
    reason: str             # Human-readable reason tags (e.g. "close_proximity+lunge_detected")
    dog_bbox: tuple[float, float, float, float]     # Dog bounding box (x1, y1, x2, y2)
    person_bbox: tuple[float, float, float, float]  # Person bounding box (x1, y1, x2, y2)


@dataclass
class _ProximityState:
    """Internal state for tracking a specific dog-person pair over time."""
    frames_close: int = 0   # How many consecutive frames this pair has been in close proximity
    last_dog_areas: Deque[float] = field(
        default_factory=lambda: deque(maxlen=8)  # Rolling window of dog bbox areas (last 8 frames)
    )


def _iou(a, b) -> float:
    """Calculate Intersection over Union between two bounding boxes.
    IoU > 0 means the dog and person bboxes physically overlap in the frame."""
    x1 = max(a[0], b[0])  # Left edge of intersection
    y1 = max(a[1], b[1])  # Top edge of intersection
    x2 = min(a[2], b[2])  # Right edge of intersection
    y2 = min(a[3], b[3])  # Bottom edge of intersection
    inter = max(0, x2 - x1) * max(0, y2 - y1)  # Intersection area (0 if no overlap)
    aa = (a[2] - a[0]) * (a[3] - a[1])  # Area of box A
    ab = (b[2] - b[0]) * (b[3] - b[1])  # Area of box B
    return inter / max(aa + ab - inter, 1e-6)  # IoU formula with epsilon to avoid div-by-zero


def _center_dist(a, b) -> float:
    """Euclidean distance between centers of two bounding boxes (in pixels)."""
    cx_a, cy_a = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2  # Center of box A
    cx_b, cy_b = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2  # Center of box B
    return math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)  # Euclidean distance


def _bbox_area(b) -> float:
    """Calculate area of a bounding box in pixels²."""
    return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))


def _bbox_diag(b) -> float:
    """Calculate diagonal length of a bounding box (used for normalization)."""
    return math.sqrt((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) or 1.0  # Avoid zero


class BiteRiskAnalyzer:
    """Stateful per-stream bite risk analyzer.

    Maintains proximity history for every dog-person pair. Each frame, computes
    a composite risk score from four factors. If score exceeds ALERT_SCORE,
    a BiteEvent is emitted.
    """

    # Tunable thresholds — adjust these to change sensitivity
    PROXIMITY_THRESH = 1.5    # Dog-person distance / dog diagonal — below this = "close"
    IOU_ALERT_THRESH = 0.05   # Any bbox overlap above this = physical contact
    LUNGE_AREA_RATIO = 1.35   # 35% bbox area growth in 4 frames = dog lunging forward
    SUSTAINED_FRAMES = 4      # Frames of proximity before "sustained" factor activates
    ALERT_SCORE = 0.55        # Composite score threshold to emit a bite risk event

    def __init__(self):
        # Dictionary mapping (dog_id, person_id) → proximity state
        # defaultdict auto-creates new _ProximityState for unseen pairs
        self._pairs: dict[tuple[int, int], _ProximityState] = defaultdict(
            _ProximityState
        )

    def analyze(
        self,
        dogs: list[dict],       # List of dog detections this frame
        persons: list[dict],    # List of person detections this frame
        frame_idx: int,         # Current frame number
        stream_id: int,         # Camera/stream ID
        t_ns: int,              # Timestamp in nanoseconds
    ) -> list[BiteEvent]:
        """Analyze all dog-person pairs and return any bite risk events."""
        events: list[BiteEvent] = []       # Events to emit this frame
        active_pairs: set[tuple[int, int]] = set()  # Pairs seen this frame (for cleanup)

        # Check every dog against every person
        for d in dogs:
            did = d.get("track_id")
            if did is None or did < 0:
                continue  # Skip untracked detections
            db = d["bbox"]           # Dog bounding box
            d_diag = _bbox_diag(db)  # Dog bbox diagonal (for normalizing distances)
            d_area = _bbox_area(db)  # Dog bbox area (for lunge detection)

            for p in persons:
                pid = p.get("track_id")
                if pid is None or pid < 0:
                    continue  # Skip untracked persons
                pb = p["bbox"]                # Person bounding box
                pair = (did, pid)             # Unique identifier for this dog-person pair
                active_pairs.add(pair)        # Mark this pair as active
                st = self._pairs[pair]        # Get/create proximity state for this pair

                # ---- FACTOR 1: PROXIMITY (30% weight) ----
                # How close is the dog to the person, relative to dog size?
                # Score 1.0 when touching, decays to 0.0 at PROXIMITY_THRESH × diagonal
                dist_norm = _center_dist(db, pb) / d_diag  # Normalize distance by dog size
                prox_score = max(0.0, 1.0 - dist_norm / self.PROXIMITY_THRESH)

                # ---- FACTOR 2: PHYSICAL OVERLAP (25% weight) ----
                # Do the dog and person bounding boxes overlap?
                # Any overlap suggests physical contact or very close approach
                overlap = _iou(db, pb)
                overlap_score = min(1.0, overlap / 0.15) if overlap > 0 else 0.0

                # ---- FACTOR 3: LUNGE DETECTION (25% weight) ----
                # Is the dog's bbox growing rapidly? (= dog approaching camera/person)
                # Compare current area to area 4 frames ago
                st.last_dog_areas.append(d_area)  # Add current area to rolling window
                lunge_score = 0.0
                if len(st.last_dog_areas) >= 4:   # Need at least 4 frames of history
                    old_area = st.last_dog_areas[-4]  # Area from 4 frames ago
                    if old_area > 0:
                        ratio = d_area / old_area  # Area growth ratio
                        if ratio > self.LUNGE_AREA_RATIO:  # >35% growth = lunging
                            lunge_score = min(1.0, (ratio - 1.0) / 0.5)

                # ---- FACTOR 4: SUSTAINED CONTACT (20% weight) ----
                # Has this pair been close for multiple consecutive frames?
                # Increments when close, decays when apart
                if prox_score > 0.3 or overlap > self.IOU_ALERT_THRESH:
                    st.frames_close += 1  # Pair is close — increment counter
                else:
                    st.frames_close = max(0, st.frames_close - 1)  # Apart — decay slowly
                sustained = min(1.0, st.frames_close / (self.SUSTAINED_FRAMES * 2))

                # ---- COMPOSITE RISK SCORE ----
                # Weighted combination of all four factors
                risk = (
                    0.30 * prox_score       # 30% — how close
                    + 0.25 * overlap_score  # 25% — physical contact
                    + 0.25 * lunge_score    # 25% — rapid approach
                    + 0.20 * sustained      # 20% — duration of proximity
                )

                # ---- EMIT EVENT IF THRESHOLD EXCEEDED ----
                if risk >= self.ALERT_SCORE:
                    # Build human-readable reason string from active factors
                    reasons = []
                    if prox_score > 0.4:
                        reasons.append("close_proximity")
                    if overlap_score > 0.3:
                        reasons.append("physical_contact")
                    if lunge_score > 0.3:
                        reasons.append("lunge_detected")
                    if sustained > 0.3:
                        reasons.append("sustained_contact")
                    events.append(BiteEvent(
                        timestamp_ns=t_ns, frame_idx=frame_idx,
                        stream_id=stream_id,
                        dog_track_id=did, person_track_id=pid,
                        risk_score=round(risk, 3),
                        reason="+".join(reasons) or "composite",
                        dog_bbox=db, person_bbox=pb,
                    ))

        # ---- CLEANUP: decay stale pairs no longer visible ----
        # Pairs not seen this frame gradually lose their proximity history
        stale = [k for k in self._pairs if k not in active_pairs]
        for k in stale:
            s = self._pairs[k]
            s.frames_close = max(0, s.frames_close - 2)  # Faster decay when fully gone
            if s.frames_close == 0:
                del self._pairs[k]  # Remove pair entirely when fully decayed

        return events
