"""Dog bite / aggression risk detector.

Heuristics-based approach using spatial + temporal cues:
  1. Dog-person proximity (IoU + center distance normalized by bbox size)
  2. Dog lunge detection (rapid bbox expansion = dog moving toward camera/person)
  3. Dog-person overlap (dog bbox overlapping person bbox = close contact)
  4. Sustained close contact duration (frames in proximity)

Each factor contributes to a 0-1 risk score. Threshold triggers alert.
Works on CPU — no pose model needed for MVP.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class BiteEvent:
    timestamp_ns: int
    frame_idx: int
    stream_id: int
    dog_track_id: int
    person_track_id: int
    risk_score: float
    reason: str
    dog_bbox: tuple[float, float, float, float]
    person_bbox: tuple[float, float, float, float]


@dataclass
class _ProximityState:
    frames_close: int = 0
    last_dog_areas: Deque[float] = field(default_factory=lambda: deque(maxlen=8))


def _iou(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-6)


def _center_dist(a, b) -> float:
    cx_a, cy_a = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    cx_b, cy_b = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)


def _bbox_area(b) -> float:
    return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))


def _bbox_diag(b) -> float:
    return math.sqrt((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) or 1.0


class BiteRiskAnalyzer:
    """Stateful per-stream bite risk analyzer."""

    PROXIMITY_THRESH = 1.5    # center distance / dog diagonal
    IOU_ALERT_THRESH = 0.05   # any overlap = close
    LUNGE_AREA_RATIO = 1.35   # 35% area jump in 4 frames
    SUSTAINED_FRAMES = 4      # frames of close contact before escalating
    ALERT_SCORE = 0.55        # above this = bite risk event

    def __init__(self):
        # (dog_id, person_id) -> state
        self._pairs: dict[tuple[int, int], _ProximityState] = defaultdict(
            _ProximityState
        )

    def analyze(
        self,
        dogs: list[dict],
        persons: list[dict],
        frame_idx: int,
        stream_id: int,
        t_ns: int,
    ) -> list[BiteEvent]:
        events: list[BiteEvent] = []
        active_pairs: set[tuple[int, int]] = set()

        for d in dogs:
            did = d.get("track_id")
            if did is None or did < 0:
                continue
            db = d["bbox"]
            d_diag = _bbox_diag(db)
            d_area = _bbox_area(db)

            for p in persons:
                pid = p.get("track_id")
                if pid is None or pid < 0:
                    continue
                pb = p["bbox"]
                pair = (did, pid)
                active_pairs.add(pair)
                st = self._pairs[pair]

                # Factor 1: proximity
                dist_norm = _center_dist(db, pb) / d_diag
                prox_score = max(0.0, 1.0 - dist_norm / self.PROXIMITY_THRESH)

                # Factor 2: overlap
                overlap = _iou(db, pb)
                overlap_score = min(1.0, overlap / 0.15) if overlap > 0 else 0.0

                # Factor 3: lunge (rapid area growth)
                st.last_dog_areas.append(d_area)
                lunge_score = 0.0
                if len(st.last_dog_areas) >= 4:
                    old_area = st.last_dog_areas[-4]
                    if old_area > 0:
                        ratio = d_area / old_area
                        if ratio > self.LUNGE_AREA_RATIO:
                            lunge_score = min(1.0, (ratio - 1.0) / 0.5)

                # Factor 4: sustained proximity
                if prox_score > 0.3 or overlap > self.IOU_ALERT_THRESH:
                    st.frames_close += 1
                else:
                    st.frames_close = max(0, st.frames_close - 1)
                sustained = min(1.0, st.frames_close / (self.SUSTAINED_FRAMES * 2))

                # Weighted composite
                risk = (
                    0.30 * prox_score
                    + 0.25 * overlap_score
                    + 0.25 * lunge_score
                    + 0.20 * sustained
                )

                if risk >= self.ALERT_SCORE:
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

        # Decay pairs no longer active
        stale = [k for k in self._pairs if k not in active_pairs]
        for k in stale:
            s = self._pairs[k]
            s.frames_close = max(0, s.frames_close - 2)
            if s.frames_close == 0:
                del self._pairs[k]

        return events
