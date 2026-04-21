"""Thin wrapper that keeps trajectories per track_id.

Ultralytics handles ByteTrack association inside `DogDetector.track()`. This
class only accumulates per-ID history for analytics.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class Track:
    track_id: int
    centers: Deque[tuple[float, float, int]] = field(
        default_factory=lambda: deque(maxlen=256)
    )
    first_seen_ns: int = 0
    last_seen_ns: int = 0


class DogTracker:
    def __init__(self):
        self._tracks: dict[int, Track] = {}

    def update(self, detections, t_ns: int) -> None:
        for d in detections:
            if d.track_id is None:
                continue
            tid = d.track_id
            x1, y1, x2, y2 = d.bbox
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            tr = self._tracks.get(tid)
            if tr is None:
                tr = Track(track_id=tid, first_seen_ns=t_ns)
                self._tracks[tid] = tr
            tr.centers.append((cx, cy, t_ns))
            tr.last_seen_ns = t_ns

    @property
    def unique_count(self) -> int:
        return len(self._tracks)

    def all_tracks(self) -> list[Track]:
        return list(self._tracks.values())
