"""Thin wrapper that keeps trajectories per track_id.

Ultralytics handles ByteTrack association inside `DogDetector.track()`. This
class only accumulates per-ID history for analytics (trajectory length, speed).

GPU ACCELERATION: ByteTrack runs inside Ultralytics' fused GPU call.
This module is CPU-side post-processing of tracker output — lightweight.
"""
from __future__ import annotations  # Allow modern type hints on Python 3.8+

from collections import deque           # Efficient fixed-size queue for center history
from dataclasses import dataclass, field  # Clean data classes
from typing import Deque                  # Type hint for deque


@dataclass
class Track:
    """State for a single tracked object (dog or person)."""
    track_id: int                    # ByteTrack persistent ID
    centers: Deque[tuple[float, float, int]] = field(
        default_factory=lambda: deque(maxlen=256)  # Last 256 center positions + timestamps
    )
    first_seen_ns: int = 0           # Nanosecond timestamp when first detected
    last_seen_ns: int = 0            # Nanosecond timestamp of most recent detection


class DogTracker:
    """Accumulates per-ID trajectory data from ByteTrack output.

    Does NOT perform association (ByteTrack does that). Just stores center
    history for each track_id so analytics can compute trajectory length,
    speed, and dwell time.
    """

    def __init__(self):
        # Dictionary: track_id → Track object
        self._tracks: dict[int, Track] = {}

    def update(self, detections, t_ns: int) -> None:
        """Update trajectory state with new detections from this frame.

        Args:
            detections: list of Detection objects from DogDetector.track()
            t_ns: nanosecond timestamp for this frame
        """
        for d in detections:
            if d.track_id is None:
                continue  # Skip untracked detections
            tid = d.track_id
            # Calculate bounding box center from (x1, y1, x2, y2)
            x1, y1, x2, y2 = d.bbox
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0  # Center point
            # Get or create Track for this ID
            tr = self._tracks.get(tid)
            if tr is None:
                # New track — create with first-seen timestamp
                tr = Track(track_id=tid, first_seen_ns=t_ns)
                self._tracks[tid] = tr
            # Append center position + timestamp to trajectory history
            tr.centers.append((cx, cy, t_ns))
            tr.last_seen_ns = t_ns  # Update last-seen time

    @property
    def unique_count(self) -> int:
        """Total number of distinct track IDs ever observed."""
        return len(self._tracks)

    def all_tracks(self) -> list[Track]:
        """Return all Track objects for analytics processing."""
        return list(self._tracks.values())
