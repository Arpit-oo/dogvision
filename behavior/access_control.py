"""Person access control — time-based per-camera authorization.

Reads a YAML config mapping camera/stream IDs to allowed time windows.
Person detections outside allowed windows are flagged as unauthorized.

Example config (configs/access_schedule.yaml):
    cameras:
      - stream_id: 0
        allowed_hours:
          - start: "06:00"
            end: "22:00"

If current time is 23:15 and a person is detected on stream 0,
an AccessViolation event is generated because 23:15 is outside 06:00-22:00.
"""
from __future__ import annotations  # Allow modern type hints on Python 3.8+

from dataclasses import dataclass       # Clean data classes
from datetime import datetime, time as dt_time  # Time comparison
from pathlib import Path                 # File path handling
from typing import Any                   # Generic type hint

import yaml  # YAML config parser


@dataclass
class AccessViolation:
    """Represents a single unauthorized access event."""
    timestamp_ns: int       # Nanosecond timestamp when violation detected
    frame_idx: int          # Video frame number
    stream_id: int          # Camera/stream where person was detected
    person_track_id: int    # ByteTrack ID of the unauthorized person
    person_bbox: tuple[float, float, float, float]  # Person bounding box (x1,y1,x2,y2)
    current_time: str       # Human-readable current time (e.g. "23:15:02")
    allowed_windows: list[str]  # Allowed windows for reference (e.g. ["06:00-22:00"])
    reason: str             # Always "person_outside_allowed_hours"


def _parse_time(s: str) -> dt_time:
    """Parse a time string like '06:00' or '22:30' into a datetime.time object."""
    parts = s.strip().split(":")         # Split "06:00" → ["06", "00"]
    return dt_time(int(parts[0]), int(parts[1]))  # Create time(6, 0)


class AccessController:
    """Check person detections against per-camera time-based access rules.

    Loads rules from a YAML config on init. Each camera/stream has a list
    of allowed time windows. Persons detected outside these windows are
    flagged as unauthorized.
    """

    def __init__(self, config_path: str | None = None):
        # Dictionary: stream_id → list of (start_time, end_time) tuples
        self._rules: dict[int, list[tuple[dt_time, dt_time]]] = {}
        self._enabled = False  # Disabled until config successfully loaded
        if config_path and Path(config_path).exists():
            self._load(config_path)  # Parse YAML and populate rules

    def _load(self, path: str) -> None:
        """Parse the access schedule YAML file into internal rule dict."""
        cfg = yaml.safe_load(Path(path).read_text())  # Parse YAML → dict
        if not cfg or "cameras" not in cfg:
            return  # Invalid or empty config — stay disabled
        for cam in cfg["cameras"]:                    # Iterate camera entries
            sid = int(cam["stream_id"])               # Camera/stream identifier
            windows = []
            for w in cam.get("allowed_hours", []):    # Each allowed time window
                start = _parse_time(w["start"])       # Parse start time
                end = _parse_time(w["end"])            # Parse end time
                windows.append((start, end))
            self._rules[sid] = windows                # Store rules for this stream
        self._enabled = True                           # Config loaded — enable checks

    @property
    def enabled(self) -> bool:
        """Whether access control is active (config was loaded successfully)."""
        return self._enabled

    def check(
        self,
        persons: list[dict],    # List of person detections this frame
        stream_id: int,         # Camera/stream ID to check rules for
        frame_idx: int,         # Current frame number
        t_ns: int,              # Timestamp in nanoseconds
    ) -> list[AccessViolation]:
        """Check all detected persons against access rules for this camera.

        Returns a list of AccessViolation events for any person detected
        outside the allowed time windows.
        """
        if not self._enabled:
            return []  # Access control not configured — allow all

        # Look up rules for this specific camera/stream
        windows = self._rules.get(stream_id)
        if windows is None:
            return []  # No rules defined for this stream — allow all

        # Get current wall-clock time for comparison
        now = datetime.now().time()

        # Check if current time falls within ANY allowed window
        # Handles overnight windows (e.g. 22:00-06:00) via the s <= e check
        authorized = any(
            (s <= now <= e) if s <= e else (now >= s or now <= e)
            for s, e in windows
        )
        if authorized:
            return []  # Current time is within allowed hours — no violations

        # ---- TIME IS OUTSIDE ALLOWED WINDOWS ----
        # Every person detected right now is unauthorized
        violations = []
        # Build human-readable window strings for the event record
        window_strs = [f"{s.strftime('%H:%M')}-{e.strftime('%H:%M')}" for s, e in windows]
        for p in persons:
            pid = p.get("track_id")
            if pid is None or pid < 0:
                continue  # Skip untracked detections
            violations.append(AccessViolation(
                timestamp_ns=t_ns,
                frame_idx=frame_idx,
                stream_id=stream_id,
                person_track_id=pid,
                person_bbox=p["bbox"],
                current_time=now.strftime("%H:%M:%S"),  # e.g. "23:15:02"
                allowed_windows=window_strs,             # e.g. ["06:00-22:00"]
                reason="person_outside_allowed_hours",
            ))
        return violations
