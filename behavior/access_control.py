"""Person access control — time-based per-camera authorization.

Reads a YAML config mapping camera/stream IDs to allowed time windows.
Person detections outside allowed windows are flagged as unauthorized.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AccessViolation:
    timestamp_ns: int
    frame_idx: int
    stream_id: int
    person_track_id: int
    person_bbox: tuple[float, float, float, float]
    current_time: str
    allowed_windows: list[str]
    reason: str


def _parse_time(s: str) -> dt_time:
    parts = s.strip().split(":")
    return dt_time(int(parts[0]), int(parts[1]))


class AccessController:
    """Check person detections against per-camera time-based access rules."""

    def __init__(self, config_path: str | None = None):
        self._rules: dict[int, list[tuple[dt_time, dt_time]]] = {}
        self._enabled = False
        if config_path and Path(config_path).exists():
            self._load(config_path)

    def _load(self, path: str) -> None:
        cfg = yaml.safe_load(Path(path).read_text())
        if not cfg or "cameras" not in cfg:
            return
        for cam in cfg["cameras"]:
            sid = int(cam["stream_id"])
            windows = []
            for w in cam.get("allowed_hours", []):
                start = _parse_time(w["start"])
                end = _parse_time(w["end"])
                windows.append((start, end))
            self._rules[sid] = windows
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def check(
        self,
        persons: list[dict],
        stream_id: int,
        frame_idx: int,
        t_ns: int,
    ) -> list[AccessViolation]:
        if not self._enabled:
            return []
        windows = self._rules.get(stream_id)
        if windows is None:
            return []

        now = datetime.now().time()
        authorized = any(
            (s <= now <= e) if s <= e else (now >= s or now <= e)
            for s, e in windows
        )
        if authorized:
            return []

        violations = []
        window_strs = [f"{s.strftime('%H:%M')}-{e.strftime('%H:%M')}" for s, e in windows]
        for p in persons:
            pid = p.get("track_id")
            if pid is None or pid < 0:
                continue
            violations.append(AccessViolation(
                timestamp_ns=t_ns, frame_idx=frame_idx,
                stream_id=stream_id, person_track_id=pid,
                person_bbox=p["bbox"],
                current_time=now.strftime("%H:%M:%S"),
                allowed_windows=window_strs,
                reason="person_outside_allowed_hours",
            ))
        return violations
