"""Unified event log for bite alerts + access violations.

Stores events in memory, prints to console via Rich, flushes to JSON.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, path: str = "out/events.json"):
        self._events: list[dict] = []
        self._path = path
        self._counts = {"bite_risk": 0, "access_violation": 0}

    def log_bite(self, event) -> None:
        d = asdict(event)
        d["event_type"] = "bite_risk"
        d["dog_bbox"] = list(d["dog_bbox"])
        d["person_bbox"] = list(d["person_bbox"])
        self._events.append(d)
        self._counts["bite_risk"] += 1

    def log_access(self, event) -> None:
        d = asdict(event)
        d["event_type"] = "access_violation"
        d["person_bbox"] = list(d["person_bbox"])
        self._events.append(d)
        self._counts["access_violation"] += 1

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    @property
    def recent(self, n: int = 5) -> list[dict]:
        return self._events[-n:]

    def flush(self) -> None:
        if not self._events:
            return
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        existing = []
        p = Path(self._path)
        if p.exists():
            try:
                existing = json.loads(p.read_text())
            except (json.JSONDecodeError, ValueError):
                pass
        existing.extend(self._events)
        p.write_text(json.dumps(existing, indent=2, default=str))
        self._events.clear()
