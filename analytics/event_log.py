"""Unified event log for bite alerts + access violations.

Stores events in memory during processing, periodically flushes to a JSON
file on disk. Provides live counts for the HUD overlay.
"""
from __future__ import annotations  # Allow modern type hints on Python 3.8+

import json       # JSON serialization for event output
import time       # Timestamps
from dataclasses import asdict  # Convert dataclass → dict for JSON
from pathlib import Path         # File path handling
from typing import Any           # Generic type hint


class EventLog:
    """Accumulates bite risk and access violation events, writes to JSON file."""

    def __init__(self, path: str = "out/events.json"):
        self._events: list[dict] = []  # In-memory event buffer (flushed periodically)
        self._path = path              # Output file path
        self._counts = {"bite_risk": 0, "access_violation": 0}  # Running totals

    def log_bite(self, event) -> None:
        """Record a bite risk event from the BiteRiskAnalyzer."""
        d = asdict(event)               # Convert BiteEvent dataclass → dict
        d["event_type"] = "bite_risk"   # Tag the event type for filtering
        d["dog_bbox"] = list(d["dog_bbox"])      # Convert tuple → list for JSON
        d["person_bbox"] = list(d["person_bbox"])
        self._events.append(d)          # Add to in-memory buffer
        self._counts["bite_risk"] += 1  # Increment running total

    def log_access(self, event) -> None:
        """Record an access violation event from the AccessController."""
        d = asdict(event)                          # Convert AccessViolation → dict
        d["event_type"] = "access_violation"      # Tag the event type
        d["person_bbox"] = list(d["person_bbox"])  # Convert tuple → list for JSON
        self._events.append(d)                     # Add to buffer
        self._counts["access_violation"] += 1      # Increment running total

    @property
    def counts(self) -> dict[str, int]:
        """Current running totals for each event type (used by HUD)."""
        return dict(self._counts)

    @property
    def recent(self, n: int = 5) -> list[dict]:
        """Return the N most recent events (for console display)."""
        return self._events[-n:]

    def flush(self) -> None:
        """Write all buffered events to the JSON file and clear the buffer.

        Append-safe: reads existing file contents first, merges, then writes.
        This prevents data loss if the process crashes between flushes.
        """
        if not self._events:
            return  # Nothing to write
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists
        existing = []
        p = Path(self._path)
        if p.exists():
            try:
                existing = json.loads(p.read_text())  # Load previously flushed events
            except (json.JSONDecodeError, ValueError):
                pass  # Corrupted file — start fresh
        existing.extend(self._events)  # Append new events to existing
        p.write_text(json.dumps(existing, indent=2, default=str))  # Write merged JSON
        self._events.clear()  # Clear in-memory buffer after successful write
