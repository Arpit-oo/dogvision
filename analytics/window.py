"""cuDF rolling-window aggregates over the DetectionRing snapshot."""
from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    import cudf
except ImportError:  # pragma: no cover
    cudf = None  # type: ignore[assignment]


@dataclass
class WindowStats:
    n_rows: int
    unique_dogs: int
    dogs_now: int                     # unique track_ids in the latest frame
    avg_dogs_per_frame: float
    peak_dogs_per_frame: int
    dogs_per_minute: float
    avg_norm_speed: float             # pixels/sec / bbox diagonal, mean across tracks

    def to_dict(self) -> dict:
        return asdict(self)


class AnalyticsWindow:
    def compute(self, snapshot) -> WindowStats:
        if cudf is None:
            raise RuntimeError("cuDF not available — install RAPIDS.")
        df = snapshot.df
        if len(df) == 0:
            return WindowStats(0, 0, 0, 0.0, 0, 0.0, 0.0)

        # Per-frame counts.
        per_frame = (
            df.groupby(["stream", "frame"])["track_id"].nunique().reset_index()
        )
        avg_per_frame = float(per_frame["track_id"].mean())
        peak = int(per_frame["track_id"].max())

        # Dogs now: the latest frame per stream — sum unique ids across streams.
        latest_frame = df["frame"].max()
        dogs_now = int(
            df[df["frame"] == latest_frame]["track_id"].nunique()
        )

        unique = int(df["track_id"].nunique())

        # Dogs per minute.
        t_min = float(df["t_ns"].min())
        t_max = float(df["t_ns"].max())
        dt_min = max((t_max - t_min) / 1e9 / 60.0, 1e-9)
        dogs_per_minute = unique / dt_min

        # Normalized speed per track_id: total path / duration / mean diag.
        df = df.assign(
            cx=(df["x1"] + df["x2"]) / 2.0,
            cy=(df["y1"] + df["y2"]) / 2.0,
            diag=((df["x2"] - df["x1"]) ** 2 + (df["y2"] - df["y1"]) ** 2) ** 0.5,
        )
        # per-track path length approx via first/last center + count
        agg = (
            df.groupby("track_id")
            .agg(
                cx_first=("cx", "min"),   # proxy — actual order requires windowing
                cx_last=("cx", "max"),
                cy_first=("cy", "min"),
                cy_last=("cy", "max"),
                t_first=("t_ns", "min"),
                t_last=("t_ns", "max"),
                diag_mean=("diag", "mean"),
            )
            .reset_index()
        )
        dx = agg["cx_last"] - agg["cx_first"]
        dy = agg["cy_last"] - agg["cy_first"]
        dt_s = (agg["t_last"] - agg["t_first"]).astype("float64") / 1e9
        dist = (dx * dx + dy * dy) ** 0.5
        speed = dist / dt_s.where(dt_s > 0, 1e9)
        norm_speed = speed / agg["diag_mean"].where(agg["diag_mean"] > 0, 1.0)
        avg_norm_speed = float(norm_speed.fillna(0).mean())

        return WindowStats(
            n_rows=int(len(df)),
            unique_dogs=unique,
            dogs_now=dogs_now,
            avg_dogs_per_frame=avg_per_frame,
            peak_dogs_per_frame=peak,
            dogs_per_minute=dogs_per_minute,
            avg_norm_speed=avg_norm_speed,
        )
