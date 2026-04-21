"""Threaded pipeline: decode → inference+track → analytics → display/write.

v1 runs a single stream. The architecture supports 2–4 streams via a
stream_id routing layer added on top of `run()`.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from analytics.ring_buffer import DetectionRing
from analytics.window import AnalyticsWindow
from detection.yolo import DogDetector
from tracking.tracker import DogTracker
from utils.draw import draw_detections, overlay_hud
from utils.video import VideoWriter, iter_frames


@dataclass
class FramePacket:
    frame: np.ndarray
    idx: int
    t_ns: int
    stream_id: int


@dataclass
class ResultPacket:
    frame: np.ndarray
    detections: list
    idx: int
    t_ns: int
    stream_id: int


class Pipeline:
    def __init__(self, cfg: dict, source, stream_id: int = 0):
        self.cfg = cfg
        self.source = source
        self.stream_id = stream_id

        mcfg = cfg["model"]
        self.detector = DogDetector(
            weights=mcfg["weights"],
            imgsz=mcfg["imgsz"],
            conf=mcfg["conf"],
            iou=mcfg["iou"],
            half=mcfg["half"],
            trt=mcfg["trt"],
            device=mcfg["device"],
            dog_class_id=mcfg["dog_class_id"],
            tracker_cfg=cfg["tracker"]["cfg"],
        )
        self.tracker = DogTracker()
        self.ring = DetectionRing(capacity=cfg["analytics"]["ring_capacity"])
        self.window = AnalyticsWindow()

        qmax = cfg["pipeline"]["queue_maxlen"]
        self._frame_q: "queue.Queue[FramePacket | None]" = queue.Queue(maxsize=qmax)
        self._result_q: "queue.Queue[ResultPacket | None]" = queue.Queue(maxsize=qmax)

        self._stop = threading.Event()
        self._stats = {"fps": 0.0, "dogs_now": 0, "unique": 0, "frame": 0}
        self._last_stats_time = time.time()
        self._frames_since = 0

        # Output artifacts
        ocfg = cfg["output"]
        Path("out").mkdir(exist_ok=True)
        self.writer = VideoWriter(ocfg["video_path"]) if ocfg["save_video"] else None
        self.display = ocfg["display"]
        self.det_path = ocfg["detections_path"]
        self.analytics_path = ocfg["analytics_path"]
        self._detection_rows: list[dict] = []

    # ---- Threads -----------------------------------------------------------

    def _producer(self):
        try:
            for frame, meta in iter_frames(
                self.source,
                stream_id=self.stream_id,
                reconnect=True,
                max_backoff_s=self.cfg["reconnect"]["max_backoff_s"],
            ):
                if self._stop.is_set():
                    break
                pkt = FramePacket(frame=frame, idx=meta.idx, t_ns=meta.t_ns,
                                  stream_id=meta.stream_id)
                try:
                    self._frame_q.put(pkt, timeout=1.0)
                except queue.Full:
                    # drop-oldest
                    try:
                        _ = self._frame_q.get_nowait()
                    except queue.Empty:
                        pass
                    self._frame_q.put(pkt)
        finally:
            self._frame_q.put(None)

    def _inference(self):
        while not self._stop.is_set():
            pkt = self._frame_q.get()
            if pkt is None:
                self._result_q.put(None)
                return
            dets = self.detector.track(pkt.frame, persist=True)
            self.tracker.update(dets, pkt.t_ns)
            self.ring.append_batch(pkt.idx, pkt.stream_id, dets, pkt.t_ns)
            # keep raw rows for parquet export
            for d in dets:
                if d.track_id is None:
                    continue
                self._detection_rows.append({
                    "frame": pkt.idx, "stream": pkt.stream_id,
                    "track_id": d.track_id,
                    "x1": d.bbox[0], "y1": d.bbox[1],
                    "x2": d.bbox[2], "y2": d.bbox[3],
                    "conf": d.conf, "t_ns": pkt.t_ns,
                })
            self._result_q.put(ResultPacket(
                frame=pkt.frame, detections=dets,
                idx=pkt.idx, t_ns=pkt.t_ns, stream_id=pkt.stream_id,
            ))

    def _consumer(self):
        win_every = self.cfg["analytics"]["window_frames"]
        flush_every = self.cfg["analytics"]["parquet_flush_every"]
        last_stats: dict = {}
        while not self._stop.is_set():
            pkt = self._result_q.get()
            if pkt is None:
                return

            # HUD stats refresh
            self._frames_since += 1
            now = time.time()
            if now - self._last_stats_time >= 1.0:
                self._stats["fps"] = self._frames_since / (now - self._last_stats_time)
                self._frames_since = 0
                self._last_stats_time = now
            self._stats["frame"] = pkt.idx
            self._stats["unique"] = self.tracker.unique_count
            self._stats["dogs_now"] = sum(
                1 for d in pkt.detections if d.track_id is not None
            )

            tracks_for_draw = [
                {"bbox": d.bbox, "track_id": d.track_id, "conf": d.conf}
                for d in pkt.detections if d.track_id is not None
            ]
            frame = draw_detections(pkt.frame, tracks_for_draw)
            frame = overlay_hud(frame, self._stats)

            if self.writer is not None:
                self.writer.write(frame)
            if self.display:
                cv2.imshow("dogvision", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    self._stop.set()
                    return

            # cuDF analytics every N frames
            if pkt.idx > 0 and pkt.idx % win_every == 0 and self.ring.size > 0:
                snap = self.ring.snapshot()
                stats = self.window.compute(snap)
                last_stats = stats.to_dict()
                Path(self.analytics_path).write_text(json.dumps(last_stats, indent=2))

            # Periodic parquet flush
            if len(self._detection_rows) >= flush_every:
                self._flush_parquet()

        # final flush
        if self._detection_rows:
            self._flush_parquet()

    def _flush_parquet(self):
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        df = pd.DataFrame(self._detection_rows)
        self._detection_rows.clear()
        path = Path(self.det_path)
        table = pa.Table.from_pandas(df)
        if path.exists():
            existing = pq.read_table(path)
            table = pa.concat_tables([existing, table])
        pq.write_table(table, path)

    # ---- Public ------------------------------------------------------------

    def run(self):
        t_prod = threading.Thread(target=self._producer, daemon=True)
        t_inf = threading.Thread(target=self._inference, daemon=True)
        t_con = threading.Thread(target=self._consumer, daemon=True)
        t_prod.start(); t_inf.start(); t_con.start()
        try:
            t_con.join()
        except KeyboardInterrupt:
            self._stop.set()
        finally:
            self._stop.set()
            if self.writer is not None:
                self.writer.release()
            if self.display:
                cv2.destroyAllWindows()
