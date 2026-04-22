"""CPU demo with high-recall settings + bbox persistence for smooth output."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from utils.draw import draw_detections, overlay_hud

DOG_CLASS = 16


def main(src: str, out_path: str = "out/annotated.mp4") -> None:
    Path("out").mkdir(exist_ok=True)
    model = YOLO("yolov8m.pt")          # bigger model, better recall
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[demo] {src}: {total} frames @ {fps_src:.1f} fps", flush=True)

    writer = None
    frame_idx = 0
    t_start = time.time()
    unique_ids: set[int] = set()
    frames_with_det = 0

    # Persistence cache: track_id -> (last bbox, last conf, frames_since)
    PERSIST_FRAMES = 10
    last_seen: dict[int, dict] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.track(
            source=frame, imgsz=960, conf=0.15, iou=0.5,
            device="cpu", half=False, classes=[DOG_CLASS],
            tracker="bytetrack.yaml", persist=True, verbose=False,
            augment=True,                # test-time augmentation
        )
        r = results[0]
        current_tracks: list[dict] = []
        live_ids: set[int] = set()

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            ids = (r.boxes.id.cpu().numpy().astype(int)
                   if r.boxes.id is not None
                   else np.full(len(xyxy), -1))
            for (x1, y1, x2, y2), c, tid in zip(xyxy, confs, ids):
                bbox = (float(x1), float(y1), float(x2), float(y2))
                if tid >= 0:
                    unique_ids.add(int(tid))
                    live_ids.add(int(tid))
                    last_seen[int(tid)] = {
                        "bbox": bbox, "conf": float(c), "age": 0,
                    }
                current_tracks.append({
                    "bbox": bbox, "conf": float(c),
                    "track_id": int(tid) if tid >= 0 else -1,
                })

        # Age out / persist ghost boxes for recently-lost tracks
        for tid in list(last_seen.keys()):
            if tid in live_ids:
                continue
            last_seen[tid]["age"] += 1
            if last_seen[tid]["age"] > PERSIST_FRAMES:
                del last_seen[tid]
                continue
            ghost = last_seen[tid]
            current_tracks.append({
                "bbox": ghost["bbox"],
                "conf": ghost["conf"] * 0.5,  # faded confidence to mark as ghost
                "track_id": tid,
            })

        if current_tracks:
            frames_with_det += 1

        elapsed = time.time() - t_start
        fps = (frame_idx + 1) / max(elapsed, 1e-9)
        coverage = 100.0 * frames_with_det / (frame_idx + 1)
        hud = {
            "fps": fps,
            "dogs_now": len([t for t in current_tracks if t["track_id"] in live_ids]),
            "unique": len(unique_ids),
            "frame": frame_idx,
        }
        annotated = draw_detections(frame, current_tracks)
        annotated = overlay_hud(annotated, hud)
        cv2.putText(annotated, f"coverage {coverage:.1f}%",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 200, 255), 2, cv2.LINE_AA)

        if writer is None:
            h, w = annotated.shape[:2]
            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h)
            )
        writer.write(annotated)

        if frame_idx % 20 == 0:
            print(f"[demo] frame {frame_idx}/{total}  "
                  f"dogs={hud['dogs_now']}  unique={len(unique_ids)}  "
                  f"coverage={coverage:.1f}%  fps={fps:.2f}", flush=True)
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    dur = time.time() - t_start
    print(f"\n[demo] done. frames={frame_idx}  unique_dogs={len(unique_ids)}  "
          f"coverage={100.0*frames_with_det/frame_idx:.1f}%  "
          f"avg_fps={frame_idx / max(dur, 1e-9):.2f}", flush=True)
    print(f"[demo] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "testdiog.mp4")
