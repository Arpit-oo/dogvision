"""Full MVP demo: dog + person detection, bite risk analysis, access control.

Two async pipelines from a single YOLO pass:
  Dog pipeline  → ByteTrack → bite risk analysis (dog-person proximity)
  Person pipeline → access control (time-based per-camera rules)

Events logged to out/events.json. Annotated video to out/annotated.mp4.

Usage:
    python run_demo_cpu.py testdiog.mp4
    python run_demo_cpu.py testdiog.mp4 --no-display
    python run_demo_cpu.py 0  # webcam
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from analytics.event_log import EventLog
from behavior.access_control import AccessController
from behavior.bite_detector import BiteRiskAnalyzer
from utils.draw import (
    draw_access_violations,
    draw_bite_alerts,
    draw_dogs,
    draw_persons,
    overlay_hud,
)

COCO_DOG = 16
COCO_PERSON = 0

PERSIST_FRAMES = 10


def _parse_source(s: str):
    return int(s) if s.isdigit() else s


def main() -> None:
    ap = argparse.ArgumentParser("dogvision-mvp")
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--access-config", default="configs/access_schedule.yaml")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--out", default="out")
    ap.add_argument("--stream-id", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    bite_analyzer = BiteRiskAnalyzer()
    access_ctrl = AccessController(args.access_config)
    event_log = EventLog(str(out_dir / "events.json"))

    source = _parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {source}")
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[mvp] source: {source}  {total} frames @ {fps_src:.1f} fps", flush=True)
    print(f"[mvp] model: {args.model}  imgsz={args.imgsz}  conf={args.conf}", flush=True)
    print(f"[mvp] access control: {'ON' if access_ctrl.enabled else 'OFF'}", flush=True)

    writer = None
    video_path = str(out_dir / "annotated.mp4")

    frame_idx = 0
    t_start = time.time()
    unique_dog_ids: set[int] = set()
    unique_person_ids: set[int] = set()
    total_bite_alerts = 0
    total_access_violations = 0

    # Ghost persistence for dogs
    last_seen_dogs: dict[int, dict] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_ns = time.time_ns()

        # --- Stage 1: unified YOLO detect+track (dog + person) ---
        results = model.track(
            source=frame, imgsz=args.imgsz, conf=args.conf, iou=0.5,
            device="cpu", half=False,
            classes=[COCO_DOG, COCO_PERSON],
            tracker="bytetrack.yaml", persist=True, verbose=False,
            augment=True,
        )
        r = results[0]

        dogs: list[dict] = []
        persons: list[dict] = []

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            ids = (r.boxes.id.cpu().numpy().astype(int)
                   if r.boxes.id is not None
                   else np.full(len(xyxy), -1))

            for (x1, y1, x2, y2), c, cls, tid in zip(xyxy, confs, clss, ids):
                entry = {
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "conf": float(c),
                    "track_id": int(tid) if tid >= 0 else -1,
                    "cls": int(cls),
                }
                if cls == COCO_DOG:
                    dogs.append(entry)
                    if tid >= 0:
                        unique_dog_ids.add(int(tid))
                        last_seen_dogs[int(tid)] = {**entry, "age": 0}
                elif cls == COCO_PERSON:
                    persons.append(entry)
                    if tid >= 0:
                        unique_person_ids.add(int(tid))

        # Ghost dogs (persist bboxes for recently-lost tracks)
        live_dog_ids = {d["track_id"] for d in dogs if d["track_id"] >= 0}
        for tid in list(last_seen_dogs.keys()):
            if tid in live_dog_ids:
                continue
            last_seen_dogs[tid]["age"] += 1
            if last_seen_dogs[tid]["age"] > PERSIST_FRAMES:
                del last_seen_dogs[tid]
                continue
            ghost = last_seen_dogs[tid]
            dogs.append({
                "bbox": ghost["bbox"],
                "conf": ghost["conf"] * 0.5,
                "track_id": tid,
                "cls": COCO_DOG,
            })

        # --- Stage 2a: Dog Pipeline — bite risk ---
        bite_events = bite_analyzer.analyze(
            dogs, persons, frame_idx, args.stream_id, t_ns
        )
        for be in bite_events:
            event_log.log_bite(be)
            total_bite_alerts += 1

        # --- Stage 2b: Person Pipeline — access control ---
        access_violations = access_ctrl.check(
            persons, args.stream_id, frame_idx, t_ns
        )
        for av in access_violations:
            event_log.log_access(av)
            total_access_violations += 1

        # --- Render ---
        elapsed = time.time() - t_start
        fps = (frame_idx + 1) / max(elapsed, 1e-9)

        annotated = draw_dogs(frame, dogs)
        annotated = draw_persons(annotated, persons)
        annotated = draw_bite_alerts(annotated, bite_events)
        annotated = draw_access_violations(annotated, access_violations)
        annotated = overlay_hud(annotated, {
            "fps": fps,
            "dogs_now": len([d for d in dogs if d["track_id"] in live_dog_ids]),
            "persons_now": len(persons),
            "unique_dogs": len(unique_dog_ids),
            "bite_alerts": total_bite_alerts,
            "access_violations": total_access_violations,
            "frame": frame_idx,
        })

        if writer is None:
            h, w = annotated.shape[:2]
            writer = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h)
            )
        writer.write(annotated)

        if not args.no_display:
            cv2.imshow("dogvision-mvp", annotated)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        if frame_idx % 20 == 0:
            print(
                f"[mvp] frame {frame_idx}/{total}  "
                f"dogs={len(live_dog_ids)}  persons={len(persons)}  "
                f"bites={total_bite_alerts}  access={total_access_violations}  "
                f"fps={fps:.2f}",
                flush=True,
            )
        frame_idx += 1

        # Periodic event flush
        if frame_idx % 200 == 0:
            event_log.flush()

    # Finalize
    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    event_log.flush()

    dur = time.time() - t_start
    summary = {
        "frames": frame_idx,
        "unique_dogs": len(unique_dog_ids),
        "unique_persons": len(unique_person_ids),
        "bite_alerts": total_bite_alerts,
        "access_violations": total_access_violations,
        "avg_fps": round(frame_idx / max(dur, 1e-9), 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[mvp] done.", flush=True)
    print(f"  frames:             {frame_idx}", flush=True)
    print(f"  unique dogs:        {len(unique_dog_ids)}", flush=True)
    print(f"  unique persons:     {len(unique_person_ids)}", flush=True)
    print(f"  bite risk alerts:   {total_bite_alerts}", flush=True)
    print(f"  access violations:  {total_access_violations}", flush=True)
    print(f"  avg FPS:            {summary['avg_fps']}", flush=True)
    print(f"  video:              {video_path}", flush=True)
    print(f"  events:             {out_dir / 'events.json'}", flush=True)
    print(f"  summary:            {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
