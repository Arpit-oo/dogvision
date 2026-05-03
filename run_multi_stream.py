"""Multi-stream CCTV grid demo: 1-4 videos displayed in a 2×2 grid.

Each grid cell runs its own detection + tracking + bite risk + access control
pipeline independently. Output is a single combined video with camera ID labels.

Usage:
    # 4 videos in a 2x2 grid
    python run_multi_stream.py --sources video1.mp4 video2.mp4 video3.mp4 video4.mp4

    # 2 videos (top row filled, bottom row black)
    python run_multi_stream.py --sources cam1.mp4 cam2.mp4

    # Same video 4 times (for demo purposes)
    python run_multi_stream.py --sources test.mp4 test.mp4 test.mp4 test.mp4

    # With custom settings
    python run_multi_stream.py --sources a.mp4 b.mp4 c.mp4 d.mp4 --model yolov8s.pt --imgsz 640
"""
from __future__ import annotations  # Allow modern type hints on Python 3.8+

import argparse    # Parse command-line arguments
import json        # JSON output for summary
import time        # FPS calculation and timestamps
from pathlib import Path  # File path handling

import cv2         # OpenCV — video capture, display, writing, image stitching
import numpy as np  # Array operations for grid assembly
from ultralytics import YOLO  # YOLOv8 object detection model

# Import custom modules
from analytics.event_log import EventLog           # Unified event logger
from behavior.access_control import AccessController  # Time-based person auth
from behavior.bite_detector import BiteRiskAnalyzer    # Dog-person bite risk
from utils.draw import (                             # Annotation renderers
    draw_access_violations,
    draw_bite_alerts,
    draw_dogs,
    draw_persons,
    overlay_hud,
)

# COCO class indices
COCO_DOG = 16      # COCO class ID for "dog"
COCO_PERSON = 0    # COCO class ID for "person"

# Ghost persistence: keep last-known dog bbox for 30 frames after tracker loses it
PERSIST_FRAMES = 30

# Grid cell resolution — each video is resized to this before stitching
CELL_W = 640       # Width of each grid cell in pixels
CELL_H = 360       # Height of each grid cell in pixels


def _parse_source(s: str):
    """Convert source string to int if webcam index, else keep as path."""
    return int(s) if s.isdigit() else s


class StreamState:
    """Per-stream state: video capture + tracking + behavior analysis."""

    def __init__(self, source, stream_id: int, access_config: str):
        self.stream_id = stream_id                          # Camera ID (0-3)
        self.source = source                                 # Video source path or index
        self.cap = cv2.VideoCapture(source)                  # Open video source
        self.bite_analyzer = BiteRiskAnalyzer()              # Independent bite risk tracker
        self.access_ctrl = AccessController(access_config)   # Access control per camera
        self.unique_dogs: set[int] = set()                   # Unique dog IDs seen
        self.unique_persons: set[int] = set()                # Unique person IDs seen
        self.last_seen_dogs: dict[int, dict] = {}            # Ghost persistence cache
        self.total_bites = 0                                  # Cumulative bite alerts
        self.total_access = 0                                 # Cumulative access violations
        self.finished = False                                 # True when video ends
        self.frame_idx = 0                                    # Frame counter for this stream
        self.last_frame: np.ndarray | None = None            # Last successfully read frame

    @property
    def is_open(self) -> bool:
        """Check if video source is still open and readable."""
        return self.cap.isOpened() and not self.finished


def process_frame(
    model: YOLO,
    state: StreamState,
    frame: np.ndarray,
    event_log: EventLog,
    conf: float,
    imgsz: int,
) -> np.ndarray:
    """Run full pipeline on one frame from one stream. Returns annotated frame."""
    t_ns = time.time_ns()  # Capture timestamp

    # ---- YOLO detect + track (dog + person) ----
    results = model.track(
        source=frame,
        imgsz=imgsz,
        conf=conf,
        iou=0.5,
        device="cpu",
        half=False,
        classes=[COCO_DOG, COCO_PERSON],
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
    )
    r = results[0]

    # ---- Parse detections by class ----
    dogs: list[dict] = []
    persons: list[dict] = []

    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()       # Bounding boxes
        confs = r.boxes.conf.cpu().numpy()       # Confidence scores
        clss = r.boxes.cls.cpu().numpy().astype(int)  # Class IDs
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
                    state.unique_dogs.add(int(tid))
                    state.last_seen_dogs[int(tid)] = {**entry, "age": 0}
            elif cls == COCO_PERSON:
                persons.append(entry)
                if tid >= 0:
                    state.unique_persons.add(int(tid))

    # ---- Ghost dog persistence ----
    live_dog_ids = {d["track_id"] for d in dogs if d["track_id"] >= 0}
    for tid in list(state.last_seen_dogs.keys()):
        if tid in live_dog_ids:
            continue
        state.last_seen_dogs[tid]["age"] += 1
        if state.last_seen_dogs[tid]["age"] > PERSIST_FRAMES:
            del state.last_seen_dogs[tid]
            continue
        ghost = state.last_seen_dogs[tid]
        dogs.append({
            "bbox": ghost["bbox"], "conf": ghost["conf"] * 0.5,
            "track_id": tid, "cls": COCO_DOG,
        })

    # ---- Bite risk analysis ----
    bite_events = state.bite_analyzer.analyze(
        dogs, persons, state.frame_idx, state.stream_id, t_ns
    )
    for be in bite_events:
        event_log.log_bite(be)
        state.total_bites += 1

    # ---- Access control ----
    access_violations = state.access_ctrl.check(
        persons, state.stream_id, state.frame_idx, t_ns
    )
    for av in access_violations:
        event_log.log_access(av)
        state.total_access += 1

    # ---- Draw annotations ----
    annotated = draw_dogs(frame, dogs)
    annotated = draw_persons(annotated, persons)
    annotated = draw_bite_alerts(annotated, bite_events)
    annotated = draw_access_violations(annotated, access_violations)
    annotated = overlay_hud(annotated, {
        "fps": 0,  # FPS shown on combined grid, not per-cell
        "dogs_now": len(live_dog_ids),
        "persons_now": len(persons),
        "unique_dogs": len(state.unique_dogs),
        "bite_alerts": state.total_bites,
        "access_violations": state.total_access,
        "frame": state.frame_idx,
    })

    # ---- Camera ID label in top-right corner ----
    cam_label = f"CAM {state.stream_id}"
    (tw, th), _ = cv2.getTextSize(cam_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    h, w = annotated.shape[:2]
    cv2.rectangle(annotated, (w - tw - 12, 4), (w - 4, th + 12), (0, 0, 0), -1)
    cv2.putText(annotated, cam_label, (w - tw - 8, th + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    state.frame_idx += 1
    return annotated


def make_grid(cells: list[np.ndarray]) -> np.ndarray:
    """Arrange 1-4 frames into a 2×2 grid. Empty cells are black."""
    # Resize all cells to uniform size
    resized = []
    for c in cells:
        resized.append(cv2.resize(c, (CELL_W, CELL_H)))

    # Pad to 4 cells with black frames if fewer than 4 sources
    while len(resized) < 4:
        resized.append(np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8))

    # Stitch into 2×2 grid: [top_left, top_right] / [bottom_left, bottom_right]
    top_row = np.hstack([resized[0], resized[1]])      # Horizontal concat top
    bottom_row = np.hstack([resized[2], resized[3]])    # Horizontal concat bottom
    grid = np.vstack([top_row, bottom_row])              # Vertical concat
    return grid


def main() -> None:
    # ==================== CLI ARGUMENTS ====================
    ap = argparse.ArgumentParser("dogvision-multi")
    ap.add_argument("--sources", nargs="+", required=True,
                    help="1-4 video file paths, webcam indices, or RTSP URLs")
    ap.add_argument("--model", default="yolov8s.pt",
                    help="YOLOv8 weights (use yolov8s for speed with 4 streams)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="YOLO input size (lower = faster for multi-stream)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Detection confidence threshold")
    ap.add_argument("--access-config", default="configs/access_schedule.yaml",
                    help="Per-camera access schedule YAML")
    ap.add_argument("--no-display", action="store_true",
                    help="Disable live OpenCV window")
    ap.add_argument("--out", default="out",
                    help="Output directory")
    args = ap.parse_args()

    # Validate source count
    if len(args.sources) > 4:
        raise SystemExit("Maximum 4 streams supported (2×2 grid)")

    # ==================== INITIALIZATION ====================
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO model once — shared across all streams
    model = YOLO(args.model)

    # Create independent event log
    event_log = EventLog(str(out_dir / "events.json"))

    # Create per-stream state (each stream has its own tracker + bite analyzer)
    streams: list[StreamState] = []
    for i, src in enumerate(args.sources):
        source = _parse_source(src)
        st = StreamState(source, stream_id=i, access_config=args.access_config)
        if not st.cap.isOpened():
            print(f"[multi] WARNING: cannot open source {i}: {src}", flush=True)
            st.finished = True
        else:
            fps = st.cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(st.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[multi] CAM {i}: {src} ({total} frames @ {fps:.1f} fps)", flush=True)
        streams.append(st)

    print(f"[multi] {len(streams)} streams → 2×2 grid ({CELL_W}×{CELL_H} per cell)", flush=True)
    print(f"[multi] model: {args.model}  imgsz={args.imgsz}  conf={args.conf}", flush=True)

    # Video writer for combined grid output
    grid_w = CELL_W * 2   # 1280 pixels wide
    grid_h = CELL_H * 2   # 720 pixels tall
    video_path = str(out_dir / "multi_stream_output.mp4")
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,            # Output at 25 FPS
        (grid_w, grid_h)
    )

    # ==================== MAIN LOOP ====================
    t_start = time.time()
    global_frame = 0

    while True:
        # Check if all streams are finished
        if all(st.finished for st in streams):
            break

        cells: list[np.ndarray] = []

        for st in streams:
            if st.finished:
                # Stream ended — show last frame (frozen) or black
                if st.last_frame is not None:
                    cells.append(st.last_frame)
                else:
                    cells.append(np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8))
                continue

            # Read next frame from this stream
            ok, frame = st.cap.read()
            if not ok:
                st.finished = True
                if st.last_frame is not None:
                    cells.append(st.last_frame)
                else:
                    cells.append(np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8))
                continue

            # Process frame through full pipeline
            annotated = process_frame(model, st, frame, event_log, args.conf, args.imgsz)
            st.last_frame = annotated  # Save for frozen display after stream ends
            cells.append(annotated)

        # Assemble 2×2 grid
        grid = make_grid(cells)

        # Add global FPS counter to grid (top-center)
        elapsed = time.time() - t_start
        fps = (global_frame + 1) / max(elapsed, 1e-9)
        fps_label = f"MULTI-STREAM  FPS: {fps:.1f}  Frame: {global_frame}"
        cv2.putText(grid, fps_label, (grid_w // 2 - 180, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Write grid frame to output video
        writer.write(grid)

        # Live display
        if not args.no_display:
            cv2.imshow("dogvision-multi", grid)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        # Progress logging every 20 frames
        if global_frame % 20 == 0:
            active = sum(1 for st in streams if not st.finished)
            total_bites = sum(st.total_bites for st in streams)
            total_access = sum(st.total_access for st in streams)
            print(
                f"[multi] frame {global_frame}  active={active}/{len(streams)}  "
                f"bites={total_bites}  access={total_access}  fps={fps:.2f}",
                flush=True,
            )

        global_frame += 1

        # Periodic event flush
        if global_frame % 200 == 0:
            event_log.flush()

    # ==================== CLEANUP ====================
    for st in streams:
        st.cap.release()
    writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    event_log.flush()

    # ==================== SUMMARY ====================
    dur = time.time() - t_start
    summary = {
        "streams": len(streams),
        "frames": global_frame,
        "per_stream": [],
        "avg_fps": round(global_frame / max(dur, 1e-9), 2),
    }
    for st in streams:
        summary["per_stream"].append({
            "stream_id": st.stream_id,
            "source": str(st.source),
            "frames": st.frame_idx,
            "unique_dogs": len(st.unique_dogs),
            "unique_persons": len(st.unique_persons),
            "bite_alerts": st.total_bites,
            "access_violations": st.total_access,
        })
    (out_dir / "multi_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[multi] done.", flush=True)
    print(f"  streams:       {len(streams)}", flush=True)
    print(f"  total frames:  {global_frame}", flush=True)
    print(f"  avg FPS:       {summary['avg_fps']}", flush=True)
    for ps in summary["per_stream"]:
        print(f"  CAM {ps['stream_id']}: {ps['frames']} frames, "
              f"{ps['unique_dogs']} dogs, {ps['unique_persons']} persons, "
              f"{ps['bite_alerts']} bites, {ps['access_violations']} access", flush=True)
    print(f"  video:         {video_path}", flush=True)
    print(f"  events:        {out_dir / 'events.json'}", flush=True)
    print(f"  summary:       {out_dir / 'multi_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
