"""Full MVP demo: dog + person detection, bite risk analysis, access control.

Two async pipelines from a single YOLO pass:
  Dog pipeline  → ByteTrack → bite risk analysis (dog-person proximity)
  Person pipeline → access control (time-based per-camera rules)

Events logged to out/events.json. Annotated video to out/dogvision_output.mp4.

Usage:
    python run_demo_cpu.py --source testdiog.mp4
    python run_demo_cpu.py --source testdiog.mp4 --no-display
    python run_demo_cpu.py --source 0  # webcam
"""
from __future__ import annotations  # Allow modern type hints on Python 3.8+

import argparse   # Parse command-line arguments
import json       # Read/write JSON files
import sys        # System utilities
import time       # Timestamps and FPS calculation
from pathlib import Path  # Cross-platform file path handling

import cv2        # OpenCV — video capture, display, writing
import numpy as np  # Array operations for detection post-processing
from ultralytics import YOLO  # YOLOv8 object detection model

# Import our custom modules for behavior analysis and rendering
from analytics.event_log import EventLog          # Unified event logger (bite + access)
from behavior.access_control import AccessController  # Time-based person access rules
from behavior.bite_detector import BiteRiskAnalyzer   # Dog-person bite risk scoring
from utils.draw import (                            # Annotation drawing functions
    draw_access_violations,  # Orange boxes for unauthorized persons
    draw_bite_alerts,        # Red lines between aggressive dogs and persons
    draw_dogs,               # Green bounding boxes for dogs
    draw_persons,            # Teal bounding boxes for persons
    overlay_hud,             # Semi-transparent heads-up display with live stats
)

# COCO dataset class indices — these are fixed by the pretrained model
COCO_DOG = 16     # COCO class ID for "dog"
COCO_PERSON = 0   # COCO class ID for "person"

# How many frames to keep showing a dog's last-known bbox after tracker loses it
# At 25 FPS, 30 frames = 1.2 seconds of ghost persistence through occlusions
PERSIST_FRAMES = 30


def _parse_source(s: str):
    """Convert source string to int if it's a webcam index (e.g. '0'), else keep as path."""
    return int(s) if s.isdigit() else s


def main() -> None:
    # ==================== CLI ARGUMENT PARSING ====================
    ap = argparse.ArgumentParser("dogvision-mvp")
    ap.add_argument("--source", required=True,
                    help="Video file path, webcam index (0), or RTSP URL")
    ap.add_argument("--model", default="yolov8m.pt",
                    help="YOLOv8 model weights file (n/s/m/l/x)")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="Input image size for YOLO inference (larger = better recall, slower)")
    ap.add_argument("--conf", type=float, default=0.30,
                    help="Minimum confidence threshold for detections")
    ap.add_argument("--access-config", default="configs/access_schedule.yaml",
                    help="Path to per-camera access schedule YAML")
    ap.add_argument("--no-display", action="store_true",
                    help="Disable live OpenCV window (for headless/server mode)")
    ap.add_argument("--out", default="out",
                    help="Output directory for video, events, summary")
    ap.add_argument("--stream-id", type=int, default=0,
                    help="Camera/stream ID for access control lookups")
    args = ap.parse_args()

    # ==================== INITIALIZATION ====================
    # Create output directory if it doesn't exist
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLOv8 model — downloads weights automatically on first run
    model = YOLO(args.model)

    # Initialize the bite risk analyzer (tracks dog-person proximity over time)
    bite_analyzer = BiteRiskAnalyzer()

    # Initialize access controller from YAML config (per-camera time windows)
    access_ctrl = AccessController(args.access_config)

    # Initialize the unified event logger (writes bite + access events to JSON)
    event_log = EventLog(str(out_dir / "events.json"))

    # Open video source (file, webcam, or RTSP stream)
    source = _parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {source}")

    # Get source video properties
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0  # Frames per second (default 25 if unknown)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frame count (0 for live streams)

    # Print startup info
    print(f"[mvp] source: {source}  {total} frames @ {fps_src:.1f} fps", flush=True)
    print(f"[mvp] model: {args.model}  imgsz={args.imgsz}  conf={args.conf}", flush=True)
    print(f"[mvp] access control: {'ON' if access_ctrl.enabled else 'OFF'}", flush=True)

    # Video writer — initialized lazily on first frame (we need frame dimensions)
    writer = None
    video_path = str(out_dir / "dogvision_output.mp4")  # Output video filename

    # ==================== TRACKING STATE ====================
    frame_idx = 0                          # Current frame counter
    t_start = time.time()                  # Wall-clock start time for FPS calculation
    unique_dog_ids: set[int] = set()       # All dog track IDs ever seen (for unique count)
    unique_person_ids: set[int] = set()    # All person track IDs ever seen
    total_bite_alerts = 0                  # Cumulative bite risk events
    total_access_violations = 0            # Cumulative access violation events

    # Ghost persistence cache: track_id → {bbox, conf, age}
    # When ByteTrack loses a dog (occlusion, brief disappearance), we keep
    # showing its last-known bounding box for PERSIST_FRAMES frames so the
    # visual output doesn't flicker and the tracker can re-link the ID
    last_seen_dogs: dict[int, dict] = {}

    # ==================== MAIN PROCESSING LOOP ====================
    while True:
        # Read next frame from video source
        ok, frame = cap.read()
        if not ok:  # End of video or read failure
            break

        # Capture high-resolution timestamp for this frame (nanoseconds)
        t_ns = time.time_ns()

        # ========== STAGE 1: UNIFIED YOLO DETECT + TRACK ==========
        # Single YOLO forward pass detects BOTH dogs and persons simultaneously.
        # ByteTrack tracker assigns persistent IDs across frames.
        # Key settings:
        #   - classes=[COCO_DOG, COCO_PERSON]: only detect these two classes
        #   - persist=True: maintain tracker state across frames (required for tracking)
        #   - augment=False: DISABLED — TTA causes bbox jitter which breaks tracker ID consistency
        #   - conf=0.25: balanced threshold — catches most dogs without too many false positives
        results = model.track(
            source=frame,            # Input BGR frame from OpenCV
            imgsz=args.imgsz,        # Resize to this resolution before inference
            conf=args.conf,          # Minimum detection confidence threshold
            iou=0.5,                 # IoU threshold for Non-Maximum Suppression (NMS)
            device="cpu",            # Run on CPU (use 0 for GPU)
            half=False,              # No FP16 on CPU (enable on GPU for 2x speed)
            classes=[COCO_DOG, COCO_PERSON],  # Only detect dogs and persons
            tracker="bytetrack.yaml",  # Use ByteTrack association algorithm
            persist=True,            # Keep tracker state between frames
            verbose=False,           # Suppress per-frame YOLO output
        )
        r = results[0]  # First (only) result — we process one frame at a time

        # ========== PARSE DETECTIONS INTO DOG AND PERSON LISTS ==========
        dogs: list[dict] = []      # Dog detections this frame
        persons: list[dict] = []   # Person detections this frame

        if r.boxes is not None and len(r.boxes) > 0:
            # Extract detection data from YOLO result tensors → NumPy arrays
            xyxy = r.boxes.xyxy.cpu().numpy()    # Bounding boxes: [[x1,y1,x2,y2], ...]
            confs = r.boxes.conf.cpu().numpy()   # Confidence scores: [0.87, 0.63, ...]
            clss = r.boxes.cls.cpu().numpy().astype(int)  # Class IDs: [0, 16, 0, ...]
            # Track IDs from ByteTrack (None if tracker hasn't assigned yet)
            ids = (r.boxes.id.cpu().numpy().astype(int)
                   if r.boxes.id is not None
                   else np.full(len(xyxy), -1))  # -1 means no track ID assigned

            # Route each detection to the appropriate pipeline by class
            for (x1, y1, x2, y2), c, cls, tid in zip(xyxy, confs, clss, ids):
                # Build detection dict with all metadata
                entry = {
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),  # Bounding box coords
                    "conf": float(c),           # Detection confidence (0-1)
                    "track_id": int(tid) if tid >= 0 else -1,  # ByteTrack persistent ID
                    "cls": int(cls),             # COCO class ID (0=person, 16=dog)
                }
                if cls == COCO_DOG:
                    dogs.append(entry)           # Route to dog pipeline
                    if tid >= 0:
                        unique_dog_ids.add(int(tid))  # Track unique dog count
                        # Update ghost cache with latest position
                        last_seen_dogs[int(tid)] = {**entry, "age": 0}
                elif cls == COCO_PERSON:
                    persons.append(entry)        # Route to person pipeline
                    if tid >= 0:
                        unique_person_ids.add(int(tid))  # Track unique person count

        # ========== GHOST DOG PERSISTENCE ==========
        # When ByteTrack loses a dog (brief occlusion, turned away from camera),
        # keep showing its last-known bbox for PERSIST_FRAMES to prevent flicker.
        # This also helps the bite risk analyzer maintain proximity state.
        live_dog_ids = {d["track_id"] for d in dogs if d["track_id"] >= 0}
        for tid in list(last_seen_dogs.keys()):
            if tid in live_dog_ids:
                continue  # Dog still actively detected — skip ghost logic
            last_seen_dogs[tid]["age"] += 1  # Increment frames since last real detection
            if last_seen_dogs[tid]["age"] > PERSIST_FRAMES:
                del last_seen_dogs[tid]  # Ghost expired — remove from cache
                continue
            # Add ghost bbox to dogs list so it renders and feeds bite analyzer
            ghost = last_seen_dogs[tid]
            dogs.append({
                "bbox": ghost["bbox"],          # Last-known position
                "conf": ghost["conf"] * 0.5,    # Halved confidence to indicate ghost
                "track_id": tid,                 # Same track ID for visual continuity
                "cls": COCO_DOG,
            })

        # ========== STAGE 2a: DOG PIPELINE — BITE RISK ANALYSIS ==========
        # Analyze every dog-person pair for aggression indicators:
        #   proximity, physical overlap, lunging motion, sustained contact
        bite_events = bite_analyzer.analyze(
            dogs, persons, frame_idx, args.stream_id, t_ns
        )
        for be in bite_events:
            event_log.log_bite(be)      # Write event to JSON log
            total_bite_alerts += 1       # Increment cumulative counter

        # ========== STAGE 2b: PERSON PIPELINE — ACCESS CONTROL ==========
        # Check each detected person against camera-specific time windows.
        # Person detected outside allowed hours → unauthorized access event.
        access_violations = access_ctrl.check(
            persons, args.stream_id, frame_idx, t_ns
        )
        for av in access_violations:
            event_log.log_access(av)     # Write event to JSON log
            total_access_violations += 1  # Increment cumulative counter

        # ========== RENDER ANNOTATED FRAME ==========
        # Calculate real-time FPS from elapsed wall-clock time
        elapsed = time.time() - t_start
        fps = (frame_idx + 1) / max(elapsed, 1e-9)  # Avoid division by zero

        # Layer annotations onto the frame (order matters — later layers on top)
        annotated = draw_dogs(frame, dogs)                  # Green bboxes for dogs
        annotated = draw_persons(annotated, persons)        # Teal bboxes for persons
        annotated = draw_bite_alerts(annotated, bite_events)  # Red alert lines
        annotated = draw_access_violations(annotated, access_violations)  # Orange violation labels
        annotated = overlay_hud(annotated, {                # Semi-transparent stats panel
            "fps": fps,
            "dogs_now": len([d for d in dogs if d["track_id"] in live_dog_ids]),
            "persons_now": len(persons),
            "unique_dogs": len(unique_dog_ids),
            "bite_alerts": total_bite_alerts,
            "access_violations": total_access_violations,
            "frame": frame_idx,
        })

        # ========== WRITE OUTPUT VIDEO ==========
        # Initialize VideoWriter on first frame (need dimensions from actual frame)
        if writer is None:
            h, w = annotated.shape[:2]  # Get frame height and width
            writer = cv2.VideoWriter(
                video_path,                              # Output file path
                cv2.VideoWriter_fourcc(*"mp4v"),         # MP4 codec
                fps_src,                                  # Match source FPS
                (w, h)                                    # Match source resolution
            )
        writer.write(annotated)  # Write annotated frame to output video

        # ========== LIVE DISPLAY (optional) ==========
        if not args.no_display:
            cv2.imshow("dogvision-mvp", annotated)  # Show frame in OpenCV window
            if (cv2.waitKey(1) & 0xFF) == ord("q"):  # Press 'q' to quit early
                break

        # ========== PROGRESS LOGGING ==========
        # Print progress every 20 frames so user can monitor the run
        if frame_idx % 20 == 0:
            print(
                f"[mvp] frame {frame_idx}/{total}  "
                f"dogs={len(live_dog_ids)}  persons={len(persons)}  "
                f"bites={total_bite_alerts}  access={total_access_violations}  "
                f"fps={fps:.2f}",
                flush=True,  # Flush immediately so output appears in real-time
            )
        frame_idx += 1  # Advance frame counter

        # Flush event log to disk every 200 frames to prevent data loss on crash
        if frame_idx % 200 == 0:
            event_log.flush()

    # ==================== CLEANUP AND FINAL REPORT ====================
    cap.release()               # Release video capture handle
    if writer is not None:
        writer.release()        # Finalize and close output MP4 file
    if not args.no_display:
        cv2.destroyAllWindows()  # Close any OpenCV windows
    event_log.flush()           # Write any remaining events to disk

    # Calculate and write run summary
    dur = time.time() - t_start  # Total elapsed time
    summary = {
        "frames": frame_idx,
        "unique_dogs": len(unique_dog_ids),
        "unique_persons": len(unique_person_ids),
        "bite_alerts": total_bite_alerts,
        "access_violations": total_access_violations,
        "avg_fps": round(frame_idx / max(dur, 1e-9), 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Print final summary to console
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
