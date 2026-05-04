# Pipeline Walkthrough — Detection to Output, Step by Step

Exact flow of one video frame through the entire system. Every step maps to a file + function.

---

## The Full Pipeline (Single Frame)

```
VIDEO FILE
    │
    ▼ cv2.VideoCapture.read()                          [run_demo_cpu.py:91]
FRAME (BGR numpy array, 854×480×3)
    │
    ▼ model.track(classes=[0,16], conf=0.30)           [run_demo_cpu.py:108-117]
YOLO INFERENCE (PyTorch CUDA / CPU)
    │ YOLOv8 forward pass → 80-class predictions
    │ Class filter → keep only person(0) + dog(16)
    │ NMS (IoU=0.5) → remove duplicate boxes
    │ ByteTrack → assign persistent track IDs
    │
    ▼ r.boxes.xyxy / .conf / .cls / .id               [run_demo_cpu.py:124-131]
RAW DETECTIONS (tensors → numpy)
    │
    ▼ Route by cls                                      [run_demo_cpu.py:135-148]
    ├──── cls == 16 → dogs list
    │     └── update ghost cache: last_seen_dogs[tid]   [run_demo_cpu.py:143]
    └──── cls == 0  → persons list
    │
    ▼ Ghost persistence check                           [run_demo_cpu.py:151-165]
    │ For each lost dog (not in live detections):
    │   age += 1
    │   if age > 30 frames → delete from cache
    │   else → add ghost bbox to dogs list (conf × 0.5)
    │
    ├────────────────────────┬──────────────────────────┐
    ▼                        ▼                          │
DOG PIPELINE             PERSON PIPELINE                │
    │                        │                          │
    ▼                        ▼                          │
BiteRiskAnalyzer         AccessController               │
.analyze()               .check()                       │
[bite_detector.py:86]    [access_control.py:75]         │
    │                        │                          │
    │ For each               │ Load rules for           │
    │ (dog, person) pair:    │ this stream_id           │
    │                        │                          │
    │ ┌─ Factor 1: PROX     │ Get datetime.now()       │
    │ │  center_dist/diag    │                          │
    │ │  [bite_detector:105] │ Check if now is          │
    │ │                      │ within any allowed       │
    │ ├─ Factor 2: OVERLAP   │ time window              │
    │ │  IoU(dog, person)    │ [access_control:88]      │
    │ │  [bite_detector:110] │                          │
    │ │                      │ If OUTSIDE allowed:      │
    │ ├─ Factor 3: LUNGE     │   → AccessViolation      │
    │ │  area_now/area_4ago  │     for each person      │
    │ │  [bite_detector:115] │                          │
    │ │                      │ If INSIDE allowed:       │
    │ ├─ Factor 4: SUSTAINED │   → empty list           │
    │ │  frames_close count  │                          │
    │ │  [bite_detector:124] │                          │
    │ │                      │                          │
    │ ▼                      ▼                          │
    │ risk = 0.30×prox       AccessViolation            │
    │      + 0.25×overlap    events                     │
    │      + 0.25×lunge      [access_control.py:96]     │
    │      + 0.20×sustained                             │
    │ [bite_detector:131]                               │
    │                                                   │
    │ if risk ≥ 0.55:                                   │
    │   → BiteEvent                                     │
    │   [bite_detector:136]                             │
    │                                                   │
    ├────────────────────────┬──────────────────────────┘
    ▼                        ▼
EVENT LOG                EVENT LOG
event_log.log_bite()     event_log.log_access()
[event_log.py:23]        [event_log.py:31]
    │                        │
    └────────┬───────────────┘
             ▼
    RENDER ANNOTATIONS                                  [run_demo_cpu.py:174-189]
             │
             ├── draw_dogs(frame, dogs)                 [draw.py:28]
             │   Green bbox + "dog" label per detection
             │   Color = hash(track_id) → HSV (unique per dog)
             │
             ├── draw_persons(frame, persons)           [draw.py:39]
             │   Teal bbox + "person#ID conf" label
             │
             ├── draw_bite_alerts(frame, events)        [draw.py:50]
             │   Red line: dog center ↔ person center
             │   "BITE RISK 68%" label at midpoint
             │   Thick red border on aggressive dog
             │
             ├── draw_access_violations(frame, viols)   [draw.py:68]
             │   Orange box around unauthorized person
             │   "UNAUTHORIZED @ 23:15:02" label
             │
             └── overlay_hud(frame, stats)              [draw.py:78]
                 Semi-transparent black panel:
                 FPS, Dogs, Persons, Unique dogs,
                 Bite alerts (red if >0),
                 Access violations (orange if >0), Frame#
             │
             ▼
    VIDEO WRITER                                        [run_demo_cpu.py:192-197]
    cv2.VideoWriter.write(annotated_frame)
    → out/dogvision_output.mp4
             │
             ▼
    PERIODIC FLUSH (every 200 frames)                   [run_demo_cpu.py:210]
    event_log.flush() → out/events.json
             │
             ▼
    NEXT FRAME (loop back to top)
```

---

## Multi-Stream Grid Addition

For `run_multi_stream.py`, the pipeline above runs independently per stream, then:

```
    CAM 0 annotated ──┐
    CAM 1 annotated ──┤
    CAM 2 annotated ──┼── make_grid() → 2×2 numpy array (1280×720)
    CAM 3 annotated ──┘   [run_multi_stream.py:178]
                          │
                          ├── Add "CAM N" label per cell   [run_multi_stream.py:157]
                          ├── Add global FPS counter       [run_multi_stream.py:195]
                          └── writer.write(grid)           [run_multi_stream.py:199]
                              → out/multi_stream_output.mp4
```

---

## File → Function Map (Detection to Output)

| Step | File | Function | Line(s) |
|------|------|----------|---------|
| Read frame | `run_demo_cpu.py` | `cap.read()` | 91 |
| YOLO detect+track | `run_demo_cpu.py` | `model.track()` | 108-117 |
| Parse detections | `run_demo_cpu.py` | for loop over `r.boxes` | 124-148 |
| Ghost persistence | `run_demo_cpu.py` | ghost dog loop | 151-165 |
| Bite risk scoring | `behavior/bite_detector.py` | `BiteRiskAnalyzer.analyze()` | 86-155 |
| ├─ Proximity | `behavior/bite_detector.py` | `_center_dist()` + normalize | 105 |
| ├─ Overlap | `behavior/bite_detector.py` | `_iou()` | 110 |
| ├─ Lunge | `behavior/bite_detector.py` | area ratio check | 115-121 |
| ├─ Sustained | `behavior/bite_detector.py` | `frames_close` counter | 124-127 |
| └─ Composite | `behavior/bite_detector.py` | weighted sum | 131 |
| Access control | `behavior/access_control.py` | `AccessController.check()` | 75-105 |
| ├─ Load rules | `behavior/access_control.py` | `_load()` | 55-65 |
| ├─ Time check | `behavior/access_control.py` | `datetime.now().time()` | 88 |
| └─ Emit violation | `behavior/access_control.py` | `AccessViolation()` | 96-104 |
| Log bite event | `analytics/event_log.py` | `EventLog.log_bite()` | 23-28 |
| Log access event | `analytics/event_log.py` | `EventLog.log_access()` | 31-36 |
| Draw dogs | `utils/draw.py` | `draw_dogs()` | 28-36 |
| Draw persons | `utils/draw.py` | `draw_persons()` | 39-48 |
| Draw bite alerts | `utils/draw.py` | `draw_bite_alerts()` | 50-65 |
| Draw violations | `utils/draw.py` | `draw_access_violations()` | 68-74 |
| Draw HUD | `utils/draw.py` | `overlay_hud()` | 78-100 |
| Write video | `run_demo_cpu.py` | `writer.write()` | 197 |
| Flush events | `analytics/event_log.py` | `EventLog.flush()` | 44-54 |
| Write summary | `run_demo_cpu.py` | `json.dumps(summary)` | 228-229 |

---

## GPU Pipeline Variant (demo.py)

Same detection→behavior→output flow, but threaded:

| Thread | What It Does | File |
|--------|-------------|------|
| Producer | `iter_frames()` → decode → queue | `pipeline/orchestrator.py:_producer()` |
| Inference | `DogDetector.track()` → CuPy ring append | `pipeline/orchestrator.py:_inference()` |
| Consumer | cuDF analytics → draw → write | `pipeline/orchestrator.py:_consumer()` |

Additional GPU stages in threaded pipeline:
- CuPy ring buffer append: `analytics/ring_buffer.py:append_batch()` — O(1) GPU write
- cuDF snapshot: `analytics/ring_buffer.py:snapshot()` — GPU arrays → GPU DataFrame
- cuDF analytics: `analytics/window.py:compute()` — groupby/agg/nunique on GPU
- ROI histogram: `analytics/roi_hist.py:roi_histograms()` — CuPy HSV bincount on GPU

---

## Bite Risk Scoring Formula

```
risk = 0.30 × proximity_score      ← how close dog is to person
     + 0.25 × overlap_score        ← bounding box physical overlap
     + 0.25 × lunge_score          ← rapid bbox area growth (approaching)
     + 0.20 × sustained_score      ← frames of continuous close contact

if risk ≥ 0.55 → BITE RISK EVENT emitted
```

Each factor outputs 0.0–1.0. Composite range: 0.0–1.0.

---

## Access Control Decision Tree

```
Person detected on stream_id S
        │
        ▼
Rules exist for S?  ──NO──→ ALLOW (no rules = open access)
        │
       YES
        │
        ▼
Current time within ANY allowed window?
        │
       YES ──→ ALLOW
        │
       NO  ──→ UNAUTHORIZED → AccessViolation event
                               → orange box on frame
                               → counter in HUD
```

Time source: `datetime.now().time()` = **system clock** (not video timestamp).
