# Unauthorized Access Detection  - How It Works

## Overview

The system detects **unauthorized human presence** using a **time-based, per-camera
access control** mechanism. Each camera/stream has configurable allowed time windows.
Any person detected outside those windows is flagged as an **access violation**.

---

## How It Gets Triggered

```
Person detected on Camera 0 at 23:15
            │
            ▼
┌─────────────────────────────────┐
│  Load rules for stream_id = 0  │
│  Allowed: 06:00 – 22:00        │
└─────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│  Is 23:15 within 06:00–22:00?  │
│  NO → UNAUTHORIZED              │
└─────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│  AccessViolation event emitted  │
│  → logged to out/events.json   │
│  → orange box drawn on frame   │
│  → counter incremented in HUD  │
└─────────────────────────────────┘
```

---

## Configuration File

**Location:** `configs/access_schedule.yaml`

```yaml
cameras:
  - stream_id: 0              # Camera index (matches --stream-id CLI flag)
    name: "Front Yard Camera"  # Human-readable label (informational)
    allowed_hours:
      - start: "06:00"        # Persons allowed from 6:00 AM
        end: "22:00"          # Until 10:00 PM
  - stream_id: 1
    name: "Back Gate Camera"
    allowed_hours:
      - start: "07:00"
        end: "19:00"
  - stream_id: 2
    name: "Parking Lot"
    allowed_hours:
      - start: "06:00"
        end: "23:00"
```

### Multiple Windows Per Camera

A camera can have multiple allowed windows (e.g., morning shift + evening shift):

```yaml
cameras:
  - stream_id: 0
    allowed_hours:
      - start: "06:00"
        end: "12:00"     # Morning shift
      - start: "18:00"
        end: "22:00"     # Evening shift
    # Gap: 12:00-18:00 → any person detected = unauthorized
```

### Overnight Windows

Windows that cross midnight are handled automatically:

```yaml
allowed_hours:
  - start: "22:00"
    end: "06:00"   # Night shift: 10 PM to 6 AM the next day
```

The system checks `now >= 22:00 OR now <= 06:00`  - both sides of midnight covered.

---

## What Gets Logged

Each unauthorized access event is written to `out/events.json`:

```json
{
    "event_type": "access_violation",
    "timestamp_ns": 1714762502000000000,
    "frame_idx": 847,
    "stream_id": 0,
    "person_track_id": 23,
    "person_bbox": [412.5, 189.3, 498.7, 421.8],
    "current_time": "23:15:02",
    "allowed_windows": ["06:00-22:00"],
    "reason": "person_outside_allowed_hours"
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | string | Always `"access_violation"` |
| `timestamp_ns` | int | Unix timestamp in nanoseconds when detected |
| `frame_idx` | int | Video frame number where person was detected |
| `stream_id` | int | Camera/stream that captured the person |
| `person_track_id` | int | ByteTrack persistent ID for this person |
| `person_bbox` | [x1,y1,x2,y2] | Bounding box of the person in pixels |
| `current_time` | string | Wall-clock time when violation occurred |
| `allowed_windows` | list[string] | What hours ARE allowed (for context) |
| `reason` | string | Always `"person_outside_allowed_hours"` |

---

## Visual Indicators

When an access violation occurs, the annotated video shows:

1. **Orange bounding box** (thicker than normal) around the unauthorized person
2. **Orange label** below the box: `UNAUTHORIZED @ 23:15:02`
3. **HUD counter** turns orange: `Access violations: 47`

Normal authorized persons are shown with teal bounding boxes and `person#ID` labels.

---

## Code Walkthrough

### File: `behavior/access_control.py`

```
AccessController.__init__(config_path)
    │
    ├── Reads configs/access_schedule.yaml
    ├── Parses each camera's stream_id and allowed_hours
    └── Stores as dict: stream_id → [(start_time, end_time), ...]

AccessController.check(persons, stream_id, frame_idx, t_ns)
    │
    ├── Look up rules for this stream_id
    ├── Get current wall-clock time via datetime.now()
    ├── Check if current time falls within ANY allowed window
    │       ├── Normal window (06:00-22:00): start <= now <= end
    │       └── Overnight window (22:00-06:00): now >= start OR now <= end
    ├── If authorized → return empty list (no violations)
    └── If unauthorized → return AccessViolation for EVERY person detected
```

### File: `analytics/event_log.py`

```
EventLog.log_access(violation)
    │
    ├── Convert AccessViolation dataclass → dict
    ├── Add event_type = "access_violation"
    ├── Append to in-memory buffer
    └── Increment running counter

EventLog.flush()
    │
    ├── Read existing out/events.json (if any)
    ├── Append new events
    └── Write merged JSON back to disk
```

### File: `run_demo_cpu.py` (integration point)

```python
# Line ~223: Person Pipeline  - access control
access_violations = access_ctrl.check(
    persons, args.stream_id, frame_idx, t_ns
)
for av in access_violations:
    event_log.log_access(av)          # Write to event log
    total_access_violations += 1       # Increment HUD counter
```

---

## How to Test Access Control

### Method 1: Change the Config to a Narrow Window

Edit `configs/access_schedule.yaml`:

```yaml
cameras:
  - stream_id: 0
    allowed_hours:
      - start: "03:00"
        end: "04:00"   # Only 1 hour allowed → most times trigger violations
```

Run the demo  - every person detection will be flagged as unauthorized.

### Method 2: Run During Off-Hours

If your config allows 06:00–22:00 and you run the demo after 10 PM,
all person detections will automatically be flagged.

### Method 3: Check the Event Log

After a run:

```bash
python -c "
import json
events = json.load(open('out/events.json'))
access = [e for e in events if e['event_type'] == 'access_violation']
print(f'{len(access)} access violations')
for e in access[:5]:
    print(f'  frame {e[\"frame_idx\"]} | person#{e[\"person_track_id\"]} | {e[\"current_time\"]} | allowed: {e[\"allowed_windows\"]}')
"
```

---

## Relationship to GPU Acceleration

While the access control logic itself is lightweight (time comparison + rule lookup),
it integrates with the GPU-accelerated pipeline:

1. **Person detection** runs on GPU via YOLOv8 CUDA + TensorRT FP16
2. **ByteTrack** assigns persistent person IDs on GPU
3. **Event metadata** (frame, bbox, track_id) comes from GPU inference results
4. **Event aggregation** uses cuDF on GPU when the full GPU pipeline is active
5. **Access control check** is the only CPU step  - a simple time comparison

The GPU does the heavy lifting (detecting and tracking persons at 25+ FPS);
the access controller just applies a business rule to the GPU's output.

---

## Architecture Position

```
GPU Layer                          CPU Layer
─────────                          ─────────
YOLOv8 (CUDA)                     
    │ detect person                
    ▼                              
ByteTrack (CUDA-assisted)          
    │ assign track_id              
    ▼                              
Person ROI (CuPy)                  AccessController.check()
    │                                  │
    └──────────────────────────────────┘
                                       │
                                       ▼
                                  AccessViolation
                                       │
                                  EventLog.log_access()
                                       │
                                  out/events.json
```
