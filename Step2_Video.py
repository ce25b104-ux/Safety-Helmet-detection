
import cv2
import csv
import time
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace

from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

VIDEO_PATH = Path(
    r"C:\Users\srina\Downloads\WhatsApp Video 2026-11-22 at 12.10.41.mp4"
)

CSV_PATH = Path(
    r"C:\Users\srina\Downloads\Part5_Output\stage1_detections.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\srina\Downloads\Part5_Output"
)

SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"

OUTPUT_VIDEO = OUTPUT_DIR / "stage2_tracked_annotated.mp4"
ALERT_CSV = OUTPUT_DIR / "alerts.csv"
TRACK_CSV = OUTPUT_DIR / "track_summary.csv"

# Your temporal rules
CONFIRMATION_SECONDS = 0.1
UNHELMETED_SECONDS = 0.15

# Brief detection interruptions tolerated
OCCLUSION_GRACE_SECONDS = 1

# ByteTrack configuration
TRACK_BUFFER_SECONDS = 1.0
TRACK_HIGH_THRESH = 0.25
TRACK_LOW_THRESH = 0.10
NEW_TRACK_THRESH = 0.25
MATCH_THRESH = 0.80


# ============================================================
# 2. LOAD DETECTIONS FROM EXISTING CSV
# ============================================================

print("Loading existing YOLO detections...")

detections_by_frame = defaultdict(list)

with open(CSV_PATH, "r", newline="") as f:
    reader = csv.DictReader(f)

    required_columns = {
        "frame", "timestamp_seconds", "class_id","class_name",
        "confidence", "x1", "y1", "x2", "y2"
    }

    if not required_columns.issubset(set(reader.fieldnames or [])):
        raise ValueError(
            f"Unexpected CSV columns: {reader.fieldnames}"
        )

    for row in reader:
        frame_id = int(row["frame"])

        detections_by_frame[frame_id].append({
            "class_id": int(row["class_id"]),
            "confidence": float(row["confidence"]),
            "x1": float(row["x1"]),
            "y1": float(row["y1"]),
            "x2": float(row["x2"]),
            "y2": float(row["y2"]),
        })


# ============================================================
# 3. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open input video.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\nVIDEO INFORMATION")
print("-----------------------------")
print("FPS:", fps)
print("Frames:", total_frames)
print("Resolution:", width, "x", height)
print("Duration:", round(total_frames / fps, 2), "seconds")


# ============================================================
# 4. INITIALIZE BYTETRACK
# ============================================================

tracker_args = SimpleNamespace(
    track_high_thresh=TRACK_HIGH_THRESH,
    track_low_thresh=TRACK_LOW_THRESH,
    new_track_thresh=NEW_TRACK_THRESH,
    track_buffer=max(1, int(TRACK_BUFFER_SECONDS * fps)),
    match_thresh=MATCH_THRESH,
    fuse_score=True,
    mot20=False
)

import inspect

tracker_parameters = inspect.signature(BYTETracker).parameters

if "frame_rate" in tracker_parameters:
    tracker = BYTETracker(
        tracker_args,
        frame_rate=int(round(fps))
    )
else:
    tracker = BYTETracker(tracker_args)

# State maintained separately for every track ID
track_states = {}

alerts = []
track_summaries = []

SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video.")


# ============================================================
# 5. PROCESS VIDEO FRAME BY FRAME
# ============================================================

frame_number = 0
start_time = time.time()

print("\nStarting ByteTrack and temporal alerting...\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    timestamp = frame_number / fps

    frame_detections = detections_by_frame.get(
        frame_number, []
    )

    # Prepare detections in Ultralytics Boxes format:
    # x1, y1, x2, y2, confidence, class_id

    detection_array = []

    for d in frame_detections:
        detection_array.append([
            d["x1"],
            d["y1"],
            d["x2"],
            d["y2"],
            d["confidence"],
            d["class_id"]
        ])

    if len(detection_array) == 0:
        detection_array = np.empty((0, 6), dtype=np.float32)

    detection_tensor = torch.tensor(
        detection_array,
        dtype=torch.float32
    )

    boxes = Boxes(
        detection_tensor,
        orig_shape=(height, width)
    )

    # Update ByteTrack
    tracked_objects = tracker.update(boxes, frame)

    currently_visible = set()

    # Each returned row:
    # x1, y1, x2, y2, track_id, confidence, class_id, detection_index

    for obj in tracked_objects:

        x1, y1, x2, y2 = map(float, obj[:4])

        track_id = int(obj[4])
        confidence = float(obj[5])
        class_id = int(obj[6])

        currently_visible.add(track_id)

        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width - 1, int(x2)))
        y2 = max(0, min(height - 1, int(y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        # ----------------------------------------------------
        # TRACK STATE
        # ----------------------------------------------------

        if track_id not in track_states:

            track_states[track_id] = {
                "first_seen": timestamp,
                "last_seen": timestamp,
                "unhelmeted_start": None,
                "confirmed": False,
                "alerted": False,
                "helmet_observations": 0,
                "unhelmeted_observations": 0
            }

        state = track_states[track_id]

        # If the gap is too large, restart continuous tracking
        if timestamp - state["last_seen"] > OCCLUSION_GRACE_SECONDS:

            state["first_seen"] = timestamp
            state["unhelmeted_start"] = None
            state["confirmed"] = False

        state["last_seen"] = timestamp

        # ----------------------------------------------------
        # TRACK CONFIRMATION: 0.5 SECONDS
        # ----------------------------------------------------

        track_duration = timestamp - state["first_seen"]

        if track_duration >= CONFIRMATION_SECONDS:
            state["confirmed"] = True

        # ----------------------------------------------------
        # TEMPORAL CLASSIFICATION
        # ----------------------------------------------------

        if class_id == 0:

            # Helmet detected
            state["helmet_observations"] += 1

            # Explicit helmet evidence resets violation timer
            state["unhelmeted_start"] = None

            label = "HELMET"

        elif class_id == 1:

            # Unhelmeted head detected
            state["unhelmeted_observations"] += 1

            label = "UNHELMETED"

            if state["confirmed"]:

                if state["unhelmeted_start"] is None:
                    state["unhelmeted_start"] = timestamp

                unhelmeted_duration = (
                    timestamp - state["unhelmeted_start"]
                )

                # Strictly MORE than one second
                if (
                    unhelmeted_duration > UNHELMETED_SECONDS
                    and not state["alerted"]
                ):

                    state["alerted"] = True

                    crop = frame[y1:y2, x1:x2]

                    snapshot_name = (
                        f"track_{track_id}_"
                        f"{timestamp:.2f}s.jpg"
                    )

                    snapshot_path = SNAPSHOT_DIR / snapshot_name

                    if crop.size > 0:
                        cv2.imwrite(str(snapshot_path), crop)

                    alerts.append({
                        "track_id": track_id,
                        "timestamp_seconds": round(timestamp, 3),
                        "timestamp_hms": time.strftime(
                            "%H:%M:%S",
                            time.gmtime(timestamp)
                        ),
                        "snapshot": str(snapshot_path),
                        "confidence": round(confidence, 4)
                    })

                    print(
                        f"ALERT! Track {track_id} "
                        f"at {timestamp:.2f} seconds"
                    )

        else:
            label = "UNKNOWN"

        # ----------------------------------------------------
        # ANNOTATION
        # ----------------------------------------------------

        if state["alerted"]:
            color = (0, 0, 255)
            status = "ALERT"
        elif not state["confirmed"]:
            color = (0, 165, 255)
            status = "CONFIRMING"
        elif class_id == 0:
            color = (0, 200, 0)
            status = "HELMET"
        else:
            color = (0, 165, 255)
            status = "UNHELMETED"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            f"ID {track_id} | {status}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    # --------------------------------------------------------
    # HANDLE TEMPORARY DISAPPEARANCE
    # --------------------------------------------------------

    for track_id, state in track_states.items():

        if track_id not in currently_visible:

            missing_duration = timestamp - state["last_seen"]

            if missing_duration > OCCLUSION_GRACE_SECONDS:

                state["unhelmeted_start"] = None
                state["confirmed"] = False

    # Timestamp on video
    cv2.putText(
        frame,
        f"Time: {timestamp:.2f}s",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    writer.write(frame)

    frame_number += 1

    if frame_number % 100 == 0:
        print(
            f"Processed {frame_number}/{total_frames} frames"
        )


# ============================================================
# 6. SAVE ALERT LOG
# ============================================================

cap.release()
writer.release()

alert_fields = [
    "track_id",
    "timestamp_seconds",
    "timestamp_hms",
    "snapshot",
    "confidence"
]

with open(ALERT_CSV, "w", newline="") as f:

    output = csv.DictWriter(f, fieldnames=alert_fields)
    output.writeheader()
    output.writerows(alerts)


# ============================================================
# 7. SAVE TRACK SUMMARY
# ============================================================

for track_id, state in track_states.items():

    track_summaries.append({
        "track_id": track_id,
        "first_seen": round(state["first_seen"], 3),
        "last_seen": round(state["last_seen"], 3),
        "duration_seconds": round(
            state["last_seen"] - state["first_seen"], 3
        ),
        "confirmed": state["confirmed"],
        "alerted": state["alerted"],
        "helmet_observations": state["helmet_observations"],
        "unhelmeted_observations": state["unhelmeted_observations"]
    })

with open(TRACK_CSV, "w", newline="") as f:

    fields = [
        "track_id",
        "first_seen",
        "last_seen",
        "duration_seconds",
        "confirmed",
        "alerted",
        "helmet_observations",
        "unhelmeted_observations"
    ]

    output = csv.DictWriter(f, fieldnames=fields)
    output.writeheader()
    output.writerows(track_summaries)


# ============================================================
# 8. FINAL REPORT
# ============================================================

elapsed = time.time() - start_time

print("\n" + "=" * 55)
print("STAGE 2 COMPLETE")
print("=" * 55)

print("Frames processed:", frame_number)
print("Unique track IDs:", len(track_states))
print("Alerts generated:", len(alerts))
print("Processing time:", round(elapsed / 60, 2), "minutes")
print("Processing FPS:", round(frame_number / elapsed, 2))

print("\nOUTPUT FILES")
print("-----------------------------")
print(OUTPUT_VIDEO)
print(ALERT_CSV)
print(TRACK_CSV)
print(SNAPSHOT_DIR)
