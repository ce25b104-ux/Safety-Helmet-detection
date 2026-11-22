
import cv2
import csv
import time
import torch
import numpy as np
import inspect

from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace

from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


# ============================================================
# 1. CONFIGURATION
# ============================================================

VIDEO_PATH = Path(
    r"C:\Users\srina\Downloads\WhatsApp Video 2026-11-22 at 14.14.14.mp4"
)

CSV_PATH = Path(
    r"C:\Users\srina\Downloads\Part5_Output\stage1_detections-2.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\srina\Downloads\Part5_Output"
)

SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"

OUTPUT_VIDEO = OUTPUT_DIR / "stage2_tracked_video_2.mp4"
ALERT_CSV = OUTPUT_DIR / "alerts_2.csv"
TRACK_CSV = OUTPUT_DIR / "track_summary-2.csv"

# Temporal rules
CONFIRMATION_SECONDS = 0.1
UNHELMETED_SECONDS = 0.15

# Short detection interruptions
OCCLUSION_GRACE_SECONDS = 1

# Custom track recovery
RECOVERY_WINDOW_SECONDS = 1.0
MAX_CENTER_DISTANCE_DIAGONALS = 1.5
MIN_IOU_FOR_RECOVERY = 0.05

# ByteTrack
TRACK_BUFFER_SECONDS = 3

TRACK_HIGH_THRESH = 0.3
TRACK_LOW_THRESH = 0.2
NEW_TRACK_THRESH = 0.20
MATCH_THRESH = 0.80


# ============================================================
# 2. LOAD YOLO DETECTIONS
# ============================================================

print("Loading YOLO detections...")

detections_by_frame = defaultdict(list)

with open(CSV_PATH, "r", newline="") as f:

    reader = csv.DictReader(f)

    required_columns = {
        "frame",
        "timestamp_seconds",
        "class_id",
        "class_name",
        "confidence",
        "x1", "y1", "x2", "y2"
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
            "y2": float(row["y2"])
        })

print("Total detections loaded:", sum(
    len(v) for v in detections_by_frame.values()
))


# ============================================================
# 3. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    raise RuntimeError("Invalid video FPS.")

print("\nVIDEO INFORMATION")
print("-----------------------------")
print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Frames:", total_frames)
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

parameters = inspect.signature(BYTETracker).parameters

if "frame_rate" in parameters:
    tracker = BYTETracker(
        tracker_args,
        frame_rate=int(round(fps))
    )
else:
    tracker = BYTETracker(tracker_args)

print("\nByteTrack initialized.")
print("Track buffer:", tracker_args.track_buffer, "frames")


# ============================================================
# 5. TRACK STATE
# ============================================================

# ByteTrack raw ID -> persistent canonical ID
raw_to_canonical = {}

# Persistent state indexed by canonical ID
track_states = {}

alerts = []
frame_number = 0

SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video.")


# ============================================================
# 6. BOUNDING BOX COMPARISON FUNCTIONS
# ============================================================

def box_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def recovery_score(old_box, new_box):

    iou = box_iou(old_box, new_box)

    old_cx = (old_box[0] + old_box[2]) / 2
    old_cy = (old_box[1] + old_box[3]) / 2

    new_cx = (new_box[0] + new_box[2]) / 2
    new_cy = (new_box[1] + new_box[3]) / 2

    distance = np.sqrt(
        (old_cx - new_cx) ** 2 +
        (old_cy - new_cy) ** 2
    )

    old_diag = np.sqrt(
        (old_box[2] - old_box[0]) ** 2 +
        (old_box[3] - old_box[1]) ** 2
    )

    new_diag = np.sqrt(
        (new_box[2] - new_box[0]) ** 2 +
        (new_box[3] - new_box[1]) ** 2
    )

    reference_diag = max((old_diag + new_diag) / 2, 1.0)

    normalized_distance = distance / reference_diag

    old_area = max(
        1.0,
        (old_box[2] - old_box[0]) *
        (old_box[3] - old_box[1])
    )

    new_area = max(
        1.0,
        (new_box[2] - new_box[0]) *
        (new_box[3] - new_box[1])
    )

    size_ratio = new_area / old_area

    acceptable_size = 0.5 <= size_ratio <= 2.0

    acceptable_match = (
        iou >= MIN_IOU_FOR_RECOVERY or
        (
            normalized_distance <= MAX_CENTER_DISTANCE_DIAGONALS
            and acceptable_size
        )
    )

    if not acceptable_match:
        return None

    return normalized_distance - 0.5 * iou


# ============================================================
# 7. CREATE TRACK STATE
# ============================================================

def create_track_state(timestamp, box):

    return {
        "first_seen": timestamp,
        "last_seen": timestamp,

        "confirmation_start": timestamp,
        "confirmed": False,

        "unhelmeted_start": None,

        "alerted": False,

        "last_box": box,

        "helmet_observations": 0,
        "unhelmeted_observations": 0,

        "recovery_count": 0
    }


# ============================================================
# 8. MAIN VIDEO LOOP
# ============================================================

print("\nStarting tracking and alerting...\n")

start_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    timestamp = frame_number / fps

    detections = detections_by_frame.get(frame_number, [])

    # Prepare boxes for ByteTrack
    detection_array = []

    for d in detections:

        detection_array.append([
            d["x1"],
            d["y1"],
            d["x2"],
            d["y2"],
            d["confidence"],
            d["class_id"]
        ])

    if detection_array:
        detection_array = np.asarray(
            detection_array,
            dtype=np.float32
        )
    else:
        detection_array = np.empty(
            (0, 6),
            dtype=np.float32
        )

    detection_boxes = Boxes(
        torch.tensor(detection_array, dtype=torch.float32),
        orig_shape=(height, width)
    )

    # ByteTrack update
    tracked_objects = tracker.update(detection_boxes, frame)

    # Convert results to a normal list
    objects = []

    for obj in tracked_objects:

        objects.append({
            "raw_id": int(obj[4]),
            "box": [
                float(obj[0]),
                float(obj[1]),
                float(obj[2]),
                float(obj[3])
            ],
            "confidence": float(obj[5]),
            "class_id": int(obj[6])
        })

    # --------------------------------------------------------
    # 8A. MAP EXISTING TRACK IDS
    # --------------------------------------------------------

    used_canonical_ids = set()
    unknown_objects = []

    for obj in objects:

        raw_id = obj["raw_id"]

        if raw_id in raw_to_canonical:

            canonical_id = raw_to_canonical[raw_id]

            obj["canonical_id"] = canonical_id
            used_canonical_ids.add(canonical_id)

        else:
            unknown_objects.append(obj)

    # --------------------------------------------------------
    # 8B. CUSTOM LOST-TRACK RECOVERY
    # --------------------------------------------------------

    possible_matches = []

    for obj in unknown_objects:

        new_box = obj["box"]

        for canonical_id, state in track_states.items():

            if canonical_id in used_canonical_ids:
                continue

            gap = timestamp - state["last_seen"]

            if gap <= 0 or gap > RECOVERY_WINDOW_SECONDS:
                continue

            score = recovery_score(
                state["last_box"],
                new_box
            )

            if score is not None:

                possible_matches.append((
                    score,
                    obj,
                    canonical_id
                ))

    # Best matches first; prevent two detections from
    # recovering the same old track.
    possible_matches.sort(key=lambda x: x[0])

    matched_objects = set()
    matched_tracks = set()

    for score, obj, canonical_id in possible_matches:

        object_key = id(obj)

        if object_key in matched_objects:
            continue

        if canonical_id in matched_tracks:
            continue

        raw_id = obj["raw_id"]

        raw_to_canonical[raw_id] = canonical_id
        obj["canonical_id"] = canonical_id

        used_canonical_ids.add(canonical_id)

        matched_objects.add(object_key)
        matched_tracks.add(canonical_id)

        track_states[canonical_id]["recovery_count"] += 1

        print(
            f"Recovered track {canonical_id} "
            f"from new ID {raw_id} at {timestamp:.2f}s"
        )

    # --------------------------------------------------------
    # 8C. CREATE NEW TRACKS WHEN NO MATCH EXISTS
    # --------------------------------------------------------

    for obj in unknown_objects:

        if id(obj) in matched_objects:
            continue

        raw_id = obj["raw_id"]

        canonical_id = raw_id

        raw_to_canonical[raw_id] = canonical_id
        obj["canonical_id"] = canonical_id

        if canonical_id not in track_states:

            track_states[canonical_id] = create_track_state(
                timestamp,
                obj["box"]
            )

        used_canonical_ids.add(canonical_id)

    # --------------------------------------------------------
    # 8D. UPDATE TRACKS AND TEMPORAL CLASSIFICATION
    # --------------------------------------------------------

    for obj in objects:

        track_id = obj["canonical_id"]

        box = obj["box"]

        class_id = obj["class_id"]
        confidence = obj["confidence"]

        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        state = track_states[track_id]

        gap = timestamp - state["last_seen"]

        # Reset timers only after a sufficiently long gap.
        if gap > OCCLUSION_GRACE_SECONDS:

            if not state["confirmed"]:
                state["confirmation_start"] = timestamp

            state["unhelmeted_start"] = None

        state["last_seen"] = timestamp
        state["last_box"] = box

        # Confirm after 0.5 seconds of continuous observation,
        # allowing only short detection interruptions.
        if (
            not state["confirmed"] and
            timestamp - state["confirmation_start"]
            >= CONFIRMATION_SECONDS
        ):
            state["confirmed"] = True

            print(
                f"Track {track_id} confirmed "
                f"at {timestamp:.2f}s"
            )

        # Class 0 = helmet-wearing head
        # Class 1 = unhelmeted head

        if class_id == 0:

            state["helmet_observations"] += 1
            state["unhelmeted_start"] = None

            status = "HELMET"

        elif class_id == 1:

            state["unhelmeted_observations"] += 1

            status = "UNHELMETED"

            if state["confirmed"]:

                if state["unhelmeted_start"] is None:
                    state["unhelmeted_start"] = timestamp

                violation_duration = (
                    timestamp - state["unhelmeted_start"]
                )

                if (
                    violation_duration > UNHELMETED_SECONDS
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
                        "confidence": round(confidence, 4),
                        "snapshot": str(snapshot_path)
                    })

                    print(
                        f"\nALERT: Track {track_id} "
                        f"at {timestamp:.2f}s\n"
                    )

        else:
            status = "UNKNOWN"

        # ----------------------------------------------------
        # ANNOTATE
        # ----------------------------------------------------

        if state["alerted"]:

            color = (0, 0, 255)
            status = "ALERT"

        elif not state["confirmed"]:

            color = (0, 165, 255)
            status = "CONFIRMING"

        elif class_id == 0:

            color = (0, 200, 0)

        else:

            color = (0, 165, 255)

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
    # 8E. HANDLE MISSING TRACKS
    # --------------------------------------------------------

    for track_id, state in track_states.items():

        if track_id not in used_canonical_ids:

            missing_duration = timestamp - state["last_seen"]

            if missing_duration > OCCLUSION_GRACE_SECONDS:

                state["unhelmeted_start"] = None

                if not state["confirmed"]:
                    state["confirmation_start"] = timestamp

    # --------------------------------------------------------
    # 8F. WRITE FRAME
    # --------------------------------------------------------

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
# 9. SAVE OUTPUTS
# ============================================================

cap.release()
writer.release()

# Alerts CSV
with open(ALERT_CSV, "w", newline="") as f:

    fields = [
        "track_id",
        "timestamp_seconds",
        "timestamp_hms",
        "confidence",
        "snapshot"
    ]

    output = csv.DictWriter(f, fieldnames=fields)

    output.writeheader()
    output.writerows(alerts)


# Track summary CSV
with open(TRACK_CSV, "w", newline="") as f:

    fields = [
        "track_id",
        "first_seen",
        "last_seen",
        "duration_seconds",
        "confirmed",
        "alerted",
        "helmet_observations",
        "unhelmeted_observations",
        "recovery_count"
    ]

    output = csv.DictWriter(f, fieldnames=fields)
    output.writeheader()

    for track_id, state in track_states.items():

        output.writerow({
            "track_id": track_id,
            "first_seen": round(state["first_seen"], 3),
            "last_seen": round(state["last_seen"], 3),
            "duration_seconds": round(
                state["last_seen"] - state["first_seen"], 3
            ),
            "confirmed": state["confirmed"],
            "alerted": state["alerted"],
            "helmet_observations": state["helmet_observations"],
            "unhelmeted_observations": state["unhelmeted_observations"],
            "recovery_count": state["recovery_count"]
        })


# ============================================================
# 10. FINAL REPORT
# ============================================================

elapsed = time.time() - start_time

print("\n" + "=" * 60)
print("STAGE 2 COMPLETE")
print("=" * 60)

print("Frames processed:", frame_number)
print("Unique persistent track IDs:", len(track_states))
print("Alerts generated:", len(alerts))

print(
    "Recovered track IDs:",
    sum(s["recovery_count"] for s in track_states.values())
)

print("Processing time:", round(elapsed / 60, 2), "minutes")
print("Processing FPS:", round(frame_number / elapsed, 2))

print("\nOUTPUT FILES")
print("-----------------------------")
print(OUTPUT_VIDEO)
print(ALERT_CSV)
print(TRACK_CSV)
print(SNAPSHOT_DIR)
