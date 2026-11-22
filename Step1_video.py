
import cv2
import csv
import time
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    r"C:\Users\srina\Downloads\deployment_exports\YOLO26m_FP32_640\YOLO26m_FP32_640.onnx"
)

# Enter your actual video path here.
VIDEO_PATH = Path(
    r"C:\Users\srina\Downloads\WhatsApp Video 2026-11-22 at 14.14.14.mp4"
)

OUTPUT_DIR = Path(
    r"C:\Users\srina\Downloads\Part5_Output"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "stage1_video2.mp4"
OUTPUT_CSV = OUTPUT_DIR / "stage1_detections-2.csv"

IMAGE_SIZE = 640

# Keep this relatively low for the initial inspection.
# We can adjust it after seeing the detections.
CONFIDENCE = 0.05
IOU_THRESHOLD = 0.5

CLASS_NAMES = {
    0: "HAT",
    1: "UNHELMETED_HEAD"
}


# ============================================================
# 2. CHECK FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}\n"
        "Please update VIDEO_PATH in the script."
    )


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("Loading YOLO26m ONNX model...")

model = YOLO(str(MODEL_PATH), task="detect")

print("Model loaded successfully.")
print("Classes:", model.names)


# ============================================================
# 4. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open input video.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    raise RuntimeError("Invalid video FPS.")

duration = total_frames / fps

print("\nVIDEO INFORMATION")
print("-----------------------------")
print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Total frames:", total_frames)
print("Duration:", round(duration, 2), "seconds")


# ============================================================
# 5. CREATE OUTPUT VIDEO
# ============================================================

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
# 6. RUN DETECTION FRAME BY FRAME
# ============================================================

csv_rows = []

frame_number = 0
start_time = time.perf_counter()

class_counts = {
    "HAT": 0,
    "UNHELMETED_HEAD": 0
}

print("\nStarting video inference...\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Actual video timestamp.
    timestamp = frame_number / fps

    # Run YOLO inference.
    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=IOU_THRESHOLD,
        device="cpu",
        verbose=False
    )

    result = results[0]

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].cpu().numpy()
            )

            class_name = CLASS_NAMES.get(
                class_id,
                f"CLASS_{class_id}"
            )

            class_counts[class_name] = (
                class_counts.get(class_name, 0) + 1
            )

            # Different colors for the two classes.
            if class_id == 0:
                color = (0, 200, 0)
            else:
                color = (0, 0, 255)

            # Draw bounding box.
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

            # Store detection.
            csv_rows.append({
                "frame": frame_number,
                "timestamp_seconds": round(timestamp, 3),
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

    # Add frame timestamp.
    cv2.putText(
        frame,
        f"Time: {timestamp:.2f}s",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    writer.write(frame)

    # Progress update.
    if frame_number % 50 == 0:

        elapsed = time.perf_counter() - start_time

        print(
            f"Processed {frame_number}/{total_frames} frames | "
            f"Elapsed: {elapsed / 60:.2f} min"
        )


# ============================================================
# 7. RELEASE RESOURCES
# ============================================================

cap.release()
writer.release()

elapsed = time.perf_counter() - start_time


# ============================================================
# 8. SAVE DETECTION CSV
# ============================================================

fieldnames = [
    "frame",
    "timestamp_seconds",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2"
]

with open(OUTPUT_CSV, "w", newline="") as f:

    csv_writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    csv_writer.writeheader()
    csv_writer.writerows(csv_rows)


# ============================================================
# 9. SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("STAGE 1 COMPLETE")
print("=" * 55)

print("Frames processed:", frame_number)
print("Video duration:", round(duration, 2), "seconds")
print("Processing time:", round(elapsed / 60, 2), "minutes")

print("\nDetection counts:")

for name, count in class_counts.items():
    print(f"{name}: {count}")

print("\nTotal detections:", len(csv_rows))

print("\nOutput files:")
print(OUTPUT_VIDEO)
print(OUTPUT_CSV)
