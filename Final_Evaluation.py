
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    r"C:\Users\srina\Downloads\deployment_exports\YOLO26m_FP32_640\YOLO26m_FP32_640.onnx"
)

DATASET_ROOT = Path(
    r"C:\Users\srina\Downloads\VOC2028\SHWD_960_Preprocessed"
)

OUTPUT_DIR = Path(
    r"C:\Users\srina\Downloads\deployment_exports\Final_Evaluation_640"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 640
BASE_CONF = 0.001
NMS_IOU = 0.7

# IoU used to decide whether a prediction matches a ground truth.
MATCH_IOU = 0.5

# Confidence thresholds to investigate.
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)

CLASS_NAMES = {
    0: "hat",
    1: "person"
}

# ============================================================
# 2. CHECK PATHS
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

for split in ["train", "val", "test"]:
    if not (DATASET_ROOT / "images" / split).exists():
        raise FileNotFoundError(f"Missing images/{split}")

    if not (DATASET_ROOT / "labels" / split).exists():
        raise FileNotFoundError(f"Missing labels/{split}")

print("=" * 65)
print("FINAL YOLO26m FP32 640x640 EVALUATION")
print("=" * 65)


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("\nLoading ONNX model...")

model = YOLO(str(MODEL_PATH), task="detect")

print("Model loaded successfully.")
print("Model classes:", model.names)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def get_image_paths(split):

    folder = DATASET_ROOT / "images" / split

    extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    return sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in extensions
    ])


def load_ground_truth(image_path):

    label_path = (
        DATASET_ROOT
        / "labels"
        / image_path.parent.name
        / f"{image_path.stem}.txt"
    )

    if not label_path.exists():
        return np.empty((0, 5), dtype=np.float32)

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    height, width = image.shape[:2]

    objects = []

    with open(label_path, "r") as f:
        for line in f:

            values = line.strip().split()

            if len(values) < 5:
                continue

            cls, xc, yc, bw, bh = map(float, values[:5])

            # YOLO normalized coordinates -> pixel coordinates.
            x1 = (xc - bw / 2) * width
            y1 = (yc - bh / 2) * height
            x2 = (xc + bw / 2) * width
            y2 = (yc + bh / 2) * height

            objects.append([
                int(cls), x1, y1, x2, y2
            ])

    if not objects:
        return np.empty((0, 5), dtype=np.float32)

    return np.array(objects, dtype=np.float32)


def calculate_iou(box, boxes):

    if len(boxes) == 0:
        return np.array([], dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(0, x2 - x1)
        * np.maximum(0, y2 - y1)
    )

    area1 = (
        max(0, box[2] - box[0])
        * max(0, box[3] - box[1])
    )

    area2 = (
        np.maximum(0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    )

    union = area1 + area2 - intersection

    return intersection / np.maximum(union, 1e-9)


# ============================================================
# 5. RUN INFERENCE AND STORE PREDICTIONS
# ============================================================

def collect_predictions(split):

    image_paths = get_image_paths(split)

    predictions = []
    ground_truths = []

    print(f"\nRunning inference on {split.upper()} set...")
    print("Number of images:", len(image_paths))

    start_time = time.perf_counter()

    for i, image_path in enumerate(image_paths):

        # Load ground truth.
        gt = load_ground_truth(image_path)

        # Inference with very low confidence threshold.
        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=BASE_CONF,
            iou=NMS_IOU,
            max_det=300,
            device="cpu",
            verbose=False
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:

            pred = np.empty((0, 6), dtype=np.float32)

        else:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            pred = np.column_stack([
                classes,
                confidences,
                boxes
            ]).astype(np.float32)

        predictions.append(pred)
        ground_truths.append(gt)

        if (i + 1) % 100 == 0 or i + 1 == len(image_paths):
            print(
                f"Processed {i + 1}/{len(image_paths)} images"
            )

    elapsed = time.perf_counter() - start_time

    print(
        f"{split.upper()} inference completed in "
        f"{elapsed / 60:.2f} minutes."
    )

    return predictions, ground_truths


# ============================================================
# 6. CALCULATE TP, FP, FN FOR EACH CLASS
# ============================================================

def calculate_class_metrics(
    predictions,
    ground_truths,
    class_id,
    confidence_threshold
):

    TP = 0
    FP = 0
    FN = 0

    for pred, gt in zip(predictions, ground_truths):

        # Keep only predictions belonging to this class
        # and above the confidence threshold.
        class_preds = pred[
            (pred[:, 0] == class_id)
            & (pred[:, 1] >= confidence_threshold)
        ]

        # Keep only ground truths belonging to this class.
        class_gt = gt[
            gt[:, 0] == class_id
        ]

        # Sort predictions from highest confidence to lowest.
        if len(class_preds) > 0:
            class_preds = class_preds[
                np.argsort(-class_preds[:, 1])
            ]

        matched_gt = set()

        for prediction in class_preds:

            pred_box = prediction[2:6]

            if len(class_gt) == 0:
                FP += 1
                continue

            ious = calculate_iou(
                pred_box,
                class_gt[:, 1:5]
            )

            # Find the highest-IoU unmatched ground truth.
            order = np.argsort(-ious)

            match_found = False

            for gt_index in order:

                if ious[gt_index] < MATCH_IOU:
                    break

                if gt_index not in matched_gt:

                    matched_gt.add(gt_index)
                    TP += 1
                    match_found = True
                    break

            if not match_found:
                FP += 1

        FN += len(class_gt) - len(matched_gt)

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0

    recall = TP / (TP + FN) if TP + FN > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# ============================================================
# 7. VALIDATION SET: CONFIDENCE THRESHOLD SWEEP
# ============================================================

val_predictions, val_ground_truths = collect_predictions("val")

threshold_rows = []
best_thresholds = {}

print("\n" + "=" * 65)
print("VALIDATION: CONFIDENCE THRESHOLD SWEEP")
print("=" * 65)

for class_id, class_name in CLASS_NAMES.items():

    print(f"\nCLASS: {class_name}")
    print("-" * 65)

    best_f1 = -1
    best_threshold = None

    for threshold in THRESHOLDS:

        metrics = calculate_class_metrics(
            val_predictions,
            val_ground_truths,
            class_id,
            threshold
        )

        row = {
            "Class": class_name,
            "Threshold": float(threshold),
            **metrics
        }

        threshold_rows.append(row)

        print(
            f"Conf={threshold:.2f} | "
            f"P={metrics['Precision']:.4f} | "
            f"R={metrics['Recall']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"TP={metrics['TP']} | "
            f"FP={metrics['FP']} | "
            f"FN={metrics['FN']}"
        )

        # Select the threshold with maximum F1.
        # On ties, prefer the higher threshold.
        if (
            metrics["F1"] > best_f1
            or (
                np.isclose(metrics["F1"], best_f1)
                and (
                    best_threshold is None
                    or threshold > best_threshold
                )
            )
        ):

            best_f1 = metrics["F1"]
            best_threshold = float(threshold)

    best_thresholds[class_id] = best_threshold

    print(
        f"\nBEST {class_name.upper()} THRESHOLD: "
        f"{best_threshold:.2f}"
    )

    print(f"Best validation F1: {best_f1:.4f}")


# Save threshold sweep.
sweep_csv = OUTPUT_DIR / "validation_threshold_sweep.csv"

with open(sweep_csv, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Class", "Threshold",
            "TP", "FP", "FN",
            "Precision", "Recall", "F1"
        ]
    )

    writer.writeheader()
    writer.writerows(threshold_rows)


# ============================================================
# 8. TEST SET: FINAL PRECISION, RECALL AND F1
# ============================================================

test_predictions, test_ground_truths = collect_predictions("test")

print("\n" + "=" * 65)
print("FINAL TEST RESULTS AT SELECTED THRESHOLDS")
print("=" * 65)

final_rows = []

for class_id, class_name in CLASS_NAMES.items():

    threshold = best_thresholds[class_id]

    metrics = calculate_class_metrics(
        test_predictions,
        test_ground_truths,
        class_id,
        threshold
    )

    row = {
        "Class": class_name,
        "Selected_threshold": threshold,
        **metrics
    }

    final_rows.append(row)

    print(f"\nCLASS: {class_name.upper()}")
    print(f"Selected confidence threshold: {threshold:.2f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1 score:  {metrics['F1']:.4f}")
    print(f"TP: {metrics['TP']}")
    print(f"FP: {metrics['FP']}")
    print(f"FN: {metrics['FN']}")


# ============================================================
# 9. TEST SET: mAP EVALUATION
# ============================================================

print("\n" + "=" * 65)
print("CALCULATING TEST-SET mAP")
print("=" * 65)

DATA_YAML = OUTPUT_DIR / "data_local.yaml"

DATA_YAML.write_text(
    f"""path: {DATASET_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test
names:
  0: hat
  1: person
""",
    encoding="utf-8"
)

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=IMAGE_SIZE,
    batch=1,
    device="cpu",
    workers=0,
    conf=BASE_CONF,
    iou=NMS_IOU,
    plots=False,
    verbose=True
)

overall_map50 = float(metrics.box.map50)
overall_map5095 = float(metrics.box.map)

# Per-class AP values.
ap_matrix = np.asarray(metrics.box.ap)
ap_class_ids = np.asarray(metrics.box.ap_class_index)

per_class_ap = {}

for index, class_id in enumerate(ap_class_ids):

    class_id = int(class_id)

    per_class_ap[class_id] = {
        "AP50": float(ap_matrix[index, 0]),
        "AP50_95": float(np.mean(ap_matrix[index]))
    }

print("\nOVERALL TEST mAP")
print(f"mAP50:    {overall_map50:.4f}")
print(f"mAP50-95: {overall_map5095:.4f}")


# ============================================================
# 10. COMBINE FINAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("FINAL CLASS-WISE RESULTS")
print("=" * 65)

for row in final_rows:

    class_id = next(
        k for k, v in CLASS_NAMES.items()
        if v == row["Class"]
    )

    ap = per_class_ap.get(
        class_id,
        {"AP50": 0.0, "AP50_95": 0.0}
    )

    row["AP50"] = ap["AP50"]
    row["AP50_95"] = ap["AP50_95"]

    print(
        f"\n{row['Class'].upper()} | "
        f"Threshold={row['Selected_threshold']:.2f}"
    )

    print(
        f"P={row['Precision']:.4f} | "
        f"R={row['Recall']:.4f} | "
        f"F1={row['F1']:.4f}"
    )

    print(
        f"AP50={row['AP50']:.4f} | "
        f"AP50-95={row['AP50_95']:.4f}"
    )


# ============================================================
# 11. SAVE FINAL CSV AND SUMMARY
# ============================================================

final_csv = OUTPUT_DIR / "final_test_class_metrics.csv"

with open(final_csv, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Class",
            "Selected_threshold",
            "TP", "FP", "FN",
            "Precision", "Recall", "F1",
            "AP50", "AP50_95"
        ]
    )

    writer.writeheader()
    writer.writerows(final_rows)


summary_file = OUTPUT_DIR / "final_summary.txt"

with open(summary_file, "w") as f:

    f.write("YOLO26m FP32 640x640 FINAL EVALUATION\n")
    f.write("=" * 55 + "\n\n")

    f.write("Selected confidence thresholds:\n")

    for class_id, class_name in CLASS_NAMES.items():
        f.write(
            f"{class_name}: {best_thresholds[class_id]:.2f}\n"
        )

    f.write("\nOverall test-set metrics:\n")
    f.write(f"mAP50: {overall_map50:.4f}\n")
    f.write(f"mAP50-95: {overall_map5095:.4f}\n")

    f.write("\nClass-wise test metrics:\n")

    for row in final_rows:

        f.write(f"\n{row['Class']}:\n")

        for key in [
            "Selected_threshold",
            "Precision",
            "Recall",
            "F1",
            "AP50",
            "AP50_95",
            "TP",
            "FP",
            "FN"
        ]:

            f.write(f"{key}: {row[key]}\n")


print("\n" + "=" * 65)
print("EVALUATION COMPLETE")
print("=" * 65)

print("\nFiles saved:")
print(sweep_csv)
print(final_csv)
print(summary_file)
