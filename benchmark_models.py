
from pathlib import Path
import time
import os
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort

from ultralytics import YOLO

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(
    r"C:\Users\srina\Downloads\VOC2028\SHWD_960_Preprocessed"
)

EXPORT_DIR = Path(
    r"C:\Users\srina\Downloads\deployment_exports"
)

MODEL_PATHS = {
    "FP32": EXPORT_DIR / "YOLO26m_FP32.onnx",
    "FP16": EXPORT_DIR / "YOLO26m_FP16.onnx",
    "INT8": EXPORT_DIR / "YOLO26m_INT8.onnx"
}

TEST_IMAGES = ROOT / "images" / "test"

# Create a local YAML because the original Colab YAML
# may contain an outdated /content path.
DATA_YAML = ROOT / "data_local.yaml"

DATA_YAML.write_text(
    f"""path: {ROOT.as_posix()}
train: images/train
val: images/val
test: images/test
names:
  0: hat
  1: person
""",
    encoding="utf-8"
)

# ============================================================
# 2. SETTINGS
# ============================================================

IMG_SIZE = 960
WARMUP_RUNS = 30
TIMED_RUNS = 200

# ============================================================
# 3. LOAD TEST IMAGES
# ============================================================

image_paths = []

for ext in ["*.jpg", "*.jpeg", "*.png"]:
    image_paths.extend(TEST_IMAGES.glob(ext))

image_paths = sorted(image_paths)

assert image_paths, "No test images found!"

print("Test images:", len(image_paths))

# ============================================================
# 4. PREPROCESSING
# ============================================================

def preprocess(path):

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32) / 255.0

    image = np.transpose(image, (2, 0, 1))

    image = np.expand_dims(image, axis=0)

    return np.ascontiguousarray(image)


# Prepare only a small set of images for repeated timing.
# This avoids keeping the entire test set in RAM.
timing_paths = image_paths[:min(20, len(image_paths))]
timing_inputs = [preprocess(p) for p in timing_paths]

# ============================================================
# 5. LATENCY AND THROUGHPUT
# ============================================================

def benchmark_onnx(model_path):

    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = max(1, os.cpu_count() or 1)
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    # Warm-up
    for i in range(WARMUP_RUNS):
        x = timing_inputs[i % len(timing_inputs)]
        session.run(None, {input_name: x})

    # Timed batch-1 inference
    latencies = []

    for i in range(TIMED_RUNS):

        x = timing_inputs[i % len(timing_inputs)]

        start = time.perf_counter()

        session.run(None, {input_name: x})

        end = time.perf_counter()

        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)

    average_latency = float(np.mean(latencies))
    median_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    throughput = 1000.0 / average_latency

    return {
        "Latency (ms)": average_latency,
        "Median latency (ms)": median_latency,
        "P95 latency (ms)": p95_latency,
        "Throughput (img/s)": throughput
    }


# ============================================================
# 6. MAP EVALUATION
# ============================================================

def evaluate_map(model_path):

    yolo_model = YOLO(str(model_path), task="detect")

    metrics = yolo_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMG_SIZE,
        batch=1,
        device="cpu",
        workers=0,
        plots=False,
        verbose=False
    )

    return {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map)
    }


# ============================================================
# 7. RUN ALL BENCHMARKS
# ============================================================

results = []

for name, model_path in MODEL_PATHS.items():

    print("\n" + "=" * 55)
    print("BENCHMARKING:", name)
    print("=" * 55)

    if not model_path.exists():
        print("Model file missing. Skipping.")
        continue

    size_mb = model_path.stat().st_size / (1024 ** 2)

    try:
        timing = benchmark_onnx(model_path)

        print("Latency:", timing["Latency (ms)"], "ms")
        print("Throughput:", timing["Throughput (img/s)"], "img/s")

    except Exception as e:
        print("INFERENCE FAILED:", e)
        continue

    try:
        accuracy = evaluate_map(model_path)

        print("mAP50:", accuracy["mAP50"])
        print("mAP50-95:", accuracy["mAP50-95"])

    except Exception as e:
        print("MAP EVALUATION FAILED:", e)
        accuracy = {
            "mAP50": np.nan,
            "mAP50-95": np.nan
        }

    results.append({
        "Variant": name,
        "Model size (MB)": size_mb,
        **timing,
        **accuracy
    })


# ============================================================
# 8. FINAL COMPARISON TABLE
# ============================================================

df = pd.DataFrame(results)

if len(df) == 0:
    raise RuntimeError("No model completed inference benchmarking.")

print("\n\nFINAL COMPARISON TABLE")

print(df.round(4).to_string(index=False))

output_csv = EXPORT_DIR / "deployment_comparison.csv"

df.to_csv(output_csv, index=False)

print("\nResults saved to:", output_csv)
