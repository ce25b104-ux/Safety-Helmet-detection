
import os
import time
import shutil
import platform
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


# ============================================================
# 1. PATHS
# ============================================================

MODEL_PATH = Path(
    r"C:\Users\srina\Downloads\YOLO26m_Contrast_ThinGrid_Online_Fixed_last.pt"
)

DATASET_ROOT = Path(
    r"C:\Users\srina\Downloads\VOC2028\SHWD_960_Preprocessed"
)

OUTPUT_DIR = Path(
    r"C:\Users\srina\Downloads\deployment_exports\YOLO26m_FP32_640"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = OUTPUT_DIR / "YOLO26m_FP32_640.onnx"
DATA_YAML = OUTPUT_DIR / "data_local.yaml"
RESULTS_FILE = OUTPUT_DIR / "benchmark_results.txt"


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = 640
WARMUP_RUNS = 30
BENCHMARK_RUNS = 200

# Set to a fixed number for reproducible CPU benchmarking.
CPU_THREADS = max(1, (os.cpu_count() or 2) // 2)

print("=" * 60)
print("YOLO26m FP32 - 640x640 DEPLOYMENT BENCHMARK")
print("=" * 60)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

if not (DATASET_ROOT / "images" / "test").exists():
    raise FileNotFoundError("SHWD test images folder not found.")

if not (DATASET_ROOT / "labels" / "test").exists():
    raise FileNotFoundError("SHWD test labels folder not found.")


# ============================================================
# 3. CREATE LOCAL DATA YAML
# ============================================================

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


# ============================================================
# 4. EXPORT CHECKPOINT TO FP32 ONNX AT 640
# ============================================================

print("\n[1/4] Exporting YOLO26m to FP32 ONNX at 640...")

model = YOLO(str(MODEL_PATH))

exported_path = model.export(
    format="onnx",
    imgsz=IMAGE_SIZE,
    batch=1,
    device="cpu",
    half=False,
    int8=False,
    dynamic=False,
    simplify=True,
    opset=17
)

exported_path = Path(exported_path)

if not exported_path.exists():
    raise FileNotFoundError("ONNX export did not produce a file.")

if exported_path.resolve() != ONNX_PATH.resolve():
    shutil.copy2(exported_path, ONNX_PATH)

print("Export successful:", ONNX_PATH)


# ============================================================
# 5. LOAD ONNX MODEL
# ============================================================

print("\n[2/4] Loading ONNX model...")

options = ort.SessionOptions()
options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)
options.intra_op_num_threads = CPU_THREADS
options.inter_op_num_threads = 1

session = ort.InferenceSession(
    str(ONNX_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

print("ONNX Runtime loading: SUCCESS")
print("Input name:", input_name)
print("Input shape:", input_shape)
print("Execution provider:", session.get_providers())
print("CPU threads:", CPU_THREADS)


# ============================================================
# 6. PREPARE TEST IMAGES
# ============================================================

print("\n[3/4] Preparing test images...")

test_dir = DATASET_ROOT / "images" / "test"

image_paths = sorted([
    p for p in test_dir.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
])

if not image_paths:
    raise RuntimeError("No test images found.")

# Keep a small fixed sample in RAM to avoid measuring disk I/O.
NUM_BENCHMARK_IMAGES = min(50, len(image_paths))

letterbox = LetterBox(
    new_shape=(IMAGE_SIZE, IMAGE_SIZE),
    auto=False,
    stride=32
)

benchmark_inputs = []

for path in image_paths[:NUM_BENCHMARK_IMAGES]:

    image = cv2.imread(str(path))

    if image is None:
        continue

    # Match YOLO's square letterbox preprocessing.
    image = letterbox(image=image)

    # BGR -> RGB, HWC -> CHW, normalize to [0, 1].
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(
        image[None, ...],
        dtype=np.float32
    ) / 255.0

    benchmark_inputs.append(image)

if not benchmark_inputs:
    raise RuntimeError("Could not prepare any benchmark images.")

print("Images prepared:", len(benchmark_inputs))


# ============================================================
# 7. LATENCY AND THROUGHPUT BENCHMARK
# ============================================================

print("\nRunning warm-up...")

for i in range(WARMUP_RUNS):
    session.run(None, {input_name: benchmark_inputs[i % len(benchmark_inputs)]})

print("Warm-up completed.")

print("\nRunning timed inference...")

latencies = []

for i in range(BENCHMARK_RUNS):

    input_tensor = benchmark_inputs[i % len(benchmark_inputs)]

    start = time.perf_counter()

    session.run(None, {input_name: input_tensor})

    end = time.perf_counter()

    latencies.append((end - start) * 1000)

avg_latency = float(np.mean(latencies))
median_latency = float(np.median(latencies))
p95_latency = float(np.percentile(latencies, 95))
throughput = 1000.0 / avg_latency

model_size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)


# ============================================================
# 8. TEST-SET MAP EVALUATION
# ============================================================

print("\n[4/4] Evaluating ONNX model on SHWD test set...")

map50 = None
map50_95 = None
map_error = None

try:
    onnx_model = YOLO(str(ONNX_PATH), task="detect")

    metrics = onnx_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=1,
        device="cpu",
        workers=0,
        plots=False,
        verbose=True
    )

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

except Exception as e:
    map_error = repr(e)
    print("\nWARNING: mAP evaluation failed.")
    print(map_error)


# ============================================================
# 9. PRINT AND SAVE RESULTS
# ============================================================

lines = [
    "=" * 60,
    "YOLO26m FP32 - 640x640 BENCHMARK RESULTS",
    "=" * 60,
    f"Checkpoint: {MODEL_PATH}",
    f"ONNX model: {ONNX_PATH}",
    f"Input resolution: {IMAGE_SIZE}x{IMAGE_SIZE}",
    "Precision: FP32",
    "Runtime: ONNX Runtime CPU",
    f"CPU: {platform.processor()}",
    f"Logical CPU cores: {os.cpu_count()}",
    f"ORT intra-op threads: {CPU_THREADS}",
    "",
    f"Model size: {model_size_mb:.2f} MB",
    f"Average latency: {avg_latency:.3f} ms",
    f"Median latency: {median_latency:.3f} ms",
    f"P95 latency: {p95_latency:.3f} ms",
    f"Throughput: {throughput:.2f} images/second",
    f"Warm-up runs: {WARMUP_RUNS}",
    f"Timed runs: {BENCHMARK_RUNS}",
    f"Benchmark images: {len(benchmark_inputs)}",
    "",
    f"mAP50: {map50 if map50 is not None else 'FAILED'}",
    f"mAP50-95: {map50_95 if map50_95 is not None else 'FAILED'}",
]

if map_error:
    lines.append(f"mAP evaluation error: {map_error}")

report = "\n".join(lines)

print("\n" + report)

RESULTS_FILE.write_text(report, encoding="utf-8")

print("\nResults saved to:", RESULTS_FILE)
print("\nBENCHMARK COMPLETE")
