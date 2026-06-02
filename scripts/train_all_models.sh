#!/usr/bin/env bash
# End-to-end pipeline: prepare merged dataset (overhead + Passenger Counter +
# all *.coco datasets), train YOLOv8 + FOMO, and emit val/test metrics.
#
# Environment overrides (all optional):
#   EPOCHS_YOLO       (default 200)   YOLOv8 training epochs
#   EPOCHS_FOMO       (default 60)    FOMO training epochs
#   IMGSZ             (default 192)   image size for both models
#   GRID_SIZE         (default 6)     FOMO grid size
#   BATCH_FOMO        (default 32)    FOMO batch size
#   YOLO_WEIGHTS      (default yolov8n.pt)
#   DEVICE            (default cpu)   training device for YOLOv8 (cpu/cuda)
#   FOMO_DEVICE       (default auto)  training device for FOMO
#   SKIP_PREPARE      set to 1 to skip dataset rebuild
#   SKIP_YOLO         set to 1 to skip YOLOv8 training
#   SKIP_FOMO         set to 1 to skip FOMO training

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EPOCHS_YOLO="${EPOCHS_YOLO:-200}"
EPOCHS_FOMO="${EPOCHS_FOMO:-60}"
IMGSZ="${IMGSZ:-192}"
GRID_SIZE="${GRID_SIZE:-6}"
BATCH_FOMO="${BATCH_FOMO:-32}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolov8n.pt}"
DEVICE="${DEVICE:-cpu}"
FOMO_DEVICE="${FOMO_DEVICE:-auto}"

DATASET_ROOT="$ROOT/prepared_datasets/nuvoton_people_v1"
DATASET_YAML="$DATASET_ROOT/dataset.yaml"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
elif [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then
    PYTHON="$ROOT/.venv/Scripts/python.exe"
else
    PYTHON="${PYTHON_BIN:-python}"
fi

echo "=========================================="
echo "  Configuration"
echo "=========================================="
echo "PYTHON       = $PYTHON"
echo "DATASET_ROOT = $DATASET_ROOT"
echo "IMGSZ        = $IMGSZ"
echo "EPOCHS_YOLO  = $EPOCHS_YOLO"
echo "EPOCHS_FOMO  = $EPOCHS_FOMO"
echo "GRID_SIZE    = $GRID_SIZE"
echo "DEVICE(yolo) = $DEVICE"
echo "DEVICE(fomo) = $FOMO_DEVICE"

# ---------------- 1. Prepare merged dataset ----------------
if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
    echo ""
    echo "=========================================="
    echo "  Step 1/3: Prepare merged dataset"
    echo "=========================================="
    "$PYTHON" scripts/prepare_nuvoton_yolo_dataset.py --force \
        2>&1 | tee "$LOG_DIR/prepare_dataset.log"
else
    echo "[skip] dataset preparation"
fi

# Quick sanity: count merged split sizes
"$PYTHON" - <<PY
from pathlib import Path
root = Path("${DATASET_ROOT}")
for split in ("train", "val", "test"):
    imgs = list((root / split / "images").glob("*"))
    print(f"[merged] {split}: {len(imgs)} images")
PY

# ---------------- 2. Train YOLOv8 ----------------
if [[ "${SKIP_YOLO:-0}" != "1" ]]; then
    echo ""
    echo "=========================================="
    echo "  Step 2/3: Train YOLOv8 (Nuvoton config)"
    echo "=========================================="
    NUVOTON_DEVICE="$DEVICE" \
        bash scripts/train_nuvoton_yolo.sh "$YOLO_WEIGHTS" "$EPOCHS_YOLO" "$IMGSZ" \
        "nuvoton_people_v2_relu6_${IMGSZ}_e${EPOCHS_YOLO}" \
        2>&1 | tee "$LOG_DIR/train_yolo.log"
else
    echo "[skip] YOLOv8 training"
fi

# ---------------- 3. Train FOMO on merged dataset ----------------
if [[ "${SKIP_FOMO:-0}" != "1" ]]; then
    echo ""
    echo "=========================================="
    echo "  Step 3/3: Train FOMO (merged dataset)"
    echo "=========================================="
    "$PYTHON" scripts/train_fomo_merged.py \
        --dataset-root "$DATASET_ROOT" \
        --output-dir "$ROOT/runs/fomo_merged" \
        --epochs "$EPOCHS_FOMO" \
        --batch-size "$BATCH_FOMO" \
        --image-size "$IMGSZ" \
        --grid-size "$GRID_SIZE" \
        --device "$FOMO_DEVICE" \
        --brightness-jitter 0.2 --contrast-jitter 0.2 \
        --pos-weight 20 --count-aux-weight 0.1 \
        2>&1 | tee "$LOG_DIR/train_fomo.log"
else
    echo "[skip] FOMO training"
fi

# ---------------- Summary ----------------
echo ""
echo "=========================================="
echo "  Final metrics"
echo "=========================================="

YOLO_RUN_DIR="$ROOT/runs/nuvoton_yolo/nuvoton_people_v2_relu6_${IMGSZ}_e${EPOCHS_YOLO}"
if [[ -f "$YOLO_RUN_DIR/results.csv" ]]; then
    echo "[YOLOv8] last 3 epochs:"
    tail -n 3 "$YOLO_RUN_DIR/results.csv"
fi
if [[ -f "$ROOT/runs/fomo_merged/summary.json" ]]; then
    echo ""
    echo "[FOMO] summary:"
    cat "$ROOT/runs/fomo_merged/summary.json"
fi
echo ""
echo "Logs in $LOG_DIR"
