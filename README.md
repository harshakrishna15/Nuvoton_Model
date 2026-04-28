# Nuvoton Model

Person detection and count-estimation experiments for overhead elevator-style imagery.

This repo has two model paths:

- **Baseline:** grayscale Faster R-CNN MobileNetV3 in `src/elevator_counter/`
- **Nuvoton YOLO:** a 192x192 YOLOv8 workflow using the local `ML_YOLO/yolov8_ultralytics` fork

Datasets, generated exports, checkpoints, and caches are intentionally ignored by Git. A fresh checkout needs the dataset setup in [Dataset Setup](docs/DATASETS.md) before training.

## Documentation

Start here:

1. [Fresh Setup](docs/SETUP.md): clone, Python environment, dependency install, and GPU check
2. [Dataset Setup](docs/DATASETS.md): source links, expected local folders, and validation commands
3. [Training](docs/TRAINING.md): baseline training, Nuvoton YOLO prep, and GPU smoke test
4. [Evaluation](docs/EVALUATION.md): baseline and YOLO evaluation commands and outputs
5. [Troubleshooting](docs/TROUBLESHOOTING.md): common setup, CUDA, path, and dataset issues

## Repository Layout

- `src/elevator_counter/`: reusable dataset, model, training, and evaluation helpers
- `scripts/build_splits.py`: creates deterministic train/val/test splits for the overhead dataset
- `scripts/inspect_dataset.py`: sanity-checks the overhead dataset loader and split manifest
- `scripts/export_label_previews.py`: exports annotated sample images for quick label review
- `scripts/train_baseline.py`: trains the grayscale Faster R-CNN baseline
- `scripts/evaluate_baseline.py`: tunes a detection threshold on validation and reports test count metrics
- `scripts/prepare_nuvoton_yolo_dataset.py`: merges datasets into a single-class YOLO dataset for Nuvoton training
- `scripts/train_nuvoton_yolo.sh`: optional Bash wrapper for YOLO training
- `scripts/evaluate_nuvoton_yolo.py`: produces count metrics, plots, CSVs, and worst-case overlays for a YOLO checkpoint
- `ML_YOLO/`: local Nuvoton/OpenNuvoton model code, including the editable `yolov8_ultralytics` package

## Quick Start

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\build_splits.py --dataset-root overhead-person-detection
python scripts\prepare_nuvoton_yolo_dataset.py --force
```

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/build_splits.py --dataset-root overhead-person-detection
python scripts/prepare_nuvoton_yolo_dataset.py --force
```

Download the two datasets listed in [Dataset Setup](docs/DATASETS.md) before running the split and preparation commands.

Run the 1-epoch GPU smoke test from [Training](docs/TRAINING.md) to verify the full pipeline.