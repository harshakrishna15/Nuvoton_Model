# Nuvoton Model

This repository contains experiments and tooling for person detection and count estimation from overhead elevator-style imagery, with two main tracks:

- a PyTorch baseline built around a grayscale Faster R-CNN model
- a Nuvoton-focused YOLOv8 workflow using a local Ultralytics fork and 192x192-compatible configs

The repo also includes local dataset snapshots, conversion scripts, and evaluation utilities for count-focused diagnostics.

## What is in this repo

The project combines two person-detection datasets:

- `overhead-person-detection/`: a parquet-backed Hugging Face-style snapshot of overhead images with bounding boxes
- `Passenger Counter.yolov8/`: a Roboflow YOLO export with 3,508 images

Those datasets are used in two ways:

- baseline experiments train directly from the parquet dataset through the Python package in `src/elevator_counter/`
- Nuvoton experiments export a merged one-class YOLO dataset, then train a custom YOLOv8 variant from `ML_YOLO/yolov8_ultralytics`

## Repository Layout

- `src/elevator_counter/`: reusable dataset, model, training, and evaluation helpers
- `scripts/build_splits.py`: creates deterministic train/val/test splits for the overhead dataset
- `scripts/inspect_dataset.py`: sanity-checks the overhead dataset loader and split manifest
- `scripts/export_label_previews.py`: exports annotated sample images for quick label review
- `scripts/train_baseline.py`: trains the grayscale Faster R-CNN baseline
- `scripts/evaluate_baseline.py`: tunes a detection threshold on validation and reports test count metrics
- `scripts/prepare_nuvoton_yolo_dataset.py`: merges datasets into a single-class YOLO dataset for Nuvoton training
- `scripts/train_nuvoton_yolo.sh`: launches YOLO training through the local Ultralytics fork
- `scripts/evaluate_nuvoton_yolo.py`: produces count metrics, plots, CSVs, and worst-case overlays for a YOLO checkpoint
- `ML_YOLO/`: local copy of the Nuvoton/OpenNuvoton model code, including the editable `yolov8_ultralytics` package
- `datasets/`: extra local dataset storage

## Environment Setup

This repo expects Python plus the dependencies in `requirements.txt`. The editable install points at the local Ultralytics fork.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Important notes:

- `requirements.txt` installs CUDA-enabled PyTorch wheels from the official PyTorch index
- `-e ./ML_YOLO/yolov8_ultralytics` means the local Ultralytics fork is part of the environment
- the scripts default to local repo paths, so they work best when run from the repository root

## Data Overview

### 1. Overhead Person Detection

- location: `overhead-person-detection/`
- format: parquet files under `overhead-person-detection/data/`
- current manifest: `overhead-person-detection/splits.json`
- size in the included dataset card: 13,448 examples

The loader in `src/elevator_counter/data.py` reads these parquet files through Hugging Face Datasets, converts images to grayscale tensors, and builds detection targets for PyTorch.

### 2. Passenger Counter

- location: `Passenger Counter.yolov8/`
- format: YOLO labels and images from Roboflow
- included export note: 3,508 images

During Nuvoton data prep, these labels are normalized to a single `person` class and merged with the overhead dataset.

## Baseline Workflow

The baseline path uses a grayscale Faster R-CNN MobileNetV3 model adapted for 1-channel 192x192 inputs.

### 1. Build or refresh the split manifest

```powershell
python scripts/build_splits.py --dataset-root overhead-person-detection
```

This creates `overhead-person-detection/splits.json` if it does not already exist.

### 2. Inspect the dataset

```powershell
python scripts/inspect_dataset.py --dataset-root overhead-person-detection --split train
```

Optional label previews:

```powershell
python scripts/export_label_previews.py --dataset-root overhead-person-detection --split val --num-samples 25
```

### 3. Train the baseline model

```powershell
python scripts/train_baseline.py `
  --dataset-root overhead-person-detection `
  --output-dir runs/baseline_frcnn `
  --epochs 10 `
  --batch-size 4 `
  --image-size 192 `
  --device auto
```

Expected outputs include:

- `runs/baseline_frcnn/run_config.json`
- `runs/baseline_frcnn/metrics.jsonl`
- `runs/baseline_frcnn/last.pt`
- `runs/baseline_frcnn/best.pt`

### 4. Evaluate the baseline checkpoint

```powershell
python scripts/evaluate_baseline.py `
  --dataset-root overhead-person-detection `
  --checkpoint runs/baseline_frcnn/best.pt `
  --output runs/baseline_frcnn/eval_summary.json
```

This script:

- sweeps score thresholds on the validation split
- selects the best threshold by count MAE, bias, and exact-match rate
- reports overall and bucketed test-set counting metrics

## Nuvoton YOLO Workflow

The Nuvoton path exports a merged one-class YOLO dataset and trains with the local Ultralytics fork using `relu6-yolov8.yaml` and `imgsz=192`.

### 1. Prepare the merged YOLO dataset

```powershell
python scripts/prepare_nuvoton_yolo_dataset.py --force
```

By default this creates:

- `prepared_datasets/nuvoton_people_v1/dataset.yaml`
- `prepared_datasets/nuvoton_people_v1/prep_summary.json`
- `prepared_datasets/nuvoton_people_v1/train/`
- `prepared_datasets/nuvoton_people_v1/val/`
- `prepared_datasets/nuvoton_people_v1/test/`

Key behavior:

- overhead dataset becomes train/val/test using the deterministic split manifest
- Passenger Counter contributes to train/val only
- all labels are normalized to a single class: `person`
- overhead images are exported as grayscale PNGs

### 2. Train the Nuvoton YOLO model

On Windows, run the shell script from Git Bash, WSL, or adapt the same command for PowerShell:

```bash
bash scripts/train_nuvoton_yolo.sh yolov8n.pt 200 192 nuvoton_people_v1_relu6_192_e200
```

The script uses:

- dataset YAML: `prepared_datasets/nuvoton_people_v1/dataset.yaml`
- model config: `ultralytics/cfg/models/v8/relu6-yolov8.yaml`
- project dir: `runs/nuvoton_yolo`
- default device: `cpu` unless overridden with `NUVOTON_DEVICE`

Useful environment overrides:

- `NUVOTON_YOLO_REPO`
- `NUVOTON_RUNS_DIR`
- `NUVOTON_DEVICE`
- `NUVOTON_WORKERS`
- `NUVOTON_PATIENCE`
- `PYTHON_BIN`

### 3. Evaluate a YOLO checkpoint

```powershell
python scripts/evaluate_nuvoton_yolo.py `
  --weights runs/nuvoton_yolo/<run-name>/weights/best.pt `
  --data prepared_datasets/nuvoton_people_v1/dataset.yaml `
  --split val `
  --imgsz 192 `
  --conf 0.25
```

The evaluation report includes:

- `summary.json`
- `per_image_counts.csv`
- `count_scatter.png`
- `count_error_hist.png`
- `worst_cases/` with GT vs prediction overlays

## Metrics Philosophy

This repo is optimized around counting quality, not only box-level detection quality. The evaluation helpers in `src/elevator_counter/evaluation.py` report metrics such as:

- count MAE
- count RMSE
- count bias
- exact-match rate
- within-one rate
- overcount and undercount rate
- empty-scene false positive rate
- bucketed metrics for `0`, `1`, `2`, `3-4`, and `5+` people

## Development Notes

- the baseline dataset loader converts images to grayscale and filters malformed boxes
- the baseline model adapts the first convolution of MobileNetV3 to one input channel
- the Nuvoton dataset YAML intentionally omits a `path:` field so the local Ultralytics fork resolves paths relative to the YAML file location
- generated artifacts such as `runs/`, `prepared_datasets/`, and preview images may not be checked in

## Quick Start

If you just want the shortest path to a working experiment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/build_splits.py
python scripts/train_baseline.py --epochs 10 --output-dir runs/baseline_frcnn
python scripts/evaluate_baseline.py --checkpoint runs/baseline_frcnn/best.pt
```

For the Nuvoton path:

```powershell
python scripts/prepare_nuvoton_yolo_dataset.py --force
```

Then train with `scripts/train_nuvoton_yolo.sh` from a Bash-compatible shell.
