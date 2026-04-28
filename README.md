# Nuvoton Model

This repository contains experiments and tooling for person detection and count estimation from overhead elevator-style imagery. It has two training paths:

- a PyTorch baseline built around a grayscale Faster R-CNN model
- a Nuvoton-focused YOLOv8 workflow using the local `ML_YOLO/yolov8_ultralytics` fork and 192x192-compatible configs

The datasets and generated training outputs are intentionally not committed to Git. A fresh checkout needs the dataset downloads described below before training can run.

## What Is In This Repo

- `src/elevator_counter/`: reusable dataset, model, training, and evaluation helpers
- `scripts/build_splits.py`: creates deterministic train/val/test splits for the overhead dataset
- `scripts/inspect_dataset.py`: sanity-checks the overhead dataset loader and split manifest
- `scripts/export_label_previews.py`: exports annotated sample images for quick label review
- `scripts/train_baseline.py`: trains the grayscale Faster R-CNN baseline
- `scripts/evaluate_baseline.py`: tunes a detection threshold on validation and reports test count metrics
- `scripts/prepare_nuvoton_yolo_dataset.py`: merges datasets into a single-class YOLO dataset for Nuvoton training
- `scripts/train_nuvoton_yolo.sh`: optional Bash wrapper for YOLO training
- `scripts/evaluate_nuvoton_yolo.py`: produces count metrics, plots, CSVs, and worst-case overlays for a YOLO checkpoint
- `ML_YOLO/`: local copy of the Nuvoton/OpenNuvoton model code, including the editable `yolov8_ultralytics` package

## Fresh Setup

These steps assume Windows PowerShell, Python 3.12, an NVIDIA GPU with a working driver, and enough disk space for the datasets, generated YOLO export, checkpoints, and caches.

### 1. Clone And Enter The Repo

```powershell
git clone <repo-url> Nuvoton_Model
cd Nuvoton_Model
```

If you already have the repo, start from the repo root:

```powershell
cd "C:\Users\Harsha Krishnaswamy\Desktop\Development\Nuvoton_Model"
```

### 2. Create The Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs CUDA-enabled PyTorch wheels from the official PyTorch index and installs the local Ultralytics fork from `ML_YOLO/yolov8_ultralytics` in editable mode.

Verify the GPU is visible to PyTorch:

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

For GPU training, this should print `cuda: True` and the NVIDIA GPU name.

### 3. Download The Training Datasets

This project trains on two datasets:

1. [bdanko/overhead-person-detection](https://huggingface.co/datasets/bdanko/overhead-person-detection)
2. [Passenger Counter](https://universe.roboflow.com/passenger-counter-project/passenger-counter)

Place them in the repo root with exactly these local folder names:

```text
Nuvoton_Model/
  overhead-person-detection/
    data/
      train-00000-of-00001.parquet
    README.md
    splits.json                  # optional; regenerated if missing

  Passenger Counter.yolov8/
    train/
      images/
      labels/
    data.yaml
    README.roboflow.txt
```

The datasets are ignored by Git. Do not commit `overhead-person-detection/`, `Passenger Counter.yolov8/`, `prepared_datasets/`, `runs/`, or model weights.

Quick local validation:

```powershell
Test-Path .\overhead-person-detection\data
Test-Path '.\Passenger Counter.yolov8\train\images'
Test-Path '.\Passenger Counter.yolov8\train\labels'
Get-ChildItem .\overhead-person-detection\data -Filter *.parquet
Get-ChildItem '.\Passenger Counter.yolov8\train\images' -File | Measure-Object
Get-ChildItem '.\Passenger Counter.yolov8\train\labels' -File | Measure-Object
```

The Passenger Counter export used here has 3,508 images and 3,508 label files.

### 4. Build Splits And Prepare The YOLO Dataset

```powershell
python scripts\build_splits.py --dataset-root overhead-person-detection
python scripts\prepare_nuvoton_yolo_dataset.py --force
```

This creates the merged one-class YOLO dataset at:

```text
prepared_datasets/nuvoton_people_v1/
  dataset.yaml
  prep_summary.json
  train/
  val/
  test/
```

Key behavior:

- the overhead dataset becomes train/val/test using the deterministic split manifest
- Passenger Counter contributes to train/val only
- all labels are normalized to a single class: `person`
- overhead images are exported as grayscale PNGs

### 5. Run A 1-Epoch GPU Smoke Test

Run this from the repo root in PowerShell:

```powershell
$repo = (Get-Location).Path
$env:MPLCONFIGDIR = "$repo\.matplotlib"
$env:YOLO_CONFIG_DIR = "$repo\.ultralytics"
$env:TEMP = "$env:LOCALAPPDATA\Temp\nuvoton_model"
$env:TMP = "$env:LOCALAPPDATA\Temp\nuvoton_model"
$env:TMPDIR = "$env:LOCALAPPDATA\Temp\nuvoton_model"
New-Item -ItemType Directory -Force $env:TEMP | Out-Null

cd .\ML_YOLO\yolov8_ultralytics

python dg_train.py `
  --model-cfg ultralytics/cfg/models/v8/relu6-yolov8.yaml `
  --data "$repo\prepared_datasets\nuvoton_people_v1\dataset.yaml" `
  --imgsz 192 `
  --weights yolov8n.pt `
  --epochs 1 `
  --patience 30 `
  --device 0 `
  --workers 0 `
  --save-period 1 `
  --project "$repo\runs\nuvoton_yolo" `
  --name smoke_1epoch_gpu
```

Important training notes:

- use an absolute `--data` path so Ultralytics does not resolve relative dataset paths against a global `datasets_dir`
- `--device 0` uses the first NVIDIA GPU; use `--device cpu` only for CPU testing
- the default batch size is `64`; add `--batch 16` or `--batch 8` if CUDA runs out of memory
- for a full run, increase `--epochs` and change `--name`

Expected smoke-test outputs:

```text
runs/nuvoton_yolo/smoke_1epoch_gpu/
  weights/
    last.pt
    best.pt
```

A Bash-compatible wrapper is also available if you are using Git Bash or WSL:

```bash
NUVOTON_DEVICE=0 bash scripts/train_nuvoton_yolo.sh yolov8n.pt 200 192 nuvoton_people_v1_relu6_192_e200
```

## Baseline Workflow

The baseline path uses a grayscale Faster R-CNN MobileNetV3 model adapted for 1-channel 192x192 inputs.

```powershell
python scripts\build_splits.py --dataset-root overhead-person-detection
python scripts\inspect_dataset.py --dataset-root overhead-person-detection --split train
python scripts\train_baseline.py `
  --dataset-root overhead-person-detection `
  --output-dir runs\baseline_frcnn `
  --epochs 10 `
  --batch-size 4 `
  --image-size 192 `
  --device auto
```

Expected baseline outputs:

- `runs/baseline_frcnn/run_config.json`
- `runs/baseline_frcnn/metrics.jsonl`
- `runs/baseline_frcnn/last.pt`
- `runs/baseline_frcnn/best.pt`

Evaluate the baseline checkpoint:

```powershell
python scripts\evaluate_baseline.py `
  --dataset-root overhead-person-detection `
  --checkpoint runs\baseline_frcnn\best.pt `
  --output runs\baseline_frcnn\eval_summary.json
```

## YOLO Evaluation

```powershell
python scripts\evaluate_nuvoton_yolo.py `
  --weights runs\nuvoton_yolo\<run-name>\weights\best.pt `
  --data prepared_datasets\nuvoton_people_v1\dataset.yaml `
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

- generated artifacts such as `.venv/`, `.hf-cache/`, `.ultralytics/`, `runs/`, `prepared_datasets/`, and preview images should not be committed
- model weights such as `*.pt`, `*.weights`, `*.onnx`, `*.engine`, and `*.tflite` should not be committed
- the baseline dataset loader converts images to grayscale and filters malformed boxes
- the baseline model adapts the first convolution of MobileNetV3 to one input channel
- for Nuvoton training, pass an absolute `--data` path to avoid Ultralytics resolving relative dataset paths against a global `datasets_dir` setting
