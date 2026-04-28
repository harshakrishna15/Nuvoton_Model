# Training

Run all commands from the repository root unless a step explicitly changes directories.

## Nuvoton YOLO Training

The Nuvoton path uses the merged dataset created by `scripts/prepare_nuvoton_yolo_dataset.py`, the local Ultralytics fork, and `relu6-yolov8.yaml` at 192x192 input size.

### 1-Epoch GPU Smoke Test

Use this first to verify the end-to-end training pipeline:

#### Windows PowerShell

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

#### Linux / WSL / macOS

```bash
repo="$(pwd)"
export MPLCONFIGDIR="$repo/.matplotlib"
export YOLO_CONFIG_DIR="$repo/.ultralytics"
export TEMP=/tmp/nuvoton_model
export TMP=/tmp/nuvoton_model
export TMPDIR=/tmp/nuvoton_model
mkdir -p /tmp/nuvoton_model

cd ML_YOLO/yolov8_ultralytics

python dg_train.py \
  --model-cfg ultralytics/cfg/models/v8/relu6-yolov8.yaml \
  --data "$repo/prepared_datasets/nuvoton_people_v1/dataset.yaml" \
  --imgsz 192 \
  --weights yolov8n.pt \
  --epochs 1 \
  --patience 30 \
  --device 0 \
  --workers 0 \
  --save-period 1 \
  --project "$repo/runs/nuvoton_yolo" \
  --name smoke_1epoch_gpu
```

Expected output directory:

```text
runs/nuvoton_yolo/smoke_1epoch_gpu/
  weights/
    last.pt
    best.pt
```

### Full Training Run

For a longer run, keep the same command but change `--epochs` and `--name`:

#### Windows PowerShell

```powershell
python dg_train.py `
  --model-cfg ultralytics/cfg/models/v8/relu6-yolov8.yaml `
  --data "$repo\prepared_datasets\nuvoton_people_v1\dataset.yaml" `
  --imgsz 192 `
  --weights yolov8n.pt `
  --epochs 200 `
  --patience 30 `
  --device 0 `
  --workers 0 `
  --save-period 1 `
  --project "$repo\runs\nuvoton_yolo" `
  --name nuvoton_people_v1_relu6_192_e200
```

#### Linux / WSL / macOS

```bash
python dg_train.py \
  --model-cfg ultralytics/cfg/models/v8/relu6-yolov8.yaml \
  --data "$repo/prepared_datasets/nuvoton_people_v1/dataset.yaml" \
  --imgsz 192 \
  --weights yolov8n.pt \
  --epochs 200 \
  --patience 30 \
  --device 0 \
  --workers 0 \
  --save-period 1 \
  --project "$repo/runs/nuvoton_yolo" \
  --name nuvoton_people_v1_relu6_192_e200
```

Important notes:

- use an absolute `--data` path so Ultralytics does not resolve relative paths against a global `datasets_dir`
- `--device 0` uses the first NVIDIA GPU; `--device cpu` forces CPU training
- default batch size is `64`; add `--batch 16` or `--batch 8` if CUDA runs out of memory
- `yolov8n.pt` downloads automatically on first use if it is not already present

### Optional Bash Wrapper

If using Git Bash or WSL:

```bash
NUVOTON_DEVICE=0 bash scripts/train_nuvoton_yolo.sh yolov8n.pt 200 192 nuvoton_people_v1_relu6_192_e200
```

## Baseline Training

The baseline path trains a grayscale Faster R-CNN MobileNetV3 model adapted for 1-channel 192x192 inputs.

### Windows PowerShell

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

### Linux / WSL / macOS

```bash
python scripts/build_splits.py --dataset-root overhead-person-detection
python scripts/inspect_dataset.py --dataset-root overhead-person-detection --split train
python scripts/train_baseline.py \
  --dataset-root overhead-person-detection \
  --output-dir runs/baseline_frcnn \
  --epochs 10 \
  --batch-size 4 \
  --image-size 192 \
  --device auto
```

Expected outputs:

- `runs/baseline_frcnn/run_config.json`
- `runs/baseline_frcnn/metrics.jsonl`
- `runs/baseline_frcnn/last.pt`
- `runs/baseline_frcnn/best.pt`

## Label Preview Utility

Optional visual sanity check for the overhead dataset:

### Windows PowerShell

```powershell
python scripts\export_label_previews.py --dataset-root overhead-person-detection --split val --num-samples 25
```

### Linux / WSL / macOS

```bash
python scripts/export_label_previews.py --dataset-root overhead-person-detection --split val --num-samples 25
```