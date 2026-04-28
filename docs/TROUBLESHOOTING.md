# Troubleshooting

## Dataset Root Does Not Exist

Error example:

```text
FileNotFoundError: Dataset root does not exist: ...\overhead-person-detection
```

Fix: download the datasets and place them in the exact folders documented in [Dataset Setup](DATASETS.md).

## YOLO Looks For Images In The Wrong Directory

Error example:

```text
Dataset images not found, missing path ...\prepared_datasets\nuvoton_people_v1\val\images
```

Fix: pass an absolute `--data` path:

Windows PowerShell:
Fix: pass an absolute `--data` path.

### Windows PowerShell

```powershell
$repo = "<path-to-repo>"
$repo = (Get-Location).Path
--data "$repo\prepared_datasets\nuvoton_people_v1\dataset.yaml"
```

Linux:

```bash
repo="<path-to-repo>"
--data "$repo/prepared_datasets/nuvoton_people_v1/dataset.yaml"
```

### Linux / WSL / macOS

```bash
repo="$(pwd)"
--data "$repo/prepared_datasets/nuvoton_people_v1/dataset.yaml"
```

Avoid relying on relative `../../prepared_datasets/...` paths from inside `ML_YOLO/yolov8_ultralytics`.

## CUDA Is Not Available

Check:

### Windows PowerShell

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Linux / WSL / macOS

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected for GPU training:

```text
True
```

If it prints `False`, check the NVIDIA driver and confirm the environment installed the CUDA-enabled PyTorch wheels from `requirements.txt`.

## CUDA Out Of Memory

Add a smaller batch size to `dg_train.py`:

```bash
--batch 16
```

If needed, use:

```bash
--batch 8
```

## Shell Activation Issues

## Virtual Environment Activation Issues

### Windows PowerShell

If PowerShell blocks `.\.venv\Scripts\Activate.ps1`, use the venv Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\build_splits.py --dataset-root overhead-person-detection
```

### Linux / WSL / macOS

If `source .venv/bin/activate` is not available, use the venv Python directly:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/build_splits.py --dataset-root overhead-person-detection
```

## Generated Files Show Up In Git Status

Most generated files are ignored, including:

- `.venv/`
- `.hf-cache/`
- `.matplotlib/`
- `.tmp/`
- `.ultralytics/`
- `prepared_datasets/`
- `runs/`
- model weights and exported model files

If `git status --short` shows only ignored files via `git status --ignored --short`, there is usually nothing to commit.
