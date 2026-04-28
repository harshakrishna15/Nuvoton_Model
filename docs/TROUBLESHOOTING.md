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

```powershell
$repo = "C:\Users\Harsha Krishnaswamy\Desktop\Development\Nuvoton_Model"
--data "$repo\prepared_datasets\nuvoton_people_v1\dataset.yaml"
```

Avoid relying on relative `..\..\prepared_datasets\...` paths from inside `ML_YOLO/yolov8_ultralytics`.

## CUDA Is Not Available

Check:

```powershell
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

```powershell
--batch 16
```

If needed, use:

```powershell
--batch 8
```

## PowerShell Activation Is Blocked

If `.\.venv\Scripts\Activate.ps1` is blocked by execution policy, use the venv Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\build_splits.py --dataset-root overhead-person-detection
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