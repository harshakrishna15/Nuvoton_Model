# Fresh Setup

This guide brings a new machine from a clean clone to a ready-to-train environment.

## Requirements

- Python 3.12
- NVIDIA GPU and a working NVIDIA driver for GPU training
- Enough disk space for datasets, generated YOLO exports, checkpoints, and caches
- Internet access for Python dependencies and the first `yolov8n.pt` download

## Clone The Repo

```powershell
git clone <repo-url> Nuvoton_Model
cd Nuvoton_Model
```

If the repo already exists locally, start from the repo root:

```powershell
cd "C:\Users\Harsha Krishnaswamy\Desktop\Development\Nuvoton_Model"
```

Or on Linux / WSL / macOS:

```bash
cd /path/to/Nuvoton_Model
```

## Create The Python Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux / WSL / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs CUDA-enabled PyTorch wheels from the official PyTorch index and installs the local Ultralytics fork from `ML_YOLO/yolov8_ultralytics` in editable mode.

## Verify CUDA

### Windows PowerShell

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

### Linux / WSL / macOS

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

For GPU training, the Python check should print `cuda: True` and the NVIDIA GPU name.

If CUDA is not available, see [Troubleshooting](TROUBLESHOOTING.md).

## Next Step

Continue with [Dataset Setup](DATASETS.md).
