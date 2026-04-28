# Evaluation

This repo evaluates models with count-focused metrics, not only box-level detection quality.

## Baseline Evaluation

### Windows PowerShell

```powershell
python scripts\evaluate_baseline.py `
  --dataset-root overhead-person-detection `
  --checkpoint runs\baseline_frcnn\best.pt `
  --output runs\baseline_frcnn\eval_summary.json
```

### Linux / WSL / macOS

```bash
python scripts/evaluate_baseline.py \
  --dataset-root overhead-person-detection \
  --checkpoint runs/baseline_frcnn/best.pt \
  --output runs/baseline_frcnn/eval_summary.json
```

The baseline evaluator sweeps score thresholds on validation, chooses a count-focused threshold, and reports test-set count quality.

## Nuvoton YOLO Evaluation

### Windows PowerShell

```powershell
python scripts\evaluate_nuvoton_yolo.py `
  --weights runs\nuvoton_yolo\<run-name>\weights\best.pt `
  --data prepared_datasets\nuvoton_people_v1\dataset.yaml `
  --split val `
  --imgsz 192 `
  --conf 0.25 `
  --device 0
```

### Linux / WSL / macOS

```bash
python scripts/evaluate_nuvoton_yolo.py \
  --weights runs/nuvoton_yolo/<run-name>/weights/best.pt \
  --data prepared_datasets/nuvoton_people_v1/dataset.yaml \
  --split val \
  --imgsz 192 \
  --conf 0.25 \
  --device 0
```

Common options:

- `--split val` or `--split test`
- `--conf 0.25` changes the counting confidence threshold
- `--max-images N` evaluates only the first N images for quick checks
- `--device 0` uses the first GPU; `--device cpu` runs on CPU

## Evaluation Outputs

The YOLO evaluator writes a report directory containing:

- `summary.json`
- `per_image_counts.csv`
- `count_scatter.png`
- `count_error_hist.png`
- `worst_cases/` with ground-truth vs prediction overlays

## Metrics

The evaluation helpers report count-first metrics including:

- count MAE
- count RMSE
- count bias
- exact-match rate
- within-one rate
- overcount and undercount rate
- empty-scene false positive rate
- bucketed metrics for `0`, `1`, `2`, `3-4`, and `5+` people