# Merged-dataset training run (Nuvoton people-counter)

Single-class `person` detector trained on the merged corpus produced by
`scripts/prepare_nuvoton_yolo_dataset.py`.

## Dataset composition

| Source                          | Images |
|---------------------------------|-------:|
| overhead-person-detection       | 13,448 |
| Passenger Counter.yolov8        |  4,906 |
| sjsu-headcount-scene-1.coco     |    580 |
| sjsu-headcount-scene-2.coco     |    350 |
| **Total**                       | **19,284** |

Splits: deterministic train/val/test, exported to `prepared_datasets/nuvoton_people_v1/`
with a YOLO-style folder layout and `dataset.yaml` (`nc: 1`, `names: [person]`).

## YOLOv8 (Nuvoton ReLU6 config, 192×192, 200 epochs)

| Metric              | Value  |
|---------------------|-------:|
| `metrics/precision` | 0.9517 |
| `metrics/recall`    | 0.9204 |
| `metrics/mAP50`     | 0.9707 |
| `metrics/mAP50-95`  | 0.6780 |

Run dir: `runs/nuvoton_yolo/nuvoton_people_v2_relu6_192_e200/`
Curves / batch previews: see `results.png`, `PR_curve.png`, `F1_curve.png`,
`confusion_matrix*.png` in the same folder.

## FOMO MobileNetV2 (192×192, 6×6 grid, 60 epochs)

| Split | count MAE | count RMSE | count bias | empty FP rate |
|-------|----------:|-----------:|-----------:|--------------:|
| val (best) | 0.6514 | – | – | – |
| test       | 0.4559 | 1.1748 | -0.3655 | 0.128 |

Run dir: `runs/fomo_merged/`
Per-epoch log: `runs/fomo_merged/metrics.jsonl`.

## Reproducing

```bash
# Rebuild merged dataset (overhead + Passenger Counter + every *.coco/)
python scripts/prepare_nuvoton_yolo_dataset.py --force

# Full pipeline (YOLOv8 200ep + FOMO 60ep)
EPOCHS_YOLO=200 EPOCHS_FOMO=60 IMGSZ=192 bash scripts/train_all_models.sh
```

## Weights

The `.pt` checkpoints are not committed (large binaries). They are published as
assets on the GitHub Release for this commit:

- `fomo_merged_best.pt` – best FOMO checkpoint
- `yolov8_relu6_192_e200_best.pt` – best YOLOv8 checkpoint
- `yolov8_relu6_192_e200_last.pt` – final YOLOv8 checkpoint
