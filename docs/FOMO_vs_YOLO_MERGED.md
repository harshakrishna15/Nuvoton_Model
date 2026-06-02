# FOMO vs YOLOv8 — merged-dataset run (with SJSU headcount)

Side-by-side of the two single-class person detectors trained from
`scripts/train_all_models.sh` on the merged corpus
(`prepared_datasets/nuvoton_people_v1`).

## Training corpus

Same merged dataset for both models:

| Source                          | Images |
|---------------------------------|-------:|
| overhead-person-detection       | 13,448 |
| Passenger Counter.yolov8        |  4,906 |
| sjsu-headcount-scene-1.coco     |    580 |
| sjsu-headcount-scene-2.coco     |    350 |
| **Total**                       | **19,284** |

Single class `person` (id 0). Image size 192×192. Same train/val/test splits.

## Headline numbers

| Model                                  | Task               | Best metric                          | Run dir |
|----------------------------------------|--------------------|--------------------------------------|---------|
| **YOLOv8n (Nuvoton ReLU6, 200 ep)**    | Detection          | mAP50 **0.9707**, mAP50-95 0.6780    | `runs/nuvoton_yolo/nuvoton_people_v2_relu6_192_e200/` |
| **FOMO MobileNetV2 (6×6 grid, 60 ep)** | Heatmap counting   | val count-MAE **0.6514**, test count-MAE **0.4559** | `runs/fomo_merged/` |

## YOLOv8 — full detection metrics

| Metric              | epoch 50 | epoch 100 | epoch 150 | epoch 200 |
|---------------------|---------:|----------:|----------:|----------:|
| `precision`         | 0.9292   | 0.9378    | 0.9365    | **0.9517** |
| `recall`            | 0.8978   | 0.9144    | 0.9263    | **0.9204** |
| `mAP50`             | 0.9564   | 0.9643    | 0.9678    | **0.9707** |
| `mAP50-95`          | 0.6237   | 0.6540    | 0.6665    | **0.6780** |
| `train/box_loss`    | 1.3499   | 1.2533    | 1.1767    | 1.0206    |
| `val/box_loss`      | 1.2369   | 1.1715    | 1.1425    | 1.1199    |

Curves: `results.png`, `PR_curve.png`, `F1_curve.png`,
`confusion_matrix.png` (committed alongside the run).

## FOMO — counting metrics

Counts derived from the 6×6 sigmoid heatmap (3×3 NMS, threshold 0.5).

| Split          | count MAE | count RMSE | count bias | empty FP rate |
|----------------|----------:|-----------:|-----------:|--------------:|
| val (best, ep 60) | **0.6514** | 1.398 | −0.577 | 0.093 |
| **test (best ckpt)** | **0.4559** | 1.175 | −0.366 | 0.128 |

Per-epoch trajectory (every 10 epochs):

| Epoch | train_loss | val_loss | val_count_mae | val_empty_fp |
|------:|-----------:|---------:|--------------:|-------------:|
| 10    | ~0.07      | ~1.5     | 0.83          | 0.21         |
| 30    | ~0.03      | ~1.3     | 0.71          | 0.13         |
| 60    | 0.0139     | 1.164    | 0.660         | 0.093        |

Full per-epoch log: `runs/fomo_merged/metrics.jsonl`.

## Side-by-side trade-offs

| Aspect                     | YOLOv8n ReLU6 (192)       | FOMO MobileNetV2 (192, 6×6) |
|----------------------------|---------------------------|------------------------------|
| Output                     | bounding boxes + scores   | 6×6 person-presence heatmap  |
| What it answers            | Where is each person?     | How many people, roughly where? |
| Best metric                | mAP50 0.9707              | count MAE 0.46 (test)        |
| Empty-image false positives | n/a (per-box NMS handles) | 12.8% on test                 |
| Checkpoint size            | **6.5 MB**                | 26 MB                        |
| Training epochs (this run) | 200                       | 60                           |
| Recommended deployment     | Detector pipelines (boxes needed, fewer simultaneous people) | On-device counting on Ethos-U55 NPU (~19.6 ms/Vela per `FIRMWARE_INTEGRATION_SUMMARY.md`) |
| Strength                   | High localisation accuracy, mAP50-95 0.68 | Tiny output tensor, fast int8 inference, robust to occlusion when boxes overlap inside one cell |
| Weakness                   | Larger compute / harder int8 quantisation than FOMO | Cannot resolve two people whose centres land in the same 32 px grid cell; biased to under-count crowded scenes (bias −0.37) |

## Take-aways

- Adding the two SJSU head-count COCO datasets (930 extra images) on top of
  overhead-person + Passenger Counter pushed the YOLOv8n run to mAP50 0.97
  and lifted FOMO test count-MAE below 0.5 — the model is essentially within
  half a person of the ground-truth count on average.
- FOMO's negative count bias (−0.37) is consistent with the known limitation
  that crowded SJSU scenes occasionally place two head centres in the same
  6×6 cell. Bumping `--grid-size` to 8 or 12 is the cheapest mitigation if
  the firmware budget allows the slightly larger output tensor.
- For the on-device people-counter pipeline described in
  `windows_deployment/FIRMWARE_INTEGRATION_SUMMARY.md`, FOMO remains the
  recommended head — YOLOv8n is the higher-quality "ground-truth" detector
  used to validate the FOMO heatmap end-to-end.

## Reproduce

```bash
python scripts/prepare_nuvoton_yolo_dataset.py --force          # rebuild merged dataset
EPOCHS_YOLO=200 EPOCHS_FOMO=60 IMGSZ=192 bash scripts/train_all_models.sh
```

Weights for both models are published as assets on the GitHub Release for
this commit (see `docs/MERGED_DATASET_RUNS.md`).
