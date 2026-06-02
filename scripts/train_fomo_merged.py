"""Train a FOMO grid detector on the merged YOLO-format dataset.

Mirrors `scripts/train_fomo.py` but reads the prepared
``prepared_datasets/nuvoton_people_v1`` directory (which fuses the
overhead-person, Passenger Counter and SJSU head-count COCO datasets).

Usage (from repo root):

    python scripts/train_fomo_merged.py \
        --dataset-root prepared_datasets/nuvoton_people_v1 \
        --epochs 30 --batch-size 32 --grid-size 6
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_counter.data import detection_collate_fn  # noqa: E402
from elevator_counter.yolo_dir_data import YoloFolderDetectionDataset  # noqa: E402
from elevator_counter.fomo import (  # noqa: E402
    FomoLoss,
    build_fomo_mobilenetv2,
    count_from_heatmap,
    targets_to_heatmaps,
)
from elevator_counter.training import (  # noqa: E402
    RandomHorizontalFlip,
    append_metrics,
    detect_device,
    seed_everything,
)


class _PhotometricJitter:
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2) -> None:
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image: torch.Tensor, target):
        if self.brightness > 0:
            shift = (torch.rand(1).item() * 2 - 1) * self.brightness
            image = (image + shift).clamp(0.0, 1.0)
        if self.contrast > 0:
            factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
            mean = image.mean()
            image = ((image - mean) * factor + mean).clamp(0.0, 1.0)
        return image, target


class _Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(ROOT / "prepared_datasets" / "nuvoton_people_v1"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "fomo_merged"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=100.0)
    parser.add_argument("--focal", action="store_true")
    parser.add_argument("--count-aux-weight", type=float, default=0.0)
    parser.add_argument("--brightness-jitter", type=float, default=0.0)
    parser.add_argument("--contrast-jitter", type=float, default=0.0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def build_loaders(args):
    transforms = [RandomHorizontalFlip(p=0.5)]
    if args.brightness_jitter > 0 or args.contrast_jitter > 0:
        transforms.append(_PhotometricJitter(args.brightness_jitter, args.contrast_jitter))
    train_transform = _Compose(transforms) if len(transforms) > 1 else transforms[0]

    train_ds = YoloFolderDetectionDataset(
        args.dataset_root, split="train", image_size=args.image_size, transform=train_transform,
    )
    val_ds = YoloFolderDetectionDataset(
        args.dataset_root, split="val", image_size=args.image_size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=detection_collate_fn, num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=detection_collate_fn, num_workers=args.num_workers,
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)


def stack_images(images) -> torch.Tensor:
    return torch.stack(list(images), dim=0)


def train_one_epoch(model, loader, optimizer, loss_fn, device, *, image_size, grid_size, max_batches=None, epoch=None):
    model.train()
    total_loss, total_batches = 0.0, 0
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for batch_index, (images, targets) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = stack_images(images).to(device)
        target_maps = targets_to_heatmaps(targets, image_size=image_size, grid_size=grid_size).to(device)
        logits = model(batch)
        loss = loss_fn(logits, target_maps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1
        progress.set_postfix(loss=f"{(total_loss/total_batches):.4f}")
    progress.close()
    return {"loss": total_loss / max(total_batches, 1), "batches": total_batches}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, *, image_size, grid_size, threshold, max_batches=None, epoch=None):
    model.eval()
    total_loss, total_batches, total_images = 0.0, 0, 0
    total_abs, total_signed, total_sq = 0.0, 0.0, 0.0
    empty_images, empty_with_fp = 0, 0
    progress = tqdm(loader, desc=f"val {epoch}", leave=False)
    for batch_index, (images, targets) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = stack_images(images).to(device)
        target_maps = targets_to_heatmaps(targets, image_size=image_size, grid_size=grid_size).to(device)
        logits = model(batch)
        loss = loss_fn(logits, target_maps)
        total_loss += float(loss.item())
        total_batches += 1
        probs = torch.sigmoid(logits)
        pred_counts = count_from_heatmap(probs, threshold=threshold).cpu()
        for i, t in enumerate(targets):
            gt = int(t["boxes"].shape[0])
            pred = int(pred_counts[i].item())
            err = pred - gt
            total_images += 1
            total_abs += abs(err)
            total_sq += err * err
            total_signed += err
            if gt == 0:
                empty_images += 1
                if pred > 0:
                    empty_with_fp += 1
        progress.set_postfix(mae=f"{(total_abs/max(total_images,1)):.3f}")
    progress.close()
    return {
        "loss": total_loss / max(total_batches, 1),
        "images": total_images,
        "count_mae": total_abs / max(total_images, 1),
        "count_rmse": math.sqrt(total_sq / max(total_images, 1)),
        "count_bias": total_signed / max(total_images, 1),
        "empty_false_positive_rate": empty_with_fp / max(empty_images, 1) if empty_images else 0.0,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(detect_device(args.device))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_loader, val_loader, n_train, n_val = build_loaders(args)
    print(f"[fomo-merged] train={n_train} val={n_val} device={device}")

    model = build_fomo_mobilenetv2(
        num_classes=1,
        grid_size=args.grid_size,
        image_size=args.image_size,
        pretrained=not args.no_pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    loss_fn = FomoLoss(
        pos_weight=args.pos_weight, use_focal=args.focal, count_aux_weight=args.count_aux_weight,
    )

    best_mae = math.inf
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            image_size=args.image_size, grid_size=args.grid_size,
            max_batches=args.max_train_batches, epoch=epoch,
        )
        val_metrics = evaluate(
            model, val_loader, loss_fn, device,
            image_size=args.image_size, grid_size=args.grid_size,
            threshold=args.threshold, max_batches=args.max_val_batches, epoch=epoch,
        )
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_count_mae": val_metrics["count_mae"],
            "val_count_rmse": val_metrics["count_rmse"],
            "val_count_bias": val_metrics["count_bias"],
            "val_empty_fp_rate": val_metrics["empty_false_positive_rate"],
        }
        append_metrics(metrics_path, record)
        print(json.dumps(record))

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, last_path)
        if val_metrics["count_mae"] < best_mae:
            best_mae = val_metrics["count_mae"]
            torch.save(ckpt, best_path)

    # Final test-split evaluation if a test split exists.
    test_split_dir = Path(args.dataset_root) / "test" / "images"
    summary = {"best_val_count_mae": best_mae, "checkpoint": str(best_path)}
    if test_split_dir.exists() and any(test_split_dir.iterdir()):
        test_ds = YoloFolderDetectionDataset(args.dataset_root, split="test", image_size=args.image_size)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=detection_collate_fn, num_workers=args.num_workers,
        )
        # Reload best checkpoint
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate(
            model, test_loader, loss_fn, device,
            image_size=args.image_size, grid_size=args.grid_size,
            threshold=args.threshold, epoch="test",
        )
        summary["test_metrics"] = test_metrics
        print("[fomo-merged] test:", json.dumps(test_metrics, indent=2))

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[fomo-merged] done. Best val count MAE: {best_mae:.4f}. Output: {output_dir}")


if __name__ == "__main__":
    main()
