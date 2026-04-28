"""Training helpers for baseline object detection experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from tqdm.auto import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device(preferred: str | None = None) -> str:
    if preferred and preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def move_targets_to_device(targets, device: torch.device):
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


class RandomHorizontalFlip:
    """Simple joint horizontal flip for grayscale detection samples."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: torch.Tensor, target: dict[str, torch.Tensor]):
        if torch.rand(1).item() >= self.p:
            return image, target

        _, _, width = image.shape
        flipped_image = torch.flip(image, dims=[2])
        flipped_target = {key: value.clone() for key, value in target.items()}
        boxes = flipped_target["boxes"]
        if len(boxes):
            x1 = width - boxes[:, 2]
            x2 = width - boxes[:, 0]
            boxes[:, 0] = x1
            boxes[:, 2] = x2
        return flipped_image, flipped_target


def append_metrics(log_path: str | Path, metrics: dict) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics) + "\n")


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device: torch.device,
    *,
    max_batches: int | None = None,
    epoch: int | None = None,
):
    model.train()
    total_loss = 0.0
    total_batches = 0
    total_steps = min(len(dataloader), max_batches) if max_batches is not None else len(dataloader)
    progress = tqdm(dataloader, total=total_steps, desc=f"train epoch {epoch}" if epoch is not None else "train", leave=False)

    for batch_index, (images, targets) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1
        progress.set_postfix(loss=f"{(total_loss / total_batches):.4f}")

    progress.close()

    return {
        "loss": total_loss / max(total_batches, 1),
        "batches": total_batches,
    }


@torch.no_grad()
def evaluate_detection_loss(
    model,
    dataloader,
    device: torch.device,
    *,
    max_batches: int | None = None,
    epoch: int | None = None,
):
    was_training = model.training
    model.train()

    total_loss = 0.0
    total_batches = 0
    total_steps = min(len(dataloader), max_batches) if max_batches is not None else len(dataloader)
    progress = tqdm(
        dataloader,
        total=total_steps,
        desc=f"val-loss epoch {epoch}" if epoch is not None else "val-loss",
        leave=False,
    )
    for batch_index, (images, targets) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)
        loss_dict = model(images, targets)
        total_loss += float(sum(loss_dict.values()).item())
        total_batches += 1
        progress.set_postfix(loss=f"{(total_loss / total_batches):.4f}")

    progress.close()

    if not was_training:
        model.eval()

    return {
        "loss": total_loss / max(total_batches, 1),
        "batches": total_batches,
    }


@torch.no_grad()
def evaluate_count_metrics(
    model,
    dataloader,
    device: torch.device,
    *,
    score_threshold: float = 0.5,
    max_batches: int | None = None,
    epoch: int | None = None,
):
    model.eval()

    total_images = 0
    total_abs_error = 0.0
    total_signed_error = 0.0
    empty_images = 0
    empty_with_false_positive = 0
    total_steps = min(len(dataloader), max_batches) if max_batches is not None else len(dataloader)
    progress = tqdm(
        dataloader,
        total=total_steps,
        desc=f"val-count epoch {epoch}" if epoch is not None else "val-count",
        leave=False,
    )

    for batch_index, (images, targets) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = [image.to(device) for image in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            gt_count = int(target["boxes"].shape[0])
            pred_count = int((output["scores"].detach().cpu() >= score_threshold).sum().item())
            error = pred_count - gt_count

            total_images += 1
            total_abs_error += abs(error)
            total_signed_error += error
            if gt_count == 0:
                empty_images += 1
                if pred_count > 0:
                    empty_with_false_positive += 1

        progress.set_postfix(mae=f"{(total_abs_error / max(total_images, 1)):.3f}")

    progress.close()

    return {
        "images": total_images,
        "count_mae": total_abs_error / max(total_images, 1),
        "count_bias": total_signed_error / max(total_images, 1),
        "empty_false_positive_rate": (
            empty_with_false_positive / max(empty_images, 1) if empty_images else 0.0
        ),
    }
