"""Evaluation helpers for elevator person-counting checkpoints."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm


@dataclass(frozen=True)
class CountBucket:
    name: str
    minimum: int
    maximum: int | None = None

    def contains(self, value: int) -> bool:
        if value < self.minimum:
            return False
        if self.maximum is None:
            return True
        return value <= self.maximum


DEFAULT_BUCKETS = (
    CountBucket("0", 0, 0),
    CountBucket("1", 1, 1),
    CountBucket("2", 2, 2),
    CountBucket("3-4", 3, 4),
    CountBucket("5+", 5, None),
)


@torch.no_grad()
def collect_count_predictions(
    model,
    dataloader,
    device: torch.device,
    *,
    max_batches: int | None = None,
    desc: str = "eval",
):
    """Run inference once and keep scores so thresholds can be swept offline."""

    model.eval()
    gt_counts: list[int] = []
    score_lists: list[list[float]] = []

    total_steps = min(len(dataloader), max_batches) if max_batches is not None else len(dataloader)
    progress = tqdm(dataloader, total=total_steps, desc=desc, leave=False)
    for batch_index, (images, targets) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        outputs = model([image.to(device) for image in images])
        for target, output in zip(targets, outputs):
            gt_counts.append(int(target["boxes"].shape[0]))
            score_lists.append(output["scores"].detach().cpu().tolist())

    progress.close()
    return gt_counts, score_lists


def counts_from_scores(score_lists: list[list[float]], threshold: float) -> list[int]:
    return [sum(score >= threshold for score in scores) for scores in score_lists]


def compute_count_metrics(gt_counts: list[int], pred_counts: list[int]) -> dict:
    if len(gt_counts) != len(pred_counts):
        raise ValueError("Ground-truth and prediction lengths do not match.")

    num_images = len(gt_counts)
    errors = [pred - gt for gt, pred in zip(gt_counts, pred_counts)]
    abs_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]

    empty_indices = [index for index, gt in enumerate(gt_counts) if gt == 0]
    empty_false_positives = sum(1 for index in empty_indices if pred_counts[index] > 0)

    return {
        "images": num_images,
        "count_mae": sum(abs_errors) / max(num_images, 1),
        "count_rmse": math.sqrt(sum(squared_errors) / max(num_images, 1)),
        "count_bias": sum(errors) / max(num_images, 1),
        "exact_match_rate": sum(1 for error in errors if error == 0) / max(num_images, 1),
        "within_one_rate": sum(1 for error in abs_errors if error <= 1) / max(num_images, 1),
        "undercount_rate": sum(1 for error in errors if error < 0) / max(num_images, 1),
        "overcount_rate": sum(1 for error in errors if error > 0) / max(num_images, 1),
        "empty_false_positive_rate": empty_false_positives / len(empty_indices) if empty_indices else 0.0,
        "mean_gt_count": sum(gt_counts) / max(num_images, 1),
        "mean_pred_count": sum(pred_counts) / max(num_images, 1),
    }


def compute_bucket_metrics(
    gt_counts: list[int],
    pred_counts: list[int],
    *,
    buckets: tuple[CountBucket, ...] = DEFAULT_BUCKETS,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for bucket in buckets:
        selected = [
            (gt, pred)
            for gt, pred in zip(gt_counts, pred_counts)
            if bucket.contains(gt)
        ]
        if not selected:
            results[bucket.name] = {"images": 0}
            continue

        bucket_gt = [gt for gt, _ in selected]
        bucket_pred = [pred for _, pred in selected]
        results[bucket.name] = compute_count_metrics(bucket_gt, bucket_pred)
    return results


def sweep_thresholds(
    gt_counts: list[int],
    score_lists: list[list[float]],
    thresholds: list[float],
) -> list[dict]:
    results = []
    for threshold in thresholds:
        pred_counts = counts_from_scores(score_lists, threshold)
        metrics = compute_count_metrics(gt_counts, pred_counts)
        results.append({"threshold": threshold, **metrics})
    return results


def select_best_threshold(sweep_results: list[dict]) -> dict:
    if not sweep_results:
        raise ValueError("sweep_results is empty")
    return min(
        sweep_results,
        key=lambda item: (
            item["count_mae"],
            abs(item["count_bias"]),
            -item["exact_match_rate"],
        ),
    )
