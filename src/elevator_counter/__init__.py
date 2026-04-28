"""Local utilities for the overhead-person detection dataset."""

from .data import (
    OverheadPersonDetectionDataset,
    SplitConfig,
    build_split_manifest,
    detection_collate_fn,
    load_local_dataset,
    load_split_manifest,
)
from .evaluation import (
    DEFAULT_BUCKETS,
    collect_count_predictions,
    compute_bucket_metrics,
    compute_count_metrics,
    counts_from_scores,
    select_best_threshold,
    sweep_thresholds,
)

__all__ = [
    "DEFAULT_BUCKETS",
    "OverheadPersonDetectionDataset",
    "SplitConfig",
    "build_split_manifest",
    "collect_count_predictions",
    "compute_bucket_metrics",
    "compute_count_metrics",
    "counts_from_scores",
    "detection_collate_fn",
    "load_local_dataset",
    "load_split_manifest",
    "select_best_threshold",
    "sweep_thresholds",
]
