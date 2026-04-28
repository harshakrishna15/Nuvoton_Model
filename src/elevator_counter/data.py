"""Dataset utilities for the overhead-person detection parquet export."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from datasets import Image, load_dataset
from PIL import Image as PILImage
from torchvision.transforms.functional import pil_to_tensor


DEFAULT_CACHE_DIRNAME = ".hf-cache"
DEFAULT_MANIFEST_NAME = "splits.json"


@dataclass(frozen=True)
class SplitConfig:
    """Deterministic split configuration."""

    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 42

    def validate(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if any(x <= 0 for x in (self.train_fraction, self.val_fraction, self.test_fraction)):
            raise ValueError("All split fractions must be > 0.")
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1.0, got {total}.")


def _dataset_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def find_parquet_files(dataset_root: str | Path) -> list[Path]:
    """Return sorted parquet files inside the downloaded Hugging Face dataset snapshot."""

    root = _dataset_root(dataset_root)
    parquet_files = sorted((root / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    return parquet_files


def load_local_dataset(
    dataset_root: str | Path,
    *,
    cache_dir: str | Path | None = None,
    decode_images: bool = False,
):
    """Load the parquet dataset through Hugging Face Datasets with a local cache directory."""

    parquet_files = [str(path) for path in find_parquet_files(dataset_root)]
    root = _dataset_root(dataset_root)
    resolved_cache_dir = Path(cache_dir) if cache_dir else root.parent / DEFAULT_CACHE_DIRNAME
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "parquet",
        data_files=parquet_files,
        split="train",
        cache_dir=str(resolved_cache_dir),
    )
    if not decode_images:
        dataset = dataset.cast_column("image", Image(decode=False))
    return dataset


def _stable_example_key(image_payload: dict[str, Any], seed: int) -> str:
    image_bytes = image_payload.get("bytes")
    if image_bytes:
        digest_source = image_bytes
    else:
        digest_source = str(image_payload.get("path", "")).encode("utf-8")
    return hashlib.sha1(f"{seed}:".encode("utf-8") + digest_source).hexdigest()


def _split_counts(size: int, config: SplitConfig) -> dict[str, int]:
    train_count = int(size * config.train_fraction)
    val_count = int(size * config.val_fraction)
    test_count = size - train_count - val_count
    return {"train": train_count, "val": val_count, "test": test_count}


def build_split_manifest(
    dataset_root: str | Path,
    *,
    output_path: str | Path | None = None,
    config: SplitConfig | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Create a deterministic split manifest from image bytes and write it to JSON."""

    split_config = config or SplitConfig()
    split_config.validate()

    dataset = load_local_dataset(dataset_root, cache_dir=cache_dir, decode_images=False)
    total_examples = len(dataset)

    ordered = []
    empty_images = 0
    total_boxes = 0
    image_size = None

    for index, row in enumerate(dataset):
        image_payload = row["image"]
        if image_size is None:
            with PILImage.open(io.BytesIO(image_payload["bytes"])) as image:
                image_size = list(image.size)

        bboxes = row["objects"]["bbox"]
        total_boxes += len(bboxes)
        if not bboxes:
            empty_images += 1

        ordered.append((index, _stable_example_key(image_payload, split_config.seed)))

    ordered.sort(key=lambda item: item[1])
    sorted_indices = [index for index, _ in ordered]
    counts = _split_counts(total_examples, split_config)

    train_end = counts["train"]
    val_end = train_end + counts["val"]
    splits = {
        "train": sorted_indices[:train_end],
        "val": sorted_indices[train_end:val_end],
        "test": sorted_indices[val_end:],
    }

    manifest = {
        "dataset_root": str(_dataset_root(dataset_root)),
        "parquet_files": [str(path) for path in find_parquet_files(dataset_root)],
        "image_size": image_size,
        "seed": split_config.seed,
        "split_config": asdict(split_config),
        "stats": {
            "total_examples": total_examples,
            "total_boxes": total_boxes,
            "empty_images": empty_images,
        },
        "splits": splits,
    }

    manifest_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else _dataset_root(dataset_root) / DEFAULT_MANIFEST_NAME
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def load_split_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load a JSON split manifest generated by build_split_manifest()."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_xywh_boxes(
    boxes: list[list[float]],
    labels: list[int],
    *,
    min_extent: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop malformed boxes with non-positive width/height and convert to xyxy."""

    if not boxes:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )

    converted = []
    kept_labels = []
    for box, label in zip(boxes, labels):
        x, y, w, h = box
        if w <= min_extent or h <= min_extent:
            continue
        converted.append([x, y, x + w, y + h])
        kept_labels.append(int(label))

    if not converted:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )

    return (
        torch.tensor(converted, dtype=torch.float32),
        torch.tensor(kept_labels, dtype=torch.int64),
    )


class OverheadPersonDetectionDataset(torch.utils.data.Dataset):
    """PyTorch dataset for the local overhead-person detection parquet snapshot."""

    class_names = ("person",)

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str,
        manifest_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        label_offset: int = 1,
        transform: Callable[[torch.Tensor, dict[str, torch.Tensor]], tuple[torch.Tensor, dict[str, torch.Tensor]]] | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.dataset_root = _dataset_root(dataset_root)
        self.manifest_path = (
            Path(manifest_path).expanduser().resolve()
            if manifest_path
            else self.dataset_root / DEFAULT_MANIFEST_NAME
        )
        manifest = load_split_manifest(self.manifest_path)
        self.indices = manifest["splits"][split]
        self.split = split
        self.label_offset = label_offset
        self.transform = transform
        self.dataset = load_local_dataset(
            self.dataset_root,
            cache_dir=cache_dir,
            decode_images=False,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source_index = int(self.indices[item])
        row = self.dataset[source_index]
        image_payload = row["image"]
        with PILImage.open(io.BytesIO(image_payload["bytes"])) as image:
            image = image.convert("L")
            width, height = image.size
            image_tensor = pil_to_tensor(image).float() / 255.0

        boxes_xywh = row["objects"]["bbox"]
        raw_labels = row["objects"]["category"]

        boxes, class_ids = _sanitize_xywh_boxes(boxes_xywh, raw_labels)
        labels = class_ids + self.label_offset
        area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if len(boxes)
            else torch.zeros((0,), dtype=torch.float32)
        )
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "class_ids": class_ids,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([source_index], dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "size": torch.tensor([height, width], dtype=torch.int64),
        }
        if self.transform is not None:
            image_tensor, target = self.transform(image_tensor, target)
        return image_tensor, target


def detection_collate_fn(batch):
    """Minimal collate function for detection batches."""

    images, targets = zip(*batch)
    return list(images), list(targets)
