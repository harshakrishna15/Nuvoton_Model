"""Torch Dataset reading the merged YOLO-format directory produced by
``scripts/prepare_nuvoton_yolo_dataset.py``.

This lets the FOMO training/eval code consume the same merged corpus that the
YOLOv8 pipeline uses, instead of being tied to the original parquet snapshot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image as PILImage
from torchvision.transforms.functional import pil_to_tensor


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class YoloFolderDetectionDataset(torch.utils.data.Dataset):
    """Read a YOLO-style ``<root>/<split>/{images,labels}`` directory.

    Returns ``(image_tensor[1, H, W] in [0,1], target_dict)`` matching the
    ``OverheadPersonDetectionDataset`` contract so the existing FOMO training
    code can be reused unchanged.
    """

    class_names = ("person",)

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str,
        image_size: int = 192,
        transform: Callable | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = split
        self.image_size = image_size
        self.transform = transform

        self.images_dir = self.dataset_root / split / "images"
        self.labels_dir = self.dataset_root / split / "labels"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing split images dir: {self.images_dir}")

        self.image_paths = sorted(
            p for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_label_boxes_xyxy(self, label_path: Path, image_size: int) -> torch.Tensor:
        if not label_path.exists():
            return torch.zeros((0, 4), dtype=torch.float32)
        boxes: list[list[float]] = []
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            try:
                _cls = int(float(parts[0]))
                cx, cy, w, h = (float(x) for x in parts[1:5])
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            x1 = (cx - w / 2.0) * image_size
            y1 = (cy - h / 2.0) * image_size
            x2 = (cx + w / 2.0) * image_size
            y2 = (cy + h / 2.0) * image_size
            x1 = max(0.0, min(float(image_size), x1))
            y1 = max(0.0, min(float(image_size), y1))
            x2 = max(0.0, min(float(image_size), x2))
            y2 = max(0.0, min(float(image_size), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.image_paths[item]
        with PILImage.open(image_path) as image:
            image = image.convert("L")
            if image.size != (self.image_size, self.image_size):
                image = image.resize((self.image_size, self.image_size), PILImage.BILINEAR)
            image_tensor = pil_to_tensor(image).float() / 255.0

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        boxes = self._load_label_boxes_xyxy(label_path, self.image_size)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        class_ids = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if boxes.numel()
            else torch.zeros((0,), dtype=torch.float32)
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "class_ids": class_ids,
            "area": area,
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
            "image_id": torch.tensor([item], dtype=torch.int64),
            "orig_size": torch.tensor([self.image_size, self.image_size], dtype=torch.int64),
            "size": torch.tensor([self.image_size, self.image_size], dtype=torch.int64),
        }
        if self.transform is not None:
            image_tensor, target = self.transform(image_tensor, target)
        return image_tensor, target
