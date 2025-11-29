import os
from typing import Callable, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


PATCH_GRID = {
    1: (1, 1),
    2: (1, 2),
    4: (2, 2),
    6: (2, 3),
}


class BiomassPatchDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        image_dir: str,
        patch_count: int,
        image_size: int,
        augment_cfg: Optional[Dict] = None,
        is_train: bool = True,
        target_columns: Optional[List[str]] = None,
    ) -> None:
        if patch_count not in PATCH_GRID:
            raise ValueError(f"Unsupported patch_count: {patch_count}")
        self.metadata = metadata.reset_index(drop=True)
        self.image_dir = image_dir
        self.patch_count = patch_count
        self.grid = PATCH_GRID[patch_count]
        self.image_size = image_size
        self.is_train = is_train
        self.target_columns = target_columns or ["Dry", "Clover", "Green"]
        print("[INFO] Dataset columns:", list(self.metadata.columns))
        for col in self.target_columns:
            if col not in self.metadata.columns:
                raise ValueError(
                    f"Expected target column '{col}' not found in dataframe columns: {list(self.metadata.columns)}"
                )
        self.transform = self._build_transform(augment_cfg or {})

    def _build_transform(self, augment_cfg: Dict) -> Callable:
        transforms: List[A.BasicTransform] = [A.Resize(self.image_size, self.image_size)]
        if augment_cfg.get("color_jitter", False):
            transforms.append(A.ColorJitter())
        if augment_cfg.get("horizontal_flip", False):
            transforms.append(A.HorizontalFlip(p=0.5))
        transforms.extend([A.Normalize(), ToTensorV2()])
        return A.Compose(transforms)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.metadata.iloc[idx]
        image_path = os.path.join(self.image_dir, row["image_path"])
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        patches = self._split_into_patches(image)
        augmented_patches = [self.transform(image=patch)["image"] for patch in patches]

        item: Dict[str, torch.Tensor] = {"patches": augmented_patches}
        if self.is_train:
            target = torch.tensor([row[col] for col in self.target_columns], dtype=torch.float32)
            item["target"] = target
        return item

    def _split_into_patches(self, image: np.ndarray) -> List[np.ndarray]:
        rows, cols = self.grid
        h, w, _ = image.shape
        patch_h, patch_w = h // rows, w // cols
        patches = []
        for r in range(rows):
            for c in range(cols):
                y0, y1 = r * patch_h, (r + 1) * patch_h
                x0, x1 = c * patch_w, (c + 1) * patch_w
                patch = image[y0:y1, x0:x1]
                patches.append(patch)
        return patches


def load_metadata(csv_path: str) -> pd.DataFrame:
    metadata = pd.read_csv(csv_path)
    if "date" in metadata.columns:
        metadata["date"] = pd.to_datetime(metadata["date"], errors="coerce")
    if "image_path" not in metadata.columns:
        raise ValueError("metadata CSV must contain an 'image_path' column")
    return metadata


