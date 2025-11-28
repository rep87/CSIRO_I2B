"""Dataset utilities for CSIRO Image2Biomass regression."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Sample:
    """Container for a dataset sample."""

    image_path: str
    targets: Optional[List[float]]
    metadata: Dict[str, str]


class BiomassDataset(Dataset):
    """Dataset supporting train/validation/test splits with grouping metadata.

    The dataset expects a CSV file with columns such as:
    - ``image_path``: relative path to the image file from ``data_root``
    - ``Sampling_Date``: string identifier used for GroupKFold
    - target columns: e.g., ``Dead_g``, ``Live_g``, etc. (5 outputs total)

    Args:
        csv_path: Path to the metadata CSV.
        data_root: Base directory where images reside (placeholder friendly).
        target_columns: List of column names representing the regression targets.
        transforms: Callable applied to the loaded PIL image.
        is_train: If ``True``, dataset returns targets; otherwise test mode.
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        target_columns: Optional[List[str]] = None,
        transforms: Optional[Callable] = None,
        is_train: bool = True,
    ) -> None:
        self.csv_path = csv_path
        self.data_root = data_root
        self.target_columns = target_columns or []
        self.transforms = transforms
        self.is_train = is_train

        self.df = pd.read_csv(self.csv_path)
        if self.is_train and not self.target_columns:
            raise ValueError("target_columns must be provided for training mode")

        self.samples: List[Sample] = []
        for _, row in self.df.iterrows():
            image_path = os.path.join(self.data_root, row["image_path"])
            metadata = {"Sampling_Date": str(row.get("Sampling_Date", ""))}
            if self.is_train:
                targets = [float(row[col]) for col in self.target_columns]
            else:
                targets = None
            self.samples.append(Sample(image_path=image_path, targets=targets, metadata=metadata))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        if self.is_train:
            return image, sample.targets, sample.metadata
        return image, sample.metadata

    def get_groups(self) -> List[str]:
        """Return group labels for GroupKFold based on Sampling_Date."""

        return [sample.metadata.get("Sampling_Date", "") for sample in self.samples]
