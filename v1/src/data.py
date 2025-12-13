import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

TARGET_COLUMNS = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g"]
ALL_TARGET_COLUMNS = TARGET_COLUMNS + ["GDM_g", "Dry_Total_g"]


class RegressionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: str, image_size: int, augment: bool = False, use_targets: bool = True):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.target_cols = TARGET_COLUMNS
        self.use_targets = use_targets
        if augment:
            self.transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["image_path"])
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        image = self.transform(image)

        targets = None
        if self.use_targets:
            targets = torch.tensor(row[self.target_cols].values.astype("float32"))
        return image, targets, row.get("sample_id_prefix", None)


def load_long_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["sample_id_prefix"] = df["sample_id"].str.split("__").str[0]
    return df


def to_wide(df: pd.DataFrame, include_targets: bool = True) -> pd.DataFrame:
    columns = ["sample_id_prefix", "image_path", "Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"]
    if not include_targets:
        return df[columns].drop_duplicates().reset_index(drop=True)

    pivot_values = "target"
    wide = df.pivot_table(index=columns, columns="target_name", values=pivot_values, aggfunc="first").reset_index()
    missing = [c for c in TARGET_COLUMNS if c not in wide.columns]
    if missing:
        raise ValueError(f"Missing target columns after pivot: {missing}")
    return wide


def create_dataloaders(
    df: pd.DataFrame,
    train_indices: List[int],
    val_indices: List[int],
    image_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = RegressionDataset(df.iloc[train_indices], image_root, image_size, augment=True)
    val_ds = RegressionDataset(df.iloc[val_indices], image_root, image_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
