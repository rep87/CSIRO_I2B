"""Single-model inference to generate submission.csv."""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import torch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import BiomassDataset
from model_convnext import build_convnext
from model_regnet import build_regnet
from model_vit import build_vit
from transforms import build_transforms

MODEL_FACTORY = {
    "convnext": build_convnext,
    "regnet": build_regnet,
    "vit": build_vit,
}


def run_inference(
    csv_path: str,
    data_root: str,
    weight_path: str,
    model_name: str = "convnext",
    image_size: int = 224,
    batch_size: int = 32,
    target_columns: List[str] | None = None,
    submission_path: str = "submission.csv",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = build_transforms(image_size)["test"]

    dataset = BiomassDataset(
        csv_path=csv_path,
        data_root=data_root,
        target_columns=target_columns or [],
        transforms=transforms,
        is_train=False,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = MODEL_FACTORY[model_name](num_outputs=5, pretrained=False)
    state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()

    preds: List[np.ndarray] = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            preds.append(outputs)

    preds_np = np.vstack(preds)
    submission = pd.DataFrame(preds_np, columns=target_columns or [f"target_{i}" for i in range(preds_np.shape[1])])
    submission.insert(0, "Id", range(len(submission)))
    submission.to_csv(submission_path, index=False)


if __name__ == "__main__":
    run_inference(
        csv_path="__DATA_ROOT__/test.csv",
        data_root="__DATA_ROOT__",
        weight_path="weights/model.pth",
        model_name="convnext",
        image_size=224,
        batch_size=32,
        target_columns=["Dead_g", "Live_g", "Conifer_g", "Broadleaf_g", "GrassForbs_g"],
        submission_path="submission.csv",
    )
