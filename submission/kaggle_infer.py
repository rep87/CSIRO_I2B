import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import BiomassPatchDataset
from src.model import PatchFusionModel
from src.utils import set_seed, tqdm


def collate_fn(batch):
    patch_count = len(batch[0]["patches"])
    patch_batches = []
    for idx in range(patch_count):
        stacked = torch.stack([sample["patches"][idx] for sample in batch])
        patch_batches.append(stacked)
    return patch_batches


def parse_args():
    parser = argparse.ArgumentParser(description="Kaggle submission inference")
    parser.add_argument("--metadata", type=str, required=True, help="CSV with image_path column")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="Folder containing test images")
    parser.add_argument("--patch_count", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--backbone", type=str, default="convnext_large")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=str, default="submission.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = pd.read_csv(args.metadata)

    dataset = BiomassPatchDataset(
        metadata,
        image_dir=args.data_root,
        patch_count=args.patch_count,
        image_size=args.image_size,
        augment_cfg={"horizontal_flip": False, "color_jitter": False},
        is_train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = PatchFusionModel(
        backbone_name=args.backbone,
        patch_count=args.patch_count,
        pretrained=False,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    predictions = []
    with torch.no_grad():
        for patches in tqdm(loader, desc="Kaggle Infer"):
            patches = [p.to(device) for p in patches]
            outputs = model(patches)
            predictions.append(outputs.cpu())
    preds = torch.cat(predictions, dim=0).numpy()

    submission = pd.DataFrame(preds, columns=["Dry", "Clover", "Green"])
    submission.insert(0, "image_path", metadata["image_path"].values)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()

