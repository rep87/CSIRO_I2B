"""Image transformations for training/validation/test."""
from typing import Dict

import torchvision.transforms as T


def build_transforms(image_size: int) -> Dict[str, T.Compose]:
    """Create safe augmentations for each split."""

    train_transforms = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
        ]
    )

    valid_transforms = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]
    )

    test_transforms = valid_transforms

    return {
        "train": train_transforms,
        "valid": valid_transforms,
        "test": test_transforms,
    }
