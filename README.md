# CSIRO I2B Patch-Based Pipeline

Patch-based ConvNeXt regression pipeline for the CSIRO Biomass Kaggle competition. The project is configured for fast experimentation by editing YAML configs without code changes.

## Directory Structure
```
./
├── configs/
│   └── train_config.yaml
├── src/
│   ├── dataset.py
│   ├── inference.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── submission/
│   └── kaggle_infer.py
├── experiments/
├── notebooks/
├── requirements.txt
└── README.md
```

## Dependencies
Install Python packages:
```bash
pip install -r requirements.txt
```

## Configuration (configs/train_config.yaml)
- **model.backbone**: `convnext_large` (default) or `convnext_base`.
- **model.patch_count**: number of image patches (1, 2, 4, or 6).
- **model.pretrained**: use ImageNet pretrained weights.
- **train.epochs / batch_size / lr / optimizer / scheduler / image_size**: core training knobs.
- **data.data_root**: folder containing images.
- **data.augment**: enable `color_jitter` and/or `horizontal_flip`.
- **loss.type**: `smooth_l1` (default), `huber`, `mae`, or `rmse`.
- **experiment.save_dir**: where dated experiment folders and checkpoints are stored.

## Patch-Based Feature Fusion
Images are split into N equal patches (grid mapping: 1→1x1, 2→1x2, 4→2x2, 6→2x3). Each patch is individually augmented and encoded by the ConvNeXt backbone. Patch features are concatenated along the channel dimension and passed through a regression head (Linear → GELU → Dropout → Linear) to predict Dry, Clover, and Green biomass values.

## Training
```bash
python -m src.train --config configs/train_config.yaml --metadata /path/to/meta.csv --cutoff_date 2023-06-01
```
- YAML is loaded for hyperparameters and augmentation.
- Date-based split uses the `date` column when provided; otherwise an 80/20 split is applied.
- Mixed precision (AMP) and tqdm progress bars are enabled.
- Checkpoints and logs are saved under a timestamped folder in `experiments/`.

## Inference
### Local/Notebook
```bash
python -m src.inference --config configs/train_config.yaml --metadata /path/to/test_meta.csv --checkpoint experiments/exp_xx/best.ckpt --output submission/submission.csv
```
- Uses the same patching logic as training.
- Generates `submission.csv` with predicted Dry, Clover, and Green columns.

### Kaggle Submission Helper
```bash
python -m submission.kaggle_infer --metadata /kaggle/input/meta.csv --checkpoint /kaggle/input/ckpt/best.ckpt --data_root /kaggle/input/images --output submission.csv
```
- Minimal script with no training dependencies; adjust `--patch_count`, `--image_size`, and `--backbone` as needed.

## Metadata Utilities
- `src.utils.create_date_split`: splits a DataFrame into train/val by cutoff date.
- `src.dataset.load_metadata`: loads CSV metadata with `image_path` (and optional `date`) columns.

## Recent Changes (UTC)
- 2025-02-06: Added patch-based ConvNeXt model with feature fusion, AMP training loop, and inference scripts. Updated README with usage instructions and directory map.

## Codex Build Report
- Created configs/train_config.yaml with default hyperparameters.
- Added patch-aware dataset loader, ConvNeXt fusion model, training, and inference utilities.
- Added Kaggle-only inference entry point and experiment scaffolding.
- Updated README with setup, config, training, inference, fusion description, and change log.
