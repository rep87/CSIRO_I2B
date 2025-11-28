# CSIRO Image2Biomass Starter

Lightweight template for the Kaggle **CSIRO – Image2Biomass Prediction** competition.

## Repository layout
- `src/`: core training code (datasets, transforms, models, utilities).
- `config/`: YAML configs with `__DATA_ROOT__` placeholders to override in Colab/Kaggle.
- `inference/`: single/ensemble inference scripts and Kaggle submission notebook.
- `notebook/`: local/Colab analysis notebooks.

## Training on Colab
1. Clone the repo in Colab and mount Google Drive.
2. Copy datasets to a Drive folder (e.g., `/content/drive/MyDrive/biomass`).
3. Override config values when launching training:
   ```bash
   python src/train.py --config config/convnext.yaml \
       --folds 5 \
       data_root=/content/drive/MyDrive/biomass \
       train_csv=/content/drive/MyDrive/biomass/train.csv
   ```
4. Weights and logs are saved to `output_dir` (also overridable).

## Kaggle inference
- Use `inference/inference_single.py` for quick runs or `inference/inference_ensemble.py` to blend predictions.
- In the Kaggle Notebook (`kaggle_submission_template.ipynb`), set `WEIGHT_PATH`, `TEST_CSV`, and `DATA_ROOT` to your dataset/weight locations, then run all cells to generate `submission.csv`.

## Data path override example
All configs default to `__DATA_ROOT__`. Replace it at runtime (env var, CLI args, or direct YAML edit) so the same code works in Colab, Kaggle, or local environments.
