# v2: grouped CV + preprocessing controls

v2 keeps the v1 baseline layout while adding configurable CV grouping and image preprocessing. Default settings target strong leakage prevention (`group_date_state`) and slightly more robust visuals (bottom crop + CLAHE) while staying Colab/Kaggle friendly.

## What changed vs v1
- **CV grouping strategies**: choose sequential (original contiguous split) or grouped splits by `Sampling_Date`, `State`, or the combination (`Sampling_Date` + `State`).
- **Image preprocessing knobs**: optional bottom crop (`crop_bottom`) and CLAHE (`use_clahe`, with PIL fallback when OpenCV is unavailable).
- **Colab/Kaggle entry points**: `v2/colab_runner.py` defaults to grouped CV + preprocessing, and `v2/kaggle_submit_flat.ipynb` inlines the v2 inference stack with the same transforms.

## Key config switches
| Field | Default (v2) | Purpose / when to use | Notes |
| --- | --- | --- | --- |
| `cv_split_strategy` | `group_date_state` | GroupKFold keyed by `Sampling_Date + '_' + State` to avoid leakage across dates/states. | Falls back to `ValueError` if grouping columns are missing or not diverse. Use `sequential` to reproduce v1. |
| `crop_bottom` | `0.1` | Drops the bottom 10% of each image before resizing to trim soil-heavy regions. | Allowed range `[0.0, 0.3]`; set to `0.0` to disable. |
| `use_clahe` | `True` | Applies CLAHE on the L channel (LAB). Improves contrast on shaded plots. | If OpenCV is absent, PIL `ImageOps.equalize` is used. Disable on memory/time constrained runtimes. |

## How to run (Colab)
1. Mount Drive and place `train.csv`, `test.csv`, and image folders under your chosen `data_root`.
2. Edit the **PATHS** block in `v2/colab_runner.py` to point `data_root` to Drive.
3. Adjust hyperparameters in the **CONFIG** block. Example:
   ```python
   train_cfg = TrainConfig(
       backbone="efficientnet_b2",
       image_size=456,
       batch_size=32,
       folds=5,
       cv_split_strategy="group_state",  # or sequential/group_date/group_date_state
       crop_bottom=0.1,
       use_clahe=True,
   )
   ```
4. Run `!python v2/colab_runner.py` to execute grouped CV, save checkpoints, and generate a submission.

## How to use on Kaggle (inference-only)
- Upload trained weights as a Dataset (e.g., `v2_weights/fold*_best.pth`).
- Open `v2/kaggle_submit_flat.ipynb`, edit `WEIGHTS_ROOT` at the top, and adjust `CROP_BOTTOM` / `USE_CLAHE` if your training used different values.
- Run all cells; the notebook rebuilds the v2 inference pipeline inline, saves `submission.csv` to both `/kaggle/working/submission.csv` and the run directory.

## Notes & cautions
- `Sampling_Date` / `State` metadata exist only in `train.csv`; grouped splits apply to CV only and are not used during test-time predictions.
- CLAHE adds minor CPU overhead; on small Kaggle machines, disable (`use_clahe=False`) if runtime is tight.
- v2 keeps v1 features intact (run summaries, efficientnet-b2 backbone, weighted R² metric) while tightening CV leakage controls.
