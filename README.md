# CS 171 Group Project - Food Price Prediction

Team 1: Nathan Montero, Suparn Posina, and Arushi Nirmal

This repository predicts USDA ERS Food-at-Home Monthly Area Prices using a leakage-aware chronological regression workflow.

## Final submission

Model and feature choices are frozen; `python src/checkpoint1.py --evaluate-test` is now the
standard run and is what produced every table and figure in `outputs/`.

The pipeline performs:

- EDA, dataset validation, and per-food-group IQR outlier analysis
- leakage-safe lag and rolling features
- chronological train/validation/test splitting (2013-2016 / 2017 / 2018)
- a mean baseline plus five model families: Ridge Regression, Decision Tree,
  Random Forest, an MLP, and a PyTorch LSTM
- hyperparameter tuning for every model, with a quantified regularization-effect
  chart/table for Ridge's L2 penalty
- validation **and** held-out 2018 test metrics, training curves, feature
  importance, error analysis by food group/region, and wall-clock runtime
  tracking for the neural network models

## Output artifacts (`outputs/`)

| File | Contents |
|---|---|
| `tables/eda_summary.csv`, `figures/target_distribution.png`, `figures/monthly_average_trend.png` | Core EDA |
| `outlier_summary.json`, `tables/outlier_by_food_group.csv` | Per-food-group IQR outlier check (1.8% of rows flagged, retained as legitimate) |
| `tables/validation_metrics.csv`, `tables/test_metrics.csv` | Model comparison on the 2017 validation set and the reserved 2018 test set |
| `tuning_results.json`, `tables/regularization_tuning_grid.csv`, `figures/ridge_regularization_effect.png` | Full hyperparameter search grid and the Ridge L2-penalty effect |
| `figures/rf_actual_vs_predicted.png` / `..._test.png` | Random Forest actual-vs-predicted, validation and test |
| `figures/rf_error_food_groups.png` / `..._test.png`, `tables/rf_error_by_food_group*.csv`, `tables/rf_error_by_region*.csv` | Error analysis by food group / region |
| `figures/rf_feature_importance.png`, `tables/rf_feature_importance.csv` | Feature importance |
| `figures/mlp_training_curve.png`, `figures/lstm_training_curve.png` | Neural network training curves |
| `figures/overview_figure.png` | Pipeline overview diagram |
| `runtime_summary.json` | LSTM/MLP wall-clock training time and GPU/CPU device |
| `checkpoint_summary.json`, `split_summary.json` | Full run metadata (seed, versions, split sizes) |

All of the above regenerate deterministically from `python src/checkpoint1.py --evaluate-test`
(seed 42) — see Reproducibility below.

## Colab setup

https://colab.research.google.com/gist/oops408/0075f7e3ad256f783b278253088a1954/cs171_checkpoint1_colab.ipynb

Open `notebooks/CS171_Checkpoint1_Colab.ipynb` in Google Colab and select **Runtime > Change runtime type > T4 GPU**. Run all cells. The notebook clones this repository, installs dependencies, trains the models, and downloads the generated outputs.

## Local setup

```bash
git clone https://github.com/oops408/cs171project.git
cd cs171project
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/checkpoint1.py --evaluate-test
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/checkpoint1.py --evaluate-test
```

Faster smoke test (skips LSTM, uses smaller tuning grids, still evaluates test):

```bash
python src/checkpoint1.py --quick --skip-lstm --evaluate-test
```

Validation-only run (no test-set evaluation):

```bash
python src/checkpoint1.py
```

## Reproducibility

Every run uses `--seed 42` by default, applied to Python, NumPy, PyTorch, and CUDA. Two
independent full runs of `python src/checkpoint1.py --evaluate-test` on this repo produced
byte-for-byte identical validation and test metrics.

## Repository layout

```text
src/checkpoint1.py                          Complete reproducible pipeline
notebooks/CS171_Checkpoint1_Colab.ipynb     Colab notebook
reports/models/CS171-FinalReport.docx       Final IEEE-format paper (export to PDF before submitting)
reports/models/Checkpoint 1 - CS171 Project Report.pdf/.docx   Checkpoint 1 submission
outputs/figures/                            All generated figures
outputs/tables/                             All generated result tables
outputs/*.json                              Run metadata (EDA, outliers, splits, tuning, runtime)
```

The raw USDA workbook and trained model binaries are excluded from Git because they can be reproduced by running the pipeline.

## Data source

U.S. Department of Agriculture, Economic Research Service, *Food-at-Home Monthly Area Prices*, 2024. Downloaded automatically by the pipeline from the URL in `DATA_URL` at the top of `src/checkpoint1.py`.
