# CS 171 Group Project - Food Price Prediction

Team 1: Nathan Montero, Suparn Posina, and Arushi Nirmal

This repository predicts USDA ERS Food-at-Home Monthly Area Prices using a leakage-aware chronological regression workflow. Model and feature choices are frozen for final submission.

## What the pipeline does

`src/pipeline.py` is the single script behind every result in the paper:

- EDA, dataset validation, and per-food-group IQR outlier analysis
- leakage-safe lag and rolling features, fit on the training split only
- chronological train/validation/test split (2013-2016 / 2017 / 2018)
- a mean baseline plus five tuned model families: Ridge Regression, Decision
  Tree, Random Forest, an MLP, and a PyTorch LSTM
- hyperparameter search for every model, with a quantified regularization-effect
  chart/table for Ridge's L2 penalty
- validation **and** held-out 2018 test metrics, training curves, feature
  importance, error analysis by food group/region, an end-to-end pipeline
  overview diagram, and wall-clock runtime tracking for the neural networks

`python src/pipeline.py --evaluate-test` is the standard run and is what
produced every table and figure in `outputs/`.

## Project structure

```text
src/pipeline.py                    Complete final pipeline (data -> features -> 5 models -> evaluation)
notebooks/
    CS171_Final_Colab.ipynb        Final-submission Colab notebook (run this one)
    checkpoint1/                   Archived Checkpoint 1 notebook, kept for the record
reports/
    CS171-FinalReport.docx         Final IEEE-format paper (export to PDF before submitting)
    references.md                  IEEE-style bibliography for the paper's References section
    checkpoint1/                   Archived Checkpoint 1 report (docx + pdf)
outputs/
    figures/, tables/              Every generated figure and table (see below)
    *.json                         Run metadata: EDA, outliers, splits, tuning grid, runtime
data/, models/                     Empty in Git; populated locally by running the pipeline
requirements.txt                   Pinned minimum dependency versions
```

`data/` and `models/` hold only `.gitkeep` in Git on purpose — the raw USDA
workbook and trained model binaries are large and fully reproducible, so they
are excluded (`.gitignore`) rather than committed.

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

All of the above regenerate deterministically from `python src/pipeline.py --evaluate-test` (seed 42) — see Reproducibility below.

## Colab setup

Open directly in Colab: https://colab.research.google.com/github/oops408/cs171project/blob/main/notebooks/CS171_Final_Colab.ipynb

Or from the repo: open `notebooks/CS171_Final_Colab.ipynb` in Google Colab and select **Runtime > Change runtime type > T4 GPU**. Run all cells. The notebook clones this repository, installs dependencies, runs the full pipeline with `--evaluate-test`, displays validation and test metrics, renders the key figures inline, and downloads the generated outputs.

The archived Checkpoint 1 run lives at `notebooks/checkpoint1/CS171_Checkpoint1_Colab.ipynb`.

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
python src/pipeline.py --evaluate-test
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/pipeline.py --evaluate-test
```

Faster smoke test (skips LSTM, uses smaller tuning grids, still evaluates test):

```bash
python src/pipeline.py --quick --skip-lstm --evaluate-test
```

Validation-only run (no test-set evaluation):

```bash
python src/pipeline.py
```

## Reproducibility

Every run uses `--seed 42` by default, applied to Python, NumPy, PyTorch, and CUDA. Independent full runs of `python src/pipeline.py --evaluate-test` on this repo have produced byte-for-byte identical validation and test metrics every time.

## Data source

U.S. Department of Agriculture, Economic Research Service, *Food-at-Home Monthly Area Prices*, 2024. Downloaded automatically by the pipeline from the URL in `DATA_URL` at the top of `src/pipeline.py`. Full citation in `reports/references.md`.
