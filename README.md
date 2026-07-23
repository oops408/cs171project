# CS 171 Group Project - Food Price Prediction

Team 1: Nathan Montero, Suparn Posina, and Arushi Nirmal

This repository predicts USDA ERS Food-at-Home Monthly Area Prices using a leakage-aware chronological regression workflow.

## Checkpoint 1

The pipeline performs EDA, leakage-safe lag and rolling features, a chronological split, a mean baseline, Ridge Regression, Decision Tree, Random Forest, MLP, and a PyTorch LSTM. It saves validation metrics, training curves, feature importance, and error analysis.

The 2018 test set is intentionally reserved by default. Do not use `--evaluate-test` until all model and feature choices are set.

## Colab setup

Open `notebooks/CS171_Checkpoint1_Colab.ipynb` in Google Colab, select **Runtime > Change runtime type > T4 GPU**, and run all cells.

## Local setup

```bash
git clone https://github.com/oops408/cs171project.git
cd cs171project
git switch checkpoint-1
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/checkpoint1.py
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/checkpoint1.py
```

Quick Test:

```bash
python src/checkpoint1.py --quick --skip-lstm
```

Final Test Evaluation:

```bash
python src/checkpoint1.py --evaluate-test
```

## Data source

U.S. Department of Agriculture, Economic Research Service, Food-at-Home Monthly Area Prices, 2024.
