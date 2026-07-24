# CS 171 Group Project - Food Price Prediction

Team 1: Nathan Montero, Suparn Posina, and Arushi Nirmal

This repository predicts USDA ERS Food-at-Home Monthly Area Prices using a leakage-aware chronological regression workflow.

## Checkpoint 1

The pipeline performs:

- EDA and dataset validation
- leakage-safe lag and rolling features
- chronological train/validation/test splitting
- mean baseline
- Ridge Regression
- Decision Tree
- Random Forest
- MLP neural network
- PyTorch LSTM recurrent neural network
- validation metrics, training curves, feature importance, and error analysis

The 2018 test set is intentionally reserved by default. Do not use `--evaluate-test` until the team has frozen all model and feature choices.

## Colab setup

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
python src/checkpoint1.py
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/checkpoint1.py
```

Faster test:

```bash
python src/checkpoint1.py --quick --skip-lstm
```

Final test evaluation, only after freezing the project design:

```bash
python src/checkpoint1.py --evaluate-test
```

## Repository layout

```text
src/checkpoint1.py                       Complete reproducible pipeline
notebooks/CS171_Checkpoint1_Colab.ipynb  Colab Notebook
reports/CS171_Checkpoint1_Report.pdf     Submission Report
reports/CS171_Checkpoint1_Report.docx    Submission Report
outputs/figures/                         Preliminary Figures
outputs/tables/                          Preliminary Result Tables
```

The raw USDA workbook and trained model binaries are excluded from Git because they can be reproduced by running the pipeline.

## Data source

U.S. Department of Agriculture, Economic Research Service, Food-at-Home Monthly Area Prices, 2024.
