#!/usr/bin/env python3
"""CS 171 Group Project - Checkpoint 1 pipeline.

This script downloads the USDA ERS Food-at-Home Monthly Area Prices dataset,
performs EDA, creates leakage-safe lag features, uses a chronological split,
trains several regression model families, and saves tables/figures for the
checkpoint report.

Default split:
    training targets:   2013-2016
    validation targets: 2017
    reserved test:      2018

The raw 2012 observations are used only as historical context for 12-month lag
features. By default, the script does not report 2018 test metrics, so the test
set remains untouched for the final paper. Use --evaluate-test only after the
team has frozen all modeling and tuning decisions.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from torch.utils.data import DataLoader, TensorDataset

DATA_URL = (
    "https://www.ers.usda.gov/media/5399/"
    "food-at-home-monthly-area-prices-2012-to-2018.xlsx?v=93448"
)
TARGET = "Unit_value_mean_wtd"
KEY_COLUMNS = ["Year", "Month", "EFPG_code", "Metroregion_code"]
CATEGORICAL_FEATURES = ["EFPG_name", "Metroregion_name"]
NUMERIC_FEATURES = [
    "Year",
    "Month",
    "month_sin",
    "month_cos",
    "Number_stores",
    "price_lag_1",
    "price_lag_3",
    "price_lag_12",
    "price_roll_3",
    "price_roll_12",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@dataclass
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass
class LSTMData:
    sequences: np.ndarray
    static_features: np.ndarray
    food_ids: np.ndarray
    region_ids: np.ndarray
    targets: np.ndarray
    years: np.ndarray
    price_mean: float
    price_std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/fmap.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--data-url", default=DATA_URL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip the PyTorch LSTM for a faster local smoke test.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller tuning grids and fewer LSTM epochs.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the reserved 2018 test set. Use only after model choices are frozen.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def download_dataset(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"Using existing dataset: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset to {path} ...")
    try:
        with requests.get(url, stream=True, timeout=180) as response:
            response.raise_for_status()
            with path.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_handle.write(chunk)
    except requests.RequestException as exc:
        raise RuntimeError(
            "Dataset download failed. Download the XLSX manually from the USDA ERS "
            f"F-MAP page and place it at {path}. Original error: {exc}"
        ) from exc
    print(f"Downloaded {path.stat().st_size / 1_000_000:.1f} MB")


def load_data(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_excel(path, sheet_name="Data", engine="openpyxl")
    except Exception as exc:
        raise RuntimeError(f"Could not read the Data sheet from {path}: {exc}") from exc

    required = {
        "Year",
        "Month",
        "EFPG_name",
        "EFPG_code",
        "Metroregion_name",
        "Metroregion_code",
        "Number_stores",
        TARGET,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return frame


def summarize_eda(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "first_year": int(frame["Year"].min()),
        "last_year": int(frame["Year"].max()),
        "unique_months": int(frame[["Year", "Month"]].drop_duplicates().shape[0]),
        "food_groups": int(frame["EFPG_name"].nunique()),
        "geographic_areas": int(frame["Metroregion_name"].nunique()),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_keys": int(frame.duplicated(KEY_COLUMNS).sum()),
        "target_mean": float(frame[TARGET].mean()),
        "target_median": float(frame[TARGET].median()),
        "target_standard_deviation": float(frame[TARGET].std()),
        "target_minimum": float(frame[TARGET].min()),
        "target_maximum": float(frame[TARGET].max()),
        "target_skewness": float(frame[TARGET].skew()),
    }


def analyze_outliers(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Flag target outliers per food group using a 1.5x IQR fence.

    A single global fence is inappropriate here: food groups span very
    different price scales (e.g. spices vs. milk), so a per-group fence
    separates genuinely unusual observations from food groups that are
    simply always expensive.
    """
    flags = pd.Series(False, index=frame.index)
    rows: list[dict[str, Any]] = []
    for name, group in frame.groupby("EFPG_name"):
        first_quartile, third_quartile = group[TARGET].quantile([0.25, 0.75])
        iqr = third_quartile - first_quartile
        low, high = first_quartile - 1.5 * iqr, third_quartile + 1.5 * iqr
        group_flags = (group[TARGET] < low) | (group[TARGET] > high)
        flags.loc[group.index] = group_flags
        rows.append(
            {
                "EFPG_name": name,
                "n_observations": int(len(group)),
                "n_outliers": int(group_flags.sum()),
                "outlier_rate": float(group_flags.mean()),
                "group_median_price": float(group[TARGET].median()),
            }
        )
    outlier_table = (
        pd.DataFrame(rows)
        .sort_values("outlier_rate", ascending=False)
        .reset_index(drop=True)
    )
    summary = {
        "method": "Per-food-group 1.5x IQR fence on Unit_value_mean_wtd",
        "total_rows": int(len(frame)),
        "total_outliers": int(flags.sum()),
        "outlier_rate": float(flags.mean()),
        "decision": (
            "Retained: flagged points are legitimately high/low-priced food "
            "groups (e.g. spices, infant formula) rather than data-entry "
            "errors, and the lag/rolling features are computed per series "
            "so within-group outliers do not leak across food groups."
        ),
    }
    return summary, outlier_table


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values(
        ["EFPG_code", "Metroregion_code", "Year", "Month"]
    ).copy()
    data["date"] = pd.to_datetime(
        {"year": data["Year"], "month": data["Month"], "day": 1}
    )
    grouped_target = data.groupby(
        ["EFPG_code", "Metroregion_code"], sort=False
    )[TARGET]

    for lag in (1, 3, 12):
        data[f"price_lag_{lag}"] = grouped_target.shift(lag)
    data["price_roll_3"] = grouped_target.transform(
        lambda series: series.shift(1).rolling(3).mean()
    )
    data["price_roll_12"] = grouped_target.transform(
        lambda series: series.shift(1).rolling(12).mean()
    )
    data["month_sin"] = np.sin(2 * np.pi * data["Month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["Month"] / 12.0)

    lag_columns = [
        "price_lag_1",
        "price_lag_3",
        "price_lag_12",
        "price_roll_3",
        "price_roll_12",
    ]
    return data.dropna(subset=lag_columns).reset_index(drop=True)


def chronological_split(model_frame: pd.DataFrame) -> SplitData:
    split = SplitData(
        train=model_frame[model_frame["Year"] <= 2016].copy(),
        validation=model_frame[model_frame["Year"] == 2017].copy(),
        test=model_frame[model_frame["Year"] == 2018].copy(),
    )
    expected = (64_800, 16_200, 16_200)
    actual = (len(split.train), len(split.validation), len(split.test))
    if actual != expected:
        warnings.warn(
            f"Unexpected split sizes {actual}; expected {expected}. Check the source file."
        )
    return split


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    return {
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def fit_tabular_models(
    split: SplitData, seed: int, quick: bool
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    x_train, y_train = split.train[FEATURES], split.train[TARGET]
    x_val, y_val = split.validation[FEATURES], split.validation[TARGET]

    one_hot_sparse = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    one_hot_dense = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    ordinal = ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )

    fitted: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    tuning: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []

    def evaluate(name: str, model: Any) -> dict[str, float]:
        prediction = model.predict(x_val)
        predictions[name] = prediction
        scores = regression_metrics(y_val, prediction)
        metric_rows.append({"Model": name, **scores})
        return scores

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(x_train, y_train)
    fitted["Mean baseline"] = baseline
    evaluate("Mean baseline", baseline)

    ridge_alphas = [1.0, 10.0] if quick else [0.1, 1.0, 10.0, 100.0]
    ridge_trials = []
    ridge_candidates = {}
    for alpha in ridge_alphas:
        model = Pipeline(
            [("preprocess", one_hot_sparse), ("regressor", Ridge(alpha=alpha))]
        )
        start = time.perf_counter()
        model.fit(x_train, y_train)
        scores = regression_metrics(y_val, model.predict(x_val))
        ridge_trials.append({"alpha": alpha, **scores, "seconds": time.perf_counter() - start})
        ridge_candidates[alpha] = model
    best_ridge_trial = min(ridge_trials, key=lambda row: row["RMSE"])
    best_ridge = ridge_candidates[best_ridge_trial["alpha"]]
    fitted["Ridge"] = best_ridge
    evaluate("Ridge", best_ridge)
    tuning["Ridge"] = ridge_trials

    depth_values = [5, 10] if quick else [5, 10, 20, None]
    leaf_values = [5, 20] if quick else [1, 5, 20]
    tree_trials = []
    tree_candidates = {}
    for depth in depth_values:
        for leaf in leaf_values:
            model = Pipeline(
                [
                    ("preprocess", ordinal),
                    (
                        "regressor",
                        DecisionTreeRegressor(
                            max_depth=depth,
                            min_samples_leaf=leaf,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            start = time.perf_counter()
            model.fit(x_train, y_train)
            scores = regression_metrics(y_val, model.predict(x_val))
            key = (depth, leaf)
            tree_trials.append(
                {
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    **scores,
                    "seconds": time.perf_counter() - start,
                }
            )
            tree_candidates[key] = model
    best_tree_trial = min(tree_trials, key=lambda row: row["RMSE"])
    best_tree = tree_candidates[
        (best_tree_trial["max_depth"], best_tree_trial["min_samples_leaf"])
    ]
    fitted["Decision Tree"] = best_tree
    evaluate("Decision Tree", best_tree)
    tuning["Decision Tree"] = tree_trials

    forest_trees = [50] if quick else [50, 100]
    forest_depths = [10, 20] if quick else [10, 20, None]
    forest_trials = []
    forest_candidates = {}
    for n_estimators in forest_trees:
        for depth in forest_depths:
            model = Pipeline(
                [
                    ("preprocess", ordinal),
                    (
                        "regressor",
                        RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=depth,
                            min_samples_leaf=2,
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            start = time.perf_counter()
            model.fit(x_train, y_train)
            scores = regression_metrics(y_val, model.predict(x_val))
            key = (n_estimators, depth)
            forest_trials.append(
                {
                    "n_estimators": n_estimators,
                    "max_depth": depth,
                    "min_samples_leaf": 2,
                    **scores,
                    "seconds": time.perf_counter() - start,
                }
            )
            forest_candidates[key] = model
    best_forest_trial = min(forest_trials, key=lambda row: row["RMSE"])
    best_forest = forest_candidates[
        (best_forest_trial["n_estimators"], best_forest_trial["max_depth"])
    ]
    fitted["Random Forest"] = best_forest
    evaluate("Random Forest", best_forest)
    tuning["Random Forest"] = forest_trials

    mlp_configs = [((64,), 1e-4)] if quick else [
        ((64,), 1e-4),
        ((64, 32), 1e-4),
        ((64, 32), 1e-3),
    ]
    mlp_trials = []
    mlp_candidates = {}
    for hidden_layers, alpha in mlp_configs:
        model = Pipeline(
            [
                ("preprocess", one_hot_dense),
                (
                    "regressor",
                    MLPRegressor(
                        hidden_layer_sizes=hidden_layers,
                        activation="relu",
                        solver="adam",
                        alpha=alpha,
                        learning_rate_init=1e-3,
                        batch_size=256,
                        max_iter=100 if quick else 150,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=10,
                        random_state=seed,
                    ),
                ),
            ]
        )
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_train)
        scores = regression_metrics(y_val, model.predict(x_val))
        regressor = model.named_steps["regressor"]
        key = (hidden_layers, alpha)
        mlp_trials.append(
            {
                "hidden_layers": str(hidden_layers),
                "alpha": alpha,
                "epochs": int(regressor.n_iter_),
                **scores,
                "seconds": time.perf_counter() - start,
            }
        )
        mlp_candidates[key] = model
    best_mlp_trial = min(mlp_trials, key=lambda row: row["RMSE"])
    best_key = None
    for key in mlp_candidates:
        if str(key[0]) == best_mlp_trial["hidden_layers"] and key[1] == best_mlp_trial["alpha"]:
            best_key = key
            break
    if best_key is None:
        raise RuntimeError("Could not identify the selected MLP configuration.")
    best_mlp = mlp_candidates[best_key]
    fitted["MLP"] = best_mlp
    evaluate("MLP", best_mlp)
    tuning["MLP"] = mlp_trials

    metrics_frame = pd.DataFrame(metric_rows).sort_values("RMSE").reset_index(drop=True)
    return fitted, metrics_frame, predictions, tuning


class PriceLSTM(nn.Module):
    def __init__(
        self,
        number_foods: int,
        number_regions: int,
        hidden_size: int = 48,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.food_embedding = nn.Embedding(number_foods, 12)
        self.region_embedding = nn.Embedding(number_regions, 4)
        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 12 + 4 + 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        static: torch.Tensor,
        food_id: torch.Tensor,
        region_id: torch.Tensor,
    ) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence)
        combined = torch.cat(
            [
                hidden[-1],
                self.food_embedding(food_id),
                self.region_embedding(region_id),
                static,
            ],
            dim=1,
        )
        return self.head(combined)


def build_lstm_data(raw_feature_frame: pd.DataFrame) -> LSTMData:
    """Create 12-month sequences efficiently, one complete series at a time."""
    sequence_length = 12
    food_names = sorted(raw_feature_frame["EFPG_name"].unique())
    region_names = sorted(raw_feature_frame["Metroregion_name"].unique())
    food_to_id = {name: index for index, name in enumerate(food_names)}
    region_to_id = {name: index for index, name in enumerate(region_names)}

    training_history = raw_feature_frame[raw_feature_frame["Year"] <= 2016]
    price_mean = float(training_history[TARGET].mean())
    price_std = float(training_history[TARGET].std())
    stores_mean = float(training_history["Number_stores"].mean())
    stores_std = float(training_history["Number_stores"].std())

    sequence_batches: list[np.ndarray] = []
    static_batches: list[np.ndarray] = []
    food_batches: list[np.ndarray] = []
    region_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    year_batches: list[np.ndarray] = []

    for _, group in raw_feature_frame.groupby(
        ["EFPG_code", "Metroregion_code"], sort=False
    ):
        group = group.sort_values("date").reset_index(drop=True)
        sequence_source = np.column_stack(
            [
                (group[TARGET].to_numpy(dtype=np.float32) - price_mean) / price_std,
                (group["Number_stores"].to_numpy(dtype=np.float32) - stores_mean) / stores_std,
                group["month_sin"].to_numpy(dtype=np.float32),
                group["month_cos"].to_numpy(dtype=np.float32),
            ]
        ).astype(np.float32)

        # sliding_window_view returns (windows, features, time); transpose to
        # the PyTorch convention (samples, time, features).  The final window
        # has no following target, so it is removed.
        windows = np.lib.stride_tricks.sliding_window_view(
            sequence_source, window_shape=sequence_length, axis=0
        )[:-1].transpose(0, 2, 1).copy()
        current = group.iloc[sequence_length:]
        static = np.column_stack(
            [
                (current["Year"].to_numpy(dtype=np.float32) - 2013.0) / 5.0,
                current["month_sin"].to_numpy(dtype=np.float32),
                current["month_cos"].to_numpy(dtype=np.float32),
                (current["Number_stores"].to_numpy(dtype=np.float32) - stores_mean)
                / stores_std,
            ]
        ).astype(np.float32)

        sample_count = len(current)
        sequence_batches.append(windows)
        static_batches.append(static)
        food_batches.append(
            np.full(sample_count, food_to_id[current.iloc[0]["EFPG_name"]], dtype=np.int64)
        )
        region_batches.append(
            np.full(
                sample_count,
                region_to_id[current.iloc[0]["Metroregion_name"]],
                dtype=np.int64,
            )
        )
        target_batches.append(
            ((current[TARGET].to_numpy(dtype=np.float32) - price_mean) / price_std).astype(
                np.float32
            )
        )
        year_batches.append(current["Year"].to_numpy(dtype=np.int16))

    return LSTMData(
        sequences=np.concatenate(sequence_batches, axis=0),
        static_features=np.concatenate(static_batches, axis=0),
        food_ids=np.concatenate(food_batches),
        region_ids=np.concatenate(region_batches),
        targets=np.concatenate(target_batches),
        years=np.concatenate(year_batches),
        price_mean=price_mean,
        price_std=price_std,
    )


def make_lstm_loader(
    data: LSTMData, mask: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(data.sequences[mask]),
        torch.tensor(data.static_features[mask]),
        torch.tensor(data.food_ids[mask]),
        torch.tensor(data.region_ids[mask]),
        torch.tensor(data.targets[mask]).unsqueeze(1),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_lstm(
    data: LSTMData, seed: int, quick: bool
) -> tuple[PriceLSTM, dict[str, list[float]], dict[str, float], np.ndarray]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LSTM device: {device}")

    train_mask = data.years <= 2016
    validation_mask = data.years == 2017
    train_loader = make_lstm_loader(data, train_mask, 512, True)
    validation_loader = make_lstm_loader(data, validation_mask, 1024, False)

    model = PriceLSTM(
        number_foods=int(data.food_ids.max()) + 1,
        number_regions=int(data.region_ids.max()) + 1,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    maximum_epochs = 12 if quick else 30
    patience = 4 if quick else 5
    best_validation_loss = float("inf")
    best_state = None
    bad_epochs = 0
    history = {"train_loss": [], "validation_loss": []}

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for sequence, static, food_id, region_id, target_values in train_loader:
            sequence = sequence.to(device)
            static = static.to(device)
            food_id = food_id.to(device)
            region_id = region_id.to(device)
            target_values = target_values.to(device)

            optimizer.zero_grad()
            prediction = model(sequence, static, food_id, region_id)
            loss = criterion(prediction, target_values)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(target_values)
            seen += len(target_values)
        training_loss = running_loss / seen

        model.eval()
        running_loss = 0.0
        seen = 0
        with torch.no_grad():
            for sequence, static, food_id, region_id, target_values in validation_loader:
                prediction = model(
                    sequence.to(device),
                    static.to(device),
                    food_id.to(device),
                    region_id.to(device),
                )
                loss = criterion(prediction, target_values.to(device))
                running_loss += float(loss.item()) * len(target_values)
                seen += len(target_values)
        validation_loss = running_loss / seen
        history["train_loss"].append(training_loss)
        history["validation_loss"].append(validation_loss)
        print(
            f"LSTM epoch {epoch:02d}: train MSE={training_loss:.6f}, "
            f"validation MSE={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss - 1e-5:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("LSTM training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)

    true_values, predictions = predict_lstm(
        model, validation_loader, device, data.price_mean, data.price_std
    )
    scores = regression_metrics(true_values, predictions)
    return model, history, scores, predictions


def predict_lstm(
    model: PriceLSTM,
    loader: DataLoader,
    device: torch.device,
    price_mean: float,
    price_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    prediction_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    with torch.no_grad():
        for sequence, static, food_id, region_id, target_values in loader:
            prediction = model(
                sequence.to(device),
                static.to(device),
                food_id.to(device),
                region_id.to(device),
            )
            prediction_batches.append(prediction.cpu().numpy().ravel())
            target_batches.append(target_values.numpy().ravel())
    standardized_predictions = np.concatenate(prediction_batches)
    standardized_targets = np.concatenate(target_batches)
    predictions = standardized_predictions * price_std + price_mean
    targets = standardized_targets * price_std + price_mean
    return targets, predictions


def save_overview_figure(figure_dir: Path) -> None:
    """Render a pipeline overview diagram: data -> features -> split -> models -> evaluation."""
    figure_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        ("F-MAP Raw Data\n113,400 rows\n2012-2018", "#cfe2f3"),
        ("Feature Engineering\nlags, rolling means,\ncyclical month", "#d9ead3"),
        ("Chronological Split\ntrain 2013-16 | val 2017\ntest 2018 (held out)", "#fff2cc"),
        ("5 Model Families\nRidge, Tree, Forest,\nMLP, LSTM", "#f4cccc"),
        ("Evaluation\nRMSE/MAE/R2,\nerror analysis", "#d0e0e3"),
    ]

    fig, ax = plt.subplots(figsize=(11.5, 2.6))
    box_width, box_height = 1.9, 1.5
    gap = 0.55
    x = 0.0
    centers = []
    for label, color in stages:
        rect = plt.Rectangle(
            (x, 0), box_width, box_height,
            facecolor=color, edgecolor="black", linewidth=1.0,
        )
        ax.add_patch(rect)
        ax.text(
            x + box_width / 2, box_height / 2, label,
            ha="center", va="center", fontsize=9.5, wrap=True,
        )
        centers.append(x + box_width)
        x += box_width + gap

    for start in centers[:-1]:
        ax.annotate(
            "", xy=(start + gap, box_height / 2), xytext=(start, box_height / 2),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.4, color="black"),
        )

    ax.set_xlim(-0.2, x - gap + 0.2)
    ax.set_ylim(-0.2, box_height + 0.2)
    ax.axis("off")
    ax.set_title("End-to-End Pipeline Overview", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(figure_dir / "overview_figure.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_eda_figures(frame: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.2, 4.4))
    plt.hist(frame[TARGET], bins=50, edgecolor="black", linewidth=0.3)
    plt.axvline(
        frame[TARGET].mean(),
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {frame[TARGET].mean():.3f}",
    )
    plt.axvline(
        frame[TARGET].median(),
        linestyle=":",
        linewidth=1.5,
        label=f"Median = {frame[TARGET].median():.3f}",
    )
    first_quartile, third_quartile = frame[TARGET].quantile([0.25, 0.75])
    iqr = third_quartile - first_quartile
    high_fence = third_quartile + 1.5 * iqr
    plt.axvline(
        high_fence,
        linestyle="-.",
        linewidth=1.5,
        color="red",
        label=f"Global 1.5x IQR fence = {high_fence:.2f}",
    )
    plt.xlabel("Weighted average unit price (dollars per 100 grams)")
    plt.ylabel("Observations")
    plt.title("Distribution of the Regression Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "target_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    monthly = frame.assign(
        date=pd.to_datetime({"year": frame["Year"], "month": frame["Month"], "day": 1})
    ).groupby("date")[TARGET].mean()
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(monthly.index, monthly.values, linewidth=1.5)
    plt.xlabel("Month")
    plt.ylabel("Mean weighted unit price")
    plt.title("Average F-MAP Unit Price Over Time")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "monthly_average_trend.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(
    actual: np.ndarray, predicted: np.ndarray, title: str, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 5.2))
    plt.scatter(actual, predicted, s=7, alpha=0.25)
    limits = [
        min(float(actual.min()), float(predicted.min())),
        max(float(actual.max()), float(predicted.max())),
    ]
    plt.plot(limits, limits, linestyle="--", linewidth=1.2)
    plt.xlim(limits)
    plt.ylim(limits)
    plt.xlabel("Actual weighted unit price")
    plt.ylabel("Predicted weighted unit price")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_model_figures(
    metrics_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    random_forest_prediction: np.ndarray,
    random_forest_model: Pipeline,
    mlp_model: Pipeline,
    figure_dir: Path,
    lstm_history: dict[str, list[float]] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_metrics = metrics_frame.sort_values("RMSE", ascending=True)
    plt.figure(figsize=(7.2, 4.4))
    bars = plt.barh(plot_metrics["Model"], plot_metrics["RMSE"])
    plt.xlabel("Validation RMSE (lower is better)")
    plt.title("Preliminary Model Comparison (2017 Validation Set)")
    maximum = float(plot_metrics["RMSE"].max())
    for bar, value in zip(bars, plot_metrics["RMSE"]):
        offset = 0.004 if value > 0.1 else 0.0006
        plt.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=9,
        )
    plt.xlim(0, maximum * 1.14)
    plt.tight_layout()
    plt.savefig(figure_dir / "model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    plot_actual_vs_predicted(
        validation_frame[TARGET].to_numpy(),
        random_forest_prediction,
        "Random Forest: Actual vs. Predicted (2017 Validation Set)",
        figure_dir / "rf_actual_vs_predicted.png",
    )

    forest_regressor = random_forest_model.named_steps["regressor"]
    importance = pd.Series(
        forest_regressor.feature_importances_,
        index=CATEGORICAL_FEATURES + NUMERIC_FEATURES,
        name="importance",
    ).sort_values(ascending=False)
    importance.head(10).sort_values().plot(kind="barh", figsize=(7.2, 4.4))
    plt.xlabel("Random Forest impurity importance")
    plt.title("Preliminary Feature Importance")
    plt.tight_layout()
    plt.savefig(figure_dir / "rf_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()

    food_error, region_error = compute_error_breakdown(
        validation_frame, random_forest_prediction
    )
    save_food_error_figure(
        food_error,
        "Food Groups With Highest Random Forest Error (2017 Validation Set)",
        figure_dir / "rf_error_food_groups.png",
    )

    mlp_regressor = mlp_model.named_steps["regressor"]
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(range(1, len(mlp_regressor.loss_curve_) + 1), mlp_regressor.loss_curve_)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("MLP Training Curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "mlp_training_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    if lstm_history is not None:
        plt.figure(figsize=(7.2, 4.2))
        epochs = range(1, len(lstm_history["train_loss"]) + 1)
        plt.plot(epochs, lstm_history["train_loss"], label="Training loss")
        plt.plot(epochs, lstm_history["validation_loss"], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE on standardized target")
        plt.title("LSTM Training Curve")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / "lstm_training_curve.png", dpi=200, bbox_inches="tight")
        plt.close()

    return importance.to_frame(), food_error, region_error


def save_regularization_effect(
    tuning: dict[str, Any], figure_dir: Path, table_dir: Path
) -> pd.DataFrame:
    ridge_trials = pd.DataFrame(tuning["Ridge"]).sort_values("alpha").reset_index(drop=True)
    ridge_trials.to_csv(table_dir / "ridge_regularization_effect.csv", index=False)

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(ridge_trials["alpha"], ridge_trials["RMSE"], marker="o")
    plt.xscale("log")
    plt.xlabel("Ridge L2 penalty (alpha, log scale)")
    plt.ylabel("Validation RMSE")
    plt.title("Effect of L2 Regularization Strength (Ridge Regression)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        figure_dir / "ridge_regularization_effect.png", dpi=200, bbox_inches="tight"
    )
    plt.close()

    combined_rows = []
    for model_name, trials in tuning.items():
        for trial in trials:
            combined_rows.append({"model": model_name, **trial})
    pd.DataFrame(combined_rows).to_csv(
        table_dir / "regularization_tuning_grid.csv", index=False
    )
    return ridge_trials


def compute_error_breakdown(
    frame: pd.DataFrame, prediction: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    error_frame = frame[["EFPG_name", "Metroregion_name", "Year", "Month", TARGET]].copy()
    error_frame["prediction"] = prediction
    error_frame["absolute_error"] = np.abs(error_frame[TARGET] - error_frame["prediction"])
    food_error = (
        error_frame.groupby("EFPG_name")["absolute_error"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    region_error = (
        error_frame.groupby("Metroregion_name")["absolute_error"]
        .mean()
        .sort_values(ascending=False)
        .rename("MAE")
        .to_frame()
    )
    return food_error, region_error


def save_food_error_figure(food_error: pd.DataFrame, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    food_error.head(8).sort_values("mean")["mean"].plot(kind="barh", figsize=(7.2, 4.5))
    plt.xlabel("MAE")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def evaluate_reserved_test(
    fitted_models: dict[str, Any], split: SplitData
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    predictions: dict[str, np.ndarray] = {}
    x_test = split.test[FEATURES]
    y_test = split.test[TARGET]
    for name, model in fitted_models.items():
        prediction = model.predict(x_test)
        predictions[name] = prediction
        rows.append({"Model": name, **regression_metrics(y_test, prediction)})
    metrics_frame = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return metrics_frame, predictions


def evaluate_lstm_reserved_test(
    model: PriceLSTM, data: LSTMData, device: torch.device
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    test_mask = data.years == 2018
    test_loader = make_lstm_loader(data, test_mask, 1024, False)
    actual, predicted = predict_lstm(
        model, test_loader, device, data.price_mean, data.price_std
    )
    return regression_metrics(actual, predicted), actual, predicted


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    table_dir = args.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    download_dataset(args.data_url, args.data_path)
    raw = load_data(args.data_path)
    eda = summarize_eda(raw)
    print(json.dumps(eda, indent=2))
    pd.DataFrame([eda]).to_csv(table_dir / "eda_summary.csv", index=False)
    save_eda_figures(raw, figure_dir)
    save_overview_figure(figure_dir)

    outlier_summary, outlier_table = analyze_outliers(raw)
    print(json.dumps(outlier_summary, indent=2))
    save_json(args.output_dir / "outlier_summary.json", outlier_summary)
    outlier_table.to_csv(table_dir / "outlier_by_food_group.csv", index=False)

    feature_frame = engineer_features(raw)
    split = chronological_split(feature_frame)
    split_summary = {
        "train_years": "2013-2016",
        "validation_year": 2017,
        "reserved_test_year": 2018,
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "test_rows": len(split.test),
        "note": "2012 is used only as historical context for lag features.",
    }
    save_json(args.output_dir / "split_summary.json", split_summary)

    fitted_models, validation_metrics, predictions, tuning = fit_tabular_models(
        split, args.seed, args.quick
    )

    lstm_model = None
    lstm_history = None
    lstm_train_seconds = None
    if not args.skip_lstm:
        raw_for_lstm = raw.sort_values(
            ["EFPG_code", "Metroregion_code", "Year", "Month"]
        ).copy()
        raw_for_lstm["date"] = pd.to_datetime(
            {
                "year": raw_for_lstm["Year"],
                "month": raw_for_lstm["Month"],
                "day": 1,
            }
        )
        raw_for_lstm["month_sin"] = np.sin(2 * np.pi * raw_for_lstm["Month"] / 12.0)
        raw_for_lstm["month_cos"] = np.cos(2 * np.pi * raw_for_lstm["Month"] / 12.0)
        lstm_data = build_lstm_data(raw_for_lstm)
        lstm_train_start = time.perf_counter()
        lstm_model, lstm_history, lstm_scores, _ = train_lstm(
            lstm_data, args.seed, args.quick
        )
        lstm_train_seconds = time.perf_counter() - lstm_train_start
        validation_metrics = pd.concat(
            [validation_metrics, pd.DataFrame([{"Model": "LSTM", **lstm_scores}])],
            ignore_index=True,
        ).sort_values("RMSE").reset_index(drop=True)
        torch.save(
            {
                "state_dict": lstm_model.state_dict(),
                "price_mean": lstm_data.price_mean,
                "price_std": lstm_data.price_std,
                "history": lstm_history,
            },
            args.model_dir / "lstm_checkpoint.pt",
        )

    validation_metrics.to_csv(table_dir / "validation_metrics.csv", index=False)
    print("\nValidation metrics:\n", validation_metrics.to_string(index=False))
    save_json(args.output_dir / "tuning_results.json", tuning)
    save_regularization_effect(tuning, figure_dir, table_dir)

    mlp_trial_seconds = pd.DataFrame(tuning["MLP"])
    runtime_summary = {
        "lstm_device": "cuda" if torch.cuda.is_available() else "cpu",
        "lstm_train_seconds": lstm_train_seconds,
        "mlp_best_config_seconds": float(
            mlp_trial_seconds.loc[mlp_trial_seconds["RMSE"].idxmin(), "seconds"]
        ),
        "mlp_total_tuning_seconds": float(mlp_trial_seconds["seconds"].sum()),
        "note": (
            "These are local/CPU or Colab timings depending on where this run "
            "executed; see README for the Colab T4 GPU run used in the paper."
        ),
    }
    print("\nRuntime summary:\n", json.dumps(runtime_summary, indent=2))
    save_json(args.output_dir / "runtime_summary.json", runtime_summary)

    for name, model in fitted_models.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, args.model_dir / f"{safe_name}.joblib")

    importance, food_error, region_error = save_model_figures(
        validation_metrics,
        split.validation,
        predictions["Random Forest"],
        fitted_models["Random Forest"],
        fitted_models["MLP"],
        figure_dir,
        lstm_history,
    )
    importance.reset_index().rename(columns={"index": "feature"}).to_csv(
        table_dir / "rf_feature_importance.csv", index=False
    )
    food_error.reset_index().rename(columns={"mean": "MAE"}).to_csv(
        table_dir / "rf_error_by_food_group.csv", index=False
    )
    region_error.reset_index().to_csv(
        table_dir / "rf_error_by_region.csv", index=False
    )

    if args.evaluate_test:
        test_metrics, test_predictions = evaluate_reserved_test(fitted_models, split)
        if lstm_model is not None:
            lstm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            lstm_test_scores, _, lstm_test_predictions = evaluate_lstm_reserved_test(
                lstm_model, lstm_data, lstm_device
            )
            test_predictions["LSTM"] = lstm_test_predictions
            test_metrics = pd.concat(
                [test_metrics, pd.DataFrame([{"Model": "LSTM", **lstm_test_scores}])],
                ignore_index=True,
            ).sort_values("RMSE").reset_index(drop=True)
        test_metrics.to_csv(table_dir / "test_metrics.csv", index=False)
        print("\nReserved 2018 test metrics:\n", test_metrics.to_string(index=False))

        plot_actual_vs_predicted(
            split.test[TARGET].to_numpy(),
            test_predictions["Random Forest"],
            "Random Forest: Actual vs. Predicted (2018 Test Set)",
            figure_dir / "rf_actual_vs_predicted_test.png",
        )

        food_error_test, region_error_test = compute_error_breakdown(
            split.test, test_predictions["Random Forest"]
        )
        save_food_error_figure(
            food_error_test,
            "Food Groups With Highest Random Forest Error (2018 Test Set)",
            figure_dir / "rf_error_food_groups_test.png",
        )
        food_error_test.reset_index().rename(columns={"mean": "MAE"}).to_csv(
            table_dir / "rf_error_by_food_group_test.csv", index=False
        )
        region_error_test.reset_index().to_csv(
            table_dir / "rf_error_by_region_test.csv", index=False
        )
    else:
        print(
            "\n2018 test metrics were intentionally not computed. "
            "Keep the test set reserved until final model choices are frozen."
        )

    run_summary = {
        "dataset": eda,
        "outliers": outlier_summary,
        "split": split_summary,
        "validation_metrics": validation_metrics.to_dict(orient="records"),
        "best_validation_model": validation_metrics.iloc[0]["Model"],
        "runtime": runtime_summary,
        "seed": args.seed,
        "quick_mode": args.quick,
        "lstm_device": "cuda" if torch.cuda.is_available() else "cpu",
        "python": sys.version,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
    }
    save_json(args.output_dir / "checkpoint_summary.json", run_summary)
    print(f"\nFinished. Tables and figures are in {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
