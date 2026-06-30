"""Rolling-origin validation framework for Ghana cocoa yield forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.metrics import metric_definitions, metric_smoke_checks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ghana_cocoa_yield_1961_2024.csv"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_README_PATH = PROJECT_ROOT / "results" / "README.md"

VALIDATION_SPLITS_PATH = TABLES_DIR / "validation_splits.csv"
ROLLING_PREDICTIONS_TEMPLATE_PATH = TABLES_DIR / "rolling_predictions_template.csv"
METRIC_DEFINITIONS_PATH = TABLES_DIR / "metric_definitions.csv"
VALIDATION_CHECKS_PATH = TABLES_DIR / "validation_framework_checks.csv"
EVALUATION_HIGH_VOLATILITY_YEARS_PATH = TABLES_DIR / "evaluation_high_volatility_years.csv"

YIELD_COLUMN = "yield_tonnes_per_hectare"
DEFAULT_INITIAL_TRAINING_OBSERVATIONS = 30
DEFAULT_FORECAST_HORIZON = 1
VOLATILITY_QUANTILE = 0.75


@dataclass(frozen=True)
class ValidationConfig:
    initial_training_observations: int = DEFAULT_INITIAL_TRAINING_OBSERVATIONS
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON
    volatility_quantile: float = VOLATILITY_QUANTILE


def ensure_directories() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_yield_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH).sort_values("year").reset_index(drop=True)
    required = {"year", YIELD_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Yield dataset is missing columns: {sorted(missing)}")
    if data[YIELD_COLUMN].isna().any():
        raise ValueError("Yield dataset contains missing yield values.")
    return data


def in_sample_naive_mae(training_values: pd.Series) -> float:
    if len(training_values) < 2:
        raise ValueError("At least two training observations are required for MASE scale.")
    scale = training_values.astype(float).diff().abs().dropna().mean()
    if scale <= 0:
        raise ValueError("In-sample naive MAE must be positive for MASE scale.")
    return float(scale)


def build_validation_splits(
    data: pd.DataFrame,
    config: ValidationConfig | None = None,
) -> pd.DataFrame:
    config = config or ValidationConfig()
    if config.forecast_horizon < 1:
        raise ValueError("forecast_horizon must be at least 1.")
    if config.initial_training_observations < 2:
        raise ValueError("initial_training_observations must be at least 2.")
    if config.initial_training_observations + config.forecast_horizon > len(data):
        raise ValueError("Not enough observations for the requested validation setup.")

    rows = []
    first_forecast_index = (
        config.initial_training_observations + config.forecast_horizon - 1
    )
    for split_number, forecast_index in enumerate(
        range(first_forecast_index, len(data)),
        start=1,
    ):
        train_end_index = forecast_index - config.forecast_horizon
        training_data = data.iloc[: train_end_index + 1]
        forecast_row = data.iloc[forecast_index]
        previous_row = data.iloc[forecast_index - config.forecast_horizon]

        actual_yield = float(forecast_row[YIELD_COLUMN])
        previous_yield = float(previous_row[YIELD_COLUMN])
        actual_change = actual_yield - previous_yield
        actual_percentage_change = 100 * actual_change / previous_yield

        rows.append(
            {
                "split_id": split_number,
                "train_start_year": int(training_data["year"].iloc[0]),
                "train_end_year": int(training_data["year"].iloc[-1]),
                "forecast_year": int(forecast_row["year"]),
                "train_observation_count": len(training_data),
                "forecast_horizon": config.forecast_horizon,
                "actual_yield": actual_yield,
                "previous_yield": previous_yield,
                "actual_change": actual_change,
                "actual_percentage_change": actual_percentage_change,
                "absolute_actual_percentage_change": abs(actual_percentage_change),
                "mase_scale": in_sample_naive_mae(training_data[YIELD_COLUMN]),
            }
        )

    splits = pd.DataFrame(rows)
    threshold = float(
        splits["absolute_actual_percentage_change"].quantile(config.volatility_quantile)
    )
    splits["evaluation_volatility_threshold"] = threshold
    splits["is_evaluation_high_volatility"] = splits[
        "absolute_actual_percentage_change"
    ].ge(threshold)
    return splits


def build_rolling_predictions_template() -> pd.DataFrame:
    columns = [
        "split_id",
        "model_id",
        "model_name",
        "model_family",
        "interval_strategy",
        "fuzzy_order",
        "interval_count",
        "parameter_setting",
        "forecast_target",
        "correction_weight",
        "anchor_model_id",
        "fit_status",
        "fallback_used",
        "selected_order",
        "selection_score",
        "transformed_forecast",
        "effective_interval_count",
        "rule_count",
        "duplicate_boundary_count",
        "train_start_year",
        "train_end_year",
        "forecast_year",
        "actual_yield",
        "forecast_yield",
        "forecast_error",
        "absolute_error",
        "squared_error",
        "mase_scale",
        "is_evaluation_high_volatility",
    ]
    return pd.DataFrame(columns=columns)


def build_evaluation_high_volatility_years(splits: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "forecast_year",
        "actual_yield",
        "previous_yield",
        "actual_change",
        "actual_percentage_change",
        "absolute_actual_percentage_change",
        "evaluation_volatility_threshold",
    ]
    return (
        splits.loc[splits["is_evaluation_high_volatility"], columns]
        .sort_values("absolute_actual_percentage_change", ascending=False)
        .reset_index(drop=True)
    )


def validation_framework_checks() -> pd.DataFrame:
    metric_checks = metric_smoke_checks()
    metric_checks.insert(0, "check_group", "metrics")

    synthetic_data = pd.DataFrame(
        {
            "year": [2000, 2001, 2002, 2003, 2004, 2005],
            YIELD_COLUMN: [1.0, 1.5, 2.0, 3.0, 2.5, 4.0],
        }
    )
    synthetic_splits = build_validation_splits(
        synthetic_data,
        ValidationConfig(initial_training_observations=3, forecast_horizon=1),
    )
    split_checks = pd.DataFrame(
        [
            {
                "check_group": "validation_splits",
                "check": "synthetic_split_count",
                "expected": 3,
                "observed": len(synthetic_splits),
                "passed": len(synthetic_splits) == 3,
            },
            {
                "check_group": "validation_splits",
                "check": "first_forecast_year",
                "expected": 2003,
                "observed": int(synthetic_splits["forecast_year"].iloc[0]),
                "passed": int(synthetic_splits["forecast_year"].iloc[0]) == 2003,
            },
            {
                "check_group": "validation_splits",
                "check": "first_train_end_year",
                "expected": 2002,
                "observed": int(synthetic_splits["train_end_year"].iloc[0]),
                "passed": int(synthetic_splits["train_end_year"].iloc[0]) == 2002,
            },
        ]
    )
    return pd.concat([metric_checks, split_checks], ignore_index=True)


def write_results_readme(
    splits: pd.DataFrame,
    high_volatility_years: pd.DataFrame,
) -> None:
    first_split = splits.iloc[0]
    last_split = splits.iloc[-1]
    threshold = float(splits["evaluation_volatility_threshold"].iloc[0])
    content = f"""# Results Artifacts

## Data Audit And Exploratory Analysis

Generated by:

```powershell
.\\.venv\\Scripts\\python.exe -m src.exploration
```

Tables:

- `results/tables/data_audit.csv`: descriptive audit of the Ghana cocoa yield series.
- `results/tables/exploratory_high_volatility_years.csv`: exploratory high-volatility years using full-series absolute percentage change.

Figures:

- `results/figures/yield_timeseries.png`: annual Ghana cocoa bean yield.
- `results/figures/yield_pct_change.png`: year-to-year percentage change.
- `results/figures/exploratory_high_volatility_years.png`: yield series with exploratory high-volatility years highlighted.
- `results/figures/yield_change_distribution.png`: distribution of absolute annual percentage changes.

## Rolling-Origin Validation Framework

Generated by:

```powershell
.\\.venv\\Scripts\\python.exe -m src.validation
```

Tables:

- `results/tables/validation_splits.csv`: expanding-window one-step-ahead validation splits.
- `results/tables/evaluation_high_volatility_years.csv`: high-volatility years computed only over the evaluation period.
- `results/tables/metric_definitions.csv`: forecast metric definitions.
- `results/tables/rolling_predictions_template.csv`: schema for model forecast outputs.
- `results/tables/validation_framework_checks.csv`: metric and split smoke checks.

Key validation values:

- Split count: {len(splits)}.
- First split: train {int(first_split["train_start_year"])}-{int(first_split["train_end_year"])} -> forecast {int(first_split["forecast_year"])}.
- Last split: train {int(last_split["train_start_year"])}-{int(last_split["train_end_year"])} -> forecast {int(last_split["forecast_year"])}.
- Evaluation high-volatility threshold: {threshold:.6g}%.
- Evaluation high-volatility year count: {len(high_volatility_years)}.

Note:

Exploratory high-volatility labels use the full observed series. Evaluation high-volatility labels use only the rolling-origin evaluation period and should be used for model comparison.
"""
    RESULTS_README_PATH.write_text(content, encoding="utf-8")


def run(config: ValidationConfig | None = None) -> None:
    ensure_directories()
    data = load_yield_data()
    splits = build_validation_splits(data, config)
    high_volatility_years = build_evaluation_high_volatility_years(splits)
    checks = validation_framework_checks()

    splits.to_csv(VALIDATION_SPLITS_PATH, index=False)
    build_rolling_predictions_template().to_csv(
        ROLLING_PREDICTIONS_TEMPLATE_PATH,
        index=False,
    )
    metric_definitions().to_csv(METRIC_DEFINITIONS_PATH, index=False)
    checks.to_csv(VALIDATION_CHECKS_PATH, index=False)
    high_volatility_years.to_csv(EVALUATION_HIGH_VOLATILITY_YEARS_PATH, index=False)
    write_results_readme(splits, high_volatility_years)

    if not checks["passed"].all():
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise RuntimeError(f"Validation framework checks failed: {failed}")

    print("Validation framework complete.")
    print(f"Validation splits: {VALIDATION_SPLITS_PATH}")
    print(f"Metric definitions: {METRIC_DEFINITIONS_PATH}")
    print(f"Prediction template: {ROLLING_PREDICTIONS_TEMPLATE_PATH}")
    print(f"Framework checks: {VALIDATION_CHECKS_PATH}")


if __name__ == "__main__":
    run()
