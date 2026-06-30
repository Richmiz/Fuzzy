"""Forecast accuracy metrics used by the Ghana cocoa yield study."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


def _numeric_array(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1:
        array = array.reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains missing or non-finite values.")
    return array


def _paired_arrays(
    actual_values: Iterable[float],
    forecast_values: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    actual = _numeric_array(actual_values, "actual_values")
    forecast = _numeric_array(forecast_values, "forecast_values")
    if actual.size != forecast.size:
        raise ValueError(
            "actual_values and forecast_values must have the same number of values."
        )
    return actual, forecast


def mean_absolute_error(
    actual_values: Iterable[float],
    forecast_values: Iterable[float],
) -> float:
    actual, forecast = _paired_arrays(actual_values, forecast_values)
    return float(np.mean(np.abs(actual - forecast)))


def root_mean_squared_error(
    actual_values: Iterable[float],
    forecast_values: Iterable[float],
) -> float:
    actual, forecast = _paired_arrays(actual_values, forecast_values)
    return float(math.sqrt(np.mean(np.square(actual - forecast))))


def symmetric_mean_absolute_percentage_error(
    actual_values: Iterable[float],
    forecast_values: Iterable[float],
) -> float:
    actual, forecast = _paired_arrays(actual_values, forecast_values)
    denominator = np.abs(actual) + np.abs(forecast)
    contribution = np.zeros_like(actual, dtype=float)
    nonzero = denominator != 0
    contribution[nonzero] = 200 * np.abs(actual[nonzero] - forecast[nonzero]) / denominator[nonzero]
    return float(np.mean(contribution))


def mean_absolute_scaled_error(
    actual_values: Iterable[float],
    forecast_values: Iterable[float],
    scale_values: Iterable[float] | float,
) -> float:
    actual, forecast = _paired_arrays(actual_values, forecast_values)
    absolute_error = np.abs(actual - forecast)

    if isinstance(scale_values, (int, float)):
        scale = np.full(actual.shape, float(scale_values), dtype=float)
    else:
        scale = _numeric_array(scale_values, "scale_values")
        if scale.size != actual.size:
            raise ValueError("scale_values must be scalar or match the forecast length.")

    if np.any(scale <= 0):
        raise ValueError("scale_values must be positive for MASE.")

    return float(np.mean(absolute_error / scale))


def forecast_metric_summary(
    actual_values: Iterable[float],
    forecast_values: Iterable[float],
    scale_values: Iterable[float] | float | None = None,
) -> dict[str, float]:
    summary = {
        "mae": mean_absolute_error(actual_values, forecast_values),
        "rmse": root_mean_squared_error(actual_values, forecast_values),
        "smape": symmetric_mean_absolute_percentage_error(actual_values, forecast_values),
    }
    if scale_values is not None:
        summary["mase"] = mean_absolute_scaled_error(
            actual_values,
            forecast_values,
            scale_values,
        )
    return summary


def grouped_forecast_metrics(
    predictions: pd.DataFrame,
    group_columns: list[str],
    actual_column: str = "actual_yield",
    forecast_column: str = "forecast_yield",
    scale_column: str = "mase_scale",
) -> pd.DataFrame:
    required = set(group_columns + [actual_column, forecast_column])
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")

    rows = []
    for group_values, group in predictions.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_columns, group_values))
        scale_values = group[scale_column] if scale_column in group.columns else None
        row.update(
            forecast_metric_summary(
                group[actual_column],
                group[forecast_column],
                scale_values,
            )
        )
        row["forecast_count"] = len(group)
        rows.append(row)
    return pd.DataFrame(rows)


def metric_definitions() -> pd.DataFrame:
    rows = [
        (
            "mae",
            "Mean Absolute Error",
            "mean(abs(actual - forecast))",
            "Lower is better; same unit as yield.",
        ),
        (
            "rmse",
            "Root Mean Squared Error",
            "sqrt(mean((actual - forecast)^2))",
            "Lower is better; penalizes larger errors more strongly.",
        ),
        (
            "smape",
            "Symmetric Mean Absolute Percentage Error",
            "mean(200 * abs(actual - forecast) / (abs(actual) + abs(forecast)))",
            "Lower is better; percentage-style scale.",
        ),
        (
            "mase",
            "Mean Absolute Scaled Error",
            "mean(abs(actual - forecast) / in-sample naive MAE)",
            "Values below 1 indicate improvement over the in-sample naive scale.",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "name", "formula", "interpretation"])


def metric_smoke_checks() -> pd.DataFrame:
    actual = [1.0, 2.0, 4.0]
    forecast = [1.0, 3.0, 2.0]
    scale = 2.0
    checks = [
        ("mae_known_values", 1.0, mean_absolute_error(actual, forecast)),
        (
            "rmse_known_values",
            math.sqrt(5.0 / 3.0),
            root_mean_squared_error(actual, forecast),
        ),
        (
            "smape_known_values",
            (0.0 + 40.0 + (200.0 * 2.0 / 6.0)) / 3.0,
            symmetric_mean_absolute_percentage_error(actual, forecast),
        ),
        (
            "mase_known_values",
            0.5,
            mean_absolute_scaled_error(actual, forecast, scale),
        ),
    ]
    rows = []
    for check_name, expected, observed in checks:
        rows.append(
            {
                "check": check_name,
                "expected": expected,
                "observed": observed,
                "passed": abs(expected - observed) < 1e-12,
            }
        )
    return pd.DataFrame(rows)
