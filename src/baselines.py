"""Baseline forecasting models for the Ghana cocoa yield study."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


ARIMA_ORDER_GRID = tuple(product((0, 1, 2), (0, 1), (0, 1, 2)))


@dataclass(frozen=True)
class ForecastResult:
    forecast_yield: float
    parameter_setting: str
    fit_status: str
    fallback_used: bool = False
    selected_order: str = ""
    selection_score: float | None = None


def _training_array(training_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(training_values), dtype=float).reshape(-1)
    if values.size < 2:
        raise ValueError("At least two training values are required.")
    if not np.isfinite(values).all():
        raise ValueError("Training values must be finite.")
    return values


def naive_forecast(training_values: Iterable[float]) -> ForecastResult:
    values = _training_array(training_values)
    return ForecastResult(
        forecast_yield=float(values[-1]),
        parameter_setting="previous_observation",
        fit_status="fitted",
    )


def mean_forecast(training_values: Iterable[float]) -> ForecastResult:
    values = _training_array(training_values)
    return ForecastResult(
        forecast_yield=float(values.mean()),
        parameter_setting="expanding_window_mean",
        fit_status="fitted",
    )


def drift_forecast(training_values: Iterable[float]) -> ForecastResult:
    values = _training_array(training_values)
    drift = (values[-1] - values[0]) / (len(values) - 1)
    return ForecastResult(
        forecast_yield=float(values[-1] + drift),
        parameter_setting="one_step_linear_drift",
        fit_status="fitted",
    )


def arima_aic_forecast(training_values: Iterable[float]) -> ForecastResult:
    values = _training_array(training_values)
    series = pd.Series(values)
    best_aic = np.inf
    best_forecast: float | None = None
    best_order: tuple[int, int, int] | None = None

    for order in ARIMA_ORDER_GRID:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted_model = ARIMA(
                    series,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()

            aic = float(fitted_model.aic)
            forecast = float(np.asarray(fitted_model.forecast(steps=1)).reshape(-1)[0])
            if np.isfinite(aic) and np.isfinite(forecast) and aic < best_aic:
                best_aic = aic
                best_forecast = forecast
                best_order = order
        except Exception:
            continue

    if best_forecast is None or best_order is None:
        fallback = naive_forecast(values)
        return ForecastResult(
            forecast_yield=fallback.forecast_yield,
            parameter_setting="aic_grid_p0-2_d0-1_q0-2",
            fit_status="fallback_naive",
            fallback_used=True,
        )

    return ForecastResult(
        forecast_yield=best_forecast,
        parameter_setting="aic_grid_p0-2_d0-1_q0-2",
        fit_status="fitted",
        selected_order=f"({best_order[0]},{best_order[1]},{best_order[2]})",
        selection_score=best_aic,
    )


def baseline_model_scenarios() -> list[dict[str, object]]:
    return [
        {
            "model_id": "baseline_naive",
            "model_name": "Naive",
            "model_family": "baseline",
            "interval_strategy": "none",
            "fuzzy_order": np.nan,
            "interval_count": np.nan,
            "parameter_setting": "previous_observation",
            "description": "One-step forecast equal to the latest training observation.",
        },
        {
            "model_id": "baseline_mean",
            "model_name": "Mean",
            "model_family": "baseline",
            "interval_strategy": "none",
            "fuzzy_order": np.nan,
            "interval_count": np.nan,
            "parameter_setting": "expanding_window_mean",
            "description": "One-step forecast equal to the expanding training-window mean.",
        },
        {
            "model_id": "baseline_drift",
            "model_name": "Drift",
            "model_family": "baseline",
            "interval_strategy": "none",
            "fuzzy_order": np.nan,
            "interval_count": np.nan,
            "parameter_setting": "one_step_linear_drift",
            "description": "One-step forecast using the linear change from first to latest training observation.",
        },
        {
            "model_id": "baseline_arima_aic",
            "model_name": "ARIMA-AIC",
            "model_family": "baseline",
            "interval_strategy": "none",
            "fuzzy_order": np.nan,
            "interval_count": np.nan,
            "parameter_setting": "aic_grid_p0-2_d0-1_q0-2",
            "description": "ARIMA order selected by AIC inside each rolling training window.",
        },
    ]


def run_baseline_models(training_values: Iterable[float]) -> list[tuple[str, ForecastResult]]:
    return [
        ("baseline_naive", naive_forecast(training_values)),
        ("baseline_mean", mean_forecast(training_values)),
        ("baseline_drift", drift_forecast(training_values)),
        ("baseline_arima_aic", arima_aic_forecast(training_values)),
    ]
