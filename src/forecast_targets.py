"""Forecast target transformations for fuzzy correction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


ABSOLUTE_CHANGE_TARGET = "absolute_change"
PERCENTAGE_CHANGE_TARGET = "percentage_change"
NAIVE_ANCHOR_MODEL_ID = "baseline_naive"


@dataclass(frozen=True)
class TransformedForecast:
    forecast_yield: float
    transformed_forecast: float
    anchor_model_id: str
    correction_weight: float
    forecast_target: str


def _numeric_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    if array.size < 2:
        raise ValueError("At least two yield values are required.")
    if not np.isfinite(array).all():
        raise ValueError("Yield values must be finite.")
    return array


def build_transformed_target(
    yield_values: Iterable[float],
    forecast_target: str,
) -> np.ndarray:
    values = _numeric_values(yield_values)
    previous_values = values[:-1]
    current_values = values[1:]

    if forecast_target == ABSOLUTE_CHANGE_TARGET:
        transformed = current_values - previous_values
    elif forecast_target == PERCENTAGE_CHANGE_TARGET:
        if np.any(previous_values == 0):
            raise ValueError("Cannot build percentage changes with zero previous yield.")
        transformed = 100 * (current_values - previous_values) / previous_values
    else:
        raise ValueError(f"Unknown forecast target: {forecast_target}")

    if transformed.size < 2:
        raise ValueError("At least two transformed values are required.")
    if not np.isfinite(transformed).all():
        raise ValueError("Transformed target contains non-finite values.")
    return transformed.astype(float)


def reconstruct_yield_forecast(
    previous_yield: float,
    transformed_forecast: float,
    forecast_target: str,
    correction_weight: float,
    anchor_model_id: str = NAIVE_ANCHOR_MODEL_ID,
) -> TransformedForecast:
    previous = float(previous_yield)
    transformed = float(transformed_forecast)
    weight = float(correction_weight)
    if not np.isfinite(previous) or not np.isfinite(transformed) or not np.isfinite(weight):
        raise ValueError("Forecast reconstruction inputs must be finite.")

    if forecast_target == ABSOLUTE_CHANGE_TARGET:
        forecast_yield = previous + weight * transformed
    elif forecast_target == PERCENTAGE_CHANGE_TARGET:
        forecast_yield = previous * (1 + weight * transformed / 100)
    else:
        raise ValueError(f"Unknown forecast target: {forecast_target}")

    if not np.isfinite(forecast_yield):
        raise ValueError("Reconstructed forecast is non-finite.")

    return TransformedForecast(
        forecast_yield=float(forecast_yield),
        transformed_forecast=transformed,
        anchor_model_id=anchor_model_id,
        correction_weight=weight,
        forecast_target=forecast_target,
    )
