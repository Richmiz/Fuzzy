"""Interval construction and fuzzification utilities for fuzzy time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MARGIN_FRACTION = 0.05
QUANTILE_METHOD = "linear"


@dataclass(frozen=True)
class FuzzificationResult:
    state: str
    interval_id: int
    midpoint: float
    clamped: bool = False


def _numeric_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    if array.size < 2:
        raise ValueError("At least two values are required to build intervals.")
    if not np.isfinite(array).all():
        raise ValueError("Interval values must be finite.")
    return array


def build_equal_intervals(
    values: Iterable[float],
    interval_count: int,
    margin_fraction: float = DEFAULT_MARGIN_FRACTION,
) -> pd.DataFrame:
    if interval_count < 1:
        raise ValueError("interval_count must be at least 1.")
    if margin_fraction < 0:
        raise ValueError("margin_fraction cannot be negative.")

    numeric_values = _numeric_values(values)
    value_min = float(numeric_values.min())
    value_max = float(numeric_values.max())
    value_range = value_max - value_min
    if value_range <= 0:
        value_range = max(abs(value_min), 1.0)

    margin = margin_fraction * value_range
    lower = value_min - margin
    upper = value_max + margin
    boundaries = np.linspace(lower, upper, interval_count + 1)

    rows = []
    for index in range(interval_count):
        lower_bound = float(boundaries[index])
        upper_bound = float(boundaries[index + 1])
        rows.append(
            {
                "interval_id": index + 1,
                "state": f"A{index + 1}",
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "midpoint": (lower_bound + upper_bound) / 2,
                "is_final_interval": index == interval_count - 1,
                "interval_strategy": "equal_length",
                "requested_interval_count": interval_count,
                "effective_interval_count": interval_count,
                "duplicate_boundary_count": 0,
                "margin_fraction": margin_fraction,
            }
        )
    return pd.DataFrame(rows)


def build_quantile_intervals(
    values: Iterable[float],
    interval_count: int,
) -> pd.DataFrame:
    if interval_count < 1:
        raise ValueError("interval_count must be at least 1.")

    numeric_values = _numeric_values(values)
    quantile_positions = np.linspace(0, 1, interval_count + 1)
    raw_boundaries = np.quantile(
        numeric_values,
        quantile_positions,
        method=QUANTILE_METHOD,
    )
    boundaries = np.unique(raw_boundaries.astype(float))
    if boundaries.size < 2:
        value = float(boundaries[0]) if boundaries.size == 1 else float(numeric_values[0])
        epsilon = max(abs(value), 1.0) * 1e-6
        boundaries = np.array([value - epsilon, value + epsilon], dtype=float)

    duplicate_boundary_count = int(len(raw_boundaries) - len(boundaries))
    effective_interval_count = int(len(boundaries) - 1)

    rows = []
    for index in range(effective_interval_count):
        lower_bound = float(boundaries[index])
        upper_bound = float(boundaries[index + 1])
        rows.append(
            {
                "interval_id": index + 1,
                "state": f"A{index + 1}",
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "midpoint": (lower_bound + upper_bound) / 2,
                "is_final_interval": index == effective_interval_count - 1,
                "interval_strategy": "adaptive_quantile",
                "requested_interval_count": interval_count,
                "effective_interval_count": effective_interval_count,
                "duplicate_boundary_count": duplicate_boundary_count,
                "margin_fraction": np.nan,
            }
        )
    return pd.DataFrame(rows)


def fuzzify_value(value: float, intervals: pd.DataFrame) -> FuzzificationResult:
    required = {"interval_id", "state", "lower_bound", "upper_bound", "midpoint"}
    missing = required.difference(intervals.columns)
    if missing:
        raise ValueError(f"Interval table is missing columns: {sorted(missing)}")

    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError("Cannot fuzzify a non-finite value.")

    ordered = intervals.sort_values("interval_id").reset_index(drop=True)
    for row_index, row in ordered.iterrows():
        lower = float(row["lower_bound"])
        upper = float(row["upper_bound"])
        is_final = bool(row_index == len(ordered) - 1)
        if lower <= numeric_value < upper or (is_final and lower <= numeric_value <= upper):
            return FuzzificationResult(
                state=str(row["state"]),
                interval_id=int(row["interval_id"]),
                midpoint=float(row["midpoint"]),
            )

    if numeric_value < float(ordered["lower_bound"].iloc[0]):
        row = ordered.iloc[0]
    else:
        row = ordered.iloc[-1]
    return FuzzificationResult(
        state=str(row["state"]),
        interval_id=int(row["interval_id"]),
        midpoint=float(row["midpoint"]),
        clamped=True,
    )


def fuzzify_series(values: Iterable[float], intervals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for position, value in enumerate(values):
        result = fuzzify_value(float(value), intervals)
        rows.append(
            {
                "position": position,
                "value": float(value),
                "state": result.state,
                "interval_id": result.interval_id,
                "midpoint": result.midpoint,
                "clamped": result.clamped,
            }
        )
    return pd.DataFrame(rows)


def midpoint_for_state(intervals: pd.DataFrame, state: str) -> float:
    match = intervals.loc[intervals["state"] == state]
    if match.empty:
        raise ValueError(f"Unknown fuzzy state: {state}")
    return float(match["midpoint"].iloc[0])
