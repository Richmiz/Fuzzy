"""Chen fuzzy time-series models for Ghana cocoa yield forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.intervals import (
    build_equal_intervals,
    build_quantile_intervals,
    fuzzify_series,
    midpoint_for_state,
)


@dataclass(frozen=True)
class ChenForecastResult:
    forecast_yield: float
    parameter_setting: str
    fit_status: str
    fallback_used: bool
    intervals: pd.DataFrame
    rule_summary: pd.DataFrame
    first_order_rule_summary: pd.DataFrame
    effective_interval_count: int
    duplicate_boundary_count: int


def build_first_order_relationships(fuzzified_values: pd.DataFrame) -> pd.DataFrame:
    required = {"position", "state"}
    missing = required.difference(fuzzified_values.columns)
    if missing:
        raise ValueError(f"Fuzzified table is missing columns: {sorted(missing)}")
    if len(fuzzified_values) < 2:
        raise ValueError("At least two fuzzified observations are required.")

    ordered = fuzzified_values.sort_values("position").reset_index(drop=True)
    rows = []
    for index in range(1, len(ordered)):
        rows.append(
            {
                "relationship_id": index,
                "antecedent_state": str(ordered.loc[index - 1, "state"]),
                "consequent_state": str(ordered.loc[index, "state"]),
            }
        )
    return pd.DataFrame(rows)


def build_first_order_rule_summary(
    relationships: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    required = {"antecedent_state", "consequent_state"}
    missing = required.difference(relationships.columns)
    if missing:
        raise ValueError(f"Relationship table is missing columns: {sorted(missing)}")

    counts = (
        relationships.groupby(["antecedent_state", "consequent_state"], as_index=False)
        .size()
        .rename(columns={"size": "transition_count"})
    )

    rows = []
    for antecedent_state, group in counts.groupby("antecedent_state", sort=False):
        group = group.sort_values(["transition_count", "consequent_state"], ascending=[False, True])
        weighted_total = 0.0
        total_count = int(group["transition_count"].sum())
        consequent_states = []
        consequent_counts = []
        for _, row in group.iterrows():
            consequent_state = str(row["consequent_state"])
            count = int(row["transition_count"])
            consequent_states.append(consequent_state)
            consequent_counts.append(f"{consequent_state}:{count}")
            weighted_total += midpoint_for_state(intervals, consequent_state) * count

        rows.append(
            {
                "antecedent_state": str(antecedent_state),
                "consequent_states": ";".join(consequent_states),
                "consequent_counts": ";".join(consequent_counts),
                "transition_count": total_count,
                "forecast_yield": weighted_total / total_count,
                "defuzzification": "frequency_weighted_midpoint",
            }
        )
    return pd.DataFrame(rows)


def build_second_order_relationships(fuzzified_values: pd.DataFrame) -> pd.DataFrame:
    required = {"position", "state"}
    missing = required.difference(fuzzified_values.columns)
    if missing:
        raise ValueError(f"Fuzzified table is missing columns: {sorted(missing)}")
    if len(fuzzified_values) < 3:
        raise ValueError("At least three fuzzified observations are required.")

    ordered = fuzzified_values.sort_values("position").reset_index(drop=True)
    rows = []
    for index in range(2, len(ordered)):
        rows.append(
            {
                "relationship_id": index - 1,
                "antecedent_states": (
                    f"{ordered.loc[index - 2, 'state']},"
                    f"{ordered.loc[index - 1, 'state']}"
                ),
                "consequent_state": str(ordered.loc[index, "state"]),
            }
        )
    return pd.DataFrame(rows)


def build_second_order_rule_summary(
    relationships: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    required = {"antecedent_states", "consequent_state"}
    missing = required.difference(relationships.columns)
    if missing:
        raise ValueError(f"Relationship table is missing columns: {sorted(missing)}")

    counts = (
        relationships.groupby(["antecedent_states", "consequent_state"], as_index=False)
        .size()
        .rename(columns={"size": "transition_count"})
    )

    rows = []
    for antecedent_states, group in counts.groupby("antecedent_states", sort=False):
        group = group.sort_values(["transition_count", "consequent_state"], ascending=[False, True])
        weighted_total = 0.0
        total_count = int(group["transition_count"].sum())
        consequent_states = []
        consequent_counts = []
        for _, row in group.iterrows():
            consequent_state = str(row["consequent_state"])
            count = int(row["transition_count"])
            consequent_states.append(consequent_state)
            consequent_counts.append(f"{consequent_state}:{count}")
            weighted_total += midpoint_for_state(intervals, consequent_state) * count

        rows.append(
            {
                "antecedent_states": str(antecedent_states),
                "consequent_states": ";".join(consequent_states),
                "consequent_counts": ";".join(consequent_counts),
                "transition_count": total_count,
                "forecast_yield": weighted_total / total_count,
                "defuzzification": "frequency_weighted_midpoint",
            }
        )
    return pd.DataFrame(rows)


def _training_values(training_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(training_values), dtype=float).reshape(-1)
    if values.size < 2:
        raise ValueError("At least two training values are required.")
    if not np.isfinite(values).all():
        raise ValueError("Training values must be finite.")
    return values


def _intervals_for_strategy(
    values: np.ndarray,
    interval_count: int,
    interval_strategy: str,
) -> pd.DataFrame:
    if interval_strategy == "equal_length":
        return build_equal_intervals(values, interval_count)
    if interval_strategy == "adaptive_quantile":
        return build_quantile_intervals(values, interval_count)
    raise ValueError(f"Unknown interval strategy: {interval_strategy}")


def _parameter_setting(
    interval_count: int,
    interval_strategy: str,
    fuzzy_order: int,
) -> str:
    if interval_strategy == "equal_length":
        interval_part = f"k={interval_count};margin_fraction=0.05"
    elif interval_strategy == "adaptive_quantile":
        interval_part = f"k={interval_count};quantiles=training_window"
    else:
        interval_part = f"k={interval_count};interval_strategy={interval_strategy}"
    return (
        f"{interval_part};fuzzy_order={fuzzy_order};"
        "defuzzification=frequency_weighted_midpoint"
    )


def forecast_first_order_equal_chen(
    training_values: Iterable[float],
    interval_count: int,
) -> ChenForecastResult:
    return forecast_first_order_chen(
        training_values,
        interval_count,
        interval_strategy="equal_length",
    )


def forecast_first_order_chen(
    training_values: Iterable[float],
    interval_count: int,
    interval_strategy: str,
) -> ChenForecastResult:
    values = _training_values(training_values)
    intervals = _intervals_for_strategy(values, interval_count, interval_strategy)
    fuzzified = fuzzify_series(values, intervals)
    relationships = build_first_order_relationships(fuzzified)
    rule_summary = build_first_order_rule_summary(relationships, intervals)
    current_state = str(fuzzified["state"].iloc[-1])
    rule = rule_summary.loc[rule_summary["antecedent_state"] == current_state]
    parameter_setting = _parameter_setting(interval_count, interval_strategy, 1)
    effective_interval_count = int(intervals["effective_interval_count"].iloc[0])
    duplicate_boundary_count = int(intervals["duplicate_boundary_count"].iloc[0])

    if rule.empty:
        current_midpoint = float(fuzzified["midpoint"].iloc[-1])
        return ChenForecastResult(
            forecast_yield=current_midpoint,
            parameter_setting=parameter_setting,
            fit_status="fallback_current_midpoint",
            fallback_used=True,
            intervals=intervals,
            rule_summary=rule_summary,
            first_order_rule_summary=rule_summary,
            effective_interval_count=effective_interval_count,
            duplicate_boundary_count=duplicate_boundary_count,
        )

    return ChenForecastResult(
        forecast_yield=float(rule["forecast_yield"].iloc[0]),
        parameter_setting=parameter_setting,
        fit_status="fitted",
        fallback_used=False,
        intervals=intervals,
        rule_summary=rule_summary,
        first_order_rule_summary=rule_summary,
        effective_interval_count=effective_interval_count,
        duplicate_boundary_count=duplicate_boundary_count,
    )


def forecast_second_order_chen(
    training_values: Iterable[float],
    interval_count: int,
    interval_strategy: str,
) -> ChenForecastResult:
    values = _training_values(training_values)
    if len(values) < 3:
        raise ValueError("At least three training values are required for order-2 FTS.")

    intervals = _intervals_for_strategy(values, interval_count, interval_strategy)
    fuzzified = fuzzify_series(values, intervals)
    first_relationships = build_first_order_relationships(fuzzified)
    first_rule_summary = build_first_order_rule_summary(first_relationships, intervals)
    second_relationships = build_second_order_relationships(fuzzified)
    second_rule_summary = build_second_order_rule_summary(second_relationships, intervals)

    current_antecedent = f"{fuzzified['state'].iloc[-2]},{fuzzified['state'].iloc[-1]}"
    second_rule = second_rule_summary.loc[
        second_rule_summary["antecedent_states"] == current_antecedent
    ]
    parameter_setting = _parameter_setting(interval_count, interval_strategy, 2)
    effective_interval_count = int(intervals["effective_interval_count"].iloc[0])
    duplicate_boundary_count = int(intervals["duplicate_boundary_count"].iloc[0])

    if not second_rule.empty:
        return ChenForecastResult(
            forecast_yield=float(second_rule["forecast_yield"].iloc[0]),
            parameter_setting=parameter_setting,
            fit_status="fitted",
            fallback_used=False,
            intervals=intervals,
            rule_summary=second_rule_summary,
            first_order_rule_summary=first_rule_summary,
            effective_interval_count=effective_interval_count,
            duplicate_boundary_count=duplicate_boundary_count,
        )

    current_state = str(fuzzified["state"].iloc[-1])
    first_rule = first_rule_summary.loc[
        first_rule_summary["antecedent_state"] == current_state
    ]
    if not first_rule.empty:
        return ChenForecastResult(
            forecast_yield=float(first_rule["forecast_yield"].iloc[0]),
            parameter_setting=parameter_setting,
            fit_status="fallback_first_order",
            fallback_used=True,
            intervals=intervals,
            rule_summary=second_rule_summary,
            first_order_rule_summary=first_rule_summary,
            effective_interval_count=effective_interval_count,
            duplicate_boundary_count=duplicate_boundary_count,
        )

    return ChenForecastResult(
        forecast_yield=float(fuzzified["midpoint"].iloc[-1]),
        parameter_setting=parameter_setting,
        fit_status="fallback_current_midpoint",
        fallback_used=True,
        intervals=intervals,
        rule_summary=second_rule_summary,
        first_order_rule_summary=first_rule_summary,
        effective_interval_count=effective_interval_count,
        duplicate_boundary_count=duplicate_boundary_count,
    )


def equal_interval_model_scenarios(interval_counts: Iterable[int]) -> list[dict[str, object]]:
    scenarios = []
    for interval_count in interval_counts:
        scenarios.append(
            {
                "model_id": f"chen_equal_k{interval_count}",
                "model_name": f"Chen FTS Equal Intervals k={interval_count}",
                "model_family": "fuzzy_time_series",
                "interval_strategy": "equal_length",
                "fuzzy_order": 1,
                "interval_count": interval_count,
                "parameter_setting": (
                    f"k={interval_count};margin_fraction=0.05;"
                    "fuzzy_order=1;defuzzification=frequency_weighted_midpoint"
                ),
                "description": "First-order Chen fuzzy time series with equal-length intervals.",
            }
        )
    return scenarios


def adaptive_interval_model_scenarios(interval_counts: Iterable[int]) -> list[dict[str, object]]:
    scenarios = []
    for interval_count in interval_counts:
        scenarios.append(
            {
                "model_id": f"chen_adaptive_k{interval_count}",
                "model_name": f"Chen FTS Adaptive Intervals k={interval_count}",
                "model_family": "fuzzy_time_series",
                "interval_strategy": "adaptive_quantile",
                "fuzzy_order": 1,
                "interval_count": interval_count,
                "parameter_setting": (
                    f"k={interval_count};quantiles=training_window;"
                    "fuzzy_order=1;defuzzification=frequency_weighted_midpoint"
                ),
                "description": "First-order Chen fuzzy time series with quantile-based adaptive intervals.",
            }
        )
    return scenarios


def high_order_adaptive_model_scenarios(interval_counts: Iterable[int]) -> list[dict[str, object]]:
    scenarios = []
    for interval_count in interval_counts:
        scenarios.append(
            {
                "model_id": f"chen_order2_adaptive_k{interval_count}",
                "model_name": f"Chen FTS Order 2 Adaptive Intervals k={interval_count}",
                "model_family": "fuzzy_time_series",
                "interval_strategy": "adaptive_quantile",
                "fuzzy_order": 2,
                "interval_count": interval_count,
                "parameter_setting": (
                    f"k={interval_count};quantiles=training_window;"
                    "fuzzy_order=2;defuzzification=frequency_weighted_midpoint"
                ),
                "description": "Order-2 Chen fuzzy time series with quantile-based adaptive intervals.",
            }
        )
    return scenarios
