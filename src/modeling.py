"""Model execution pipeline for Ghana cocoa yield forecasting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import baseline_model_scenarios, run_baseline_models
from src.forecast_targets import (
    ABSOLUTE_CHANGE_TARGET,
    PERCENTAGE_CHANGE_TARGET,
    build_transformed_target,
    reconstruct_yield_forecast,
)
from src.fts import (
    adaptive_interval_model_scenarios,
    equal_interval_model_scenarios,
    forecast_first_order_chen,
    forecast_first_order_equal_chen,
    forecast_second_order_chen,
    high_order_adaptive_model_scenarios,
)
from src.metrics import grouped_forecast_metrics
from src.validation import (
    YIELD_COLUMN,
    build_validation_splits,
    load_yield_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MODEL_OUTPUTS_DIR = PROJECT_ROOT / "results" / "model_outputs"
RESULTS_README_PATH = PROJECT_ROOT / "results" / "README.md"

MODEL_SCENARIOS_PATH = TABLES_DIR / "model_scenarios.csv"
ROLLING_PREDICTIONS_PATH = TABLES_DIR / "rolling_predictions.csv"
OVERALL_METRICS_PATH = TABLES_DIR / "overall_metrics.csv"
VOLATILITY_METRICS_PATH = TABLES_DIR / "volatility_metrics.csv"
FORECAST_TARGET_COMPARISON_PATH = TABLES_DIR / "forecast_target_comparison.csv"
FUZZY_CORRECTION_COMPARISON_PATH = TABLES_DIR / "fuzzy_correction_comparison.csv"
FUZZY_BASELINE_GAP_ANALYSIS_PATH = TABLES_DIR / "fuzzy_baseline_gap_analysis.csv"
ENHANCED_MODEL_RECOMMENDATIONS_PATH = TABLES_DIR / "enhanced_model_recommendations.csv"
INTERVAL_STRATEGY_COMPARISON_PATH = (
    TABLES_DIR / "interval_strategy_comparison.csv"
)
RULE_ORDER_COMPARISON_PATH = TABLES_DIR / "rule_order_comparison.csv"
FINAL_WINDOW_BOUNDARIES_PATH = (
    MODEL_OUTPUTS_DIR / "equal_interval_boundaries_final_window.csv"
)
FINAL_WINDOW_RULES_PATH = MODEL_OUTPUTS_DIR / "equal_interval_rules_final_window.csv"
ADAPTIVE_WINDOW_BOUNDARIES_PATH = (
    MODEL_OUTPUTS_DIR / "adaptive_interval_boundaries_final_window.csv"
)
ADAPTIVE_WINDOW_RULES_PATH = MODEL_OUTPUTS_DIR / "adaptive_interval_rules_final_window.csv"
HIGH_ORDER_RULES_PATH = MODEL_OUTPUTS_DIR / "high_order_rules_final_window.csv"
CHANGE_TARGET_RULES_PATH = MODEL_OUTPUTS_DIR / "change_interval_rules_final_window.csv"
PERCENTAGE_CHANGE_TARGET_RULES_PATH = (
    MODEL_OUTPUTS_DIR / "percentage_change_interval_rules_final_window.csv"
)

EQUAL_INTERVAL_COUNTS = (5, 7, 9, 11)
ADAPTIVE_INTERVAL_COUNTS = (5, 7, 9, 11)
HIGH_ORDER_INTERVAL_COUNTS = (5, 7, 9, 11)
CORRECTION_INTERVAL_COUNTS = (5, 7, 9, 11)
CORRECTION_WEIGHTS = (0.5, 1.0)
CORRECTION_TARGETS = (ABSOLUTE_CHANGE_TARGET, PERCENTAGE_CHANGE_TARGET)
CORRECTION_INTERVAL_STRATEGIES = ("equal_length", "adaptive_quantile")
ACCURACY_METRICS = ("mae", "rmse", "smape", "mase")
MODEL_GROUP_COLUMNS = [
    "model_id",
    "model_name",
    "model_family",
    "interval_strategy",
    "fuzzy_order",
    "interval_count",
    "parameter_setting",
]
MODEL_GROUP_COLUMNS.extend(
    [
        "forecast_target",
        "correction_weight",
        "anchor_model_id",
    ]
)


def ensure_directories() -> None:
    for path in [TABLES_DIR, MODEL_OUTPUTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def build_model_scenarios() -> pd.DataFrame:
    rows = baseline_model_scenarios()
    rows.extend(equal_interval_model_scenarios(EQUAL_INTERVAL_COUNTS))
    rows.extend(adaptive_interval_model_scenarios(ADAPTIVE_INTERVAL_COUNTS))
    rows.extend(high_order_adaptive_model_scenarios(HIGH_ORDER_INTERVAL_COUNTS))
    rows.extend(correction_model_scenarios())
    scenarios = pd.DataFrame(rows)
    scenarios["forecast_target"] = scenarios["forecast_target"].fillna("yield_level")
    scenarios["correction_weight"] = scenarios["correction_weight"].astype(float)
    scenarios["anchor_model_id"] = scenarios["anchor_model_id"].fillna("")
    return scenarios


def _correction_weight_label(correction_weight: float) -> str:
    return f"w{int(round(correction_weight * 100)):03d}"


def _target_model_prefix(forecast_target: str) -> str:
    if forecast_target == ABSOLUTE_CHANGE_TARGET:
        return "chen_change"
    if forecast_target == PERCENTAGE_CHANGE_TARGET:
        return "chen_pct_change"
    raise ValueError(f"Unknown forecast target: {forecast_target}")


def _strategy_model_suffix(interval_strategy: str) -> str:
    if interval_strategy == "equal_length":
        return "equal"
    if interval_strategy == "adaptive_quantile":
        return "adaptive"
    raise ValueError(f"Unknown interval strategy: {interval_strategy}")


def correction_model_id(
    forecast_target: str,
    interval_strategy: str,
    interval_count: int,
    correction_weight: float,
) -> str:
    return (
        f"{_target_model_prefix(forecast_target)}_"
        f"{_strategy_model_suffix(interval_strategy)}_"
        f"k{interval_count}_{_correction_weight_label(correction_weight)}"
    )


def correction_model_scenarios() -> list[dict[str, object]]:
    scenarios = []
    target_names = {
        ABSOLUTE_CHANGE_TARGET: "Absolute Change",
        PERCENTAGE_CHANGE_TARGET: "Percentage Change",
    }
    strategy_names = {
        "equal_length": "Equal Intervals",
        "adaptive_quantile": "Adaptive Intervals",
    }
    for forecast_target in CORRECTION_TARGETS:
        for interval_strategy in CORRECTION_INTERVAL_STRATEGIES:
            for correction_weight in CORRECTION_WEIGHTS:
                for interval_count in CORRECTION_INTERVAL_COUNTS:
                    model_id = correction_model_id(
                        forecast_target,
                        interval_strategy,
                        interval_count,
                        correction_weight,
                    )
                    scenarios.append(
                        {
                            "model_id": model_id,
                            "model_name": (
                                "Chen FTS "
                                f"{target_names[forecast_target]} Correction "
                                f"{strategy_names[interval_strategy]} "
                                f"k={interval_count} "
                                f"{_correction_weight_label(correction_weight)}"
                            ),
                            "model_family": "fuzzy_time_series",
                            "interval_strategy": interval_strategy,
                            "fuzzy_order": 1,
                            "interval_count": interval_count,
                            "parameter_setting": (
                                f"forecast_target={forecast_target};"
                                f"k={interval_count};"
                                f"correction_weight={correction_weight:g};"
                                "anchor=baseline_naive;"
                                "fuzzy_order=1;"
                                "defuzzification=frequency_weighted_midpoint"
                            ),
                            "description": (
                                "First-order Chen fuzzy time series predicting a "
                                "training-window yield correction around the naive anchor."
                            ),
                            "forecast_target": forecast_target,
                            "correction_weight": correction_weight,
                            "anchor_model_id": "baseline_naive",
                        }
                    )
    return scenarios


def _scenario_lookup(model_scenarios: pd.DataFrame) -> dict[str, dict[str, object]]:
    return {
        str(row["model_id"]): row.to_dict()
        for _, row in model_scenarios.iterrows()
    }


def _base_prediction_row(
    split: pd.Series,
    scenario: dict[str, object],
) -> dict[str, object]:
    return {
        "split_id": int(split["split_id"]),
        "model_id": scenario["model_id"],
        "model_name": scenario["model_name"],
        "model_family": scenario["model_family"],
        "interval_strategy": scenario["interval_strategy"],
        "fuzzy_order": scenario["fuzzy_order"],
        "interval_count": scenario["interval_count"],
        "parameter_setting": scenario["parameter_setting"],
        "forecast_target": scenario.get("forecast_target", "yield_level"),
        "correction_weight": scenario.get("correction_weight", np.nan),
        "anchor_model_id": scenario.get("anchor_model_id", ""),
        "train_start_year": int(split["train_start_year"]),
        "train_end_year": int(split["train_end_year"]),
        "forecast_year": int(split["forecast_year"]),
        "actual_yield": float(split["actual_yield"]),
        "mase_scale": float(split["mase_scale"]),
        "is_evaluation_high_volatility": bool(split["is_evaluation_high_volatility"]),
    }


def _add_error_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    enriched = predictions.copy()
    enriched["forecast_error"] = enriched["actual_yield"] - enriched["forecast_yield"]
    enriched["absolute_error"] = enriched["forecast_error"].abs()
    enriched["squared_error"] = enriched["forecast_error"] ** 2
    return enriched


def _add_model_result_columns(
    row: dict[str, object],
    forecast_result,
) -> dict[str, object]:
    row.update(
        {
            "forecast_yield": forecast_result.forecast_yield,
            "fit_status": forecast_result.fit_status,
            "fallback_used": forecast_result.fallback_used,
            "selected_order": "",
            "selection_score": np.nan,
            "transformed_forecast": np.nan,
            "effective_interval_count": forecast_result.effective_interval_count,
            "rule_count": len(forecast_result.rule_summary),
            "duplicate_boundary_count": forecast_result.duplicate_boundary_count,
        }
    )
    return row


def build_rolling_predictions(
    data: pd.DataFrame,
    splits: pd.DataFrame,
    model_scenarios: pd.DataFrame,
) -> pd.DataFrame:
    scenario_lookup = _scenario_lookup(model_scenarios)
    rows = []

    for _, split in splits.iterrows():
        training_data = data.loc[
            data["year"].between(
                int(split["train_start_year"]),
                int(split["train_end_year"]),
            )
        ]
        training_values = training_data[YIELD_COLUMN].astype(float).to_numpy()

        for model_id, forecast_result in run_baseline_models(training_values):
            scenario = scenario_lookup[model_id]
            row = _base_prediction_row(split, scenario)
            row.update(
                {
                    "forecast_yield": forecast_result.forecast_yield,
                    "fit_status": forecast_result.fit_status,
                    "fallback_used": forecast_result.fallback_used,
                    "selected_order": forecast_result.selected_order,
                    "selection_score": forecast_result.selection_score,
                    "transformed_forecast": np.nan,
                    "effective_interval_count": np.nan,
                    "rule_count": np.nan,
                    "duplicate_boundary_count": np.nan,
                }
            )
            rows.append(row)

        for interval_count in EQUAL_INTERVAL_COUNTS:
            model_id = f"chen_equal_k{interval_count}"
            scenario = scenario_lookup[model_id]
            forecast_result = forecast_first_order_equal_chen(
                training_values,
                interval_count,
            )
            row = _base_prediction_row(split, scenario)
            rows.append(_add_model_result_columns(row, forecast_result))

        for interval_count in ADAPTIVE_INTERVAL_COUNTS:
            model_id = f"chen_adaptive_k{interval_count}"
            scenario = scenario_lookup[model_id]
            forecast_result = forecast_first_order_chen(
                training_values,
                interval_count,
                interval_strategy="adaptive_quantile",
            )
            row = _base_prediction_row(split, scenario)
            rows.append(_add_model_result_columns(row, forecast_result))

        for interval_count in HIGH_ORDER_INTERVAL_COUNTS:
            model_id = f"chen_order2_adaptive_k{interval_count}"
            scenario = scenario_lookup[model_id]
            forecast_result = forecast_second_order_chen(
                training_values,
                interval_count,
                interval_strategy="adaptive_quantile",
            )
            row = _base_prediction_row(split, scenario)
            rows.append(_add_model_result_columns(row, forecast_result))

        for forecast_target in CORRECTION_TARGETS:
            transformed_values = build_transformed_target(training_values, forecast_target)
            for interval_strategy in CORRECTION_INTERVAL_STRATEGIES:
                for correction_weight in CORRECTION_WEIGHTS:
                    for interval_count in CORRECTION_INTERVAL_COUNTS:
                        model_id = correction_model_id(
                            forecast_target,
                            interval_strategy,
                            interval_count,
                            correction_weight,
                        )
                        scenario = scenario_lookup[model_id]
                        forecast_result = forecast_first_order_chen(
                            transformed_values,
                            interval_count,
                            interval_strategy=interval_strategy,
                        )
                        reconstructed = reconstruct_yield_forecast(
                            previous_yield=float(split["previous_yield"]),
                            transformed_forecast=forecast_result.forecast_yield,
                            forecast_target=forecast_target,
                            correction_weight=correction_weight,
                        )
                        row = _base_prediction_row(split, scenario)
                        row.update(
                            {
                                "forecast_yield": reconstructed.forecast_yield,
                                "fit_status": forecast_result.fit_status,
                                "fallback_used": forecast_result.fallback_used,
                                "selected_order": "",
                                "selection_score": np.nan,
                                "transformed_forecast": reconstructed.transformed_forecast,
                                "anchor_model_id": reconstructed.anchor_model_id,
                                "correction_weight": reconstructed.correction_weight,
                                "forecast_target": reconstructed.forecast_target,
                                "effective_interval_count": forecast_result.effective_interval_count,
                                "rule_count": len(forecast_result.rule_summary),
                                "duplicate_boundary_count": forecast_result.duplicate_boundary_count,
                            }
                        )
                        rows.append(row)

    predictions = pd.DataFrame(rows)
    predictions = _add_error_columns(predictions)
    ordered_columns = [
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
    return predictions[ordered_columns]


def build_metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_metrics = grouped_forecast_metrics(predictions, MODEL_GROUP_COLUMNS)
    overall_metrics = overall_metrics.sort_values(["mae", "rmse"]).reset_index(drop=True)
    overall_metrics["mae_rank"] = overall_metrics["mae"].rank(method="min").astype(int)
    overall_metrics["rmse_rank"] = overall_metrics["rmse"].rank(method="min").astype(int)
    overall_metrics["smape_rank"] = overall_metrics["smape"].rank(method="min").astype(int)
    overall_metrics["mase_rank"] = overall_metrics["mase"].rank(method="min").astype(int)

    volatility_metrics = grouped_forecast_metrics(
        predictions,
        MODEL_GROUP_COLUMNS + ["is_evaluation_high_volatility"],
    )
    volatility_metrics = volatility_metrics.sort_values(
        ["is_evaluation_high_volatility", "mae", "rmse"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return overall_metrics, volatility_metrics


def _metric_lookup(metrics: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        str(row["model_id"]): row
        for _, row in metrics.iterrows()
    }


def _comparison_rows(
    left_metrics: pd.DataFrame,
    right_metrics: pd.DataFrame,
    left_label: str,
    right_label: str,
    left_model_prefix: str,
    right_model_prefix: str,
    interval_counts: tuple[int, ...],
    evaluation_view: str,
) -> list[dict[str, object]]:
    left_lookup = _metric_lookup(left_metrics)
    right_lookup = _metric_lookup(right_metrics)
    rows = []
    for interval_count in interval_counts:
        left_model_id = f"{left_model_prefix}{interval_count}"
        right_model_id = f"{right_model_prefix}{interval_count}"
        if left_model_id not in left_lookup or right_model_id not in right_lookup:
            continue
        left_row = left_lookup[left_model_id]
        right_row = right_lookup[right_model_id]
        for metric in ACCURACY_METRICS:
            left_value = float(left_row[metric])
            right_value = float(right_row[metric])
            rows.append(
                {
                    "evaluation_view": evaluation_view,
                    "interval_count": interval_count,
                    "metric": metric,
                    "left_label": left_label,
                    "left_model_id": left_model_id,
                    "left_value": left_value,
                    "right_label": right_label,
                    "right_model_id": right_model_id,
                    "right_value": right_value,
                    "right_minus_left": right_value - left_value,
                    "right_improved": right_value < left_value,
                }
            )
    return rows


def build_interval_strategy_comparison(
    overall_metrics: pd.DataFrame,
    volatility_metrics: pd.DataFrame,
) -> pd.DataFrame:
    high_volatility_metrics = volatility_metrics.loc[
        volatility_metrics["is_evaluation_high_volatility"]
    ]
    rows = []
    rows.extend(
        _comparison_rows(
            overall_metrics,
            overall_metrics,
            "equal_length",
            "adaptive_quantile",
            "chen_equal_k",
            "chen_adaptive_k",
            ADAPTIVE_INTERVAL_COUNTS,
            "overall",
        )
    )
    rows.extend(
        _comparison_rows(
            high_volatility_metrics,
            high_volatility_metrics,
            "equal_length",
            "adaptive_quantile",
            "chen_equal_k",
            "chen_adaptive_k",
            ADAPTIVE_INTERVAL_COUNTS,
            "high_volatility",
        )
    )
    return pd.DataFrame(rows)


def build_rule_order_comparison(
    overall_metrics: pd.DataFrame,
    volatility_metrics: pd.DataFrame,
) -> pd.DataFrame:
    high_volatility_metrics = volatility_metrics.loc[
        volatility_metrics["is_evaluation_high_volatility"]
    ]
    rows = []
    rows.extend(
        _comparison_rows(
            overall_metrics,
            overall_metrics,
            "first_order",
            "second_order",
            "chen_adaptive_k",
            "chen_order2_adaptive_k",
            HIGH_ORDER_INTERVAL_COUNTS,
            "overall",
        )
    )
    rows.extend(
        _comparison_rows(
            high_volatility_metrics,
            high_volatility_metrics,
            "first_order",
            "second_order",
            "chen_adaptive_k",
            "chen_order2_adaptive_k",
            HIGH_ORDER_INTERVAL_COUNTS,
            "high_volatility",
        )
    )
    return pd.DataFrame(rows)


def _best_metric_row(metrics: pd.DataFrame, selector: pd.Series | None = None) -> pd.Series:
    selected = metrics if selector is None else metrics.loc[selector]
    if selected.empty:
        raise ValueError("Cannot select a best model from an empty metric table.")
    return selected.sort_values(["mae", "rmse", "smape"]).iloc[0]


def _metric_views(
    overall_metrics: pd.DataFrame,
    volatility_metrics: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame]]:
    high_volatility = volatility_metrics.loc[
        volatility_metrics["is_evaluation_high_volatility"]
    ]
    normal_volatility = volatility_metrics.loc[
        ~volatility_metrics["is_evaluation_high_volatility"]
    ]
    return [
        ("overall", overall_metrics),
        ("high_volatility", high_volatility),
        ("normal_volatility", normal_volatility),
    ]


def build_forecast_target_comparison(
    overall_metrics: pd.DataFrame,
    volatility_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for evaluation_view, metrics in _metric_views(overall_metrics, volatility_metrics):
        correction_metrics = metrics.loc[metrics["forecast_target"].isin(CORRECTION_TARGETS)]
        for forecast_target, target_group in correction_metrics.groupby("forecast_target"):
            best = _best_metric_row(target_group)
            rows.append(
                {
                    "evaluation_view": evaluation_view,
                    "forecast_target": forecast_target,
                    "best_model_id": best["model_id"],
                    "interval_strategy": best["interval_strategy"],
                    "interval_count": best["interval_count"],
                    "correction_weight": best["correction_weight"],
                    "mae": best["mae"],
                    "rmse": best["rmse"],
                    "smape": best["smape"],
                    "mase": best["mase"],
                }
            )
    return pd.DataFrame(rows)


def build_fuzzy_correction_comparison(
    overall_metrics: pd.DataFrame,
    volatility_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for evaluation_view, metrics in _metric_views(overall_metrics, volatility_metrics):
        level_fuzzy = metrics.loc[
            metrics["model_id"].str.startswith("chen_")
            & metrics["forecast_target"].eq("yield_level")
        ]
        correction_fuzzy = metrics.loc[metrics["forecast_target"].isin(CORRECTION_TARGETS)]

        best_level = _best_metric_row(level_fuzzy)
        best_correction = _best_metric_row(correction_fuzzy)
        for metric in ACCURACY_METRICS:
            level_value = float(best_level[metric])
            correction_value = float(best_correction[metric])
            rows.append(
                {
                    "evaluation_view": evaluation_view,
                    "metric": metric,
                    "best_level_model_id": best_level["model_id"],
                    "best_level_value": level_value,
                    "best_correction_model_id": best_correction["model_id"],
                    "best_correction_value": correction_value,
                    "correction_minus_level": correction_value - level_value,
                    "correction_improved": correction_value < level_value,
                }
            )
    return pd.DataFrame(rows)


def build_fuzzy_baseline_gap_analysis(
    overall_metrics: pd.DataFrame,
    volatility_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for evaluation_view, metrics in _metric_views(overall_metrics, volatility_metrics):
        naive = metrics.loc[metrics["model_id"].eq("baseline_naive")].iloc[0]
        arima = metrics.loc[metrics["model_id"].eq("baseline_arima_aic")].iloc[0]
        level_fuzzy = metrics.loc[
            metrics["model_id"].str.startswith("chen_")
            & metrics["forecast_target"].eq("yield_level")
        ]
        correction_fuzzy = metrics.loc[metrics["forecast_target"].isin(CORRECTION_TARGETS)]
        best_level = _best_metric_row(level_fuzzy)
        best_correction = _best_metric_row(correction_fuzzy)

        for metric in ACCURACY_METRICS:
            naive_value = float(naive[metric])
            arima_value = float(arima[metric])
            level_value = float(best_level[metric])
            correction_value = float(best_correction[metric])
            level_gap = level_value - naive_value
            correction_gap = correction_value - naive_value
            if level_gap == 0:
                gap_reduction = np.nan
            else:
                gap_reduction = 100 * (level_gap - correction_gap) / abs(level_gap)
            rows.append(
                {
                    "evaluation_view": evaluation_view,
                    "metric": metric,
                    "baseline_naive_value": naive_value,
                    "baseline_arima_aic_value": arima_value,
                    "best_level_model_id": best_level["model_id"],
                    "best_level_value": level_value,
                    "best_correction_model_id": best_correction["model_id"],
                    "best_correction_value": correction_value,
                    "level_minus_naive": level_gap,
                    "correction_minus_naive": correction_gap,
                    "gap_reduction_percent": gap_reduction,
                    "correction_beats_naive": correction_value < naive_value,
                    "correction_beats_arima_aic": correction_value < arima_value,
                }
            )
    return pd.DataFrame(rows)


def build_enhanced_model_recommendations(
    overall_metrics: pd.DataFrame,
    volatility_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    gap_analysis = build_fuzzy_baseline_gap_analysis(overall_metrics, volatility_metrics)
    for evaluation_view in ["overall", "high_volatility"]:
        view_rows = gap_analysis.loc[
            gap_analysis["evaluation_view"].eq(evaluation_view)
            & gap_analysis["metric"].eq("mae")
        ]
        row = view_rows.iloc[0]
        correction_is_best_fuzzy = row["best_correction_value"] < row["best_level_value"]
        if correction_is_best_fuzzy and row["correction_beats_naive"]:
            recommendation = row["best_correction_model_id"]
            claim = "A fuzzy correction model beats the naive benchmark for this evaluation view."
        elif correction_is_best_fuzzy and row["gap_reduction_percent"] > 0:
            recommendation = row["best_correction_model_id"]
            claim = "A fuzzy correction model narrows the gap to the naive benchmark but does not beat it."
        else:
            recommendation = row["best_level_model_id"]
            if row["best_level_value"] < row["baseline_naive_value"]:
                claim = (
                    "A level-based fuzzy model remains the strongest fuzzy design and beats "
                    "the naive benchmark for this evaluation view."
                )
            else:
                claim = (
                    "The existing level-based fuzzy model remains stronger than fuzzy "
                    "correction for this evaluation view."
                )
        rows.append(
            {
                "evaluation_view": evaluation_view,
                "recommended_model_id": recommendation,
                "naive_mae": row["baseline_naive_value"],
                "recommended_mae": (
                    row["best_correction_value"]
                    if recommendation == row["best_correction_model_id"]
                    else row["best_level_value"]
                ),
                "gap_to_naive": (
                    row["correction_minus_naive"]
                    if recommendation == row["best_correction_model_id"]
                    else row["level_minus_naive"]
                ),
                "claim_boundary": claim,
            }
        )
    return pd.DataFrame(rows)


def build_final_window_model_outputs(
    data: pd.DataFrame,
    splits: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    final_split = splits.sort_values("split_id").iloc[-1]
    training_data = data.loc[
        data["year"].between(
            int(final_split["train_start_year"]),
            int(final_split["train_end_year"]),
        )
    ]
    training_values = training_data[YIELD_COLUMN].astype(float).to_numpy()

    boundary_tables = []
    rule_tables = []
    adaptive_boundary_tables = []
    adaptive_rule_tables = []
    high_order_rule_tables = []
    change_rule_tables = []
    percentage_change_rule_tables = []
    for interval_count in EQUAL_INTERVAL_COUNTS:
        model_id = f"chen_equal_k{interval_count}"
        forecast_result = forecast_first_order_equal_chen(training_values, interval_count)

        boundaries = forecast_result.intervals.copy()
        boundaries.insert(0, "model_id", model_id)
        boundaries.insert(1, "interval_count", interval_count)
        boundaries.insert(2, "train_start_year", int(final_split["train_start_year"]))
        boundaries.insert(3, "train_end_year", int(final_split["train_end_year"]))
        boundary_tables.append(boundaries)

        rules = forecast_result.rule_summary.copy()
        rules.insert(0, "model_id", model_id)
        rules.insert(1, "interval_count", interval_count)
        rules.insert(2, "train_start_year", int(final_split["train_start_year"]))
        rules.insert(3, "train_end_year", int(final_split["train_end_year"]))
        rule_tables.append(rules)

    for interval_count in ADAPTIVE_INTERVAL_COUNTS:
        model_id = f"chen_adaptive_k{interval_count}"
        forecast_result = forecast_first_order_chen(
            training_values,
            interval_count,
            interval_strategy="adaptive_quantile",
        )

        boundaries = forecast_result.intervals.copy()
        boundaries.insert(0, "model_id", model_id)
        boundaries.insert(1, "interval_count", interval_count)
        boundaries.insert(2, "train_start_year", int(final_split["train_start_year"]))
        boundaries.insert(3, "train_end_year", int(final_split["train_end_year"]))
        adaptive_boundary_tables.append(boundaries)

        rules = forecast_result.rule_summary.copy()
        rules.insert(0, "model_id", model_id)
        rules.insert(1, "interval_count", interval_count)
        rules.insert(2, "train_start_year", int(final_split["train_start_year"]))
        rules.insert(3, "train_end_year", int(final_split["train_end_year"]))
        adaptive_rule_tables.append(rules)

    for interval_count in HIGH_ORDER_INTERVAL_COUNTS:
        model_id = f"chen_order2_adaptive_k{interval_count}"
        forecast_result = forecast_second_order_chen(
            training_values,
            interval_count,
            interval_strategy="adaptive_quantile",
        )

        rules = forecast_result.rule_summary.copy()
        rules.insert(0, "model_id", model_id)
        rules.insert(1, "interval_count", interval_count)
        rules.insert(2, "train_start_year", int(final_split["train_start_year"]))
        rules.insert(3, "train_end_year", int(final_split["train_end_year"]))
        high_order_rule_tables.append(rules)

    transformed_rule_tables = {
        ABSOLUTE_CHANGE_TARGET: change_rule_tables,
        PERCENTAGE_CHANGE_TARGET: percentage_change_rule_tables,
    }
    for forecast_target, target_rule_tables in transformed_rule_tables.items():
        transformed_values = build_transformed_target(training_values, forecast_target)
        for interval_strategy in CORRECTION_INTERVAL_STRATEGIES:
            for interval_count in CORRECTION_INTERVAL_COUNTS:
                model_group_id = (
                    f"{_target_model_prefix(forecast_target)}_"
                    f"{_strategy_model_suffix(interval_strategy)}_"
                    f"k{interval_count}"
                )
                forecast_result = forecast_first_order_chen(
                    transformed_values,
                    interval_count,
                    interval_strategy=interval_strategy,
                )
                rules = forecast_result.rule_summary.copy()
                rules.insert(0, "model_group_id", model_group_id)
                rules.insert(1, "forecast_target", forecast_target)
                rules.insert(2, "interval_strategy", interval_strategy)
                rules.insert(3, "interval_count", interval_count)
                rules.insert(4, "train_start_year", int(final_split["train_start_year"]))
                rules.insert(5, "train_end_year", int(final_split["train_end_year"]))
                rules.insert(
                    6,
                    "effective_interval_count",
                    forecast_result.effective_interval_count,
                )
                rules.insert(
                    7,
                    "duplicate_boundary_count",
                    forecast_result.duplicate_boundary_count,
                )
                target_rule_tables.append(rules)

    return (
        pd.concat(boundary_tables, ignore_index=True),
        pd.concat(rule_tables, ignore_index=True),
        pd.concat(adaptive_boundary_tables, ignore_index=True),
        pd.concat(adaptive_rule_tables, ignore_index=True),
        pd.concat(high_order_rule_tables, ignore_index=True),
        pd.concat(change_rule_tables, ignore_index=True),
        pd.concat(percentage_change_rule_tables, ignore_index=True),
    )


def validate_outputs(
    predictions: pd.DataFrame,
    model_scenarios: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    volatility_metrics: pd.DataFrame,
    forecast_target_comparison: pd.DataFrame,
    fuzzy_correction_comparison: pd.DataFrame,
    fuzzy_baseline_gap_analysis: pd.DataFrame,
    enhanced_model_recommendations: pd.DataFrame,
    boundaries: pd.DataFrame,
    rules: pd.DataFrame,
    adaptive_boundaries: pd.DataFrame,
    adaptive_rules: pd.DataFrame,
    high_order_rules: pd.DataFrame,
    change_rules: pd.DataFrame,
    percentage_change_rules: pd.DataFrame,
) -> None:
    expected_model_count = len(model_scenarios)
    expected_split_count = predictions["split_id"].nunique()
    expected_rows = expected_model_count * expected_split_count
    if len(predictions) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} prediction rows, found {len(predictions)}."
        )

    per_split_counts = predictions.groupby("split_id")["model_id"].nunique()
    if not per_split_counts.eq(expected_model_count).all():
        raise RuntimeError("Every split must contain one prediction for each model.")

    if not np.isfinite(predictions["forecast_yield"]).all():
        raise RuntimeError("Prediction table contains missing or non-finite forecasts.")

    correction_predictions = predictions.loc[
        predictions["forecast_target"].isin(CORRECTION_TARGETS)
    ]
    if correction_predictions.empty:
        raise RuntimeError("Correction model predictions are required.")
    metadata_columns = [
        "forecast_target",
        "correction_weight",
        "transformed_forecast",
        "anchor_model_id",
    ]
    if correction_predictions[metadata_columns].isna().any().any():
        raise RuntimeError("Correction model metadata contains missing values.")
    if not np.isfinite(correction_predictions["transformed_forecast"]).all():
        raise RuntimeError("Correction model transformed forecasts must be finite.")

    expected_model_ids = set(model_scenarios["model_id"])
    if set(overall_metrics["model_id"]) != expected_model_ids:
        raise RuntimeError("Overall metrics do not contain all model IDs.")

    volatility_counts = volatility_metrics.groupby("model_id")[
        "is_evaluation_high_volatility"
    ].nunique()
    if not volatility_counts.eq(2).all():
        raise RuntimeError("Volatility metrics must contain both groups for every model.")

    if boundaries.empty or rules.empty or adaptive_boundaries.empty:
        raise RuntimeError("Final-window fuzzy interval and rule outputs are required.")
    if adaptive_rules.empty or high_order_rules.empty:
        raise RuntimeError("Final-window adaptive and high-order rules are required.")
    if change_rules.empty or percentage_change_rules.empty:
        raise RuntimeError("Final-window transformed-target rules are required.")

    comparison_tables = [
        forecast_target_comparison,
        fuzzy_correction_comparison,
        fuzzy_baseline_gap_analysis,
        enhanced_model_recommendations,
    ]
    if any(table.empty for table in comparison_tables):
        raise RuntimeError("Correction comparison tables are required.")


def write_results_readme(
    splits: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    volatility_metrics: pd.DataFrame,
) -> None:
    first_split = splits.iloc[0]
    last_split = splits.iloc[-1]
    threshold = float(splits["evaluation_volatility_threshold"].iloc[0])
    best_overall = overall_metrics.sort_values("mae").iloc[0]
    high_volatility = volatility_metrics.loc[
        volatility_metrics["is_evaluation_high_volatility"]
    ]
    best_high_volatility = high_volatility.sort_values("mae").iloc[0]

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
- Evaluation high-volatility year count: {int(splits["is_evaluation_high_volatility"].sum())}.

## Baseline And Fuzzy Time-Series Models

Generated by:

```powershell
.\\.venv\\Scripts\\python.exe -m src.modeling
```

Tables:

- `results/tables/model_scenarios.csv`: model IDs, families, settings, and descriptions.
- `results/tables/rolling_predictions.csv`: one-step-ahead forecasts for each model and validation split.
- `results/tables/overall_metrics.csv`: MAE, RMSE, sMAPE, and MASE for all evaluation years.
- `results/tables/volatility_metrics.csv`: metrics split by evaluation high-volatility label.
- `results/tables/interval_strategy_comparison.csv`: equal-length versus adaptive interval comparisons.
- `results/tables/rule_order_comparison.csv`: first-order versus order-2 adaptive rule comparisons.
- `results/tables/forecast_target_comparison.csv`: absolute-change versus percentage-change correction comparison.
- `results/tables/fuzzy_correction_comparison.csv`: level-based fuzzy versus correction fuzzy comparison.
- `results/tables/fuzzy_baseline_gap_analysis.csv`: fuzzy model gaps to Naive and ARIMA-AIC.
- `results/tables/enhanced_model_recommendations.csv`: concise model-selection and claim-boundary guidance.

Model outputs:

- `results/model_outputs/equal_interval_boundaries_final_window.csv`: equal interval boundaries for the final training window.
- `results/model_outputs/equal_interval_rules_final_window.csv`: Chen first-order FLRG summaries for the final training window.
- `results/model_outputs/adaptive_interval_boundaries_final_window.csv`: adaptive interval boundaries for the final training window.
- `results/model_outputs/adaptive_interval_rules_final_window.csv`: adaptive Chen first-order FLRG summaries for the final training window.
- `results/model_outputs/high_order_rules_final_window.csv`: order-2 adaptive FLRG summaries for the final training window.
- `results/model_outputs/change_interval_rules_final_window.csv`: transformed-target FLRG summaries for absolute-change correction.
- `results/model_outputs/percentage_change_interval_rules_final_window.csv`: transformed-target FLRG summaries for percentage-change correction.

Key modeling values:

- Model count: {overall_metrics["model_id"].nunique()}.
- Prediction rows: {len(splits) * overall_metrics["model_id"].nunique()}.
- Best overall MAE model: {best_overall["model_id"]} ({best_overall["mae"]:.6g}).
- Best high-volatility MAE model: {best_high_volatility["model_id"]} ({best_high_volatility["mae"]:.6g}).

Note:

Exploratory high-volatility labels use the full observed series. Evaluation high-volatility labels use only the rolling-origin evaluation period and should be used for model comparison.
"""
    RESULTS_README_PATH.write_text(content, encoding="utf-8")


def run() -> None:
    ensure_directories()
    data = load_yield_data()
    splits = build_validation_splits(data)
    model_scenarios = build_model_scenarios()
    predictions = build_rolling_predictions(data, splits, model_scenarios)
    overall_metrics, volatility_metrics = build_metric_tables(predictions)
    interval_strategy_comparison = build_interval_strategy_comparison(
        overall_metrics,
        volatility_metrics,
    )
    rule_order_comparison = build_rule_order_comparison(
        overall_metrics,
        volatility_metrics,
    )
    forecast_target_comparison = build_forecast_target_comparison(
        overall_metrics,
        volatility_metrics,
    )
    fuzzy_correction_comparison = build_fuzzy_correction_comparison(
        overall_metrics,
        volatility_metrics,
    )
    fuzzy_baseline_gap_analysis = build_fuzzy_baseline_gap_analysis(
        overall_metrics,
        volatility_metrics,
    )
    enhanced_model_recommendations = build_enhanced_model_recommendations(
        overall_metrics,
        volatility_metrics,
    )
    (
        boundaries,
        rules,
        adaptive_boundaries,
        adaptive_rules,
        high_order_rules,
        change_rules,
        percentage_change_rules,
    ) = build_final_window_model_outputs(data, splits)

    validate_outputs(
        predictions,
        model_scenarios,
        overall_metrics,
        volatility_metrics,
        forecast_target_comparison,
        fuzzy_correction_comparison,
        fuzzy_baseline_gap_analysis,
        enhanced_model_recommendations,
        boundaries,
        rules,
        adaptive_boundaries,
        adaptive_rules,
        high_order_rules,
        change_rules,
        percentage_change_rules,
    )

    model_scenarios.to_csv(MODEL_SCENARIOS_PATH, index=False)
    predictions.to_csv(ROLLING_PREDICTIONS_PATH, index=False)
    overall_metrics.to_csv(OVERALL_METRICS_PATH, index=False)
    volatility_metrics.to_csv(VOLATILITY_METRICS_PATH, index=False)
    interval_strategy_comparison.to_csv(
        INTERVAL_STRATEGY_COMPARISON_PATH,
        index=False,
    )
    rule_order_comparison.to_csv(RULE_ORDER_COMPARISON_PATH, index=False)
    forecast_target_comparison.to_csv(FORECAST_TARGET_COMPARISON_PATH, index=False)
    fuzzy_correction_comparison.to_csv(FUZZY_CORRECTION_COMPARISON_PATH, index=False)
    fuzzy_baseline_gap_analysis.to_csv(FUZZY_BASELINE_GAP_ANALYSIS_PATH, index=False)
    enhanced_model_recommendations.to_csv(
        ENHANCED_MODEL_RECOMMENDATIONS_PATH,
        index=False,
    )
    boundaries.to_csv(FINAL_WINDOW_BOUNDARIES_PATH, index=False)
    rules.to_csv(FINAL_WINDOW_RULES_PATH, index=False)
    adaptive_boundaries.to_csv(ADAPTIVE_WINDOW_BOUNDARIES_PATH, index=False)
    adaptive_rules.to_csv(ADAPTIVE_WINDOW_RULES_PATH, index=False)
    high_order_rules.to_csv(HIGH_ORDER_RULES_PATH, index=False)
    change_rules.to_csv(CHANGE_TARGET_RULES_PATH, index=False)
    percentage_change_rules.to_csv(PERCENTAGE_CHANGE_TARGET_RULES_PATH, index=False)
    write_results_readme(splits, overall_metrics, volatility_metrics)

    print("Modeling pipeline complete.")
    print(f"Model scenarios: {MODEL_SCENARIOS_PATH}")
    print(f"Rolling predictions: {ROLLING_PREDICTIONS_PATH}")
    print(f"Overall metrics: {OVERALL_METRICS_PATH}")
    print(f"Volatility metrics: {VOLATILITY_METRICS_PATH}")
    print(f"Interval strategy comparison: {INTERVAL_STRATEGY_COMPARISON_PATH}")
    print(f"Rule order comparison: {RULE_ORDER_COMPARISON_PATH}")
    print(f"Forecast target comparison: {FORECAST_TARGET_COMPARISON_PATH}")
    print(f"Fuzzy correction comparison: {FUZZY_CORRECTION_COMPARISON_PATH}")
    print(f"Fuzzy baseline gap analysis: {FUZZY_BASELINE_GAP_ANALYSIS_PATH}")
    print(f"Enhanced model recommendations: {ENHANCED_MODEL_RECOMMENDATIONS_PATH}")
    print(f"Final-window interval boundaries: {FINAL_WINDOW_BOUNDARIES_PATH}")
    print(f"Final-window fuzzy rules: {FINAL_WINDOW_RULES_PATH}")
    print(f"Adaptive interval boundaries: {ADAPTIVE_WINDOW_BOUNDARIES_PATH}")
    print(f"Adaptive fuzzy rules: {ADAPTIVE_WINDOW_RULES_PATH}")
    print(f"High-order fuzzy rules: {HIGH_ORDER_RULES_PATH}")
    print(f"Change-target fuzzy rules: {CHANGE_TARGET_RULES_PATH}")
    print(f"Percentage-change fuzzy rules: {PERCENTAGE_CHANGE_TARGET_RULES_PATH}")


if __name__ == "__main__":
    run()
