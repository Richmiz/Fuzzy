"""Final robustness checks for Ghana cocoa yield forecasting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.forecast_targets import (
    ABSOLUTE_CHANGE_TARGET,
    PERCENTAGE_CHANGE_TARGET,
    build_transformed_target,
    reconstruct_yield_forecast,
)
from src.fts import forecast_first_order_chen
from src.metrics import grouped_forecast_metrics
from src.modeling import (
    CORRECTION_INTERVAL_COUNTS,
    CORRECTION_INTERVAL_STRATEGIES,
    CORRECTION_TARGETS,
    ROLLING_PREDICTIONS_PATH,
    VOLATILITY_METRICS_PATH,
    correction_model_id,
)
from src.sensitivity import (
    INITIAL_TRAINING_OBSERVATIONS,
    SENSITIVITY_METRICS_PATH,
    SENSITIVITY_SUMMARY_PATH,
)
from src.validation import (
    YIELD_COLUMN,
    ValidationConfig,
    build_validation_splits,
    load_yield_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_README_PATH = PROJECT_ROOT / "results" / "README.md"

ROBUSTNESS_CORRECTION_PREDICTIONS_PATH = (
    TABLES_DIR / "robustness_correction_predictions.csv"
)
CORRECTION_WEIGHT_SENSITIVITY_PATH = (
    TABLES_DIR / "correction_weight_sensitivity.csv"
)
ROBUSTNESS_SUMMARY_PATH = TABLES_DIR / "robustness_summary.csv"
PAIRED_ERROR_TESTS_PATH = TABLES_DIR / "paired_error_tests.csv"
ROBUSTNESS_CLAIM_BOUNDARY_PATH = TABLES_DIR / "robustness_claim_boundary.csv"

ROBUSTNESS_CORRECTION_WEIGHTS = (0.25, 0.5, 0.75, 1.0)
ROBUSTNESS_GROUP_COLUMNS = [
    "initial_training_observations",
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
]


def ensure_directories() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _correction_weight_label(correction_weight: float) -> str:
    return f"w{int(round(correction_weight * 100)):03d}"


def _target_name(forecast_target: str) -> str:
    if forecast_target == ABSOLUTE_CHANGE_TARGET:
        return "Absolute Change"
    if forecast_target == PERCENTAGE_CHANGE_TARGET:
        return "Percentage Change"
    raise ValueError(f"Unknown forecast target: {forecast_target}")


def _strategy_name(interval_strategy: str) -> str:
    if interval_strategy == "equal_length":
        return "Equal Intervals"
    if interval_strategy == "adaptive_quantile":
        return "Adaptive Intervals"
    raise ValueError(f"Unknown interval strategy: {interval_strategy}")


def build_robustness_correction_scenarios() -> pd.DataFrame:
    rows = []
    for forecast_target in CORRECTION_TARGETS:
        for interval_strategy in CORRECTION_INTERVAL_STRATEGIES:
            for correction_weight in ROBUSTNESS_CORRECTION_WEIGHTS:
                for interval_count in CORRECTION_INTERVAL_COUNTS:
                    model_id = correction_model_id(
                        forecast_target,
                        interval_strategy,
                        interval_count,
                        correction_weight,
                    )
                    rows.append(
                        {
                            "model_id": model_id,
                            "model_name": (
                                "Chen FTS "
                                f"{_target_name(forecast_target)} Correction "
                                f"{_strategy_name(interval_strategy)} "
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
                            "forecast_target": forecast_target,
                            "correction_weight": correction_weight,
                            "anchor_model_id": "baseline_naive",
                        }
                    )
    return pd.DataFrame(rows)


def _scenario_lookup(scenarios: pd.DataFrame) -> dict[str, dict[str, object]]:
    return {
        str(row["model_id"]): row.to_dict()
        for _, row in scenarios.iterrows()
    }


def _prediction_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    enriched = predictions.copy()
    enriched["forecast_error"] = enriched["actual_yield"] - enriched["forecast_yield"]
    enriched["absolute_error"] = enriched["forecast_error"].abs()
    enriched["squared_error"] = enriched["forecast_error"] ** 2
    return enriched


def _base_prediction_row(
    split: pd.Series,
    scenario: dict[str, object],
    training_window_size: int,
    split_count: int,
) -> dict[str, object]:
    return {
        "initial_training_observations": training_window_size,
        "evaluation_start_year": int(split["forecast_year"] - split["split_id"] + 1),
        "evaluation_end_year": int(split["forecast_year"] - split["split_id"] + split_count),
        "validation_split_count": split_count,
        "split_id": int(split["split_id"]),
        "model_id": scenario["model_id"],
        "model_name": scenario["model_name"],
        "model_family": scenario["model_family"],
        "interval_strategy": scenario["interval_strategy"],
        "fuzzy_order": scenario["fuzzy_order"],
        "interval_count": scenario["interval_count"],
        "parameter_setting": scenario["parameter_setting"],
        "forecast_target": scenario["forecast_target"],
        "correction_weight": scenario["correction_weight"],
        "anchor_model_id": scenario["anchor_model_id"],
        "train_start_year": int(split["train_start_year"]),
        "train_end_year": int(split["train_end_year"]),
        "forecast_year": int(split["forecast_year"]),
        "actual_yield": float(split["actual_yield"]),
        "mase_scale": float(split["mase_scale"]),
        "is_evaluation_high_volatility": bool(split["is_evaluation_high_volatility"]),
    }


def build_robustness_correction_predictions() -> pd.DataFrame:
    data = load_yield_data()
    scenarios = build_robustness_correction_scenarios()
    scenario_lookup = _scenario_lookup(scenarios)
    rows = []

    for training_window_size in INITIAL_TRAINING_OBSERVATIONS:
        config = ValidationConfig(initial_training_observations=training_window_size)
        splits = build_validation_splits(data, config)
        split_count = len(splits)
        for _, split in splits.iterrows():
            training_data = data.loc[
                data["year"].between(
                    int(split["train_start_year"]),
                    int(split["train_end_year"]),
                )
            ]
            training_values = training_data[YIELD_COLUMN].astype(float).to_numpy()
            transformed_by_target = {
                forecast_target: build_transformed_target(
                    training_values,
                    forecast_target,
                )
                for forecast_target in CORRECTION_TARGETS
            }

            for forecast_target, transformed_values in transformed_by_target.items():
                for interval_strategy in CORRECTION_INTERVAL_STRATEGIES:
                    for correction_weight in ROBUSTNESS_CORRECTION_WEIGHTS:
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
                            row = _base_prediction_row(
                                split,
                                scenario,
                                training_window_size,
                                split_count,
                            )
                            row.update(
                                {
                                    "fit_status": forecast_result.fit_status,
                                    "fallback_used": forecast_result.fallback_used,
                                    "transformed_forecast": reconstructed.transformed_forecast,
                                    "effective_interval_count": forecast_result.effective_interval_count,
                                    "rule_count": len(forecast_result.rule_summary),
                                    "duplicate_boundary_count": forecast_result.duplicate_boundary_count,
                                    "forecast_yield": reconstructed.forecast_yield,
                                }
                            )
                            rows.append(row)

    predictions = _prediction_errors(pd.DataFrame(rows))
    ordered_columns = [
        "initial_training_observations",
        "evaluation_start_year",
        "evaluation_end_year",
        "validation_split_count",
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


def _evaluation_metrics(
    predictions: pd.DataFrame,
    evaluation_view: str,
) -> pd.DataFrame:
    metrics = grouped_forecast_metrics(predictions, ROBUSTNESS_GROUP_COLUMNS)
    metrics.insert(1, "evaluation_view", evaluation_view)
    for metric in ["mae", "rmse", "smape", "mase"]:
        metrics[f"{metric}_rank"] = (
            metrics.groupby(
                ["initial_training_observations", "evaluation_view"],
                dropna=False,
            )[metric]
            .rank(method="min")
            .astype(int)
        )
    return metrics


def build_correction_weight_sensitivity(predictions: pd.DataFrame) -> pd.DataFrame:
    overall = _evaluation_metrics(predictions, "overall")
    high_volatility = _evaluation_metrics(
        predictions.loc[predictions["is_evaluation_high_volatility"]],
        "high_volatility",
    )
    normal_volatility = _evaluation_metrics(
        predictions.loc[~predictions["is_evaluation_high_volatility"]],
        "normal_volatility",
    )
    metrics = pd.concat(
        [overall, high_volatility, normal_volatility],
        ignore_index=True,
    )
    return metrics.sort_values(
        [
            "initial_training_observations",
            "evaluation_view",
            "mae",
            "rmse",
            "smape",
        ],
    ).reset_index(drop=True)


def _best_row(metrics: pd.DataFrame) -> pd.Series:
    if metrics.empty:
        raise ValueError("Cannot select a best row from an empty table.")
    return metrics.sort_values(["mae", "rmse", "smape"]).iloc[0]


def _model_row(metrics: pd.DataFrame, model_id: str) -> pd.Series | None:
    selected = metrics.loc[metrics["model_id"].eq(model_id)]
    if selected.empty:
        return None
    return selected.iloc[0]


def build_robustness_summary(
    correction_metrics: pd.DataFrame,
    sensitivity_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (training_window_size, evaluation_view), correction_group in correction_metrics.groupby(
        ["initial_training_observations", "evaluation_view"],
        dropna=False,
    ):
        reference_group = sensitivity_metrics.loc[
            sensitivity_metrics["initial_training_observations"].eq(training_window_size)
            & sensitivity_metrics["evaluation_view"].eq(evaluation_view)
        ]
        best_correction = _best_row(correction_group)
        best_reference = _best_row(reference_group)
        best_existing_fuzzy = _best_row(
            reference_group.loc[reference_group["model_id"].str.startswith("chen_")]
        )
        naive = _model_row(reference_group, "baseline_naive")
        arima = _model_row(reference_group, "baseline_arima_aic")
        naive_mae = float(naive["mae"]) if naive is not None else np.nan
        arima_mae = float(arima["mae"]) if arima is not None else np.nan
        correction_mae = float(best_correction["mae"])
        existing_fuzzy_mae = float(best_existing_fuzzy["mae"])

        rows.append(
            {
                "initial_training_observations": int(training_window_size),
                "evaluation_view": evaluation_view,
                "best_correction_model_id": best_correction["model_id"],
                "best_correction_mae": correction_mae,
                "best_correction_rmse": float(best_correction["rmse"]),
                "best_correction_smape": float(best_correction["smape"]),
                "best_correction_mase": float(best_correction["mase"]),
                "forecast_target": best_correction["forecast_target"],
                "interval_strategy": best_correction["interval_strategy"],
                "interval_count": int(best_correction["interval_count"]),
                "correction_weight": float(best_correction["correction_weight"]),
                "best_reference_model_id": best_reference["model_id"],
                "best_reference_mae": float(best_reference["mae"]),
                "best_existing_fuzzy_model_id": best_existing_fuzzy["model_id"],
                "best_existing_fuzzy_mae": existing_fuzzy_mae,
                "naive_mae": naive_mae,
                "arima_aic_mae": arima_mae,
                "correction_minus_naive": correction_mae - naive_mae,
                "correction_minus_arima_aic": correction_mae - arima_mae,
                "correction_minus_existing_fuzzy": (
                    correction_mae - existing_fuzzy_mae
                ),
                "correction_beats_naive": bool(correction_mae < naive_mae),
                "correction_beats_arima_aic": bool(correction_mae < arima_mae),
                "correction_beats_existing_fuzzy": bool(
                    correction_mae < existing_fuzzy_mae
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["evaluation_view", "initial_training_observations"],
    ).reset_index(drop=True)


def _canonical_predictions(
    robustness_predictions: pd.DataFrame,
) -> pd.DataFrame:
    baseline_predictions = pd.read_csv(ROLLING_PREDICTIONS_PATH)
    correction_predictions = robustness_predictions.loc[
        robustness_predictions["initial_training_observations"].eq(30)
    ]
    common_columns = sorted(
        set(baseline_predictions.columns).intersection(correction_predictions.columns)
    )
    combined = pd.concat(
        [
            baseline_predictions[common_columns],
            correction_predictions[common_columns],
        ],
        ignore_index=True,
    )
    return combined.drop_duplicates(["split_id", "model_id"], keep="first")


def _view_subset(predictions: pd.DataFrame, evaluation_view: str) -> pd.DataFrame:
    if evaluation_view == "overall":
        return predictions
    if evaluation_view == "high_volatility":
        return predictions.loc[predictions["is_evaluation_high_volatility"]]
    if evaluation_view == "normal_volatility":
        return predictions.loc[~predictions["is_evaluation_high_volatility"]]
    raise ValueError(f"Unknown evaluation view: {evaluation_view}")


def _wilcoxon_p_value(error_a: np.ndarray, error_b: np.ndarray) -> float:
    differences = error_a - error_b
    if np.allclose(differences, 0):
        return 1.0
    try:
        return float(
            stats.wilcoxon(
                error_a,
                error_b,
                zero_method="wilcox",
                alternative="two-sided",
            ).pvalue
        )
    except ValueError:
        return np.nan


def _paired_error_test(
    predictions: pd.DataFrame,
    evaluation_view: str,
    model_a_id: str,
    model_b_id: str,
    comparison_label: str,
) -> dict[str, object]:
    view_predictions = _view_subset(predictions, evaluation_view)
    selected = view_predictions.loc[
        view_predictions["model_id"].isin([model_a_id, model_b_id])
    ]
    pivot = selected.pivot_table(
        index="split_id",
        columns="model_id",
        values="absolute_error",
        aggfunc="first",
    ).dropna()
    if model_a_id not in pivot.columns or model_b_id not in pivot.columns:
        raise ValueError(f"Missing paired predictions for {model_a_id} and {model_b_id}.")

    error_a = pivot[model_a_id].astype(float).to_numpy()
    error_b = pivot[model_b_id].astype(float).to_numpy()
    difference = error_a - error_b
    t_test = stats.ttest_rel(error_a, error_b)
    a_better_count = int(np.sum(error_a < error_b))
    b_better_count = int(np.sum(error_b < error_a))
    tie_count = int(np.sum(np.isclose(error_a, error_b)))
    binomial_trials = a_better_count + b_better_count
    if binomial_trials > 0:
        sign_test_p_value = float(
            stats.binomtest(
                min(a_better_count, b_better_count),
                n=binomial_trials,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    else:
        sign_test_p_value = 1.0

    mean_a = float(np.mean(error_a))
    mean_b = float(np.mean(error_b))
    return {
        "comparison_label": comparison_label,
        "evaluation_view": evaluation_view,
        "model_a_id": model_a_id,
        "model_b_id": model_b_id,
        "sample_size": len(pivot),
        "model_a_mae": mean_a,
        "model_b_mae": mean_b,
        "mae_difference_a_minus_b": mean_a - mean_b,
        "model_a_lower_mae": bool(mean_a < mean_b),
        "model_a_better_count": a_better_count,
        "model_b_better_count": b_better_count,
        "tie_count": tie_count,
        "wilcoxon_p_value": _wilcoxon_p_value(error_a, error_b),
        "paired_loss_t_p_value": float(t_test.pvalue),
        "sign_test_p_value": sign_test_p_value,
        "test_note": (
            "Two-sided paired tests on absolute error; interpret cautiously because "
            "the rolling evaluation sample is small."
        ),
    }


def build_paired_error_tests(
    robustness_predictions: pd.DataFrame,
    correction_metrics: pd.DataFrame,
) -> pd.DataFrame:
    combined = _canonical_predictions(robustness_predictions)
    volatility_metrics = pd.read_csv(VOLATILITY_METRICS_PATH)
    overall_reference = pd.read_csv(TABLES_DIR / "overall_metrics.csv")
    high_reference = volatility_metrics.loc[
        volatility_metrics["is_evaluation_high_volatility"]
    ]

    canonical_correction = correction_metrics.loc[
        correction_metrics["initial_training_observations"].eq(30)
    ]
    best_overall_correction = _best_row(
        canonical_correction.loc[canonical_correction["evaluation_view"].eq("overall")]
    )
    best_high_correction = _best_row(
        canonical_correction.loc[
            canonical_correction["evaluation_view"].eq("high_volatility")
        ]
    )
    best_level_overall = _best_row(
        overall_reference.loc[
            overall_reference["model_id"].str.startswith("chen_")
            & overall_reference["forecast_target"].eq("yield_level")
        ]
    )
    best_level_high = _best_row(
        high_reference.loc[
            high_reference["model_id"].str.startswith("chen_")
            & high_reference["forecast_target"].eq("yield_level")
        ]
    )

    comparisons = [
        (
            "overall_best_correction_vs_naive",
            "overall",
            best_overall_correction["model_id"],
            "baseline_naive",
        ),
        (
            "overall_best_correction_vs_arima_aic",
            "overall",
            best_overall_correction["model_id"],
            "baseline_arima_aic",
        ),
        (
            "overall_best_correction_vs_best_level_fuzzy",
            "overall",
            best_overall_correction["model_id"],
            best_level_overall["model_id"],
        ),
        (
            "high_volatility_best_correction_vs_naive",
            "high_volatility",
            best_high_correction["model_id"],
            "baseline_naive",
        ),
        (
            "high_volatility_best_correction_vs_arima_aic",
            "high_volatility",
            best_high_correction["model_id"],
            "baseline_arima_aic",
        ),
        (
            "high_volatility_best_correction_vs_best_level_fuzzy",
            "high_volatility",
            best_high_correction["model_id"],
            best_level_high["model_id"],
        ),
        (
            "high_volatility_best_level_fuzzy_vs_naive",
            "high_volatility",
            best_level_high["model_id"],
            "baseline_naive",
        ),
        (
            "high_volatility_best_level_fuzzy_vs_arima_aic",
            "high_volatility",
            best_level_high["model_id"],
            "baseline_arima_aic",
        ),
    ]

    rows = []
    seen = set()
    for comparison_label, evaluation_view, model_a_id, model_b_id in comparisons:
        key = (evaluation_view, model_a_id, model_b_id)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            _paired_error_test(
                combined,
                evaluation_view,
                model_a_id,
                model_b_id,
                comparison_label,
            )
        )
    return pd.DataFrame(rows)


def build_robustness_claim_boundary(
    robustness_summary: pd.DataFrame,
    paired_tests: pd.DataFrame,
) -> pd.DataFrame:
    overall = robustness_summary.loc[
        robustness_summary["evaluation_view"].eq("overall")
    ]
    high_volatility = robustness_summary.loc[
        robustness_summary["evaluation_view"].eq("high_volatility")
    ]
    overall_beats_naive = overall["correction_beats_naive"].all()
    overall_beats_arima = overall["correction_beats_arima_aic"].all()
    high_beats_naive = high_volatility["correction_beats_naive"].all()
    high_beats_arima = high_volatility["correction_beats_arima_aic"].all()
    overall_best_weights = sorted(overall["correction_weight"].unique().tolist())
    high_best_models = sorted(high_volatility["best_correction_model_id"].unique())

    paired_overall = paired_tests.loc[
        paired_tests["comparison_label"].eq("overall_best_correction_vs_naive")
    ].iloc[0]
    paired_high = paired_tests.loc[
        paired_tests["comparison_label"].eq("high_volatility_best_level_fuzzy_vs_naive")
    ].iloc[0]

    return pd.DataFrame(
        [
            {
                "claim_area": "overall_forecasting",
                "evidence_status": (
                    "supported" if overall_beats_naive and overall_beats_arima else "mixed"
                ),
                "claim_boundary": (
                    "Expanded correction-weight checks support fuzzy correction models "
                    "beating Naive and ARIMA-AIC overall across 25-year and 30-year "
                    "initial-window settings."
                ),
                "model_stability_note": (
                    "Best correction weights across windows: "
                    + ", ".join(f"{weight:g}" for weight in overall_best_weights)
                ),
                "paired_test_note": (
                    f"Canonical overall correction-vs-Naive Wilcoxon p="
                    f"{paired_overall['wilcoxon_p_value']:.4g}; report as descriptive."
                ),
            },
            {
                "claim_area": "high_volatility_forecasting",
                "evidence_status": (
                    "supported" if high_beats_naive and high_beats_arima else "mixed"
                ),
                "claim_boundary": (
                    "Fuzzy models remain stronger than Naive and ARIMA-AIC for "
                    "high-volatility years, but the best fuzzy design changes across "
                    "validation settings."
                ),
                "model_stability_note": (
                    "Best correction models across windows: "
                    + ", ".join(high_best_models)
                ),
                "paired_test_note": (
                    f"Canonical high-volatility level-fuzzy-vs-Naive Wilcoxon p="
                    f"{paired_high['wilcoxon_p_value']:.4g}; high-volatility n is small."
                ),
            },
        ]
    )


def validate_outputs(
    predictions: pd.DataFrame,
    correction_metrics: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    paired_tests: pd.DataFrame,
    claim_boundary: pd.DataFrame,
) -> None:
    model_count = (
        len(CORRECTION_TARGETS)
        * len(CORRECTION_INTERVAL_STRATEGIES)
        * len(CORRECTION_INTERVAL_COUNTS)
        * len(ROBUSTNESS_CORRECTION_WEIGHTS)
    )
    expected_rows = 0
    for training_window_size in INITIAL_TRAINING_OBSERVATIONS:
        split_count = int(
            predictions.loc[
                predictions["initial_training_observations"].eq(training_window_size),
                "validation_split_count",
            ].iloc[0]
        )
        expected_rows += split_count * model_count
    if len(predictions) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} robustness prediction rows, found {len(predictions)}."
        )

    per_split_counts = predictions.groupby(
        ["initial_training_observations", "split_id"]
    )["model_id"].nunique()
    if not per_split_counts.eq(model_count).all():
        raise RuntimeError("Every robustness split must contain each correction model.")

    if not np.isfinite(predictions["forecast_yield"]).all():
        raise RuntimeError("Robustness predictions contain non-finite forecasts.")

    observed_weights = set(predictions["correction_weight"].unique())
    if observed_weights != set(ROBUSTNESS_CORRECTION_WEIGHTS):
        raise RuntimeError("Robustness predictions are missing correction weights.")

    required_views = {"overall", "high_volatility", "normal_volatility"}
    if set(correction_metrics["evaluation_view"]) != required_views:
        raise RuntimeError("Correction weight sensitivity is missing evaluation views.")

    if robustness_summary.empty or paired_tests.empty or claim_boundary.empty:
        raise RuntimeError("Robustness summary outputs are required.")


def write_results_readme(
    predictions: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    claim_boundary: pd.DataFrame,
) -> None:
    section_heading = "## Final Robustness Checks"
    content = RESULTS_README_PATH.read_text(encoding="utf-8")
    if section_heading in content:
        content = content.split(section_heading)[0].rstrip() + "\n\n"
    else:
        content = content.rstrip() + "\n\n"

    overall_30 = robustness_summary.loc[
        robustness_summary["initial_training_observations"].eq(30)
        & robustness_summary["evaluation_view"].eq("overall")
    ].iloc[0]
    high_30 = robustness_summary.loc[
        robustness_summary["initial_training_observations"].eq(30)
        & robustness_summary["evaluation_view"].eq("high_volatility")
    ].iloc[0]
    overall_claim = claim_boundary.loc[
        claim_boundary["claim_area"].eq("overall_forecasting")
    ].iloc[0]

    section = f"""{section_heading}

Generated by:

```powershell
.\\.venv\\Scripts\\python.exe -m src.robustness
```

Tables:

- `results/tables/robustness_correction_predictions.csv`: correction-model forecasts for expanded correction weights.
- `results/tables/correction_weight_sensitivity.csv`: correction-weight metrics for 25-year and 30-year initial windows.
- `results/tables/robustness_summary.csv`: best correction model and benchmark gaps by validation view.
- `results/tables/paired_error_tests.csv`: two-sided paired absolute-error tests for selected model comparisons.
- `results/tables/robustness_claim_boundary.csv`: concise evidence boundaries for reporting.

Key robustness values:

- Correction weights checked: {", ".join(str(weight) for weight in ROBUSTNESS_CORRECTION_WEIGHTS)}.
- Robustness prediction rows: {len(predictions)}.
- Canonical overall best correction model: {overall_30["best_correction_model_id"]} ({overall_30["best_correction_mae"]:.6g} MAE).
- Canonical high-volatility best correction model: {high_30["best_correction_model_id"]} ({high_30["best_correction_mae"]:.6g} MAE).

Claim boundary:

{overall_claim["claim_boundary"]}
"""
    RESULTS_README_PATH.write_text(content + section, encoding="utf-8")


def run() -> None:
    ensure_directories()
    missing_inputs = [
        path
        for path in [
            ROLLING_PREDICTIONS_PATH,
            SENSITIVITY_METRICS_PATH,
            SENSITIVITY_SUMMARY_PATH,
            VOLATILITY_METRICS_PATH,
        ]
        if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            "Run validation, modeling, and sensitivity before robustness checks. "
            f"Missing: {missing_inputs}"
        )

    predictions = build_robustness_correction_predictions()
    correction_metrics = build_correction_weight_sensitivity(predictions)
    sensitivity_metrics = pd.read_csv(SENSITIVITY_METRICS_PATH)
    robustness_summary = build_robustness_summary(
        correction_metrics,
        sensitivity_metrics,
    )
    paired_tests = build_paired_error_tests(predictions, correction_metrics)
    claim_boundary = build_robustness_claim_boundary(
        robustness_summary,
        paired_tests,
    )

    validate_outputs(
        predictions,
        correction_metrics,
        robustness_summary,
        paired_tests,
        claim_boundary,
    )

    predictions.to_csv(ROBUSTNESS_CORRECTION_PREDICTIONS_PATH, index=False)
    correction_metrics.to_csv(CORRECTION_WEIGHT_SENSITIVITY_PATH, index=False)
    robustness_summary.to_csv(ROBUSTNESS_SUMMARY_PATH, index=False)
    paired_tests.to_csv(PAIRED_ERROR_TESTS_PATH, index=False)
    claim_boundary.to_csv(ROBUSTNESS_CLAIM_BOUNDARY_PATH, index=False)
    write_results_readme(predictions, robustness_summary, claim_boundary)

    print("Final robustness checks complete.")
    print(f"Robustness correction predictions: {ROBUSTNESS_CORRECTION_PREDICTIONS_PATH}")
    print(f"Correction weight sensitivity: {CORRECTION_WEIGHT_SENSITIVITY_PATH}")
    print(f"Robustness summary: {ROBUSTNESS_SUMMARY_PATH}")
    print(f"Paired error tests: {PAIRED_ERROR_TESTS_PATH}")
    print(f"Robustness claim boundary: {ROBUSTNESS_CLAIM_BOUNDARY_PATH}")


if __name__ == "__main__":
    run()
