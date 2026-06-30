"""Sensitivity and robustness analysis for Ghana cocoa yield forecasting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.metrics import grouped_forecast_metrics
from src.modeling import (
    MODEL_GROUP_COLUMNS,
    build_model_scenarios,
    build_rolling_predictions,
)
from src.validation import ValidationConfig, build_validation_splits, load_yield_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_README_PATH = PROJECT_ROOT / "results" / "README.md"

SENSITIVITY_PREDICTIONS_PATH = TABLES_DIR / "sensitivity_predictions.csv"
SENSITIVITY_METRICS_PATH = TABLES_DIR / "sensitivity_metrics.csv"
SENSITIVITY_SUMMARY_PATH = TABLES_DIR / "sensitivity_summary.csv"
RECOMMENDED_MODEL_SELECTION_PATH = TABLES_DIR / "recommended_model_selection.csv"

INITIAL_TRAINING_OBSERVATIONS = (25, 30)
PRIMARY_METRIC = "mae"


def ensure_directories() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _evaluation_metrics(
    predictions: pd.DataFrame,
    evaluation_view: str,
) -> pd.DataFrame:
    group_columns = ["initial_training_observations", *MODEL_GROUP_COLUMNS]
    metrics = grouped_forecast_metrics(predictions, group_columns)
    metrics.insert(1, "evaluation_view", evaluation_view)
    rank_columns = ["mae", "rmse", "smape", "mase"]
    for metric in rank_columns:
        metrics[f"{metric}_rank"] = (
            metrics.groupby(
                ["initial_training_observations", "evaluation_view"],
                dropna=False,
            )[metric]
            .rank(method="min")
            .astype(int)
        )
    return metrics


def build_sensitivity_predictions() -> pd.DataFrame:
    data = load_yield_data()
    model_scenarios = build_model_scenarios()
    prediction_tables = []

    for training_window_size in INITIAL_TRAINING_OBSERVATIONS:
        config = ValidationConfig(initial_training_observations=training_window_size)
        splits = build_validation_splits(data, config)
        predictions = build_rolling_predictions(data, splits, model_scenarios)
        predictions.insert(0, "initial_training_observations", training_window_size)
        predictions.insert(1, "evaluation_start_year", int(splits["forecast_year"].min()))
        predictions.insert(2, "evaluation_end_year", int(splits["forecast_year"].max()))
        predictions.insert(3, "validation_split_count", len(splits))
        prediction_tables.append(predictions)

    return pd.concat(prediction_tables, ignore_index=True)


def build_sensitivity_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
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
        ["initial_training_observations", "evaluation_view", PRIMARY_METRIC],
    ).reset_index(drop=True)


def _model_row(metrics: pd.DataFrame, model_id: str) -> pd.Series | None:
    match = metrics.loc[metrics["model_id"] == model_id]
    if match.empty:
        return None
    return match.iloc[0]


def _best_row(metrics: pd.DataFrame, selector: pd.Series | None = None) -> pd.Series:
    selected = metrics if selector is None else metrics.loc[selector]
    if selected.empty:
        raise ValueError("Cannot choose a best model from an empty metric table.")
    return selected.sort_values([PRIMARY_METRIC, "rmse", "smape"]).iloc[0]


def build_sensitivity_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (training_window_size, evaluation_view), group in metrics.groupby(
        ["initial_training_observations", "evaluation_view"],
        dropna=False,
    ):
        best_model = _best_row(group)
        best_fuzzy = _best_row(group, group["model_id"].str.startswith("chen_"))
        best_baseline = _best_row(group, group["model_family"].eq("baseline"))
        naive = _model_row(group, "baseline_naive")
        arima = _model_row(group, "baseline_arima_aic")

        rows.append(
            {
                "initial_training_observations": int(training_window_size),
                "evaluation_view": evaluation_view,
                "best_model_id": best_model["model_id"],
                "best_model_mae": float(best_model["mae"]),
                "best_baseline_id": best_baseline["model_id"],
                "best_baseline_mae": float(best_baseline["mae"]),
                "best_fuzzy_model_id": best_fuzzy["model_id"],
                "best_fuzzy_mae": float(best_fuzzy["mae"]),
                "best_fuzzy_interval_strategy": best_fuzzy["interval_strategy"],
                "best_fuzzy_order": best_fuzzy["fuzzy_order"],
                "naive_mae": float(naive["mae"]) if naive is not None else pd.NA,
                "arima_aic_mae": float(arima["mae"]) if arima is not None else pd.NA,
                "fuzzy_beats_naive": bool(
                    naive is not None and float(best_fuzzy["mae"]) < float(naive["mae"])
                ),
                "fuzzy_beats_arima_aic": bool(
                    arima is not None and float(best_fuzzy["mae"]) < float(arima["mae"])
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["evaluation_view", "initial_training_observations"],
    ).reset_index(drop=True)


def build_recommended_model_selection(summary: pd.DataFrame) -> pd.DataFrame:
    overall = summary.loc[summary["evaluation_view"] == "overall"]
    high_volatility = summary.loc[summary["evaluation_view"] == "high_volatility"]

    overall_best_consistent = overall["best_model_id"].nunique() == 1
    high_volatility_fuzzy_consistent = (
        high_volatility["best_fuzzy_model_id"].nunique() == 1
    )
    high_volatility_fuzzy_beats_naive = high_volatility["fuzzy_beats_naive"].all()
    high_volatility_fuzzy_beats_arima = high_volatility["fuzzy_beats_arima_aic"].all()
    if high_volatility_fuzzy_beats_naive and high_volatility_fuzzy_beats_arima:
        if high_volatility_fuzzy_consistent:
            high_volatility_claim = (
                "The selected fuzzy design is robust across initial-window settings for high-volatility years."
            )
        else:
            high_volatility_claim = (
                "Fuzzy models beat baseline references for high-volatility years, but the best fuzzy design changes across initial-window settings."
            )
    else:
        high_volatility_claim = (
            "Treat the high-volatility fuzzy result as promising but not uniformly dominant across robustness checks."
        )

    rows = [
        {
            "recommendation_area": "overall_forecasting",
            "selected_model_id": (
                overall["best_model_id"].iloc[0]
                if overall_best_consistent
                else "metric_dependent"
            ),
            "evidence": (
                "Same best overall model across 25-year and 30-year initial windows."
                if overall_best_consistent
                else "Best overall model changes across initial-window settings."
            ),
            "claim_boundary": (
                "Use the best overall baseline as the primary overall benchmark."
            ),
        },
        {
            "recommendation_area": "high_volatility_forecasting",
            "selected_model_id": (
                high_volatility["best_fuzzy_model_id"].iloc[0]
                if high_volatility_fuzzy_consistent
                else "setting_dependent_fuzzy_model"
            ),
            "evidence": (
                "Best fuzzy high-volatility model is consistent across initial-window settings."
                if high_volatility_fuzzy_consistent
                else "Best fuzzy high-volatility model changes across initial-window settings."
            ),
            "claim_boundary": high_volatility_claim,
        },
    ]
    return pd.DataFrame(rows)


def validate_outputs(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    expected_rows = 0
    for training_window_size in INITIAL_TRAINING_OBSERVATIONS:
        split_count = int(
            predictions.loc[
                predictions["initial_training_observations"].eq(training_window_size),
                "validation_split_count",
            ].iloc[0]
        )
        model_count = predictions.loc[
            predictions["initial_training_observations"].eq(training_window_size),
            "model_id",
        ].nunique()
        expected_rows += split_count * model_count

    if len(predictions) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} sensitivity prediction rows, found {len(predictions)}."
        )

    per_split_counts = predictions.groupby(
        ["initial_training_observations", "split_id"]
    )["model_id"].nunique()
    expected_model_counts = predictions.groupby(
        "initial_training_observations"
    )["model_id"].nunique()
    for training_window_size, model_count in expected_model_counts.items():
        counts = per_split_counts.loc[training_window_size]
        if not counts.eq(model_count).all():
            raise RuntimeError(
                "Every sensitivity split must contain one prediction for each model."
            )

    if predictions["forecast_yield"].isna().any():
        raise RuntimeError("Sensitivity predictions contain missing forecasts.")

    required_views = {"overall", "high_volatility", "normal_volatility"}
    if set(metrics["evaluation_view"]) != required_views:
        raise RuntimeError("Sensitivity metrics are missing evaluation views.")

    if summary.empty or recommendations.empty:
        raise RuntimeError("Sensitivity summary and model-selection outputs are required.")


def write_results_readme(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    section_heading = "## Sensitivity And Robustness Analysis"
    content = RESULTS_README_PATH.read_text(encoding="utf-8")
    if section_heading in content:
        content = content.split(section_heading)[0].rstrip() + "\n\n"
    else:
        content = content.rstrip() + "\n\n"

    overall_selection = recommendations.loc[
        recommendations["recommendation_area"].eq("overall_forecasting")
    ].iloc[0]
    high_volatility_selection = recommendations.loc[
        recommendations["recommendation_area"].eq("high_volatility_forecasting")
    ].iloc[0]

    section = f"""{section_heading}

Generated by:

```powershell
.\\.venv\\Scripts\\python.exe -m src.sensitivity
```

Tables:

- `results/tables/sensitivity_predictions.csv`: rolling forecasts for 25-year and 30-year initial training windows.
- `results/tables/sensitivity_metrics.csv`: overall, high-volatility, and normal-volatility metrics for each sensitivity setting.
- `results/tables/sensitivity_summary.csv`: best baseline, best fuzzy model, and baseline comparison flags.
- `results/tables/recommended_model_selection.csv`: concise model-selection guidance for reporting.

Key robustness values:

- Initial training windows checked: {", ".join(str(value) for value in INITIAL_TRAINING_OBSERVATIONS)} years.
- Sensitivity prediction rows: {len(predictions)}.
- Validation designs summarized: {int(summary["initial_training_observations"].nunique())}.
- Recommended overall model: {overall_selection["selected_model_id"]}.
- Recommended high-volatility fuzzy model: {high_volatility_selection["selected_model_id"]}.

Claim boundary:

{high_volatility_selection["claim_boundary"]}
"""
    RESULTS_README_PATH.write_text(content + section, encoding="utf-8")


def run() -> None:
    ensure_directories()
    predictions = build_sensitivity_predictions()
    metrics = build_sensitivity_metrics(predictions)
    summary = build_sensitivity_summary(metrics)
    recommendations = build_recommended_model_selection(summary)
    validate_outputs(predictions, metrics, summary, recommendations)

    predictions.to_csv(SENSITIVITY_PREDICTIONS_PATH, index=False)
    metrics.to_csv(SENSITIVITY_METRICS_PATH, index=False)
    summary.to_csv(SENSITIVITY_SUMMARY_PATH, index=False)
    recommendations.to_csv(RECOMMENDED_MODEL_SELECTION_PATH, index=False)
    write_results_readme(predictions, summary, recommendations)

    print("Sensitivity analysis complete.")
    print(f"Sensitivity predictions: {SENSITIVITY_PREDICTIONS_PATH}")
    print(f"Sensitivity metrics: {SENSITIVITY_METRICS_PATH}")
    print(f"Sensitivity summary: {SENSITIVITY_SUMMARY_PATH}")
    print(f"Recommended model selection: {RECOMMENDED_MODEL_SELECTION_PATH}")


if __name__ == "__main__":
    run()
