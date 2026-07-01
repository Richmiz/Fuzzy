"""Publication-ready figures for Ghana cocoa yield forecasting."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_README_PATH = PROJECT_ROOT / "results" / "README.md"

ROLLING_PREDICTIONS_PATH = TABLES_DIR / "rolling_predictions.csv"
OVERALL_METRICS_PATH = TABLES_DIR / "overall_metrics.csv"
VOLATILITY_METRICS_PATH = TABLES_DIR / "volatility_metrics.csv"
ROBUSTNESS_CORRECTION_PREDICTIONS_PATH = (
    TABLES_DIR / "robustness_correction_predictions.csv"
)
CORRECTION_WEIGHT_SENSITIVITY_PATH = (
    TABLES_DIR / "correction_weight_sensitivity.csv"
)
ROBUSTNESS_SUMMARY_PATH = TABLES_DIR / "robustness_summary.csv"
PAIRED_ERROR_TESTS_PATH = TABLES_DIR / "paired_error_tests.csv"
PUBLICATION_FIGURE_MANIFEST_PATH = TABLES_DIR / "publication_figure_manifest.csv"

OVERALL_CORRECTION_MODEL_ID = "chen_change_adaptive_k9_w075"
HIGH_VOLATILITY_FUZZY_MODEL_ID = "chen_adaptive_k9"
SELECTED_MODEL_IDS = [
    "baseline_naive",
    "baseline_arima_aic",
    OVERALL_CORRECTION_MODEL_ID,
    HIGH_VOLATILITY_FUZZY_MODEL_ID,
]
TRAJECTORY_MODEL_IDS = [
    "baseline_naive",
    OVERALL_CORRECTION_MODEL_ID,
]
MODEL_LABELS = {
    "actual_yield": "Observed yield",
    "baseline_naive": "Naive",
    "baseline_arima_aic": "ARIMA-AIC",
    OVERALL_CORRECTION_MODEL_ID: "Fuzzy correction",
    HIGH_VOLATILITY_FUZZY_MODEL_ID: "Adaptive Chen FTS",
}
MODEL_COLORS = {
    "actual_yield": "#222222",
    "baseline_naive": "#6b6b6b",
    "baseline_arima_aic": "#c44e52",
    OVERALL_CORRECTION_MODEL_ID: "#2a7f62",
    HIGH_VOLATILITY_FUZZY_MODEL_ID: "#4c72b0",
}
MODEL_MARKERS = {
    "baseline_naive": "o",
    "baseline_arima_aic": "s",
    OVERALL_CORRECTION_MODEL_ID: "^",
    HIGH_VOLATILITY_FUZZY_MODEL_ID: "D",
}
TRAJECTORY_LINE_STYLES = {
    "baseline_naive": "--",
    OVERALL_CORRECTION_MODEL_ID: "-",
}


def ensure_directories() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def apply_axis_style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)


def save_figure(fig: plt.Figure, figure_stem: str) -> tuple[Path, Path]:
    png_path = FIGURES_DIR / f"{figure_stem}.png"
    svg_path = FIGURES_DIR / f"{figure_stem}.svg"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def load_required_tables() -> dict[str, pd.DataFrame]:
    required_paths = {
        "rolling_predictions": ROLLING_PREDICTIONS_PATH,
        "overall_metrics": OVERALL_METRICS_PATH,
        "volatility_metrics": VOLATILITY_METRICS_PATH,
        "robustness_predictions": ROBUSTNESS_CORRECTION_PREDICTIONS_PATH,
        "weight_sensitivity": CORRECTION_WEIGHT_SENSITIVITY_PATH,
        "robustness_summary": ROBUSTNESS_SUMMARY_PATH,
        "paired_tests": PAIRED_ERROR_TESTS_PATH,
    }
    missing = [path for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run modeling, sensitivity, and robustness before publication figures. "
            f"Missing: {missing}"
        )
    return {
        name: pd.read_csv(path)
        for name, path in required_paths.items()
    }


def canonical_selected_predictions(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rolling = tables["rolling_predictions"]
    robustness = tables["robustness_predictions"]
    baseline_and_level = rolling.loc[
        rolling["model_id"].isin(
            ["baseline_naive", "baseline_arima_aic", HIGH_VOLATILITY_FUZZY_MODEL_ID]
        )
    ].copy()
    correction = robustness.loc[
        robustness["initial_training_observations"].eq(30)
        & robustness["model_id"].eq(OVERALL_CORRECTION_MODEL_ID)
    ].copy()
    selected = pd.concat([baseline_and_level, correction], ignore_index=True)
    selected["model_label"] = selected["model_id"].map(MODEL_LABELS)
    return selected.sort_values(["forecast_year", "model_id"]).reset_index(drop=True)


def selected_metric_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    overall_metrics = tables["overall_metrics"]
    weight_sensitivity = tables["weight_sensitivity"]
    rows = []
    for model_id in ["baseline_naive", "baseline_arima_aic", HIGH_VOLATILITY_FUZZY_MODEL_ID]:
        row = overall_metrics.loc[overall_metrics["model_id"].eq(model_id)].iloc[0]
        rows.append(
            {
                "model_id": model_id,
                "model_label": MODEL_LABELS[model_id],
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "smape": float(row["smape"]),
                "mase": float(row["mase"]),
            }
        )
    correction = weight_sensitivity.loc[
        weight_sensitivity["initial_training_observations"].eq(30)
        & weight_sensitivity["evaluation_view"].eq("overall")
        & weight_sensitivity["model_id"].eq(OVERALL_CORRECTION_MODEL_ID)
    ].iloc[0]
    rows.append(
        {
            "model_id": OVERALL_CORRECTION_MODEL_ID,
            "model_label": MODEL_LABELS[OVERALL_CORRECTION_MODEL_ID],
            "mae": float(correction["mae"]),
            "rmse": float(correction["rmse"]),
            "smape": float(correction["smape"]),
            "mase": float(correction["mase"]),
        }
    )
    order = {model_id: index for index, model_id in enumerate(SELECTED_MODEL_IDS)}
    return pd.DataFrame(rows).sort_values(
        by="model_id",
        key=lambda series: series.map(order),
    )


def save_forecast_trajectory(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    selected = canonical_selected_predictions(tables)
    actual = (
        selected[["forecast_year", "actual_yield", "is_evaluation_high_volatility"]]
        .drop_duplicates("forecast_year")
        .sort_values("forecast_year")
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(
        actual["forecast_year"],
        actual["actual_yield"],
        color=MODEL_COLORS["actual_yield"],
        linewidth=2.6,
        marker="o",
        markersize=3.2,
        label=MODEL_LABELS["actual_yield"],
        zorder=5,
    )
    for model_id in TRAJECTORY_MODEL_IDS:
        model_data = selected.loc[selected["model_id"].eq(model_id)]
        ax.plot(
            model_data["forecast_year"],
            model_data["forecast_yield"],
            color=MODEL_COLORS[model_id],
            linewidth=2.0 if model_id == OVERALL_CORRECTION_MODEL_ID else 1.6,
            linestyle=TRAJECTORY_LINE_STYLES[model_id],
            marker=MODEL_MARKERS[model_id],
            markersize=3.0,
            alpha=0.95 if model_id == OVERALL_CORRECTION_MODEL_ID else 0.75,
            label=MODEL_LABELS[model_id],
        )
    ax.set_title("Observed Yield and Selected One-Step Forecasts", fontsize=12, pad=10)
    ax.set_xlabel("Forecast year")
    ax.set_ylabel("Yield (tonnes per hectare)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    apply_axis_style(ax)
    png_path, svg_path = save_figure(fig, "publication_forecast_trajectory")
    return {
        "figure_id": "figure_forecast_trajectory",
        "file_png": png_path.name,
        "file_svg": svg_path.name,
        "manuscript_use": "Main results",
        "description": (
            "Observed Ghana cocoa yield and focused one-step forecasts for the Naive "
            "baseline and the selected fuzzy correction model."
        ),
        "selected_models": ";".join(TRAJECTORY_MODEL_IDS),
    }


def save_metric_comparison(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    metrics = selected_metric_table(tables)
    metric_panels = [
        ("mae", "MAE", "Yield units"),
        ("rmse", "RMSE", "Yield units"),
        ("smape", "sMAPE", "Percent-style"),
        ("mase", "MASE", "Scaled error"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))
    for ax, (metric, title, ylabel) in zip(axes.ravel(), metric_panels):
        values = metrics[metric].astype(float)
        colors = [MODEL_COLORS[model_id] for model_id in metrics["model_id"]]
        bars = ax.bar(metrics["model_label"], values, color=colors, width=0.72)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=25)
        apply_axis_style(ax)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}" if value >= 1 else f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    fig.suptitle("Overall Forecast Accuracy for Selected Models", fontsize=13, y=1.02)
    png_path, svg_path = save_figure(fig, "publication_overall_metric_comparison")
    return {
        "figure_id": "figure_overall_metric_comparison",
        "file_png": png_path.name,
        "file_svg": svg_path.name,
        "manuscript_use": "Main results",
        "description": "Overall MAE, RMSE, sMAPE, and MASE for selected benchmark and fuzzy models.",
        "selected_models": ";".join(metrics["model_id"].tolist()),
    }


def save_high_volatility_errors(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    selected = canonical_selected_predictions(tables)
    high = selected.loc[selected["is_evaluation_high_volatility"]].copy()
    years = sorted(high["forecast_year"].unique())
    x = np.arange(len(years))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    for index, model_id in enumerate(SELECTED_MODEL_IDS):
        model_data = (
            high.loc[high["model_id"].eq(model_id)]
            .set_index("forecast_year")
            .reindex(years)
        )
        offset = (index - (len(SELECTED_MODEL_IDS) - 1) / 2) * width
        ax.bar(
            x + offset,
            model_data["absolute_error"],
            width=width,
            color=MODEL_COLORS[model_id],
            label=MODEL_LABELS[model_id],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(year) for year in years], rotation=30, ha="right")
    ax.set_title("Absolute Forecast Error in Evaluation High-Volatility Years", fontsize=12, pad=10)
    ax.set_xlabel("Forecast year")
    ax.set_ylabel("Absolute error (tonnes per hectare)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    apply_axis_style(ax)
    png_path, svg_path = save_figure(fig, "publication_high_volatility_errors")
    return {
        "figure_id": "figure_high_volatility_errors",
        "file_png": png_path.name,
        "file_svg": svg_path.name,
        "manuscript_use": "High-volatility analysis",
        "description": "Absolute forecast errors for selected models in evaluation high-volatility years.",
        "selected_models": ";".join(SELECTED_MODEL_IDS),
    }


def save_correction_weight_sensitivity(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    metrics = tables["weight_sensitivity"]
    subset = metrics.loc[metrics["evaluation_view"].isin(["overall", "high_volatility"])]
    best_by_weight = (
        subset.sort_values(["mae", "rmse", "smape"])
        .groupby(
            ["initial_training_observations", "evaluation_view", "correction_weight"],
            as_index=False,
        )
        .first()
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True)
    view_titles = {
        "overall": "Overall evaluation",
        "high_volatility": "High-volatility years",
    }
    for ax, evaluation_view in zip(axes, ["overall", "high_volatility"]):
        view_data = best_by_weight.loc[
            best_by_weight["evaluation_view"].eq(evaluation_view)
        ]
        for training_window_size, group in view_data.groupby(
            "initial_training_observations"
        ):
            group = group.sort_values("correction_weight")
            ax.plot(
                group["correction_weight"],
                group["mae"],
                marker="o",
                linewidth=2,
                label=f"{training_window_size}-year initial window",
            )
            best = group.sort_values("mae").iloc[0]
            ax.scatter(
                [best["correction_weight"]],
                [best["mae"]],
                s=55,
                color="#222222",
                zorder=4,
            )
        ax.set_title(view_titles[evaluation_view], fontsize=11)
        ax.set_xlabel("Correction weight")
        ax.set_ylabel("Best correction MAE")
        ax.set_xticks([0.25, 0.5, 0.75, 1.0])
        ax.legend(frameon=False, fontsize=8)
        apply_axis_style(ax)
    fig.suptitle("Correction Weight Sensitivity", fontsize=13, y=1.02)
    png_path, svg_path = save_figure(fig, "publication_correction_weight_sensitivity")
    return {
        "figure_id": "figure_correction_weight_sensitivity",
        "file_png": png_path.name,
        "file_svg": svg_path.name,
        "manuscript_use": "Robustness",
        "description": "Best correction-model MAE by correction weight across validation windows.",
        "selected_models": "expanded_correction_models",
    }


def save_window_robustness(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    summary = tables["robustness_summary"]
    views = ["overall", "high_volatility"]
    labels = ["Naive", "ARIMA-AIC", "Existing fuzzy", "Best correction"]
    value_columns = [
        "naive_mae",
        "arima_aic_mae",
        "best_existing_fuzzy_mae",
        "best_correction_mae",
    ]
    colors = ["#6b6b6b", "#c44e52", "#4c72b0", "#2a7f62"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharey=False)
    for ax, evaluation_view in zip(axes, views):
        view_data = summary.loc[summary["evaluation_view"].eq(evaluation_view)].sort_values(
            "initial_training_observations"
        )
        x = np.arange(len(view_data))
        width = 0.18
        for index, (label, column, color) in enumerate(zip(labels, value_columns, colors)):
            offset = (index - (len(labels) - 1) / 2) * width
            ax.bar(
                x + offset,
                view_data[column].astype(float),
                width=width,
                label=label,
                color=color,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{int(value)} years" for value in view_data["initial_training_observations"]]
        )
        ax.set_title(
            "Overall" if evaluation_view == "overall" else "High-volatility",
            fontsize=11,
        )
        ax.set_xlabel("Initial training window")
        ax.set_ylabel("MAE")
        apply_axis_style(ax)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Robustness Across Initial Training Windows", fontsize=13, y=1.02)
    png_path, svg_path = save_figure(fig, "publication_window_robustness")
    return {
        "figure_id": "figure_window_robustness",
        "file_png": png_path.name,
        "file_svg": svg_path.name,
        "manuscript_use": "Robustness",
        "description": "MAE comparison for baselines, existing fuzzy models, and best correction models across initial windows.",
        "selected_models": "best_by_validation_window",
    }


def save_paired_error_deltas(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    tests = tables["paired_tests"].copy()
    tests["display_label"] = tests["comparison_label"].str.replace("_", " ").str.title()
    tests = tests.sort_values("mae_difference_a_minus_b")

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    y = np.arange(len(tests))
    colors = tests["mae_difference_a_minus_b"].map(
        lambda value: "#2a7f62" if value < 0 else "#c44e52"
    )
    ax.barh(y, tests["mae_difference_a_minus_b"], color=colors)
    ax.axvline(0, color="#222222", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(tests["display_label"], fontsize=7)
    ax.set_xlabel("MAE difference for model A minus model B")
    ax.set_title("Paired Absolute-Error Comparisons", fontsize=12, pad=10)
    for ypos, (_, row) in zip(y, tests.iterrows()):
        ax.text(
            row["mae_difference_a_minus_b"],
            ypos,
            f"  p={row['wilcoxon_p_value']:.3f}",
            va="center",
            ha="left" if row["mae_difference_a_minus_b"] >= 0 else "right",
            fontsize=7,
        )
    apply_axis_style(ax)
    png_path, svg_path = save_figure(fig, "publication_paired_error_deltas")
    return {
        "figure_id": "figure_paired_error_deltas",
        "file_png": png_path.name,
        "file_svg": svg_path.name,
        "manuscript_use": "Supplementary robustness",
        "description": "Paired absolute-error MAE differences with descriptive Wilcoxon p-values.",
        "selected_models": "selected_pairwise_comparisons",
    }


def build_publication_figures() -> pd.DataFrame:
    tables = load_required_tables()
    figure_rows = [
        save_forecast_trajectory(tables),
        save_metric_comparison(tables),
        save_high_volatility_errors(tables),
        save_correction_weight_sensitivity(tables),
        save_window_robustness(tables),
        save_paired_error_deltas(tables),
    ]
    manifest = pd.DataFrame(figure_rows)
    manifest.to_csv(PUBLICATION_FIGURE_MANIFEST_PATH, index=False)
    return manifest


def validate_outputs(manifest: pd.DataFrame) -> None:
    if len(manifest) != 6:
        raise RuntimeError(f"Expected 6 publication figures, found {len(manifest)}.")
    for _, row in manifest.iterrows():
        for column in ["file_png", "file_svg"]:
            path = FIGURES_DIR / row[column]
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty figure file: {path}")
    if manifest["figure_id"].duplicated().any():
        raise RuntimeError("Publication figure manifest contains duplicate IDs.")


def write_results_readme(manifest: pd.DataFrame) -> None:
    section_heading = "## Publication Figures"
    content = RESULTS_README_PATH.read_text(encoding="utf-8")
    if section_heading in content:
        content = content.split(section_heading)[0].rstrip() + "\n\n"
    else:
        content = content.rstrip() + "\n\n"

    figure_lines = "\n".join(
        f"- `results/figures/{row.file_png}` and `results/figures/{row.file_svg}`: {row.description}"
        for row in manifest.itertuples(index=False)
    )
    section = f"""{section_heading}

Generated by:

```powershell
.\\.venv\\Scripts\\python.exe -m src.publication_figures
```

Table:

- `results/tables/publication_figure_manifest.csv`: figure IDs, file names, manuscript use, and selected model notes.

Figures:

{figure_lines}
"""
    RESULTS_README_PATH.write_text(content + section, encoding="utf-8")


def run() -> None:
    ensure_directories()
    manifest = build_publication_figures()
    validate_outputs(manifest)
    write_results_readme(manifest)

    print("Publication figures complete.")
    print(f"Figure manifest: {PUBLICATION_FIGURE_MANIFEST_PATH}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    run()
