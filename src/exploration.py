"""Data audit and exploratory analysis for Ghana cocoa yield."""

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
import pandas as pd


PROCESSED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "ghana_cocoa_yield_1961_2024.csv"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_README_PATH = PROJECT_ROOT / "results" / "README.md"

DATA_AUDIT_PATH = TABLES_DIR / "data_audit.csv"
EXPLORATORY_HIGH_VOLATILITY_YEARS_PATH = (
    TABLES_DIR / "exploratory_high_volatility_years.csv"
)
YIELD_TIMESERIES_PATH = FIGURES_DIR / "yield_timeseries.png"
YIELD_PCT_CHANGE_PATH = FIGURES_DIR / "yield_pct_change.png"
EXPLORATORY_HIGH_VOLATILITY_FIGURE_PATH = (
    FIGURES_DIR / "exploratory_high_volatility_years.png"
)
CHANGE_DISTRIBUTION_PATH = FIGURES_DIR / "yield_change_distribution.png"

YIELD_COLUMN = "yield_tonnes_per_hectare"
VOLATILITY_QUANTILE = 0.75


def ensure_directories() -> None:
    for path in [TABLES_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_processed_data() -> pd.DataFrame:
    if not PROCESSED_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {PROCESSED_CSV_PATH}. "
            "Run `python -m src.data` first."
        )

    df = pd.read_csv(PROCESSED_CSV_PATH)
    required = {
        "year",
        YIELD_COLUMN,
        "yield_lag1",
        "absolute_change",
        "percentage_change",
        "absolute_percentage_change",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Processed dataset is missing columns: {sorted(missing)}")

    df = df.sort_values("year").reset_index(drop=True)
    return df


def missing_years_for(df: pd.DataFrame) -> list[int]:
    first_year = int(df["year"].min())
    last_year = int(df["year"].max())
    observed = set(df["year"].astype(int).tolist())
    return [year for year in range(first_year, last_year + 1) if year not in observed]


def exploratory_volatility_threshold(df: pd.DataFrame) -> float:
    changes = df["absolute_percentage_change"].dropna()
    if changes.empty:
        raise ValueError("Cannot compute volatility threshold without percentage changes.")
    return float(changes.quantile(VOLATILITY_QUANTILE))


def add_exploratory_volatility_label(df: pd.DataFrame) -> pd.DataFrame:
    threshold = exploratory_volatility_threshold(df)
    labeled = df.copy()
    labeled["exploratory_high_volatility"] = (
        labeled["absolute_percentage_change"].ge(threshold).fillna(False)
    )
    labeled["exploratory_volatility_threshold"] = threshold
    return labeled


def build_data_audit(df: pd.DataFrame) -> pd.DataFrame:
    missing_years = missing_years_for(df)
    threshold = exploratory_volatility_threshold(df)
    changes = df["percentage_change"].dropna()
    abs_changes = df["absolute_percentage_change"].dropna()
    high_count = int(df["absolute_percentage_change"].ge(threshold).fillna(False).sum())

    audit_rows = [
        ("country", "Ghana"),
        ("commodity", "cocoa beans"),
        ("indicator", "yield"),
        ("unit", "tonnes per hectare"),
        ("frequency", "annual"),
        ("observation_count", len(df)),
        ("first_year", int(df["year"].min())),
        ("last_year", int(df["year"].max())),
        ("missing_year_count", len(missing_years)),
        ("missing_years", ", ".join(str(year) for year in missing_years) or "none"),
        ("missing_yield_values", int(df[YIELD_COLUMN].isna().sum())),
        ("duplicate_year_count", int(df.duplicated(subset=["year"]).sum())),
        ("yield_min", df[YIELD_COLUMN].min()),
        ("yield_min_year", int(df.loc[df[YIELD_COLUMN].idxmin(), "year"])),
        ("yield_max", df[YIELD_COLUMN].max()),
        ("yield_max_year", int(df.loc[df[YIELD_COLUMN].idxmax(), "year"])),
        ("yield_mean", df[YIELD_COLUMN].mean()),
        ("yield_median", df[YIELD_COLUMN].median()),
        ("yield_std", df[YIELD_COLUMN].std(ddof=1)),
        ("yield_first_value", df.iloc[0][YIELD_COLUMN]),
        ("yield_last_value", df.iloc[-1][YIELD_COLUMN]),
        ("total_change", df.iloc[-1][YIELD_COLUMN] - df.iloc[0][YIELD_COLUMN]),
        (
            "total_percentage_change",
            100
            * (df.iloc[-1][YIELD_COLUMN] - df.iloc[0][YIELD_COLUMN])
            / df.iloc[0][YIELD_COLUMN],
        ),
        ("annual_change_mean", changes.mean()),
        ("annual_change_median", changes.median()),
        ("annual_change_std", changes.std(ddof=1)),
        ("absolute_annual_change_mean", abs_changes.mean()),
        ("absolute_annual_change_median", abs_changes.median()),
        ("absolute_annual_change_75th_percentile", threshold),
        ("exploratory_high_volatility_year_count", high_count),
        (
            "exploratory_high_volatility_rule",
            "absolute percentage change >= full-series 75th percentile",
        ),
        (
            "validation_note",
            "Final modeling should recompute volatility labels on the evaluation period.",
        ),
    ]
    return pd.DataFrame(audit_rows, columns=["metric", "value"])


def build_high_volatility_years(df: pd.DataFrame) -> pd.DataFrame:
    labeled = add_exploratory_volatility_label(df)
    columns = [
        "year",
        YIELD_COLUMN,
        "yield_lag1",
        "absolute_change",
        "percentage_change",
        "absolute_percentage_change",
        "exploratory_volatility_threshold",
    ]
    high = labeled.loc[labeled["exploratory_high_volatility"], columns].copy()
    return high.sort_values("absolute_percentage_change", ascending=False).reset_index(
        drop=True
    )


def apply_axis_style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def save_yield_timeseries(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    ax.plot(df["year"], df[YIELD_COLUMN], color="#1f6f8b", linewidth=2.2)
    ax.scatter(df["year"], df[YIELD_COLUMN], color="#1f6f8b", s=14, zorder=3)
    ax.set_title("Ghana Cocoa Bean Yield, 1961-2024", fontsize=13, pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (tonnes per hectare)")
    apply_axis_style(ax)
    fig.tight_layout()
    fig.savefig(YIELD_TIMESERIES_PATH, bbox_inches="tight")
    plt.close(fig)


def save_yield_pct_change(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["percentage_change"]).copy()
    threshold = exploratory_volatility_threshold(df)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    colors = plot_df["percentage_change"].map(lambda value: "#2a7f62" if value >= 0 else "#b24b48")
    ax.bar(plot_df["year"], plot_df["percentage_change"], color=colors, width=0.85)
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.axhline(threshold, color="#555555", linestyle="--", linewidth=1.0)
    ax.axhline(-threshold, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_title("Year-to-Year Change in Ghana Cocoa Bean Yield", fontsize=13, pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Change (%)")
    apply_axis_style(ax)
    fig.tight_layout()
    fig.savefig(YIELD_PCT_CHANGE_PATH, bbox_inches="tight")
    plt.close(fig)


def save_high_volatility_figure(df: pd.DataFrame) -> None:
    labeled = add_exploratory_volatility_label(df)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    ax.plot(labeled["year"], labeled[YIELD_COLUMN], color="#828282", linewidth=1.8)
    ax.scatter(
        labeled.loc[~labeled["exploratory_high_volatility"], "year"],
        labeled.loc[~labeled["exploratory_high_volatility"], YIELD_COLUMN],
        color="#828282",
        s=16,
        label="Normal exploratory change",
    )
    ax.scatter(
        labeled.loc[labeled["exploratory_high_volatility"], "year"],
        labeled.loc[labeled["exploratory_high_volatility"], YIELD_COLUMN],
        color="#b24b48",
        s=36,
        label="High-volatility year",
        zorder=4,
    )
    ax.set_title("Exploratory High-Volatility Years in Ghana Cocoa Yield", fontsize=13, pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (tonnes per hectare)")
    ax.legend(frameon=False, fontsize=9, loc="best")
    apply_axis_style(ax)
    fig.tight_layout()
    fig.savefig(EXPLORATORY_HIGH_VOLATILITY_FIGURE_PATH, bbox_inches="tight")
    plt.close(fig)


def save_change_distribution(df: pd.DataFrame) -> None:
    changes = df["absolute_percentage_change"].dropna()
    threshold = exploratory_volatility_threshold(df)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.hist(changes, bins=12, color="#5f8fb6", edgecolor="white")
    ax.axvline(threshold, color="#b24b48", linestyle="--", linewidth=1.5)
    ax.set_title("Distribution of Absolute Annual Yield Changes", fontsize=13, pad=12)
    ax.set_xlabel("Absolute change (%)")
    ax.set_ylabel("Number of years")
    apply_axis_style(ax)
    fig.tight_layout()
    fig.savefig(CHANGE_DISTRIBUTION_PATH, bbox_inches="tight")
    plt.close(fig)


def write_results_readme(audit: pd.DataFrame, high_years: pd.DataFrame) -> None:
    get_value = lambda metric: audit.loc[audit["metric"] == metric, "value"].iloc[0]
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

Key audit values:

- Observation count: {get_value("observation_count")}.
- Year range: {get_value("first_year")}-{get_value("last_year")}.
- Missing year count: {get_value("missing_year_count")}.
- Missing yield values: {get_value("missing_yield_values")}.
- Exploratory high-volatility threshold: {float(get_value("absolute_annual_change_75th_percentile")):.6g}%.
- Exploratory high-volatility year count: {len(high_years)}.

Note:

The high-volatility label in these exploratory artifacts uses the full observed series. The modeling workflow should recompute high-volatility labels using only the evaluation period.
"""
    RESULTS_README_PATH.write_text(content, encoding="utf-8")


def run() -> None:
    ensure_directories()
    df = load_processed_data()

    audit = build_data_audit(df)
    high_years = build_high_volatility_years(df)

    audit.to_csv(DATA_AUDIT_PATH, index=False)
    high_years.to_csv(EXPLORATORY_HIGH_VOLATILITY_YEARS_PATH, index=False)

    save_yield_timeseries(df)
    save_yield_pct_change(df)
    save_high_volatility_figure(df)
    save_change_distribution(df)
    write_results_readme(audit, high_years)

    print("Data audit and exploratory analysis complete.")
    print(f"Data audit: {DATA_AUDIT_PATH}")
    print(
        "Exploratory high-volatility years: "
        f"{EXPLORATORY_HIGH_VOLATILITY_YEARS_PATH}"
    )
    print(f"Figures: {FIGURES_DIR}")
    print(f"Results README: {RESULTS_README_PATH}")


if __name__ == "__main__":
    run()
