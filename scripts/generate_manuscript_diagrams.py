"""Generate manuscript method diagrams for the Ghana cocoa yield study."""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import fill

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams["svg.fonttype"] = "none"

FIGURE_DIR = PROJECT_ROOT / "paper" / "figures"

COLOR_TEXT = "#1f2933"
COLOR_MUTED = "#52606d"
COLOR_BLUE = "#dbeafe"
COLOR_GREEN = "#dcfce7"
COLOR_AMBER = "#fef3c7"
COLOR_ROSE = "#ffe4e6"
COLOR_PURPLE = "#ede9fe"
COLOR_BORDER = "#334e68"


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    facecolor: str = COLOR_BLUE,
    fontsize: int = 9,
    title_size: int = 10.5,
    wrap_width: int = 33,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        linewidth=1.25,
        edgecolor=COLOR_BORDER,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.035,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=COLOR_TEXT,
    )
    if body:
        ax.text(
            x + w / 2,
            y + h - 0.085,
            fill(body, width=wrap_width),
            ha="center",
            va="top",
            fontsize=fontsize,
            color=COLOR_TEXT,
            linespacing=1.15,
        )
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    rad: float = 0.0,
    color: str = COLOR_BORDER,
    lw: float = 1.45,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def forecasting_framework() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.965,
        "Fig. 1. Forecasting and Evaluation Framework",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=COLOR_TEXT,
    )

    _box(
        ax,
        0.05,
        0.72,
        0.22,
        0.16,
        "Official Yield Data",
        "FAOSTAT via OWID Grapher. Ghana cocoa bean yield, 1961-2024, 64 annual observations.",
        COLOR_BLUE,
        wrap_width=34,
    )
    _box(
        ax,
        0.39,
        0.72,
        0.22,
        0.16,
        "Preprocessing",
        "Filter Ghana, sort by year, verify no missing years or missing yield values, derive annual changes.",
        COLOR_GREEN,
        wrap_width=34,
    )
    _box(
        ax,
        0.73,
        0.72,
        0.22,
        0.16,
        "Rolling-Origin Splits",
        "Expanding training window. Main setting: 30 initial observations and 34 one-step forecasts, 1991-2024.",
        COLOR_AMBER,
        wrap_width=38,
    )
    _arrow(ax, (0.27, 0.80), (0.39, 0.80))
    _arrow(ax, (0.61, 0.80), (0.73, 0.80))

    _box(
        ax,
        0.05,
        0.43,
        0.24,
        0.17,
        "Baseline References",
        "Naive, expanding-window mean, drift, and ARIMA selected by AIC inside each training window.",
        "#e0f2fe",
        wrap_width=34,
    )
    _box(
        ax,
        0.38,
        0.43,
        0.24,
        0.17,
        "Fuzzy Level Models",
        "First-order Chen FTS with equal-length and adaptive quantile intervals; order-2 adaptive FTS.",
        COLOR_PURPLE,
        wrap_width=34,
    )
    _box(
        ax,
        0.71,
        0.43,
        0.24,
        0.17,
        "Fuzzy Correction Models",
        "Chen FTS predicts absolute or percentage yield change; forecast reconstructed around the Naive anchor.",
        COLOR_ROSE,
        wrap_width=34,
    )
    _arrow(ax, (0.84, 0.72), (0.17, 0.60), rad=0.0)
    _arrow(ax, (0.84, 0.72), (0.50, 0.60), rad=0.0)
    _arrow(ax, (0.84, 0.72), (0.83, 0.60), rad=0.0)

    _box(
        ax,
        0.16,
        0.16,
        0.29,
        0.16,
        "Evaluation Outputs",
        "Rolling predictions, MAE, RMSE, sMAPE, MASE, high-volatility subgroup metrics.",
        "#f8fafc",
        wrap_width=36,
    )
    _box(
        ax,
        0.56,
        0.16,
        0.29,
        0.16,
        "Robustness and Comparison",
        "25-year and 30-year initial windows, expanded correction weights, paired absolute-error tests.",
        "#f8fafc",
        wrap_width=36,
    )
    for x in (0.17, 0.50, 0.83):
        _arrow(ax, (x, 0.43), (0.31, 0.32), rad=0.05 if x < 0.31 else -0.05)
    _arrow(ax, (0.45, 0.24), (0.56, 0.24), rad=0.0)

    ax.text(
        0.5,
        0.06,
        "All interval boundaries, ARIMA orders, fuzzy rules, transformations, and correction weights are estimated from the training window only.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLOR_MUTED,
    )
    _save(fig, "fig_1")


def correction_mechanism() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.965,
        "Fig. 3. Adaptive Fuzzy Correction Mechanism",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=COLOR_TEXT,
    )

    _box(
        ax,
        0.05,
        0.67,
        0.24,
        0.16,
        "Training Yields",
        "Historical yield values available before the forecast year.",
        COLOR_BLUE,
        wrap_width=30,
    )
    _box(
        ax,
        0.38,
        0.67,
        0.24,
        0.16,
        "Transformed Target",
        "Absolute change: y_t - y_(t-1). Percentage change also evaluated.",
        COLOR_GREEN,
        wrap_width=30,
    )
    _box(
        ax,
        0.71,
        0.67,
        0.24,
        0.16,
        "Chen FTS Rules",
        "Adaptive quantile intervals, fuzzification, FLRG grouping, weighted defuzzification.",
        COLOR_PURPLE,
        wrap_width=34,
    )
    _arrow(ax, (0.29, 0.75), (0.38, 0.75))
    _arrow(ax, (0.62, 0.75), (0.71, 0.75))

    _box(
        ax,
        0.08,
        0.35,
        0.25,
        0.16,
        "Naive Anchor",
        "Previous observed yield y_(t-1). This preserves local persistence.",
        COLOR_AMBER,
        wrap_width=30,
    )
    _box(
        ax,
        0.41,
        0.35,
        0.18,
        0.16,
        "Correction Weight",
        "lambda controls the size of the fuzzy adjustment. Best overall value: 0.75.",
        "#fef9c3",
        wrap_width=28,
    )
    _box(
        ax,
        0.67,
        0.35,
        0.25,
        0.16,
        "Fuzzy Change Forecast",
        "Predicted change from the transformed-target fuzzy rules.",
        COLOR_ROSE,
        wrap_width=30,
    )
    _arrow(ax, (0.83, 0.67), (0.80, 0.51), rad=0.0)

    _box(
        ax,
        0.28,
        0.10,
        0.44,
        0.14,
        "Reconstructed Yield Forecast",
        "Absolute-change model: forecast_yield = previous_yield + lambda * predicted_change",
        "#f8fafc",
        fontsize=10,
        title_size=12,
        wrap_width=52,
    )
    _arrow(ax, (0.205, 0.35), (0.39, 0.24), rad=-0.08)
    _arrow(ax, (0.50, 0.35), (0.50, 0.24), rad=0.0)
    _arrow(ax, (0.795, 0.35), (0.61, 0.24), rad=0.08)

    ax.text(
        0.5,
        0.045,
        "The final model improves accuracy by adding a damped fuzzy correction to the previous observed yield instead of replacing the persistence benchmark.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLOR_MUTED,
    )
    _save(fig, "fig_3")


def rolling_origin_validation() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.965,
        "Fig. 2. Rolling-Origin One-Step Validation",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=COLOR_TEXT,
    )

    y_positions = [0.73, 0.56, 0.39, 0.22]
    rows = [
        ("Split 1", "Train 1961-1990", "Forecast 1991"),
        ("Split 2", "Train 1961-1991", "Forecast 1992"),
        ("...", "Expanding training window", "..."),
        ("Split 34", "Train 1961-2023", "Forecast 2024"),
    ]
    train_start = 0.20
    train_end_values = [0.57, 0.60, 0.74, 0.85]
    forecast_x_values = [0.62, 0.65, 0.79, 0.90]

    for y, row, train_end, forecast_x in zip(y_positions, rows, train_end_values, forecast_x_values):
        label, train_label, forecast_label = row
        ax.text(0.06, y, label, ha="left", va="center", fontsize=11.5, fontweight="bold", color=COLOR_TEXT)
        ax.plot([train_start, train_end], [y, y], color="#2563eb", lw=11, solid_capstyle="round")
        ax.text(
            (train_start + train_end) / 2,
            y + 0.045,
            train_label,
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=COLOR_TEXT,
        )
        ax.scatter([forecast_x], [y], s=230, color="#fb7185", edgecolor=COLOR_BORDER, zorder=3)
        ax.text(
            forecast_x,
            y - 0.06,
            forecast_label,
            ha="center",
            va="top",
            fontsize=10.5,
            color=COLOR_TEXT,
        )
        ax.plot(
            [train_end + 0.025, forecast_x - 0.035],
            [y, y],
            color=COLOR_BORDER,
            lw=1.2,
            marker=">",
            markevery=[1],
            markersize=8,
        )

    ax.plot([train_start, 0.90], [0.10, 0.10], color=COLOR_BORDER, lw=1.0)
    for x, year in [(train_start, "1961"), (0.57, "1990"), (0.65, "1992"), (0.90, "2024")]:
        ax.plot([x, x], [0.085, 0.115], color=COLOR_BORDER, lw=1.0)
        ax.text(x, 0.06, year, ha="center", va="top", fontsize=10, color=COLOR_MUTED)

    ax.text(
        0.5,
        0.015,
        "Each forecast is generated after fitting the model only on observations available before the forecast year.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=COLOR_MUTED,
    )
    _save(fig, "fig_2")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
        }
    )
    forecasting_framework()
    rolling_origin_validation()
    correction_mechanism()


if __name__ == "__main__":
    main()
