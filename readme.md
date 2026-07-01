# Ghana Cocoa Yield Fuzzy Time-Series Forecasting

This repository contains a reproducible forecasting study of Ghana annual cocoa bean yield using public FAOSTAT data accessed through the Our World in Data Grapher service. The study compares classical baseline forecasts with Chen fuzzy time-series models, adaptive interval designs, higher-order fuzzy rules, and fuzzy correction models under rolling-origin one-step-ahead validation.

The strongest empirical result is an adaptive absolute-change fuzzy correction model with nine intervals and correction weight 0.75. In the canonical 30-year initial-window evaluation, it achieved MAE 0.029255 tonnes per hectare, compared with 0.031665 for the Naive baseline and 0.033814 for ARIMA-AIC. High-volatility years are reported separately because those years are more relevant for planning risk and produce the clearest fuzzy-model gains.

## Repository Structure

```text
data/
  raw/                         Archived OWID/FAOSTAT source files
  processed/                   Ghana-only processed yield series
docs/
  literature/                  Literature matrix, references, and gap notes
notebooks/
  ghana_cocoa_yield_forecasting.ipynb
paper/
  manuscript.md                Submission-oriented manuscript draft
  figures/                     Numbered paper figures
results/
  figures/                     Generated exploratory and publication figures
  model_outputs/               Final-window interval and fuzzy-rule exports
  tables/                      Validation, prediction, metric, and robustness tables
scripts/
  generate_manuscript_diagrams.py
src/
  data.py                      Data retrieval and cleaning
  exploration.py               Data audit and exploratory figures
  validation.py                Rolling-origin split and metric framework
  modeling.py                  Baselines, FTS models, and main predictions
  sensitivity.py               Initial-window sensitivity analysis
  robustness.py                Expanded correction-weight and paired tests
  publication_figures.py       Publication figure generation
```

## Environment

Use Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The dependency list is intentionally compact: pandas, numpy, matplotlib, scikit-learn, statsmodels, scipy, and jupyter.

## Reproduce The Study

Run the commands from the repository root.

```powershell
python -m src.data
python -m src.exploration
python -m src.validation
python -m src.modeling
python -m src.sensitivity
python -m src.robustness
python -m src.publication_figures
```

To execute the canonical notebook:

```powershell
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks\ghana_cocoa_yield_forecasting.ipynb
```

## Data

The original provider is the Food and Agriculture Organization of the United Nations through FAOSTAT. The project accesses the series through Our World in Data:

- Data page: https://ourworldindata.org/grapher/cocoa-bean-yields
- CSV endpoint: https://ourworldindata.org/grapher/cocoa-bean-yields.csv?v=1&csvType=full&useColumnShortNames=false
- Metadata endpoint: https://ourworldindata.org/grapher/cocoa-bean-yields.metadata.json

The processed dataset is `data/processed/ghana_cocoa_yield_1961_2024.csv`. It contains 64 annual observations from 1961 to 2024 and no missing years after filtering to Ghana.

## Validation Design

The main experiment uses rolling-origin validation with 30 initial training observations. The first forecast year is 1991 and the final forecast year is 2024, producing 34 one-step-ahead forecasts per model. A 25-observation initial training window is used in robustness analysis.

All interval boundaries, ARIMA orders, fuzzy rules, target transformations, and correction weights used by a model are estimated from training data available before the forecast year. Evaluation high-volatility labels are used only for subgroup reporting.

## Model Families

The main experiment evaluates 48 model scenarios:

- Baseline references: Naive, expanding-window mean, drift, and ARIMA selected by AIC.
- Level-based Chen fuzzy time-series models with equal-length and adaptive quantile intervals.
- Order-2 adaptive Chen fuzzy time-series models.
- Fuzzy correction models using absolute-change and percentage-change targets.

Candidate interval counts are 5, 7, 9, and 11. Fuzzy forecasts use frequency-weighted midpoint defuzzification.

## Key Outputs

Core result tables:

- `results/tables/rolling_predictions.csv`
- `results/tables/overall_metrics.csv`
- `results/tables/volatility_metrics.csv`
- `results/tables/sensitivity_summary.csv`
- `results/tables/robustness_summary.csv`
- `results/tables/paired_error_tests.csv`
- `results/tables/publication_figure_manifest.csv`

Interpretability outputs:

- `results/model_outputs/equal_interval_rules_final_window.csv`
- `results/model_outputs/adaptive_interval_rules_final_window.csv`
- `results/model_outputs/high_order_rules_final_window.csv`
- `results/model_outputs/change_interval_rules_final_window.csv`
- `results/model_outputs/percentage_change_interval_rules_final_window.csv`

Paper artifacts:

- `paper/manuscript.md`
- `paper/figures/fig_1.png` through `paper/figures/fig_9.png`

## Verification

Use these checks before publishing or submitting:

```powershell
python -m compileall src scripts
python -m src.validation
python -m src.modeling
python -m src.robustness
python -m src.publication_figures
```

Expected high-level checks:

- Main predictions: 1,632 rows, equal to 34 validation splits times 48 model scenarios.
- Robustness correction predictions: 4,672 rows.
- Forecast values are finite.
- Every validation split has one forecast per model scenario.
- Publication figure links in `paper/manuscript.md` resolve to files in `paper/figures/`.
