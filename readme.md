# Ghana Cocoa Yield FTS Forecasting

This repository contains a fuzzy time-series study of Ghana cocoa bean yield. The project prepares a reproducible Python workspace and downloads official public data from the Our World in Data Grapher endpoint, which is based on FAOSTAT data.

The previous root README described a cocoa bean production data package. That production file is not the main study target. This project uses cocoa bean yield for Ghana.

## Current Status

Implemented:

- Project folder structure.
- Python dependency list.
- Data retrieval and cleaning script.
- Raw OWID cocoa yield CSV archive.
- Ghana-only processed yield dataset.
- Data card.
- Basic data integrity report.
- Data audit and exploratory figures.
- Rolling-origin validation framework.
- Forecast metric definitions and validation checks.

Not implemented yet:

- Fuzzy time-series models.
- Naive, ARIMA, or ETS baselines.
- Manuscript, slides, or presentation assets.

Model implementation starts after the validation framework.

## Project Structure

```text
data/
  raw/
  processed/
notebooks/
src/
results/
  tables/
  figures/
  model_outputs/
paper/
slides/
```

## Setup

Use Python 3.11 for the project environment. The scientific Python stack installs cleanly on Python 3.11, while Python 3.14 may trigger package-resolution or wheel-availability delays.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The dependency set is intentionally small:

- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels
- scipy
- jupyter

Optional packages such as pyFTS, seaborn, and plotly are not included yet.

## Prepare Data

Run the canonical data preparation command from the project root:

```bash
python -m src.data
```

The script downloads:

```text
https://ourworldindata.org/grapher/cocoa-bean-yields.csv?v=1&csvType=full&useColumnShortNames=false
```

It also attempts to download:

```text
https://ourworldindata.org/grapher/cocoa-bean-yields.metadata.json
```

If metadata is unavailable, the script continues and records the issue in the data card and integrity report.

## Expected Outputs

```text
data/raw/cocoa-bean-yields.csv
data/raw/cocoa-bean-yields.metadata.json
data/processed/ghana_cocoa_yield_1961_2024.csv
data/DATA_CARD.md
results/tables/data_integrity_report.csv
```

The processed dataset contains:

```text
year
yield_tonnes_per_hectare
yield_lag1
absolute_change
percentage_change
absolute_percentage_change
is_high_volatility
```

The exploratory high-volatility flag in the processed dataset uses the 75th percentile of full-series absolute percentage change. Model evaluation uses a separate evaluation-period volatility label.

## Run Data Audit And Exploration

```powershell
python -m src.exploration
```

This generates:

```text
results/tables/data_audit.csv
results/tables/exploratory_high_volatility_years.csv
results/figures/yield_timeseries.png
results/figures/yield_pct_change.png
results/figures/exploratory_high_volatility_years.png
results/figures/yield_change_distribution.png
results/README.md
```

## Build Validation Framework

```powershell
python -m src.validation
```

This generates:

```text
results/tables/validation_splits.csv
results/tables/evaluation_high_volatility_years.csv
results/tables/metric_definitions.csv
results/tables/rolling_predictions_template.csv
results/tables/validation_framework_checks.csv
```

## Study Notebook

Use one notebook for the study:

```text
notebooks/ghana_cocoa_yield_forecasting.ipynb
```

Launch it with the project environment:

```powershell
.\.venv\Scripts\jupyter.exe notebook notebooks\ghana_cocoa_yield_forecasting.ipynb
```

## Data Quality Rules

- Filter to `Entity == "Ghana"`.
- Keep only year and the detected cocoa yield value column.
- Sort by year ascending.
- Do not interpolate missing years.
- Stop if the processed dataset has missing yield values.
- Remove duplicate years only if duplicate values are identical.
- Stop if duplicate years contain conflicting yield values.

## Next Work

Next work should implement the baseline and fuzzy time-series models, then write final result tables, figures, and manuscript artifacts from saved outputs.
