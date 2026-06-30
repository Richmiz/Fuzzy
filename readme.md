# Ghana Cocoa Yield FTS Forecasting

This repository contains the Phase 2-3 setup for a fuzzy time-series study of Ghana cocoa bean yield. The project prepares a reproducible Python workspace and downloads official public data from the Our World in Data Grapher endpoint, which is based on FAOSTAT data.

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

Not implemented yet:

- Exploratory plots.
- Fuzzy time-series models.
- Naive, ARIMA, or ETS baselines.
- Rolling-origin validation.
- Manuscript, slides, or presentation assets.

Modeling starts in Phase 4 and later.

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

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The Phase 2-3 dependency set is intentionally small:

- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels
- scipy
- jupyter

Optional packages such as pyFTS, seaborn, and plotly are not included yet.

## Download And Prepare Data

Run the canonical Phase 3 command from the project root:

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

The Phase 3 high-volatility flag is provisional. It uses the 75th percentile of full-series absolute percentage change and should be recomputed on the evaluation period during rolling-origin validation.

## Data Quality Rules

- Filter to `Entity == "Ghana"`.
- Keep only year and the detected cocoa yield value column.
- Sort by year ascending.
- Do not interpolate missing years.
- Stop if the processed dataset has missing yield values.
- Remove duplicate years only if duplicate values are identical.
- Stop if duplicate years contain conflicting yield values.

## Next Work

Phase 4 should add data audit and exploratory figures. Later phases should implement the baseline and fuzzy time-series models, rolling-origin validation, result tables, figures, and writing artifacts.
