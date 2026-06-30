# Ghana Cocoa Yield Data Card

## Source

- Original provider: Food and Agriculture Organization of the United Nations (FAOSTAT).
- Retrieval route: Our World in Data Grapher.
- Data page: https://ourworldindata.org/grapher/cocoa-bean-yields
- CSV endpoint: https://ourworldindata.org/grapher/cocoa-bean-yields.csv?v=1&csvType=full&useColumnShortNames=false
- Metadata endpoint: https://ourworldindata.org/grapher/cocoa-bean-yields.metadata.json
- Retrieval date: 2026-06-30

## Dataset

- Country: Ghana.
- Indicator: cocoa bean yield.
- Detected source column: `Cocoa beans - Yield (tonnes per hectare)`.
- Project column: `yield_tonnes_per_hectare`.
- Unit: tonnes per hectare.
- Frequency: annual.
- Year range: 1961-2024.
- Processed observations: 64.

## Saved Files

- Raw CSV: `data/raw/cocoa-bean-yields.csv`.
- Metadata JSON: `data/raw/cocoa-bean-yields.metadata.json`.
- Processed Ghana dataset: `data/processed/ghana_cocoa_yield_1961_2024.csv`.
- Integrity report: `results/tables/data_integrity_report.csv`.

## Column Mapping

| Source column | Project column |
|---|---|
| `Year` | `year` |
| `Cocoa beans - Yield (tonnes per hectare)` | `yield_tonnes_per_hectare` |

## Derived Columns

- `yield_lag1`: previous year's yield.
- `absolute_change`: current yield minus previous year's yield.
- `percentage_change`: year-to-year percentage change.
- `absolute_percentage_change`: absolute value of `percentage_change`.
- `exploratory_high_volatility`: provisional label using the 75th percentile of full-series absolute percentage change.

## Cleaning Rules

- Filtered the raw OWID file to `Entity == "Ghana"`.
- Sorted annual observations by year.
- Removed exact duplicate years only when duplicate values were identical.
- Stopped execution if duplicate years had conflicting yield values.
- Did not interpolate missing years or missing yield values.

## Integrity Notes

- Missing years: none.
- Missing yield values: 0.
- Metadata available: True.
- Metadata error: none.
- Provisional high-volatility threshold: 13.7489.

## Limitations

- This data card documents dataset preparation only.
- The high-volatility flag is exploratory and uses the full Ghana series. The modeling workflow should recompute volatility labels on the evaluation period only.
- This project forecasts yield only; it does not model production volume, prices, climate variables, or causal drivers.
