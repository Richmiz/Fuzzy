# Study Handoff Specification

## 1. Study Identity

**Working title:** Volatility-Aware Fuzzy Time-Series Forecasting of Ghana Cocoa Bean Yield Using Adaptive Intervals and High-Order Rules

**Short title:** Ghana Cocoa Yield FTS Forecasting

**Target article type:** empirical forecasting study with reproducible code and official public data.

**Target publication level:** Scopus Q3 or Q4. The final journal must be verified close to submission because indexing status and quartiles can change.

## 2. One-Paragraph Summary

This study investigates one-step-ahead annual forecasting of Ghana cocoa bean yield using fuzzy time-series models. It compares classical Chen FTS with equal-length intervals, Chen FTS with adaptive intervals, and high-order FTS using rolling-origin validation. The study also includes simple non-fuzzy baselines, such as Naive and ARIMA or ETS, so the fuzzy models are evaluated against standard forecasting references. Accuracy is measured using MAE, RMSE, sMAPE, and MASE. The analysis is reported for all test years and separately for high-volatility years, defined by large absolute year-to-year changes in yield. The contribution is not the invention of a new fuzzy algorithm, but a reproducible, volatility-aware comparison of fuzzy time-series design choices for a strategically important African agricultural yield series.

## 3. Final Scope

**Included:**

- Ghana only.
- Cocoa beans only.
- Yield only, not production volume.
- Annual time series.
- FAOSTAT / UN FAO data retrieved through the Our World in Data Grapher CSV endpoint.
- Univariate forecasting.
- One-step-ahead forecasting.
- Rolling-origin validation.
- Fuzzy time-series model comparison.
- Naive and ARIMA or ETS baselines.
- Separate evaluation for high-volatility years.

**Excluded:**

- Multi-country forecasting.
- Cocoa price forecasting.
- Climate-variable modeling.
- Fertilizer, rainfall, temperature, and remote-sensing covariates.
- Deep learning.
- Long-horizon 2030 policy forecasting.
- Causal claims about why yield changed.
- Claims that fuzzy methods are universally better than statistical models.

## 4. Research Problem

Ghana is a major cocoa-producing country, but cocoa production volume alone can be misleading because production may change due to harvested area rather than land productivity. Cocoa bean yield is a more focused measure because it expresses output per land area. Forecasting yield is useful for understanding productivity movement, but annual agricultural series can be unstable because of agronomic, environmental, economic, and reporting influences. Fuzzy time-series models are attractive for small and uncertain time series, but their performance depends heavily on interval partitioning and fuzzy relationship design. This study asks whether adaptive intervals and high-order fuzzy logical relationships improve forecasting reliability for Ghana cocoa bean yield, especially during high-volatility years.

## 5. Main Research Question

Which fuzzy time-series design provides the most reliable one-year-ahead forecasts of Ghana cocoa bean yield under normal and high-volatility yield conditions?

## 6. Sub-Questions

1. Does adaptive interval partitioning improve Chen FTS performance compared with equal-length intervals?
2. Does high-order FTS improve forecasting accuracy compared with first-order FTS?
3. Do fuzzy time-series models outperform simple non-fuzzy baselines such as Naive and ARIMA or ETS?
4. Which model is most reliable during high-volatility years?
5. How sensitive are the results to the number of intervals and fuzzy rule order?

## 7. Expected Contribution

The study's defensible contribution is:

```text
This study provides a volatility-aware empirical comparison of fuzzy time-series design choices for Ghana cocoa bean yield forecasting. It evaluates equal-length interval Chen FTS, adaptive-interval Chen FTS, and high-order FTS under rolling-origin validation, with additional comparison against standard non-fuzzy baselines.
```

Do not claim:

- a new fuzzy algorithm, unless one is explicitly developed;
- a new Ghana cocoa dataset;
- causal explanation of cocoa yield changes;
- policy effectiveness;
- superiority of fuzzy methods without evidence from the saved metrics.

## 8. Dataset

**Original source:** FAOSTAT / Food and Agriculture Organization of the United Nations.

**Automated retrieval route:** Our World in Data Grapher.

**Data page:** https://ourworldindata.org/grapher/cocoa-bean-yields

**CSV endpoint:**

```text
https://ourworldindata.org/grapher/cocoa-bean-yields.csv?v=1&csvType=full&useColumnShortNames=false
```

**Python loading code:**

```python
import pandas as pd

url = (
    "https://ourworldindata.org/grapher/"
    "cocoa-bean-yields.csv?v=1&csvType=full&useColumnShortNames=false"
)

df = pd.read_csv(url)
ghana = df[df["Entity"] == "Ghana"].copy()

print(ghana.head())
print(ghana.tail())
```

**Expected columns:**

- `Entity`
- `Code`
- `Year`
- cocoa bean yield column, exact name should be confirmed after loading

**Expected filtered dataset:**

- Entity: Ghana.
- Years: 1961-2024, subject to final retrieval verification.
- Observations: approximately 64.
- Frequency: annual.
- Unit: tonnes per hectare, subject to final source verification.

**Processed dataset columns:**

```text
year
yield_tonnes_per_hectare
yield_lag1
absolute_change
percentage_change
absolute_percentage_change
is_high_volatility
```

## 9. Data Processing Rules

1. Load the full CSV from the OWID endpoint.
2. Filter `Entity == "Ghana"`.
3. Keep only year and yield columns.
4. Rename columns to stable project names.
5. Sort by year ascending.
6. Remove duplicate years if any are found, but document the issue.
7. Check for missing years.
8. Check for missing yield values.
9. Save the raw downloaded file before cleaning.
10. Save the cleaned Ghana-only file separately.
11. Never interpolate missing values silently. If missing values exist, document and choose a transparent rule.

## 10. High-Volatility Definition

Compute year-to-year percentage change:

```text
percentage_change_t = 100 * (yield_t - yield_t-1) / yield_t-1
absolute_percentage_change_t = abs(percentage_change_t)
```

Define high-volatility years inside the evaluation period:

```text
high_volatility_t = absolute_percentage_change_t >= 75th percentile of absolute percentage changes among test years
```

Important rule:

- The threshold used for analysis must be reported.
- For strict forecasting purity, the volatility label can be computed after observing the actual test year because it is used for evaluation grouping, not for training.
- Do not use volatility labels as model inputs in the main study.

## 11. Models

### Model A - Naive Forecast

Forecast:

```text
y_hat_t = y_t-1
```

Purpose:

- Minimum benchmark.
- Any useful model should be compared against it.

### Model B - ARIMA Or ETS

Purpose:

- Standard statistical time-series benchmark.
- Use only historical yield values in each training window.

Recommended implementation:

- Use `statsmodels`.
- Keep the selection simple and reproducible.
- If automatic order selection is not available, test a small set of ARIMA orders and select by AIC inside each training window.

Possible ARIMA grid:

```text
p in {0, 1, 2}
d in {0, 1}
q in {0, 1, 2}
```

### Model C - Chen FTS With Equal-Length Intervals

Purpose:

- Classical fuzzy time-series baseline.

Core steps:

1. Define universe of discourse using training data.
2. Add margins around min and max.
3. Create equal-length intervals.
4. Fuzzify observations.
5. Build first-order fuzzy logical relationships.
6. Group rules into FLRGs.
7. Forecast using midpoint averages.
8. Defuzzify into numeric yield values.

### Model D - Chen FTS With Adaptive Intervals

Purpose:

- Test whether data-aware interval partitioning improves forecasting.

Primary method:

- Quantile-based adaptive intervals.

Reason:

- Simple.
- Reproducible.
- Handles uneven distribution better than equal-length intervals.

Optional sensitivity method:

- Average-based interval length.

### Model E - High-Order FTS

Purpose:

- Test whether using more fuzzy-state history improves forecast reliability.

Main model:

```text
Order-2 FTS
```

Optional model:

```text
Order-3 FTS
```

Rule examples:

```text
Order 1: A(t-1) -> A(t)
Order 2: A(t-2), A(t-1) -> A(t)
Order 3: A(t-3), A(t-2), A(t-1) -> A(t)
```

Fallback rule:

```text
If the high-order antecedent is unseen, fall back to order-1.
If order-1 is also unseen, fall back to the nearest interval midpoint or Naive forecast.
```

Document the fallback choice exactly.

## 12. Interval Construction

### Equal-Length Intervals

For training values `y_train`:

```text
lower = min(y_train) - margin
upper = max(y_train) + margin
width = (upper - lower) / k
```

Suggested margin:

```text
margin = 0.05 * (max(y_train) - min(y_train))
```

Candidate interval counts:

```text
k = 5, 7, 9, 11
```

### Quantile-Based Adaptive Intervals

For training values `y_train`:

```text
boundaries = quantiles of y_train at 0, 1/k, 2/k, ..., 1
```

Rules:

- Remove duplicate boundaries if the data cause repeated quantiles.
- Ensure the minimum and maximum values are included.
- Use only the training window to compute boundaries.
- Do not compute intervals from the full dataset before rolling validation.

## 13. Fuzzification

Each numeric yield value is assigned to an interval:

```text
y_t in interval_i -> fuzzy state A_i
```

Boundary rule:

- Intervals are left-closed and right-open, except the final interval, which is closed on both sides.

Example:

```text
[lower_i, upper_i) -> A_i
[lower_last, upper_last] -> A_last
```

## 14. Defuzzification

Each interval has a midpoint:

```text
midpoint_i = (lower_i + upper_i) / 2
```

For a fuzzy logical relationship group:

```text
A_i -> A_j, A_k, A_l
```

Numeric forecast:

```text
y_hat = average(midpoint_j, midpoint_k, midpoint_l)
```

If frequency-weighted relationships are used, document the weighting:

```text
y_hat = weighted average of consequent midpoints using consequent frequencies
```

Recommended primary setting:

- Use frequency-weighted consequent midpoints.
- Include unweighted midpoint average as optional sensitivity if time allows.

## 15. Validation Design

Use expanding-window rolling-origin validation.

Recommended initial training window:

```text
30 annual observations
```

If the dataset starts in 1961:

```text
Train 1961-1990 -> forecast 1991
Train 1961-1991 -> forecast 1992
Train 1961-1992 -> forecast 1993
...
Train 1961-2023 -> forecast 2024
```

Rules:

- Recompute intervals inside each training window.
- Rebuild fuzzy rules inside each training window.
- Refit ARIMA/ETS inside each training window.
- Store every one-step forecast.
- Do not use random splits.
- Do not tune settings using the final test year only.

## 16. Evaluation Metrics

### MAE

```text
MAE = mean(abs(y_t - y_hat_t))
```

### RMSE

```text
RMSE = sqrt(mean((y_t - y_hat_t)^2))
```

### sMAPE

```text
sMAPE = mean(200 * abs(y_t - y_hat_t) / (abs(y_t) + abs(y_hat_t)))
```

### MASE

```text
MASE = MAE_model / MAE_naive_insample
```

where:

```text
MAE_naive_insample = mean(abs(y_t - y_t-1)) over the training data
```

Interpretation:

```text
MASE < 1 means the model improves over the in-sample naive benchmark.
```

## 17. Main Experiment Table

| ID | Model | Interval Strategy | Order | Purpose |
|---|---|---|---:|---|
| B1 | Naive | None | None | Minimum benchmark |
| B2 | ARIMA or ETS | None | None | Statistical benchmark |
| F1 | Chen FTS | Equal length | 1 | Classical fuzzy baseline |
| F2 | Chen FTS | Adaptive quantile | 1 | Interval improvement test |
| F3 | High-order FTS | Best from F1/F2 | 2 | Rule-order improvement test |
| F4 | High-order FTS | Best from F1/F2 | 3 | Optional sensitivity |

## 18. Required Output Files

### Data

```text
data/raw/cocoa-bean-yields.csv
data/processed/ghana_cocoa_yield_1961_2024.csv
```

### Tables

```text
results/tables/data_audit.csv
results/tables/high_volatility_years.csv
results/tables/model_scenarios.csv
results/tables/rolling_predictions.csv
results/tables/overall_metrics.csv
results/tables/volatility_metrics.csv
results/tables/sensitivity_metrics.csv
```

### Figures

```text
results/figures/yield_timeseries.png
results/figures/yield_pct_change.png
results/figures/high_volatility_years.png
results/figures/actual_vs_predicted_best_model.png
results/figures/model_error_comparison.png
```

### Paper And Presentation

```text
paper/manuscript.docx
paper/manuscript.pdf
slides/presentation.pptx
```

## 19. Suggested Code Architecture

### `src/data.py`

Responsibilities:

- Download CSV.
- Filter Ghana.
- Rename columns.
- Save raw and processed files.
- Produce data audit.

### `src/intervals.py`

Responsibilities:

- Equal interval generation.
- Quantile interval generation.
- Interval midpoint calculation.
- Boundary validation.

### `src/fts.py`

Responsibilities:

- Fuzzification.
- First-order FLR and FLRG construction.
- High-order FLR and FLRG construction.
- Defuzzification.
- Fallback handling.

### `src/baselines.py`

Responsibilities:

- Naive forecast.
- ARIMA or ETS forecast.

### `src/metrics.py`

Responsibilities:

- MAE.
- RMSE.
- sMAPE.
- MASE.
- Grouped metric calculation.

### `src/validation.py`

Responsibilities:

- Rolling-origin split generation.
- Model execution loop.
- Storage of predictions.
- Storage of per-split metadata.

### `src/plots.py`

Responsibilities:

- Time-series plots.
- Change plots.
- Actual vs predicted plots.
- Error comparison plots.

## 20. Main Script Behavior

Create a script such as:

```text
run_experiment.py
```

Expected behavior:

1. Load or download data.
2. Create processed dataset.
3. Run data audit.
4. Run rolling-origin validation for all models.
5. Compute metrics.
6. Save all tables.
7. Save all figures.
8. Print a short summary of best models.

Example command:

```bash
python run_experiment.py
```

## 21. Main Results To Report

The paper must answer these questions directly:

1. Which model has the lowest overall MAE?
2. Which model has the lowest overall RMSE?
3. Which model has the lowest overall sMAPE?
4. Which model has the lowest high-volatility-year sMAPE?
5. Does adaptive interval FTS beat equal-interval FTS?
6. Does high-order FTS beat first-order FTS?
7. Do fuzzy models beat Naive and ARIMA or ETS?
8. Is the improvement meaningful or very small?

## 22. Interpretation Rules

If adaptive FTS wins:

```text
The result suggests that data-aware partitioning is useful for this sparse agricultural yield series.
```

If high-order FTS wins:

```text
The result suggests that recent fuzzy-state history contains useful temporal information for Ghana cocoa yield forecasting.
```

If ARIMA or ETS wins:

```text
The result shows that fuzzy models are interpretable alternatives but did not outperform the statistical benchmark overall.
```

If fuzzy models win only during volatile years:

```text
The strongest contribution becomes volatility-specific reliability rather than overall accuracy.
```

If no model clearly wins:

```text
The study remains useful as evidence that model selection for short agricultural yield series is sensitive to interval construction, rule order, and evaluation regime.
```

## 23. Manuscript Structure

### 1. Introduction

Include:

- importance of cocoa to Ghana;
- why yield is more meaningful than production volume;
- forecasting challenge;
- fuzzy time-series motivation;
- gap;
- contribution.

### 2. Related Work

Include:

- fuzzy time-series foundations;
- Chen FTS;
- adaptive interval FTS;
- high-order FTS;
- agricultural FTS applications;
- cocoa or Ghana cocoa forecasting studies;
- forecast validation literature.

### 3. Data And Preprocessing

Include:

- data source;
- period;
- variable;
- unit;
- cleaning steps;
- volatility definition.

### 4. Methodology

Include:

- baselines;
- Chen FTS;
- adaptive intervals;
- high-order FTS;
- metrics.

### 5. Experimental Design

Include:

- rolling-origin validation;
- model scenarios;
- parameter grid;
- evaluation groups.

### 6. Results

Include:

- overall metrics;
- volatility-group metrics;
- sensitivity analysis;
- plots.

### 7. Discussion

Include:

- answer each research question;
- explain where adaptive intervals helped or failed;
- explain where high-order rules helped or failed;
- compare fuzzy models against baselines;
- limitations.

### 8. Conclusion

Include:

- concise findings;
- best model;
- practical implication;
- limitations;
- future work.

## 24. Claim Boundaries

Safe claims:

- The study compares fuzzy time-series design choices.
- The dataset is official public agricultural data.
- Yield is more productivity-focused than production volume.
- Rolling-origin validation is more appropriate than random splitting for time series.
- The best model is best under the reported metrics and validation setup.

Unsafe claims:

- This is the first ever cocoa forecasting paper.
- This is the first ever FTS agricultural study.
- This proves fuzzy models are superior.
- This explains the causes of Ghana cocoa yield changes.
- This gives policy prescriptions without external evidence.
- This guarantees future yield values.

## 25. Literature Gap Wording

Use wording like this:

```text
Although fuzzy time-series forecasting has been widely studied, fewer works evaluate how interval partitioning and fuzzy relationship order affect short annual agricultural yield series. In cocoa-related forecasting, many studies focus on production volume, prices, or broader agronomic modeling, while yield-focused fuzzy time-series evidence for Ghana remains limited. This study addresses that gap by comparing equal-interval Chen FTS, adaptive-interval Chen FTS, and high-order FTS for Ghana cocoa bean yield using rolling-origin validation and volatility-specific error analysis.
```

Avoid wording like this unless verified:

```text
No previous study has ever forecast Ghana cocoa yield.
```

## 26. Journal Strategy

For Scopus Q3/Q4:

1. Confirm the journal is currently indexed in Scopus.
2. Confirm the current CiteScore quartile.
3. Check whether forecasting, fuzzy systems, agriculture, informatics, or applied mathematics fits the journal scope.
4. Check article processing charges.
5. Check publication ethics and indexing warnings.
6. Avoid journals with unclear websites or suspicious acceptance promises.
7. Keep the manuscript general enough for applied computing, agricultural informatics, or decision science journals.

Useful verification pages:

```text
https://www.scopus.com/sources
https://www.scimagojr.com/journalrank.php
```

## 27. Team Role Split

### Person 1 - Literature And Writing

- Build literature matrix.
- Draft Introduction and Related Work.
- Verify journal scope and format.
- Manage references.

### Person 2 - Data And Modeling

- Download and clean dataset.
- Implement baselines and FTS models.
- Run rolling-origin validation.
- Save tables and figures.

### Person 3 - Results And Presentation

- Check metrics and plots.
- Draft Results and Discussion.
- Prepare slides.
- Prepare presentation script and video.

If only one person works on the project, follow the phase order in `GHANA_COCOA_FTS_IMPLEMENTATION_PHASES.md`.

## 28. Acceptance Criteria

The project is ready for manuscript writing when:

- The dataset can be loaded from code.
- Cleaned Ghana-only data is saved.
- All model predictions are saved.
- Overall and volatility-specific metrics are saved.
- At least five models are compared: Naive, ARIMA or ETS, equal Chen FTS, adaptive Chen FTS, and order-2 FTS.
- Sensitivity over interval count is complete.
- Figures are publication-ready.
- The main claim is supported by the result tables.

The project is ready for submission when:

- Manuscript follows the target journal template.
- References are formatted correctly.
- All tables and figures match saved outputs.
- Code runs from a fresh environment.
- Journal indexing and quartile are verified.
- Claims are not stronger than the evidence.

## 29. Immediate Next Actions

1. Create the project folder structure.
2. Download and inspect the Ghana dataset.
3. Build the data audit table.
4. Implement rolling-origin validation.
5. Implement Naive baseline.
6. Implement Chen FTS with equal intervals.
7. Add adaptive intervals.
8. Add order-2 high-order FTS.
9. Add ARIMA or ETS baseline.
10. Generate final result tables and figures.

