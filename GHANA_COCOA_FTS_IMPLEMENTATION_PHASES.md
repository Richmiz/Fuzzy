# Implementation Phases

## Project

**Working title:** Volatility-Aware Fuzzy Time-Series Forecasting of Ghana Cocoa Bean Yield Using Adaptive Intervals and High-Order Rules

**Core objective:** compare fuzzy time-series model designs for one-step-ahead annual forecasting of Ghana cocoa bean yield, with special attention to high-volatility years.

**Target publication level:** Scopus Q3 or Q4, if the final article has clean literature positioning, strong baselines, reproducible code, and careful claims.

## Phase 0 - Scope Lock And Journal Strategy

**Goal:** make the project narrow enough to finish and strong enough for Q3/Q4 positioning.

**Decisions to lock:**

- Country: Ghana only.
- Commodity: cocoa beans only.
- Forecast target: yield, not production volume.
- Frequency: annual.
- Period: 1961-2024, subject to final dataset verification.
- Main method family: fuzzy time series.
- Main comparison: equal intervals, adaptive intervals, and high-order fuzzy logical relationships.
- Validation: rolling-origin one-step-ahead forecasting.
- Evaluation focus: overall accuracy and high-volatility-year accuracy.

**Publication framing:**

- Do not claim a new fuzzy algorithm unless a genuinely new algorithm is developed.
- Present the paper as a reproducible empirical comparison and evaluation framework.
- For Q3/Q4, include non-fuzzy baselines such as Naive and ARIMA or ETS.
- Verify the selected journal's current Scopus status, quartile, aims, scope, article type, fees, and review history before submission.

**Deliverables:**

- Final title.
- Final research question.
- Target journal shortlist.
- One-paragraph scope statement.
- One-paragraph contribution statement.

## Phase 1 - Literature Search And Gap Confirmation

**Goal:** confirm that the study is not an obvious duplicate and build the paper's intellectual foundation.

**Search databases:**

- Scopus.
- Google Scholar.
- Semantic Scholar.
- IEEE Xplore, ScienceDirect, SpringerLink, MDPI, Taylor & Francis, or other relevant platforms depending on access.

**Core search strings:**

```text
"Ghana cocoa yield" +"fuzzy time series"
"cocoa bean yield" +"fuzzy time series"
"Ghana cocoa" +"Chen fuzzy time series"
"cocoa yield" +"high-order fuzzy time series"
"cocoa production" +"fuzzy time series"
"adaptive interval" +"fuzzy time series" +"agriculture"
"high-order fuzzy time series" +"crop yield"
"fuzzy time series" +"agricultural forecasting"
"Ghana cocoa yield" +"forecasting"
"cocoa production" +"ARIMA" +"Ghana"
```

**Literature groups to collect:**

- Foundational fuzzy time-series papers.
- Chen FTS papers.
- High-order FTS papers.
- Adaptive or unequal interval FTS papers.
- FTS applications in agriculture.
- Cocoa forecasting or Ghana cocoa studies.
- Rolling-origin validation and forecast evaluation references.

**Deliverables:**

- Literature matrix with at least 25 papers.
- Short duplicate-risk note.
- Final research gap paragraph.
- Reference manager file, for example BibTeX or Zotero export.

## Phase 2 - Project Repository Setup

**Goal:** prepare a reproducible workspace that a reviewer or teammate can run.

**Recommended structure:**

```text
ghana-cocoa-fts/
  data/
    raw/
    processed/
  notebooks/
  src/
    data.py
    intervals.py
    fts.py
    baselines.py
    metrics.py
    validation.py
    plots.py
  results/
    tables/
    figures/
    model_outputs/
  paper/
  slides/
  README.md
  requirements.txt
```

**Recommended Python packages:**

```text
pandas
numpy
matplotlib
scikit-learn
statsmodels
scipy
jupyter
```

Optional:

```text
pyFTS
seaborn
plotly
```

**Deliverables:**

- Executable project environment.
- `README.md` with setup and run instructions.
- Fixed folder structure.
- Versioned requirements file.

## Phase 3 - Dataset Retrieval And Data Card

**Goal:** collect the official data in a reproducible way and document exactly what was used.

**Primary automated retrieval route:**

```python
import pandas as pd

url = (
    "https://ourworldindata.org/grapher/"
    "cocoa-bean-yields.csv?v=1&csvType=full&useColumnShortNames=false"
)

df = pd.read_csv(url)
ghana = df[df["Entity"] == "Ghana"].copy()
```

**Source description:**

- Original provider: FAOSTAT / Food and Agriculture Organization of the United Nations.
- Retrieval route: Our World in Data Grapher CSV endpoint.
- Indicator: cocoa bean yield.
- Unit: tonnes per hectare, subject to final source verification.
- Country: Ghana.

**Deliverables:**

- `data/raw/cocoa-bean-yields.csv`.
- `data/processed/ghana_cocoa_yield_1961_2024.csv`.
- Data card with source, retrieval date, variable definitions, units, and known limitations.
- Basic data integrity report.

## Phase 4 - Data Audit And Exploratory Analysis

**Goal:** understand the time series before modeling.

**Checks:**

- Number of observations.
- First and last year.
- Missing years.
- Duplicate years.
- Missing yield values.
- Unit consistency.
- Minimum, maximum, mean, median, standard deviation.
- Year-to-year absolute change.
- Year-to-year percentage change.

**High-volatility-year definition:**

```text
A high-volatility year is a test year whose absolute year-to-year percentage change in yield is in the top 25 percent of the evaluation period.
```

**Figures:**

- Ghana cocoa yield over time.
- Year-to-year percentage change.
- Yield with high-volatility years highlighted.
- Histogram or boxplot of annual yield changes.

**Deliverables:**

- `results/tables/data_audit.csv`.
- `results/tables/high_volatility_years.csv`.
- `results/figures/yield_timeseries.png`.
- `results/figures/yield_pct_change.png`.

## Phase 5 - Metric And Validation Framework

**Goal:** implement the evaluation logic before implementing models.

**Validation design:**

- Use rolling-origin expanding-window validation.
- Forecast horizon: one year ahead.
- No random train-test splitting.
- Suggested initial training window: 30 years.

**Example:**

```text
Train 1961-1990 -> forecast 1991
Train 1961-1991 -> forecast 1992
Train 1961-1992 -> forecast 1993
...
Train 1961-2023 -> forecast 2024
```

**Metrics:**

- MAE.
- RMSE.
- sMAPE.
- MASE, recommended for Scopus-level comparison.

**Evaluation groups:**

- All test years.
- High-volatility test years only.
- Normal-volatility test years only.

**Deliverables:**

- `src/metrics.py`.
- `src/validation.py`.
- Unit checks using small synthetic time series.
- Empty result table template.

## Phase 6 - Baseline Forecasting Models

**Goal:** add simple non-fuzzy benchmarks so the fuzzy models are not evaluated in isolation.

**Minimum baselines:**

- Naive forecast: forecast next year as the current year's yield.
- Mean forecast or drift forecast, optional.

**Scopus-strength baseline:**

- ARIMA or ETS using only historical yield data inside each rolling training window.

**Rules:**

- Baselines must use the same rolling-origin splits as fuzzy models.
- No future information can enter model fitting.
- Keep automatic ARIMA/ETS selection simple and reproducible.

**Deliverables:**

- `src/baselines.py`.
- Prediction table for each baseline.
- Baseline metric table.

## Phase 7 - Scenario 1: Chen FTS With Equal-Length Intervals

**Goal:** implement the classical fuzzy time-series baseline.

**Steps:**

1. Define universe of discourse from the rolling training data.
2. Add a small lower and upper margin around min and max.
3. Split the universe into equal-length intervals.
4. Fuzzify each observation by assigning it to an interval.
5. Build first-order fuzzy logical relationships.
6. Group relationships into FLRGs.
7. Forecast one step ahead using interval midpoints.
8. Defuzzify to numeric yield values.

**Parameters to test:**

- Number of intervals: 5, 7, 9, 11.
- Primary selected setting: choose based on rolling validation, not test hindsight.

**Deliverables:**

- `src/intervals.py`.
- `src/fts.py`.
- Equal-interval prediction table.
- Equal-interval rule table for the final training window.
- Metric table.

## Phase 8 - Scenario 2: Chen FTS With Adaptive Intervals

**Goal:** test whether data-aware intervals improve forecast reliability.

**Primary adaptive interval method:**

- Quantile-based intervals.
- This creates intervals with more balanced data frequency.
- It is easy to explain and reproducible.

**Optional sensitivity method:**

- Average-based interval length.
- Use only if time allows or if the selected journal expects closer FTS literature alignment.

**Steps:**

1. Generate adaptive interval boundaries from training data only.
2. Fuzzify observations using adaptive intervals.
3. Build first-order fuzzy logical relationships.
4. Forecast and defuzzify using adaptive interval midpoints.
5. Compare against equal-length Chen FTS under the same validation design.

**Deliverables:**

- Adaptive interval generator.
- Adaptive interval table.
- Adaptive Chen FTS prediction table.
- Metric table.
- Comparison against Scenario 1.

## Phase 9 - Scenario 3: High-Order FTS

**Goal:** test whether adding lagged fuzzy states improves forecasting.

**Primary high-order model:**

- Order-2 FTS.

**Optional sensitivity model:**

- Order-3 FTS.

**Recommended setup:**

- Use the best interval strategy from Scenarios 1 and 2.
- Compare high-order results against first-order FTS.

**Rule form:**

```text
Order 1: A(t-1) -> A(t)
Order 2: A(t-2), A(t-1) -> A(t)
Order 3: A(t-3), A(t-2), A(t-1) -> A(t)
```

**Unseen-rule fallback:**

- If an antecedent pattern is not found, fall back to the lower-order rule.
- If the lower-order rule is also not found, fall back to the naive forecast or nearest interval midpoint.
- Document this rule clearly.

**Deliverables:**

- High-order FTS implementation.
- Order-2 prediction table.
- Optional order-3 prediction table.
- Rule table for final training window.
- Metric table.

## Phase 10 - Sensitivity And Robustness Analysis

**Goal:** show that the conclusion is not based on a single arbitrary setting.

**Sensitivity checks:**

- Number of intervals: 5, 7, 9, 11.
- Interval strategy: equal vs adaptive.
- FTS order: 1, 2, and optionally 3.
- Initial training window: 25 vs 30 years, optional.

**Analysis questions:**

- Does adaptive partitioning help consistently or only under one interval count?
- Does order-2 improve volatile-year performance?
- Does order-3 overfit because the dataset is small?
- Does any fuzzy model beat Naive and ARIMA/ETS?

**Deliverables:**

- Sensitivity summary table.
- Robustness discussion notes.
- Recommended final model selection.

## Phase 11 - Result Tables And Figures

**Goal:** produce publication-ready evidence.

**Main tables:**

- Dataset summary.
- High-volatility years.
- Scenario design.
- Overall model comparison.
- High-volatility model comparison.
- Sensitivity analysis.

**Main figures:**

- Ghana cocoa yield time series.
- Yield change with volatility threshold.
- Actual vs forecast for best fuzzy model.
- Actual vs forecast for all main models, if readable.
- Error comparison bar chart.

**Deliverables:**

- `results/tables/*.csv`.
- `results/figures/*.png`.
- A short `results/README.md` explaining each artifact.

## Phase 12 - Article Writing

**Goal:** write the paper from verified artifacts, not from assumptions.

**Recommended sections:**

1. Introduction.
2. Related Work.
3. Data and Preprocessing.
4. Methodology.
5. Experimental Design.
6. Results.
7. Discussion.
8. Conclusion.

**Writing rules:**

- Keep claims bounded by actual results.
- Do not claim policy impact unless the evidence supports it.
- Do not claim that fuzzy methods are universally better.
- If fuzzy methods do not beat ARIMA/ETS overall, discuss where they help or fail.
- Separate overall performance from high-volatility performance.

**Deliverables:**

- Full manuscript draft.
- Tables and figures inserted.
- References formatted for target journal.
- Limitations section.

## Phase 13 - Submission Package And Reproducibility Check

**Goal:** make the project runnable and submit-ready.

**Final package:**

- Manuscript `.docx` and `.pdf`.
- Source code.
- Cleaned dataset.
- Result tables.
- Figures.
- README.
- Slides.
- Presentation video, if needed for the course.

**Reproducibility checklist:**

- Fresh environment can install dependencies.
- Raw data can be downloaded or loaded from the archived raw file.
- Running the main script regenerates all tables and figures.
- Metrics in the paper match saved CSV files.
- All figures in the paper match saved figure files.
- Journal status and quartile have been checked close to submission date.

**Deliverables:**

- Final zipped project folder.
- Final article.
- Final presentation assets.
- Submission checklist.

## Suggested Timeline

| Week | Work |
|---|---|
| 1 | Scope, journal shortlist, literature search |
| 2 | Dataset, data audit, EDA, volatility definition |
| 3 | Metrics, rolling-origin validation, baselines |
| 4 | Chen FTS equal and adaptive intervals |
| 5 | High-order FTS and sensitivity checks |
| 6 | Results, tables, figures |
| 7 | Manuscript draft |
| 8 | Revision, formatting, reproducibility check, submission package |

