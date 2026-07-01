# Literature Gap Confirmation

Search date: 2026-06-30

Working title: Volatility-Aware Fuzzy Time-Series Forecasting of Ghana Cocoa Bean Yield Using Adaptive Intervals and High-Order Rules

## Bottom-Line Decision

The idea should continue, but it must be framed as a reproducible empirical extension, not as a new fuzzy algorithm.

Accessible literature searches did not surface an obvious exact duplicate combining all of these elements:

- Ghana only.
- Cocoa beans only.
- Yield as the forecast target, not production volume.
- Fuzzy time-series forecasting.
- Comparison of Chen equal-interval FTS, adaptive-interval FTS, and high-order FTS.
- Rolling-origin one-step-ahead validation.
- Naive and ARIMA or ETS baselines.
- Separate high-volatility-year evaluation.

The idea is therefore not an obvious direct duplicate based on the accessible search evidence. Its originality is moderate and evaluation-driven. The strongest contribution is the combined study design.

## Search Protocol

Sources used in this first pass:

- OpenAlex live metadata search.
- Targeted web search.
- Project planning notes.
- Study implementation notes.

Search strings used:

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
Ghana cocoa yield fuzzy time series
cocoa bean yield fuzzy time series
Ghana cocoa Chen fuzzy time series
cocoa production fuzzy time series
Ghana cocoa yield forecasting
cocoa production Ghana ARIMA forecasting
adaptive interval fuzzy time series agriculture
high order fuzzy time series crop yield
fuzzy time series agricultural forecasting
fuzzy time series cocoa
```

Limitations:

- This is not yet a final Scopus novelty clearance.
- Scopus and Google Scholar should still be checked manually before submission.
- Crossref API calls from this Windows environment had TLS/time-out issues, so OpenAlex was used as the main reproducible metadata source.

## Duplicate-Risk Screen

| Dimension | Evidence found | Duplicate risk |
|---|---|---|
| Ghana cocoa yield studies | Ghana cocoa yield-gap and agronomic-yield papers exist, especially Aneani and Ofori-Frimpong (2013). | Medium domain overlap, but not FTS forecasting. |
| Cocoa forecasting studies | Cocoa production has been forecast using ARIMA and grey models, including Quartey-Papafio et al. (2020). | High neighboring risk, but production is not yield and the method is not FTS. |
| Cocoa plus fuzzy logic | Fuzzy AHP has been used for cocoa post-harvest technology selection. | Low duplicate risk because it is decision support, not time-series forecasting. |
| Cocoa plus FTS | A 2023 FTS-Markov paper forecasts cocoa plant disease counts in Bendungan district. | High neighboring risk, but the case is disease counts in Indonesia, not Ghana cocoa yield. |
| Agricultural FTS | FTS has been used for rice production, agricultural processes, and palm oil production. | Medium method-domain overlap, but not Ghana cocoa yield. |
| Adaptive interval FTS | Interval-length and partitioning studies exist. | Low duplicate risk; these are method foundations to cite and adapt. |
| High-order FTS | High-order FTS is established from Chen and later work. | Low duplicate risk; this is a method component, not the full case study. |
| Rolling-origin validation plus baselines | Forecast-evaluation literature supports this design. | Low duplicate risk; it strengthens rigor rather than creating overlap. |

## Nearest Studies And Distinctions

1. Quartey-Papafio et al. (2020), "Forecasting cocoa production of six major producers through ARIMA and grey models."

   This is the closest cocoa forecasting paper found in the accessible search. It is not a direct duplicate because it forecasts production, covers six major producers, and uses ARIMA and grey models rather than Ghana-only cocoa yield with fuzzy time-series design comparisons.

2. "Forecasting cocoa yields for 2050" (2018, CGSpace).

   This overlaps with cocoa yield forecasting language, but it appears to be a long-horizon cocoa-yield scenario or domain forecast, not a rolling-origin Chen/adaptive/high-order FTS comparison. It needs manual full-text review before final submission.

3. Salsabila et al. (2023), "Python script fuzzy time series Markov chain model for forecasting the number of diseases cocoa plant in Bendungan district."

   This is the closest cocoa plus FTS paper found. It is not a direct duplicate because it forecasts disease counts in Indonesia, not Ghana cocoa bean yield.

4. Aneani and Ofori-Frimpong (2013), "An Analysis of Yield Gap and Some Factors of Cocoa (Theobroma cacao) Yields in Ghana."

   This is important Ghana cocoa yield background, but it is not a time-series forecasting paper and does not implement fuzzy time-series models.

5. Rice and palm-oil FTS papers.

   These show that fuzzy time-series forecasting has already been used for agricultural production tasks. They are useful as literature anchors, but they do not remove the novelty of the Ghana cocoa yield case.

## Novelty Claim That Is Currently Defensible

Use this claim:

```text
This study contributes a volatility-aware empirical evaluation of fuzzy time-series design choices for Ghana cocoa bean yield forecasting. It compares equal-interval Chen FTS, adaptive-interval Chen FTS, and high-order FTS under rolling-origin validation, and benchmarks the fuzzy models against simple non-fuzzy forecasting references.
```

Do not use these claims:

```text
This is the first cocoa forecasting study.
This is the first fuzzy time-series agricultural forecasting study.
This paper proposes a new fuzzy time-series algorithm.
This proves fuzzy methods are superior for cocoa forecasting.
```

## Method Baseline Versus Original Contribution

This project is framed as a Scopus Q3/Q4-oriented empirical forecasting study. It should still use established fuzzy time-series papers as method baselines, because the manuscript does not propose a new algorithm.

Method baseline:

- Implement the classical Chen FTS workflow from Chen (1996) as the core fuzzy baseline.
- Use Song and Chissom (1993, 1994), Huarng (2001), Chen (2002), Yu (2004), and later FTS surveys to justify the method design choices.

Original contribution:

- Apply the method family to Ghana cocoa bean yield rather than a common enrollment, stock-index, or non-cocoa agricultural dataset.
- Add adaptive interval partitioning.
- Add order-2 and optionally order-3 FTS.
- Add Naive and ARIMA or ETS baselines.
- Use rolling-origin validation instead of a simple static split.
- Add high-volatility-year evaluation.
- Add sensitivity analysis over interval count.

This makes the manuscript an application-and-evaluation contribution rather than a direct copy of an existing FTS paper.

## Final Research Gap Paragraph

Although fuzzy time-series forecasting has been widely studied, the accessible literature search did not identify a direct study that evaluates Chen FTS, adaptive-interval FTS, and high-order FTS for Ghana cocoa bean yield under rolling-origin validation. Existing cocoa-related studies more often address production forecasting, yield gaps, cocoa-sector modelling, disease forecasting, or fuzzy decision support rather than Ghana-only yield forecasting with FTS design comparisons. Agricultural FTS studies exist for other crops such as rice and palm oil, but their datasets, commodities, and evaluation designs differ from the proposed Ghana cocoa yield setting. This study addresses that gap by providing a reproducible, volatility-aware comparison of fuzzy time-series design choices for a short official agricultural yield series, with non-fuzzy benchmarks used to keep the performance claims bounded.

## Go / No-Go Decision

Decision: Go, with corrections.

Completed corrections:

1. The dataset mismatch was fixed. The project now archives and processes cocoa bean yield, not cocoa bean production.
2. The actual cocoa bean yield CSV and metadata are saved under `data/raw/`, and the Ghana-only processed dataset is saved under `data/processed/`.

Remaining check before submission:

1. Repeat exact-phrase searches in Scopus and Google Scholar:

```text
"Ghana cocoa yield" "fuzzy time series"
"Ghana cocoa bean yield" "Chen fuzzy time series"
"cocoa bean yield" "high-order fuzzy time series"
"cocoa yield" "adaptive interval" "fuzzy time series"
"cocoa production" "fuzzy time series"
"cocoa yield" "rolling-origin" forecasting
```

If those searches still do not reveal an exact duplicate, the manuscript can safely present the study as an original application-and-evaluation contribution.
