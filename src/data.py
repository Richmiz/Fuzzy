"""Download, clean, and document the Ghana cocoa bean yield dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
DATA_CARD_PATH = PROJECT_ROOT / "data" / "DATA_CARD.md"

YIELD_CSV_URL = (
    "https://ourworldindata.org/grapher/"
    "cocoa-bean-yields.csv?v=1&csvType=full&useColumnShortNames=false"
)
METADATA_URL = "https://ourworldindata.org/grapher/cocoa-bean-yields.metadata.json"

RAW_CSV_PATH = RAW_DIR / "cocoa-bean-yields.csv"
METADATA_PATH = RAW_DIR / "cocoa-bean-yields.metadata.json"
PROCESSED_CSV_PATH = PROCESSED_DIR / "ghana_cocoa_yield_1961_2024.csv"
INTEGRITY_REPORT_PATH = TABLES_DIR / "data_integrity_report.csv"

EXPECTED_COLUMNS = [
    "year",
    "yield_tonnes_per_hectare",
    "yield_lag1",
    "absolute_change",
    "percentage_change",
    "absolute_percentage_change",
    "is_high_volatility",
]


@dataclass
class MetadataResult:
    available: bool
    error: str = ""
    content: dict[str, Any] | None = None


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, TABLES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, output_path: Path) -> None:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; GhanaCocoaFTS/0.1; "
                "+https://ourworldindata.org/grapher/cocoa-bean-yields)"
            )
        },
    )
    with urlopen(request, timeout=60) as response:
        output_path.write_bytes(response.read())


def download_metadata() -> MetadataResult:
    try:
        download_file(METADATA_URL, METADATA_PATH)
        content = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return MetadataResult(available=True, content=content)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return MetadataResult(available=False, error=str(exc))


def load_raw_data() -> pd.DataFrame:
    try:
        download_file(YIELD_CSV_URL, RAW_CSV_PATH)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Could not download yield CSV from OWID: {exc}") from exc
    return pd.read_csv(RAW_CSV_PATH)


def detect_yield_column(df: pd.DataFrame) -> str:
    identifier_columns = {"Entity", "Code", "Year", "Day"}
    candidates = [
        column
        for column in df.columns
        if column not in identifier_columns and pd.api.types.is_numeric_dtype(df[column])
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one numeric cocoa yield column after excluding "
            f"identifier columns, found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def build_ghana_dataset(raw_df: pd.DataFrame, yield_column: str) -> pd.DataFrame:
    if "Entity" not in raw_df.columns or "Year" not in raw_df.columns:
        raise ValueError("Raw dataset must contain Entity and Year columns.")

    ghana = raw_df.loc[raw_df["Entity"] == "Ghana", ["Year", yield_column]].copy()
    if ghana.empty:
        raise ValueError("No rows found for Entity == 'Ghana'.")

    ghana = ghana.rename(
        columns={"Year": "year", yield_column: "yield_tonnes_per_hectare"}
    )
    ghana["year"] = ghana["year"].astype(int)
    ghana = resolve_duplicate_years(ghana)
    ghana = ghana.sort_values("year").reset_index(drop=True)

    if ghana["yield_tonnes_per_hectare"].isna().any():
        missing_years = ghana.loc[
            ghana["yield_tonnes_per_hectare"].isna(), "year"
        ].tolist()
        raise ValueError(f"Processed Ghana dataset has missing yield values: {missing_years}")

    ghana["yield_lag1"] = ghana["yield_tonnes_per_hectare"].shift(1)
    ghana["absolute_change"] = (
        ghana["yield_tonnes_per_hectare"] - ghana["yield_lag1"]
    )
    ghana["percentage_change"] = (
        100 * ghana["absolute_change"] / ghana["yield_lag1"]
    )
    ghana["absolute_percentage_change"] = ghana["percentage_change"].abs()

    threshold = ghana["absolute_percentage_change"].quantile(0.75)
    ghana["is_high_volatility"] = (
        ghana["absolute_percentage_change"].ge(threshold).fillna(False)
    )

    return ghana[EXPECTED_COLUMNS]


def resolve_duplicate_years(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_mask = df.duplicated(subset=["year"], keep=False)
    if not duplicate_mask.any():
        return df

    duplicated = df.loc[duplicate_mask].sort_values("year")
    conflicting_years = []
    for year, group in duplicated.groupby("year"):
        if group["yield_tonnes_per_hectare"].nunique(dropna=False) > 1:
            conflicting_years.append(int(year))

    if conflicting_years:
        raise ValueError(
            "Conflicting duplicate values found for years: "
            f"{conflicting_years}. Manual review is required."
        )

    return df.drop_duplicates(subset=["year"], keep="first").copy()


def missing_years_for(df: pd.DataFrame) -> list[int]:
    first_year = int(df["year"].min())
    last_year = int(df["year"].max())
    observed = set(df["year"].astype(int).tolist())
    return [year for year in range(first_year, last_year + 1) if year not in observed]


def metadata_unit(metadata: MetadataResult, detected_column: str) -> str:
    if not metadata.available or not metadata.content:
        return "Unable to verify from metadata"

    columns = metadata.content.get("columns", {})
    column_meta = columns.get(detected_column, {})
    if not column_meta and len(columns) == 1:
        column_meta = next(iter(columns.values()))
    unit = column_meta.get("unit")
    if unit:
        return str(unit)
    return "Unable to verify from metadata"


def build_integrity_report(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    detected_column: str,
    metadata: MetadataResult,
) -> pd.DataFrame:
    ghana_raw = raw_df.loc[raw_df["Entity"] == "Ghana"].copy()
    raw_duplicate_year_count = int(ghana_raw.duplicated(subset=["Year"]).sum())
    duplicate_count = int(processed_df.duplicated(subset=["year"]).sum())
    missing_years = missing_years_for(processed_df)

    checks = [
        ("raw_row_count", len(raw_df)),
        ("ghana_row_count", len(ghana_raw)),
        ("first_year", int(processed_df["year"].min())),
        ("last_year", int(processed_df["year"].max())),
        ("duplicate_years_in_ghana_raw", raw_duplicate_year_count),
        ("duplicate_years_after_cleaning", duplicate_count),
        ("missing_year_count", len(missing_years)),
        ("missing_years", ", ".join(str(year) for year in missing_years) or "none"),
        (
            "missing_yield_values",
            int(processed_df["yield_tonnes_per_hectare"].isna().sum()),
        ),
        ("detected_yield_column_name", detected_column),
        ("metadata_available", metadata.available),
        ("metadata_error", metadata.error or "none"),
    ]
    return pd.DataFrame(checks, columns=["check", "value"])


def write_data_card(
    processed_df: pd.DataFrame,
    detected_column: str,
    metadata: MetadataResult,
) -> None:
    retrieval_date = datetime.now().strftime("%Y-%m-%d")
    first_year = int(processed_df["year"].min())
    last_year = int(processed_df["year"].max())
    missing_years = missing_years_for(processed_df)
    unit = metadata_unit(metadata, detected_column)
    threshold = processed_df["absolute_percentage_change"].quantile(0.75)

    content = f"""# Ghana Cocoa Yield Data Card

## Source

- Original provider: Food and Agriculture Organization of the United Nations (FAOSTAT).
- Retrieval route: Our World in Data Grapher.
- Data page: https://ourworldindata.org/grapher/cocoa-bean-yields
- CSV endpoint: {YIELD_CSV_URL}
- Metadata endpoint: {METADATA_URL}
- Retrieval date: {retrieval_date}

## Dataset

- Country: Ghana.
- Indicator: cocoa bean yield.
- Detected source column: `{detected_column}`.
- Project column: `yield_tonnes_per_hectare`.
- Unit: {unit}.
- Frequency: annual.
- Year range: {first_year}-{last_year}.
- Processed observations: {len(processed_df)}.

## Saved Files

- Raw CSV: `data/raw/cocoa-bean-yields.csv`.
- Metadata JSON: `data/raw/cocoa-bean-yields.metadata.json`.
- Processed Ghana dataset: `data/processed/ghana_cocoa_yield_1961_2024.csv`.
- Integrity report: `results/tables/data_integrity_report.csv`.

## Column Mapping

| Source column | Project column |
|---|---|
| `Year` | `year` |
| `{detected_column}` | `yield_tonnes_per_hectare` |

## Derived Columns

- `yield_lag1`: previous year's yield.
- `absolute_change`: current yield minus previous year's yield.
- `percentage_change`: year-to-year percentage change.
- `absolute_percentage_change`: absolute value of `percentage_change`.
- `is_high_volatility`: provisional Phase 3 label using the 75th percentile of full-series absolute percentage change.

## Cleaning Rules

- Filtered the raw OWID file to `Entity == "Ghana"`.
- Sorted annual observations by year.
- Removed exact duplicate years only when duplicate values were identical.
- Stopped execution if duplicate years had conflicting yield values.
- Did not interpolate missing years or missing yield values.

## Integrity Notes

- Missing years: {", ".join(str(year) for year in missing_years) if missing_years else "none"}.
- Missing yield values: {int(processed_df["yield_tonnes_per_hectare"].isna().sum())}.
- Metadata available: {metadata.available}.
- Metadata error: {metadata.error or "none"}.
- Provisional high-volatility threshold: {threshold:.6g}.

## Limitations

- This data card documents Phase 3 dataset preparation only.
- The high-volatility flag is provisional and uses the full Ghana series. The validation phase should recompute volatility labels on the evaluation period only.
- This project forecasts yield only; it does not model production volume, prices, climate variables, or causal drivers.
"""
    DATA_CARD_PATH.write_text(content, encoding="utf-8")


def run() -> None:
    ensure_directories()
    raw_df = load_raw_data()
    metadata = download_metadata()
    detected_column = detect_yield_column(raw_df)
    processed_df = build_ghana_dataset(raw_df, detected_column)

    processed_df.to_csv(PROCESSED_CSV_PATH, index=False)
    report = build_integrity_report(raw_df, processed_df, detected_column, metadata)
    report.to_csv(INTEGRITY_REPORT_PATH, index=False)
    write_data_card(processed_df, detected_column, metadata)

    print("Phase 3 data preparation complete.")
    print(f"Raw CSV: {RAW_CSV_PATH}")
    print(f"Processed CSV: {PROCESSED_CSV_PATH}")
    print(f"Data card: {DATA_CARD_PATH}")
    print(f"Integrity report: {INTEGRITY_REPORT_PATH}")


if __name__ == "__main__":
    run()
