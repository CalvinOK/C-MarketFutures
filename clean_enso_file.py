from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
LOGDATA_DIR = BASE_DIR / "logdata"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGDATA_DIR.mkdir(parents=True, exist_ok=True)

# Priority:
# 1. already cleaned enso.csv (data/)
# 2. raw NOAA file (data/)
# 3. fallbacks in logdata/
RAW_CANDIDATES = [
    DATA_DIR / "enso.csv",
    DATA_DIR / "nino34.long.anom.csv",
    LOGDATA_DIR / "enso.csv",
    LOGDATA_DIR / "nino34.long.anom.csv",
]

OUTPUT_FILE_LOGDATA = LOGDATA_DIR / "enso.csv"
OUTPUT_FILE_DATA = DATA_DIR / "enso.csv"

MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


# ============================================================
# HELPERS
# ============================================================


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")



def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None



def standardize_date_column(df: pd.DataFrame) -> pd.Series | None:
    for candidate in ["Date", "date", "DATE"]:
        if candidate in df.columns:
            return pd.to_datetime(df[candidate], errors="coerce")
    return None



def build_from_clean_format(df: pd.DataFrame) -> pd.DataFrame:
    date_col = standardize_date_column(df)
    if date_col is None:
        raise ValueError("ENSO file must contain a Date column")

    value_col = None
    for candidate in ["enso_index", "ENSO_INDEX", "anomaly", "anom", "value", "Value"]:
        if candidate in df.columns:
            value_col = candidate
            break

    if value_col is None:
        non_date_cols = [c for c in df.columns if c.lower() != "date"]
        if not non_date_cols:
            raise ValueError("ENSO file has no value column")
        value_col = non_date_cols[0]

    out = pd.DataFrame({
        "Date": date_col,
        "enso_index": safe_numeric(df[value_col]).replace(-99.99, np.nan),
    })
    return out



def build_from_year_month_wide(df: pd.DataFrame) -> pd.DataFrame | None:
    year_col = None
    for candidate in ["YR", "Year", "YEAR", "year"]:
        if candidate in df.columns:
            year_col = candidate
            break

    if year_col is None:
        return None

    month_cols = [col for col in df.columns if str(col).strip().upper()[:3] in MONTH_MAP]
    if not month_cols:
        return None

    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        year = pd.to_numeric(row[year_col], errors="coerce")
        if pd.isna(year):
            continue
        year = int(year)

        for col in month_cols:
            month = MONTH_MAP[str(col).strip().upper()[:3]]
            value = pd.to_numeric(row[col], errors="coerce")
            rows.append(
                {
                    "Date": pd.Timestamp(year=year, month=month, day=1),
                    "enso_index": np.nan if pd.isna(value) or value == -99.99 else float(value),
                }
            )

    if not rows:
        return None

    return pd.DataFrame(rows)



def clean_enso_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = build_from_year_month_wide(df)
    if out is None:
        out = build_from_clean_format(df)

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["enso_index"] = safe_numeric(out["enso_index"]).replace(-99.99, np.nan)

    out = (
        out.dropna(subset=["Date"])
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    out["enso_missing_flag"] = out["enso_index"].isna().astype(int)
    return out


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    raw_path = first_existing(RAW_CANDIDATES)
    if raw_path is None:
        raise FileNotFoundError("Could not find ENSO file in data/ or logdata/")

    print(f"Using ENSO source: {raw_path}")
    df = pd.read_csv(raw_path)
    out = clean_enso_dataframe(df)

    out.to_csv(OUTPUT_FILE_LOGDATA, index=False)
    out.to_csv(OUTPUT_FILE_DATA, index=False)

    print(f"\nSaved cleaned ENSO file -> {OUTPUT_FILE_LOGDATA}")
    print(f"Saved cleaned ENSO file -> {OUTPUT_FILE_DATA}")
    print(f"Rows: {len(out):,}")
    print(f"Date range: {out['Date'].min()} to {out['Date'].max()}")
    print(out.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
