
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(".")
LOGDATA_DIR = BASE_DIR / "logdata"
LOGDATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_CANDIDATES = [
    BASE_DIR / "nino34.long.anom.csv",
    LOGDATA_DIR / "nino34.long.anom.csv",
]
OUTPUT_FILE = LOGDATA_DIR / "enso.csv"

def main():
    raw_path = None
    for path in RAW_CANDIDATES:
        if path.exists():
            raw_path = path
            break

    if raw_path is None:
        raise FileNotFoundError("Could not find nino34.long.anom.csv")

    df = pd.read_csv(raw_path)
    if "Date" not in df.columns:
        raise ValueError("ENSO file must contain a Date column")

    non_date_cols = [c for c in df.columns if c != "Date"]
    if not non_date_cols:
        raise ValueError("ENSO file has no value column")

    value_col = non_date_cols[0]
    out = df[["Date", value_col]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce").replace(-99.99, np.nan)
    out = out.rename(columns={value_col: "enso_index"}).dropna(subset=["Date"]).sort_values("Date")

    out.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved cleaned ENSO file: {OUTPUT_FILE}")
    print(out.tail(12).to_string(index=False))

if __name__ == "__main__":
    main()
