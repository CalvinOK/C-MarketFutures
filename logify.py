from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(".")

FILES = [
    "CoffeeCData.csv",
    "US Soybeans Futures Historical Data.csv",
    "US Sugar #11 Futures Historical Data.csv",
    "USD_BRLT Historical Data.csv",
]

# If True, keep original columns and add log-return columns.
# If False, output a smaller cleaned file with Date, Price, log_return only.
KEEP_EXTRA_COLUMNS = True


# ============================================================
# HELPERS
# ============================================================

def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


def process_market_csv(file_path: Path):
    df = pd.read_csv(file_path)

    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError(f"{file_path.name} must contain 'Date' and 'Price' columns.")

    # Parse date and clean price
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = clean_numeric(df["Price"])

    # Sort oldest to newest for return calculation
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Log price and log return
    df["log_price"] = np.log(df["Price"])
    df["log_return"] = df["log_price"].diff()

    # Optional simple return for comparison
    df["pct_return"] = df["Price"].pct_change()

    if KEEP_EXTRA_COLUMNS:
        out_df = df.copy()
    else:
        out_df = df[["Date", "Price", "log_price", "log_return", "pct_return"]].copy()

    output_file = file_path.with_name(file_path.stem + "_log_returns.csv")
    out_df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")


# ============================================================
# MAIN
# ============================================================

def main():
    for filename in FILES:
        file_path = BASE_DIR / filename
        process_market_csv(file_path)

    print("\nDone.")
    print("Use 'log_return' as the modeling column, not raw Price.")
    print("Do not log-return the climate CSV.")


if __name__ == "__main__":
    main()