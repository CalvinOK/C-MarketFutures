from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
LOGDATA_DIR = BASE_DIR / "logdata"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COFFEE_FILE = DATA_DIR / "CoffeeCData.csv"
CLIMATE_FILE = LOGDATA_DIR / "coffee_climate_sao_paulo.csv"
SOY_FILE = DATA_DIR / "US Soybeans Futures Historical Data.csv"
SUGAR_FILE = DATA_DIR / "US Sugar #11 Futures Historical Data.csv"
FX_FILE = DATA_DIR / "USD_BRLT Historical Data.csv"

MAX_LAG = 200  # analyze lags from -200 to +200 days

OUTPUT_MERGED = OUTPUT_DIR / "merged_correlation_dataset.csv"
OUTPUT_LEVEL_CORR = OUTPUT_DIR / "correlation_matrix_levels.csv"
OUTPUT_RETURN_CORR = OUTPUT_DIR / "correlation_matrix_returns.csv"
OUTPUT_LAGGED = OUTPUT_DIR / "lagged_correlation_results.csv"
OUTPUT_BEST_LAGS = OUTPUT_DIR / "best_lag_summary.csv"


# ============================================================
# HELPERS
# ============================================================

def clean_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


def load_market_file(file_path: Path, value_name: str) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_csv(file_path)

    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError(f"{file_path.name} must contain Date and Price columns")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = clean_numeric_series(df["Price"])

    # Use Change % if present, otherwise compute from Price
    if "Change %" in df.columns:
        df["Change %"] = clean_numeric_series(df["Change %"])
        df[f"{value_name}_return_pct"] = df["Change %"]
    else:
        df[f"{value_name}_return_pct"] = df["Price"].pct_change() * 100

    df = df.rename(columns={"Price": value_name})
    df = df[["Date", value_name, f"{value_name}_return_pct"]].copy()
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    return df


def load_climate_file(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing climate file: {file_path}")

    df = pd.read_csv(file_path)

    if "Date" not in df.columns:
        raise ValueError("Climate file must contain a Date column")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["tmax", "tmax_change_pct", "tmin", "tmin_change_pct", "rainfall"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("Date").reset_index(drop=True)


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    temp = pd.concat([x, y], axis=1).dropna()
    if len(temp) < 3:
        return np.nan
    return temp.iloc[:, 0].corr(temp.iloc[:, 1])


def lagged_corr(target: pd.Series, feature: pd.Series, lag: int) -> float:
    """
    corr(target_t, feature_{t-lag})

    positive lag  => feature leads target
    negative lag  => feature lags target
    """
    shifted_feature = feature.shift(lag)
    return safe_corr(target, shifted_feature)


def build_lag_table(df: pd.DataFrame, target_col: str, feature_cols: list[str], max_lag: int) -> pd.DataFrame:
    rows = []

    for feature in feature_cols:
        if feature not in df.columns:
            continue

        for lag in range(-max_lag, max_lag + 1):
            joined = pd.concat([df[target_col], df[feature].shift(lag)], axis=1).dropna()
            corr = joined.iloc[:, 0].corr(joined.iloc[:, 1]) if len(joined) >= 3 else np.nan

            rows.append({
                "target": target_col,
                "feature": feature,
                "lag_days": lag,
                "correlation": corr,
                "abs_correlation": abs(corr) if pd.notna(corr) else np.nan,
                "n_obs": len(joined),
            })

    return pd.DataFrame(rows)


def best_lag_summary(lag_df: pd.DataFrame) -> pd.DataFrame:
    lag_df = lag_df.dropna(subset=["correlation"]).copy()
    if lag_df.empty:
        return lag_df

    idx = lag_df.groupby(["target", "feature"])["abs_correlation"].idxmax()
    out = (
        lag_df.loc[idx]
        .sort_values(["target", "abs_correlation"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    coffee = load_market_file(COFFEE_FILE, "coffee_c")
    soy = load_market_file(SOY_FILE, "soybeans")
    sugar = load_market_file(SUGAR_FILE, "sugar")
    fx = load_market_file(FX_FILE, "usd_brl")
    climate = load_climate_file(CLIMATE_FILE)

    # Merge all on Date
    df = coffee.merge(soy, on="Date", how="outer")
    df = df.merge(sugar, on="Date", how="outer")
    df = df.merge(fx, on="Date", how="outer")
    df = df.merge(climate, on="Date", how="outer")
    df = df.sort_values("Date").reset_index(drop=True)

    # Save merged dataset
    df.to_csv(OUTPUT_MERGED, index=False)

    # Correlation matrices
    level_cols = [
        "coffee_c",
        "soybeans",
        "sugar",
        "usd_brl",
        "tmax",
        "tmin",
        "rainfall",
    ]
    level_cols = [c for c in level_cols if c in df.columns]

    return_cols = [
        "coffee_c_return_pct",
        "soybeans_return_pct",
        "sugar_return_pct",
        "usd_brl_return_pct",
        "tmax_change_pct",
        "tmin_change_pct",
    ]
    return_cols = [c for c in return_cols if c in df.columns]

    level_corr = df[level_cols].corr() if level_cols else pd.DataFrame()
    return_corr = df[return_cols].corr() if return_cols else pd.DataFrame()

    level_corr.to_csv(OUTPUT_LEVEL_CORR)
    return_corr.to_csv(OUTPUT_RETURN_CORR)

    # Lagged correlation analysis
    lag_targets = [c for c in ["coffee_c", "coffee_c_return_pct"] if c in df.columns]
    lag_features = [
        "soybeans",
        "soybeans_return_pct",
        "sugar",
        "sugar_return_pct",
        "usd_brl",
        "usd_brl_return_pct",
        "tmax",
        "tmax_change_pct",
        "tmin",
        "tmin_change_pct",
        "rainfall",
    ]
    lag_features = [c for c in lag_features if c in df.columns]

    lag_frames = []
    for target_col in lag_targets:
        lag_frames.append(build_lag_table(df, target_col, lag_features, MAX_LAG))

    lagged_df = pd.concat(lag_frames, ignore_index=True) if lag_frames else pd.DataFrame()
    best_lags_df = best_lag_summary(lagged_df) if not lagged_df.empty else pd.DataFrame()

    lagged_df.to_csv(OUTPUT_LAGGED, index=False)
    best_lags_df.to_csv(OUTPUT_BEST_LAGS, index=False)

    print(f"Saved merged dataset: {OUTPUT_MERGED}")
    print(f"Saved level correlations: {OUTPUT_LEVEL_CORR}")
    print(f"Saved return correlations: {OUTPUT_RETURN_CORR}")
    print(f"Saved lagged correlations: {OUTPUT_LAGGED}")
    print(f"Saved best lags: {OUTPUT_BEST_LAGS}")


if __name__ == "__main__":
    main()