from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(".")

COFFEE_FILE = BASE_DIR / "CoffeeCData.csv"
CLIMATE_FILE = BASE_DIR / "coffee_climate_sao_paulo.csv"
SOY_FILE = BASE_DIR / "US Soybeans Futures Historical Data.csv"
SUGAR_FILE = BASE_DIR / "US Sugar #11 Futures Historical Data.csv"
FX_FILE = BASE_DIR / "USD_BRLT Historical Data.csv"

MAX_LAG = 200  # analyze lags from -30 to +30 trading/calendar days

OUTPUT_MERGED = BASE_DIR / "merged_correlation_dataset.csv"
OUTPUT_LEVEL_CORR = BASE_DIR / "correlation_matrix_levels.csv"
OUTPUT_RETURN_CORR = BASE_DIR / "correlation_matrix_returns.csv"
OUTPUT_LAGGED = BASE_DIR / "lagged_correlation_results.csv"
OUTPUT_BEST_LAGS = BASE_DIR / "best_lag_summary.csv"


# ============================================================
# HELPERS
# ============================================================

def clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Clean numeric strings like:
    '1,163.50', '15.29', '-0.81%', etc.
    """
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


def load_market_file(file_path: Path, value_name: str) -> pd.DataFrame:
    """
    Load futures/FX files from Investing.com-style exports.
    Keeps Date, Price, Change %.
    """
    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = clean_numeric_series(df["Price"])

    # If Change % already exists, use it; otherwise compute from Price
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
    """
    Load climate file produced earlier.
    Expected columns:
    Date, tmax, tmax_change_pct, tmin, tmin_change_pct, rainfall
    """
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Ensure numeric
    for col in ["tmax", "tmax_change_pct", "tmin", "tmin_change_pct", "rainfall"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create rainfall return/change only if user wants it later.
    # For now we keep rainfall raw only.
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    """
    Pearson correlation after dropping NaNs.
    """
    temp = pd.concat([x, y], axis=1).dropna()
    if len(temp) < 3:
        return np.nan
    return temp.iloc[:, 0].corr(temp.iloc[:, 1])


def lagged_corr(target: pd.Series, feature: pd.Series, lag: int) -> float:
    """
    Correlation between target and lagged feature.

    Convention:
    corr(target_t, feature_{t-lag})

    So:
    - positive lag => feature leads target
      Example lag=5 means compare coffee_t with feature shifted 5 days earlier.
    - negative lag => feature lags target
    """
    shifted_feature = feature.shift(lag)
    return safe_corr(target, shifted_feature)


def build_lag_table(df: pd.DataFrame, target_col: str, feature_cols: list[str], max_lag: int) -> pd.DataFrame:
    rows = []

    for feature in feature_cols:
        for lag in range(-max_lag, max_lag + 1):
            corr = lagged_corr(df[target_col], df[feature], lag)
            rows.append({
                "target": target_col,
                "feature": feature,
                "lag_days": lag,
                "correlation": corr,
                "abs_correlation": abs(corr) if pd.notna(corr) else np.nan,
                "n_obs": len(pd.concat([df[target_col], df[feature].shift(lag)], axis=1).dropna())
            })

    return pd.DataFrame(rows)


def best_lag_summary(lag_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each feature, return the lag with the strongest absolute correlation.
    """
    lag_df = lag_df.dropna(subset=["correlation"]).copy()
    if lag_df.empty:
        return lag_df

    idx = lag_df.groupby(["target", "feature"])["abs_correlation"].idxmax()
    out = lag_df.loc[idx].sort_values(["target", "abs_correlation"], ascending=[True, False]).reset_index(drop=True)
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    # ---------------------------
    # Load data
    # ---------------------------
    coffee = load_market_file(COFFEE_FILE, "coffee_c")
    soy = load_market_file(SOY_FILE, "soybeans")
    sugar = load_market_file(SUGAR_FILE, "sugar")
    fx = load_market_file(FX_FILE, "usd_brl")
    climate = load_climate_file(CLIMATE_FILE)

    # ---------------------------
    # Merge all on Date
    # ---------------------------
    df = coffee.merge(soy, on="Date", how="outer")
    df = df.merge(sugar, on="Date", how="outer")
    df = df.merge(fx, on="Date", how="outer")
    df = df.merge(climate, on="Date", how="outer")

    df = df.sort_values("Date").reset_index(drop=True)

    # ---------------------------
    # Save merged dataset
    # ---------------------------
    df.to_csv(OUTPUT_MERGED, index=False)

    # ---------------------------
    # Correlation matrices
    # ---------------------------
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
        # rainfall change intentionally omitted
    ]
    return_cols = [c for c in return_cols if c in df.columns]

    level_corr = df[level_cols].corr()
    return_corr = df[return_cols].corr()

    level_corr.to_csv(OUTPUT_LEVEL_CORR)
    return_corr.to_csv(OUTPUT_RETURN_CORR)

    # ---------------------------
    # Lag analysis
    # ---------------------------
    level_features = [
        "soybeans",
        "sugar",
        "usd_brl",
        "tmax",
        "tmin",
        "rainfall",
    ]
    level_features = [c for c in level_features if c in df.columns]

    return_features = [
        "soybeans_return_pct",
        "sugar_return_pct",
        "usd_brl_return_pct",
        "tmax_change_pct",
        "tmin_change_pct",
    ]
    return_features = [c for c in return_features if c in df.columns]

    lagged_levels = build_lag_table(
        df=df,
        target_col="coffee_c",
        feature_cols=level_features,
        max_lag=MAX_LAG
    )

    lagged_returns = build_lag_table(
        df=df,
        target_col="coffee_c_return_pct",
        feature_cols=return_features,
        max_lag=MAX_LAG
    )

    lagged_levels["analysis_type"] = "levels"
    lagged_returns["analysis_type"] = "returns"

    lagged_all = pd.concat([lagged_levels, lagged_returns], ignore_index=True)
    lagged_all = lagged_all[
        ["analysis_type", "target", "feature", "lag_days", "correlation", "abs_correlation", "n_obs"]
    ].sort_values(["analysis_type", "feature", "lag_days"])

    lagged_all.to_csv(OUTPUT_LAGGED, index=False)

    # ---------------------------
    # Best lag summary
    # ---------------------------
    best_levels = best_lag_summary(lagged_levels)
    best_returns = best_lag_summary(lagged_returns)

    if not best_levels.empty:
        best_levels["analysis_type"] = "levels"
    if not best_returns.empty:
        best_returns["analysis_type"] = "returns"

    best_all = pd.concat([best_levels, best_returns], ignore_index=True)
    best_all = best_all[
        ["analysis_type", "target", "feature", "lag_days", "correlation", "abs_correlation", "n_obs"]
    ].sort_values(["analysis_type", "abs_correlation"], ascending=[True, False])

    best_all.to_csv(OUTPUT_BEST_LAGS, index=False)

    # ---------------------------
    # Print summary to console
    # ---------------------------
    print("\nSaved files:")
    print(f"- {OUTPUT_MERGED}")
    print(f"- {OUTPUT_LEVEL_CORR}")
    print(f"- {OUTPUT_RETURN_CORR}")
    print(f"- {OUTPUT_LAGGED}")
    print(f"- {OUTPUT_BEST_LAGS}")

    print("\nDate coverage:")
    for name, subdf in {
        "coffee": coffee,
        "soybeans": soy,
        "sugar": sugar,
        "usd_brl": fx,
        "climate": climate
    }.items():
        print(f"{name:10s}: {subdf['Date'].min().date()} to {subdf['Date'].max().date()}  ({len(subdf):,} rows)")

    print("\nTop best lags:")
    print(best_all.head(15).to_string(index=False))


if __name__ == "__main__":
    main()