"""
coffee_xgboost_projection_macro_model_b.py
===========================================
MODEL B – Macro Trend-Oriented Coffee Futures Forecasting
----------------------------------------------------------

HOW MODEL B DIFFERS FROM THE CURRENT MODEL (Model A):
------------------------------------------------------
Model A (coffee_xgboost_projection_macro.py):
  - Target: forward *cumulative daily log returns* summed over horizon
  - Features: short-lag-heavy (lags 1, 2, 3, 5, 10, 20d), short rolling windows (5, 10, 20d)
  - Horizons: 1–52 weeks, equal emphasis
  - Prediction shrinkage: 30% at h=1 scaling to ~85% at h=52 (aggressive)
  - Use case: capturing near-term momentum and short-horizon noise

Model B (this file):
  - Target: log price level change from anchor to horizon date
              = log(price_{t+h}) - log(price_t)
              This is equivalent to cumulative log return but computed directly
              from the price series. Cleaner for macro forecasting because it
              explicitly anchors to a stable price level rather than summing noisy
              daily returns. Also more natural for predicting "where will price be
              in 6 months" questions.
  - Features: macro-oriented. Long rolling windows (60/120/252d), cumulative
              changes (30/90/180d), regime states, slow-moving inventory/FX/ENSO
              signals. Very few short lags. Designed for signal persistence.
  - Horizons: 12, 26, 52 weeks emphasized (medium and long term)
  - Prediction shrinkage: minimal – macro models should speak with their own
              confidence at longer horizons. Slight clipping only.
  - Use case: capturing multi-month directional trends

DATA AUDIT (checked against actual project files before writing):
-----------------------------------------------------------------
Confirmed available inputs (from logdata/ and data/):
  - coffee_c, coffee_c_log_return              [daily, continuous]
  - soybeans, soybeans_log_return              [daily, continuous]
  - sugar, sugar_log_return                    [daily, continuous]
  - usd_brl, usd_brl_log_return                [daily, continuous]
  - tmax, tmin, rainfall                        [daily climate, Sao Paulo]
  - enso_index                                  [monthly, forward-filled to daily]
  - drought_index, drought_flag                 [derived daily from climate]
  - frost_flag, frost_severity                  [derived daily from climate]
  - inventory_certified_bags, etc.              [weekly ICE stock, forward-filled]

NOT POSSIBLE from current data (skipped):
  - Term structure / futures curve slope: only spot/nearby price available;
    no spread between deferred contracts
  - COT (Commitments of Traders) net positioning: not in any data file
  - Brazilian production forecast / USDA WASDE supply/demand estimates:
    not in any data file
  - Vietnam/Colombian price differential: no origin-specific price data
  - Ocean freight rates: not in any data file
  - GBP/EUR FX cross rates: only USD/BRL available

Author: Calvin Chen / BOND Great Lakes Coffee
Date:   2026-04-05
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")  # headless – consistent with existing project pattern
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths – mirror the existing project structure exactly
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGDATA_DIR = BASE_DIR / "logdata"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Horizons: Model B emphasises medium and long horizons.
# 4w is included for reference / comparison; 12/26/52 are the focus.
HORIZONS_WEEKS = [4, 12, 26, 52]
BUSINESS_DAYS_PER_WEEK = 5

# Train / test split (same convention as Model A)
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Minimum rows needed before the first test observation
# 252 business days ≈ 1 year; needed so long rolling windows are valid
MIN_TRAIN_ROWS = 400

# XGBoost hyperparameters – tuned for slower, macro-oriented signal.
# Fewer trees, deeper trees, higher regularisation to avoid short-term
# pattern memorisation.
XGBOOST_PARAMS = dict(
    n_estimators=350,
    max_depth=5,           # deeper than Model A (4) to capture regime interactions
    learning_rate=0.02,    # slower learning rate to reduce overfit on short moves
    subsample=0.80,
    colsample_bytree=0.70, # stronger column subsampling for feature diversity
    min_child_weight=5,    # higher than Model A (3); suppresses noise splits
    reg_lambda=2.0,        # heavier L2 than Model A (1.0)
    reg_alpha=0.1,         # L1 regularisation (not used in Model A)
    random_state=RANDOM_STATE,
    verbosity=0,
    n_jobs=-1,
)

# Sample weight half-life for recency weighting (business days).
# Model A uses 730d; here we use 3 years to value older macro regimes
# more than Model A does.
WEIGHT_HALFLIFE_DAYS = 1095  # ~3 years

# Rolling window lengths (in business days) for macro features
MACRO_ROLL_WINDOWS = [20, 60, 120, 252]      # short→1yr
CUMULATIVE_WINDOWS = [30, 90, 180, 252]      # cumulative change windows

# FX and exogenous trend windows
FX_TREND_WINDOWS = [60, 120, 252]            # business days

# Inventory windows (in business days; weekly data so multiples of ~5)
INVENTORY_WINDOWS = [20, 65, 130, 260]       # ~4w, 13w, 26w, 52w

# Minimum observations required to compute z-scores reliably
ZSCORE_MIN_PERIODS = 126

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_numeric(s: pd.Series) -> pd.Series:
    """Coerce to float, replace sentinel -99.99 with NaN."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s != -99.99, other=np.nan)
    return s


def standardise_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise Date column → sort → drop duplicates → set as index."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date")
    df = df.set_index("Date")
    return df


def rolling_zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    if min_periods is None:
        min_periods = max(ZSCORE_MIN_PERIODS, window // 2)
    mu = s.rolling(window, min_periods=min_periods).mean()
    sigma = s.rolling(window, min_periods=min_periods).std()
    return (s - mu) / sigma.replace(0, np.nan)


def log_price_change_target(price: pd.Series, horizon_days: int) -> pd.Series:
    """
    Compute the forward log price level change over horizon_days.

    target_t = log(price_{t + horizon_days}) - log(price_t)

    This is equivalent to the sum of daily log returns over the horizon but
    computed directly from prices, anchoring to a stable level. It is the
    natural macro-forecasting target: "how much will price move from today
    to horizon h?"

    Unlike Model A's cumulative_target() which sums daily log_return forward,
    this is computed from the price series itself to avoid accumulation of
    forward-fill artefacts in log_return on non-trading days.
    """
    log_price = np.log(price)
    return log_price.shift(-horizon_days) - log_price


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_first(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_market_file(name_stem: str, price_col: str, return_col: str) -> pd.DataFrame | None:
    """
    Load a log-return market file from LOGDATA_DIR or DATA_DIR.

    Returns a DataFrame indexed by Date with columns [price_col, return_col],
    or None if no file is found.
    """
    candidates = [
        LOGDATA_DIR / f"{name_stem}_log_returns.csv",
        LOGDATA_DIR / f"{name_stem}.csv",
        DATA_DIR / f"{name_stem}.csv",
    ]
    path = _find_first(candidates)
    if path is None:
        print(f"  [MISSING] {name_stem} – skipping")
        return None

    df = pd.read_csv(path, thousands=",")
    df = standardise_date_index(df)

    # Identify price column
    price_raw = None
    for col in ["Price", "price", "Close", "close"]:
        if col in df.columns:
            price_raw = col
            break
    if price_raw is None:
        print(f"  [WARN] {path.name}: no recognised price column")
        return None

    df[price_col] = safe_numeric(df[price_raw])

    # Compute log return here (don't rely on pre-computed column – may differ)
    df[return_col] = np.log(df[price_col]).diff()
    return df[[price_col, return_col]].copy()


def load_climate_file() -> pd.DataFrame | None:
    """Load Sao Paulo climate data. Returns None if not found."""
    candidates = [
        LOGDATA_DIR / "coffee_climate_sao_paulo.csv",
        DATA_DIR / "coffee_climate_sao_paulo.csv",
    ]
    path = _find_first(candidates)
    if path is None:
        print("  [MISSING] coffee_climate_sao_paulo.csv – climate features unavailable")
        return None

    df = pd.read_csv(path)
    df = standardise_date_index(df)
    for col in ["tmax", "tmin", "rainfall"]:
        if col in df.columns:
            df[col] = safe_numeric(df[col])
    df["tavg"] = (df.get("tmax", np.nan) + df.get("tmin", np.nan)) / 2
    return df


def load_enso_file() -> pd.DataFrame | None:
    """
    Load ENSO index. Handles two formats:
      - Clean: Date + enso_index columns
      - NOAA wide: YR + JAN ... DEC columns
    Returns monthly data or None.
    """
    candidates = [
        LOGDATA_DIR / "enso.csv",
        DATA_DIR / "enso.csv",
        DATA_DIR / "nina34.long.anom.csv",
    ]
    path = _find_first(candidates)
    if path is None:
        print("  [MISSING] enso.csv – ENSO features unavailable")
        return None

    df = pd.read_csv(path)
    # Detect NOAA wide format
    if "YR" in df.columns and "JAN" in df.columns:
        month_cols = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
        month_map = {m: i+1 for i, m in enumerate(month_cols)}
        rows = []
        for _, row in df.iterrows():
            yr = int(row["YR"])
            for m, mnum in month_map.items():
                if m in row:
                    val = safe_numeric(pd.Series([row[m]])).iloc[0]
                    rows.append({"Date": pd.Timestamp(yr, mnum, 1), "enso_index": val})
        df = pd.DataFrame(rows)
    else:
        df = df.rename(columns={c: "Date" for c in df.columns if "date" in c.lower() or c == "Date"})
        df = df.rename(columns={c: "enso_index" for c in df.columns if "enso" in c.lower() and c != "Date"})
    if "enso_index" not in df.columns:
        print("  [WARN] enso file has no enso_index column – skipping")
        return None

    df = standardise_date_index(df)
    df["enso_index"] = safe_numeric(df["enso_index"])
    return df[["enso_index"]]


def load_optional_event_file(stem: str, value_col: str, flag_col: str | None) -> pd.DataFrame | None:
    """Load drought or frost event file. Returns None if not found."""
    candidates = [LOGDATA_DIR / f"{stem}.csv", DATA_DIR / f"{stem}.csv"]
    path = _find_first(candidates)
    if path is None:
        print(f"  [MISSING] {stem}.csv – features unavailable")
        return None

    df = pd.read_csv(path)
    df = standardise_date_index(df)
    keep = [c for c in [value_col, flag_col] if c and c in df.columns]
    if not keep:
        print(f"  [WARN] {stem}.csv: expected columns not found; skipping")
        return None
    for c in keep:
        df[c] = safe_numeric(df[c])
    return df[keep]


def load_inventory_file() -> pd.DataFrame | None:
    """
    Load ICE standardised inventory (weekly). Returns daily forward-filled
    series or None.
    """
    candidates = [
        LOGDATA_DIR / "standardized_inventory.csv",
        DATA_DIR / "standardized_inventory.csv",
    ]
    path = _find_first(candidates)
    if path is None:
        print("  [MISSING] standardized_inventory.csv – inventory features unavailable")
        return None

    df = pd.read_csv(path)
    if "report_date" not in df.columns:
        print("  [WARN] inventory file missing report_date column; skipping")
        return None

    df["Date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Aggregate certified and transition bags
    cert = None
    trans = None
    if "section" in df.columns and "bags" in df.columns:
        df["bags"] = safe_numeric(df["bags"])
        cert_mask = df["section"].str.upper().str.contains("CERTIFIED", na=False) & \
                    ~df["section"].str.upper().str.contains("TRANSITION", na=False)
        trans_mask = df["section"].str.upper().str.contains("TRANSITION", na=False)

        # Warehouse == "Total" or aggregate per date
        def agg_section(mask):
            sub = df[mask].copy()
            # Prefer rows with warehouse == "Total"
            tot = sub[sub["warehouse"].astype(str).str.lower().str.strip() == "total"]
            if tot.empty:
                tot = sub.groupby("Date")["bags"].sum().reset_index()
            else:
                tot = tot.groupby("Date")["bags"].sum().reset_index()
            tot = tot.set_index("Date")["bags"]
            return tot

        cert = agg_section(cert_mask).rename("inventory_certified_bags")
        trans = agg_section(trans_mask).rename("inventory_transition_bags")

    if cert is None:
        print("  [WARN] Could not extract certified bags from inventory; skipping")
        return None

    out = pd.DataFrame({"inventory_certified_bags": cert})
    if trans is not None:
        out["inventory_transition_bags"] = trans
    else:
        out["inventory_transition_bags"] = np.nan

    out["inventory_total_bags"] = out["inventory_certified_bags"].fillna(0) + \
                                   out["inventory_transition_bags"].fillna(0)

    out = out.sort_index()
    out["inventory_available_flag"] = 1  # mark real (not imputed) rows

    # Forward-fill weekly data to daily (max 7 days)
    full_idx = pd.date_range(out.index.min(), out.index.max(), freq="B")
    out = out.reindex(full_idx)
    out["inventory_available_flag"] = out["inventory_available_flag"].fillna(0)
    out = out.ffill(limit=7)
    out.index.name = "Date"
    return out


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_macro_dataset() -> pd.DataFrame:
    """
    Load and merge all available data sources into a single daily DataFrame.
    Prints a data audit summary.
    """
    print("\n=== MODEL B – DATA AUDIT ===")

    # ---- Market data -------------------------------------------------------
    coffee = load_market_file("CoffeeCData", "coffee_c", "coffee_c_log_return")
    if coffee is None:
        sys.exit("ERROR: coffee price data is required and was not found.")

    soybeans = load_market_file("US Soybeans Futures Historical Data", "soybeans", "soybeans_log_return")
    sugar    = load_market_file("US Sugar #11 Futures Historical Data", "sugar", "sugar_log_return")
    fx       = load_market_file("USD_BRLT Historical Data", "usd_brl", "usd_brl_log_return")

    # Merge market data on coffee index (left join, sorted)
    df = coffee.copy()
    for other in [soybeans, sugar, fx]:
        if other is not None:
            df = df.join(other, how="left")

    # Forward-fill market prices up to 5 days (holiday / weekend gaps)
    for col in ["soybeans", "soybeans_log_return", "sugar", "sugar_log_return",
                "usd_brl", "usd_brl_log_return"]:
        if col in df.columns:
            df[col] = df[col].ffill(limit=5)

    print(f"\nMarket data:")
    print(f"  Coffee:    {df['coffee_c'].notna().sum()} obs  "
          f"({df.index.min().date()} – {df.index.max().date()})")
    for asset in ["soybeans", "sugar", "usd_brl"]:
        if asset in df.columns:
            print(f"  {asset:<12}{df[asset].notna().sum()} obs")
        else:
            print(f"  {asset:<12}NOT LOADED")

    # ---- Climate data ------------------------------------------------------
    climate = load_climate_file()
    if climate is not None:
        df = df.join(climate[["tmax", "tmin", "rainfall", "tavg"]], how="left")
        df[["tmax","tmin","rainfall","tavg"]] = df[["tmax","tmin","rainfall","tavg"]].ffill(limit=5)
        print(f"\nClimate:   {df['tmax'].notna().sum()} obs with valid tmax")
    else:
        for col in ["tmax", "tmin", "rainfall", "tavg"]:
            df[col] = np.nan

    # ---- ENSO (monthly → daily) --------------------------------------------
    enso = load_enso_file()
    if enso is not None:
        # Reindex to daily and forward-fill up to 31 days (monthly data)
        df = df.join(enso.reindex(df.index, method="ffill"), how="left")
        # If not aligned, try a merge approach
        if "enso_index" not in df.columns:
            enso_daily = enso.reindex(
                pd.date_range(enso.index.min(), enso.index.max(), freq="D")
            ).ffill(limit=31)
            df = df.join(enso_daily, how="left")
        df["enso_index"] = df["enso_index"].ffill(limit=31)
        print(f"ENSO:      {df['enso_index'].notna().sum()} obs (after daily fwd-fill)")
    else:
        df["enso_index"] = np.nan

    # ---- Drought -----------------------------------------------------------
    drought = load_optional_event_file("brazil_drought", "drought_index", "drought_flag")
    if drought is not None:
        df = df.join(drought, how="left")
        df[drought.columns] = df[drought.columns].ffill(limit=14)
        print(f"Drought:   {df['drought_index'].notna().sum()} obs")
    else:
        df["drought_index"] = np.nan
        df["drought_flag"] = np.nan

    # ---- Frost -------------------------------------------------------------
    frost = load_optional_event_file("brazil_frost", "frost_severity", "frost_flag")
    if frost is not None:
        df = df.join(frost, how="left")
        df[frost.columns] = df[frost.columns].ffill(limit=14)
        if "frost_severity" not in df.columns:
            df["frost_severity"] = np.nan
        if "frost_flag" not in df.columns:
            df["frost_flag"] = np.nan
        print(f"Frost:     {df['frost_severity'].notna().sum()} obs")
    else:
        df["frost_severity"] = np.nan
        df["frost_flag"] = np.nan

    # ---- Inventory ---------------------------------------------------------
    inventory = load_inventory_file()
    if inventory is not None:
        df = df.join(inventory, how="left")
        df["inventory_available_flag"] = df.get("inventory_available_flag", pd.Series(0, index=df.index)).fillna(0)
        print(f"Inventory: {df['inventory_certified_bags'].notna().sum()} obs (weekly → daily)")
    else:
        df["inventory_certified_bags"] = np.nan
        df["inventory_total_bags"] = np.nan
        df["inventory_available_flag"] = 0

    print(f"\nMerged dataset shape before features: {df.shape}")
    print("=== END DATA AUDIT ===\n")
    return df


# ---------------------------------------------------------------------------
# Feature engineering – MACRO ORIENTED
# Model B uses LONGER windows, CUMULATIVE changes, REGIME states, and
# SLOW-MOVING signals. Short lags are deliberately minimal.
# ---------------------------------------------------------------------------

def add_macro_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long-horizon price level and trend features for coffee.

    Differs from Model A by using 60/120/252d windows instead of 5/10/20d.
    Adds cumulative log return over 30/90/180/252d as direct momentum features.
    Adds regime z-scores anchored to multi-year baselines.
    """
    lp = np.log(df["coffee_c"])

    # --- Cumulative log return features (momentum at macro horizons) ---
    for w in CUMULATIVE_WINDOWS:
        # w-day backward-looking cumulative log return
        col = f"coffee_cum_logret_{w}d"
        df[col] = lp - lp.shift(w)

    # --- Rolling mean and std at macro windows ---
    for w in MACRO_ROLL_WINDOWS:
        df[f"coffee_sma_{w}d"] = df["coffee_c"].rolling(w, min_periods=w // 2).mean()
        df[f"coffee_std_{w}d"] = df["coffee_c"].rolling(w, min_periods=w // 2).std()

    # --- Price relative to SMA (trend position) ---
    for w in MACRO_ROLL_WINDOWS:
        sma = df[f"coffee_sma_{w}d"]
        df[f"coffee_vs_sma_{w}d"] = df["coffee_c"] / sma - 1

    # --- Rolling z-score (multi-year) ---
    # 252d z-score: where is price vs. its own 1-year distribution?
    df["coffee_zscore_252d"] = rolling_zscore(df["coffee_c"], 252)
    # 504d z-score: 2-year distribution
    df["coffee_zscore_504d"] = rolling_zscore(df["coffee_c"], 504)

    # --- Distance from 52-week and 2-year high / low ---
    df["coffee_52w_high"] = df["coffee_c"].rolling(252, min_periods=126).max()
    df["coffee_52w_low"]  = df["coffee_c"].rolling(252, min_periods=126).min()
    df["coffee_dist_52w_high"] = df["coffee_c"] / df["coffee_52w_high"] - 1   # negative = below high
    df["coffee_dist_52w_low"]  = df["coffee_c"] / df["coffee_52w_low"]  - 1   # positive = above low

    # --- Volatility at macro windows ---
    for w in [60, 120, 252]:
        df[f"coffee_vol_{w}d"] = (
            df["coffee_c_log_return"].rolling(w, min_periods=w // 2).std() * np.sqrt(252)
        )

    # --- Volatility regime ratio ---
    if "coffee_vol_60d" in df.columns and "coffee_vol_252d" in df.columns:
        df["coffee_vol_regime_60_252"] = df["coffee_vol_60d"] / df["coffee_vol_252d"].replace(0, np.nan)

    # --- Long-term directional consistency ---
    pos = (df["coffee_c_log_return"] > 0).astype(float)
    df["coffee_dir_consistency_60d"]  = pos.rolling(60,  min_periods=30).mean()
    df["coffee_dir_consistency_120d"] = pos.rolling(120, min_periods=60).mean()
    df["coffee_dir_consistency_252d"] = pos.rolling(252, min_periods=126).mean()

    # --- Rolling Sharpe ratio at macro horizons ---
    for w in [60, 120, 252]:
        mu  = df["coffee_c_log_return"].rolling(w, min_periods=w // 2).mean()
        sig = df["coffee_c_log_return"].rolling(w, min_periods=w // 2).std()
        df[f"coffee_sharpe_{w}d"] = (mu / sig.replace(0, np.nan)) * np.sqrt(252)

    # --- Trend regime flags ---
    df["coffee_macro_uptrend"]   = (df["coffee_vs_sma_252d"] > 0.05).astype(float)
    df["coffee_macro_downtrend"] = (df["coffee_vs_sma_252d"] < -0.05).astype(float)

    # --- Price acceleration (difference of medium-term momentum) ---
    ret_60  = df["coffee_cum_logret_30d"]   # short momentum
    ret_252 = df["coffee_cum_logret_180d"]  # long momentum
    df["coffee_momentum_acceleration"] = ret_60 - ret_252 / 3  # ~scale-neutral

    return df


def add_macro_fx_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    BRL/USD macro trend features. Coffee is priced in USD at origin in BRL,
    so persistent BRL weakness (usd_brl rises) lowers Brazilian producer costs
    and can incentivise oversupply → bearish for USD-denominated prices.
    """
    if "usd_brl" not in df.columns or df["usd_brl"].isna().all():
        print("  [INFO] usd_brl not available; skipping FX macro features")
        return df

    lp_fx = np.log(df["usd_brl"])

    for w in FX_TREND_WINDOWS:
        df[f"usd_brl_trend_{w}d"] = lp_fx - lp_fx.shift(w)
        df[f"usd_brl_sma_{w}d"]   = df["usd_brl"].rolling(w, min_periods=w // 2).mean()

    df["usd_brl_zscore_252d"] = rolling_zscore(df["usd_brl"], 252)

    # BRL regime
    df["brl_weakening_regime"]    = (df.get("usd_brl_trend_120d", pd.Series(0.0, index=df.index)) > 0.03).astype(float)
    df["brl_strengthening_regime"] = (df.get("usd_brl_trend_120d", pd.Series(0.0, index=df.index)) < -0.03).astype(float)

    # Cumulative FX momentum
    for w in CUMULATIVE_WINDOWS:
        df[f"usd_brl_cum_logret_{w}d"] = lp_fx - lp_fx.shift(w)

    return df


def add_macro_exog_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long-horizon exogenous soft commodity trends.
    Sugar and soybeans move with broader EM agriculture risk appetite.
    """
    for asset, col in [("soybeans", "soybeans"), ("sugar", "sugar")]:
        price_col = col
        ret_col   = f"{col}_log_return"
        if price_col not in df.columns or df[price_col].isna().all():
            continue
        lp = np.log(df[price_col])
        for w in [60, 120, 252]:
            df[f"{asset}_trend_{w}d"] = lp - lp.shift(w)
        df[f"{asset}_zscore_252d"] = rolling_zscore(df[price_col], 252)
        for w in [60, 120]:
            df[f"{asset}_vol_{w}d"] = (
                df[ret_col].rolling(w, min_periods=w // 2).std() * np.sqrt(252)
            )
    return df


def add_macro_enso_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ENSO macro regime features.
    ENSO operates on multi-month timescales; Model A uses short lags (1, 5, 20d)
    which are mostly noise. Model B uses persistent rolling means and regime
    classification.
    """
    if "enso_index" not in df.columns or df["enso_index"].isna().all():
        print("  [INFO] ENSO index unavailable; skipping ENSO features")
        return df

    enso = df["enso_index"]

    # Rolling means at monthly and seasonal timescales
    df["enso_mean_90d"]  = enso.rolling(90,  min_periods=45).mean()
    df["enso_mean_180d"] = enso.rolling(180, min_periods=90).mean()
    df["enso_mean_365d"] = enso.rolling(365, min_periods=180).mean()

    # Regime classification (canonical ONI thresholds)
    df["enso_el_nino_90d"]  = (df["enso_mean_90d"]  > 0.5).astype(float)
    df["enso_la_nina_90d"]  = (df["enso_mean_90d"]  < -0.5).astype(float)
    df["enso_el_nino_180d"] = (df["enso_mean_180d"] > 0.5).astype(float)
    df["enso_la_nina_180d"] = (df["enso_mean_180d"] < -0.5).astype(float)

    # ENSO trend (is it strengthening?)
    df["enso_trend_90d"] = df["enso_mean_90d"] - df["enso_mean_90d"].shift(45)

    # ENSO z-score (unusual strength of current ENSO state)
    df["enso_zscore_365d"] = rolling_zscore(enso, 365)

    # Phase transition indicator: ENSO crossing zero
    sign_now   = np.sign(df["enso_mean_90d"])
    sign_prior = np.sign(df["enso_mean_90d"].shift(45))
    df["enso_phase_flip"] = (sign_now != sign_prior).astype(float)

    return df


def add_macro_climate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long-horizon climate features for Sao Paulo coffee region.

    Model B uses cumulative/persistent signals (90d, 180d) rather than
    short lags. Drought and frost are especially relevant for multi-month
    price impact.
    """
    has_climate = "tmax" in df.columns and df["tmax"].notna().sum() > 100

    if has_climate:
        rain = df["rainfall"].fillna(0)
        tmax = df["tmax"]
        tmin = df["tmin"]

        # Rainfall accumulation at macro horizons
        for w in [30, 90, 180]:
            df[f"rainfall_cum_{w}d"]   = rain.rolling(w, min_periods=w // 3).sum()
        df["rainfall_deficit_90_180"]  = (
            df["rainfall_cum_90d"] - df["rainfall_cum_180d"] / 2
        )  # negative = drying recently vs past half-year

        # Heat stress persistence
        heat = (tmax > 30).astype(float)
        for w in [30, 90]:
            df[f"heat_stress_frac_{w}d"] = heat.rolling(w, min_periods=w // 3).mean()

        # Frost accumulation (frost_severity > 0 on frost days)
        if "frost_severity" in df.columns:
            fsev = df["frost_severity"].fillna(0)
            for w in [30, 90]:
                df[f"frost_severity_cum_{w}d"] = fsev.rolling(w, min_periods=w // 4).sum()
            df["frost_season"] = (
                (df.index.month.isin([6, 7, 8])).astype(float)
            )

        # Temperature anomaly vs rolling baseline
        for col_name, col in [("tmax", tmax), ("tmin", tmin)]:
            baseline = col.rolling(365, min_periods=180).mean()
            df[f"{col_name}_anomaly_30d"] = (
                col.rolling(30, min_periods=15).mean() - baseline
            )

    # Drought persistence (from derived file – available even without raw climate)
    if "drought_index" in df.columns and df["drought_index"].notna().sum() > 100:
        di = df["drought_index"].fillna(0)
        df["drought_mean_90d"]  = di.rolling(90,  min_periods=45).mean()
        df["drought_mean_180d"] = di.rolling(180, min_periods=90).mean()
        df["drought_regime"]    = (df["drought_mean_90d"] > 0.3).astype(float)

    if "drought_flag" in df.columns:
        df["drought_streak_90d"] = df["drought_flag"].fillna(0).rolling(90, min_periods=30).sum()

    if "frost_flag" in df.columns:
        df["frost_streak_90d"] = df["frost_flag"].fillna(0).rolling(90, min_periods=30).sum()

    return df


def add_macro_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Slow-moving ICE certified inventory features.

    Model A uses 4w/13w/26w windows for signals. Model B also adds the
    full-year (52w) baseline and z-score over the longest available window
    to capture multi-year supply cycle position.
    """
    cert_col = "inventory_certified_bags"
    if cert_col not in df.columns or df[cert_col].isna().all():
        print("  [INFO] Inventory data unavailable; skipping inventory features")
        return df

    avail = df.get("inventory_available_flag", pd.Series(1.0, index=df.index))
    cert  = df[cert_col]

    # Rolling statistics at increasing horizons
    for w in INVENTORY_WINDOWS:
        df[f"inventory_roll_mean_{w}d"] = cert.rolling(w, min_periods=w // 3).mean()
        df[f"inventory_roll_std_{w}d"]  = cert.rolling(w, min_periods=w // 3).std()

    # Z-score relative to each baseline
    for w in INVENTORY_WINDOWS:
        mu  = df[f"inventory_roll_mean_{w}d"]
        sig = df[f"inventory_roll_std_{w}d"]
        df[f"inventory_zscore_{w}d"] = (cert - mu) / sig.replace(0, np.nan)

    # Inventory relative to slow moving average (supply surplus / deficit)
    for w in [130, 260]:
        df[f"inventory_vs_{w}d_mean"] = cert / df[f"inventory_roll_mean_{w}d"].replace(0, np.nan) - 1

    # Inventory trend (direction of change over quarter)
    df["inventory_trend_65d"]  = cert - cert.shift(65)
    df["inventory_trend_130d"] = cert - cert.shift(130)

    # Supply regime flags (only where inventory data is real, not imputed)
    df["inventory_low_supply_130d"]  = (
        (df["inventory_vs_130d_mean"] < -0.10) & (avail == 1)
    ).astype(float)
    df["inventory_high_supply_130d"] = (
        (df["inventory_vs_130d_mean"] > 0.10) & (avail == 1)
    ).astype(float)

    # Total supply (certified + transition)
    if "inventory_total_bags" in df.columns:
        tot = df["inventory_total_bags"]
        df["inventory_total_roll_mean_260d"] = tot.rolling(260, min_periods=130).mean()
        df["inventory_total_vs_260d_mean"]   = (
            tot / df["inventory_total_roll_mean_260d"].replace(0, np.nan) - 1
        )
        df["inventory_total_zscore_260d"] = rolling_zscore(tot, 260, min_periods=130)

    # Brazil share trend
    if "inventory_brazil_share" in df.columns:
        bs = df["inventory_brazil_share"]
        df["brazil_share_roll_mean_130d"] = bs.rolling(130, min_periods=65).mean()
        df["brazil_share_trend_65d"] = bs - bs.shift(65)

    return df


def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar and harvest cycle features.
    These are slow by nature – they provide the macro seasonal backdrop.
    """
    m = df.index.month
    w = df.index.isocalendar().week.astype(int)
    q = df.index.quarter

    # Circular encoding
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    df["week_sin"]  = np.sin(2 * np.pi * w / 52)
    df["week_cos"]  = np.cos(2 * np.pi * w / 52)

    # Raw calendar
    df["month"]   = m
    df["quarter"] = q

    # Brazilian coffee calendar
    # Flowering:   Sep–Nov (affects next year's crop)
    # Harvest:     May–Sep (affects current year supply)
    # Off-season:  Oct–Apr (inventory draws / demand season)
    df["brazil_flowering"]   = m.isin([9, 10, 11]).astype(float)
    df["brazil_harvest"]     = m.isin([5, 6, 7, 8, 9]).astype(float)
    df["brazil_off_season"]  = m.isin([10, 11, 12, 1, 2, 3, 4]).astype(float)

    # Northern hemisphere winter (Northern roaster demand peak)
    df["nh_demand_peak"]     = m.isin([10, 11, 12, 1, 2]).astype(float)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Macro-level interaction features between regimes and fundamentals.
    Only constructed when the underlying columns exist.
    """
    # Drought + harvest season → acute supply risk
    if "drought_regime" in df.columns and "brazil_harvest" in df.columns:
        df["drought_during_harvest"] = df["drought_regime"] * df["brazil_harvest"]

    # La Niña + flowering → increased frost/drought risk (historically bearish yield)
    if "enso_la_nina_90d" in df.columns and "brazil_flowering" in df.columns:
        df["la_nina_during_flowering"] = df["enso_la_nina_90d"] * df["brazil_flowering"]

    # BRL weakening + low inventory → reduced supply incentive despite low cost
    if "brl_weakening_regime" in df.columns and "inventory_low_supply_130d" in df.columns:
        df["brl_weak_and_low_supply"] = df["brl_weakening_regime"] * df["inventory_low_supply_130d"]

    # Macro uptrend + harvest → seasonal pressure on sustained rally
    if "coffee_macro_uptrend" in df.columns and "brazil_harvest" in df.columns:
        df["uptrend_during_harvest"] = df["coffee_macro_uptrend"] * df["brazil_harvest"]

    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all macro feature engineering steps in order."""
    df = add_macro_price_features(df)
    df = add_macro_fx_features(df)
    df = add_macro_exog_features(df)
    df = add_macro_enso_features(df)
    df = add_macro_climate_features(df)
    df = add_macro_inventory_features(df)
    df = add_seasonality_features(df)
    df = add_interaction_features(df)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add forward log price level change targets for each horizon.

    Target: log(price_{t + h_days}) - log(price_t)
    (See module docstring for why this is preferred for Model B)
    """
    price = df["coffee_c"]
    for h_weeks in HORIZONS_WEEKS:
        h_days = h_weeks * BUSINESS_DAYS_PER_WEEK
        col = f"target_log_price_change_{h_weeks}w"
        df[col] = log_price_change_target(price, h_days)
    return df


# ---------------------------------------------------------------------------
# Feature column selector
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return all engineered feature columns present in df.
    Excludes raw price/return columns and target columns.
    """
    exclude_prefixes = ("target_", "coffee_c", "soybeans", "sugar", "usd_brl")
    exclude_exact    = {
        "coffee_c", "coffee_c_log_return",
        "soybeans", "soybeans_log_return",
        "sugar", "sugar_log_return",
        "usd_brl", "usd_brl_log_return",
        "tmax", "tmin", "rainfall", "tavg",
        "enso_index",
        "drought_index", "drought_flag",
        "frost_severity", "frost_flag",
        "inventory_certified_bags", "inventory_transition_bags",
        "inventory_total_bags", "inventory_available_flag",
    }
    features = []
    for col in df.columns:
        if col in exclude_exact:
            continue
        if col.startswith("target_"):
            continue
        # Keep engineered columns (they don't start with raw names, or they do
        # but have a suffix like _trend_, _zscore_, etc.)
        features.append(col)
    return features


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------

def compute_sample_weights(train_df: pd.DataFrame) -> np.ndarray:
    """
    Exponential recency weighting with a 3-year half-life.

    Model B values older macro regimes more than Model A (730d half-life),
    using 1095d (3 years) to give meaningful weight to full commodity cycles.
    """
    n = len(train_df)
    days_ago = np.arange(n - 1, -1, -1, dtype=float)
    weights  = np.exp(-days_ago / WEIGHT_HALFLIFE_DAYS)
    weights  = weights / weights.mean()  # normalise so XGBoost scale is stable
    return weights


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute metrics suited for macro trend capture.

    In addition to RMSE / MAE / R², Model B reports:
      - directional_accuracy: fraction of correct sign predictions
      - pearson_r:            linear correlation (trend co-movement)
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 5:
        return dict(n_obs=len(y_true), rmse=np.nan, mae=np.nan, r2=np.nan,
                    directional_accuracy=np.nan, pearson_r=np.nan)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 2 else np.nan

    return dict(n_obs=len(y_true), rmse=rmse, mae=mae, r2=r2,
                directional_accuracy=dir_acc, pearson_r=corr)


def train_model_b(df: pd.DataFrame, horizon_weeks: int,
                  feature_cols: list[str]) -> tuple:
    """
    Train an XGBoost model for a single horizon.

    Returns (model, test_predictions_df, train_metrics, test_metrics).
    """
    h_days  = horizon_weeks * BUSINESS_DAYS_PER_WEEK
    tgt_col = f"target_log_price_change_{horizon_weeks}w"

    if tgt_col not in df.columns:
        raise ValueError(f"Target column {tgt_col} not found in dataset")

    # Drop rows where we can't know target (future rows) or features are all-NaN
    work = df[feature_cols + [tgt_col, "coffee_c"]].copy()
    work = work.dropna(subset=[tgt_col])

    # Drop rows with >50% missing features (overly sparse rows)
    feature_missing = work[feature_cols].isna().mean(axis=1)
    work = work[feature_missing < 0.5]

    n = len(work)
    split_idx = int(n * (1 - TEST_SIZE))
    split_idx = max(split_idx, MIN_TRAIN_ROWS)
    if split_idx >= n:
        print(f"  [WARN] Horizon {horizon_weeks}w: insufficient data for train/test split")
        return None, None, {}, {}

    train = work.iloc[:split_idx]
    test  = work.iloc[split_idx:]

    X_train = train[feature_cols].fillna(0)
    y_train = train[tgt_col].values
    X_test  = test[feature_cols].fillna(0)
    y_test  = test[tgt_col].values

    weights = compute_sample_weights(train)

    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, sample_weight=weights)

    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    # Mild clipping: clip to 2nd–98th percentile of training targets
    # (less aggressive than Model A's 5th–95th, reflecting macro forecast confidence)
    lo, hi = np.nanpercentile(y_train, 2), np.nanpercentile(y_train, 98)
    y_test_pred  = np.clip(y_test_pred,  lo, hi)
    y_train_pred = np.clip(y_train_pred, lo, hi)

    train_metrics = {**evaluate(y_train, y_train_pred), "horizon_weeks": horizon_weeks,
                     "split": "train"}
    test_metrics  = {**evaluate(y_test,  y_test_pred),  "horizon_weeks": horizon_weeks,
                     "split": "test"}

    pred_df = test[["coffee_c"]].copy()
    pred_df["horizon_weeks"]    = horizon_weeks
    pred_df["actual_target"]    = y_test
    pred_df["predicted_target"] = y_test_pred
    pred_df["direction_match"]  = (np.sign(y_test) == np.sign(y_test_pred)).astype(int)

    return model, pred_df, train_metrics, test_metrics


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def walk_forward_backtest(df: pd.DataFrame, feature_cols: list[str],
                          horizon_weeks: int,
                          n_test_points: int = 30,
                          step_weeks: int = 4) -> pd.DataFrame:
    """
    Rolling walk-forward backtest for a single horizon.

    At each step:
      - Train on all data up to the step boundary
      - Predict one horizon window ahead

    Returns a DataFrame of predictions with actual and predicted targets.
    """
    tgt_col = f"target_log_price_change_{horizon_weeks}w"
    if tgt_col not in df.columns:
        return pd.DataFrame()

    work = df[feature_cols + [tgt_col, "coffee_c"]].copy()
    work = work.dropna(subset=[tgt_col])
    feature_missing = work[feature_cols].isna().mean(axis=1)
    work = work[feature_missing < 0.5]

    n = len(work)
    step_days = step_weeks * BUSINESS_DAYS_PER_WEEK
    results   = []

    # Start the walk-forward window at MIN_TRAIN_ROWS
    test_start = MIN_TRAIN_ROWS
    step_idx   = 0

    while test_start + step_idx + step_days <= n:
        train_end = test_start + step_idx
        test_end  = min(train_end + step_days, n)

        if test_end - train_end < 1:
            break

        train = work.iloc[:train_end]
        test  = work.iloc[train_end:test_end]

        X_train = train[feature_cols].fillna(0)
        y_train = train[tgt_col].values
        X_test  = test[feature_cols].fillna(0)
        y_test  = test[tgt_col].values

        weights = compute_sample_weights(train)
        model   = XGBRegressor(**XGBOOST_PARAMS)
        model.fit(X_train, y_train, sample_weight=weights)

        lo, hi        = np.nanpercentile(y_train, 2), np.nanpercentile(y_train, 98)
        y_pred_clipped = np.clip(model.predict(X_test), lo, hi)

        for i, (idx, row) in enumerate(test.iterrows()):
            results.append({
                "forecast_date":     idx,
                "horizon_weeks":     horizon_weeks,
                "actual_target":     y_test[i],
                "predicted_target":  y_pred_clipped[i],
                "direction_match":   int(np.sign(y_test[i]) == np.sign(y_pred_clipped[i])),
                "train_end_date":    work.index[train_end - 1],
                "step_idx":          step_idx,
            })

        step_idx += step_days
        if step_idx >= n_test_points * step_days:
            break

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

def generate_forecast(df: pd.DataFrame, models: dict, feature_cols: list[str],
                      anchor_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Generate forward price-level projections using trained models.

    Returns a DataFrame with columns:
      anchor_date, forecast_date, horizon_weeks,
      anchor_price, predicted_log_change, projected_price, projected_pct_change
    """
    if anchor_date is None:
        anchor_date = df.index.max()

    # Use last available feature row at or before anchor
    available = df.loc[df.index <= anchor_date]
    if available.empty:
        print("[WARN] No data at or before anchor date; cannot generate forecast")
        return pd.DataFrame()

    last_row    = available.iloc[[-1]]
    anchor_price = last_row["coffee_c"].iloc[0]
    feature_row  = last_row[feature_cols].fillna(0)

    rows = []
    for h_weeks, model in models.items():
        if model is None:
            continue
        pred_log_change = float(model.predict(feature_row)[0])
        projected_price = anchor_price * np.exp(pred_log_change)
        forecast_date   = anchor_date + pd.offsets.BDay(h_weeks * BUSINESS_DAYS_PER_WEEK)
        rows.append({
            "anchor_date":           anchor_date.date(),
            "model_asof_date":       last_row.index[0].date(),
            "forecast_date":         forecast_date.date(),
            "horizon_weeks":         h_weeks,
            "anchor_price":          anchor_price,
            "predicted_log_change":  pred_log_change,
            "projected_price":       projected_price,
            "projected_pct_change":  projected_price / anchor_price - 1,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame,
                  anchor_date: pd.Timestamp, out_path: Path) -> None:
    """
    Plot 2-year price history + forward projections at each horizon.
    Follows the same 2-panel convention as the existing project.
    """
    history_start = anchor_date - pd.DateOffset(years=2)
    hist = df.loc[(df.index >= history_start) & (df.index <= anchor_date), "coffee_c"].dropna()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    ax1, ax2  = axes

    # --- Top panel: price history + forecasts ---
    ax1.plot(hist.index, hist.values, color="steelblue", linewidth=1.5, label="Coffee C (history)")
    ax1.axvline(anchor_date, color="black", linestyle="--", linewidth=0.8, label="Anchor date")

    colors = {4: "goldenrod", 12: "darkorange", 26: "tomato", 52: "firebrick"}
    anchor_price = hist.iloc[-1] if not hist.empty else 1.0

    if not forecast_df.empty:
        fdf = forecast_df.sort_values("horizon_weeks")
        # Build line: anchor point + all forecast points
        line_dates  = [anchor_date] + [pd.Timestamp(r["forecast_date"]) for _, r in fdf.iterrows()]
        line_prices = [anchor_price] + list(fdf["projected_price"])
        ax1.plot(line_dates, line_prices, color="firebrick", linewidth=1.8,
                 linestyle="--", zorder=4, label="Projection")
        for _, row in fdf.iterrows():
            h = int(row["horizon_weeks"])
            fd = pd.Timestamp(row["forecast_date"])
            ax1.scatter(fd, row["projected_price"], color=colors.get(h, "grey"),
                        s=80, zorder=5, label=f"h={h}w")
            ax1.annotate(f"{row['projected_pct_change']:+.1%}",
                         xy=(fd, row["projected_price"]),
                         xytext=(5, 5), textcoords="offset points",
                         fontsize=7, color=colors.get(h, "grey"))

    ax1.set_ylabel("Price (cents/lb)")
    ax1.set_title("Model B – Macro Trend Forecast (Coffee C Futures)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Bottom panel: percentage change from anchor ---
    pct_hist = (hist / anchor_price - 1) * 100
    ax2.plot(hist.index, pct_hist.values, color="steelblue", linewidth=1.5)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.axvline(anchor_date, color="black", linestyle="--", linewidth=0.8)

    if not forecast_df.empty:
        fdf = forecast_df.sort_values("horizon_weeks")
        line_dates_pct  = [anchor_date] + [pd.Timestamp(r["forecast_date"]) for _, r in fdf.iterrows()]
        line_pcts       = [0.0] + list(fdf["projected_pct_change"] * 100)
        ax2.plot(line_dates_pct, line_pcts, color="firebrick", linewidth=1.8,
                 linestyle="--", zorder=4)
        for _, row in fdf.iterrows():
            h = int(row["horizon_weeks"])
            fd = pd.Timestamp(row["forecast_date"])
            ax2.scatter(fd, row["projected_pct_change"] * 100,
                        color=colors.get(h, "grey"), s=80, zorder=5)

    ax2.set_ylabel("% Change from Anchor")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot → {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("MODEL B – MACRO TREND-ORIENTED COFFEE FUTURES FORECAST")
    print("=" * 60)

    # ---- 1. Build dataset --------------------------------------------------
    df = build_macro_dataset()
    df = build_feature_set(df)
    df = add_targets(df)

    feature_cols = get_feature_columns(df)
    # Restrict to columns that are numeric and have >20% non-NaN coverage
    valid_features = [
        c for c in feature_cols
        if c in df.columns
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().mean() > 0.20
    ]
    print(f"\nFeature set: {len(valid_features)} valid features (from {len(feature_cols)} engineered)")

    # Save full dataset
    dataset_path = OUTPUT_DIR / "coffee_feature_dataset_macro_model_b.csv"
    df.to_csv(dataset_path)
    print(f"Full dataset saved → {dataset_path.name} ({df.shape})")

    # ---- 2. Train per-horizon models ---------------------------------------
    print("\n--- Training per-horizon models ---")
    trained_models = {}
    all_train_metrics = []
    all_test_metrics  = []
    all_test_preds    = []

    for h_weeks in HORIZONS_WEEKS:
        print(f"\n  Horizon {h_weeks}w ({h_weeks * BUSINESS_DAYS_PER_WEEK} business days):")
        model, pred_df, train_m, test_m = train_model_b(df, h_weeks, valid_features)

        if model is None:
            print(f"    [SKIP] Not enough data for horizon {h_weeks}w")
            continue

        trained_models[h_weeks] = model

        print(f"    Train → RMSE={train_m['rmse']:.4f}  R²={train_m['r2']:.3f}  "
              f"DirAcc={train_m['directional_accuracy']:.3f}  n={train_m['n_obs']}")
        print(f"    Test  → RMSE={test_m['rmse']:.4f}  R²={test_m['r2']:.3f}  "
              f"DirAcc={test_m['directional_accuracy']:.3f}  "
              f"Corr={test_m['pearson_r']:.3f}  n={test_m['n_obs']}")

        all_train_metrics.append(train_m)
        all_test_metrics.append(test_m)
        if pred_df is not None:
            all_test_preds.append(pred_df)

    # ---- 3. Walk-forward backtest ------------------------------------------
    print("\n--- Walk-forward backtests (medium/long horizons) ---")
    backtest_results = []
    backtest_horizons = [h for h in HORIZONS_WEEKS if h >= 12]

    for h_weeks in backtest_horizons:
        print(f"  Backtesting h={h_weeks}w ...", end=" ", flush=True)
        bt = walk_forward_backtest(df, valid_features, h_weeks,
                                   n_test_points=24, step_weeks=4)
        if bt.empty:
            print("no results")
            continue

        m = evaluate(bt["actual_target"].values, bt["predicted_target"].values)
        print(f"RMSE={m['rmse']:.4f}  R²={m['r2']:.3f}  "
              f"DirAcc={m['directional_accuracy']:.3f}  "
              f"Corr={m['pearson_r']:.3f}  n={m['n_obs']}")
        bt["backtest_type"] = "walk_forward"
        backtest_results.append(bt)

    # ---- 4. Save outputs ---------------------------------------------------
    print("\n--- Saving outputs ---")

    # Metrics
    metrics_rows = []
    for m in all_test_metrics:
        metrics_rows.append({**m, "model": "model_b_xgboost"})
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = OUTPUT_DIR / "model_b_metrics_by_horizon.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Metrics → {metrics_path.name}")

    # Train metrics
    train_metrics_df = pd.DataFrame(all_train_metrics)
    train_metrics_df["model"] = "model_b_xgboost"
    train_metrics_df.to_csv(OUTPUT_DIR / "model_b_train_metrics.csv", index=False)

    # Test predictions
    if all_test_preds:
        preds_df = pd.concat(all_test_preds, ignore_index=False)
        preds_path = OUTPUT_DIR / "model_b_test_predictions.csv"
        preds_df.to_csv(preds_path)
        print(f"  Test predictions → {preds_path.name}")

    # Backtest results
    if backtest_results:
        bt_df = pd.concat(backtest_results, ignore_index=True)
        bt_path = OUTPUT_DIR / "model_b_backtest_predictions.csv"
        bt_df.to_csv(bt_path, index=False)
        print(f"  Backtest predictions → {bt_path.name}")

        # Backtest metrics summary
        bt_metrics_rows = []
        for h_weeks in backtest_horizons:
            sub = bt_df[bt_df["horizon_weeks"] == h_weeks]
            if sub.empty:
                continue
            m = evaluate(sub["actual_target"].values, sub["predicted_target"].values)
            bt_metrics_rows.append({**m, "horizon_weeks": h_weeks, "model": "model_b_walkforward"})
        if bt_metrics_rows:
            bt_metrics_df = pd.DataFrame(bt_metrics_rows)
            bt_metrics_path = OUTPUT_DIR / "model_b_backtest_metrics.csv"
            bt_metrics_df.to_csv(bt_metrics_path, index=False)
            print(f"  Backtest metrics → {bt_metrics_path.name}")

    # Feature importance for 26-week horizon (macro focus)
    if 26 in trained_models:
        importances = trained_models[26].feature_importances_
        fi_df = pd.DataFrame({
            "feature":    valid_features,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        fi_path = OUTPUT_DIR / "model_b_feature_importance_h26w.csv"
        fi_df.to_csv(fi_path, index=False)
        print(f"  Feature importance (h=26w) → {fi_path.name}")
        print(f"\n  Top 15 features at h=26w:")
        print(fi_df.head(15).to_string(index=False))

    # ---- 5. Forecast -------------------------------------------------------
    print("\n--- Generating forward forecast ---")
    anchor_date = df["coffee_c"].dropna().index.max()
    forecast_df = generate_forecast(df, trained_models, valid_features, anchor_date)

    if not forecast_df.empty:
        fc_path = OUTPUT_DIR / "model_b_forecast_output.csv"
        forecast_df.to_csv(fc_path, index=False)
        print(f"  Forecast → {fc_path.name}")
        print(f"\n  Anchor: {anchor_date.date()}  |  "
              f"Price: {forecast_df['anchor_price'].iloc[0]:.2f} cents/lb")
        print(forecast_df[["horizon_weeks","forecast_date","projected_price",
                            "projected_pct_change","predicted_log_change"]].to_string(index=False))

    # ---- 6. Plot -----------------------------------------------------------
    print("\n--- Plotting ---")
    plot_path = OUTPUT_DIR / "model_b_projection_plot.png"
    try:
        plot_forecast(df, forecast_df, anchor_date, plot_path)
    except Exception as e:
        print(f"  [WARN] Plot failed: {e}")

    print("\n=== Model B complete ===")
    print(f"All outputs in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
