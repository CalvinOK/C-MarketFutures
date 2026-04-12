from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = OUTPUT_DIR / "coffee_model_dataset_merged.csv"
METRICS_FILE = OUTPUT_DIR / "coffee_xgb_proj4_metrics.csv"
WF_PREDS_FILE = OUTPUT_DIR / "coffee_xgb_proj4_walkforward_predictions.csv"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "coffee_xgb_proj4_feature_importance.csv"
PROJECTION_FILE = OUTPUT_DIR / "coffee_xgb_proj4_latest_projection.csv"
PATH_FILE = OUTPUT_DIR / "coffee_xgb_proj4_rolling_path.csv"
PROJECTION_PNG_FILE = OUTPUT_DIR / "coffee_xgb_proj4_6m_projection.png"

BUSINESS_DAYS_PER_WEEK = 5
HORIZONS_WEEKS = [1, 4, 12, 26, 52]
RANDOM_STATE = 42
MIN_TRAIN_ROWS = 400
WF_N_FOLDS = 5
WF_MIN_TEST_ROWS = 20
PRUNE_CORR_THRESHOLD = 0.95
PRUNE_IMP_THRESHOLD = 0.005

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.80,
    colsample_bytree=0.70,
    min_child_weight=5,
    reg_lambda=2.0,
    reg_alpha=0.1,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)
EARLY_STOPPING_ROUNDS = 30
EVAL_FRACTION = 0.15

SHORT_LAGS = [1, 2, 3, 5, 10, 20]
SHORT_ROLLS = [5, 10, 20]
MACRO_ROLLS = [20, 60, 120, 252]
CUM_WINDOWS = [30, 90, 180, 252]
INVENTORY_WINDOWS = [20, 65, 130, 260]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def rolling_zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(20, window // 2)
    mu = s.rolling(window, min_periods=min_periods).mean()
    sig = s.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return (s - mu) / sig


def days_since_last_flag(flag: pd.Series) -> pd.Series:
    dates = pd.to_datetime(flag.index)
    last = pd.NaT
    vals: list[float] = []
    for dt, f in zip(dates, flag.fillna(0).astype(float)):
        if f > 0:
            last = dt
        vals.append(np.nan if pd.isna(last) else float((dt - last).days))
    return pd.Series(vals, index=flag.index, dtype=float)


def evaluate(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if len(yt) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "dir_acc": np.nan, "corr": np.nan}
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)) if len(yt) > 1 else np.nan,
        "dir_acc": float(np.mean(np.sign(yt) == np.sign(yp))),
        "corr": float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 else np.nan,
    }


def log_price_change_target(price: pd.Series, horizon_days: int) -> pd.Series:
    lp = np.log(price.where(price > 0))
    return lp.shift(-horizon_days) - lp


def build_model(early_stopping: bool) -> XGBRegressor:
    params = dict(XGB_PARAMS)
    if early_stopping:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return XGBRegressor(**params)


def compute_sample_weights(train_df: pd.DataFrame, horizon_weeks: int) -> np.ndarray:
    days_ago = (pd.to_datetime(train_df["Date"]).max() - pd.to_datetime(train_df["Date"])).dt.days.to_numpy(dtype=float)
    weights = np.exp(-days_ago / 730.0)

    if "coffee_high_vol_regime" in train_df.columns:
        weights *= 1.0 + 0.50 * train_df["coffee_high_vol_regime"].fillna(0.0).to_numpy(dtype=float)

    if horizon_weeks >= 12 and "coffee_macro_uptrend" in train_df.columns:
        weights *= 1.0 + 0.30 * train_df["coffee_macro_uptrend"].fillna(0.0).to_numpy(dtype=float)

    mean_w = weights.mean()
    return weights / mean_w if mean_w > 0 else weights


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_short_term_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "coffee_c_log_return" in out.columns:
        for lag in SHORT_LAGS:
            out[f"coffee_c_log_return_lag_{lag}"] = out["coffee_c_log_return"].shift(lag)
        for w in SHORT_ROLLS:
            out[f"coffee_c_log_return_roll_mean_{w}"] = out["coffee_c_log_return"].rolling(w, min_periods=max(2, w // 2)).mean()
            out[f"coffee_c_log_return_roll_std_{w}"] = out["coffee_c_log_return"].rolling(w, min_periods=max(2, w // 2)).std()

        out["coffee_abs_return_5d"] = out["coffee_c_log_return"].abs().rolling(5, min_periods=2).mean()
        out["coffee_abs_return_20d"] = out["coffee_c_log_return"].abs().rolling(20, min_periods=5).mean()
        out["coffee_abs_return_60d"] = out["coffee_c_log_return"].abs().rolling(60, min_periods=15).mean()
        out["coffee_vol_20d"] = out["coffee_c_log_return"].rolling(20, min_periods=5).std() * np.sqrt(252)
        out["coffee_vol_60d"] = out["coffee_c_log_return"].rolling(60, min_periods=15).std() * np.sqrt(252)
        out["coffee_vol_regime_ratio_20_60"] = out["coffee_vol_20d"] / out["coffee_vol_60d"].replace(0, np.nan)
        out["coffee_high_vol_regime"] = (out["coffee_vol_regime_ratio_20_60"] > 1.2).astype(float)

    for exog in ["soybeans_log_return", "sugar_log_return", "usd_brl_log_return"]:
        if exog in out.columns:
            for lag in [1, 2, 3, 5, 10]:
                out[f"{exog}_lag_{lag}"] = out[exog].shift(lag)

    return out


def add_macro_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    price = out["coffee_c"]
    lp = np.log(price.where(price > 0))

    for w in CUM_WINDOWS:
        out[f"coffee_cum_logret_{w}d"] = lp - lp.shift(w)

    for w in MACRO_ROLLS:
        out[f"coffee_sma_{w}d"] = price.rolling(w, min_periods=max(20, w // 2)).mean()
        out[f"coffee_std_{w}d"] = price.rolling(w, min_periods=max(20, w // 2)).std()
        out[f"coffee_vs_sma_{w}d"] = price / out[f"coffee_sma_{w}d"].replace(0, np.nan) - 1.0

    out["coffee_zscore_252d"] = rolling_zscore(price, 252)
    out["coffee_zscore_504d"] = rolling_zscore(price, 504, min_periods=126)
    out["coffee_52w_high"] = price.rolling(252, min_periods=126).max()
    out["coffee_52w_low"] = price.rolling(252, min_periods=126).min()
    out["coffee_dist_52w_high"] = price / out["coffee_52w_high"].replace(0, np.nan) - 1.0
    out["coffee_dist_52w_low"] = price / out["coffee_52w_low"].replace(0, np.nan) - 1.0
    out["coffee_macro_uptrend"] = (out["coffee_vs_sma_252d"] > 0.05).astype(float)
    out["coffee_macro_downtrend"] = (out["coffee_vs_sma_252d"] < -0.05).astype(float)

    if "coffee_c_log_return" in out.columns:
        pos = (out["coffee_c_log_return"] > 0).astype(float)
        out["coffee_dir_consistency_60d"] = pos.rolling(60, min_periods=15).mean()
        out["coffee_dir_consistency_120d"] = pos.rolling(120, min_periods=30).mean()
        out["coffee_dir_consistency_252d"] = pos.rolling(252, min_periods=60).mean()
        for w in [60, 120, 252]:
            mu = out["coffee_c_log_return"].rolling(w, min_periods=max(10, w // 4)).mean()
            sig = out["coffee_c_log_return"].rolling(w, min_periods=max(10, w // 4)).std().replace(0, np.nan)
            out[f"coffee_sharpe_{w}d"] = mu / sig * np.sqrt(252)

    if {"coffee_cum_logret_30d", "coffee_cum_logret_180d"}.issubset(out.columns):
        out["coffee_momentum_acceleration"] = out["coffee_cum_logret_30d"] - out["coffee_cum_logret_180d"] / 3.0

    return out


def add_macro_fx_exog_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "usd_brl" in out.columns:
        lp_fx = np.log(out["usd_brl"].where(out["usd_brl"] > 0))
        for w in [60, 120, 252]:
            out[f"usd_brl_trend_{w}d"] = lp_fx - lp_fx.shift(w)
            out[f"usd_brl_sma_{w}d"] = out["usd_brl"].rolling(w, min_periods=max(20, w // 2)).mean()
        out["usd_brl_zscore_252d"] = rolling_zscore(out["usd_brl"], 252)
        out["brl_weakening_regime"] = (out["usd_brl_trend_120d"] > 0.03).astype(float)
        out["brl_strengthening_regime"] = (out["usd_brl_trend_120d"] < -0.03).astype(float)
        for w in CUM_WINDOWS:
            out[f"usd_brl_cum_logret_{w}d"] = lp_fx - lp_fx.shift(w)

    for asset in ["soybeans", "sugar"]:
        if asset in out.columns:
            lp = np.log(out[asset].where(out[asset] > 0))
            for w in [60, 120, 252]:
                out[f"{asset}_trend_{w}d"] = lp - lp.shift(w)
            out[f"{asset}_zscore_252d"] = rolling_zscore(out[asset], 252)

    if {"coffee_c_log_return", "sugar_log_return"}.issubset(out.columns):
        out["coffee_minus_sugar_log_return"] = out["coffee_c_log_return"] - out["sugar_log_return"]
    if {"coffee_c_log_return", "soybeans_log_return"}.issubset(out.columns):
        out["coffee_minus_soybeans_log_return"] = out["coffee_c_log_return"] - out["soybeans_log_return"]
    if {"coffee_c_log_return", "usd_brl_log_return"}.issubset(out.columns):
        out["coffee_minus_fx_log_return"] = out["coffee_c_log_return"] - out["usd_brl_log_return"]

    return out


def add_climate_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "rainfall" in out.columns:
        out["rainfall_roll_mean_7"] = out["rainfall"].rolling(7, min_periods=3).mean()
        out["rainfall_roll_mean_30"] = out["rainfall"].rolling(30, min_periods=10).mean()
        out["rainfall_rolling_30"] = out["rainfall"].rolling(30, min_periods=10).sum()
        out["rainfall_rolling_90"] = out["rainfall"].rolling(90, min_periods=30).sum()
        out["rainfall_rolling_180"] = out["rainfall"].rolling(180, min_periods=60).sum()
        out["rainfall_regime_ratio_30_90"] = out["rainfall_rolling_30"] / out["rainfall_rolling_90"].replace(0, np.nan)
        out["rainfall_regime_ratio_90_180"] = out["rainfall_rolling_90"] / out["rainfall_rolling_180"].replace(0, np.nan)
        for lag in [1, 3, 5, 10]:
            out[f"rainfall_lag_{lag}"] = out["rainfall"].shift(lag)

    if "tmax" in out.columns:
        out["heat_stress_30d"] = (out["tmax"] - out["tmax"].rolling(252, min_periods=60).mean()).rolling(30, min_periods=10).mean()
        out["heat_stress_90d"] = (out["tmax"] - out["tmax"].rolling(252, min_periods=60).mean()).rolling(90, min_periods=30).mean()
        for lag in [1, 3, 5, 10]:
            out[f"tmax_lag_{lag}"] = out["tmax"].shift(lag)

    if "tmin" in out.columns:
        for lag in [1, 3, 5, 10]:
            out[f"tmin_lag_{lag}"] = out["tmin"].shift(lag)
    if "tavg" in out.columns:
        for lag in [1, 3, 5, 10]:
            out[f"tavg_lag_{lag}"] = out["tavg"].shift(lag)
    if "trange" in out.columns:
        for lag in [1, 3, 5, 10]:
            out[f"trange_lag_{lag}"] = out["trange"].shift(lag)

    if "enso_index" in out.columns:
        out["enso_positive"] = (out["enso_index"] > 0).astype(float)
        out["enso_el_nino_flag"] = (out["enso_index"] >= 0.5).astype(float)
        out["enso_la_nina_flag"] = (out["enso_index"] <= -0.5).astype(float)
        out["enso_roll_mean_30"] = out["enso_index"].rolling(30, min_periods=10).mean()
        out["enso_roll_mean_90"] = out["enso_index"].rolling(90, min_periods=30).mean()
        out["enso_roll_mean_180"] = out["enso_index"].rolling(180, min_periods=60).mean()
        for lag in [1, 5, 10, 20]:
            out[f"enso_index_lag_{lag}"] = out["enso_index"].shift(lag)

    if "drought_index" in out.columns:
        out["drought_roll_mean_30"] = out["drought_index"].rolling(30, min_periods=10).mean()
        out["drought_roll_mean_90"] = out["drought_index"].rolling(90, min_periods=30).mean()
        out["drought_persistence_90d"] = out["drought_flag"].rolling(90, min_periods=30).mean() if "drought_flag" in out.columns else np.nan
        for lag in [1, 5, 10, 20]:
            out[f"drought_index_lag_{lag}"] = out["drought_index"].shift(lag)

    if "drought_flag" in out.columns:
        for lag in [1, 5, 10, 20]:
            out[f"drought_flag_lag_{lag}"] = out["drought_flag"].shift(lag)

    if "frost_severity" in out.columns:
        out["frost_roll_sum_30"] = out["frost_severity"].rolling(30, min_periods=10).sum()
        out["frost_roll_sum_90"] = out["frost_severity"].rolling(90, min_periods=30).sum()
        out["frost_severity_temp_30d"] = out["frost_severity"].rolling(30, min_periods=10).mean()
        out["frost_severity_temp_90d"] = out["frost_severity"].rolling(90, min_periods=30).mean()
        for lag in [1, 5, 10, 20]:
            out[f"frost_severity_lag_{lag}"] = out["frost_severity"].shift(lag)

    if "frost_flag" in out.columns:
        for lag in [1, 5, 10, 20]:
            out[f"frost_flag_lag_{lag}"] = out["frost_flag"].shift(lag)

    return out


def add_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "inventory_certified_bags" not in out.columns:
        return out

    avail = out.get("inventory_available_flag", pd.Series(1.0, index=out.index)).fillna(0.0)
    cert = out["inventory_certified_bags"].where(avail == 1)

    for window in INVENTORY_WINDOWS:
        weeks = int(round(window / 5))
        out[f"inventory_certified_roll_mean_{weeks}w"] = cert.rolling(window, min_periods=2).mean().fillna(0.0)
        out[f"inventory_certified_roll_std_{weeks}w"] = cert.rolling(window, min_periods=2).std().fillna(0.0)

    base13 = out.get("inventory_certified_roll_mean_13w", cert.rolling(65, min_periods=2).mean()).replace(0, np.nan)
    base26 = out.get("inventory_certified_roll_mean_26w", cert.rolling(130, min_periods=2).mean()).replace(0, np.nan)
    out["inventory_cert_vs_13w_mean"] = (cert / base13 - 1.0).fillna(0.0)
    out["inventory_cert_vs_26w_mean"] = (cert / base26 - 1.0).fillna(0.0)
    out["inventory_low_supply_flag"] = ((out["inventory_cert_vs_26w_mean"] < -0.05) & (avail == 1)).astype(float)
    out["inventory_high_supply_flag"] = ((out["inventory_cert_vs_26w_mean"] > 0.05) & (avail == 1)).astype(float)

    for lag in [5, 10, 20, 60]:
        out[f"inventory_certified_bags_lag_{lag}"] = cert.shift(lag).fillna(0.0)
        out[f"inventory_cert_vs_26w_mean_lag_{lag}"] = out["inventory_cert_vs_26w_mean"].shift(lag).fillna(0.0)

    out["days_since_inventory_report"] = days_since_last_flag(avail).fillna(365.0)
    return out


def add_api_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "api_curve_slope_1_6" in out.columns:
        out["api_curve_slope_1_6_z26"] = rolling_zscore(out["api_curve_slope_1_6"], 130, min_periods=26)
    if "api_curve_curvature_1_3_6" in out.columns:
        out["api_curve_curvature_1_3_6_z26"] = rolling_zscore(out["api_curve_curvature_1_3_6"], 130, min_periods=26)
    if "api_spec_net" in out.columns:
        out["api_spec_net_z52"] = rolling_zscore(out["api_spec_net"], 260, min_periods=52)
        out["api_spec_net_4w_chg"] = out["api_spec_net"].diff(20)
    if "api_brl_per_usd" in out.columns:
        out["api_brl_per_usd_13w_chg"] = np.log(out["api_brl_per_usd"].where(out["api_brl_per_usd"] > 0)).diff(65)

    forecast_cols = [c for c in out.columns if c.startswith("api_fcst_") or c.startswith("api_weather_fcst_")]
    for c in forecast_cols:
        out[f"{c}_z26"] = rolling_zscore(safe_numeric(out[c]), 130, min_periods=26)

    out["days_since_api_panel"] = out.get("days_since_api_panel", pd.Series(999.0, index=out.index)).fillna(999.0)
    return out


def add_calendar_and_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["Date"])

    month = dates.dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    week = dates.dt.isocalendar().week.astype(int)
    out["week_sin"] = np.sin(2 * np.pi * week / 52.0)
    out["week_cos"] = np.cos(2 * np.pi * week / 52.0)
    out["quarter"] = dates.dt.quarter.astype(float)
    out["day_of_week"] = dates.dt.dayofweek.astype(float)

    out["brazil_harvest_flag"] = month.isin([5, 6, 7, 8, 9]).astype(float)
    out["brazil_flowering_flag"] = month.isin([9, 10, 11]).astype(float)

    if {"brazil_flowering_flag", "rainfall_rolling_90"}.issubset(out.columns):
        out["flowering_rainfall_90_interaction"] = out["brazil_flowering_flag"] * out["rainfall_rolling_90"]
    if {"brazil_harvest_flag", "coffee_vs_sma_120d"}.issubset(out.columns):
        out["harvest_trend_120_interaction"] = out["brazil_harvest_flag"] * out["coffee_vs_sma_120d"]
    if {"brl_weakening_regime", "inventory_low_supply_flag"}.issubset(out.columns):
        out["brl_weak_and_low_supply"] = out["brl_weakening_regime"] * out["inventory_low_supply_flag"]
    if {"coffee_macro_uptrend", "brazil_harvest_flag"}.issubset(out.columns):
        out["uptrend_during_harvest"] = out["coffee_macro_uptrend"] * out["brazil_harvest_flag"]
    if {"api_spec_net_z52", "api_curve_slope_1_6_z26"}.issubset(out.columns):
        out["api_flow_curve_interaction"] = out["api_spec_net_z52"] * out["api_curve_slope_1_6_z26"]

    return out


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for horizon in HORIZONS_WEEKS:
        out[f"target_log_price_change_{horizon}w"] = log_price_change_target(
            out["coffee_c"],
            horizon * BUSINESS_DAYS_PER_WEEK,
        )
    return out


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_short_term_features(out)
    out = add_macro_price_features(out)
    out = add_macro_fx_exog_features(out)
    out = add_climate_features(out)
    out = add_inventory_features(out)
    out = add_api_features(out)
    out = add_calendar_and_interactions(out)
    out = add_targets(out)
    return out


# ---------------------------------------------------------------------------
# Feature selection / modelling
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame, horizon_weeks: int) -> list[str]:
    exclude_exact = {
        "Date", "coffee_c", "coffee_log_price", "coffee_c_log_return",
        "soybeans", "soybeans_log_return", "sugar", "sugar_log_return",
        "usd_brl", "usd_brl_log_return",
        "tmax", "tmin", "tavg", "trange", "rainfall",
        "enso_index", "drought_index", "drought_flag", "frost_severity", "frost_flag",
        "inventory_certified_bags", "inventory_transition_bags", "inventory_total_bags",
    }

    features: list[str] = []
    for col in df.columns:
        if col in exclude_exact:
            continue
        if col.startswith("target_"):
            continue
        if df[col].dtype == object:
            continue
        features.append(col)

    if horizon_weeks >= 12:
        drop_short = {f"coffee_c_log_return_lag_{lag}" for lag in [1, 2]}
        features = [c for c in features if c not in drop_short]

    seen: set[str] = set()
    out: list[str] = []
    for c in features:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def prune_features(X_train: pd.DataFrame, y_train: pd.Series, features: list[str]) -> list[str]:
    if not features:
        return []

    X = X_train[features].fillna(0.0).astype(float)
    y = y_train.astype(float)

    scout = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    scout.fit(X, y)

    importances = scout.feature_importances_
    survivors = [f for f, imp in zip(features, importances) if imp >= PRUNE_IMP_THRESHOLD]
    if not survivors:
        survivors = features[: min(10, len(features))]

    Xs = X[survivors]
    corr = Xs.corr().abs()
    imp_map = {f: imp for f, imp in zip(features, importances)}

    drop: set[str] = set()
    cols = list(Xs.columns)
    for i in range(len(cols)):
        if cols[i] in drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in drop:
                continue
            if corr.iloc[i, j] > PRUNE_CORR_THRESHOLD:
                if imp_map.get(cols[i], 0.0) >= imp_map.get(cols[j], 0.0):
                    drop.add(cols[j])
                else:
                    drop.add(cols[i])

    return [c for c in survivors if c not in drop]


def trailing_baseline(ret: pd.Series, horizon_days: int) -> pd.Series:
    out = pd.Series(0.0, index=ret.index)
    for i in range(1, horizon_days + 1):
        out = out + ret.shift(i)
    return out


def walk_forward_validate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for horizon in [4, 12, 26, 52]:
        target_col = f"target_log_price_change_{horizon}w"
        features = get_feature_columns(df, horizon)
        work = df[["Date", "coffee_c", "coffee_c_log_return", target_col] + features].copy()
        work["trailing_baseline"] = trailing_baseline(work["coffee_c_log_return"], horizon * BUSINESS_DAYS_PER_WEEK)
        work = work.dropna(subset=[target_col]).reset_index(drop=True)

        feature_missing = work[features].isna().mean(axis=1)
        work = work.loc[feature_missing < 0.50].reset_index(drop=True)

        n = len(work)
        min_train = max(MIN_TRAIN_ROWS, n // (WF_N_FOLDS + 2))
        remaining = n - min_train
        if remaining < WF_MIN_TEST_ROWS:
            continue
        test_step = max(WF_MIN_TEST_ROWS, remaining // WF_N_FOLDS)

        for fold in range(WF_N_FOLDS):
            train_end = min_train + fold * test_step
            test_start = train_end
            test_end = min(test_start + test_step, n)
            if train_end >= n or (test_end - test_start) < WF_MIN_TEST_ROWS:
                break

            train = work.iloc[:train_end].copy()
            test = work.iloc[test_start:test_end].copy()

            X_train_full = train[features].fillna(0.0)
            y_train = train[target_col].astype(float)
            X_test_full = test[features].fillna(0.0)
            y_test = test[target_col].astype(float)

            survived = prune_features(X_train_full, y_train, features)
            X_train = X_train_full[survived]
            X_test = X_test_full[survived]
            weights = compute_sample_weights(train, horizon)

            n_eval = max(10, int(len(X_train) * EVAL_FRACTION))
            if len(X_train) <= n_eval + 10:
                n_eval = max(5, len(X_train) // 5)

            X_fit, X_eval = X_train.iloc[:-n_eval], X_train.iloc[-n_eval:]
            y_fit, y_eval = y_train.iloc[:-n_eval], y_train.iloc[-n_eval:]
            w_fit = weights[:-n_eval]

            xgb = build_model(early_stopping=True)
            xgb.fit(X_fit, y_fit, sample_weight=w_fit, eval_set=[(X_eval, y_eval)], verbose=False)
            pred_xgb = xgb.predict(X_test)

            lo, hi = np.nanpercentile(y_train, 2), np.nanpercentile(y_train, 98)
            pred_xgb_clip = np.clip(pred_xgb, lo, hi)

            ridge = Ridge(alpha=100.0)
            ridge.fit(X_train, y_train, sample_weight=weights)
            pred_ridge = ridge.predict(X_test)
            pred_zero = np.zeros(len(y_test), dtype=float)
            pred_trailing = test["trailing_baseline"].to_numpy(dtype=float)

            for model_name, pred in [
                ("xgb", pred_xgb_clip),
                ("ridge", pred_ridge),
                ("zero", pred_zero),
                ("trailing", pred_trailing),
            ]:
                metric_rows.append({
                    "horizon_weeks": horizon,
                    "fold": fold + 1,
                    "model": model_name,
                    "n_obs": len(y_test),
                    **evaluate(y_test, pred),
                    "n_features_used": len(survived),
                })

            for i in range(len(test)):
                pred_rows.append({
                    "Date": test.iloc[i]["Date"],
                    "horizon_weeks": horizon,
                    "fold": fold + 1,
                    "y_true": float(y_test.iloc[i]),
                    "y_pred_xgb": float(pred_xgb_clip[i]),
                    "y_pred_ridge": float(pred_ridge[i]),
                    "y_pred_zero": float(pred_zero[i]),
                    "y_pred_trailing": float(pred_trailing[i]) if np.isfinite(pred_trailing[i]) else np.nan,
                    "n_features_used": len(survived),
                })

            if fold == WF_N_FOLDS - 1:
                fi = pd.Series(xgb.feature_importances_, index=survived).sort_values(ascending=False)
                for feat, imp in fi.head(50).items():
                    feature_rows.append({
                        "horizon_weeks": horizon,
                        "feature": feat,
                        "importance": float(imp),
                    })

    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows), feature_rows


def fit_one_model_for_horizon(df: pd.DataFrame, horizon: int) -> dict[str, object] | None:
    target_col = f"target_log_price_change_{horizon}w"
    features = get_feature_columns(df, horizon)
    work = df[["Date", "coffee_c", target_col] + features].copy()
    work = work.dropna(subset=[target_col]).reset_index(drop=True)

    feature_missing = work[features].isna().mean(axis=1)
    work = work.loc[feature_missing < 0.50].reset_index(drop=True)
    if len(work) < MIN_TRAIN_ROWS:
        return None

    X_full = work[features].fillna(0.0)
    y_full = work[target_col].astype(float)
    survived = prune_features(X_full, y_full, features)
    X_train = X_full[survived]
    weights = compute_sample_weights(work, horizon)

    model = build_model(early_stopping=False)
    model.fit(X_train, y_full, sample_weight=weights)

    lo, hi = np.nanpercentile(y_full, 2), np.nanpercentile(y_full, 98)

    return {
        "horizon_weeks": horizon,
        "model": model,
        "features": survived,
        "clip_lo": float(lo),
        "clip_hi": float(hi),
        "train_df": work.copy(),
    }


def predict_horizon_from_latest(model_info: dict[str, object], latest_row: pd.DataFrame) -> float:
    model = model_info["model"]
    features = model_info["features"]
    lo = model_info["clip_lo"]
    hi = model_info["clip_hi"]

    X_latest = latest_row.reindex(columns=features).fillna(0.0)
    raw_pred = float(model.predict(X_latest)[0])
    return float(np.clip(raw_pred, lo, hi))


def build_all_final_models(df: pd.DataFrame) -> dict[int, dict[str, object]]:
    models: dict[int, dict[str, object]] = {}
    for horizon in HORIZONS_WEEKS:
        info = fit_one_model_for_horizon(df, horizon)
        if info is not None:
            models[horizon] = info
    return models


def make_future_exog_row(last_row: pd.Series, next_date: pd.Timestamp) -> dict[str, object]:
    row = last_row.to_dict()
    row["Date"] = next_date

    # Keep most exogenous values persistent unless they are calendar-derived.
    # This is conservative and matches how the merged dataset already forward-fills
    # many slower exogenous series in the data build process.
    return row


def recursive_weekly_path(
    feature_df: pd.DataFrame,
    models: dict[int, dict[str, object]],
    n_weeks: int = 26,
    anchor_weight_start: float = 0.35,
    shock_cap_sigma: float = 1.25,
    noise_fraction: float = 0.70,
    seed: int | None = None,
) -> pd.DataFrame:
    if 1 not in models:
        raise ValueError("Need a 1-week model for recursive rolling forecasts.")
    if 26 not in models:
        raise ValueError("Need a 26-week model for long-horizon anchoring.")

    rng = np.random.default_rng(seed)

    hist = feature_df.copy().sort_values("Date").reset_index(drop=True)
    last_obs = hist.iloc[-1].copy()
    current_price = float(last_obs["coffee_c"])
    current_date = pd.to_datetime(last_obs["Date"])

    latest_row = hist.iloc[[-1]].copy()
    long_total_log_change = predict_horizon_from_latest(models[26], latest_row)

    train_1w = models[1]["train_df"]
    sigma_1w = float(train_1w["target_log_price_change_1w"].std())
    if not np.isfinite(sigma_1w) or sigma_1w <= 0:
        sigma_1w = 0.03

    # AR(1) noise state for autocorrelated volatility (phi ≈ 0.35 gives realistic clustering)
    ar_phi = 0.35
    ar_noise_state = 0.0

    remaining_target = long_total_log_change
    path_rows: list[dict[str, object]] = []

    sim_raw = hist[["Date", "coffee_c"]].copy()

    for step in range(1, n_weeks + 1):
        sim_feature = build_feature_dataset(sim_raw.copy())
        sim_latest = sim_feature.iloc[[-1]].copy()

        pred_1w = predict_horizon_from_latest(models[1], sim_latest)

        weeks_left_including_this = n_weeks - step + 1
        anchor_weekly = remaining_target / weeks_left_including_this

        # Blend recursive short-term signal with long-horizon anchor.
        # Anchor fades over time so early path shape is realistic but final level still converges.
        progress = (step - 1) / max(1, n_weeks - 1)
        anchor_weight = anchor_weight_start * (1.0 - progress)
        blended = (1.0 - anchor_weight) * pred_1w + anchor_weight * anchor_weekly

        # AR(1) noise: persistent volatility shocks like real markets
        innovation = float(rng.normal(0.0, sigma_1w))
        ar_noise_state = ar_phi * ar_noise_state + np.sqrt(1 - ar_phi ** 2) * innovation
        noise = noise_fraction * ar_noise_state

        blended_noisy = blended + noise

        # Cap unrealistically large weekly jumps.
        blended_noisy = float(np.clip(blended_noisy, -shock_cap_sigma * 2.0 * sigma_1w, shock_cap_sigma * 2.0 * sigma_1w))

        next_price = float(current_price * np.exp(blended_noisy))
        next_date = current_date + pd.Timedelta(weeks=1)

        path_rows.append({
            "step_week": step,
            "Date": next_date,
            "predicted_weekly_log_return": blended_noisy,
            "projected_price": next_price,
            "anchor_weekly_log_return": float(anchor_weekly),
            "raw_1w_log_return": float(pred_1w),
        })

        remaining_target -= blended_noisy
        current_price = next_price
        current_date = next_date

        sim_raw = pd.concat(
            [
                sim_raw,
                pd.DataFrame([{"Date": next_date, "coffee_c": next_price}]),
            ],
            ignore_index=True,
        )

    # Small terminal alignment so final point exactly matches anchored 26w target.
    anchored_final_price = float(last_obs["coffee_c"] * np.exp(long_total_log_change))
    if len(path_rows) > 0:
        end_price = path_rows[-1]["projected_price"]
        terminal_ratio = anchored_final_price / end_price if end_price > 0 else 1.0

        for i, row in enumerate(path_rows, start=1):
            frac = i / len(path_rows)
            row["projected_price"] = float(row["projected_price"] * (terminal_ratio ** frac))

    return pd.DataFrame(path_rows)


def fit_final_models_and_project(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = build_all_final_models(df)
    if 1 not in models or 26 not in models:
        raise ValueError("Final forecasting now requires both 1-week and 26-week models.")

    latest_row = df.iloc[[-1]].copy()
    current_price = float(latest_row["coffee_c"].iloc[0])
    as_of_date = pd.to_datetime(latest_row["Date"].iloc[0])

    projection_rows: list[dict[str, object]] = []
    for horizon in [4, 12, 26, 52]:
        if horizon not in models:
            continue
        clipped = predict_horizon_from_latest(models[horizon], latest_row)
        projected_price = float(current_price * np.exp(clipped))
        projection_rows.append({
            "as_of_date": as_of_date.date().isoformat(),
            "horizon_weeks": horizon,
            "current_price": current_price,
            "predicted_log_change": clipped,
            "projected_price": projected_price,
            "n_features_used": len(models[horizon]["features"]),
        })

    path_df = recursive_weekly_path(df, models, n_weeks=26)
    path_df.insert(0, "as_of_date", as_of_date.date().isoformat())
    return pd.DataFrame(projection_rows), path_df


def weekly_path_to_business_days(as_of_date: pd.Timestamp, current_price: float, weekly_path: pd.DataFrame) -> pd.DataFrame:
    points = pd.concat(
        [
            pd.DataFrame([{"Date": as_of_date, "projected_price": current_price}]),
            weekly_path[["Date", "projected_price"]].rename(columns={"projected_price": "projected_price"}),
        ],
        ignore_index=True,
    ).sort_values("Date").reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for i in range(len(points) - 1):
        start_date = pd.to_datetime(points.loc[i, "Date"])
        end_date = pd.to_datetime(points.loc[i + 1, "Date"])
        start_price = float(points.loc[i, "projected_price"])
        end_price = float(points.loc[i + 1, "projected_price"])

        bdays = pd.date_range(start=start_date, end=end_date, freq="B")
        if len(bdays) == 0:
            continue

        total_log_move = np.log(end_price / start_price) if start_price > 0 and end_price > 0 else 0.0
        step_log = np.linspace(0.0, total_log_move, len(bdays))
        prices = start_price * np.exp(step_log)

        for dt, px in zip(bdays, prices):
            rows.append({"Date": dt, "projected_price": float(px)})

    out = pd.DataFrame(rows).drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)
    return out


def save_six_month_projection_plot(raw_df: pd.DataFrame, weekly_path: pd.DataFrame, output_path: Path) -> None:
    if weekly_path.empty:
        return

    as_of_date = pd.to_datetime(weekly_path["as_of_date"].iloc[0])
    current_price_series = raw_df.loc[raw_df["Date"] == as_of_date, "coffee_c"]
    if current_price_series.empty:
        # Fall back to last known price if exact date not found
        current_price = float(raw_df["coffee_c"].iloc[-1])
    else:
        current_price = float(current_price_series.iloc[-1])

    history_start = as_of_date - pd.Timedelta(days=365)
    history = (
        raw_df.loc[(raw_df["Date"] >= history_start) & (raw_df["Date"] <= as_of_date), ["Date", "coffee_c"]]
        .dropna()
        .sort_values("Date")
        .copy()
    )
    if history.empty:
        return

    business_path = weekly_path_to_business_days(as_of_date, current_price, weekly_path)

    # Build confidence cone from weekly path sigma (annualised vol -> weekly vol)
    # Use realized weekly returns over the path to estimate forward uncertainty
    weekly_log_rets = weekly_path["predicted_weekly_log_return"].values
    sigma_weekly = float(np.std(weekly_log_rets)) if len(weekly_log_rets) > 2 else 0.03
    # Derive times-in-weeks for each business-day point
    bp_dates = pd.to_datetime(business_path["Date"])
    weeks_elapsed = (bp_dates - as_of_date).dt.days / 7.0
    # Cone is ±1 sigma (fan widens with sqrt of time)
    cone_half = current_price * (np.exp(sigma_weekly * np.sqrt(weeks_elapsed)) - 1.0)
    upper_band = business_path["projected_price"].values + cone_half.values
    lower_band = np.maximum(business_path["projected_price"].values - cone_half.values, 1.0)

    fig, ax = plt.subplots(figsize=(13, 6))

    # Historical price
    ax.plot(history["Date"], history["coffee_c"],
            color="#1f4e79", linewidth=1.8, label="Historical price", zorder=3)

    # Shaded confidence cone
    ax.fill_between(
        business_path["Date"],
        lower_band,
        upper_band,
        color="#f4a261",
        alpha=0.25,
        label="±1σ forecast cone",
        zorder=1,
    )

    # Forecast path
    ax.plot(business_path["Date"], business_path["projected_price"],
            color="#e76f51", linewidth=2.0, linestyle="--", label="6-month forecast path", zorder=4)

    # Weekly waypoint dots
    wk_dates = pd.to_datetime(weekly_path["Date"])
    wk_prices = weekly_path["projected_price"].values
    ax.scatter(wk_dates, wk_prices, color="#e76f51", s=28, zorder=5, alpha=0.85)

    # As-of vertical line
    ax.axvline(as_of_date, color="grey", linewidth=1.2, linestyle=":", label="Forecast start", zorder=2)

    # Annotate final target price
    final_price = float(weekly_path["projected_price"].iloc[-1])
    final_date = pd.to_datetime(weekly_path["Date"].iloc[-1])
    ax.annotate(
        f"  {final_price:.1f}¢",
        xy=(final_date, final_price),
        fontsize=9,
        color="#e76f51",
        va="center",
    )

    ax.set_title("Coffee C: 1-Year History + Recursive 6-Month Forecast Path", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Price (¢/lb)", fontsize=10)
    ax.legend(framealpha=0.85, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input dataset: {INPUT_FILE}\nRun coffee_data7_merged.py first.")

    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "coffee_c"]).sort_values("Date").reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c != "Date"]
    for col in numeric_cols:
        df[col] = safe_numeric(df[col])

    feature_df = build_feature_dataset(df)

    metrics, wf_preds, feature_rows = walk_forward_validate(feature_df)
    projections, weekly_path = fit_final_models_and_project(feature_df)

    metrics.to_csv(METRICS_FILE, index=False)
    wf_preds.to_csv(WF_PREDS_FILE, index=False)
    pd.DataFrame(feature_rows).to_csv(FEATURE_IMPORTANCE_FILE, index=False)
    projections.to_csv(PROJECTION_FILE, index=False)
    weekly_path.to_csv(PATH_FILE, index=False)
    save_six_month_projection_plot(df, weekly_path, PROJECTION_PNG_FILE)

    print("Saved training outputs:")
    for path in [
        METRICS_FILE,
        WF_PREDS_FILE,
        FEATURE_IMPORTANCE_FILE,
        PROJECTION_FILE,
        PATH_FILE,
        PROJECTION_PNG_FILE,
    ]:
        print(f"  {path}")

    if not metrics.empty:
        summary = (
            metrics.groupby(["horizon_weeks", "model"], as_index=False)[["rmse", "mae", "r2", "dir_acc", "corr"]]
            .mean()
            .sort_values(["horizon_weeks", "rmse"])
        )
        print("\nWalk-forward mean metrics:")
        print(summary.to_string(index=False))

    if not projections.empty:
        print("\nLatest projection:")
        print(projections.to_string(index=False))


if __name__ == "__main__":
    main()