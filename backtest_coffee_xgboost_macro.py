from __future__ import annotations

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import argparse
import importlib.util
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
CWD_DIR = Path.cwd()
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_FILE = SCRIPT_DIR / "coffee_xgboost_projection_macro.py"
ALTERNATE_MODEL_FILES = [
    SCRIPT_DIR / "coffee_xgboost_projection.py",
    SCRIPT_DIR / "coffee_xgboost_projection_fixed.py",
    SCRIPT_DIR / "coffee_xgboost_projection_macro.py",
    SCRIPT_DIR / "coffee_xgboost_projection_rewritten.py",
    SCRIPT_DIR / "coffee_xgboost_projection_rewritten_1y_plot.py",
    CWD_DIR / "coffee_xgboost_projection.py",
    CWD_DIR / "coffee_xgboost_projection_fixed.py",
    CWD_DIR / "coffee_xgboost_projection_macro.py",
    CWD_DIR / "coffee_xgboost_projection_rewritten.py",
    CWD_DIR / "coffee_xgboost_projection_rewritten_1y_plot.py",
]
DEFAULT_HORIZONS = [1, 4, 12, 26, 52]
DEFAULT_TEST_POINTS = 60
DEFAULT_STEP_SIZE = 5
DEFAULT_MIN_TRAIN_ROWS = 252
DEFAULT_RANDOM_STATE = 42

PREDICTIONS_FILE = OUTPUT_DIR / "coffee_backtest_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "coffee_backtest_metrics.csv"
PLOT_FILE = OUTPUT_DIR / "coffee_backtest_plot_12w.png"


# ============================================================
# HELPERS
# ============================================================

def resolve_model_file(module_path: Path) -> Path:
    candidates = []

    provided = Path(module_path).expanduser()
    candidates.append(provided)
    if not provided.is_absolute():
        candidates.append((CWD_DIR / provided).resolve())
        candidates.append((SCRIPT_DIR / provided).resolve())

    candidates.extend(path.resolve() for path in ALTERNATE_MODEL_FILES)

    seen = set()
    ordered = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            ordered.append(candidate)

    for candidate in ordered:
        if candidate.exists():
            return candidate

    checked = "\n - ".join(str(path) for path in ordered)
    raise FileNotFoundError(
        "Could not find a projection model file. Checked:\n - " + checked +
        "\nPass --model-file with an explicit path if your file lives elsewhere."
    )


def load_projection_module(module_path: Path):
    module_path = resolve_model_file(module_path)

    spec = importlib.util.spec_from_file_location("coffee_projection_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["coffee_projection_model"] = module
    spec.loader.exec_module(module)
    return module


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n_obs": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def direction_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    yt = np.sign(np.asarray(y_true, dtype=float))
    yp = np.sign(np.asarray(y_pred, dtype=float))
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        return np.nan
    return float((yt[mask] == yp[mask]).mean())


def _coerce_dataframe_from_builder_output(builder_output) -> pd.DataFrame:
    if isinstance(builder_output, pd.DataFrame):
        return builder_output.copy()

    if isinstance(builder_output, tuple):
        for item in builder_output:
            if isinstance(item, pd.DataFrame):
                return item.copy()

    raise TypeError(
        "Could not extract a pandas DataFrame from the model builder output. "
        f"Got type: {type(builder_output)!r}"
    )


def prepare_model_dataframe(module, horizon_weeks: int) -> tuple[pd.DataFrame, list[str], str]:
    if hasattr(module, "build_merged_dataset"):
        builder_output = module.build_merged_dataset()
    elif hasattr(module, "build_feature_dataset"):
        builder_output = module.build_feature_dataset()
    else:
        raise AttributeError(
            "The model module must define build_merged_dataset() or build_feature_dataset()."
        )

    df = _coerce_dataframe_from_builder_output(builder_output)

    if hasattr(module, "get_feature_columns"):
        features = module.get_feature_columns(df)
    else:
        exclude = {"Date", "coffee_c", "coffee_c_log_return"}
        features = [c for c in df.columns if c not in exclude]

    horizon_days = horizon_weeks * module.BUSINESS_DAYS_PER_WEEK
    target_col = f"target_log_return_{horizon_weeks}w"

    if hasattr(module, "cumulative_forward_log_return"):
        df[target_col] = module.cumulative_forward_log_return(df["coffee_c_log_return"], horizon_days)
    elif hasattr(module, "cumulative_target"):
        df[target_col] = module.cumulative_target(df["coffee_c_log_return"], horizon_days)
    else:
        acc = pd.Series(0.0, index=df.index)
        base = pd.to_numeric(df["coffee_c_log_return"], errors="coerce")
        for i in range(1, horizon_days + 1):
            acc = acc + base.shift(-i)
        df[target_col] = acc

    model_df = (
        df.dropna(subset=features + [target_col, "coffee_c", "Date"])
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return model_df, features, target_col


def fit_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def walk_forward_backtest(
    module,
    horizon_weeks: int,
    test_points: int,
    step_size: int,
    min_train_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_df, features, target_col = prepare_model_dataframe(module, horizon_weeks)

    if len(model_df) < (min_train_rows + 5):
        raise ValueError(
            f"Not enough usable rows for {horizon_weeks}w horizon. "
            f"Need at least {min_train_rows + 5}, found {len(model_df)}."
        )

    test_start = max(min_train_rows, len(model_df) - test_points)
    pred_rows: list[dict] = []

    for idx in range(test_start, len(model_df), step_size):
        train_df = model_df.iloc[:idx].copy()
        test_df = model_df.iloc[idx: min(idx + step_size, len(model_df))].copy()

        if len(train_df) < min_train_rows or test_df.empty:
            continue

        X_train = train_df[features].apply(pd.to_numeric, errors="coerce")
        y_train = train_df[target_col].astype(float)
        X_test = test_df[features].apply(pd.to_numeric, errors="coerce")
        y_test = test_df[target_col].astype(float)

        model = fit_model(X_train, y_train, random_state=random_state)
        raw_pred = model.predict(X_test)

        lo = float(np.nanpercentile(y_train, 5))
        hi = float(np.nanpercentile(y_train, 95))
        clipped_pred = np.clip(raw_pred, lo, hi)
        shrunk_pred = clipped_pred * 0.5
        baseline_pred = np.zeros(len(y_test), dtype=float)

        last_price = test_df["coffee_c"].astype(float).to_numpy()
        actual_price = last_price * np.exp(y_test.to_numpy())
        predicted_price = last_price * np.exp(shrunk_pred)
        baseline_price = last_price * np.exp(baseline_pred)

        for i, row in enumerate(test_df.itertuples(index=False)):
            forecast_date = pd.to_datetime(row.Date) + pd.offsets.BDay(
                horizon_weeks * module.BUSINESS_DAYS_PER_WEEK
            )
            pred_rows.append({
                "asof_date": pd.to_datetime(row.Date),
                "forecast_date": forecast_date,
                "horizon_weeks": horizon_weeks,
                "last_observed_price": float(last_price[i]),
                "actual_cumulative_log_return": float(y_test.iloc[i]),
                "predicted_cumulative_log_return": float(shrunk_pred[i]),
                "baseline_cumulative_log_return": float(baseline_pred[i]),
                "actual_price": float(actual_price[i]),
                "predicted_price": float(predicted_price[i]),
                "baseline_price": float(baseline_price[i]),
                "clip_lo": lo,
                "clip_hi": hi,
                "train_rows": int(len(train_df)),
            })

    preds = pd.DataFrame(pred_rows).sort_values(["horizon_weeks", "asof_date"]).reset_index(drop=True)

    if preds.empty:
        raise RuntimeError(f"Backtest produced no predictions for {horizon_weeks}w horizon.")

    xgb_ret = evaluate(preds["actual_cumulative_log_return"], preds["predicted_cumulative_log_return"].to_numpy())
    base_ret = evaluate(preds["actual_cumulative_log_return"], preds["baseline_cumulative_log_return"].to_numpy())
    xgb_px = evaluate(preds["actual_price"], preds["predicted_price"].to_numpy())
    base_px = evaluate(preds["actual_price"], preds["baseline_price"].to_numpy())

    metrics = pd.DataFrame([
        {
            "model": "xgboost",
            "target": "cumulative_log_return",
            "horizon_weeks": horizon_weeks,
            **xgb_ret,
            "direction_accuracy": direction_accuracy(preds["actual_cumulative_log_return"], preds["predicted_cumulative_log_return"].to_numpy()),
        },
        {
            "model": "zero_return_baseline",
            "target": "cumulative_log_return",
            "horizon_weeks": horizon_weeks,
            **base_ret,
            "direction_accuracy": direction_accuracy(preds["actual_cumulative_log_return"], preds["baseline_cumulative_log_return"].to_numpy()),
        },
        {
            "model": "xgboost",
            "target": "price",
            "horizon_weeks": horizon_weeks,
            **xgb_px,
            "direction_accuracy": direction_accuracy(preds["actual_price"].diff().fillna(0.0), preds["predicted_price"].diff().fillna(0.0).to_numpy()),
        },
        {
            "model": "zero_return_baseline",
            "target": "price",
            "horizon_weeks": horizon_weeks,
            **base_px,
            "direction_accuracy": direction_accuracy(preds["actual_price"].diff().fillna(0.0), preds["baseline_price"].diff().fillna(0.0).to_numpy()),
        },
    ])

    return preds, metrics


def make_plot(predictions: pd.DataFrame, horizon_weeks: int, out_file: Path) -> None:
    plot_df = predictions[predictions["horizon_weeks"] == horizon_weeks].copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("forecast_date")

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["forecast_date"], plot_df["actual_price"], label="Actual future price")
    plt.plot(plot_df["forecast_date"], plot_df["predicted_price"], label="Predicted future price")
    plt.plot(plot_df["forecast_date"], plot_df["baseline_price"], label="Zero-return baseline")
    plt.xlabel("Forecast date")
    plt.ylabel("Coffee price")
    plt.title(f"Coffee backtest: {horizon_weeks}-week horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def parse_horizons(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("No horizons were provided.")
    return sorted(set(values))


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest for the coffee XGBoost forecast model.")
    parser.add_argument("--model-file", type=str, default=str(DEFAULT_MODEL_FILE), help="Path to the projection model file (defaults to coffee_xgboost_projection_macro.py).")
    parser.add_argument("--horizons", type=str, default=",".join(str(x) for x in DEFAULT_HORIZONS), help="Comma-separated forecast horizons in weeks. Example: 1,4,12,26,52")
    parser.add_argument("--test-points", type=int, default=DEFAULT_TEST_POINTS, help="Approximate number of final eligible rows to reserve for walk-forward backtesting.")
    parser.add_argument("--step-size", type=int, default=DEFAULT_STEP_SIZE, help="Number of rows predicted per walk-forward retrain step.")
    parser.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS, help="Minimum number of rows required before first backtest fit.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--plot-horizon", type=int, default=12, help="Which horizon to plot in the output chart.")
    args = parser.parse_args()

    module = load_projection_module(Path(args.model_file))
    horizons = parse_horizons(args.horizons)

    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []

    for horizon_weeks in horizons:
        print(f"Backtesting {horizon_weeks}w horizon...")
        preds, metrics = walk_forward_backtest(
            module=module,
            horizon_weeks=horizon_weeks,
            test_points=args.test_points,
            step_size=args.step_size,
            min_train_rows=args.min_train_rows,
            random_state=args.random_state,
        )
        all_predictions.append(preds)
        all_metrics.append(metrics)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)

    predictions_df.to_csv(PREDICTIONS_FILE, index=False)
    metrics_df.to_csv(METRICS_FILE, index=False)
    make_plot(predictions_df, args.plot_horizon, PLOT_FILE)

    print(f"\nSaved predictions: {PREDICTIONS_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")
    print(f"Saved plot: {PLOT_FILE}")

    display_cols = [
        "model",
        "target",
        "horizon_weeks",
        "n_obs",
        "rmse",
        "mae",
        "r2",
        "direction_accuracy",
    ]
    print("\nBacktest summary:")
    print(metrics_df[display_cols].sort_values(["target", "horizon_weeks", "model"]).to_string(index=False))


if __name__ == "__main__":
    main()
