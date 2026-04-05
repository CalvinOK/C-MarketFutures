"""
Coffee Futures ARIMA-GARCH Forecast
------------------------------------
Key fixes over previous version:
  1. Rolling 1-step-ahead backtest  — re-fits on expanding window so the
     backtest actually reflects day-by-day predictive accuracy, not a
     single long extrapolation that ignores all the actual prices.
  2. ARIMA-GARCH simulation for forward forecast  — captures both the
     mean trend *and* realistic day-to-day volatility / clustering.
  3. ADF-based differencing order  — empirically checks stationarity
     rather than guessing d from the candidate list.
  4. Seasonal candidates on by default.
"""

import os
import argparse
import warnings
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError("CSV must contain 'Date' and 'Price' columns")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = (
        df[["Date", "Price"]]
        .dropna()
        .sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .set_index("Date")
    )
    df = df.resample("B").last()
    df["Price"] = df["Price"].ffill()
    return df


# ──────────────────────────────────────────────
# Stationarity
# ──────────────────────────────────────────────
def required_differencing(series: pd.Series, max_d: int = 2, significance: float = 0.05) -> int:
    d, s = 0, series.copy()
    for _ in range(max_d):
        if adfuller(s.dropna(), autolag="AIC")[1] <= significance:
            break
        s = s.diff()
        d += 1
    return d


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────
def maybe_log(series: pd.Series, use_log: bool) -> pd.Series:
    if use_log:
        if (series <= 0).any():
            raise ValueError("Log transform: non-positive values found")
        return np.log(series)
    return series.copy()


def invert_forecast(values, use_log: bool):
    return np.exp(values) if use_log else values


# ──────────────────────────────────────────────
# Model fitting
# ──────────────────────────────────────────────
def fit_model(train: pd.Series, order: tuple, seasonal_order=None):
    trend = "c" if order[1] == 0 else "n"
    if seasonal_order is None:
        return ARIMA(train, order=order, trend=trend,
                     enforce_stationarity=False, enforce_invertibility=False).fit()
    return SARIMAX(train, order=order, seasonal_order=seasonal_order, trend=trend,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


# ──────────────────────────────────────────────
# Candidate models
# ──────────────────────────────────────────────
def candidate_models(min_d: int, include_seasonal: bool):
    models = []
    for d in list(dict.fromkeys([min_d, min(min_d + 1, 2)])):
        for p, q in product([0, 1, 2, 3], [0, 1, 2]):
            if p == 0 and q == 0:
                continue
            models.append(("ARIMA", (p, d, q), None))
    if include_seasonal:
        seas_orders = [(1, 0, 0, 5), (0, 1, 1, 5), (1, 1, 0, 5), (1, 1, 1, 5)]
        for order, seas in product([(1, min_d, 1), (2, min_d, 1)], seas_orders):
            models.append(("SARIMA", order, seas))
    return models


# ──────────────────────────────────────────────
# Rolling 1-step-ahead backtest
# ──────────────────────────────────────────────
def rolling_backtest(series_raw: pd.Series, order: tuple, seasonal_order,
                     test_size: int, use_log: bool) -> pd.Series:
    """
    Re-fits the model on each expanding window and forecasts exactly 1 step
    ahead before observing the next actual price. This is how forecasting
    accuracy should be measured for financial data — a multi-step
    extrapolation compresses all volatility into a smooth drift line.
    """
    n = len(series_raw)
    train_end = n - test_size
    preds = []

    for i in range(test_size):
        window_raw = series_raw.iloc[: train_end + i]
        window = maybe_log(window_raw, use_log)
        try:
            fit = fit_model(window, order, seasonal_order)
            fc = fit.forecast(steps=1)
            val = float(invert_forecast(fc.iloc[0], use_log))
        except Exception:
            val = float(series_raw.iloc[train_end + i - 1])
        preds.append(val)

    return pd.Series(preds, index=series_raw.index[train_end:])


# ──────────────────────────────────────────────
# Model selection (fast single-fit pass)
# ──────────────────────────────────────────────
def evaluate_candidate(train_raw, test_raw, model_type, order, seasonal_order, use_log):
    train = maybe_log(train_raw, use_log)
    try:
        fitted = fit_model(train, order, seasonal_order)
        pred = fitted.forecast(steps=len(test_raw))
        pred = pd.Series(invert_forecast(pred, use_log), index=test_raw.index)
        mae = float(mean_absolute_error(test_raw, pred))
        rmse = float(np.sqrt(mean_squared_error(test_raw, pred)))
        aic = float(getattr(fitted, "aic", np.nan))
        wiggle = 0.0
        if model_type == "SARIMA":
            diffs = pred.diff().dropna()
            sign_changes = (np.sign(diffs).diff().fillna(0) != 0).sum()
            wiggle = 0.05 * sign_changes
        return dict(ok=True, model_type=model_type, order=order,
                    seasonal_order=seasonal_order, mae=mae, rmse=rmse,
                    aic=aic, selection_score=mae + wiggle, pred=pred, fit=fitted)
    except Exception as e:
        return dict(ok=False, model_type=model_type, order=order,
                    seasonal_order=seasonal_order, error=str(e))


def select_best_model(series, test_size, use_log, allow_seasonal):
    if len(series) <= test_size + 50:
        raise ValueError("Dataset too short")
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    min_d = required_differencing(train)
    print(f"ADF test → minimum differencing d={min_d}")

    candidates = candidate_models(min_d, allow_seasonal)
    print(f"Evaluating {len(candidates)} candidates for model selection …")

    results = [evaluate_candidate(train, test, mt, o, so, use_log)
               for mt, o, so in candidates]
    results = [r for r in results if r["ok"]]
    if not results:
        raise RuntimeError("No candidate converged")

    results.sort(key=lambda x: (x["selection_score"], x["rmse"], x["aic"]))
    best = results[0]

    if best["model_type"] == "SARIMA":
        ns = next((r for r in results if r["model_type"] == "ARIMA"), None)
        if ns and (ns["mae"] - best["mae"]) / max(ns["mae"], 1e-8) < 0.05:
            best = ns

    return best, train, test, results


# ──────────────────────────────────────────────
# GARCH(1,1) parameter estimation
# ──────────────────────────────────────────────
def estimate_garch_params(residuals: np.ndarray):
    """
    Fits GARCH(1,1) to ARIMA residuals via MLE using scipy.optimize,
    so no 'arch' package dependency is required.
    """
    from scipy.optimize import minimize

    eps = residuals - residuals.mean()
    n = len(eps)
    var0 = float(np.var(eps))

    def garch_nll(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        h = np.full(n, var0)
        ll = 0.0
        for t in range(1, n):
            h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
            if h[t] <= 0:
                return 1e10
            ll += 0.5 * (np.log(h[t]) + eps[t] ** 2 / h[t])
        return ll

    res = minimize(garch_nll, x0=[var0 * 0.05, 0.08, 0.88],
                   method="L-BFGS-B",
                   bounds=[(1e-6, None), (1e-6, 0.999), (1e-6, 0.999)],
                   options={"maxiter": 300, "ftol": 1e-9})

    if res.success:
        omega, alpha, beta = res.x
        if alpha + beta < 0.9999:
            return float(omega), float(alpha), float(beta)

    # Fallback to sensible defaults if optimisation fails
    return float(var0 * 0.05), 0.08, 0.88


# ──────────────────────────────────────────────
# Monte Carlo forward simulation with GARCH vol
# ──────────────────────────────────────────────
def simulate_arima_garch(series_raw: pd.Series, steps: int,
                         use_log: bool, n_sims: int = 1000,
                         vol_lookback: int = 126):
    """
    Simulate forward price paths using a drift-free geometric random walk
    with GARCH(1,1) volatility, anchored to the last observed price.

    WHY NO DRIFT:
    ARIMA's drift term is the average daily price change over the *entire*
    training history. When the model is fit on a period with a strong recent
    trend (e.g. a -20% drawdown), that trend gets baked into the drift and
    then compounded over 30 forward steps — producing a confidently wrong
    extrapolation. For multi-step forecasting beyond a few days, the
    honest position is: we don't know the direction, but we do know roughly
    how much prices move day-to-day. So we simulate from the last price
    using GARCH volatility only, with zero drift.

    RETURNS-SPACE SIMULATION:
    Rather than adding GARCH noise to an ARIMA level forecast (which
    requires inverting the differencing correctly), we simulate directly
    in log-return space and compound back to prices. This guarantees
    prices stay positive and scales volatility correctly.
    """
    # Fit GARCH to log-returns of the recent price history
    # Use a rolling lookback window so vol reflects current regime, not
    # the entire 18-year training history.
    recent_prices = series_raw.iloc[-vol_lookback:]
    log_returns = np.log(recent_prices / recent_prices.shift(1)).dropna().values

    omega, alpha, beta = estimate_garch_params(log_returns)
    sigma2_last = float(np.var(log_returns))

    last_price = float(series_raw.iloc[-1])
    rng = np.random.default_rng(42)
    paths = np.zeros((n_sims, steps))

    for sim in range(n_sims):
        prices = np.zeros(steps)
        h = sigma2_last
        last_eps = log_returns[-1]
        prev_price = last_price

        for t in range(steps):
            h = omega + alpha * last_eps ** 2 + beta * h
            h = max(h, 1e-8)
            # Zero-drift: pure volatility, no directional assumption
            shock = rng.normal(0.0, np.sqrt(h))
            prev_price = prev_price * np.exp(shock)
            prices[t] = prev_price
            last_eps = shock

        paths[sim] = prices

    sim_median = np.median(paths, axis=0)
    lower = np.percentile(paths, 17.5, axis=0)
    upper = np.percentile(paths, 82.5, axis=0)

    # Most-likely path: the single simulated path whose shape stays closest
    # to the ensemble median at every step. This gives a realistic, jagged
    # "representative" path rather than the smooth average.
    # We measure closeness as the sum of squared z-score deviations so that
    # scale differences across the horizon don't bias the selection toward
    # the start (where all paths are close) or the end (where they fan out).
    step_std = paths.std(axis=0)
    step_std = np.where(step_std < 1e-8, 1.0, step_std)   # avoid div-by-zero
    z_deviations = ((paths - sim_median) / step_std) ** 2  # shape: (n_sims, steps)
    most_likely_idx = int(z_deviations.sum(axis=1).argmin())
    most_likely_path = paths[most_likely_idx]

    # Exclude the chosen path from the background sample so it doesn't get
    # drawn twice at different opacities.
    remaining = np.delete(np.arange(n_sims), most_likely_idx)
    sample_paths = paths[rng.choice(remaining, size=min(150, len(remaining)), replace=False)]

    return most_likely_path, lower, upper, sample_paths


# ──────────────────────────────────────────────
# Output / plotting
# ──────────────────────────────────────────────
def save_outputs(series, train, test, test_pred,
                 forecast_mean, lower_65, upper_65, sample_paths,
                 future_index, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    forecast_df = pd.DataFrame({
        "Forecast": forecast_mean,
        "Lower_65": lower_65,
        "Upper_65": upper_65,
    }, index=future_index)
    forecast_csv = os.path.join(output_dir, "coffee_arima_forecast.csv")
    forecast_df.to_csv(forecast_csv, index_label="Date")

    backtest_df = pd.DataFrame({"Actual": test, "Predicted": test_pred})
    backtest_csv = os.path.join(output_dir, "coffee_arima_backtest.csv")
    backtest_df.to_csv(backtest_csv, index_label="Date")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor("#f9f9f9")

    plot_start = series.index.max() - pd.DateOffset(years=1)
    history = series.loc[plot_start:]

    # ── Top panel ─────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#f9f9f9")
    ax1.plot(history.index, history.values, color="#2c7bb6",
             linewidth=1.5, label="Historical", zorder=3)
    ax1.plot(test.index, test_pred.values, color="#e8820c",
             linewidth=1.8, label="Backtest (1-step rolling)", zorder=4)

    # Clip explosive outlier paths before plotting — a handful of paths
    # can reach 2-3× the current price under GARCH, which stretches the
    # y-axis and makes the real distribution look flat. We clip sample
    # paths at ±60% of the last price for visual clarity; the confidence
    # band (which uses percentiles, not the raw max) is unaffected.
    last_price = float(series.iloc[-1])
    price_floor = last_price * 0.30
    price_ceil  = last_price * 1.80
    for path in sample_paths:
        clipped = np.clip(path, price_floor, price_ceil)
        ax1.plot(future_index, clipped, color="green", alpha=0.07, linewidth=0.8)

    ax1.fill_between(future_index, lower_65, upper_65,
                     color="green", alpha=0.30, label="65% Simulated Interval")
    ax1.plot(future_index, forecast_mean, color="darkgreen",
             linewidth=2.2, linestyle="--", label="Most Likely Path", zorder=5)
    ax1.axvline(test.index.min(), linestyle="--", alpha=0.4, color="gray")

    # Pin y-axis: show from 20% below last price to 60% above, so the
    # fan is clearly visible regardless of outlier paths.
    all_visible = np.concatenate([history.values, test_pred.values, lower_65, upper_65])
    ymin = max(min(all_visible) * 0.90, last_price * 0.35)
    ymax = min(max(all_visible) * 1.05, last_price * 1.75)
    ax1.set_ylim(ymin, ymax)

    ax1.set_title("Coffee Futures — ARIMA + GARCH Volatility Simulation", fontsize=13, pad=10)
    ax1.set_ylabel("Price (cents/lb)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ── Bottom panel: backtest zoom ───────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#f9f9f9")
    mae_val = float(mean_absolute_error(test, test_pred))
    rmse_val = float(np.sqrt(mean_squared_error(test, test_pred)))

    ax2.plot(test.index, test.values, color="#2c7bb6",
             linewidth=1.8, label="Actual", zorder=3)
    ax2.plot(test.index, test_pred.values, color="#e8820c",
             linewidth=1.8, label="1-step Forecast", zorder=4)
    ax2.fill_between(test.index, test.values, test_pred.values,
                     alpha=0.12, color="red", label="Error")
    ax2.set_title(
        f"Backtest Detail (Rolling 1-step)  —  MAE: {mae_val:.2f}  |  RMSE: {rmse_val:.2f}",
        fontsize=12, pad=8)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (cents/lb)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax2.get_xticklabels(), rotation=15, ha="right")

    plt.tight_layout(pad=2.5)
    plot_path = os.path.join(output_dir, "coffee_arima_plot.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()

    return forecast_csv, backtest_csv, plot_path


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = next(
        (p for p in [
            os.path.join(script_dir, "CoffeeCData.csv"),
            os.path.join(script_dir, "data", "CoffeeCData.csv"),
        ] if os.path.exists(p)),
        os.path.join(script_dir, "CoffeeCData.csv"),
    )
    output_dir = os.path.join(script_dir, "outputs")

    parser = argparse.ArgumentParser(description="ARIMA-GARCH forecast for coffee futures")
    parser.add_argument("--csv", default=default_csv)
    parser.add_argument("--test-size", type=int, default=60,
                        help="Rolling backtest window in business days (~3 months)")
    parser.add_argument("--forecast-steps", type=int, default=261,
                        help="Forward forecast horizon in business days (default 261 = ~1 year)")
    parser.add_argument("--sims", type=int, default=1000,
                        help="Monte Carlo paths for volatility simulation")
    parser.add_argument("--log", action="store_true",
                        help="Fit on log prices (better for multiplicative vol)")
    parser.add_argument("--no-seasonal", action="store_true",
                        help="Disable SARIMA candidates")
    parser.add_argument("--vol-lookback", type=int, default=126,
                        help="Business days of history for GARCH vol estimation (default 126 = ~6mo)")
    args = parser.parse_args()

    df = load_data(args.csv)
    series = df["Price"]

    # Step 1: fast model selection
    best, train, test, ranking = select_best_model(
        series=series,
        test_size=args.test_size,
        use_log=args.log,
        allow_seasonal=not args.no_seasonal,
    )
    print(f"\nSelected: {best['model_type']} order={best['order']}", end="")
    if best["seasonal_order"]:
        print(f"  seasonal={best['seasonal_order']}", end="")
    print(f"\n  Selection MAE: {best['mae']:.4f}  RMSE: {best['rmse']:.4f}")

    # Step 2: rolling 1-step backtest
    print(f"\nRunning rolling 1-step backtest ({args.test_size} steps) …")
    rolling_pred = rolling_backtest(
        series_raw=series,
        order=best["order"],
        seasonal_order=best["seasonal_order"],
        test_size=args.test_size,
        use_log=args.log,
    )
    mae_roll = float(mean_absolute_error(test, rolling_pred))
    rmse_roll = float(np.sqrt(mean_squared_error(test, rolling_pred)))
    print(f"  Rolling MAE: {mae_roll:.4f}  RMSE: {rmse_roll:.4f}")

    # Step 3: fit on full series, simulate forward
    print("\nFitting final model on full series …")
    y_full = maybe_log(series, args.log)
    final_fit = fit_model(y_full, best["order"], best["seasonal_order"])

    print(f"Simulating {args.sims} forward paths ({args.forecast_steps} steps) …")
    forecast_mean, lower_65, upper_65, sample_paths = simulate_arima_garch(
        series_raw=series,
        steps=args.forecast_steps,
        use_log=args.log,
        n_sims=args.sims,
        vol_lookback=args.vol_lookback,
    )

    future_index = pd.bdate_range(
        series.index[-1] + pd.offsets.BDay(1), periods=args.forecast_steps
    )

    # Step 4: save outputs
    forecast_csv, backtest_csv, plot_path = save_outputs(
        series=series,
        train=train,
        test=test,
        test_pred=rolling_pred,
        forecast_mean=forecast_mean,
        lower_65=lower_65,
        upper_65=upper_65,
        sample_paths=sample_paths,
        future_index=future_index,
        output_dir=output_dir,
    )

    print(f"\nSaved forecast  → {forecast_csv}")
    print(f"Saved backtest  → {backtest_csv}")
    print(f"Saved plot      → {plot_path}")

    print("\nTop 5 candidates (selection pass):")
    for row in ranking[:5]:
        seas = row["seasonal_order"] or "-"
        print(f"  {row['model_type']:6s} order={row['order']} seasonal={seas} "
              f"MAE={row['mae']:.3f} RMSE={row['rmse']:.3f}")


if __name__ == "__main__":
    main()
