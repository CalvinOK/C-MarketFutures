import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def estimate_weekly_sigma(history: pd.DataFrame) -> float:
    """
    Estimate weekly log-return volatility from historical prices.
    Uses Wednesday-to-Wednesday prices when possible, otherwise falls back
    to 5-row log returns.
    """
    history = history.sort_values("date").copy()

    wed = history[history["date"].dt.day_name() == "Wednesday"].copy()
    if len(wed) >= 8:
        weekly_returns = np.log(wed["price"] / wed["price"].shift(1)).dropna()
    else:
        weekly_returns = np.log(history["price"] / history["price"].shift(5)).dropna()

    sigma = weekly_returns.std()

    if pd.isna(sigma) or sigma <= 0:
        sigma = 0.05  # safe fallback

    return float(sigma)


def make_forecast_plot(
    csv_path: str,
    output_path: str = "coffee_forecast_plot.png",
    show: bool = False,
):
    # -----------------------------
    # Load and prepare data
    # -----------------------------
    df = pd.read_csv(csv_path)

    required_cols = {"series", "date", "price"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["series"] = df["series"].str.lower().str.strip()

    history = df[df["series"].eq("history")].copy().sort_values("date")
    forecast = df[df["series"].eq("forecast")].copy().sort_values("date")

    if history.empty:
        raise ValueError("CSV must contain rows where series == 'history'.")

    if forecast.empty:
        raise ValueError("CSV must contain rows where series == 'forecast'.")

    forecast_start = history["date"].max()

    # Step number for forecast cone
    if "stepWeek" in forecast.columns and forecast["stepWeek"].notna().any():
        forecast["stepWeek"] = forecast["stepWeek"].fillna(
            pd.Series(np.arange(1, len(forecast) + 1), index=forecast.index)
        )
    else:
        forecast["stepWeek"] = np.arange(1, len(forecast) + 1)

    # -----------------------------
    # Compute summary lines
    # -----------------------------
    two_month_start = forecast_start - pd.Timedelta(days=60)
    recent_history = history[
        (history["date"] > two_month_start) & (history["date"] <= forecast_start)
    ]

    if recent_history.empty:
        historical_monthly_avg = history.tail(42)["price"].mean()
    else:
        historical_monthly_avg = recent_history["price"].mean()

    projected_avg = forecast["price"].mean()
    final_forecast = forecast["price"].iloc[-1]

    # -----------------------------
    # Forecast cone
    # -----------------------------
    # If your CSV already has lower/upper cone columns, use them.
    possible_lower_cols = ["lower", "lo", "forecast_lower", "lower_1sigma"]
    possible_upper_cols = ["upper", "hi", "forecast_upper", "upper_1sigma"]

    lower_col = next((c for c in possible_lower_cols if c in forecast.columns), None)
    upper_col = next((c for c in possible_upper_cols if c in forecast.columns), None)

    if lower_col and upper_col:
        lower = forecast[lower_col].astype(float)
        upper = forecast[upper_col].astype(float)
    else:
        sigma = estimate_weekly_sigma(history)
        cumulative_sigma = sigma * np.sqrt(forecast["stepWeek"].astype(float))
        lower = forecast["price"] * np.exp(-cumulative_sigma)
        upper = forecast["price"] * np.exp(cumulative_sigma)

    # -----------------------------
    # Colors and styling
    # -----------------------------
    bg = "#f3f4f6"
    grid_color = "#d9dce1"

    historical_color = "#183f78"
    cone_color = "#c8b5db"
    forecast_color = "#7b3fa1"

    monthly_avg_color = "#183f78"
    projected_avg_color = "#7b3fa1"
    forecast_start_color = "#9a9a9a"

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.plot(
        history["date"],
        history["price"],
        color=historical_color,
        linewidth=1.55,
        label="Historical price",
    )

    ax.fill_between(
        forecast["date"],
        lower,
        upper,
        color=cone_color,
        alpha=0.55,
        linewidth=0,
        label="±1σ forecast cone",
    )

    ax.plot(
        forecast["date"],
        forecast["price"],
        color=forecast_color,
        linestyle="--",
        marker="o",
        markersize=3.8,
        linewidth=1.25,
        label="6-month forecast path",
    )

    ax.axhline(
        historical_monthly_avg,
        color=monthly_avg_color,
        linestyle="-.",
        linewidth=1.35,
        label=f"Historical 2-month avg ({historical_monthly_avg:.1f}¢)",
    )

    ax.axhline(
        projected_avg,
        color=projected_avg_color,
        linestyle=":",
        linewidth=1.6,
        label=f"Projected avg ({projected_avg:.1f}¢)",
    )

    ax.axvline(
        forecast_start,
        color=forecast_start_color,
        linestyle=":",
        linewidth=1.15,
        label="Forecast start",
    )

    # Final value annotation
    ax.annotate(
        f"{final_forecast:.1f}¢",
        xy=(forecast["date"].iloc[-1], final_forecast),
        xytext=(5, -2),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
        color=forecast_color,
    )

    # -----------------------------
    # Axes formatting
    # -----------------------------
    ax.set_ylabel("Price (¢/lb)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)

    ax.set_ylim(200, 500)
    ax.set_yticks(np.arange(200, 476, 25))

    ax.set_xlim(
        history["date"].min() - pd.Timedelta(days=15),
        forecast["date"].max() + pd.Timedelta(days=25),
    )

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax.grid(axis="y", color=grid_color, linestyle="--", linewidth=0.75, alpha=0.85)
    ax.grid(axis="x", alpha=0)

    ax.tick_params(axis="both", labelsize=9, colors="#333333")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(0.85)

    # -----------------------------
    # Legend
    # -----------------------------
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=historical_color,
            linewidth=1.55,
            label="Historical price",
        ),
        Patch(
            facecolor=cone_color,
            edgecolor="none",
            alpha=0.55,
            label="±1σ forecast cone",
        ),
        Line2D(
            [0],
            [0],
            color=forecast_color,
            linestyle="--",
            marker="o",
            markersize=4,
            linewidth=1.25,
            label="6-month forecast path",
        ),
        Line2D(
            [0],
            [0],
            color=monthly_avg_color,
            linestyle="-.",
            linewidth=1.35,
            label=f"Historical 2-month avg ({historical_monthly_avg:.1f}¢)",
        ),
        Line2D(
            [0],
            [0],
            color=projected_avg_color,
            linestyle=":",
            linewidth=1.6,
            label=f"Projected avg ({projected_avg:.1f}¢)",
        ),
        Line2D(
            [0],
            [0],
            color=forecast_start_color,
            linestyle=":",
            linewidth=1.15,
            label="Forecast start",
        ),
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=8.5,
        borderpad=0.6,
        handlelength=2.4,
    )

    legend.get_frame().set_facecolor(bg)
    legend.get_frame().set_edgecolor("#bbbbbb")
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())

    if show:
        plt.show()

    plt.close(fig)

    return output_path


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    default_csv_path = script_dir / "coffeechartdata.csv"
    default_output_path = script_dir / "coffeechartdata_plot.png"

    parser = argparse.ArgumentParser(description="Create a coffee plot from the local CSV.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(default_csv_path),
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(default_output_path),
        help="Output image path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot after saving.",
    )

    args = parser.parse_args()

    saved_path = make_forecast_plot(
        csv_path=args.csv_path,
        output_path=args.output,
        show=args.show,
    )

    print(f"Saved plot to: {saved_path}")