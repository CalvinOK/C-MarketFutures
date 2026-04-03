# Coffee XGBoost Forecast

This project reads a local `CoffeeCData.csv`, adjusts prices for inflation, augments the series with API-updated external factors, trains an XGBoost regressor, and produces a 5-year monthly forecast plus a black/blue chart.

## What it uses

- Your local `CoffeeCData.csv` for the core coffee price history.
- Alpha Vantage for:
  - `CPI` (inflation adjustment)
  - `COFFEE` (global coffee benchmark)
  - `SUGAR`, `WHEAT`, `COTTON` (correlated crop/soft commodity signals)
  - `WTI` (energy / transport-cost proxy)
  - `FX_MONTHLY` for `BRL/USD` and `COP/USD` (important producer-currency proxies)
- Open-Meteo Historical Weather API for lagged weather.
- Open-Meteo Climate API for future climate-aware weather inputs.

## Expected CSV columns

- `Date`
- `Price`
- `Open`
- `High`
- `Low`
- `Vol.`
- `Change %`

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## API key

Set an Alpha Vantage API key as either an environment variable or in a local `.env` file next to `coffee_forecast.py`.

Examples:

```bash
export ALPHAVANTAGE_API_KEY=your_key_here
```

`.env` format:

```bash
ALPHAVANTAGE_API_KEY=your_key_here
```

Windows PowerShell:

```powershell
$env:ALPHAVANTAGE_API_KEY="your_key_here"
```

## Run

```bash
python coffee_forecast.py --csv CoffeeCData.csv --outdir outputs
```

Optional:

```bash
python coffee_forecast.py --csv CoffeeCData.csv --outdir outputs --horizon-months 60 --climate-model MRI_AGCM3_2_S
```

## Outputs

- `outputs/coffee_model_dataset.csv`
- `outputs/coffee_forecast_5y.csv`
- `outputs/feature_importance.csv`
- `outputs/model_metrics.json`
- `outputs/coffee_history_forecast.png`

## Notes

- Forecasting is done at **monthly** frequency because the external macro / crop APIs are monthly.
- If your raw CSV is daily, the script resamples it to monthly open-high-low-close-volume style features.
- Future weather is taken from Open-Meteo climate-model data; future market variables such as CPI, sugar, wheat, cotton, WTI, and FX are seasonally trend-projected from recent history.
- You can swap regions, weights, or add more producer areas in `data_sources.py`.
