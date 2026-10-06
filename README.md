# Coffee spot price projection with XGBoost

The deployed Flask API uses the lightweight dependencies in `api/requirements.txt`. The offline refresh and training scripts rely on the heavier scientific stack in `api/requirements-offline.txt`.

This script is designed for the weekly output files created by your uploaded data-prep script, especially:

- `kc_weekly_with_fx_cot_weather_overlap_only.csv`
- `kc_weekly_with_fx_cot_weather.csv`
- `kc_weekly_with_fx_cot_overlap_only.csv`
- `kc_weekly_with_fx.csv`
- `kc_continuous_weekly_friday.csv`

## What it does

- infers the weekly date column
- infers the coffee price target column
- builds lag, rolling, return, and calendar features
- trains an `XGBRegressor`
- performs a walk-forward backtest to estimate forecast error
- recursively projects the next 26 weeks (about 6 months)
- saves a chart with historical data plus the projection band

## Run

```bash
python coffee_xgboost_projection.py --input data/kc_weekly_with_fx_cot_weather_overlap_only.csv --outdir outputs
```

## Optional arguments

```bash
python coffee_xgboost_projection.py \
  --input data/kc_weekly_with_fx_cot_weather_overlap_only.csv \
  --target settlement \
  --date-col friday_week \
  --horizon-weeks 26 \
  --history-weeks 104 \
  --outdir outputs
```

## Outputs

- `coffee_spot_projection_6m.csv`
- `coffee_spot_backtest.csv`
- `coffee_spot_projection_6m.png`
- `coffee_xgb_feature_importance.csv`

## Assumptions

For future exogenous variables such as FX, COT, and weather, the script carries the latest observed values forward unless you supply a richer future scenario file. That makes this a baseline conditional projection, not a structural market forecast.

## Website market-data refresh

The scheduled GitHub Actions workflow updates both the current Coffee C snapshot and
the historical data used by the website chart. Configure these repository secrets:

- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY`
- `CFTC_APP_TOKEN`
- `WEBSITE_URL` — the deployed website origin, without a trailing slash
- `CRON_SECRET` — the same value configured in the website deployment

The website's `COFFEE_HISTORY_CSV_URL` or `MARKET_API_BASE_URL` must also be configured
in the deployment so `/api/coffee/update-history` can retrieve the latest daily row.
