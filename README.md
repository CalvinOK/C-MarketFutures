# Coffee spot price projection with XGBoost

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
