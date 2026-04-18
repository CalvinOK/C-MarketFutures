# Coffee XGBoost FastAPI for Vercel

This repository wraps your uploaded coffee data pipeline and XGBoost training code in a FastAPI app that Vercel can deploy.

## What it does

- exposes a `GET /predict` endpoint
- computes fresh projections when needed
- caches results by date
- uses **Vercel Blob** for online storage when `BLOB_READ_WRITE_TOKEN` is set
- falls back to local ephemeral storage when Blob is not configured

## Important note about your uploaded files

Your uploaded files were:

- `coffee_data7_merged(3).py`
- `coffee_xgboost_proj4_train(4).py`

The second file is a **training / projection script**, not a serialized trained model file like `.json`, `.bin`, or `.pkl`. This repo therefore **rebuilds the prediction models on demand** from your script instead of loading a pre-trained XGBoost artifact.

If you later want faster responses, replace that behavior with a saved booster file and load it directly.

## Repo layout

```text
api/index.py                 # Vercel entrypoint
backend/main.py              # FastAPI app
backend/pipeline.py          # refresh + caching logic
backend/storage/cache.py     # Vercel Blob / local cache adapter
backend/ml/coffee_data_merged.py
backend/ml/coffee_xgboost_train.py
data/                        # add source data files here
logdata/                     # add source data files here
outputs/                     # generated outputs
vercel.json
requirements.txt
```

## Required data files

Your `coffee_data_merged.py` script expects CSV files under `./data` and `./logdata`.
Before deployment, copy those files into this repo.

## Environment variables

Set these in Vercel Project Settings:

- `BLOB_READ_WRITE_TOKEN` — enables persistent online cache in Vercel Blob
- `CACHE_PREFIX` — optional, defaults to `coffee-predictions`
- `REFRESH_WEEKDAY` — optional, defaults to `4` for Friday
- `DEFAULT_FORCE_ON_REFRESH_DAY` — optional, defaults to `true`
- `ALLOW_LOCAL_FALLBACK` — optional, defaults to `true`

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

## Deploy to Vercel

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial Vercel FastAPI app"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

### 2. Import the repo into Vercel

Vercel can deploy FastAPI apps directly, and FastAPI apps become a single Vercel Function. Official docs also note you can expose the `app` instance from standard entrypoints and optionally configure function duration in `vercel.json`. Vercel Blob provides object storage, and creating a Blob store adds `BLOB_READ_WRITE_TOKEN` to the project environment. citeturn207069view2turn583055view1turn207069view1

### 3. Add your storage and data

- create a Vercel Blob store in the project
- confirm `BLOB_READ_WRITE_TOKEN` exists
- add your source CSV files to `data/` and `logdata/`

## API behavior

### `GET /health`
Shows whether Blob and source folders are available.

### `GET /predict?date=2026-04-17`
Behavior:

- if exact cached data exists and refresh is not forced, return it
- if the date is **not Friday**, return the most recent cached historical result on or before that date
- if the date **is Friday**, recompute when no fresh result exists or when `force_refresh=true`

### Example response

```json
{
  "requested_date": "2026-04-17",
  "resolved_as_of_date": "2026-04-17",
  "cache_hit": false,
  "refresh_reason": "fresh_compute",
  "generated_at": "2026-04-17T00:00:00+00:00",
  "storage_backend": "vercel_blob",
  "source": "xgboost_training_script",
  "latest_price": 385.2,
  "projections": []
}
```

## Recommendation

This works for a first version, but training XGBoost inside a request can still be slow. On Vercel, FastAPI runs as a function and function duration limits still apply, so a stronger production setup is:

- use this API to serve cached results
- refresh with a scheduled job on Fridays
- load a saved XGBoost model artifact instead of retraining every time

That reduces response time and lowers the risk of hitting function limits. citeturn207069view2turn583055view2turn207069view3
