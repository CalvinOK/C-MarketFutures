from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
LOGDATA_DIR = BASE_DIR / 'logdata'

# On Vercel (serverless), the source tree is read-only at /var/task.
# RUNTIME_DATA_DIR points to a writable directory (/tmp/...) set by runner.py.
_runtime = os.getenv('RUNTIME_DATA_DIR')
OUTPUT_DIR = Path(_runtime) if _runtime else BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RUNTIME_LOGDATA_DIR is set by run_old_projection_pipeline.py on serverless so
# that fresh logdata fetched at runtime can be written to /tmp and read back by
# coffee_data_merged.py without touching the read-only /var/task tree.
_runtime_logdata = os.getenv('RUNTIME_LOGDATA_DIR')
RUNTIME_LOGDATA_DIR: Path | None = Path(_runtime_logdata) if _runtime_logdata else None

BLOB_TOKEN = os.getenv('BLOB_READ_WRITE_TOKEN')
CACHE_PREFIX = os.getenv('CACHE_PREFIX', 'coffee-predictions')
REFRESH_WEEKDAY = int(os.getenv('REFRESH_WEEKDAY', '4'))  # Friday=4
DEFAULT_FORCE_ON_REFRESH_DAY = os.getenv('DEFAULT_FORCE_ON_REFRESH_DAY', 'true').lower() == 'true'
ALLOW_LOCAL_FALLBACK = os.getenv('ALLOW_LOCAL_FALLBACK', 'true').lower() == 'true'
DATA_FETCH_TIMEOUT_SECONDS = int(os.getenv('DATA_FETCH_TIMEOUT_SECONDS', '120'))
