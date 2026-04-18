from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
LOGDATA_DIR = BASE_DIR / 'logdata'
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLOB_TOKEN = os.getenv('BLOB_READ_WRITE_TOKEN')
CACHE_PREFIX = os.getenv('CACHE_PREFIX', 'coffee-predictions')
REFRESH_WEEKDAY = int(os.getenv('REFRESH_WEEKDAY', '4'))  # Friday=4
DEFAULT_FORCE_ON_REFRESH_DAY = os.getenv('DEFAULT_FORCE_ON_REFRESH_DAY', 'true').lower() == 'true'
ALLOW_LOCAL_FALLBACK = os.getenv('ALLOW_LOCAL_FALLBACK', 'true').lower() == 'true'
