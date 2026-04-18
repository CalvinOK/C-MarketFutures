from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.config import ALLOW_LOCAL_FALLBACK, BLOB_TOKEN, CACHE_PREFIX, OUTPUT_DIR

try:
    from vercel.blob import put, list as blob_list
except Exception:  # pragma: no cover
    put = None
    blob_list = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


class CacheStore:
    def __init__(self) -> None:
        self.local_dir = OUTPUT_DIR / 'cache'
        self.local_dir.mkdir(parents=True, exist_ok=True)

    @property
    def backend_name(self) -> str:
        if BLOB_TOKEN and put is not None and blob_list is not None:
            return 'vercel_blob'
        if ALLOW_LOCAL_FALLBACK:
            return 'local_ephemeral'
        return 'none'

    def _blob_path(self, key: str) -> str:
        return f"{CACHE_PREFIX}/{key}.json"

    def _local_path(self, key: str) -> Path:
        return self.local_dir / f'{key}.json'

    def read(self, key: str) -> dict[str, Any] | None:
        if self.backend_name == 'vercel_blob':
            return self._read_blob(key)
        if self.backend_name == 'local_ephemeral':
            path = self._local_path(key)
            return json.loads(path.read_text()) if path.exists() else None
        return None

    def write(self, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.backend_name == 'vercel_blob':
            return self._write_blob(key, payload)
        if self.backend_name == 'local_ephemeral':
            path = self._local_path(key)
            path.write_text(json.dumps(payload, indent=2))
            return {'storage_backend': 'local_ephemeral', 'path': str(path)}
        raise RuntimeError('No storage backend configured')

    def latest_on_or_before(self, requested_date: str) -> dict[str, Any] | None:
        if self.backend_name == 'vercel_blob':
            return self._latest_blob_on_or_before(requested_date)
        if self.backend_name == 'local_ephemeral':
            candidates: list[tuple[str, Path]] = []
            for path in self.local_dir.glob('*.json'):
                key = path.stem
                if key <= requested_date:
                    candidates.append((key, path))
            if not candidates:
                return None
            _, best_path = sorted(candidates)[-1]
            return json.loads(best_path.read_text())
        return None

    def _read_blob(self, key: str) -> dict[str, Any] | None:
        if blob_list is None or requests is None:
            return None
        result = blob_list(prefix=self._blob_path(key), token=BLOB_TOKEN)
        blobs = getattr(result, 'blobs', None) or result.get('blobs', [])
        if not blobs:
            return None
        url = blobs[0].url if hasattr(blobs[0], 'url') else blobs[0]['url']
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _latest_blob_on_or_before(self, requested_date: str) -> dict[str, Any] | None:
        if blob_list is None or requests is None:
            return None
        result = blob_list(prefix=f'{CACHE_PREFIX}/', token=BLOB_TOKEN)
        blobs = getattr(result, 'blobs', None) or result.get('blobs', [])
        dated: list[tuple[str, str]] = []
        for blob in blobs:
            pathname = blob.pathname if hasattr(blob, 'pathname') else blob.get('pathname', '')
            if not pathname.endswith('.json'):
                continue
            key = pathname.split('/')[-1].removesuffix('.json')
            if key <= requested_date:
                url = blob.url if hasattr(blob, 'url') else blob['url']
                dated.append((key, url))
        if not dated:
            return None
        _, url = sorted(dated)[-1]
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _write_blob(self, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        if put is None:
            raise RuntimeError('vercel.blob SDK not available')
        result = put(
            self._blob_path(key),
            json.dumps(payload, indent=2).encode('utf-8'),
            access='private',
            content_type='application/json',
            add_random_suffix=False,
            overwrite=True,
            token=BLOB_TOKEN,
        )
        url = result.url if hasattr(result, 'url') else result.get('url')
        return {'storage_backend': 'vercel_blob', 'url': url}
