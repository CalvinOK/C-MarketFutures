from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
API_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# On Vercel, /var/task is read-only. RUNTIME_DATA_DIR is set to /tmp/... by
# runner.py and is the only writable location. When not on Vercel, fall back
# to the local outputs/ directory so local runs still work unchanged.
_runtime = os.environ.get("RUNTIME_DATA_DIR", "").strip()
API_OUTPUTS = Path(_runtime) if _runtime else API_ROOT / "outputs"
API_OUTPUTS.mkdir(parents=True, exist_ok=True)

# Detect serverless environment so we can skip steps that require a writable
# source tree (logdata fetch) or that are already unnecessary (sync).
_IS_SERVERLESS = bool(
    os.environ.get("VERCEL")
    or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    or os.environ.get("AWS_EXECUTION_ENV")
    or str(API_ROOT).startswith("/var/task")
)

WEBSITE_PUBLIC_DATA = PROJECT_ROOT / "website" / "public" / "data"
API_PUBLIC_DATA = API_ROOT / "public" / "data"


def _python_executable() -> str:
    return sys.executable


def _run_step(
    label: str,
    script_path: Path,
    cwd: Path,
    timeout_seconds: int = 1800,
    extra_env: dict[str, str] | None = None,
) -> None:
    cmd = [_python_executable(), str(script_path)]
    print(f"[pipeline] Running {label}: {' '.join(cmd)}")

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{API_ROOT}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(API_ROOT)

    # Propagate RUNTIME_DATA_DIR to sub-scripts so config.py and data_fetch.py
    # also write to the writable /tmp path.
    if _runtime:
        env["RUNTIME_DATA_DIR"] = _runtime

    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )

    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.returncode != 0:
        if proc.stderr.strip():
            print(proc.stderr.strip())
        raise RuntimeError(f"Step failed: {label} (exit={proc.returncode})")


def _build_history_csv() -> Path:
    merged_path = API_OUTPUTS / "coffee_model_dataset_merged.csv"
    if not merged_path.exists():
        raise FileNotFoundError(f"Missing merged dataset: {merged_path}")

    out_path = API_OUTPUTS / "coffee_xgb_proj4_history.csv"
    with merged_path.open(newline="", encoding="utf-8") as infile, \
         out_path.open("w", newline="", encoding="utf-8") as outfile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None or not {"Date", "coffee_c"}.issubset(reader.fieldnames):
            raise ValueError("Merged dataset missing required columns: Date, coffee_c")
        writer = csv.DictWriter(outfile, fieldnames=["Date", "coffee_c"])
        writer.writeheader()
        for row in reader:
            if row.get("Date", "").strip() and row.get("coffee_c", "").strip():
                writer.writerow({"Date": row["Date"], "coffee_c": row["coffee_c"]})
    return out_path


def _sync_outputs(paths: list[Path]) -> None:
    # On Vercel, outputs are already in RUNTIME_DATA_DIR which the Flask API
    # searches first. Writing to WEBSITE_PUBLIC_DATA / API_PUBLIC_DATA would
    # hit read-only /var/task and fail, so skip the sync entirely.
    if _IS_SERVERLESS:
        print("[pipeline] Serverless environment — skipping filesystem sync.")
        return

    WEBSITE_PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
    API_PUBLIC_DATA.mkdir(parents=True, exist_ok=True)

    for path in paths:
        if not path.exists():
            continue
        website_dst = WEBSITE_PUBLIC_DATA / path.name
        api_dst = API_PUBLIC_DATA / path.name

        if path.resolve() != website_dst.resolve():
            shutil.copy2(path, website_dst)
        if path.resolve() != api_dst.resolve():
            shutil.copy2(path, api_dst)


def _prepare_runtime_logdata() -> dict[str, str]:
    """
    On serverless, copy the bundled logdata CSVs to a writable /tmp directory
    so fetch_logdata.py can extend them with an incremental Databento fetch.
    Returns a dict of extra env vars to pass to subprocesses.
    """
    if not _IS_SERVERLESS or not _runtime:
        return {}

    runtime_logdata = Path(_runtime) / "logdata"
    runtime_logdata.mkdir(parents=True, exist_ok=True)

    bundled_logdata = API_ROOT / "logdata"
    if bundled_logdata.exists():
        for src in bundled_logdata.glob("*.csv"):
            dst = runtime_logdata / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"[pipeline] Copied bundled logdata: {src.name}")

    return {"RUNTIME_LOGDATA_DIR": str(runtime_logdata)}


def main() -> None:
    if not API_ROOT.exists():
        raise FileNotFoundError(f"Missing API folder: {API_ROOT}")

    # On serverless, copy bundled logdata to /tmp then run an incremental fetch
    # (only the days missing since the last bundled date — typically just a few
    # rows, so this completes in seconds).  On local dev the full fetch runs
    # against the bundled logdata directory directly.
    logdata_env = _prepare_runtime_logdata()
    _run_step("logdata fetch", SCRIPT_PATH.parent / "fetch_logdata.py", API_ROOT, extra_env=logdata_env)

    _run_step("data fetch", API_ROOT / "data_fetch.py", API_ROOT)
    _run_step(
        "data merge",
        API_ROOT / "backend" / "ml" / "coffee_data_merged.py",
        API_ROOT,
        extra_env=logdata_env,
    )
    _run_step(
        "xgboost train",
        API_ROOT / "backend" / "ml" / "coffee_xgboost_train.py",
        API_ROOT,
    )

    history_path = _build_history_csv()

    _sync_outputs(
        [
            history_path,
            API_OUTPUTS / "coffee_xgb_proj4_rolling_path.csv",
            API_OUTPUTS / "coffee_xgb_proj4_latest_projection.csv",
            API_OUTPUTS / "coffee_xgb_proj4_feature_importance.csv",
            WEBSITE_PUBLIC_DATA / "contracts.json",
            WEBSITE_PUBLIC_DATA / "snapshot.json",
        ]
    )

    print(f"[pipeline] Outputs written to: {API_OUTPUTS}")
    print("[pipeline] Completed projection pipeline refresh.")


if __name__ == "__main__":
    main()
