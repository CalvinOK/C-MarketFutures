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
API_OUTPUTS = API_ROOT / "outputs"
WEBSITE_PUBLIC_DATA = PROJECT_ROOT / "website" / "public" / "data"
API_PUBLIC_DATA = API_ROOT / "public" / "data"


def _python_executable() -> str:
    return sys.executable


def _run_step(label: str, script_path: Path, cwd: Path, timeout_seconds: int = 1800) -> None:
    cmd = [_python_executable(), str(script_path)]
    print(f"[pipeline] Running {label}: {' '.join(cmd)}")

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{API_ROOT}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(API_ROOT)

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


def main() -> None:
    if not API_ROOT.exists():
        raise FileNotFoundError(f"Missing API folder: {API_ROOT}")

    _run_step("data fetch", API_ROOT / "data_fetch.py", API_ROOT)
    _run_step(
        "data merge",
        API_ROOT / "backend" / "ml" / "coffee_data_merged.py",
        API_ROOT,
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

    print("[pipeline] Completed projection pipeline refresh from api/")


if __name__ == "__main__":
    main()