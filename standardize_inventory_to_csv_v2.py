#!/usr/bin/env python3
"""Standardize ICE coffee inventory Excel reports into a single CSV.

What it does
------------
- Scans a directory for .xlsx and .xls files.
- Extracts the daily inventory tables from each workbook.
- Standardizes them into a tidy / long CSV with columns:
    report_date, section, country, warehouse, bags, source_file
- Fills in missing calendar days by carrying the last known value forward.
- Lets you keep only data on or after a chosen output start date.

Notes
-----
- .xlsx files are read with openpyxl.
- .xls files are read with xlrd. Install it with: pip install xlrd
  (or: .venv/bin/pip install xlrd)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


SECTION_NAMES = [
    "TOTAL BAGS CERTIFIED",
    "TRANSITION BAGS CERTIFIED",
]

STOP_MARKERS = {
    "TODAY'S GRADING SUMMARY",
    "BAGS PASSED GRADING",
}

DATE_PATTERNS = [
    re.compile(r"(20\d{2})(\d{2})(\d{2})"),  # 20250424
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),  # 2025-04-24
    re.compile(r"(\d{2})_(\d{2})_(\d{4})"),  # 04_24_2025
    re.compile(r"(\d{2})-(\d{2})-(\d{4})"),  # 04-24-2025
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/inventory data"),
        help='Directory containing the Excel reports (.xlsx and/or .xls). Default: "data/inventory data"',
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/inventory data/standardized_inventory.csv"),
        help='Path to the output CSV file. Default: "data/inventory data/standardized_inventory.csv"',
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-04-24",
        help="Only keep output rows on/after this date (YYYY-MM-DD). Default: 2025-04-24",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional output end date (YYYY-MM-DD). Default: today's date.",
    )
    parser.add_argument(
        "--include-source-before-start",
        action="store_true",
        default=True,
        help=(
            "Use files before --start-date to seed forward-fill values. "
            "Recommended when the first kept day may need prior-day values."
        ),
    )
    return parser.parse_args()


def parse_iso_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def coerce_number(value: object) -> Optional[int]:
    if pd.isna(value) or value == "":
        return None
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    text = str(value).strip().replace(",", "")
    if text == "":
        return None
    try:
        return int(round(float(text)))
    except ValueError:
        return None


def extract_date_from_filename(path: Path) -> Optional[dt.date]:
    name = path.stem
    for pattern in DATE_PATTERNS:
        match = pattern.search(name)
        if not match:
            continue
        groups = match.groups()
        try:
            if len(groups[0]) == 4:
                year, month, day = map(int, groups)
            else:
                month, day, year = map(int, groups)
            return dt.date(year, month, day)
        except ValueError:
            continue
    return None


def load_workbook_as_df(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".xlsx":
        return pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")

    if suffix == ".xls":
        try:
            import xlrd  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "This file is .xls, which needs the 'xlrd' package.\n"
                "Install it with:\n"
                "  .venv/bin/pip install xlrd\n"
                "or:\n"
                "  pip install xlrd"
            ) from exc

        return pd.read_excel(path, sheet_name=0, header=None, engine="xlrd")

    raise ValueError(f"Unsupported file type: {path.name}")


def extract_report_date(df: pd.DataFrame, source_path: Path) -> dt.date:
    for value in df.iloc[:10, 0].tolist():
        text = normalize_text(value)
        if text.lower().startswith("as of:"):
            cleaned = re.sub(r"^As of:\s*", "", text, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+\d{1,2}:\d{2}:\d{2}[AP]M.*$", "", cleaned)
            try:
                return dt.datetime.strptime(cleaned.strip(), "%b %d, %Y").date()
            except ValueError:
                pass

    file_date = extract_date_from_filename(source_path)
    if file_date is not None:
        return file_date

    raise ValueError(f"Could not determine report date for file: {source_path}")


def find_row(df: pd.DataFrame, needle: str) -> Optional[int]:
    target = needle.strip().upper()
    for i in range(len(df)):
        row_values = [normalize_text(v).upper() for v in df.iloc[i].tolist()]
        if target in row_values:
            return i
        if row_values and row_values[0] == target:
            return i
    return None


def parse_section(df: pd.DataFrame, report_date: dt.date, source_file: str, section_name: str) -> List[dict]:
    start_row = find_row(df, section_name)
    if start_row is None:
        return []

    header_row_idx = None
    for r in range(start_row + 1, min(start_row + 8, len(df))):
        row_values = [normalize_text(v) for v in df.iloc[r].tolist()]
        non_empty = [v for v in row_values if v]
        if "Total" in non_empty and len(non_empty) >= 2:
            header_row_idx = r
            break

    if header_row_idx is None:
        raise ValueError(f"Could not find header row for section '{section_name}' in {source_file}")

    headers = [normalize_text(v) for v in df.iloc[header_row_idx].tolist()]

    records: List[dict] = []
    for r in range(header_row_idx + 1, len(df)):
        first_cell = normalize_text(df.iat[r, 0])
        first_upper = first_cell.upper()

        if first_upper == "TOTAL IN BAGS":
            break
        if first_upper in STOP_MARKERS:
            break
        if first_cell == "":
            continue

        country = first_cell
        for c in range(1, len(headers)):
            warehouse = headers[c]
            if warehouse == "":
                continue
            bags = coerce_number(df.iat[r, c])
            if bags is None:
                continue
            records.append(
                {
                    "report_date": report_date,
                    "section": section_name,
                    "country": country,
                    "warehouse": warehouse,
                    "bags": bags,
                    "source_file": source_file,
                }
            )

    return records


def parse_report(path: Path) -> List[dict]:
    df = load_workbook_as_df(path)
    report_date = extract_report_date(df, path)

    all_records: List[dict] = []
    for section_name in SECTION_NAMES:
        all_records.extend(parse_section(df, report_date, path.name, section_name))
    return all_records


def scan_excel_files(input_dir: Path) -> List[Path]:
    return sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in {".xlsx", ".xls"}]
    )


def build_standardized_csv(
    input_dir: Path,
    output_csv: Path,
    start_date: dt.date,
    end_date: dt.date,
    include_source_before_start: bool,
) -> pd.DataFrame:
    files = scan_excel_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No .xlsx or .xls files found in: {input_dir}")

    records: List[dict] = []
    parsed_files = 0

    for path in files:
        try:
            report_records = parse_report(path)
        except Exception as exc:
            print(f"WARNING: skipping {path.name}: {exc}", file=sys.stderr)
            continue

        parsed_files += 1
        if include_source_before_start:
            records.extend(report_records)
        else:
            report_date = report_records[0]["report_date"] if report_records else None
            if report_date is not None and report_date >= start_date:
                records.extend(report_records)

    if parsed_files == 0:
        raise ValueError(
            "No files were successfully parsed.\n"
            "If your files are .xls, install xlrd first with:\n"
            "  .venv/bin/pip install xlrd"
        )

    if not records:
        raise ValueError("Files were parsed, but no report records were extracted.")

    df = pd.DataFrame.from_records(records)
    df["report_date"] = pd.to_datetime(df["report_date"])

    series_cols = ["section", "country", "warehouse"]
    dedupe_cols = ["report_date"] + series_cols
    df = df.sort_values(["report_date", "source_file"]).drop_duplicates(dedupe_cols, keep="last")

    all_days = pd.date_range(start=start_date, end=end_date, freq="D")
    filled_frames: List[pd.DataFrame] = []

    for keys, group in df.groupby(series_cols, dropna=False):
        group = group.sort_values("report_date").set_index("report_date")
        group = group[["bags", "source_file"]]

        reindex_start = min(group.index.min(), pd.Timestamp(start_date))
        full_index = pd.date_range(start=reindex_start, end=end_date, freq="D")
        expanded = group.reindex(full_index)
        expanded[["bags", "source_file"]] = expanded[["bags", "source_file"]].ffill()

        expanded = expanded.loc[all_days]
        expanded = expanded.reset_index().rename(columns={"index": "report_date"})
        expanded["section"] = keys[0]
        expanded["country"] = keys[1]
        expanded["warehouse"] = keys[2]
        filled_frames.append(expanded)

    out = pd.concat(filled_frames, ignore_index=True)
    out = out[["report_date", "section", "country", "warehouse", "bags", "source_file"]]
    out = out.sort_values(["report_date", "section", "country", "warehouse"])
    out["report_date"] = out["report_date"].dt.strftime("%Y-%m-%d")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    return out


def main() -> None:
    args = parse_args()

    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date) if args.end_date else dt.date.today()

    if end_date < start_date:
        raise ValueError("--end-date cannot be earlier than --start-date")

    output = build_standardized_csv(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        start_date=start_date,
        end_date=end_date,
        include_source_before_start=args.include_source_before_start,
    )

    min_date = output["report_date"].min()
    max_date = output["report_date"].max()
    print(f"Wrote {len(output):,} rows to {args.output_csv}")
    print(f"Date range in output: {min_date} to {max_date}")
    print(f"Unique sections: {output['section'].nunique()}")
    print(f"Unique countries: {output['country'].nunique()}")
    print(f"Unique warehouses: {output['warehouse'].nunique()}")


if __name__ == "__main__":
    main()
