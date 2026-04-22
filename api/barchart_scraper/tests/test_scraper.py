from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scraper import (
    RawRow,
    derive_curve_shape,
    derive_snapshot,
    extract_rows_from_table_html,
    month_year_from_symbol,
    parse_compact_number,
    parse_float,
    parse_pct,
    symbol_to_expiry_date,
    transform_rows_to_contracts,
    validate_contracts,
)


def test_parse_compact_number():
    assert parse_compact_number("73.6K") == 73600
    assert parse_compact_number("1.2M") == 1200000
    assert parse_compact_number("369,400") == 369400
    assert parse_compact_number("N/A") is None


def test_parse_float_and_pct():
    assert parse_float("193.40") == 193.40
    assert parse_float("+1.25") == 1.25
    assert parse_float("(1.25)") == -1.25
    assert parse_pct("0.65%") == 0.65


def test_symbol_helpers():
    assert symbol_to_expiry_date("KCK26") == "2026-05-01"
    assert symbol_to_expiry_date("KCN26") == "2026-07-01"
    month, year = month_year_from_symbol("KCU26")
    assert month == "Sep"
    assert year == 2026


def test_extract_rows_from_table_html():
    html = """
    <html>
      <body>
        <table>
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Last</th>
              <th>Change</th>
              <th>% Change</th>
              <th>Volume</th>
              <th>Open Interest</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>KCK26</td>
              <td>193.40</td>
              <td>1.25</td>
              <td>0.65%</td>
              <td>73.6K</td>
              <td>369.4K</td>
            </tr>
            <tr>
              <td>KCN26</td>
              <td>195.10</td>
              <td>0.50</td>
              <td>0.26%</td>
              <td>25.0K</td>
              <td>180.2K</td>
            </tr>
          </tbody>
        </table>
      </body>
    </html>
    """
    rows = extract_rows_from_table_html(html)
    assert len(rows) == 2
    assert rows[0].symbol == "KCK26"
    assert rows[0].volume == "73.6K"


def test_transform_and_snapshot():
    rows = [
        RawRow(
            symbol="KCK26",
            last_price="193.40",
            price_change="1.25",
            price_change_pct="0.65%",
            volume="73.6K",
            open_interest="369.4K",
        ),
        RawRow(
            symbol="KCN26",
            last_price="195.10",
            price_change="0.50",
            price_change_pct="0.26%",
            volume="25.0K",
            open_interest="180.2K",
        ),
    ]
    contracts = transform_rows_to_contracts(rows, "2026-04-22T00:00:00+00:00", logger=_StubLogger())
    validate_contracts(contracts)

    assert contracts[0]["symbol"] == "KCK26"
    assert contracts[0]["volume"] == 73600
    assert contracts[0]["open_interest"] == 369400

    curve = derive_curve_shape(contracts)
    assert curve == "contango"

    snapshot = derive_snapshot(contracts)
    assert snapshot["frontSymbol"] == "KCK26"
    assert snapshot["totalVolume"] == 98600


class _StubLogger:
    def info(self, *_args, **_kwargs):
        return None
