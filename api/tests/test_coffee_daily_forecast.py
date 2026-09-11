from __future__ import annotations

import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scripts.fetch_logdata import validate_ohlcv_rows

from backend.ml.coffee_daily_forecast import (
    build_features,
    compare_price_series,
    train_direct_forecast,
)


class CoffeeDailyForecastTests(unittest.TestCase):
    def setUp(self) -> None:
        start = date(2005, 1, 3)
        rows = []
        for index in range(900):
            current = start + timedelta(days=index)
            if current.weekday() >= 5:
                continue
            price = 100 + index * 0.03 + 4 * np.sin(index / 20)
            rows.append({"Date": current.isoformat(), "Price": price})
        self.rows = rows

    def test_forecast_has_31_ordered_weekday_points(self) -> None:
        result = train_direct_forecast(self.rows)
        self.assertEqual(len(result.forecast), 31)
        self.assertEqual([row["horizon"] for row in result.forecast], list(range(1, 32)))
        self.assertTrue(all(row["price"] > 0 for row in result.forecast))
        dates = [date.fromisoformat(row["date"]) for row in result.forecast]
        self.assertEqual(dates, sorted(dates))
        self.assertTrue(all(item.weekday() < 5 for item in dates))

    def test_features_are_available_without_future_inputs(self) -> None:
        frame = build_features(pd.DataFrame(self.rows).rename(columns={"Price": "price"}).assign(Date=lambda value: pd.to_datetime(value["Date"])))
        self.assertNotIn("Price", frame.columns)
        self.assertGreater(frame.shape[1], 10)

    def test_compatibility_reports_material_mismatch(self) -> None:
        left = pd.DataFrame([{"Date": "2025-01-01", "price": 100, "open": 99, "high": 101, "low": 98}])
        right = pd.DataFrame([{"Date": "2025-01-01", "price": 900, "open": 900, "high": 901, "low": 899}])
        self.assertFalse(compare_price_series(left, right)["compatible"])

    def test_rejects_zero_ohlc_provider_rows(self) -> None:
        frame = pd.DataFrame([{"Date": "2025-05-15", "open": 0, "high": 0, "low": 0, "close": 0}])
        with self.assertRaisesRegex(ValueError, "invalid OHLC"):
            validate_ohlcv_rows(frame, "coffee")


if __name__ == "__main__":
    unittest.main()
