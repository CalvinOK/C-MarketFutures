from __future__ import annotations

import unittest
from datetime import date, timedelta

import numpy as np

from backend.ml.coffee_daily_forecast import (
    build_features,
    normalize_history,
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
        dates, prices = normalize_history(self.rows)
        matrix, names = build_features((dates, prices))
        self.assertGreater(matrix.shape[1], 10)
        self.assertEqual(matrix.shape[1], len(names))



if __name__ == "__main__":
    unittest.main()
