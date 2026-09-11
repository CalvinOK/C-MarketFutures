from __future__ import annotations

import unittest
from unittest.mock import patch

import databento_contracts


class DatabentoContractsTests(unittest.TestCase):
    def test_definition_filter_and_ice_symbol_label(self) -> None:
        definitions = [
            {
                "raw_symbol": "KC  FMZ0026!",
                "instrument_class": "F",
                "security_type": "FUT",
                "asset": "KC",
                "expiration": "1794950000000000000",
                "instrument_id": 1,
            },
            {
                "raw_symbol": "KC  FMZ0026-KC  FMH0027",
                "instrument_class": "S",
                "security_type": "FUT",
                "asset": "KC",
                "expiration": "1794950000000000000",
                "instrument_id": 2,
            },
        ]
        with patch.object(databento_contracts, "_request_jsonl", return_value=definitions):
            result = databento_contracts._definition_rows()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "KC  FMZ0026!")
        self.assertEqual(result[0]["year"], 2026)
        self.assertEqual(databento_contracts._display_symbol("x", 2026, "Z"), "KCZ26")

    def test_statistics_are_grouped_by_instrument_id(self) -> None:
        definitions = [{"symbol": "KC  FMZ0026!", "instrumentId": 10}]
        records = [
            {"hd": {"instrument_id": 10, "ts_event": "2000000000000000000"}, "stat_type": 3, "price": "288150000000"},
            {"hd": {"instrument_id": 10, "ts_event": "2000000000000000000"}, "stat_type": 6, "quantity": "1200"},
            {"hd": {"instrument_id": 10, "ts_event": "2000000000000000000"}, "stat_type": 9, "quantity": "5000"},
        ]
        with patch.object(databento_contracts, "_request_jsonl", return_value=records):
            result = databento_contracts._latest_statistics(definitions)
        self.assertEqual(result["KC  FMZ0026!"]["settlement"], "288150000000")
        self.assertEqual(result["KC  FMZ0026!"]["cleared_volume"], "1200")
        self.assertEqual(result["KC  FMZ0026!"]["open_interest"], "5000")


if __name__ == "__main__":
    unittest.main()
