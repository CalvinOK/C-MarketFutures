from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from databento_contracts import DatabentoProviderError, _request_jsonl


class DatabentoProviderErrorTests(unittest.TestCase):
    def test_402_is_insufficient_budget(self) -> None:
        response = Mock(ok=False, status_code=402, text='secret response body')
        with patch('databento_contracts.requests.post', return_value=response):
            with self.assertRaises(DatabentoProviderError) as context:
                _request_jsonl({})
        self.assertEqual(context.exception.reason, 'insufficient_budget')
        self.assertEqual(context.exception.status, 402)
        self.assertNotIn('secret', str(context.exception))

    def test_auth_and_entitlement_categories(self) -> None:
        for status, reason in ((401, 'authentication_failure'), (403, 'entitlement_failure')):
            response = Mock(ok=False, status_code=status, text='private')
            with patch('databento_contracts.requests.post', return_value=response):
                with self.assertRaises(DatabentoProviderError) as context:
                    _request_jsonl({})
            self.assertEqual(context.exception.reason, reason)

    def test_timeout_is_safe_category(self) -> None:
        with patch('databento_contracts.requests.post', side_effect=requests.Timeout()):
            with self.assertRaises(DatabentoProviderError) as context:
                _request_jsonl({})
        self.assertEqual(context.exception.reason, 'timeout')


if __name__ == '__main__':
    unittest.main()
