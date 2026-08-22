import pytest

from sentiment_analyzer.data import normalize_ticker


def test_normalize_ticker_strips_and_uppercases():
    assert normalize_ticker(" msft ") == "MSFT"


def test_normalize_ticker_rejects_blank_input():
    with pytest.raises(ValueError, match="Please enter"):
        normalize_ticker("   ")
