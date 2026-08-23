import pytest

from sentiment_analyzer import data
from sentiment_analyzer.data import get_news, normalize_ticker


def test_normalize_ticker_strips_and_uppercases():
    assert normalize_ticker(" msft ") == "MSFT"


def test_normalize_ticker_rejects_blank_input():
    with pytest.raises(ValueError, match="Please enter"):
        normalize_ticker("   ")


@pytest.mark.parametrize(
    ("content", "expected_publisher", "expected_link"),
    [
        (None, "Fallback publisher", "https://example.com/fallback"),
        ("not a dictionary", "Fallback publisher", "https://example.com/fallback"),
        ({"title": "Nested headline", "provider": None, "canonicalUrl": None}, "Fallback publisher", "https://example.com/fallback"),
        ({"title": "Nested headline", "provider": "not a dictionary", "canonicalUrl": "not a dictionary"}, "Fallback publisher", "https://example.com/fallback"),
    ],
)
def test_get_news_handles_malformed_nested_values(monkeypatch, content, expected_publisher, expected_link):
    article = {
        "title": "Fallback headline",
        "publisher": "Fallback publisher",
        "link": "https://example.com/fallback",
        "content": content,
    }

    class FakeTicker:
        news = [article]

    monkeypatch.setattr(data.yf, "Ticker", lambda symbol: FakeTicker())

    news = get_news("msft")

    assert news.to_dict("records") == [
        {
            "title": "Nested headline" if isinstance(content, dict) else "Fallback headline",
            "publisher": expected_publisher,
            "link": expected_link,
            "published_at": None,
        }
    ]


def test_get_news_handles_missing_content_provider_and_canonical_url(monkeypatch):
    article = {"title": "Legacy headline", "publisher": "Legacy publisher", "link": "https://example.com/legacy"}

    class FakeTicker:
        news = [article]

    monkeypatch.setattr(data.yf, "Ticker", lambda symbol: FakeTicker())

    news = get_news("msft")

    assert news.to_dict("records") == [
        {
            "title": "Legacy headline",
            "publisher": "Legacy publisher",
            "link": "https://example.com/legacy",
            "published_at": None,
        }
    ]
