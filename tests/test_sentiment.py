import pandas as pd
import pytest

from sentiment_analyzer import sentiment
from sentiment_analyzer.sentiment import (
    FinBertAnalyzer,
    average_sentiment,
    get_sentiment_analyzer,
    label_score,
    score_headlines,
)


class FakeVaderAnalyzer:
    name = "VADER (fallback)"

    def score(self, title):
        return 0.8 if "gain" in title else -0.7


class FakeFinBertPipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, title, **kwargs):
        self.calls.append((title, kwargs))
        if "gain" in title:
            return [{"label": "positive", "score": 0.91}]
        if "loss" in title:
            return [{"label": "negative", "score": 0.88}]
        return [{"label": "neutral", "score": 0.99}]


@pytest.fixture(autouse=True)
def clear_analyzer_cache():
    get_sentiment_analyzer.cache_clear()
    yield
    get_sentiment_analyzer.cache_clear()


def test_label_score_uses_sentiment_thresholds():
    assert label_score(0.05) == "Positive"
    assert label_score(-0.05) == "Negative"
    assert label_score(0.0) == "Neutral"


def test_finbert_is_primary_and_cached(monkeypatch):
    pipeline = FakeFinBertPipeline()
    monkeypatch.setattr(sentiment, "_load_finbert_pipeline", lambda: pipeline)

    analyzer = get_sentiment_analyzer()

    assert isinstance(analyzer, FinBertAnalyzer)
    assert analyzer is get_sentiment_analyzer()
    assert analyzer.name == "FinBERT (ProsusAI/finbert)"


def test_score_headlines_with_mocked_finbert_requires_no_model_download(monkeypatch):
    pipeline = FakeFinBertPipeline()
    monkeypatch.setattr(sentiment, "_load_finbert_pipeline", lambda: pipeline)
    news = pd.DataFrame({"title": ["Shares gain after earnings", "Stock faces a loss", "Company holds guidance"]})

    result = score_headlines(news)

    assert result["compound"].tolist() == [0.91, -0.88, 0.0]
    assert result["sentiment"].tolist() == ["Positive", "Negative", "Neutral"]
    assert result.attrs["sentiment_engine"] == "FinBERT (ProsusAI/finbert)"
    assert all(kwargs == {"truncation": True} for _, kwargs in pipeline.calls)


def test_vader_is_used_when_finbert_cannot_load(monkeypatch):
    monkeypatch.setattr(sentiment, "_load_finbert_pipeline", lambda: (_ for _ in ()).throw(OSError("offline")))
    fake_vader = FakeVaderAnalyzer()
    monkeypatch.setattr(sentiment, "VaderAnalyzer", lambda: fake_vader)

    analyzer = get_sentiment_analyzer()
    result = score_headlines(pd.DataFrame({"title": ["Shares gain after earnings"]}), analyzer=analyzer)

    assert analyzer is fake_vader
    assert result["compound"].tolist() == [0.8]
    assert result.attrs["sentiment_engine"] == "VADER (fallback)"


def test_score_headlines_handles_empty_news():
    result = score_headlines(pd.DataFrame(columns=["title"]), analyzer=FakeVaderAnalyzer())
    assert result.empty
    assert result.attrs["sentiment_engine"] == "VADER (fallback)"
    assert average_sentiment(result) is None


def test_average_sentiment():
    scored = pd.DataFrame({"compound": [0.8, -0.7]})
    assert average_sentiment(scored) == pytest.approx(0.05)
