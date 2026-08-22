import pandas as pd
import pytest

from sentiment_analyzer.sentiment import average_sentiment, label_score, score_headlines


class FakeAnalyzer:
    def polarity_scores(self, title):
        return {"compound": 0.8 if "gain" in title else -0.7}


def test_label_score_uses_vader_thresholds():
    assert label_score(0.05) == "Positive"
    assert label_score(-0.05) == "Negative"
    assert label_score(0.0) == "Neutral"


def test_score_headlines_adds_scores_and_labels():
    news = pd.DataFrame({"title": ["Shares gain after earnings", "Stock faces a loss"]})
    result = score_headlines(news, analyzer=FakeAnalyzer())
    assert result["compound"].tolist() == [0.8, -0.7]
    assert result["sentiment"].tolist() == ["Positive", "Negative"]
    assert average_sentiment(result) == pytest.approx(0.05)


def test_score_headlines_handles_empty_news():
    result = score_headlines(pd.DataFrame(columns=["title"]))
    assert result.empty
    assert average_sentiment(result) is None
