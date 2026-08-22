"""FinBERT-first sentiment scoring with a VADER fallback."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

FINBERT_MODEL = "ProsusAI/finbert"


class HeadlineAnalyzer(Protocol):
    """Common interface for the available headline sentiment engines."""

    name: str

    def score(self, title: str) -> float:
        """Return a signed sentiment score in the inclusive range [-1, 1]."""


class VaderAnalyzer:
    """Adapter around VADER's compound sentiment score."""

    name = "VADER (fallback)"

    def __init__(self, analyzer: SentimentIntensityAnalyzer | None = None) -> None:
        self._analyzer = analyzer or get_analyzer()

    def score(self, title: str) -> float:
        return float(self._analyzer.polarity_scores(title)["compound"])


class FinBertAnalyzer:
    """Adapter that converts FinBERT label confidence into a signed score."""

    name = "FinBERT (ProsusAI/finbert)"

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    def score(self, title: str) -> float:
        result = self._pipeline(title, truncation=True)[0]
        confidence = float(result["score"])
        label = str(result["label"]).upper()
        if label == "POSITIVE":
            return confidence
        if label == "NEGATIVE":
            return -confidence
        return 0.0


def get_analyzer() -> SentimentIntensityAnalyzer:
    """Create a VADER analyzer, downloading its small lexicon when needed."""
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def _load_finbert_pipeline() -> Any:
    """Load FinBERT only when it is first needed by an analysis."""
    from transformers import pipeline

    return pipeline("sentiment-analysis", model=FINBERT_MODEL, tokenizer=FINBERT_MODEL)


@lru_cache(maxsize=1)
def get_sentiment_analyzer() -> HeadlineAnalyzer:
    """Return cached FinBERT, or VADER when FinBERT cannot be loaded."""
    try:
        return FinBertAnalyzer(_load_finbert_pipeline())
    except Exception:
        # Missing optional ML dependencies, model download failures, and model-loading
        # errors should not prevent the dashboard from returning a sentiment result.
        return VaderAnalyzer()


def sentiment_engine_name(analyzer: HeadlineAnalyzer) -> str:
    """Return a human-readable name for the analyzer currently in use."""
    return analyzer.name


def label_score(compound: float) -> str:
    """Translate a signed score into the dashboard's sentiment bands."""
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def score_headlines(news: pd.DataFrame, analyzer: HeadlineAnalyzer | None = None) -> pd.DataFrame:
    """Add signed scores, labels, and the active engine name to a news table."""
    scored = news.copy()
    active_analyzer = analyzer or get_sentiment_analyzer()
    scored["compound"] = (
        pd.Series(dtype=float)
        if scored.empty
        else scored["title"].map(active_analyzer.score)
    )
    scored["sentiment"] = scored["compound"].map(label_score).astype(str)
    scored.attrs["sentiment_engine"] = sentiment_engine_name(active_analyzer)
    return scored


def average_sentiment(scored_news: pd.DataFrame) -> float | None:
    """Return the average signed sentiment score, if headlines are available."""
    if scored_news.empty:
        return None
    return float(scored_news["compound"].mean())
