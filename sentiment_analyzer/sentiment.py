"""VADER sentiment scoring helpers."""

from __future__ import annotations

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


def get_analyzer() -> SentimentIntensityAnalyzer:
    """Create a VADER analyzer, downloading its small lexicon when needed."""
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def label_score(compound: float) -> str:
    """Translate VADER's compound score into its standard sentiment band."""
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def score_headlines(news: pd.DataFrame, analyzer: SentimentIntensityAnalyzer | None = None) -> pd.DataFrame:
    """Add VADER compound scores and readable labels to a news table."""
    scored = news.copy()
    if scored.empty:
        scored["compound"] = pd.Series(dtype=float)
        scored["sentiment"] = pd.Series(dtype=str)
        return scored
    vader = analyzer or get_analyzer()
    scored["compound"] = scored["title"].map(lambda title: vader.polarity_scores(title)["compound"])
    scored["sentiment"] = scored["compound"].map(label_score)
    return scored


def average_sentiment(scored_news: pd.DataFrame) -> float | None:
    """Return the average VADER compound score, if headlines are available."""
    if scored_news.empty:
        return None
    return float(scored_news["compound"].mean())
