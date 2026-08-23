"""Stock-price and news retrieval helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf


def normalize_ticker(symbol: str) -> str:
    """Return a Yahoo Finance-compatible ticker or raise a helpful error."""
    ticker = symbol.strip().upper()
    if not ticker:
        raise ValueError("Please enter a stock ticker, such as AAPL or MSFT.")
    return ticker


def get_price_history(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch daily historical prices from Yahoo Finance."""
    ticker = normalize_ticker(symbol)
    history = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if history.empty:
        raise ValueError(f"No price data was found for {ticker}. Check the ticker and try again.")
    return history.reset_index()


def _published_at(timestamp: Any) -> datetime | None:
    """Convert Yahoo's Unix publication timestamp to a timezone-aware datetime."""
    if not timestamp:
        return None
    if isinstance(timestamp, str):
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    try:
        return datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def get_news(symbol: str, limit: int = 10) -> pd.DataFrame:
    """Fetch recent Yahoo Finance news headlines for a ticker.

    The Yahoo response format may vary, so entries without a title are skipped.
    """
    ticker = normalize_ticker(symbol)
    articles = yf.Ticker(ticker).news or []
    records: list[dict[str, Any]] = []
    for article in articles[:limit]:
        content_value = article.get("content", article)
        content = content_value if isinstance(content_value, dict) else {}
        title = content.get("title") or article.get("title")
        if not title:
            continue
        provider_value = content.get("provider", {})
        provider = provider_value if isinstance(provider_value, dict) else {}
        canonical_url_value = content.get("canonicalUrl", {})
        canonical_url = canonical_url_value if isinstance(canonical_url_value, dict) else {}
        records.append(
            {
                "title": title,
                "publisher": provider.get("displayName") or article.get("publisher") or "Unknown source",
                "link": canonical_url.get("url") or article.get("link") or "",
                "published_at": _published_at(content.get("pubDate") or article.get("providerPublishTime")),
            }
        )
    return pd.DataFrame(records, columns=["title", "publisher", "link", "published_at"])
