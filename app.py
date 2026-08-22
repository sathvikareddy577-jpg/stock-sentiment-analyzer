"""A beginner-friendly Streamlit dashboard for stock prices and news sentiment."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from sentiment_analyzer.data import get_news, get_price_history, normalize_ticker
from sentiment_analyzer.sentiment import average_sentiment, label_score, score_headlines

st.set_page_config(page_title="Stock Sentiment Analyzer", page_icon="📈", layout="wide")
st.title("📈 Stock Sentiment Analyzer")
st.caption("Compare a stock's recent price with the tone of its Yahoo Finance news headlines.")

with st.sidebar:
    st.header("Choose a stock")
    symbol = st.text_input("Ticker symbol", value="AAPL", help="Examples: AAPL, MSFT, TSLA, NVDA")
    period_label = st.selectbox("Price history", ["1 month", "3 months", "6 months", "1 year"], index=2)
    analyze = st.button("Analyze", type="primary", use_container_width=True)

periods = {"1 month": "1mo", "3 months": "3mo", "6 months": "6mo", "1 year": "1y"}


def sentiment_color(label: str) -> str:
    return {"Positive": "green", "Negative": "red", "Neutral": "gray"}[label]


if not analyze:
    st.info("Enter a ticker symbol in the sidebar, then select **Analyze**.")
else:
    try:
        ticker = normalize_ticker(symbol)
        with st.spinner(f"Getting data for {ticker}…"):
            prices = get_price_history(ticker, periods[period_label])
            news = score_headlines(get_news(ticker))
    except (ValueError, OSError, KeyError) as error:
        st.error(str(error))
    else:
        latest_close = float(prices["Close"].iloc[-1])
        change = float(prices["Close"].pct_change().iloc[-1] * 100)
        score = average_sentiment(news)
        first, second, third = st.columns(3)
        first.metric("Latest close", f"${latest_close:,.2f}", f"{change:+.2f}% today")
        second.metric("News headlines", len(news))
        third.metric("Average news sentiment", "No news" if score is None else f"{score:+.2f}")

        st.subheader(f"{ticker} closing price")
        st.plotly_chart(
            px.line(prices, x="Date", y="Close", title=f"{period_label.title()} adjusted closing price"),
            use_container_width=True,
        )

        st.subheader("News sentiment")
        if news.empty:
            st.warning("Yahoo Finance did not return any recent headlines for this ticker.")
        else:
            counts = news["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
            chart_data = counts.rename_axis("sentiment").reset_index(name="headlines")
            st.plotly_chart(
                px.bar(chart_data, x="sentiment", y="headlines", color="sentiment",
                       color_discrete_map={"Positive": "#2ca02c", "Neutral": "#7f7f7f", "Negative": "#d62728"}),
                use_container_width=True,
            )
            for article in news.itertuples(index=False):
                label = f"{article.sentiment} ({article.compound:+.2f})"
                st.markdown(f"**:{sentiment_color(article.sentiment)}[{label}]** — [{article.title}]({article.link})")
                st.caption(f"{article.publisher} · {article.published_at or 'Publication date unavailable'}")

st.divider()
st.caption("Sentiment uses VADER's rule-based compound score. It measures headline tone, not investment value. This app is for education only—not financial advice.")
