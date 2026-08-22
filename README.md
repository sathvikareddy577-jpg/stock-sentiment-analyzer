# Stock Sentiment Analyzer

A beginner-friendly [Streamlit](https://streamlit.io/) dashboard that shows a stock's recent price history alongside the sentiment of recent Yahoo Finance news headlines.

> **Educational use only.** News sentiment is not a trading signal or financial advice.

## What it does

- Downloads recent adjusted stock-price data with `yfinance` (Yahoo Finance).
- Retrieves recent news headlines for the selected ticker.
- Scores each headline with NLTK's VADER sentiment analyzer.
- Shows interactive Plotly charts for closing prices and headline sentiment counts.

## Quick start

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

5. Open the local address printed by Streamlit (usually `http://localhost:8501`), enter a ticker such as `AAPL`, and click **Analyze**.

The first analysis downloads VADER's small sentiment lexicon automatically.

## Run tests

```bash
pytest
```

## Project layout

```text
app.py                       # Streamlit user interface
sentiment_analyzer/data.py   # Yahoo Finance price and news retrieval
sentiment_analyzer/sentiment.py  # VADER scoring helpers
tests/                       # Fast unit tests (no network needed)
```
