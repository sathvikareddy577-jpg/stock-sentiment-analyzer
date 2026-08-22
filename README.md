# Stock Sentiment Analyzer

A beginner-friendly [Streamlit](https://streamlit.io/) dashboard that shows a stock's recent price history alongside the sentiment of recent Yahoo Finance news headlines.

> **Educational use only.** News sentiment is not a trading signal or financial advice.

## What it does

- Downloads recent adjusted stock-price data with `yfinance` (Yahoo Finance).
- Retrieves recent news headlines for the selected ticker.
- Scores each headline with financial-language-aware [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) by default.
- Automatically falls back to NLTK's VADER analyzer if FinBERT or its model cannot be loaded.
- Shows the active sentiment engine in the dashboard.
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

The first analysis lazy-loads and caches the FinBERT model. This may download the model on first use. If FinBERT is unavailable (for example, because its dependencies or model files cannot be loaded), the app automatically uses VADER instead; VADER's small lexicon is downloaded as needed.

## Run tests

```bash
pytest
```

## Project layout

```text
app.py                       # Streamlit user interface
sentiment_analyzer/data.py   # Yahoo Finance price and news retrieval
sentiment_analyzer/sentiment.py  # Lazy FinBERT scoring with VADER fallback
tests/                       # Fast unit tests (no network needed)
```
