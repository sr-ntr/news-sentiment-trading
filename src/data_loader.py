"""
Data Loader Module
==================
Handles fetching stock market data via yfinance and loading/generating
news headline data for sentiment analysis.

Inspired by: https://www.kaggle.com/code/shtrausslearning/news-sentiment-based-trading-strategy
"""

import random
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download historical OHLCV data for a given ticker using yfinance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').
        start: Start date string in 'YYYY-MM-DD' format.
        end: End date string in 'YYYY-MM-DD' format.

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume,
        ticker, daily_return.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df["ticker"] = ticker
    df["daily_return"] = df["Close"].pct_change()

    return df[["Date", "Open", "High", "Low", "Close", "Volume", "ticker", "daily_return"]]


def load_news_from_json(filepath: str) -> pd.DataFrame:
    """
    Load news headline data from a JSON file.

    Expected JSON format: list of objects with keys 'date', 'headline', 'ticker'.

    Args:
        filepath: Path to the JSON file.

    Returns:
        DataFrame with columns: date, headline, ticker.
    """
    df = pd.read_json(filepath)
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "headline", "ticker"]]


def generate_sample_news(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Generate realistic sample financial news headlines for demo purposes.

    Creates ~150-200 headlines spread across the date range for the given
    tickers, mixing positive, negative, and neutral sentiment.

    Args:
        tickers: List of stock ticker symbols.
        start_date: Start date string in 'YYYY-MM-DD' format.
        end_date: End date string in 'YYYY-MM-DD' format.

    Returns:
        DataFrame with columns: date, headline, ticker.
    """
    positive_templates = [
        "{ticker} beats Q{q} earnings expectations with record revenue",
        "{ticker} announces major stock buyback program worth $5B",
        "{ticker} reports strong quarterly growth, stock surges",
        "{ticker} lands massive government contract worth billions",
        "{ticker} CEO optimistic about future growth prospects",
        "{ticker} expands into new international markets successfully",
        "{ticker} raises full-year guidance above analyst estimates",
        "{ticker} achieves record profit margins in latest quarter",
        "{ticker} new product launch receives overwhelmingly positive reviews",
        "{ticker} partnership deal boosts investor confidence significantly",
        "{ticker} named top innovator in industry report",
        "{ticker} dividend increase delights shareholders",
        "{ticker} cloud revenue growth exceeds Wall Street forecasts",
        "{ticker} announces breakthrough technology advancement",
        "{ticker} market share gains accelerate in key segments",
    ]

    negative_templates = [
        "{ticker} misses Q{q} earnings estimates, shares drop sharply",
        "{ticker} reports weak revenue growth amid market uncertainty",
        "{ticker} faces regulatory scrutiny over business practices",
        "{ticker} announces major layoffs affecting thousands of workers",
        "{ticker} warns of slowing demand in key product segments",
        "{ticker} loses key executive in surprise departure",
        "{ticker} hit by supply chain disruptions impacting production",
        "{ticker} cuts full-year revenue guidance below consensus",
        "{ticker} faces class action lawsuit from investors",
        "{ticker} market share declines as competition intensifies",
        "{ticker} reports unexpected quarterly loss for first time",
        "{ticker} data breach affects millions of customer accounts",
        "{ticker} product recall impacts consumer trust and sales",
        "{ticker} under investigation by federal trade commission",
        "{ticker} debt levels raise concerns among credit analysts",
    ]

    neutral_templates = [
        "{ticker} schedules annual shareholder meeting for next month",
        "{ticker} releases quarterly earnings report on schedule",
        "{ticker} appoints new board member from technology sector",
        "{ticker} plans to present at upcoming industry conference",
        "{ticker} files routine SEC disclosure documents",
        "{ticker} maintains current dividend policy unchanged",
        "{ticker} opens new regional office in expanding market",
        "{ticker} updates corporate governance guidelines",
        "{ticker} announces routine management restructuring plan",
        "{ticker} participates in analyst day presentation event",
    ]

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days

    random.seed(42)
    num_headlines = random.randint(150, 200)
    records: list[dict] = []

    for _ in range(num_headlines):
        ticker = random.choice(tickers)
        days_offset = random.randint(0, total_days)
        news_date = start + timedelta(days=days_offset)
        quarter = random.randint(1, 4)

        # 40% positive, 35% negative, 25% neutral
        roll = random.random()
        if roll < 0.40:
            template = random.choice(positive_templates)
        elif roll < 0.75:
            template = random.choice(negative_templates)
        else:
            template = random.choice(neutral_templates)

        headline = template.format(ticker=ticker, q=quarter)
        records.append({"date": news_date, "headline": headline, "ticker": ticker})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df
