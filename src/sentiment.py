"""
Sentiment Analysis Module
=========================
Uses TextBlob NLP to extract polarity and subjectivity scores from
financial news headlines, then aggregates daily sentiment metrics.

Inspired by: https://www.kaggle.com/code/shtrausslearning/news-sentiment-based-trading-strategy
"""

import pandas as pd
from textblob import TextBlob


def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of a single text string using TextBlob.

    Args:
        text: Input text (headline) to analyze.

    Returns:
        Dictionary with keys:
            - polarity (float): Sentiment polarity in [-1.0, 1.0].
            - subjectivity (float): Subjectivity score in [0.0, 1.0].
            - label (str): 'Positive', 'Negative', or 'Neutral'.
    """
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        label = "Positive"
    elif polarity < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return {"polarity": polarity, "subjectivity": subjectivity, "label": label}


def add_sentiment_scores(
    df: pd.DataFrame, text_column: str = "headline"
) -> pd.DataFrame:
    """
    Apply sentiment analysis to a DataFrame column and add score columns.

    Args:
        df: DataFrame containing text data.
        text_column: Name of the column with text to analyze.

    Returns:
        DataFrame with added columns: polarity, subjectivity, sentiment_label.
    """
    df = df.copy()
    sentiment_results = df[text_column].apply(analyze_sentiment)
    df["polarity"] = sentiment_results.apply(lambda x: x["polarity"])
    df["subjectivity"] = sentiment_results.apply(lambda x: x["subjectivity"])
    df["sentiment_label"] = sentiment_results.apply(lambda x: x["label"])
    return df


def get_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment scores to daily level per ticker.

    Groups by date and ticker, computing mean polarity, headline count,
    and the dominant (most frequent) sentiment label for each day.

    Args:
        df: DataFrame with columns: date, ticker, polarity, sentiment_label.

    Returns:
        DataFrame with columns: date, ticker, avg_polarity, headline_count,
        dominant_sentiment.
    """
    daily = (
        df.groupby(["date", "ticker"])
        .agg(
            avg_polarity=("polarity", "mean"),
            headline_count=("polarity", "count"),
            dominant_sentiment=(
                "sentiment_label",
                lambda x: x.mode().iloc[0] if not x.mode().empty else "Neutral",
            ),
        )
        .reset_index()
    )
    return daily
