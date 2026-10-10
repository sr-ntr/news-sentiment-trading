"""
Visualization Module
====================
Builds interactive Plotly charts for stock prices, sentiment analysis,
trading signals, portfolio performance, and event study results.

Inspired by: https://www.kaggle.com/code/shtrausslearning/news-sentiment-based-trading-strategy
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Consistent color palette
COLORS = {
    "primary": "#00D4AA",
    "secondary": "#FF6B6B",
    "neutral": "#FFC107",
    "bg": "#0E1117",
    "card_bg": "#1A1F2E",
    "text": "#FAFAFA",
    "buy": "#00E676",
    "sell": "#FF5252",
    "hold": "#78909C",
    "positive": "#00E676",
    "negative": "#FF5252",
    "benchmark": "#42A5F5",
}

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["card_bg"],
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    margin=dict(l=40, r=40, t=60, b=40),
)


def plot_stock_with_signals(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Line chart of stock closing price with BUY/SELL signal markers.

    Args:
        df: DataFrame with columns: date, Close, signal, ticker.
        ticker: Ticker symbol to filter and display.

    Returns:
        Plotly Figure object.
    """
    data = df[df["ticker"] == ticker].copy()

    fig = go.Figure()

    # Price line
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color=COLORS["primary"], width=2),
        )
    )

    # BUY signals
    buys = data[data["signal"] == "BUY"]
    fig.add_trace(
        go.Scatter(
            x=buys["date"],
            y=buys["Close"],
            mode="markers",
            name="BUY Signal",
            marker=dict(
                symbol="triangle-up",
                size=14,
                color=COLORS["buy"],
                line=dict(width=1, color="white"),
            ),
        )
    )

    # SELL signals
    sells = data[data["signal"] == "SELL"]
    fig.add_trace(
        go.Scatter(
            x=sells["date"],
            y=sells["Close"],
            mode="markers",
            name="SELL Signal",
            marker=dict(
                symbol="triangle-down",
                size=14,
                color=COLORS["sell"],
                line=dict(width=1, color="white"),
            ),
        )
    )

    fig.update_layout(
        title=f"{ticker} — Price with Trading Signals",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        **LAYOUT_DEFAULTS,
    )
    return fig


def plot_sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Histogram of sentiment polarity scores with color coding.

    Args:
        df: DataFrame with column: polarity.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    for label, color in [
        ("Positive", COLORS["positive"]),
        ("Neutral", COLORS["neutral"]),
        ("Negative", COLORS["negative"]),
    ]:
        subset = df[df["sentiment_label"] == label]
        fig.add_trace(
            go.Histogram(
                x=subset["polarity"],
                name=label,
                marker_color=color,
                opacity=0.75,
                nbinsx=30,
            )
        )

    fig.update_layout(
        title="Sentiment Polarity Distribution",
        xaxis_title="Polarity Score",
        yaxis_title="Number of Headlines",
        barmode="overlay",
        **LAYOUT_DEFAULTS,
    )
    return fig


def plot_sentiment_over_time(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Dual-axis chart: line for average daily polarity + bar for headline count.

    Args:
        df: DataFrame with columns: date, ticker, avg_polarity, headline_count.
        ticker: Ticker symbol to filter.

    Returns:
        Plotly Figure object.
    """
    data = df[df["ticker"] == ticker].copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=data["date"],
            y=data["headline_count"],
            name="Headline Count",
            marker_color=COLORS["hold"],
            opacity=0.4,
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["avg_polarity"],
            mode="lines+markers",
            name="Avg Polarity",
            line=dict(color=COLORS["primary"], width=2),
            marker=dict(size=5),
        ),
        secondary_y=False,
    )

    # Zero line reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=False)

    fig.update_layout(
        title=f"{ticker} — Daily Sentiment Over Time",
        hovermode="x unified",
        **LAYOUT_DEFAULTS,
    )
    fig.update_yaxes(title_text="Avg Polarity", secondary_y=False)
    fig.update_yaxes(title_text="Headline Count", secondary_y=True)

    return fig


def plot_portfolio_performance(backtest_df: pd.DataFrame) -> go.Figure:
    """
    Line chart comparing strategy portfolio value vs buy-and-hold benchmark.

    Args:
        backtest_df: DataFrame from backtest_strategy() with columns:
            date, portfolio_value, buy_hold_value.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=backtest_df["date"],
            y=backtest_df["portfolio_value"],
            mode="lines",
            name="Sentiment Strategy",
            line=dict(color=COLORS["primary"], width=2.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=backtest_df["date"],
            y=backtest_df["buy_hold_value"],
            mode="lines",
            name="Buy & Hold",
            line=dict(color=COLORS["benchmark"], width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title="Portfolio Performance — Strategy vs Buy & Hold",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        **LAYOUT_DEFAULTS,
    )
    return fig


def plot_event_returns(event_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart showing average returns around news events
    for positive vs negative sentiment.

    Args:
        event_df: DataFrame with columns: day_offset,
            positive_avg_return, negative_avg_return.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=event_df["day_offset"],
            y=event_df["positive_avg_return"] * 100,
            name="Positive Sentiment",
            marker_color=COLORS["positive"],
            opacity=0.8,
        )
    )

    fig.add_trace(
        go.Bar(
            x=event_df["day_offset"],
            y=event_df["negative_avg_return"] * 100,
            name="Negative Sentiment",
            marker_color=COLORS["negative"],
            opacity=0.8,
        )
    )

    fig.update_layout(
        title="Event Study — Average Returns Around News Events",
        xaxis_title="Days Relative to News Event (t=0)",
        yaxis_title="Average Return (%)",
        barmode="group",
        **LAYOUT_DEFAULTS,
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap of correlations between sentiment scores and returns.

    Args:
        df: DataFrame with numeric columns including polarity,
            daily_return, avg_polarity, etc.

    Returns:
        Plotly Figure object.
    """
    numeric_cols = ["avg_polarity", "headline_count", "daily_return", "Close", "Volume"]
    available_cols = [c for c in numeric_cols if c in df.columns]

    if len(available_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for correlation heatmap", showarrow=False)
        fig.update_layout(**LAYOUT_DEFAULTS)
        return fig

    corr_matrix = df[available_cols].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdYlGn",
            zmid=0,
            text=corr_matrix.values.round(3),
            texttemplate="%{text}",
            textfont=dict(size=12),
        )
    )

    fig.update_layout(
        title="Correlation Heatmap — Sentiment & Market Features",
        **LAYOUT_DEFAULTS,
    )
    return fig
