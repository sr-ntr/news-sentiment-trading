"""
Trading Strategy Module
=======================
Implements signal generation, backtesting, event study analysis,
and performance metric calculations for the sentiment-based strategy.

Inspired by: https://www.kaggle.com/code/shtrausslearning/news-sentiment-based-trading-strategy
"""

import numpy as np
import pandas as pd


def merge_stock_and_sentiment(
    stock_df: pd.DataFrame, sentiment_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge stock price data with daily sentiment data on date and ticker.

    Args:
        stock_df: DataFrame with stock OHLCV data (columns: Date, ticker, ...).
        sentiment_df: DataFrame with daily sentiment (columns: date, ticker, avg_polarity, ...).

    Returns:
        Merged DataFrame with both price and sentiment data.
    """
    stock_df = stock_df.copy()
    sentiment_df = sentiment_df.copy()

    stock_df["date"] = pd.to_datetime(stock_df["Date"]).dt.normalize()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.normalize()

    merged = pd.merge(
        stock_df,
        sentiment_df,
        how="left",
        on=["date", "ticker"],
    )

    # Fill days without news with neutral sentiment
    merged["avg_polarity"] = merged["avg_polarity"].fillna(0.0)
    merged["headline_count"] = merged["headline_count"].fillna(0).astype(int)
    merged["dominant_sentiment"] = merged["dominant_sentiment"].fillna("Neutral")

    return merged.sort_values("date").reset_index(drop=True)


def generate_signals(
    merged_df: pd.DataFrame, polarity_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Generate trading signals based on sentiment polarity thresholds.

    Args:
        merged_df: DataFrame with merged stock + sentiment data.
        polarity_threshold: Polarity threshold for signal generation.
            BUY when avg_polarity > threshold, SELL when < -threshold.

    Returns:
        DataFrame with added 'signal' column ('BUY', 'SELL', or 'HOLD').
    """
    df = merged_df.copy()
    conditions = [
        df["avg_polarity"] > polarity_threshold,
        df["avg_polarity"] < -polarity_threshold,
    ]
    choices = ["BUY", "SELL"]
    df["signal"] = np.select(conditions, choices, default="HOLD")
    return df


def calculate_event_returns(
    merged_df: pd.DataFrame, window: int = 3
) -> pd.DataFrame:
    """
    Calculate cumulative returns in a window around news events.

    Evaluates returns from t-window to t+window for days with news headlines,
    grouped by positive vs negative sentiment.

    Args:
        merged_df: DataFrame with merged stock + sentiment data.
        window: Number of days before and after the event to include.

    Returns:
        DataFrame with columns: day_offset, positive_avg_return,
        negative_avg_return, all_avg_return.
    """
    df = merged_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Only look at days with actual news
    news_days = df[df["headline_count"] > 0].index.tolist()

    event_returns: list[dict] = []

    for idx in news_days:
        sentiment = df.loc[idx, "avg_polarity"]
        is_positive = sentiment > 0.1
        is_negative = sentiment < -0.1

        for offset in range(-window, window + 1):
            target_idx = idx + offset
            if 0 <= target_idx < len(df):
                ret = df.loc[target_idx, "daily_return"]
                if pd.notna(ret):
                    event_returns.append(
                        {
                            "day_offset": offset,
                            "return": ret,
                            "sentiment_type": (
                                "Positive" if is_positive
                                else "Negative" if is_negative
                                else "Neutral"
                            ),
                        }
                    )

    if not event_returns:
        return pd.DataFrame(
            columns=["day_offset", "positive_avg_return", "negative_avg_return", "all_avg_return"]
        )

    event_df = pd.DataFrame(event_returns)

    # Pivot to get avg returns by sentiment type and day offset
    pos_returns = (
        event_df[event_df["sentiment_type"] == "Positive"]
        .groupby("day_offset")["return"]
        .mean()
        .rename("positive_avg_return")
    )
    neg_returns = (
        event_df[event_df["sentiment_type"] == "Negative"]
        .groupby("day_offset")["return"]
        .mean()
        .rename("negative_avg_return")
    )
    all_returns = (
        event_df.groupby("day_offset")["return"]
        .mean()
        .rename("all_avg_return")
    )

    result = pd.concat([pos_returns, neg_returns, all_returns], axis=1).reset_index()
    result = result.fillna(0.0)
    return result


def backtest_strategy(
    df: pd.DataFrame, initial_capital: float = 100000.0
) -> pd.DataFrame:
    """
    Run a simple backtest: buy on BUY signal, sell on SELL signal.

    Tracks portfolio value over time assuming full capital deployment.

    Args:
        df: DataFrame with signal column and daily_return.
        initial_capital: Starting portfolio capital.

    Returns:
        DataFrame with columns: date, signal, position, portfolio_value,
        daily_return, buy_hold_value.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    portfolio_value = initial_capital
    buy_hold_value = initial_capital
    position = 0  # 0 = cash, 1 = long

    records: list[dict] = []

    for i, row in df.iterrows():
        daily_ret = row["daily_return"] if pd.notna(row["daily_return"]) else 0.0

        # Update buy-and-hold
        buy_hold_value *= 1 + daily_ret

        # Signal-based position changes
        if row["signal"] == "BUY" and position == 0:
            position = 1
        elif row["signal"] == "SELL" and position == 1:
            position = 0

        # Update portfolio value based on position
        if position == 1:
            portfolio_value *= 1 + daily_ret

        records.append(
            {
                "date": row["date"],
                "signal": row["signal"],
                "position": position,
                "portfolio_value": portfolio_value,
                "daily_return": daily_ret,
                "buy_hold_value": buy_hold_value,
            }
        )

    return pd.DataFrame(records)


def calculate_performance_metrics(backtest_df: pd.DataFrame) -> dict:
    """
    Calculate key performance metrics from backtest results.

    Args:
        backtest_df: DataFrame from backtest_strategy().

    Returns:
        Dictionary with: total_return, annualized_return, sharpe_ratio,
        max_drawdown, win_rate, total_trades.
    """
    if backtest_df.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
        }

    # Total return
    initial_value = backtest_df["portfolio_value"].iloc[0]
    final_value = backtest_df["portfolio_value"].iloc[-1]
    total_return = (final_value - initial_value) / initial_value

    # Annualized return
    num_days = len(backtest_df)
    annualized_return = (1 + total_return) ** (252 / max(num_days, 1)) - 1

    # Daily returns when in position
    in_position = backtest_df[backtest_df["position"] == 1]["daily_return"]
    if len(in_position) > 1 and in_position.std() > 0:
        sharpe_ratio = (in_position.mean() / in_position.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    cumulative = backtest_df["portfolio_value"]
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    # Win rate (days with positive returns while in position)
    if len(in_position) > 0:
        win_rate = (in_position > 0).sum() / len(in_position)
    else:
        win_rate = 0.0

    # Total trades (signal changes)
    signals = backtest_df["signal"]
    total_trades = ((signals == "BUY") | (signals == "SELL")).sum()

    return {
        "total_return": round(total_return * 100, 2),
        "annualized_return": round(annualized_return * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "win_rate": round(win_rate * 100, 2),
        "total_trades": int(total_trades),
    }
