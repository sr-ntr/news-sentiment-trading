"""
📈 News Sentiment Trading Strategy Dashboard
=============================================
Interactive Streamlit app for analyzing financial news sentiment
and generating trading signals with backtesting capabilities.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.data_loader import fetch_stock_data, generate_sample_news, load_news_from_json
from src.sentiment import add_sentiment_scores, get_daily_sentiment
from src.strategy import (
    generate_signals,
    merge_stock_and_sentiment,
    backtest_strategy,
    calculate_performance_metrics,
    calculate_event_returns,
)
from src.visualization import (
    plot_stock_with_signals,
    plot_sentiment_distribution,
    plot_sentiment_over_time,
    plot_portfolio_performance,
    plot_event_returns,
    plot_correlation_heatmap,
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="News Sentiment Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 News Sentiment Trader")
    st.markdown("---")

    # Ticker selection
    available_tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "NFLX", "AMD", "BA", "ADBE"]
    selected_tickers = st.multiselect(
        "Select Tickers",
        options=available_tickers,
        default=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
        help="Choose one or more stock tickers to analyze.",
    )

    st.markdown("---")

    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            help="Analysis start date.",
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            help="Analysis end date.",
        )

    st.markdown("---")

    # Strategy parameters
    polarity_threshold = st.slider(
        "Polarity Threshold",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.05,
        help="Sentiment score above/below this triggers BUY/SELL signals.",
    )

    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=1000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
        help="Starting capital for the backtest.",
    )

    st.markdown("---")

    # Data source
    data_source = st.radio(
        "Data Source",
        options=["Generate Sample Data", "Upload JSON"],
        help="Choose whether to generate demo data or upload your own.",
    )

    uploaded_file = None
    if data_source == "Upload JSON":
        uploaded_file = st.file_uploader(
            "Upload News JSON",
            type=["json"],
            help="JSON file with format: [{date, headline, ticker}, ...]",
        )

    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ── Main Content ─────────────────────────────────────────────────────────────
st.markdown("# 📈 News Sentiment Trading Dashboard")
st.markdown(
    "Analyze financial news sentiment and generate data-driven trading signals "
    "with interactive backtesting."
)

if not run_analysis and "results" not in st.session_state:
    st.info(
        "👈 Configure parameters in the sidebar and click **Run Analysis** to get started."
    )
    st.stop()

# ── Run the Pipeline ─────────────────────────────────────────────────────────
if run_analysis:
    if not selected_tickers:
        st.error("Please select at least one ticker.")
        st.stop()

    with st.spinner("Fetching stock data..."):
        all_stock_data = []
        for ticker in selected_tickers:
            stock_df = fetch_stock_data(
                ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )
            if not stock_df.empty:
                all_stock_data.append(stock_df)

        if not all_stock_data:
            st.error("Could not fetch stock data. Please check your tickers and date range.")
            st.stop()

        stock_data = pd.concat(all_stock_data, ignore_index=True)

    with st.spinner("Loading news data..."):
        if data_source == "Upload JSON" and uploaded_file is not None:
            import json, tempfile, os
            # Save uploaded file temporarily
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            tmp.write(uploaded_file.read())
            tmp.close()
            news_data = load_news_from_json(tmp.name)
            os.unlink(tmp.name)
        else:
            news_data = generate_sample_news(
                selected_tickers,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )

    with st.spinner("Analyzing sentiment..."):
        news_with_sentiment = add_sentiment_scores(news_data)
        daily_sentiment = get_daily_sentiment(news_with_sentiment)

    with st.spinner("Generating trading signals..."):
        merged = merge_stock_and_sentiment(stock_data, daily_sentiment)
        signals = generate_signals(merged, polarity_threshold)

    with st.spinner("Running backtest..."):
        # Backtest per ticker, then combine
        all_backtests = []
        all_metrics = {}
        for ticker in selected_tickers:
            ticker_signals = signals[signals["ticker"] == ticker].copy()
            if not ticker_signals.empty:
                bt = backtest_strategy(ticker_signals, initial_capital)
                bt["ticker"] = ticker
                all_backtests.append(bt)
                all_metrics[ticker] = calculate_performance_metrics(bt)

        if all_backtests:
            backtest_results = pd.concat(all_backtests, ignore_index=True)
        else:
            backtest_results = pd.DataFrame()

    with st.spinner("Computing event returns..."):
        event_returns = {}
        for ticker in selected_tickers:
            ticker_data = signals[signals["ticker"] == ticker]
            er = calculate_event_returns(ticker_data, window=3)
            if not er.empty:
                event_returns[ticker] = er

    # Store results in session state
    st.session_state["results"] = {
        "stock_data": stock_data,
        "news_data": news_data,
        "news_with_sentiment": news_with_sentiment,
        "daily_sentiment": daily_sentiment,
        "signals": signals,
        "backtest_results": backtest_results,
        "metrics": all_metrics,
        "event_returns": event_returns,
        "selected_tickers": selected_tickers,
    }

# ── Display Results ──────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.stop()

r = st.session_state["results"]

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "💬 Sentiment Analysis", "📈 Trading Signals", "💰 Backtest Results", "🔍 Event Study"]
)

# ── Tab 1: Overview ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("📊 Analysis Overview")

    news_df = r["news_with_sentiment"]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Headlines", len(news_df))
    with col2:
        st.metric("Avg Sentiment", f"{news_df['polarity'].mean():.3f}")
    with col3:
        pct_pos = (news_df["sentiment_label"] == "Positive").sum() / len(news_df) * 100
        st.metric("% Positive", f"{pct_pos:.1f}%")
    with col4:
        pct_neg = (news_df["sentiment_label"] == "Negative").sum() / len(news_df) * 100
        st.metric("% Negative", f"{pct_neg:.1f}%")

    st.markdown("---")

    view_ticker = st.selectbox(
        "Select Ticker to View",
        options=r["selected_tickers"],
        key="overview_ticker",
    )

    ticker_stock = r["stock_data"][r["stock_data"]["ticker"] == view_ticker]
    if not ticker_stock.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ticker_stock["Date"],
                y=ticker_stock["Close"],
                mode="lines",
                name=f"{view_ticker} Close",
                line=dict(color="#00D4AA", width=2),
            )
        )
        fig.update_layout(
            title=f"{view_ticker} — Stock Price",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#1A1F2E",
            font=dict(color="#FAFAFA"),
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Sentiment Analysis ───────────────────────────────────────────────
with tab2:
    st.subheader("💬 Sentiment Analysis")

    # Sentiment distribution
    st.plotly_chart(
        plot_sentiment_distribution(r["news_with_sentiment"]),
        use_container_width=True,
    )

    st.markdown("---")

    # Sentiment over time
    sent_ticker = st.selectbox(
        "Select Ticker",
        options=r["selected_tickers"],
        key="sentiment_ticker",
    )
    st.plotly_chart(
        plot_sentiment_over_time(r["daily_sentiment"], sent_ticker),
        use_container_width=True,
    )

    st.markdown("---")

    # Headlines table
    st.subheader("📰 Recent Headlines with Sentiment")
    display_df = (
        r["news_with_sentiment"]
        .sort_values("date", ascending=False)
        .head(50)[["date", "ticker", "headline", "polarity", "subjectivity", "sentiment_label"]]
        .reset_index(drop=True)
    )

    def color_sentiment(val):
        if val == "Positive":
            return "color: #00E676"
        elif val == "Negative":
            return "color: #FF5252"
        return "color: #FFC107"

    st.dataframe(
        display_df.style.applymap(color_sentiment, subset=["sentiment_label"]),
        use_container_width=True,
        height=400,
    )

# ── Tab 3: Trading Signals ──────────────────────────────────────────────────
with tab3:
    st.subheader("📈 Trading Signals")

    signal_ticker = st.selectbox(
        "Select Ticker",
        options=r["selected_tickers"],
        key="signal_ticker",
    )

    st.plotly_chart(
        plot_stock_with_signals(r["signals"], signal_ticker),
        use_container_width=True,
    )

    st.markdown("---")

    # Signal summary
    ticker_signals = r["signals"][r["signals"]["ticker"] == signal_ticker]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 BUY Signals", (ticker_signals["signal"] == "BUY").sum())
    with col2:
        st.metric("🔴 SELL Signals", (ticker_signals["signal"] == "SELL").sum())
    with col3:
        st.metric("⚪ HOLD Days", (ticker_signals["signal"] == "HOLD").sum())

    # Recent signals table
    st.subheader("Recent Signals")
    recent_signals = (
        ticker_signals[ticker_signals["signal"] != "HOLD"]
        .sort_values("date", ascending=False)
        .head(20)[["date", "Close", "avg_polarity", "signal"]]
        .reset_index(drop=True)
    )
    st.dataframe(recent_signals, use_container_width=True)

# ── Tab 4: Backtest Results ─────────────────────────────────────────────────
with tab4:
    st.subheader("💰 Backtest Results")

    if not r["backtest_results"].empty:
        bt_ticker = st.selectbox(
            "Select Ticker",
            options=r["selected_tickers"],
            key="backtest_ticker",
        )

        if bt_ticker in r["metrics"]:
            metrics = r["metrics"][bt_ticker]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Return", f"{metrics['total_return']}%")
                st.metric("Annualized Return", f"{metrics['annualized_return']}%")
            with col2:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']}")
                st.metric("Max Drawdown", f"{metrics['max_drawdown']}%")
            with col3:
                st.metric("Win Rate", f"{metrics['win_rate']}%")
                st.metric("Total Trades", f"{metrics['total_trades']}")

            st.markdown("---")

            # Portfolio performance chart
            bt_data = r["backtest_results"][r["backtest_results"]["ticker"] == bt_ticker]
            st.plotly_chart(
                plot_portfolio_performance(bt_data),
                use_container_width=True,
            )

        st.markdown("---")

        # Correlation heatmap
        st.subheader("Feature Correlations")
        st.plotly_chart(
            plot_correlation_heatmap(r["signals"]),
            use_container_width=True,
        )
    else:
        st.warning("No backtest results available.")

# ── Tab 5: Event Study ──────────────────────────────────────────────────────
with tab5:
    st.subheader("🔍 Event Study Analysis")

    st.markdown(
        """
        The **Event Study** examines stock returns in a window around news events to determine
        whether positive or negative sentiment headlines correlate with subsequent price movements.

        - **Day 0** = Date of the news headline
        - **Negative days** (t-1, t-2, ...) = Days *before* the news
        - **Positive days** (t+1, t+2, ...) = Days *after* the news

        We compare average returns for **positive sentiment** vs **negative sentiment** headlines
        to see if the market reacts consistently to news tone.
        """
    )

    event_ticker = st.selectbox(
        "Select Ticker",
        options=r["selected_tickers"],
        key="event_ticker",
    )

    if event_ticker in r["event_returns"] and not r["event_returns"][event_ticker].empty:
        st.plotly_chart(
            plot_event_returns(r["event_returns"][event_ticker]),
            use_container_width=True,
        )
    else:
        st.info(f"Not enough event data for {event_ticker} to display event study results.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #78909C; font-size: 0.85em;'>"
    "📈 News Sentiment Trading Dashboard · Built with Streamlit & Plotly · "
    "<a href='https://www.kaggle.com/code/shtrausslearning/news-sentiment-based-trading-strategy/notebook' "
    "target='_blank' style='color: #00D4AA;'>Inspired by shtrausslearning</a> · "
    "For educational purposes only — not financial advice."
    "</div>",
    unsafe_allow_html=True,
)
