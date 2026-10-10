#  News Sentiment Trading Strategy Dashboard

> **An interactive algorithmic trading and financial NLP dashboard that transforms news headline sentiment into quantitative alpha, featuring dynamic trading signals, robust backtesting, and event study analysis.**

##  Overview

The **News Sentiment Trading Strategy Dashboard** bridges quantitative finance and natural language processing (NLP). Financial markets are intensely reactive to news narratives, yet traditional retail tools rarely quantify headline sentiment against equity price trajectories.

This project delivers an end-to-end algorithmic research framework that:
1. Streams real-time market data and headline feeds.
2. Extracts polarity and subjectivity scores via TextBlob NLP.
3. Formulates disciplined trading signals based on sentiment momentum.
4. Backtests strategy profitability vs. a Buy & Hold benchmark.
5. Evaluates market efficiency around headline releases through rigorous **Event Studies**.

---


##  How It Works

The strategy follows a 5-stage quantitative pipeline:

```mermaid
flowchart LR
    A[Data Collection\n(yfinance & News)] --> B[Sentiment Analysis\n(TextBlob NLP)]
    B --> C[Signal Generation\n(Thresholds & Moving Avg)]
    C --> D[Backtesting Engine\n(Returns & Metrics)]
    D --> E[Interactive Visualization\n(Plotly & Streamlit)]
```

### 1. Data Collection (`src/data_loader.py`)
- Fetches historical price bars (Open, High, Low, Close, Volume) using `yfinance`.
- Collects news headlines or generates synthetic market news items for selected tickers.
- Synchronizes timestamps between news publications and trading sessions.

### 2. Sentiment Analysis (`src/sentiment.py`)
- Cleans and tokenizes headline strings.
- Uses **TextBlob** lexicon-based NLP to extract:
  - **Polarity** $\in [-1.0, 1.0]$: Indicates negative (bearish), neutral, or positive (bullish) tone.
  - **Subjectivity** $\in [0.0, 1.0]$: Quantifies factual reporting vs. subjective opinion.
- Computes rolling sentiment averages (e.g., 3-day / 7-day EMA) to dampen news cycle noise.

### 3. Signal Generation (`src/strategy.py`)
Trading decisions are governed by parameterized sentiment thresholds:
$$\text{Signal}_t = \begin{cases} 
\text{BUY (+1)} & \text{if } \text{Sentiment}_t > \theta_{\text{upper}} \\
\text{SELL (-1)} & \text{if } \text{Sentiment}_t < \theta_{\text{lower}} \\
\text{HOLD (0)} & \text{otherwise}
\end{cases}$$

### 4. Backtesting & Performance Analytics (`src/strategy.py`)
- Simulates capital allocation, trade execution, and portfolio mark-to-market valuations.
- Calculates trade log, win/loss ratio, transaction friction, and risk-adjusted metrics:
  $$\text{Sharpe Ratio} = \frac{\mathbb{E}[R_p - R_f]}{\sigma_p}$$
  $$\text{Max Drawdown} = \max_{\tau \in [0, t]} \left( \frac{\text{Peak}_\tau - \text{Equity}_t}{\text{Peak}_\tau} \right)$$

### 5. Event Study Framework (`src/strategy.py`)
- Isolates major sentiment outliers (e.g., $|\text{Sentiment}| > 0.5$).
- Tracks daily Abnormal Returns ($AR_t = R_{i,t} - R_{m,t}$) across an event window $[-T_1, +T_2]$.
- Aggregates Cumulative Abnormal Returns ($CAR$) to test market overreaction or drift hypotheses.

---

##  Project Structure

```
news-sentiment-trading/
├── app.py                  # Streamlit dashboard application entry point
├── requirements.txt        # Python package dependencies
├── LICENSE                 # MIT License
├── README.md               # Project documentation and guide
├── .gitignore              # Git ignore rules
├── .streamlit/
│   └── config.toml         # Custom Streamlit UI theme configuration
├── src/
│   ├── __init__.py         # Package initialization
│   ├── data_loader.py      # Stock data fetching (yfinance) & sample news generator
│   ├── sentiment.py        # TextBlob NLP sentiment extraction & rolling calculations
│   ├── strategy.py         # Signal generation, backtesting engine & event study math
│   └── visualization.py    # Plotly interactive charts & styling components
└── data/
    └── .gitkeep            # Directory placeholder for cached market datasets
```