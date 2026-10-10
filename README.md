# 📈 News Sentiment Trading Strategy Dashboard

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/python/)
[![NLP](https://img.shields.io/badge/TextBlob-NLP_Sentiment-FF6F00?style=for-the-badge)](https://textblob.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **An interactive algorithmic trading and financial NLP dashboard that transforms news headline sentiment into quantitative alpha, featuring dynamic trading signals, robust backtesting, and event study analysis.**

---

## 📸 Dashboard Preview

<!-- Add screenshot here -->
```
+----------------------------------------------------------------------------------------------------+
|  📈 NEWS SENTIMENT TRADING STRATEGY DASHBOARD                                 [Ticker: AAPL  v]     |
+----------------------------------------------------------------------------------------------------+
|  [ Overview ]  [ Sentiment Analysis ]  [ Trading Signals ]  [ Backtest Results ]  [ Event Study ]  |
+----------------------------------------------------------------------------------------------------+
|  METRICS SUMMARY                                                                                   |
|  * Current Price: $182.40     * 30D Sentiment Score: +0.42     * Strategy Return: +18.4%           |
|  * Benchmark Return: +11.2%   * Sharpe Ratio: 1.84             * Max Drawdown: -6.2%               |
+----------------------------------------------------------------------------------------------------+
|  EQUITY CURVE & SIGNAL VISUALIZATION                                                               |
|  [=========================== Strategy Equity Curve vs. Benchmark ===========================]    |
|  [             ▲ Buy Signal (Polarity > +0.20)     ▼ Sell Signal (Polarity < -0.20)          ]    |
+----------------------------------------------------------------------------------------------------+
```

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [How It Works (Architecture & Pipeline)](#-how-it-works)
- [Dashboard Walkthrough](#-dashboard-walkthrough)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Acknowledgments](#-acknowledgments)
- [Disclaimer](#-disclaimer)
- [License](#-license)

---

## 🌟 Overview

The **News Sentiment Trading Strategy Dashboard** bridges quantitative finance and natural language processing (NLP). Financial markets are intensely reactive to news narratives, yet traditional retail tools rarely quantify headline sentiment against equity price trajectories.

This project delivers an end-to-end algorithmic research framework that:
1. Streams real-time market data and headline feeds.
2. Extracts polarity and subjectivity scores via TextBlob NLP.
3. Formulates disciplined trading signals based on sentiment momentum.
4. Backtests strategy profitability vs. a Buy & Hold benchmark.
5. Evaluates market efficiency around headline releases through rigorous **Event Studies**.

---

## ✨ Key Features

- 📊 **Real-Time Stock Market Data**: Retrieves live and historical OHLCV data directly via `yfinance` across equities, ETFs, and indices.
- 🧠 **NLP Sentiment Engine**: Analyzes financial headline tone using TextBlob to compute bounded polarity scores ($[-1.0, +1.0]$) and subjectivity levels.
- 🎯 **Algorithmic Signal Generation**: Produces actionable **BUY**, **SELL**, and **HOLD** signals using customizable polarity thresholds and moving-average smoothing.
- 🧪 **Quantitative Backtesting Engine**: Evaluates strategy performance with metrics including Cumulative Return, Annualized Return, Sharpe Ratio, Maximum Drawdown, and Win Rate.
- 📅 **Event Study Analysis**: Measures post-event price impact and Cumulative Abnormal Returns (CAR) across customizable pre- and post-announcement event windows.
- 🖥️ **5-Tab Interactive Streamlit Dashboard**: Provides an intuitive analytics experience powered by Plotly charts with full zoom, pan, and hover inspection.
- 🎲 **Sample Data Generator**: Includes an offline demo mode with synthetic headline and price data, allowing immediate testing without external API limitations.

---

## 🔄 How It Works

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

## 🖥️ Dashboard Walkthrough

The application organizes insights into 5 focused tabs:

| Tab | Focus & Contents |
| :--- | :--- |
| **1. Overview** | Executive summary, key ticker statistics, real-time price snapshot, and recent news stream. |
| **2. Sentiment Analysis** | Distribution histograms, sentiment time series with rolling averages, and polarity vs. subjectivity scatter plots. |
| **3. Trading Signals** | Candlestick price chart overlaid with buy (🟢) and sell (🔴) markers, plus tabular signal logs. |
| **4. Backtest Results** | Strategy Equity Curve vs. Buy & Hold Benchmark, Drawdown profiles, and quantitative risk metrics table. |
| **5. Event Study** | Cumulative Abnormal Return (CAR) trajectories surrounding major news events to evaluate market efficiency. |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/news-sentiment-trading.git
   cd news-sentiment-trading
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download TextBlob corpora (if prompted):**
   ```bash
   python -m textblob.download_corpora
   ```

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

6. Open your browser and navigate to `http://localhost:8501`.

---

## ⚙️ Configuration

The dashboard exposes real-time sidebar controls to tune strategy and data parameters:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `Ticker Symbol` | `AAPL` | Stock ticker to analyze (e.g., `MSFT`, `TSLA`, `NVDA`, `SPY`). |
| `Date Range` | Past 1 Year | Historical analysis horizon. |
| `Positive Threshold` ($\theta_{\text{upper}}$) | `+0.20` | Sentiment score above which a BUY signal is generated. |
| `Negative Threshold` ($\theta_{\text{lower}}$) | `-0.20` | Sentiment score below which a SELL signal is generated. |
| `Smoothing Window` | `3 Days` | Moving average window applied to raw headline sentiment. |
| `Initial Capital` | `$10,000` | Starting portfolio cash for backtesting simulations. |
| `Event Window` | `[-3, +5]` | Days before and after a news headline for event study tracking. |
| `Data Source` | Live / Sample | Toggle between live `yfinance` feed or built-in sample generator. |

---

## 📂 Project Structure

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

### Module Responsibilities
- [`app.py`](file:///home/henry-k/news-sentiment-trading/app.py): Manages Streamlit layout, session state, tabs, and user input widgets.
- [`src/data_loader.py`](file:///home/henry-k/news-sentiment-trading/src/data_loader.py): Handles asynchronous yfinance queries, OHLCV formatting, and fallback synthetic datasets.
- [`src/sentiment.py`](file:///home/henry-k/news-sentiment-trading/src/sentiment.py): Encapsulates text cleaning, TextBlob polarity/subjectivity computation, and time-series aggregation.
- [`src/strategy.py`](file:///home/henry-k/news-sentiment-trading/src/strategy.py): Contains trading logic, portfolio equity tracking, trade recording, and event study abnormal return calculators.
- [`src/visualization.py`](file:///home/henry-k/news-sentiment-trading/src/visualization.py): Builds high-performance, responsive Plotly figures (candlesticks, equity curves, CAR plots).

---

## 🛠️ Technologies Used

| Technology | Purpose | Documentation |
| :--- | :--- | :--- |
| **Python** | Core programming language | [python.org](https://www.python.org/) |
| **Streamlit** | Interactive web dashboard framework | [streamlit.io](https://streamlit.io/) |
| **TextBlob** | Natural language processing & sentiment scoring | [textblob.readthedocs.io](https://textblob.readthedocs.io/) |
| **yfinance** | Real-time and historical financial market data | [github.com/ranaroussi/yfinance](https://github.com/ranaroussi/yfinance) |
| **Plotly** | Financial charts, candlesticks, and heatmaps | [plotly.com/python](https://plotly.com/python/) |
| **Pandas** | Time-series data manipulation and alignment | [pandas.pydata.org](https://pandas.pydata.org/) |
| **NumPy** | Vectorized mathematical and statistical routines | [numpy.org](https://numpy.org/) |

---

## 🙏 Acknowledgments

- Inspired by and adapted from the Kaggle research notebook:  
  **[News Sentiment Based Trading Strategy](https://www.kaggle.com/code/shtrausslearning/news-sentiment-based-trading-strategy/notebook)** by [shtrausslearning](https://www.kaggle.com/shtrausslearning).
- Built with financial data accessible through the Yahoo Finance API via `yfinance`.
- Sentiment analysis powered by TextBlob's pattern-analyzer lexicon.

---

## ⚠️ Disclaimer

> [!WARNING]
> **Educational & Research Purposes Only**: This software is designed exclusively for educational, illustrative, and academic research purposes. It does **not** constitute financial, investment, legal, or tax advice. Past backtested performance is no guarantee of future returns. Algorithmic trading entails significant financial risk, including the possible loss of principal capital. Always conduct independent due diligence or consult a licensed financial advisor before deploying real capital.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details.
