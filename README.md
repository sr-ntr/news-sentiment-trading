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