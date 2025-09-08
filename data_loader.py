import yfinance as yf, pandas as pd, streamlit as st
from utils import unify_ticker, pause_every_100, save_cache, load_cache

@st.cache_data(show_spinner=False)
def download_prices(tickers, start, end):
    cache_file = "prices.pkl"
    cached = load_cache(cache_file)
    if cached is not None:
        return cached

    prices = pd.DataFrame()
    for i, ticker in enumerate(tickers):
        ticker = unify_ticker(ticker)
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
            if not df.empty:
                prices[ticker] = df["Close"]
        except Exception as e:
            print(f"Error {ticker}: {e}")
        pause_every_100(i)
    save_cache(prices, cache_file)
    return prices
