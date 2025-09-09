import yfinance as yf, pandas as pd, streamlit as st
from utils import unify_ticker, pause_every_100, save_cache, load_cache

# ----------  data_loader.py (solo añadir estas líneas) ----------
def get_constituents_at_date(index_name, start, end):
    """
    Devuelve (DataFrame[ticker, name], DataFrame de cambios)
    index_name : 'SP500' | 'NASDAQ100'
    start, end : solo para firma compatible; usamos lista actual
    """
    import pandas as pd
    if index_name.upper() == "SP500":
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, match="Ticker")
        df = tables[0][["Symbol", "Security"]].rename(
            columns={"Symbol": "Ticker", "Security": "Name"})
    elif index_name.upper() == "NASDAQ100":
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url, match="Ticker")
        df = tables[2][["Ticker", "Company"]].rename(
            columns={"Company": "Name"})
    else:
        raise ValueError("index_name debe ser 'SP500' o 'NASDAQ100'")

    df["Ticker"] = df["Ticker"].str.replace("-", ".")
    changes = pd.DataFrame()  # lista vacía de cambios
    return df, changes


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
