import os, time, pickle, streamlit as st

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def pause_every_100(i, step=100, seconds=2):
    if (i + 1) % step == 0:
        time.sleep(seconds)

@st.cache_data(show_spinner=False)
def load_tickers_sp500():
    import pandas as pd
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = tables[0]
    changes = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", match="Changes")[0]
    return df, changes

@st.cache_data(show_spinner=False)
def load_tickers_nasdaq100():
    import pandas as pd
    tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    df = tables[2]  # tabla "constituents"
    changes = tables[3] if len(tables) > 3 else pd.DataFrame()
    return df, changes

def unify_ticker(ticker):
    """
    Unifica el formato de los tickers
    """
    if pd.isna(ticker) or ticker is None:
        return None
    
    ticker = str(ticker).strip().upper()
    
    # Manejar casos especiales
    ticker = ticker.replace('.', '-')  # Convertir puntos a guiones
    
    return ticker if ticker else None

def save_cache(df, filename):
    with open(os.path.join(CACHE_DIR, filename), "wb") as f:
        pickle.dump(df, f)

def load_cache(filename):
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None
