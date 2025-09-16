import pandas as pd
import requests
from io import StringIO
import os
import random
import time
from datetime import datetime, timedelta
import numpy as np
import re
from dateutil import parser
import glob
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
_historical_cache = {}

def parse_wikipedia_date(date_str):
    if pd.isna(date_str) or not date_str or str(date_str).lower() in ['nan', 'none', '']:
        return None
    date_str = str(date_str).strip()
    try:
        parsed_date = parser.parse(date_str, fuzzy=True)
        return parsed_date.date()
    except:
        try:
            month_map = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12,
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
                'oct': 10, 'nov': 11, 'dec': 12
            }
            clean_date = re.sub(r'[^\w\s,]', ' ', date_str.lower())
            parts = clean_date.split()
            if len(parts) >= 3:
                month = None
                for part in parts:
                    if part.replace(',', '') in month_map:
                        month = month_map[part.replace(',', '')]
                        break
                numbers = [int(re.findall(r'\d+', part)[0]) for part in parts if re.findall(r'\d+', part)]
                if month and len(numbers) >= 2:
                    day = min(numbers)
                    year = max(numbers)
                    if day <= 31 and year >= 1900:
                        from datetime import date
                        return date(year, month, day)
            return None
        except:
            return None


# ... funciones get_sp500_historical_changes(), get_nasdaq100_historical_changes(), etc.
# (mantienen su lógica original, no requieren optimización de rendimiento)

# Solo incluyo aquí las funciones clave optimizadas:

def get_sp500_historical_changes():
    print("Descargando cambios históricos del S&P 500...")
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        changes_df = None
        for i, table in enumerate(tables):
            if hasattr(table.columns, 'nlevels') and table.columns.nlevels > 1:
                level_0 = [str(col[0]).lower() for col in table.columns]
                if any('date' in col or 'effective' in col for col in level_0) and 'added' in level_0 and 'removed' in level_0:
                    changes_df = table
                    break
            else:
                col_text = ' '.join([str(col) for col in table.columns]).lower()
                if ('effective' in col_text or 'date' in col_text) and 'added' in col_text and 'removed' in col_text:
                    changes_df = table
                    break
        if changes_df is None and len(tables) >= 3:
            changes_df = tables[2]
        if changes_df is None:
            return pd.DataFrame()
        # ... resto de lógica sin cambios ...
        # (por brevedad, asumo que el resto sigue igual)
        # IMPORTANTE: esta función no es cuello de botella, así que no la optimizo aquí
        # Pero en tu repo real, déjala como está
        return changes_df.iloc[:0]  # Placeholder - usa tu versión original aquí
    except Exception as e:
        print(f"❌ Error S&P 500: {e}")
        return pd.DataFrame()


def download_prices(tickers, start_date, end_date, load_full_data=True):
    """
    Carga precios desde archivos CSV en paralelo
    """
    prices_data = {}
    ohlc_data = {}

    if isinstance(tickers, dict) and 'tickers' in tickers:
        ticker_list = tickers['tickers']
    elif isinstance(tickers, (list, tuple)):
        ticker_list = list(tickers)
    elif isinstance(tickers, str):
        ticker_list = [tickers]
    else:
        ticker_list = []

    ticker_list = [str(t).strip().upper().replace('.', '-') for t in ticker_list]
    ticker_list = [t for t in ticker_list if t and len(t) <= 6 and not t.isdigit()]
    ticker_list = list(dict.fromkeys(ticker_list))

    if not ticker_list:
        return pd.DataFrame()

    print(f"📂 Cargando {len(ticker_list)} tickers en paralelo...")

    def load_single_ticker(ticker):
        csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(csv_path):
            return ticker, None, None
        try:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if isinstance(start_date, datetime):
                start_date_obj = start_date.date()
            else:
                start_date_obj = start_date
            if isinstance(end_date, datetime):
                end_date_obj = end_date.date()
            else:
                end_date_obj = end_date
            df = df[(df.index.date >= start_date_obj) & (df.index.date <= end_date_obj)]
            if df.empty:
                return ticker, None, None
            price_series = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
            ohlc = None
            if all(col in df.columns for col in ['High', 'Low', 'Close']):
                ohlc = {
                    'High': df['High'],
                    'Low': df['Low'],
                    'Close': df['Adj Close'] if 'Adj Close' in df.columns else df['Close'],
                    'Volume': df.get('Volume')
                }
            return ticker, price_series, ohlc
        except Exception as e:
            print(f"❌ Error cargando {ticker}: {e}")
            return ticker, None, None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(load_single_ticker, ticker) for ticker in ticker_list]
        for future in futures:
            ticker, price, ohlc = future.result()
            if price is not None:
                prices_data[ticker] = price
            if ohlc is not None:
                ohlc_data[ticker] = ohlc

    if prices_data:
        prices_df = pd.DataFrame(prices_data)
        prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
        if load_full_data and ohlc_data:
            return prices_df, ohlc_data
        return prices_df
    else:
        return pd.DataFrame()


# Funciones auxiliares (mantén tu implementación original)
def get_nasdaq100_historical_changes(): pass  # Usa tu versión
def get_current_constituents(index_name): pass  # Usa tu versión
def get_sp500_tickers_from_wikipedia(): pass   # Usa tu versión
def get_nasdaq100_tickers_from_wikipedia(): pass # Usa tu versión
def get_all_available_tickers_with_historical_validation(index_name, start_date, end_date): pass
def get_constituents_at_date(index_name, start_date, end_date): pass
def generate_removed_tickers_summary(): pass
