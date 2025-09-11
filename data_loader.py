"""
data_loader.py
Descarga históricos de yfinance con:
  - rate-limiting (1 req/s)
  - caché local SQLite (1 h)
  - descarga por lotes (20 tickers / lote)
  - manejo de shape-2D y errores de red
"""

import os
import random
import time
from datetime import timedelta
from typing import List, Union, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------------
# 1. Session con caché + rate-limit
# ------------------------------------------------------------------
try:
    from requests_cache import CacheMixin
    from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
    from pyrate_limiter import Duration, RequestRate, Limiter
    from requests import Session

    class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
        pass

    limiter = Limiter(RequestRate(1, Duration.SECOND))  # 1 req/s
    SESSION = CachedLimiterSession(
        limiter=limiter,
        bucket_class=MemoryQueueBucket,
        backend="sqlite",
        expire_after=timedelta(hours=1),
    )
except ImportError:  # si no tienes las libs, session normal
    SESSION = None
    print("⚠️  requests-cache / requests-ratelimiter no encontradas. Sin caché ni rate-limit.")

# ------------------------------------------------------------------
# 2. Descarga individual de un ticker
# ------------------------------------------------------------------
def download_single(ticker: str, start, end, max_retry: int = 5) -> pd.Series:
    """Devuelve Serie de precios de cierre (Adj Close o Close)."""
    ticker = ticker.replace(".", "-").strip().upper()
    for attempt in range(1, max_retry + 1):
        try:
            time.sleep(2 ** (attempt - 1) * random.uniform(0.8, 1.2))  # backoff
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                repair=True,
                threads=False,
                timeout=30,
                session=SESSION,
            )
            if df.empty:
                raise ValueError("DataFrame vacío")

            # columna de cierre
            close = df["Adj Close"] if "Adj Close" in df else df["Close"]

            # si Yahoo devolvió (N,1) → aplanar
            if isinstance(close, pd.DataFrame) and close.shape[1] == 1:
                close = close.squeeze(axis=1)
            if not isinstance(close, pd.Series):
                close = pd.Series(close, index=df.index)

            close = close.replace([np.inf, -np.inf], np.nan).dropna()
            if close.empty:
                raise ValueError("Serie vacía tras limpieza")
            close.name = ticker
            return close

        except Exception as e:
            print(f"⚠️  {ticker} intento {attempt}/{max_retry}: {e}")
            if attempt == max_retry:
                return pd.Series(dtype=float, name=ticker)
    return pd.Series(dtype=float, name=ticker)

# ------------------------------------------------------------------
# 3. Descarga por lotes
# ------------------------------------------------------------------
def download_prices(
    tickers: Union[List[str], Dict[str, List[str]], str],
    start_date,
    end_date,
    batch_size: int = 20,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Devuelve DataFrame (fecha × tickers) con precios de cierre."""
    # normalizar entrada
    if isinstance(tickers, dict) and "tickers" in tickers:
        tickers = tickers["tickers"]
    if isinstance(tickers, str):
        tickers = [tickers]

    tickers = list(
        {
            t.replace(".", "-").strip().upper()
            for t in map(str, tickers)
            if t.strip() and len(t) <= 5 and not t.isdigit()
        }
    )
    if not tickers:
        print("❌ No hay tickers válidos")
        return pd.DataFrame()

    print(f"Descargando {len(tickers)} tickers en lotes de {batch_size}...")

    all_series = {}
    total_lotes = (len(tickers) - 1) // batch_size + 1
    for i in range(0, len(tickers), batch_size):
        lote = tickers[i : i + batch_size]
        print(f"Lote {i//batch_size + 1}/{total_lotes}")
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            fut_to_t = {
                exe.submit(download_single, t, start_date, end_date): t for t in lote
            }
            for fut in as_completed(fut_to_t):
                t = fut_to_t[fut]
                serie = fut.result()
                if not serie.empty:
                    all_series[t] = serie
        # pausa entre lotes
        time.sleep(random.uniform(2, 4))

    if not all_series:
        print("❌ Ningún ticker pudo descargarse")
        return pd.DataFrame()

    prices = pd.DataFrame(all_series)
    prices = prices.dropna(axis=1, how="all").dropna(axis=0, how="all")
    print(
        f"✅ Descarga finalizada: {prices.shape[1]} tickers, {len(prices)} filas"
    )
    return prices

# ------------------------------------------------------------------
# 4. Funciones legacy (mantener compatibilidad)
# ------------------------------------------------------------------
def download_prices_with_retry(tickers, start_date, end_date, max_retries=5):
    """Wrapper legacy: usa el nuevo download_prices."""
    return download_prices(tickers, start_date, end_date)

def get_sp500_tickers_cached():
    """Devuelve {'tickers': [...], 'data': [...], 'timestamp': ...}"""
    import utils
    return utils.load_tickers_sp500()

def get_nasdaq100_tickers_cached():
    import utils
    return utils.load_tickers_nasdaq100()

def get_constituents_at_date(index_name, start_date, end_date):
    if index_name == "SP500":
        return get_sp500_tickers_cached(), None
    if index_name == "NDX":
        return get_nasdaq100_tickers_cached(), None
    raise ValueError(f"Índice {index_name} no soportado")
