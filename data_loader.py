import pandas as pd
import yfinance as yf
import requests
from io import StringIO
import os
import pickle
from datetime import datetime, timedelta
import time
import random
import streamlit as st
import numpy as np

# Directorio para caché
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_sp500_tickers_cached():
    """Obtiene los tickers del S&P 500 con caché para evitar scraping frecuente"""
    cache_file = os.path.join(CACHE_DIR, "sp500_tickers.pkl")
    # Verificar si existe caché válido (menos de 7 días)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cache_time = cached_data.get('timestamp', 0)
                if (datetime.now().timestamp() - cache_time) < (7 * 24 * 3600):  # 7 días
                    print("Usando tickers S&P 500 desde caché")
                    return cached_data
        except Exception as e:
            print(f"Error leyendo caché S&P 500: {e}")
    # Headers para evitar bloqueos
    headers_list = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        },
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    ]
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    for attempt in range(3):
        try:
            headers = random.choice(headers_list)
            print(f"Intento {attempt + 1} de scraping S&P 500...")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            tables = pd.read_html(StringIO(response.text))
            if not tables:
                raise ValueError("No se encontraron tablas en la página")
            # Buscar la tabla principal (usualmente la primera)
            df = tables[0]
            # Verificar columnas esperadas
            if 'Symbol' not in df.columns and 'Ticker' in df.columns:
                df = df.rename(columns={'Ticker': 'Symbol'})
            if 'Symbol' not in df.columns:
                # Intentar encontrar la columna de tickers
                ticker_columns = [col for col in df.columns if 'symbol' in col.lower() or 'ticker' in col.lower()]
                if ticker_columns:
                    df = df.rename(columns={ticker_columns[0]: 'Symbol'})
                else:
                    raise ValueError("No se encontró columna de símbolos")
            tickers = df['Symbol'].tolist()
            # Limpiar tickers
            tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
            tickers = [t for t in tickers if t and t != 'nan']
            if not tickers:
                raise ValueError("No se encontraron tickers válidos")
            print(f"Obtenidos {len(tickers)} tickers S&P 500")
            # Guardar en caché
            tickers_data = {
                'tickers': tickers,
                'data': df.to_dict('records'),
                'timestamp': datetime.now().timestamp(),
                'date': datetime.now()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(tickers_data, f)
            # Esperar un poco para evitar bloqueos
            time.sleep(random.uniform(1, 2))
            return tickers_data
        except Exception as e:
            print(f"Intento {attempt + 1} falló: {e}")
            if attempt < 2:  # No esperar después del último intento
                time.sleep(random.uniform(2, 4))
            continue
    # Fallback: lista básica de tickers comunes
    print("Usando fallback de tickers S&P 500")
    fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'HD', 'DIS', 'PYPL']
    return {
        'tickers': fallback_tickers, 
        'data': [{'Symbol': t} for t in fallback_tickers], 
        'timestamp': datetime.now().timestamp(),
        'date': datetime.now()
    }

def get_nasdaq100_tickers_cached():
    """Obtiene los tickers del Nasdaq-100 con caché - CORREGIDO"""
    cache_file = os.path.join(CACHE_DIR, "nasdaq100_tickers.pkl")
    # Verificar si existe caché válido (menos de 7 días)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cache_time = cached_data.get('timestamp', 0)
                if (datetime.now().timestamp() - cache_time) < (7 * 24 * 3600):  # 7 días
                    print("Usando tickers Nasdaq-100 desde caché")
                    return cached_data
        except Exception as e:
            print(f"Error leyendo caché Nasdaq-100: {e}")
    # Headers para evitar bloqueos
    headers_list = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        },
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    ]
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    for attempt in range(5):  # Aumentado a 5 intentos
        try:
            headers = random.choice(headers_list)
            print(f"Intento {attempt + 1} de scraping Nasdaq-100...")
            response = requests.get(url, headers=headers, timeout=15)  # Aumentado timeout
            response.raise_for_status()
            tables = pd.read_html(StringIO(response.text))
            if not tables:
                raise ValueError("No se encontraron tablas en la página")
            
            # Estrategia mejorada para encontrar la tabla de constituyentes
            df = None
            target_table_index = None
            
            # Buscar tabla con columna "Ticker" o "Symbol"
            for i, table in enumerate(tables):
                print(f"Revisando tabla {i} con columnas: {list(table.columns)}")
                # Buscar columnas que contengan "Ticker" o "Symbol"
                ticker_cols = [col for col in table.columns if 'Ticker' in str(col) or 'Symbol' in str(col)]
                if ticker_cols:
                    df = table
                    target_table_index = i
                    print(f"Encontrada tabla {i} con columna ticker: {ticker_cols[0]}")
                    break
            
            # Si no encontramos por nombre de columna, usar la tercera tabla (índice 2) que típicamente es la de constituyentes
            if df is None:
                if len(tables) >= 3:
                    df = tables[2]
                    target_table_index = 2
                    print("Usando tabla 3 (índice 2) como tabla de constituyentes por defecto")
                else:
                    raise ValueError("No se encontró tabla de constituyentes")
            
            # Verificar y renombrar columna de tickers si es necesario
            ticker_column = None
            for col in df.columns:
                if 'Ticker' in str(col) or 'Symbol' in str(col):
                    ticker_column = col
                    break
            
            if ticker_column is None:
                # Usar la primera columna como fallback
                ticker_column = df.columns[0]
                print(f"No se encontró columna de ticker clara, usando la primera columna: '{ticker_column}'")
            
            # Extraer tickers
            tickers = df[ticker_column].tolist()
            print(f"Tickers brutos extraídos: {tickers[:10]}...")  # Mostrar primeros 10 para debugging
            
            # Limpiar tickers con validaciones más estrictas
            clean_tickers = []
            for t in tickers:
                t_str = str(t).strip().upper()
                # Validar que sea un ticker real (no numérico, no vacío, no NaN)
                if t_str and t_str != 'NAN' and not t_str.isdigit() and len(t_str) <= 5:
                    clean_tickers.append(t_str)
            
            tickers = list(set(clean_tickers))  # Eliminar duplicados
            print(f"Tickers limpios: {tickers[:10]}...")  # Mostrar primeros 10 para debugging
            
            if not tickers:
                raise ValueError("No se encontraron tickers válidos después de limpieza")
            
            print(f"Obtenidos {len(tickers)} tickers Nasdaq-100")
            print(f"Primeros 10 tickers: {tickers[:10]}")
            
            # Guardar en caché
            tickers_data = {
                'tickers': tickers,
                'data': df.to_dict('records'),
                'timestamp': datetime.now().timestamp(),
                'date': datetime.now()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(tickers_data, f)
            # Esperar un poco
            time.sleep(random.uniform(1, 2))
            return tickers_data
        except Exception as e:
            print(f"Intento {attempt + 1} falló: {e}")
            import traceback
            traceback.print_exc()
            if attempt < 4:  # 5 intentos en total
                time.sleep(random.uniform(3, 6))  # Espera más larga entre intentos
            continue
    
    # Fallback más completo
    print("Usando fallback de tickers Nasdaq-100")
    fallback_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'PEP', 'COST', 
        'AVGO', 'CMCSA', 'CSCO', 'INTC', 'QCOM', 'TXN', 'NFLX', 'PYPL', 'SBUX', 'ADP',
        'VRTX', 'GILD', 'MDLZ', 'ISRG', 'CSX', 'ATVI', 'TMUS', 'AMD', 'REGN', 'BKNG'
    ]
    return {
        'tickers': fallback_tickers, 
        'data': [{'Symbol': t} for t in fallback_tickers], 
        'timestamp': datetime.now().timestamp(),
        'date': datetime.now()
    }

def get_constituents_at_date(index_name, start_date, end_date):
    """
    Obtiene los constituyentes de un índice en una fecha dada
    """
    if index_name == "SP500":
        tickers_data = get_sp500_tickers_cached()
    elif index_name == "NDX":
        tickers_data = get_nasdaq100_tickers_cached()
    else:
        raise ValueError(f"Índice {index_name} no soportado")
    return tickers_data, None

def download_prices_with_retry(tickers, start_date, end_date, max_retries=5):
    """
    Descarga precios con reintentos y manejo de errores mejorado
    Descarga un ticker a la vez para mayor estabilidad
    """
    all_data = {}
    
    # Si tickers es una lista, procesar uno a uno
    if isinstance(tickers, list):
        for i, ticker in enumerate(tickers):
            print(f"Descargando {ticker} ({i+1}/{len(tickers)})...")
            
            ticker_data = None
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        print(f"  Reintento {attempt} para {ticker}...")
                        time.sleep(random.uniform(5, 10))
                    
                    # Descargar un solo ticker a la vez
                    data = yf.download(
                        ticker,  # Solo un ticker
                        start=start_date, 
                        end=end_date, 
                        group_by='ticker',
                        progress=False,
                        threads=False,  # Desactivar threads para estabilidad
                        timeout=30,
                        auto_adjust=True,
                        repair=True
                    )
                    
                    if not data.empty:
                        all_data[ticker] = data
                        ticker_data = data
                        print(f"  ✅ {ticker} descargado correctamente")
                        break
                    else:
                        raise ValueError(f"No se recibieron datos para {ticker}")
                        
                except Exception as e:
                    print(f"  Error en intento {attempt + 1} para {ticker}: {e}")
                    if "delisted" in str(e).lower() or "timezone" in str(e).lower():
                        print(f"  Ticker probablemente delistado, saltando: {ticker}")
                        break
                    if attempt == max_retries - 1:
                        print(f"  ❌ Fallo definitivo para {ticker}")
                        break
                    continue
            
            # Espera fija de 10 segundos cada 25 tickers (excepto al final)
            if (i + 1) % 25 == 0 and i < len(tickers) - 1:
                print(f"  Esperando 10 segundos después de {i+1} tickers...")
                time.sleep(10)
    
    else:
        # Si es un solo ticker (string)
        ticker = tickers
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"Reintento {attempt} de descarga de precios...")
                    time.sleep(random.uniform(5, 10))
                
                data = yf.download(
                    ticker,
                    start=start_date, 
                    end=end_date, 
                    group_by='ticker',
                    progress=False,
                    threads=False,
                    timeout=30,
                    auto_adjust=True,
                    repair=True
                )
                
                if data.empty:
                    raise ValueError("No se recibieron datos")
                return data
            except Exception as e:
                print(f"Error en intento {attempt + 1}: {e}")
                if "delisted" in str(e).lower() or "timezone" in str(e).lower():
                    print(f"Ticker probablemente delistado, saltando: {ticker}")
                    return pd.DataFrame()
                if attempt == max_retries - 1:
                    return pd.DataFrame()
                continue
    
    # Combinar todos los datos en un DataFrame
    if all_data:
        # Para datos individuales, mantener la estructura original
        combined_data = pd.concat(all_data, axis=1)
        return combined_data
    else:
        return pd.DataFrame()

# data_loader.py
import pandas as pd
import yfinance as yf
import numpy as np
import time, random, datetime as dt

def download_prices(tickers, start_date, end_date):
    """
    Descarga historiales para S&P-500 / Nasdaq-100
    100 % en modo "1-ticker-a-la-vez" para evitar el cuelgue masivo.
    """
    # ---------- 1. normalizar entrada ----------
    if isinstance(tickers, dict) and 'tickers' in tickers:
        ticker_list = tickers['tickers']
    elif isinstance(tickers, pd.DataFrame):
        col = 'Symbol' if 'Symbol' in tickers.columns else tickers.columns[0]
        ticker_list = tickers[col].tolist()
    elif isinstance(tickers, str):
        ticker_list = [tickers]
    else:
        ticker_list = list(tickers)

    ticker_list = [str(t).strip().upper().replace('.', '-') for t in ticker_list]
    ticker_list = [t for t in ticker_list if t and not t.isdigit() and len(t) <= 6]
    ticker_list = list(dict.fromkeys(ticker_list))          # orden + únicos
    if not ticker_list:
        return pd.DataFrame()

    print(f"Descargando {len(ticker_list)} tickers (modo 1×1)…")

    # ---------- 2. bucle 1×1 ----------
    prices = {}
    for i, tk in enumerate(ticker_list, 1):
        print(f"  {i:3d}/{len(ticker_list)}  {tk}")
        for attempt in range(1, 4):                         # 3 intentos
            try:
                df0 = yf.download(
                    tk,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    threads=False,          # ➕ clave
                    timeout=30,
                    auto_adjust=True,
                    repair=True,
                    interval="1d"
                )
                if df0.empty:
                    raise ValueError("vacío")

                # ---------- 3. forzar 1-D ----------
                # prioridad: Adj Close → Close → 1ª numérica
                for col in ("Adj Close", "Close"):
                    if col in df0.columns:
                        sr = df0[col].copy()
                        break
                else:
                    sr = df0.select_dtypes(np.number).iloc[:, 0].copy()

                sr = pd.Series(sr.squeeze(), name=tk)        # ➕ squeeze + Series
                sr.index = pd.to_datetime(sr.index)
                sr = sr.dropna()
                sr = sr[~sr.index.duplicated()]              # índice único

                if sr.empty:
                    raise ValueError("serie vacía")

                prices[tk] = sr
                break                                      # éxito → siguiente ticker

            except Exception as e:
                print(f"    └─ intento {attempt}: {e}")   # log breve
                time.sleep(random.uniform(2, 6))
        else:
            print(f"    └─ ❌  {tk}  – descartado")        # agotados intentos
            continue

        # ---------- 4. pausa cada 50 ----------
        if i % 50 == 0 and i < len(ticker_list):
            print("⏱  pausa 10 s …")
            time.sleep(10)

    # ---------- 5. armar DataFrame ----------
    if not prices:
        return pd.DataFrame()

    master = pd.DataFrame(prices)
    master = master.dropna(how='all').dropna(axis=1, how='all')
    print(f"✅  DataFrame final  {master.shape}")
    return master
