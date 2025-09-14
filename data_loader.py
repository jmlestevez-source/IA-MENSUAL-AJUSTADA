import pandas as pd
import requests
from io import StringIO
import os
import random
import time
from datetime import datetime
import numpy as np

# Directorio para datos
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_sp500_tickers_from_wikipedia():
    """Obtiene los tickers del S&P 500 directamente de Wikipedia"""
    print("Obteniendo tickers S&P 500 desde Wikipedia...")
    
    # Headers para evitar bloqueos
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        
        if not tables:
            raise ValueError("No se encontraron tablas en la página de Wikipedia")
        
        # Usar la primera tabla (que contiene los constituyentes)
        df = tables[0]
        
        # Verificar que tenemos la columna de símbolos
        symbol_column = None
        for col in df.columns:
            if 'symbol' in str(col).lower() or 'ticker' in str(col).lower():
                symbol_column = col
                break
        
        if symbol_column is None:
            # Usar la primera columna como fallback
            symbol_column = df.columns[0]
        
        # Extraer y limpiar tickers
        tickers = df[symbol_column].astype(str).str.strip().str.upper().tolist()
        tickers = [t.replace('.', '-') for t in tickers]  # Convertir puntos a guiones
        tickers = [t for t in tickers if t and t != 'nan' and len(t) <= 6 and not t.isdigit()]
        
        if not tickers:
            raise ValueError("No se encontraron tickers válidos")
        
        print(f"Obtenidos {len(tickers)} tickers S&P 500 desde Wikipedia")
        
        return {
            'tickers': tickers,
            'data': df.to_dict('records'),
            'timestamp': datetime.now().timestamp(),
            'date': datetime.now()
        }
        
    except Exception as e:
        print(f"Error obteniendo tickers S&P 500 desde Wikipedia: {e}")
        # Fallback básico
        fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
        return {
            'tickers': fallback_tickers,
            'data': [{'Symbol': t} for t in fallback_tickers],
            'timestamp': datetime.now().timestamp(),
            'date': datetime.now()
        }

def get_nasdaq100_tickers_from_wikipedia():
    """Obtiene los tickers del Nasdaq-100 directamente de Wikipedia"""
    print("Obteniendo tickers Nasdaq-100 desde Wikipedia...")
    
    # Headers para evitar bloqueos
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        
        if not tables:
            raise ValueError("No se encontraron tablas en la página de Wikipedia")
        
        # Buscar la tabla que contiene los constituyentes (usualmente una de las primeras)
        df = None
        for table in tables:
            # Buscar tabla con columnas que contengan "Ticker" o "Symbol"
            ticker_cols = [col for col in table.columns if 'Ticker' in str(col) or 'Symbol' in str(col)]
            if ticker_cols:
                df = table
                break
        
        # Si no encontramos por nombre de columna, usar la tercera tabla (índice 2) que típicamente es la de constituyentes
        if df is None:
            if len(tables) >= 3:
                df = tables[2]
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
        
        # Extraer y limpiar tickers
        tickers = df[ticker_column].astype(str).str.strip().str.upper().tolist()
        tickers = [t.replace('.', '-') for t in tickers]  # Convertir puntos a guiones
        tickers = [t for t in tickers if t and t != 'nan' and len(t) <= 6 and not t.isdigit()]
        
        if not tickers:
            raise ValueError("No se encontraron tickers válidos")
        
        print(f"Obtenidos {len(tickers)} tickers Nasdaq-100 desde Wikipedia")
        
        return {
            'tickers': tickers,
            'data': df.to_dict('records'),
            'timestamp': datetime.now().timestamp(),
            'date': datetime.now()
        }
        
    except Exception as e:
        print(f"Error obteniendo tickers Nasdaq-100 desde Wikipedia: {e}")
        # Fallback básico
        fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'PEP', 'COST']
        return {
            'tickers': fallback_tickers,
            'data': [{'Symbol': t} for t in fallback_tickers],
            'timestamp': datetime.now().timestamp(),
            'date': datetime.now()
        }

def get_constituents_at_date(index_name, start_date, end_date):
    """
    Obtiene los constituyentes de un índice en una fecha dada
    Siempre lee directamente de Wikipedia
    """
    try:
        if index_name == "SP500":
            tickers_data = get_sp500_tickers_from_wikipedia()
        elif index_name == "NDX":
            tickers_data = get_nasdaq100_tickers_from_wikipedia()
        else:
            raise ValueError(f"Índice {index_name} no soportado")
        return tickers_data, None
    except Exception as e:
        return None, str(e)

def download_prices(tickers, start_date, end_date):
    """
    Carga precios desde archivos CSV en la carpeta data/
    Formato esperado: data/AAPL.csv, data/MSFT.csv, etc.
    """
    prices_data = {}
    
    # Normalizar entrada de tickers
    if isinstance(tickers, dict) and 'tickers' in tickers:
        ticker_list = tickers['tickers']
    elif isinstance(tickers, (list, tuple)):
        ticker_list = list(tickers)
    elif isinstance(tickers, str):
        ticker_list = [tickers]
    else:
        ticker_list = []
    
    # Limpiar y normalizar tickers
    ticker_list = [str(t).strip().upper().replace('.', '-') for t in ticker_list]
    ticker_list = [t for t in ticker_list if t and not t.isdigit() and len(t) <= 6]
    ticker_list = list(dict.fromkeys(ticker_list))  # Eliminar duplicados
    
    if not ticker_list:
        return pd.DataFrame()
    
    print(f"Cargando {len(ticker_list)} tickers desde CSV...")
    
    for ticker in ticker_list:
        csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if os.path.exists(csv_path):
            try:
                # Leer CSV con Date como índice
                df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
                
                # Filtrar por rango de fechas
                if isinstance(start_date, datetime):
                    start_date = start_date.date()
                if isinstance(end_date, datetime):
                    end_date = end_date.date()
                    
                df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
                
                if not df.empty:
                    # Prioridad: Adj Close → Close → primera columna numérica
                    price_series = None
                    for col in ["Adj Close", "Close"]:
                        if col in df.columns:
                            price_series = df[col]
                            break
                    
                    if price_series is None:
                        # Usar primera columna numérica si no hay Close/Adj Close
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            price_series = df[numeric_cols[0]]
                    
                    if price_series is not None:
                        price_series.name = ticker  # Nombrar la serie con el ticker
                        prices_data[ticker] = price_series
                else:
                    print(f"  No hay datos para {ticker} en el rango de fechas")
            except Exception as e:
                print(f"  Error cargando {ticker}: {e}")
        else:
            print(f"  Archivo no encontrado: {csv_path}")
    
    # Combinar en DataFrame
    if prices_data:
        prices_df = pd.DataFrame(prices_data)
        # Rellenar valores faltantes hacia adelante y hacia atrás
        prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
        return prices_df
    else:
        return pd.DataFrame()
