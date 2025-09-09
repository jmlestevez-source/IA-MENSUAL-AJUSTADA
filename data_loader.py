import pandas as pd
import yfinance as yf
import requests
from io import StringIO
import os
import pickle
from datetime import datetime

# Directorio para caché
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_sp500_tickers_cached():
    """Obtiene los tickers del S&P 500 con caché para evitar scraping frecuente"""
    cache_file = os.path.join(CACHE_DIR, "sp500_tickers.pkl")
    
    # Verificar si existe caché válido (menos de 7 días)
    if os.path.exists(cache_file):
        cache_time = os.path.getmtime(cache_file)
        if (datetime.now().timestamp() - cache_time) < (7 * 24 * 3600):  # 7 días
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    # Si no hay caché válido, hacer scraping
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text), match="Ticker")
        df = tables[0]
        
        # Guardar en caché
        tickers_data = {
            'tickers': df['Symbol'].tolist(),
            'data': df.to_dict('records'),
            'date': datetime.now()
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(tickers_data, f)
            
        return tickers_data
        
    except Exception as e:
        print(f"Error obteniendo tickers S&P 500: {e}")
        # Fallback: lista básica de tickers comunes
        fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
        return {'tickers': fallback_tickers, 'data': [], 'date': datetime.now()}

def get_nasdaq100_tickers_cached():
    """Obtiene los tickers del Nasdaq-100 con caché"""
    cache_file = os.path.join(CACHE_DIR, "nasdaq100_tickers.pkl")
    
    # Verificar si existe caché válido (menos de 7 días)
    if os.path.exists(cache_file):
        cache_time = os.path.getmtime(cache_file)
        if (datetime.now().timestamp() - cache_time) < (7 * 24 * 3600):  # 7 días
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    # Si no hay caché válido, hacer scraping
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        url = "https://en.wikipedia.org/wiki/NASDAQ-100"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text), match="Ticker")
        df = tables[0]
        
        # Guardar en caché
        tickers_data = {
            'tickers': df['Ticker'].tolist(),
            'data': df.to_dict('records'),
            'date': datetime.now()
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(tickers_data, f)
            
        return tickers_data
        
    except Exception as e:
        print(f"Error obteniendo tickers Nasdaq-100: {e}")
        # Fallback: lista básica de tickers comunes
        fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'PEP', 'COST']
        return {'tickers': fallback_tickers, 'data': [], 'date': datetime.now()}

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
    
    # Crear DataFrame con los datos
    if tickers_data['data']:
        df = pd.DataFrame(tickers_data['data'])
    else:
        df = pd.DataFrame({'Symbol': tickers_data['tickers']}) if index_name == "SP500" else pd.DataFrame({'Ticker': tickers_data['tickers']})
    
    return df, None

def download_prices(tickers, start_date, end_date):
    """
    Descarga precios históricos para una lista de tickers
    """
    try:
        # Unificar formato de tickers
        if isinstance(tickers, pd.DataFrame):
            # Extraer tickers del DataFrame
            if 'Symbol' in tickers.columns:
                ticker_list = tickers['Symbol'].tolist()
            elif 'Ticker' in tickers.columns:
                ticker_list = tickers['Ticker'].tolist()
            else:
                ticker_list = tickers.iloc[:, 0].tolist()
        else:
            ticker_list = list(tickers)
        
        # Limpiar tickers (eliminar espacios y vacíos)
        ticker_list = [str(t).strip() for t in ticker_list if str(t).strip()]
        
        if not ticker_list:
            raise ValueError("No se encontraron tickers válidos")
        
        print(f"Descargando datos para {len(ticker_list)} tickers...")
        
        # Descargar datos
        data = yfinance.download(ticker_list, start=start_date, end=end_date, group_by='ticker')
        
        if data.empty:
            raise ValueError("No se pudieron descargar datos")
        
        # Procesar datos para crear un DataFrame limpio
        prices = {}
        for ticker in ticker_list:
            try:
                if len(ticker_list) > 1:
                    ticker_data = data[ticker]
                else:
                    ticker_data = data
                
                if not ticker_data.empty and 'Adj Close' in ticker_data.columns:
                    prices[ticker] = ticker_data['Adj Close']
                elif not ticker_data.empty and 'Close' in ticker_data.columns:
                    prices[ticker] = ticker_data['Close']
            except Exception as e:
                print(f"Error procesando {ticker}: {e}")
                continue
        
        if not prices:
            raise ValueError("No se pudieron procesar los datos descargados")
        
        # Crear DataFrame final
        prices_df = pd.DataFrame(prices)
        
        return prices_df
        
    except Exception as e:
        print(f"Error en download_prices: {e}")
        return None
