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

# Directorio para datos
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Cache para datos históricos
_historical_cache = {}

def parse_wikipedia_date(date_str):
    """Parsea fechas de Wikipedia en diferentes formatos"""
    if pd.isna(date_str) or not date_str or str(date_str).lower() in ['nan', 'none', '']:
        return None
    
    date_str = str(date_str).strip()
    
    # Patrones comunes en Wikipedia
    patterns = [
        r'(\d{4})-(\d{1,2})-(\d{1,2})',  # 2020-12-15
        r'(\d{1,2})/(\d{1,2})/(\d{4})',  # 12/15/2020
        r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', # December 15, 2020
        r'(\d{1,2})\s+(\w+)\s+(\d{4})',   # 15 December 2020
    ]
    
    try:
        # Intentar parsing directo
        return parser.parse(date_str, fuzzy=True).date()
    except:
        try:
            # Limpiar string
            date_clean = re.sub(r'[^\w\s\-/,]', '', date_str)
            return parser.parse(date_clean, fuzzy=True).date()
        except:
            return None

def get_sp500_historical_changes():
    """Obtiene los cambios históricos del S&P 500 desde Wikipedia"""
    print("Descargando cambios históricos del S&P 500...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        
        # Buscar tabla de cambios (normalmente la segunda tabla)
        changes_df = None
        for i, table in enumerate(tables):
            # Buscar tabla que contenga columnas relacionadas con cambios
            cols = [str(col).lower() for col in table.columns]
            if any('date' in col for col in cols) and any('add' in col for col in cols):
                changes_df = table
                print(f"Encontrada tabla de cambios en posición {i}")
                break
        
        if changes_df is None:
            print("No se encontró tabla de cambios, usando tabla por defecto")
            if len(tables) > 1:
                changes_df = tables[1]  # Tabla típica de cambios
            else:
                return pd.DataFrame()
        
        # Normalizar nombres de columnas
        changes_df.columns = [str(col).strip() for col in changes_df.columns]
        
        # Buscar columnas relevantes
        date_col = None
        added_col = None
        removed_col = None
        
        for col in changes_df.columns:
            col_lower = col.lower()
            if 'date' in col_lower and date_col is None:
                date_col = col
            elif 'add' in col_lower and added_col is None:
                added_col = col
            elif 'remov' in col_lower and removed_col is None:
                removed_col = col
        
        if not all([date_col, added_col, removed_col]):
            print(f"Columnas encontradas: {changes_df.columns.tolist()}")
            # Intentar con las primeras 3 columnas
            if len(changes_df.columns) >= 3:
                date_col, added_col, removed_col = changes_df.columns[:3]
            else:
                return pd.DataFrame()
        
        # Procesar datos
        changes_clean = []
        for _, row in changes_df.iterrows():
            try:
                date_str = str(row[date_col]).strip()
                date_parsed = parse_wikipedia_date(date_str)
                
                if date_parsed is None:
                    continue
                
                added_ticker = str(row[added_col]).strip().upper() if pd.notna(row[added_col]) else None
                removed_ticker = str(row[removed_col]).strip().upper() if pd.notna(row[removed_col]) else None
                
                # Limpiar tickers
                if added_ticker and added_ticker not in ['NAN', 'NONE', '']:
                    added_ticker = added_ticker.replace('.', '-')
                    if len(added_ticker) <= 6 and not added_ticker.isdigit():
                        changes_clean.append({
                            'Date': date_parsed,
                            'Action': 'Added',
                            'Ticker': added_ticker
                        })
                
                if removed_ticker and removed_ticker not in ['NAN', 'NONE', '']:
                    removed_ticker = removed_ticker.replace('.', '-')
                    if len(removed_ticker) <= 6 and not removed_ticker.isdigit():
                        changes_clean.append({
                            'Date': date_parsed,
                            'Action': 'Removed',
                            'Ticker': removed_ticker
                        })
                        
            except Exception as e:
                continue
        
        if changes_clean:
            result_df = pd.DataFrame(changes_clean)
            result_df = result_df.sort_values('Date', ascending=False)  # Más reciente primero
            print(f"Procesados {len(result_df)} cambios históricos del S&P 500")
            return result_df
        else:
            print("No se pudieron procesar cambios históricos")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error obteniendo cambios históricos S&P 500: {e}")
        return pd.DataFrame()

def get_nasdaq100_historical_changes():
    """Obtiene los cambios históricos del NASDAQ-100 desde Wikipedia"""
    print("Descargando cambios históricos del NASDAQ-100...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        
        # Buscar tabla de cambios
        changes_df = None
        for i, table in enumerate(tables):
            cols = [str(col).lower() for col in table.columns]
            if any('date' in col for col in cols) and any('add' in col for col in cols):
                changes_df = table
                print(f"Encontrada tabla de cambios NASDAQ-100 en posición {i}")
                break
        
        if changes_df is None:
            print("No se encontró tabla de cambios para NASDAQ-100")
            return pd.DataFrame()
        
        # Procesar similar a S&P 500
        changes_df.columns = [str(col).strip() for col in changes_df.columns]
        
        date_col = None
        added_col = None
        removed_col = None
        
        for col in changes_df.columns:
            col_lower = col.lower()
            if 'date' in col_lower and date_col is None:
                date_col = col
            elif 'add' in col_lower and added_col is None:
                added_col = col
            elif 'remov' in col_lower and removed_col is None:
                removed_col = col
        
        if not all([date_col, added_col, removed_col]):
            if len(changes_df.columns) >= 3:
                date_col, added_col, removed_col = changes_df.columns[:3]
            else:
                return pd.DataFrame()
        
        changes_clean = []
        for _, row in changes_df.iterrows():
            try:
                date_str = str(row[date_col]).strip()
                date_parsed = parse_wikipedia_date(date_str)
                
                if date_parsed is None:
                    continue
                
                added_ticker = str(row[added_col]).strip().upper() if pd.notna(row[added_col]) else None
                removed_ticker = str(row[removed_col]).strip().upper() if pd.notna(row[removed_col]) else None
                
                if added_ticker and added_ticker not in ['NAN', 'NONE', '']:
                    added_ticker = added_ticker.replace('.', '-')
                    if len(added_ticker) <= 6 and not added_ticker.isdigit():
                        changes_clean.append({
                            'Date': date_parsed,
                            'Action': 'Added',
                            'Ticker': added_ticker
                        })
                
                if removed_ticker and removed_ticker not in ['NAN', 'NONE', '']:
                    removed_ticker = removed_ticker.replace('.', '-')
                    if len(removed_ticker) <= 6 and not removed_ticker.isdigit():
                        changes_clean.append({
                            'Date': date_parsed,
                            'Action': 'Removed',
                            'Ticker': removed_ticker
                        })
                        
            except Exception as e:
                continue
        
        if changes_clean:
            result_df = pd.DataFrame(changes_clean)
            result_df = result_df.sort_values('Date', ascending=False)
            print(f"Procesados {len(result_df)} cambios históricos del NASDAQ-100")
            return result_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error obteniendo cambios históricos NASDAQ-100: {e}")
        return pd.DataFrame()

def get_current_constituents(index_name):
    """Obtiene los constituyentes actuales de un índice"""
    if index_name == "SP500":
        return get_sp500_tickers_from_wikipedia()
    elif index_name == "NDX":
        return get_nasdaq100_tickers_from_wikipedia()
    else:
        raise ValueError(f"Índice {index_name} no soportado")

def get_sp500_tickers_from_wikipedia():
    """Obtiene los tickers actuales del S&P 500"""
    print("Obteniendo constituyentes actuales S&P 500...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        
        df = tables[0]  # Primera tabla contiene constituyentes actuales
        
        symbol_column = None
        for col in df.columns:
            if 'symbol' in str(col).lower() or 'ticker' in str(col).lower():
                symbol_column = col
                break
        
        if symbol_column is None:
            symbol_column = df.columns[0]
        
        tickers = df[symbol_column].astype(str).str.strip().str.upper().tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        tickers = [t for t in tickers if t and t != 'nan' and len(t) <= 6 and not t.isdigit()]
        
        print(f"Obtenidos {len(tickers)} constituyentes actuales S&P 500")
        
        return {
            'tickers': tickers,
            'data': df.to_dict('records'),
            'timestamp': datetime.now().timestamp(),
            'date': datetime.now()
        }
        
    except Exception as e:
        print(f"Error obteniendo constituyentes actuales S&P 500: {e}")
        fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
        return {
            'tickers': fallback_tickers,
            'data': [{'Symbol': t} for t in fallback_tickers],
            'timestamp': datetime.now().timestamp(),
            'date': datetime.now()
        }

def get_nasdaq100_tickers_from_wikipedia():
    """Obtiene los tickers actuales del NASDAQ-100"""
    print("Obteniendo constituyentes actuales NASDAQ-100...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        
        df = None
        for table in tables:
            ticker_cols = [col for col in table.columns if 'Ticker' in str(col) or 'Symbol' in str(col)]
            if ticker_cols:
                df = table
                break
        
        if df is None:
            if len(tables) >= 3:
                df = tables[2]
            else:
                raise ValueError("No se encontró tabla de constituyentes")
        
        ticker_column = None
        for col in df.columns:
            if 'Ticker' in str(col) or 'Symbol' in str(col):
                ticker_column = col
                break
        
        if ticker_column is None:
            ticker_column = df.columns[0]
        
        tickers = df[ticker_column].astype(str).str.strip().str.upper().tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        tickers = [t for t in tickers if t and t != 'nan' and len(t) <= 6 and not t.isdigit()]
        
        print(f"Obtenidos {len(tickers)} constituyentes actuales NASDAQ-100")
        
        return {
            'tickers': tickers,
            'data': df.to_dict('records'),
            'timestamp': datetime.now().timestamp(),
            'date': datetime.now()
        }
        
    except Exception as e:
        print(f"Error obteniendo constituyentes actuales NASDAQ-100: {e}")
        fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'PEP', 'COST']
        return {
            'tickers': fallback_tickers,
            'data': [{'Symbol': t} for t in fallback_tickers],
            'timestamp': datetime.now().timestamp(),
            'date': datetime.now()
        }

def get_constituents_at_date(index_name, start_date, end_date):
    """
    Obtiene constituyentes históricos válidos para el rango de fechas
    Ahora implementa verificación de fechas históricas real
    """
    cache_key = f"{index_name}_{start_date}_{end_date}"
    
    if cache_key in _historical_cache:
        print(f"Usando cache para {index_name}")
        return _historical_cache[cache_key], None
    
    try:
        # Obtener constituyentes actuales
        current_data = get_current_constituents(index_name)
        current_tickers = set(current_data['tickers'])
        
        # Obtener cambios históricos
        if index_name == "SP500":
            changes_df = get_sp500_historical_changes()
        elif index_name == "NDX":
            changes_df = get_nasdaq100_historical_changes()
        else:
            raise ValueError(f"Índice {index_name} no soportado")
        
        # Si no hay datos históricos, usar lista actual con advertencia
        if changes_df.empty:
            print(f"⚠️  No hay datos históricos para {index_name}, usando constituyentes actuales")
            result = {
                'tickers': current_data['tickers'],
                'data': current_data['data'],
                'historical_data_available': False,
                'note': 'Using current constituents - no historical data available'
            }
            _historical_cache[cache_key] = result
            return result, None
        
        # Crear conjunto de constituyentes históricos válidos
        # Estrategia: usar tickers que estuvieron en el índice durante el período de backtest
        
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()
        
        # Reconstruir constituyentes históricos
        historical_tickers = set(current_tickers)  # Empezar con actuales
        
        # Procesar cambios para el período de interés
        relevant_changes = changes_df[
            (changes_df['Date'] >= start_date) & 
            (changes_df['Date'] <= end_date)
        ].sort_values('Date', ascending=True)  # Cronológico para el período
        
        # También incluir cambios anteriores al start_date para tickers que estuvieron
        pre_period_changes = changes_df[changes_df['Date'] < start_date]
        
        # Tickers que fueron agregados/removidos en el período de interés
        added_in_period = set()
        removed_in_period = set()
        
        for _, change in relevant_changes.iterrows():
            if change['Action'] == 'Added':
                added_in_period.add(change['Ticker'])
            elif change['Action'] == 'Removed':
                removed_in_period.add(change['Ticker'])
        
        # Tickers que estuvieron presentes en algún momento del período
        # = actuales + agregados en período - removidos solo si no fueron re-agregados
        valid_tickers_for_period = set()
        
        # 1. Tickers actuales que no fueron removidos permanentemente
        for ticker in current_tickers:
            # Verificar si fue removido en el período y no re-agregado
            ticker_removals = relevant_changes[
                (relevant_changes['Ticker'] == ticker) & 
                (relevant_changes['Action'] == 'Removed')
            ]
            ticker_additions = relevant_changes[
                (relevant_changes['Ticker'] == ticker) & 
                (relevant_changes['Action'] == 'Added')
            ]
            
            # Si fue removido pero luego re-agregado, incluirlo
            # Si solo fue removido, excluirlo
            # Si no aparece en cambios, incluirlo (estaba presente todo el tiempo)
            if ticker_removals.empty or not ticker_additions.empty:
                valid_tickers_for_period.add(ticker)
        
        # 2. Agregar tickers que fueron agregados en el período
        valid_tickers_for_period.update(added_in_period)
        
        # 3. Agregar tickers que estuvieron presentes antes del período y no fueron removidos permanentemente
        for _, change in pre_period_changes.iterrows():
            ticker = change['Ticker']
            if change['Action'] == 'Added':
                # Verificar si fue removido posteriormente en nuestro período
                was_removed = not relevant_changes[
                    (relevant_changes['Ticker'] == ticker) & 
                    (relevant_changes['Action'] == 'Removed')
                ].empty
                
                was_readded = not relevant_changes[
                    (relevant_changes['Ticker'] == ticker) & 
                    (relevant_changes['Action'] == 'Added')
                ].empty
                
                # Si no fue removido, o fue removido pero re-agregado, incluirlo
                if not was_removed or was_readded:
                    valid_tickers_for_period.add(ticker)
        
        # Convertir a lista y limpiar
        final_tickers = sorted(list(valid_tickers_for_period))
        final_tickers = [t for t in final_tickers if t and len(t) <= 6 and not t.isdigit()]
        
        print(f"✅ {index_name} - Constituyentes históricos válidos para {start_date} a {end_date}: {len(final_tickers)}")
        print(f"   Cambios procesados: {len(relevant_changes)} en el período, {len(pre_period_changes)} anteriores")
        
        # Crear estructura de datos con información histórica
        historical_data = []
        for ticker in final_tickers:
            # Buscar fecha de incorporación más reciente
            ticker_additions = changes_df[
                (changes_df['Ticker'] == ticker) & 
                (changes_df['Action'] == 'Added')
            ].sort_values('Date', ascending=False)
            
            added_date = ticker_additions['Date'].iloc[0] if not ticker_additions.empty else None
            
            historical_data.append({
                'ticker': ticker,
                'added': added_date.strftime('%Y-%m-%d') if added_date else 'Unknown',
                'in_current': ticker in current_tickers,
                'status': 'Historical constituent'
            })
        
        result = {
            'tickers': final_tickers,
            'data': historical_data,
            'historical_data_available': True,
            'changes_processed': len(relevant_changes),
            'period_start': start_date.strftime('%Y-%m-%d'),
            'period_end': end_date.strftime('%Y-%m-%d'),
            'note': f'Historical constituents for {index_name} from {start_date} to {end_date}'
        }
        
        _historical_cache[cache_key] = result
        return result, None
        
    except Exception as e:
        error_msg = f"Error obteniendo constituyentes históricos para {index_name}: {e}"
        print(error_msg)
        
        # Fallback a constituyentes actuales
        try:
            current_data = get_current_constituents(index_name)
            fallback_result = {
                'tickers': current_data['tickers'],
                'data': current_data['data'],
                'historical_data_available': False,
                'note': f'Fallback to current constituents due to error: {str(e)}'
            }
            return fallback_result, f"Warning: {error_msg}"
        except:
            return None, error_msg

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
