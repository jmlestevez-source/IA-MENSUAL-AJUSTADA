import pandas as pd
import requests
from io import StringIO
import os
import numpy as np
import re
from dateutil import parser
import glob
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import hashlib
import warnings
import base64
warnings.filterwarnings('ignore')

# Directorios
DATA_DIR = "data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache global mejorado
_historical_cache = {}
_constituents_cache = {}

def get_cache_key(*args):
    """Genera clave de caché única"""
    return hashlib.md5(str(args).encode()).hexdigest()

def save_cache(key, data, prefix="cache"):
    """Guarda datos en caché persistente"""
    try:
        cache_file = os.path.join(CACHE_DIR, f"{prefix}_{key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error guardando caché: {e}")

def load_cache(key, prefix="cache", max_age_days=7):
    """Carga datos de caché si no son muy antiguos"""
    try:
        cache_file = os.path.join(CACHE_DIR, f"{prefix}_{key}.pkl")
        if os.path.exists(cache_file):
            # Verificar antigüedad
            file_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days
            if file_age <= max_age_days:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
    except Exception as e:
        print(f"Error cargando caché: {e}")
    return None

def parse_wikipedia_date(date_str):
    """Parsea fechas de Wikipedia"""
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
                        return date(year, month, day)
            
            return None
        except:
            return None

def get_sp500_tickers_from_wikipedia():
    """Obtiene los tickers actuales del S&P 500"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].str.replace('.', '-').tolist()
        print(f"✅ Obtenidos {len(tickers)} tickers del S&P 500")
        return {'tickers': tickers}
    except Exception as e:
        print(f"❌ Error obteniendo S&P 500: {e}")
        return {'tickers': []}

def get_nasdaq100_tickers_from_wikipedia():
    """Obtiene los tickers actuales del NASDAQ-100"""
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        # La tabla puede estar en diferentes posiciones, intentar varias
        for i in [4, 3, 2, 1]:
            try:
                df = tables[i]
                if 'Ticker' in df.columns or 'Symbol' in df.columns:
                    col_name = 'Ticker' if 'Ticker' in df.columns else 'Symbol'
                    tickers = df[col_name].str.replace('.', '-').tolist()
                    print(f"✅ Obtenidos {len(tickers)} tickers del NASDAQ-100")
                    return {'tickers': tickers}
            except:
                continue
        # Si no encuentra la tabla correcta, usar una lista conocida
        print("⚠️ No se pudo encontrar la tabla del NASDAQ-100, usando lista de respaldo")
        return {'tickers': []}
    except Exception as e:
        print(f"❌ Error obteniendo NASDAQ-100: {e}")
        return {'tickers': []}

def get_sp500_historical_changes():
    """Obtiene los cambios históricos del S&P 500 desde CSV local"""
    local_csv_path = "sp500_changes.csv"
    fallback_path = "data/sp500_changes.csv"
    
    changes_df = pd.DataFrame()
    
    for csv_path in [local_csv_path, fallback_path]:
        if os.path.exists(csv_path):
            try:
                print(f"📂 Cargando cambios históricos S&P 500 desde {csv_path}...")
                changes_df = pd.read_csv(csv_path, parse_dates=['Date'])
                print(f"✅ Cargados {len(changes_df)} cambios históricos del S&P 500")
                break
            except Exception as e:
                print(f"⚠️ Error leyendo {csv_path}: {e}")
    
    if changes_df.empty:
        print("❌ No se pudieron obtener cambios históricos del S&P 500")
    else:
        # Asegurarse de que las fechas estén en formato datetime
        changes_df['Date'] = pd.to_datetime(changes_df['Date'])
        # Ordenar por fecha descendente
        changes_df = changes_df.sort_values('Date', ascending=False)
        
        if not changes_df.empty:
            date_range = f"{changes_df['Date'].min().date()} a {changes_df['Date'].max().date()}"
            print(f"📅 Rango de cambios S&P 500: {date_range}")
    
    return changes_df

def get_nasdaq100_historical_changes():
    """Obtiene los cambios históricos del NASDAQ-100 desde CSV local"""
    local_csv_path = "ndx_changes.csv"
    fallback_path = "data/ndx_changes.csv"
    
    changes_df = pd.DataFrame()
    
    for csv_path in [local_csv_path, fallback_path]:
        if os.path.exists(csv_path):
            try:
                print(f"📂 Cargando cambios históricos NASDAQ-100 desde {csv_path}...")
                changes_df = pd.read_csv(csv_path, parse_dates=['Date'])
                print(f"✅ Cargados {len(changes_df)} cambios históricos del NASDAQ-100")
                break
            except Exception as e:
                print(f"⚠️ Error leyendo {csv_path}: {e}")
    
    if changes_df.empty:
        print("❌ No se pudieron obtener cambios históricos del NASDAQ-100")
    else:
        # Asegurarse de que las fechas estén en formato datetime
        changes_df['Date'] = pd.to_datetime(changes_df['Date'])
        # Ordenar por fecha descendente
        changes_df = changes_df.sort_values('Date', ascending=False)
        
        if not changes_df.empty:
            date_range = f"{changes_df['Date'].min().date()} a {changes_df['Date'].max().date()}"
            print(f"📅 Rango de cambios NASDAQ-100: {date_range}")
    
    return changes_df

def get_current_constituents(index_name):
    """Obtiene constituyentes actuales con caché"""
    cache_key = f"current_{index_name}"
    
    cached_data = load_cache(cache_key, prefix="constituents", max_age_days=1)
    if cached_data is not None:
        return cached_data
    
    if index_name == "SP500":
        result = get_sp500_tickers_from_wikipedia()
    elif index_name == "NDX":
        result = get_nasdaq100_tickers_from_wikipedia()
    elif index_name == "Ambos (SP500 + NDX)":
        # Combinar ambos índices
        sp500 = get_sp500_tickers_from_wikipedia()
        ndx = get_nasdaq100_tickers_from_wikipedia()
        # Unir los tickers de ambos índices (sin duplicados)
        combined_tickers = list(set(sp500['tickers'] + ndx['tickers']))
        result = {'tickers': combined_tickers}
        print(f"✅ Combinados: {len(sp500['tickers'])} S&P500 + {len(ndx['tickers'])} NDX = {len(combined_tickers)} únicos")
    else:
        # En lugar de lanzar error, intentar interpretar
        print(f"⚠️ Índice '{index_name}' no reconocido, intentando interpretar...")
        if "SP500" in index_name or "S&P" in index_name:
            result = get_sp500_tickers_from_wikipedia()
        elif "NDX" in index_name or "NASDAQ" in index_name:
            result = get_nasdaq100_tickers_from_wikipedia()
        else:
            print(f"❌ No se pudo interpretar el índice '{index_name}'")
            return {'tickers': []}
    
    save_cache(cache_key, result, prefix="constituents")
    return result

def get_all_available_tickers_with_historical_validation(index_name, start_date, end_date):
    """
    Obtiene los tickers que realmente estaban en el índice durante el período especificado
    VERSIÓN SIMPLIFICADA Y ROBUSTA
    """
    try:
        print(f"🔍 Validando constituyentes históricos de {index_name} para {start_date} a {end_date}")
        
        # 1. Obtener constituyentes actuales y cambios históricos
        if index_name == "SP500":
            current_constituents = get_sp500_tickers_from_wikipedia()
            historical_changes = get_sp500_historical_changes()
        elif index_name == "NDX":
            current_constituents = get_nasdaq100_tickers_from_wikipedia()
            historical_changes = get_nasdaq100_historical_changes()
        elif index_name == "Ambos (SP500 + NDX)":
            # Combinar ambos índices
            sp500_current = get_sp500_tickers_from_wikipedia()
            ndx_current = get_nasdaq100_tickers_from_wikipedia()
            current_constituents = {
                'tickers': list(set(sp500_current['tickers'] + ndx_current['tickers']))
            }
            
            sp500_changes = get_sp500_historical_changes()
            ndx_changes = get_nasdaq100_historical_changes()
            
            # Combinar cambios históricos
            if not sp500_changes.empty and not ndx_changes.empty:
                historical_changes = pd.concat([sp500_changes, ndx_changes], ignore_index=True)
                historical_changes = historical_changes.drop_duplicates(subset=['Date', 'Ticker', 'Action'])
            elif not sp500_changes.empty:
                historical_changes = sp500_changes
            else:
                historical_changes = ndx_changes
        else:
            print(f"❌ Índice no soportado: {index_name}")
            return get_current_constituents(index_name), None
        
        # 2. Si no hay datos históricos, usar solo constituyentes actuales
        if historical_changes.empty:
            print("⚠️ No hay cambios históricos disponibles, usando solo constituyentes actuales")
            return {
                'tickers': current_constituents['tickers'],
                'data': [],
                'historical_data_available': False,
                'note': 'No historical data available'
            }, None
        
        # 3. Convertir fechas a datetime para comparación
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        elif isinstance(start_date, date):
            start_date = pd.to_datetime(start_date)
            
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        elif isinstance(end_date, date):
            end_date = pd.to_datetime(end_date)
        
        print(f"📅 Período de validación: {start_date.date()} a {end_date.date()}")
        
        # 4. LÓGICA SIMPLIFICADA: Solo filtrar los que fueron removidos ANTES del start_date
        valid_tickers = set(current_constituents['tickers'])
        
        # Agregar tickers que fueron removidos DESPUÉS del start_date (estaban presentes durante el período)
        for _, change in historical_changes.iterrows():
            change_date = pd.to_datetime(change['Date'])
            ticker = str(change['Ticker']).upper()
            action = change['Action']
            
            if action == 'Removed' and change_date > start_date:
                # Si fue removido después del start_date, SÍ estaba presente durante el backtest
                valid_tickers.add(ticker)
            elif action == 'Removed' and change_date <= start_date:
                # Si fue removido antes o en el start_date, NO estaba presente
                valid_tickers.discard(ticker)
        
        print(f"📊 Tickers válidos históricamente: {len(valid_tickers)}")
        
        # 5. Filtrar solo los tickers que tienen datos CSV disponibles
        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        available_csv_tickers = set()
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            if filename.endswith('.csv'):
                ticker = filename.replace('.csv', '').upper()
                available_csv_tickers.add(ticker)
        
        print(f"📁 Tickers con datos CSV disponibles: {len(available_csv_tickers)}")
        
        # Intersección: tickers válidos históricamente Y con datos disponibles
        final_tickers = list(valid_tickers & available_csv_tickers)
        
        # 6. Verificar específicamente los tickers problemáticos
        problematic_tickers = ['RIG', 'OI', 'VNT']
        removed_problematic = []
        
        for ticker in problematic_tickers:
            if ticker in final_tickers:
                # Verificar cuándo fue removido
                removal_info = historical_changes[
                    (historical_changes['Ticker'] == ticker) & 
                    (historical_changes['Action'] == 'Removed')
                ]
                if not removal_info.empty:
                    removal_date = pd.to_datetime(removal_info.iloc[0]['Date'])
                    if removal_date <= start_date:
                        print(f"⚠️ REMOVIENDO {ticker} - fue eliminado del índice el {removal_date.date()}")
                        final_tickers.remove(ticker)
                        removed_problematic.append(ticker)
        
        print(f"📈 Tickers válidos finales: {len(final_tickers)}")
        
        if removed_problematic:
            print(f"🚫 Tickers problemáticos removidos: {removed_problematic}")
        
        return {
            'tickers': final_tickers,
            'data': [],
            'historical_data_available': True,
            'removed_problematic': removed_problematic
        }, None
        
    except Exception as e:
        error_msg = f"Error en validación histórica: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Fallback a constituyentes actuales
        try:
            print("🔄 Intentando fallback a constituyentes actuales...")
            current = get_current_constituents(index_name)
            if current and 'tickers' in current and current['tickers']:
                print(f"✅ Fallback exitoso con {len(current['tickers'])} tickers")
                return {
                    'tickers': current['tickers'],
                    'data': [],
                    'historical_data_available': False,
                    'note': f'Fallback due to error: {str(e)}'
                }, f"Warning: Using current constituents as fallback - {str(e)}"
            else:
                print("❌ Fallback también falló")
                return None, error_msg
        except Exception as fallback_error:
            print(f"❌ Error en fallback: {fallback_error}")
            return None, f"{error_msg} | Fallback error: {str(fallback_error)}"

def get_constituents_at_date(index_name, start_date, end_date):
    """Obtiene constituyentes con caché mejorado"""
    cache_key = get_cache_key(index_name, start_date, end_date)
    
    # Caché en memoria
    if cache_key in _constituents_cache:
        print(f"⚡ Usando caché en memoria para {index_name}")
        return _constituents_cache[cache_key], None
    
    # Caché en disco
    cached_data = load_cache(cache_key, prefix="constituents", max_age_days=7)
    if cached_data is not None:
        print(f"📦 Usando caché en disco para {index_name}")
        _constituents_cache[cache_key] = cached_data
        return cached_data, None
    
    try:
        # Obtener datos frescos con validación histórica
        result, error = get_all_available_tickers_with_historical_validation(
            index_name, start_date, end_date
        )
        
        if result and 'tickers' in result and result['tickers']:
            # Guardar en ambos cachés
            _constituents_cache[cache_key] = result
            save_cache(cache_key, result, prefix="constituents")
            print(f"✅ Constituyentes obtenidos: {len(result['tickers'])} tickers")
            return result, error
        else:
            # Fallback
            print("⚠️ No se obtuvieron resultados, intentando fallback...")
            current_data = get_current_constituents(index_name)
            
            if current_data and 'tickers' in current_data and current_data['tickers']:
                fallback_result = {
                    'tickers': current_data['tickers'],
                    'data': [],
                    'historical_data_available': False,
                    'note': 'Fallback to current constituents'
                }
                print(f"✅ Usando fallback con {len(current_data['tickers'])} tickers actuales")
                return fallback_result, "Warning: Using current constituents as fallback"
            else:
                print("❌ Fallback también falló")
                return None, "Error: No se pudieron obtener constituyentes"
            
    except Exception as e:
        error_msg = f"Error obteniendo constituyentes para {index_name}: {e}"
        print(f"⚠️ {error_msg}")
        
        # Fallback
        try:
            print("🔄 Intentando fallback a constituyentes actuales...")
            current_data = get_current_constituents(index_name)
            
            if current_data and 'tickers' in current_data and current_data['tickers']:
                fallback_result = {
                    'tickers': current_data['tickers'],
                    'data': [],
                    'historical_data_available': False,
                    'note': f'Fallback to current constituents due to: {str(e)}'
                }
                print(f"✅ Fallback exitoso con {len(current_data['tickers'])} tickers")
                return fallback_result, f"Warning: Using current constituents as fallback - {str(e)}"
            else:
                print("❌ Fallback también falló")
                return None, error_msg
        except Exception as fallback_error:
            print(f"❌ Error en fallback: {fallback_error}")
            return None, f"{error_msg} | Fallback error: {str(fallback_error)}"

def load_prices_from_csv_parallel(tickers, start_date, end_date, load_full_data=True):
    """Carga precios desde CSV en PARALELO"""
    prices_data = {}
    ohlc_data = {}
    
    def load_single_ticker(ticker):
        csv_path = f"data/{ticker}.csv"
        if not os.path.exists(csv_path):
            return ticker, None, None
        
        try:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            start_filter = start_date.date() if isinstance(start_date, datetime) else start_date
            end_filter = end_date.date() if isinstance(end_date, datetime) else end_date
            
            df = df[(df.index.date >= start_filter) & (df.index.date <= end_filter)]
            
            if df.empty:
                return ticker, None, 
