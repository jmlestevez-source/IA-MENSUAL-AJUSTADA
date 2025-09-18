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
import streamlit as st
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
    """Obtiene constituyentes actuales con caché - CORREGIDA para manejar 'Ambos'"""
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
    VERSIÓN CORREGIDA que valida correctamente los cambios históricos
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
        
        # 4. Obtener todos los tickers que alguna vez estuvieron en el índice
        all_tickers_ever = set(current_constituents['tickers'])
        removed_tickers = historical_changes[historical_changes['Action'] == 'Removed']['Ticker'].unique()
        all_tickers_ever.update(removed_tickers)
        
        print(f"📊 Total tickers históricos encontrados: {len(all_tickers_ever)}")
        
        # 5. Validar para cada mes del backtest
        valid_tickers_for_period = set()
        
        # Para cada mes en el período del backtest
        current_date = start_date
        months_checked = 0
        
        while current_date <= end_date:
            # Obtener constituyentes válidos para esta fecha específica
            constituents_at_date = set(current_constituents['tickers'].copy())
            
            # Procesar cambios históricos
            for _, change in historical_changes.iterrows():
                change_date = pd.to_datetime(change['Date'])
                ticker = str(change['Ticker']).upper()
                action = change['Action']
                
                # Si el cambio ocurrió DESPUÉS de current_date, revertirlo
                if change_date > current_date:
                    if action == 'Added':
                        # Si fue añadido después, no estaba en current_date
                        constituents_at_date.discard(ticker)
                    elif action == 'Removed':
                        # Si fue removido después, SÍ estaba en current_date
                        constituents_at_date.add(ticker)
            
            # Agregar estos constituyentes al conjunto válido
            valid_tickers_for_period.update(constituents_at_date)
            
            # Avanzar al siguiente mes
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1, day=1)
            
            months_checked += 1
            if months_checked % 12 == 0:
                print(f"  Validados {months_checked} meses...")
        
        print(f"✅ Validación completa: {months_checked} meses verificados")
        print(f"📊 Tickers válidos históricamente: {len(valid_tickers_for_period)}")
        
        # 6. Filtrar solo los tickers que tienen datos CSV disponibles
        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        available_csv_tickers = set()
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            if filename.endswith('.csv'):
                ticker = filename.replace('.csv', '').upper()
                available_csv_tickers.add(ticker)
        
        print(f"📁 Tickers con datos CSV disponibles: {len(available_csv_tickers)}")
        
        # Intersección: tickers válidos históricamente Y con datos disponibles
        final_tickers = list(valid_tickers_for_period & available_csv_tickers)
        
        # 7. Verificar específicamente los tickers problemáticos
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
                    if removal_date < start_date:
                        print(f"⚠️ REMOVIENDO {ticker} - fue eliminado del índice el {removal_date.date()}")
                        final_tickers.remove(ticker)
                        removed_problematic.append(ticker)
        
        print(f"📈 Tickers válidos finales: {len(final_tickers)}")
        
        if removed_problematic:
            print(f"🚫 Tickers problemáticos removidos: {removed_problematic}")
        
        # 8. Crear información detallada para debugging
        detailed_data = []
        for ticker in final_tickers[:10]:  # Solo los primeros 10 para no sobrecargar
            ticker_info = {
                'ticker': ticker,
                'in_current': ticker in current_constituents['tickers'],
                'has_csv': ticker in available_csv_tickers
            }
            
            # Buscar información de cambios
            ticker_changes = historical_changes[historical_changes['Ticker'] == ticker]
            if not ticker_changes.empty:
                last_change = ticker_changes.iloc[0]
                ticker_info['last_action'] = last_change['Action']
                ticker_info['last_change_date'] = str(last_change['Date'])
            
            detailed_data.append(ticker_info)
        
        return {
            'tickers': final_tickers,
            'data': detailed_data,
            'historical_data_available': True,
            'total_months_validated': months_checked,
            'removed_problematic': removed_problematic
        }, None
        
    except Exception as e:
        error_msg = f"Error en validación histórica: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Fallback mejorado a constituyentes actuales
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
    """Obtiene constituyentes con caché mejorado y mejor manejo de errores"""
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
            # Fallback mejorado
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
        
        # Mejor manejo del fallback
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

@st.cache_data(ttl=3600*24*7)
def load_prices_from_csv_parallel(tickers, start_date, end_date, load_full_data=True):
    """Carga precios desde CSV en PARALELO - NOMBRE CORREGIDO"""
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
                return ticker, None, None
            
            # Usar Close ya que los datos están ajustados
            if 'Close' in df.columns:
                price = df['Close']
            else:
                return ticker, None, None
            
            ohlc = None
            if load_full_data and all(col in df.columns for col in ['High', 'Low', 'Close']):
                ohlc = {
                    'High': df['High'],
                    'Low': df['Low'],
                    'Close': df['Close'],
                    'Volume': df['Volume'] if 'Volume' in df.columns else None
                }
            
            return ticker, price, ohlc
            
        except Exception as e:
            return ticker, None, None
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(load_single_ticker, ticker) for ticker in tickers]
        
        for future in futures:
            ticker, price, ohlc = future.result()
            if price is not None:
                prices_data[ticker] = price
            if ohlc is not None:
                ohlc_data[ticker] = ohlc
    
    if prices_data:
        prices_df = pd.DataFrame(prices_data)
        prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
        return prices_df, ohlc_data
    else:
        return pd.DataFrame(), {}

def create_download_link(df, filename, link_text):
    """Crea un enlace de descarga para un DataFrame"""
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
        return href
    except Exception as e:
        st.error(f"Error creando enlace de descarga: {e}")
        return None

def generate_removed_tickers_summary():
    """Genera resumen de tickers removidos con caché"""
    cache_key = "removed_tickers_summary"
    
    cached_data = load_cache(cache_key, max_age_days=30)
    if cached_data is not None:
        return cached_data
    
    summary_data = []
    
    # Procesar S&P 500
    sp500_changes = get_sp500_historical_changes()
    if not sp500_changes.empty:
        removed = sp500_changes[sp500_changes['Action'] == 'Removed']
        for _, row in removed.iterrows():
            summary_data.append({
                'Index': 'S&P 500',
                'Ticker': row['Ticker'],
                'Date': row['Date'],
                'Action': 'Removed'
            })
    
    # Procesar NASDAQ-100
    ndx_changes = get_nasdaq100_historical_changes()
    if not ndx_changes.empty:
        removed = ndx_changes[ndx_changes['Action'] == 'Removed']
        for _, row in removed.iterrows():
            summary_data.append({
                'Index': 'NASDAQ-100',
                'Ticker': row['Ticker'],
                'Date': row['Date'],
                'Action': 'Removed'
            })
    
    if summary_data:
        summary = pd.DataFrame(summary_data)
        summary['Date'] = pd.to_datetime(summary['Date'])
        summary = summary.sort_values('Date', ascending=False)
        save_cache(cache_key, summary)
        return summary
    
    return pd.DataFrame()

def clear_cache():
    """Limpia toda la caché"""
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR)
    print("✅ Caché limpiado")

def get_cache_size():
    """Obtiene el tamaño de la caché en MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(CACHE_DIR):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convertir a MB

# TEST: Verificar que funciona para todos los índices
if __name__ == "__main__":
    print(f"💾 Tamaño actual de caché: {get_cache_size():.2f} MB")
    
    test_indices = ["SP500", "NDX", "Ambos (SP500 + NDX)"]
    for idx in test_indices:
        print(f"\n🧪 Probando índice: {idx}")
        result = get_current_constituents(idx)
        if result and 'tickers' in result:
            print(f"✅ {idx}: {len(result['tickers'])} tickers encontrados")
        else:
            print(f"❌ {idx}: Error obteniendo tickers")
    
    # Test de validación histórica
    print("\n🧪 Probando validación histórica...")
    test_date = datetime(2023, 8, 1)
    result, error = get_all_available_tickers_with_historical_validation(
        "SP500", test_date, test_date + timedelta(days=30)
    )
    
    if result:
        print(f"✅ Validación histórica: {len(result['tickers'])} tickers")
        if 'removed_problematic' in result:
            print(f"🚫 Tickers problemáticos removidos: {result['removed_problematic']}")
        
        # Verificar que RIG, OI, VNT no estén en los resultados
        problematic = ['RIG', 'OI', 'VNT']
        for ticker in problematic:
            if ticker in result['tickers']:
                print(f"❌ ERROR: {ticker} no debería estar en los resultados de 2023!")
            else:
                print(f"✅ Correcto: {ticker} no está en los resultados")

print(f"💾 Tamaño actual de caché: {get_cache_size():.2f} MB")
