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
        print("⚠️ No se pudo encontrar la tabla del NASDAQ-100")
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
        print(f"❌ Índice '{index_name}' no reconocido")
        return {'tickers': []}
    
    save_cache(cache_key, result, prefix="constituents")
    return result

def get_all_available_tickers_with_historical_validation(index_name, start_date, end_date):
    """
    VERSIÓN SIMPLIFICADA - Solo usa constituyentes actuales con datos disponibles
    """
    try:
        print(f"🔍 Obteniendo constituyentes de {index_name}")
        
        # 1. Obtener constituyentes actuales
        current_constituents = get_current_constituents(index_name)
        
        if not current_constituents or 'tickers' not in current_constituents or not current_constituents['tickers']:
            print("❌ No se pudieron obtener constituyentes")
            return {'tickers': []}, "No se pudieron obtener constituyentes"
        
        print(f"✅ Constituyentes obtenidos: {len(current_constituents['tickers'])}")
        
        # 2. Verificar qué tickers tienen datos CSV disponibles
        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        available_csv_tickers = set()
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            if filename.endswith('.csv') and not filename.endswith('_changes.csv'):
                ticker = filename.replace('.csv', '').upper()
                available_csv_tickers.add(ticker)
        
        print(f"📁 Archivos CSV disponibles: {len(available_csv_tickers)}")
        
        # 3. Intersección: tickers del índice que tienen datos disponibles
        final_tickers = list(set(current_constituents['tickers']) & available_csv_tickers)
        
        print(f"✅ Tickers finales con datos: {len(final_tickers)}")
        
        if len(final_tickers) == 0:
            print("❌ No hay tickers con datos disponibles")
            return {'tickers': []}, "No hay tickers con datos disponibles"
        
        return {
            'tickers': final_tickers,
            'data': [],
            'historical_data_available': False
        }, None
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {'tickers': []}, error_msg

def get_constituents_at_date(index_name, start_date, end_date):
    """Obtiene constituyentes - VERSIÓN SIMPLIFICADA"""
    try:
        # Intentar obtener con validación
        result, error = get_all_available_tickers_with_historical_validation(
            index_name, start_date, end_date
        )
        
        if result and 'tickers' in result and result['tickers']:
            print(f"✅ Constituyentes obtenidos: {len(result['tickers'])} tickers")
            return result, error
        else:
            print("❌ No se obtuvieron constituyentes")
            return None, error or "No se pudieron obtener constituyentes"
            
    except Exception as e:
        error_msg = f"Error obteniendo constituyentes: {e}"
        print(f"❌ {error_msg}")
        return None, error_msg

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
        print(f"Error creando enlace de descarga: {e}")
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

# Función de debug para verificar el estado del sistema
def debug_system_status():
    """Función de debug para diagnosticar problemas"""
    print("\n=== DEBUG: ESTADO DEL SISTEMA ===")
    
    # 1. Verificar directorio de datos
    print(f"📁 Directorio de datos: {DATA_DIR}")
    print(f"   Existe: {'✅' if os.path.exists(DATA_DIR) else '❌'}")
    
    # 2. Verificar archivos CSV de datos
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    ticker_files = [f for f in csv_files if not f.endswith('_changes.csv')]
    print(f"\n📊 Archivos de datos CSV encontrados: {len(ticker_files)}")
    
    if ticker_files:
        print(f"   Primeros 10: {[os.path.basename(f) for f in ticker_files[:10]]}")
        
        # Verificar un archivo al azar
        sample_file = ticker_files[0]
        try:
            sample_df = pd.read_csv(sample_file, nrows=5)
            print(f"   Muestra de {os.path.basename(sample_file)}:")
            print(f"   Columnas: {list(sample_df.columns)}")
            print(f"   Filas: {len(sample_df)}")
        except Exception as e:
            print(f"   ❌ Error leyendo archivo muestra: {e}")
    
    # 3. Verificar archivos de cambios históricos
    change_files = [
        'sp500_changes.csv',
        'ndx_changes.csv',
        'data/sp500_changes.csv',
        'data/ndx_changes.csv'
    ]
    
    print(f"\n📋 Archivos de cambios históricos:")
    for file_path in change_files:
        exists = os.path.exists(file_path)
        print(f"   {file_path}: {'✅' if exists else '❌'}")
        if exists:
            try:
                df = pd.read_csv(file_path)
                print(f"     - {len(df)} registros")
                print(f"     - Columnas: {list(df.columns)}")
            except Exception as e:
                print(f"     - ❌ Error: {e}")
    
    # 4. Probar conexión a Wikipedia
    print(f"\n🌐 Probando conexión a Wikipedia:")
    try:
        sp500_data = get_sp500_tickers_from_wikipedia()
        ndx_data = get_nasdaq100_tickers_from_wikipedia()
        print(f"   S&P 500: {len(sp500_data.get('tickers', []))} tickers")
        print(f"   NASDAQ-100: {len(ndx_data.get('tickers', []))} tickers")
    except Exception as e:
        print(f"   ❌ Error de conexión: {e}")
    
    # 5. Verificar caché
    print(f"\n💾 Estado de caché:")
    print(f"   Directorio: {CACHE_DIR}")
    print(f"   Existe: {'✅' if os.path.exists(CACHE_DIR) else '❌'}")
    try:
        cache_size = get_cache_size()
        print(f"   Tamaño: {cache_size:.2f} MB")
    except:
        print(f"   Tamaño: Error calculando")
    
    print("=== FIN DEBUG ===\n")

# Solo ejecutar si se llama directamente
if __name__ == "__main__":
    debug_system_status()
    
    print(f"💾 Tamaño actual de caché: {get_cache_size():.2f} MB")
    
    test_indices = ["SP500", "NDX", "Ambos (SP500 + NDX)"]
    for idx in test_indices:
        print(f"\n🧪 Probando índice: {idx}")
        result = get_current_constituents(idx)
        if result and 'tickers' in result:
            print(f"✅ {idx}: {len(result['tickers'])} tickers encontrados")
            if result['tickers']:
                print(f"   Primeros 5: {result['tickers'][:5]}")
        else:
            print(f"❌ {idx}: Error obteniendo tickers")
    
    # Test de validación completa
    print("\n🧪 Probando validación completa...")
    test_start = datetime(2023, 1, 1)
    test_end = datetime(2023, 12, 31)
    
    result, error = get_constituents_at_date("SP500", test_start, test_end)
    
    if result and 'tickers' in result:
        print(f"✅ Validación completa exitosa: {len(result['tickers'])} tickers")
        if result['tickers']:
            print(f"   Primeros 10: {result['tickers'][:10]}")
    else:
        print(f"❌ Error en validación completa: {error}")
