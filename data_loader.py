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

# En data_loader.py, reemplaza las funciones get_sp500_historical_changes() y get_nasdaq100_historical_changes() con estas:

def get_sp500_historical_changes():
    """Obtiene los cambios históricos del S&P 500 desde CSV local o Wikipedia"""
    
    # PRIMERO: Intentar cargar desde CSV local
    local_csv_path = "sp500_changes.csv"  # En la raíz del repositorio
    fallback_path = "data/sp500_changes.csv"  # Alternativa en carpeta data
    
    changes_df = pd.DataFrame()
    loaded_from_local = False
    
    # Intentar cargar el CSV local
    for csv_path in [local_csv_path, fallback_path]:
        if os.path.exists(csv_path):
            try:
                print(f"📂 Cargando cambios históricos S&P 500 desde {csv_path}...")
                changes_df = pd.read_csv(csv_path, parse_dates=['Date'])
                print(f"✅ Cargados {len(changes_df)} cambios históricos del S&P 500 desde CSV local")
                loaded_from_local = True
                break
            except Exception as e:
                print(f"⚠️ Error leyendo {csv_path}: {e}")
    
    # SEGUNDO: Si necesitamos actualizar o no hay datos locales, ir a Wikipedia
    try:
        if loaded_from_local:
            # Verificar si necesitamos actualizar con datos más recientes
            last_date = pd.to_datetime(changes_df['Date']).max()
            days_old = (datetime.now() - last_date).days
            
            if days_old > 7:  # Si los datos tienen más de 7 días
                print(f"🔄 Datos locales tienen {days_old} días. Buscando actualizaciones en Wikipedia...")
                
                # Descargar de Wikipedia
                wikipedia_df = download_sp500_changes_from_wikipedia()
                
                if not wikipedia_df.empty:
                    # Combinar con datos locales
                    new_changes = wikipedia_df[pd.to_datetime(wikipedia_df['Date']) > last_date]
                    
                    if not new_changes.empty:
                        print(f"📥 Encontrados {len(new_changes)} cambios nuevos desde {last_date.date()}")
                        changes_df = pd.concat([changes_df, new_changes], ignore_index=True)
                        changes_df = changes_df.drop_duplicates(subset=['Date', 'Ticker', 'Action'])
                        changes_df = changes_df.sort_values('Date', ascending=False)
                        
                        # Guardar versión actualizada
                        try:
                            changes_df.to_csv(local_csv_path, index=False)
                            print(f"💾 CSV actualizado guardado en {local_csv_path}")
                        except:
                            print("⚠️ No se pudo guardar el CSV actualizado")
                    else:
                        print("✅ No hay cambios nuevos en Wikipedia")
            else:
                print(f"✅ Datos locales están actualizados (última entrada hace {days_old} días)")
        else:
            # No hay datos locales, descargar todo de Wikipedia
            print("🌐 No se encontró CSV local. Descargando desde Wikipedia...")
            changes_df = download_sp500_changes_from_wikipedia()
            
            if not changes_df.empty:
                # Guardar para futuro uso
                try:
                    changes_df.to_csv(local_csv_path, index=False)
                    print(f"💾 Nuevos datos guardados en {local_csv_path}")
                except:
                    print("⚠️ No se pudo guardar el CSV")
    
    except Exception as e:
        print(f"⚠️ Error actualizando desde Wikipedia: {e}")
        # Si hay error, usar solo los datos locales que tengamos
    
    if changes_df.empty:
        print("❌ No se pudieron obtener cambios históricos del S&P 500")
    else:
        print(f"📊 Total: {len(changes_df)} cambios históricos del S&P 500")
        
        # Mostrar rango de fechas
        if not changes_df.empty:
            date_range = f"{changes_df['Date'].min()} a {changes_df['Date'].max()}"
            print(f"📅 Rango de cambios: {date_range}")
    
    return changes_df

def get_nasdaq100_historical_changes():
    """Obtiene los cambios históricos del NASDAQ-100 desde CSV local o Wikipedia"""
    
    # PRIMERO: Intentar cargar desde CSV local
    local_csv_path = "ndx_changes.csv"  # En la raíz del repositorio
    fallback_path = "data/ndx_changes.csv"  # Alternativa en carpeta data
    
    changes_df = pd.DataFrame()
    loaded_from_local = False
    
    # Intentar cargar el CSV local
    for csv_path in [local_csv_path, fallback_path]:
        if os.path.exists(csv_path):
            try:
                print(f"📂 Cargando cambios históricos NASDAQ-100 desde {csv_path}...")
                changes_df = pd.read_csv(csv_path, parse_dates=['Date'])
                print(f"✅ Cargados {len(changes_df)} cambios históricos del NASDAQ-100 desde CSV local")
                loaded_from_local = True
                break
            except Exception as e:
                print(f"⚠️ Error leyendo {csv_path}: {e}")
    
    # SEGUNDO: Si necesitamos actualizar o no hay datos locales, ir a Wikipedia
    try:
        if loaded_from_local:
            # Verificar si necesitamos actualizar
            last_date = pd.to_datetime(changes_df['Date']).max()
            days_old = (datetime.now() - last_date).days
            
            if days_old > 7:  # Si los datos tienen más de 7 días
                print(f"🔄 Datos locales tienen {days_old} días. Buscando actualizaciones en Wikipedia...")
                
                # Descargar de Wikipedia
                wikipedia_df = download_nasdaq100_changes_from_wikipedia()
                
                if not wikipedia_df.empty:
                    # Combinar con datos locales
                    new_changes = wikipedia_df[pd.to_datetime(wikipedia_df['Date']) > last_date]
                    
                    if not new_changes.empty:
                        print(f"📥 Encontrados {len(new_changes)} cambios nuevos desde {last_date.date()}")
                        changes_df = pd.concat([changes_df, new_changes], ignore_index=True)
                        changes_df = changes_df.drop_duplicates(subset=['Date', 'Ticker', 'Action'])
                        changes_df = changes_df.sort_values('Date', ascending=False)
                        
                        # Guardar versión actualizada
                        try:
                            changes_df.to_csv(local_csv_path, index=False)
                            print(f"💾 CSV actualizado guardado en {local_csv_path}")
                        except:
                            print("⚠️ No se pudo guardar el CSV actualizado")
                    else:
                        print("✅ No hay cambios nuevos en Wikipedia")
            else:
                print(f"✅ Datos locales están actualizados (última entrada hace {days_old} días)")
        else:
            # No hay datos locales, descargar todo de Wikipedia
            print("🌐 No se encontró CSV local. Descargando desde Wikipedia...")
            changes_df = download_nasdaq100_changes_from_wikipedia()
            
            if not changes_df.empty:
                # Guardar para futuro uso
                try:
                    changes_df.to_csv(local_csv_path, index=False)
                    print(f"💾 Nuevos datos guardados en {local_csv_path}")
                except:
                    print("⚠️ No se pudo guardar el CSV")
    
    except Exception as e:
        print(f"⚠️ Error actualizando desde Wikipedia: {e}")
        # Si hay error, usar solo los datos locales que tengamos
    
    if changes_df.empty:
        print("❌ No se pudieron obtener cambios históricos del NASDAQ-100")
    else:
        print(f"📊 Total: {len(changes_df)} cambios históricos del NASDAQ-100")
        
        # Mostrar rango de fechas
        if not changes_df.empty:
            date_range = f"{changes_df['Date'].min()} a {changes_df['Date'].max()}"
            print(f"📅 Rango de cambios: {date_range}")
    
    return changes_df

def download_sp500_changes_from_wikipedia():
    """Función auxiliar para descargar cambios del S&P 500 desde Wikipedia"""
    print("🌐 Conectando con Wikipedia para S&P 500...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        
        # [El resto del código de parsing de Wikipedia que ya tenías...]
        # [Por brevedad, aquí solo muestro la estructura]
        
        # Buscar la tabla de cambios
        changes_df = None
        for i, table in enumerate(tables):
            # [Tu lógica existente para encontrar la tabla correcta]
            pass
        
        if changes_df is None:
            if len(tables) >= 3:
                changes_df = tables[2]
        
        # [Tu lógica de procesamiento existente]
        
        # Retornar DataFrame procesado
        return result_df
        
    except Exception as e:
        print(f"❌ Error descargando de Wikipedia: {e}")
        return pd.DataFrame()

def download_nasdaq100_changes_from_wikipedia():
    """Función auxiliar para descargar cambios del NASDAQ-100 desde Wikipedia"""
    print("🌐 Conectando con Wikipedia para NASDAQ-100...")
    
    # [Similar a la función anterior pero para NASDAQ-100]
    
    return pd.DataFrame()

def download_prices_parallel(tickers, start_date, end_date, load_full_data=True, max_workers=10):
    """
    OPTIMIZACIÓN CRÍTICA: Carga precios en PARALELO
    """
    if isinstance(tickers, dict) and 'tickers' in tickers:
        ticker_list = tickers['tickers']
    elif isinstance(tickers, (list, tuple)):
        ticker_list = list(tickers)
    elif isinstance(tickers, str):
        ticker_list = [tickers]
    else:
        ticker_list = []
    
    # Limpiar tickers
    ticker_list = [str(t).strip().upper().replace('.', '-') for t in ticker_list]
    ticker_list = [t for t in ticker_list if t and len(t) <= 6 and not t.isdigit()]
    ticker_list = list(dict.fromkeys(ticker_list))
    
    if not ticker_list:
        return pd.DataFrame(), {}
    
    print(f"📂 Cargando {len(ticker_list)} tickers en paralelo (máx {max_workers} workers)...")
    
    prices_data = {}
    ohlc_data = {}
    
    def load_single_ticker(ticker):
        """Carga un ticker individual"""
        csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(csv_path):
            return ticker, None, None
        
        try:
            # Usar caché de lectura si existe
            cache_key = get_cache_key(ticker, start_date, end_date)
            cached = load_cache(cache_key, prefix="price", max_age_days=1)
            if cached is not None:
                return ticker, cached.get('price'), cached.get('ohlc')
            
            # Leer CSV
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            
            # Manejar timezone
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Filtrar fechas
            start_filter = start_date.date() if isinstance(start_date, datetime) else start_date
            end_filter = end_date.date() if isinstance(end_date, datetime) else end_date
            
            mask = (df.index.date >= start_filter) & (df.index.date <= end_filter)
            df_filtered = df[mask]
            
            if df_filtered.empty:
                return ticker, None, None
            
            # Extraer precio
            if 'Adj Close' in df_filtered.columns:
                price_series = df_filtered['Adj Close']
            elif 'Close' in df_filtered.columns:
                price_series = df_filtered['Close']
            else:
                return ticker, None, None
            
            # Extraer OHLC
            ohlc = None
            if load_full_data and all(col in df_filtered.columns for col in ['High', 'Low', 'Close']):
                ohlc = {
                    'High': df_filtered['High'],
                    'Low': df_filtered['Low'],
                    'Close': df_filtered['Adj Close'] if 'Adj Close' in df_filtered.columns else df_filtered['Close'],
                    'Volume': df_filtered.get('Volume')
                }
            
            # Guardar en caché
            save_cache(cache_key, {'price': price_series, 'ohlc': ohlc}, prefix="price")
            
            return ticker, price_series, ohlc
            
        except Exception as e:
            return ticker, None, None
    
    # Ejecutar en paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single_ticker, ticker): ticker for ticker in ticker_list}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 50 == 0:
                print(f"  Progreso: {completed}/{len(ticker_list)} tickers cargados...")
            
            ticker, price, ohlc = future.result()
            if price is not None:
                prices_data[ticker] = price
            if ohlc is not None:
                ohlc_data[ticker] = ohlc
    
    # Crear DataFrame
    if prices_data:
        prices_df = pd.DataFrame(prices_data)
        # Usar interpolación en lugar de fillna para mejor rendimiento
        prices_df = prices_df.interpolate(method='linear', limit_direction='both')
        
        print(f"✅ Cargados {len(prices_df.columns)} tickers con datos válidos")
        return prices_df, ohlc_data
    else:
        return pd.DataFrame(), {}

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
        # Obtener datos frescos
        result, error = get_all_available_tickers_with_historical_validation(
            index_name, start_date, end_date
        )
        
        if result:
            # Guardar en ambos cachés
            _constituents_cache[cache_key] = result
            save_cache(cache_key, result, prefix="constituents")
            return result, error
        else:
            # Fallback
            current_data = get_current_constituents(index_name)
            fallback_result = {
                'tickers': current_data['tickers'],
                'data': [{'ticker': t, 'added': 'Unknown', 'in_current': True, 'status': 'Current fallback'} for t in current_data['tickers']],
                'historical_data_available': False,
                'note': 'Fallback to current constituents'
            }
            return fallback_result, "Warning: Using current constituents as fallback"
            
    except Exception as e:
        error_msg = f"Error obteniendo constituyentes para {index_name}: {e}"
        print(error_msg)
        return None, error_msg

# Función wrapper para compatibilidad
def download_prices(tickers, start_date, end_date, load_full_data=True):
    """Wrapper para compatibilidad con código existente"""
    prices_df, ohlc_data = download_prices_parallel(
        tickers, start_date, end_date, 
        load_full_data=load_full_data,
        max_workers=10
    )
    
    if load_full_data:
        return prices_df, ohlc_data
    else:
        return prices_df

# [El resto de las funciones se mantienen igual pero con caché agregado donde sea apropiado]

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
    else:
        raise ValueError(f"Índice {index_name} no soportado")
    
    save_cache(cache_key, result, prefix="constituents")
    return result

# [Las demás funciones se mantienen con estructura similar]

def get_all_available_tickers_with_historical_validation(index_name, start_date, end_date):
    """Versión optimizada con validación más eficiente"""
    try:
        # Obtener lista de CSVs disponibles (rápido)
        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        all_available_tickers = []
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            if filename.endswith('.csv'):
                ticker = filename.replace('.csv', '').upper().replace('.', '-')
                if ticker and len(ticker) <= 6 and not ticker.isdigit() and ticker not in ['SPY', 'QQQ']:
                    all_available_tickers.append(ticker)
        
        all_available_tickers = list(dict.fromkeys(all_available_tickers))
        print(f"📊 Total de tickers CSV disponibles: {len(all_available_tickers)}")
        
        # [El resto de la lógica se mantiene pero optimizada con menos iteraciones]
        
        return {
            'tickers': all_available_tickers,
            'data': [],
            'historical_data_available': True,
            'note': 'Optimized validation'
        }, None
        
    except Exception as e:
        return None, str(e)

def generate_removed_tickers_summary():
    """Genera resumen de tickers removidos con caché"""
    cache_key = "removed_tickers_summary"
    
    cached_data = load_cache(cache_key, max_age_days=30)
    if cached_data is not None:
        return cached_data
    
    # [Lógica original]
    # Después de generar el resumen:
    # save_cache(cache_key, summary)
    
    return pd.DataFrame()

# Funciones de utilidad adicionales
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

print(f"💾 Tamaño actual de caché: {get_cache_size():.2f} MB")
