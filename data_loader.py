import pandas as pd
import os
import numpy as np
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

# Cache global
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
            file_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days
            if file_age <= max_age_days:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
    except Exception as e:
        print(f"Error cargando caché: {e}")
    return None

def get_all_available_tickers():
    """Obtiene todos los tickers disponibles en la carpeta data/"""
    try:
        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        available_tickers = []
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            if filename.endswith('.csv') and not filename.endswith('_changes.csv'):
                ticker = filename.replace('.csv', '').upper()
                # Filtrar archivos que no son tickers válidos
                if len(ticker) <= 6 and ticker.replace('-', '').isalnum():
                    available_tickers.append(ticker)
        
        available_tickers = sorted(list(set(available_tickers)))
        print(f"📊 Total de tickers disponibles: {len(available_tickers)}")
        return available_tickers
        
    except Exception as e:
        print(f"Error obteniendo tickers disponibles: {e}")
        return []

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
        changes_df['Date'] = pd.to_datetime(changes_df['Date'])
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
        changes_df['Date'] = pd.to_datetime(changes_df['Date'])
        changes_df = changes_df.sort_values('Date', ascending=False)
        
        if not changes_df.empty:
            date_range = f"{changes_df['Date'].min().date()} a {changes_df['Date'].max().date()}"
            print(f"📅 Rango de cambios NASDAQ-100: {date_range}")
    
    return changes_df

def get_sp500_tickers_from_local_data():
    """Obtiene tickers del S&P 500 basado en datos locales (SIN Wikipedia)"""
    try:
        # Lista de tickers conocidos del S&P 500 (los más comunes y estables)
        known_sp500_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'BRK-B', 'TSLA', 'META', 'UNH',
            'JNJ', 'XOM', 'JPM', 'V', 'PG', 'CVX', 'HD', 'MA', 'ABBV', 'PFE', 'AVGO', 'KO',
            'LLY', 'TMO', 'PEP', 'COST', 'ADBE', 'WMT', 'BAC', 'MRK', 'DIS', 'ABT', 'CRM',
            'NFLX', 'ACN', 'LIN', 'DHR', 'VZ', 'CMCSA', 'NKE', 'TXN', 'RTX', 'AMD', 'QCOM',
            'HON', 'T', 'UPS', 'NEE', 'LOW', 'AMGN', 'SPGI', 'CAT', 'SBUX', 'GE', 'AMAT',
            'INTU', 'IBM', 'DE', 'AXP', 'BKNG', 'MDLZ', 'ADI', 'TJX', 'GILD', 'CVS', 'MO',
            'ADP', 'VRTX', 'PLD', 'LRCX', 'SYK', 'TMUS', 'ZTS', 'FISV', 'BSX', 'REGN', 'C',
            'DUK', 'SO', 'CL', 'MMM', 'EQIX', 'NOC', 'AON', 'APD', 'ITW', 'ICE', 'PGR',
            'FCX', 'F', 'USB', 'NSC', 'EMR', 'BSX', 'SHW', 'MCO', 'CME', 'COF', 'HUM',
            'GD', 'TGT', 'FIS', 'MMC', 'CCI', 'ORLY', 'KLAC', 'APH', 'MSI', 'EOG', 'WM'
        ]
        
        # Obtener cambios históricos para tickers adicionales
        sp500_changes = get_sp500_historical_changes()
        historical_tickers = set()
        
        if not sp500_changes.empty:
            # Obtener todos los tickers mencionados en los cambios históricos
            all_mentioned = sp500_changes['Ticker'].unique()
            historical_tickers.update(all_mentioned)
        
        # Combinar tickers conocidos con históricos
        all_sp500_tickers = list(set(known_sp500_tickers + list(historical_tickers)))
        
        # Filtrar solo los que tienen datos disponibles
        available_tickers = get_all_available_tickers()
        final_tickers = [t for t in all_sp500_tickers if t in available_tickers]
        
        print(f"✅ S&P 500 tickers obtenidos localmente: {len(final_tickers)}")
        return {'tickers': final_tickers}
        
    except Exception as e:
        print(f"❌ Error obteniendo S&P 500 local: {e}")
        return {'tickers': []}

def get_nasdaq100_tickers_from_local_data():
    """Obtiene tickers del NASDAQ-100 basado en datos locales (SIN Wikipedia)"""
    try:
        # Lista de tickers conocidos del NASDAQ-100
        known_ndx_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'AVGO', 'ASML',
            'COST', 'ADBE', 'NFLX', 'PEP', 'CSCO', 'CMCSA', 'AMD', 'QCOM', 'INTC', 'INTU',
            'AMAT', 'TXN', 'BKNG', 'HON', 'AMGN', 'SBUX', 'GILD', 'ADI', 'VRTX', 'ADP',
            'LRCX', 'FISV', 'REGN', 'ATVI', 'KLAC', 'MRNA', 'ORLY', 'MCHP', 'CSX', 'DXCM',
            'ABNB', 'PANW', 'CRWD', 'FTNT', 'TEAM', 'DOCU', 'ZM', 'PTON', 'OKTA', 'MTCH',
            'DDOG', 'ZS', 'LCID', 'RIVN', 'SHOP', 'PLTR', 'MSTR', 'AXON', 'APP', 'SMCI'
        ]
        
        # Obtener cambios históricos para tickers adicionales
        ndx_changes = get_nasdaq100_historical_changes()
        historical_tickers = set()
        
        if not ndx_changes.empty:
            all_mentioned = ndx_changes['Ticker'].unique()
            historical_tickers.update(all_mentioned)
        
        # Combinar tickers conocidos con históricos
        all_ndx_tickers = list(set(known_ndx_tickers + list(historical_tickers)))
        
        # Filtrar solo los que tienen datos disponibles
        available_tickers = get_all_available_tickers()
        final_tickers = [t for t in all_ndx_tickers if t in available_tickers]
        
        print(f"✅ NASDAQ-100 tickers obtenidos localmente: {len(final_tickers)}")
        return {'tickers': final_tickers}
        
    except Exception as e:
        print(f"❌ Error obteniendo NASDAQ-100 local: {e}")
        return {'tickers': []}

def get_current_constituents(index_name):
    """Obtiene constituyentes actuales usando SOLO datos locales"""
    cache_key = f"current_{index_name}"
    
    cached_data = load_cache(cache_key, prefix="constituents", max_age_days=1)
    if cached_data is not None:
        print(f"⚡ Usando caché para {index_name}")
        return cached_data
    
    print(f"🔍 Obteniendo constituyentes locales para {index_name}...")
    
    if index_name == "SP500":
        result = get_sp500_tickers_from_local_data()
    elif index_name == "NDX":
        result = get_nasdaq100_tickers_from_local_data()
    elif index_name == "Ambos (SP500 + NDX)":
        sp500 = get_sp500_tickers_from_local_data()
        ndx = get_nasdaq100_tickers_from_local_data()
        combined_tickers = list(set(sp500['tickers'] + ndx['tickers']))
        result = {'tickers': combined_tickers}
        print(f"✅ Combinados: {len(sp500['tickers'])} S&P500 + {len(ndx['tickers'])} NDX = {len(combined_tickers)} únicos")
    else:
        print(f"❌ Índice '{index_name}' no reconocido")
        return {'tickers': []}
    
    save_cache(cache_key, result, prefix="constituents")
    return result

def get_all_available_tickers_with_historical_validation(index_name, start_date, end_date):
    """Obtiene tickers válidos para el período usando datos locales"""
    try:
        print(f"🔍 Validando constituyentes históricos de {index_name} para {start_date} a {end_date}")
        
        # 1. Obtener constituyentes actuales (locales)
        current_constituents = get_current_constituents(index_name)
        
        if not current_constituents or 'tickers' not in current_constituents:
            print("❌ No se pudieron obtener constituyentes actuales")
            return {'tickers': []}, "No se pudieron obtener constituyentes"
        
        print(f"✅ Constituyentes actuales: {len(current_constituents['tickers'])}")
        
        # 2. Obtener cambios históricos
        if index_name == "SP500":
            historical_changes = get_sp500_historical_changes()
        elif index_name == "NDX":
            historical_changes = get_nasdaq100_historical_changes()
        else:  # Ambos
            sp500_changes = get_sp500_historical_changes()
            ndx_changes = get_nasdaq100_historical_changes()
            if not sp500_changes.empty and not ndx_changes.empty:
                historical_changes = pd.concat([sp500_changes, ndx_changes], ignore_index=True)
                historical_changes = historical_changes.drop_duplicates(subset=['Date', 'Ticker', 'Action'])
            elif not sp500_changes.empty:
                historical_changes = sp500_changes
            else:
                historical_changes = ndx_changes
        
        # 3. Convertir fechas
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        elif isinstance(start_date, date):
            start_date = pd.to_datetime(start_date)
            
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        elif isinstance(end_date, date):
            end_date = pd.to_datetime(end_date)
        
        print(f"📅 Período de validación: {start_date.date()} a {end_date.date()}")
        
        # 4. Empezar con constituyentes actuales
        valid_tickers = set(current_constituents['tickers'])
        print(f"🎯 Tickers iniciales: {len(valid_tickers)}")
        
        # 5. Aplicar validación histórica si tenemos datos
        if not historical_changes.empty:
            print(f"📊 Procesando {len(historical_changes)} cambios históricos...")
            
            # Filtrar cambios futuros
            today = pd.to_datetime(datetime.now().date())
            historical_changes = historical_changes[pd.to_datetime(historical_changes['Date']) <= today]
            
            for _, change in historical_changes.iterrows():
                change_date = pd.to_datetime(change['Date'])
                ticker = str(change['Ticker']).upper()
                action = change['Action']
                
                if change_date > end_date:
                    continue
                
                if action == 'Added' and change_date > start_date:
                    # Añadido durante el período, no estaba al inicio
                    valid_tickers.discard(ticker)
                elif action == 'Removed' and change_date <= start_date:
                    # Removido antes del período, no estaba presente
                    valid_tickers.discard(ticker)
        
        print(f"📊 Tickers válidos históricamente: {len(valid_tickers)}")
        
        # 6. Verificar disponibilidad de datos
        available_tickers = get_all_available_tickers()
        final_tickers = list(valid_tickers & set(available_tickers))
        
        print(f"✅ Tickers finales con datos: {len(final_tickers)}")
        
        if len(final_tickers) == 0:
            print("⚠️ No hay tickers válidos, usando todos los disponibles como fallback")
            final_tickers = available_tickers[:100]  # Usar primeros 100
        
        return {
            'tickers': final_tickers,
            'data': [],
            'historical_data_available': True if not historical_changes.empty else False
        }, None
        
    except Exception as e:
        error_msg = f"Error en validación histórica: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Fallback a todos los tickers disponibles
        try:
            available_tickers = get_all_available_tickers()
            if available_tickers:
                return {
                    'tickers': available_tickers[:200],  # Usar primeros 200
                    'data': [],
                    'historical_data_available': False,
                    'note': f'Fallback due to error: {str(e)}'
                }, f"Warning: Using available tickers as fallback - {str(e)}"
            else:
                return {'tickers': []}, error_msg
        except Exception as fallback_error:
            return {'tickers': []}, f"{error_msg} | Fallback error: {str(fallback_error)}"

def get_constituents_at_date(index_name, start_date, end_date):
    """Obtiene constituyentes para una fecha específica"""
    try:
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
    """Genera resumen de tickers removidos"""
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
    return total_size / (1024 * 1024)

def debug_system_status():
    """Función de debug para verificar el estado del sistema"""
    print("\n=== DEBUG: ESTADO DEL SISTEMA ===")
    print(f"📁 Directorio de datos: {DATA_DIR}")
    print(f"   Existe: {'✅' if os.path.exists(DATA_DIR) else '❌'}")
    
    # Verificar archivos disponibles
    available_tickers = get_all_available_tickers()
    print(f"📊 Tickers disponibles: {len(available_tickers)}")
    if available_tickers:
        print(f"   Primeros 10: {available_tickers[:10]}")
    
    # Verificar cambios históricos
    sp500_changes = get_sp500_historical_changes()
    ndx_changes = get_nasdaq100_historical_changes()
    print(f"📋 Cambios S&P 500: {len(sp500_changes)}")
    print(f"📋 Cambios NASDAQ-100: {len(ndx_changes)}")
    
    print("=== FIN DEBUG ===\n")

# Solo ejecutar si se llama directamente
if __name__ == "__main__":
    debug_system_status()
    
    # Probar índices
    test_indices = ["SP500", "NDX", "Ambos (SP500 + NDX)"]
    for idx in test_indices:
        print(f"\n🧪 Probando índice: {idx}")
        result = get_current_constituents(idx)
        if result and 'tickers' in result:
            print(f"✅ {idx}: {len(result['tickers'])} tickers encontrados")
        else:
            print(f"❌ {idx}: Error obteniendo tickers")
