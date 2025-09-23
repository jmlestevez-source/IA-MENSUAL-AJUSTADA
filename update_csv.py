import sys
import yfinance as yf
import pandas as pd
import os
import glob
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

FORCE_UPDATE = '--force' in sys.argv

def normalize_symbol(ticker: str) -> str:
    """Normaliza el símbolo a formato Yahoo y de nombre de archivo (BRK.B -> BRK-B)."""
    return str(ticker).strip().upper().replace('.', '-')

def normalize_adjusted_dataframe(df):
    """Normaliza el DataFrame con datos ajustados"""
    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    if any(isinstance(col, int) for col in df.columns) or len(df.columns) > 5:
        new_columns = {}
        col_list = list(df.columns)
        
        for i, col in enumerate(col_list):
            if i < len(expected_columns):
                new_columns[col] = expected_columns[i]
        
        if new_columns:
            df = df.rename(columns=new_columns)
    
    valid_cols = [col for col in expected_columns if col in df.columns]
    df = df[valid_cols]
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = pd.NA
    
    df = df[expected_columns]
    df = df.dropna(how='all')
    
    price_cols = ['Open', 'High', 'Low', 'Close']
    df = df.dropna(subset=price_cols, how='all')
    
    return df

def get_existing_tickers():
    """Obtiene tickers existentes (por nombre de archivo en data/)."""
    csv_files = glob.glob("data/*.csv")
    tickers = []
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        if filename.endswith('.csv'):
            ticker = filename.replace('.csv', '').upper()
            tickers.append(ticker)
    return sorted(list(set(tickers)))

def load_extra_tickers(path="data/extra_tickers.txt"):
    """Carga tickers extra declarados (uno por línea) para crear/actualizar."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return sorted(list({normalize_symbol(line) for line in f if line.strip()}))
    return []

def save_extra_tickers(tickers, path="data/extra_tickers.txt"):
    """Guarda el conjunto de tickers extra (uno por línea) de forma ordenada."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    unique_sorted = sorted(list({normalize_symbol(t) for t in tickers}))
    with open(path, "w") as f:
        for t in unique_sorted:
            f.write(f"{t}\n")
    return unique_sorted

def append_extra_tickers(new_tickers, path="data/extra_tickers.txt"):
    """Añade tickers al archivo extra_tickers.txt evitando duplicados y devolviendo los nuevos realmente añadidos."""
    existing = set(load_extra_tickers(path))
    normalized_new = {normalize_symbol(t) for t in new_tickers if str(t).strip()}
    to_add = normalized_new - existing
    if to_add:
        updated = sorted(list(existing | to_add))
        save_extra_tickers(updated, path)
    return sorted(list(to_add))

def read_changes_csv(file_path):
    """
    Lee un CSV de cambios y lo normaliza a columnas: Date, Ticker, Action.
    Maneja delimitadores inferidos (coma, tab, etc.).
    """
    try:
        df = pd.read_csv(file_path, sep=None, engine='python')
        if df.empty:
            return pd.DataFrame(columns=['Date', 'Ticker', 'Action'])
        
        # Normalizar nombres de columnas a minúsculas para detectar equivalencias
        cols = {c.lower().strip(): c for c in df.columns}
        
        # Detectar columnas base
        ticker_col = cols.get('ticker') or cols.get('symbol') or cols.get('símbolo') or cols.get('simbolo')
        action_col = cols.get('action') or cols.get('accion') or cols.get('change') or cols.get('evento')
        date_col = cols.get('date') or cols.get('fecha') or cols.get('effective date') or cols.get('effective')
        
        if not ticker_col or not action_col:
            return pd.DataFrame(columns=['Date', 'Ticker', 'Action'])
        
        work = df.copy()
        work.rename(columns={
            ticker_col: 'Ticker',
            action_col: 'Action',
            **({date_col: 'Date'} if date_col else {})
        }, inplace=True)
        
        work['Ticker'] = work['Ticker'].astype(str).str.strip().str.upper()
        work['Ticker'] = work['Ticker'].str.replace('.', '-', regex=False)  # Normalizar tipo BRK.B
        work['Action'] = work['Action'].astype(str).str.strip()
        if 'Date' in work.columns:
            work['Date'] = pd.to_datetime(work['Date'], errors='coerce')
        else:
            work['Date'] = pd.NaT
        
        work = work[['Date', 'Ticker', 'Action']].dropna(subset=['Ticker', 'Action'])
        return work
    except Exception:
        return pd.DataFrame(columns=['Date', 'Ticker', 'Action'])

def get_added_tickers_from_changes():
    """
    Busca archivos de cambios en raíz o en data/ y devuelve el conjunto de tickers añadidos ('Added').
    """
    candidates = [
        "sp500_changes.csv", "ndx_changes.csv",
        "data/sp500_changes.csv", "data/ndx_changes.csv"
    ]
    added = set()
    for p in candidates:
        if os.path.exists(p):
            df = read_changes_csv(p)
            if not df.empty:
                mask_added = df['Action'].str.lower().str.contains('add')
                added |= set(df.loc[mask_added, 'Ticker'].dropna().astype(str))
    # Normalizar formateo
    added = {normalize_symbol(t) for t in added}
    # Evitar símbolos raros que no sean de Yahoo
    added = {t for t in added if t and not t.startswith('^')}
    return sorted(list(added))

def regenerate_full_history(ticker):
    """Regenera historial completo (auto_adjust=True) usando símbolo normalizado."""
    try:
        t_norm = normalize_symbol(ticker)
        print(f"  🔄 Regenerando {t_norm}...")
        
        yticker = yf.Ticker(t_norm)
        data = yticker.history(period="max", auto_adjust=True)
        
        if data.empty:
            print(f"    ❌ No se pudo obtener datos")
            return None
        
        data = normalize_adjusted_dataframe(data)
        print(f"    ✅ {len(data)} días descargados")
        return data
        
    except Exception as e:
        print(f"    ❌ Error: {str(e)[:100]}")
        return None

def update_ticker_data(ticker, force_regenerate=False):
    """Actualiza datos de ticker (usa ticker normalizado para Yahoo y nombre de archivo)."""
    try:
        t_norm = normalize_symbol(ticker)
        filename = f"data/{t_norm}.csv"
        
        # Si no existe, crearlo desde cero
        if not os.path.exists(filename):
            print(f"📄 {t_norm}: No existía, descargando historial completo...")
            data = regenerate_full_history(t_norm)
            if data is not None:
                data.to_csv(filename)
                print(f"✅ {t_norm}: Creado")
                return True
            else:
                print(f"❌ {t_norm}: No se pudo crear")
                return False
        
        needs_full_regeneration = force_regenerate
        
        if not needs_full_regeneration and os.path.exists(filename):
            try:
                df_existing = pd.read_csv(filename, index_col="Date", parse_dates=True)
                
                if 'Adj Close' in df_existing.columns:
                    print(f"⚠️ {t_norm}: Datos no ajustados detectados")
                    needs_full_regeneration = True
                elif len([c for c in df_existing.columns if c in ['Open','High','Low','Close','Volume']]) != 5:
                    print(f"⚠️ {t_norm}: Estructura incorrecta")
                    needs_full_regeneration = True
                
            except Exception:
                print(f"⚠️ {t_norm}: Error leyendo archivo")
                needs_full_regeneration = True
        
        if needs_full_regeneration:
            data = regenerate_full_history(t_norm)
            if data is not None:
                data.to_csv(filename)
                print(f"✅ {t_norm}: Regenerado")
                return True
            else:
                return False
        
        # Actualización incremental
        df_existing = pd.read_csv(filename, index_col="Date", parse_dates=True)
        df_existing = normalize_adjusted_dataframe(df_existing)
        
        if len(df_existing) > 0:
            last_date = df_existing.index[-1]
            today = datetime.now()
            
            if not FORCE_UPDATE and today.weekday() >= 5:
                print(f"📅 {t_norm}: Fin de semana")
                return True
            
            if last_date.date() >= today.date():
                print(f"✅ {t_norm}: Actualizado")
                return True
            
            est = pytz.timezone('US/Eastern')
            now_est = datetime.now(est)
            
            if not FORCE_UPDATE and now_est.hour < 17:
                yesterday = today - timedelta(days=1)
                while yesterday.weekday() > 4:
                    yesterday = yesterday - timedelta(days=1)
                
                if last_date.date() >= yesterday.date():
                    print(f"⏰ {t_norm}: Datos recientes")
                    return True
            
            print(f"📥 {t_norm}: Actualizando...")
            
            yticker = yf.Ticker(t_norm)
            new_data = yticker.history(period="1mo", auto_adjust=True)
            
            if not new_data.empty:
                new_data = normalize_adjusted_dataframe(new_data)
                new_data.index = pd.to_datetime(new_data.index)
                
                new_data = new_data[new_data.index > last_date]
                
                if len(new_data) > 0:
                    combined = pd.concat([df_existing, new_data])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined = combined.sort_index()
                    
                    combined.to_csv(filename)
                    print(f"✅ {t_norm}: +{len(new_data)} días")
                    return True
                else:
                    print(f"ℹ️ {t_norm}: Sin datos nuevos")
                    return True
            else:
                print(f"⚠️ {t_norm}: Sin respuesta de Yahoo")
                return True
        else:
            data = regenerate_full_history(t_norm)
            if data is not None:
                data.to_csv(filename)
                return True
            return False
            
    except Exception as e:
        print(f"❌ {ticker}: Error - {str(e)[:200]}")
        return False

def check_and_fix_all_csvs():
    """Verifica CSVs que necesitan regeneración"""
    print("\n🔍 VERIFICANDO CSVs")
    print("=" * 50)
    
    csv_files = glob.glob("data/*.csv")
    need_fix = []
    ok_files = []
    
    for file_path in csv_files:
        ticker = os.path.basename(file_path).replace('.csv', '').upper()
        try:
            df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
            
            if 'Adj Close' in df.columns:
                need_fix.append(ticker)
                print(f"  ⚠️ {ticker}: Necesita regeneración")
            elif any(isinstance(col, int) for col in df.columns):
                need_fix.append(ticker)
                print(f"  ⚠️ {ticker}: Columnas numéricas")
            elif len(df.columns) > 5:
                need_fix.append(ticker)
                print(f"  ⚠️ {ticker}: Demasiadas columnas")
            else:
                ok_files.append(ticker)
            
        except Exception:
            need_fix.append(ticker)
            print(f"  ❌ {ticker}: Error leyendo")
    
    print(f"\n📊 Resumen:")
    print(f"  ✅ OK: {len(ok_files)}")
    print(f"  ⚠️ Necesitan fix: {len(need_fix)}")
    
    return need_fix

def register_new_index_additions_and_update_extra():
    """
    Busca nuevas incorporaciones (Action contiene 'add') en los CSVs de cambios,
    y las añade a data/extra_tickers.txt. Devuelve la lista de tickers realmente añadidos.
    """
    added_from_changes = get_added_tickers_from_changes()
    if not added_from_changes:
        print("\nℹ️ No se encontraron incorporaciones en CSVs de cambios")
        return []
    
    print("\n🧭 Tickers detectados como 'Added' en cambios:", added_from_changes)
    newly_added = append_extra_tickers(added_from_changes, path="data/extra_tickers.txt")
    if newly_added:
        print("➕ Añadidos a data/extra_tickers.txt:", newly_added)
    else:
        print("ℹ️ No había nuevos tickers para añadir a extra_tickers.txt")
    return newly_added

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ACTUALIZACIÓN DIARIA")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    if FORCE_UPDATE:
        print("⚡ MODO FORCE ACTIVADO")
    
    # INICIALIZAR VARIABLES DE CONTEO
    regenerated = 0
    failed = 0
    updated = 0
    errors = 0

    # 1) Registrar incorporaciones nuevas en extra_tickers.txt (si las hay)
    newly_added_to_extra = register_new_index_additions_and_update_extra()

    # 2) Cargar y crear tickers extra (p.ej. IEF, BIL y nuevas incorporaciones)
    extra = load_extra_tickers()
    if extra:
        print("\n➕ Tickers extra declarados:", extra)
        for t in extra:
            t_norm = normalize_symbol(t)
            if not os.path.exists(f"data/{t_norm}.csv"):
                data = regenerate_full_history(t_norm)
                if data is not None:
                    data.to_csv(f"data/{t_norm}.csv")
                    print(f"✅ {t_norm}: CSV creado desde cero")
                else:
                    print(f"❌ {t_norm}: No se pudo crear CSV inicial")
    
    # 3) Verificar CSVs existentes
    tickers_to_regenerate = check_and_fix_all_csvs()
    
    if tickers_to_regenerate:
        print(f"\n🔧 REGENERANDO {len(tickers_to_regenerate)} ARCHIVOS")
        print("=" * 60)
        
        priority = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
        priority_regen = [t for t in priority if t in tickers_to_regenerate]
        other_regen = [t for t in tickers_to_regenerate if t not in priority]
        
        for i, ticker in enumerate(priority_regen + other_regen, 1):
            print(f"\n[{i}/{len(tickers_to_regenerate)}] {ticker}")
            if update_ticker_data(ticker, force_regenerate=True):
                regenerated += 1
            else:
                failed += 1
            
            if i % 20 == 0:
                import time
                time.sleep(1)
        
        print(f"\n✅ Regenerados: {regenerated}")
        print(f"❌ Fallidos: {failed}")
    else:
        print("\n✅ Todos los CSVs están en formato correcto")
    
    # 4) Actualizar datos de todos los CSVs
    print("\n📊 ACTUALIZANDO DATOS")
    print("=" * 60)
    
    all_tickers = get_existing_tickers()
    # incluir extra por si acabamos de crearlos
    all_tickers = sorted(list({*all_tickers, *extra}))
    print(f"\n📁 Total: {len(all_tickers)}")
    
    priority_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    priority_found = [t for t in priority_tickers if t in all_tickers]
    other_tickers = [t for t in all_tickers if t not in priority_tickers]
    
    tickers_ordered = priority_found + other_tickers
    
    for i, ticker in enumerate(tickers_ordered, 1):
        if i % 10 == 1:
            print(f"\n--- {i} a {min(i+9, len(tickers_ordered))} de {len(tickers_ordered)} ---")
        
        if ticker not in tickers_to_regenerate:
            result = update_ticker_data(ticker, force_regenerate=False)
            if result:
                updated += 1
            else:
                errors += 1
        
        if i % 50 == 0:
            import time
            time.sleep(2)
    
    print("\n📊 RESUMEN FINAL")
    print("=" * 60)
    print(f"✅ Procesados: {updated + regenerated}")
    print(f"❌ Errores: {errors + failed}")
    print(f"📊 Total: {len(tickers_ordered)}")
    print("\n✅ Proceso completado")
