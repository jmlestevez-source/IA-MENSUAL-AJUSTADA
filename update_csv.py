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
    """Obtiene tickers existentes"""
    csv_files = glob.glob("data/*.csv")
    tickers = []
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        if filename.endswith('.csv'):
            ticker = filename.replace('.csv', '')
            tickers.append(ticker)
    return sorted(list(set(tickers)))

def regenerate_full_history(ticker):
    """Regenera historial completo"""
    try:
        print(f"  🔄 Regenerando {ticker}...")
        
        yticker = yf.Ticker(ticker)
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
    """Actualiza datos de ticker"""
    try:
        filename = f"data/{ticker}.csv"
        
        if not os.path.exists(filename) and not force_regenerate:
            print(f"❌ {ticker}: Archivo no encontrado")
            return False
        
        needs_full_regeneration = force_regenerate
        
        if not needs_full_regeneration and os.path.exists(filename):
            try:
                df_existing = pd.read_csv(filename, index_col="Date", parse_dates=True)
                
                if 'Adj Close' in df_existing.columns:
                    print(f"⚠️ {ticker}: Datos no ajustados detectados")
                    needs_full_regeneration = True
                elif len([c for c in df_existing.columns if c in ['Open','High','Low','Close','Volume']]) != 5:
                    print(f"⚠️ {ticker}: Estructura incorrecta")
                    needs_full_regeneration = True
                
            except Exception as e:
                print(f"⚠️ {ticker}: Error leyendo archivo")
                needs_full_regeneration = True
        
        if needs_full_regeneration:
            data = regenerate_full_history(ticker)
            if data is not None:
                data.to_csv(filename)
                print(f"✅ {ticker}: Regenerado")
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
                print(f"📅 {ticker}: Fin de semana")
                return True
            
            if last_date.date() >= today.date():
                print(f"✅ {ticker}: Actualizado")
                return True
            
            est = pytz.timezone('US/Eastern')
            now_est = datetime.now(est)
            
            if not FORCE_UPDATE and now_est.hour < 17:
                yesterday = today - timedelta(days=1)
                while yesterday.weekday() > 4:
                    yesterday = yesterday - timedelta(days=1)
                
                if last_date.date() >= yesterday.date():
                    print(f"⏰ {ticker}: Datos recientes")
                    return True
            
            print(f"📥 {ticker}: Actualizando...")
            
            yticker = yf.Ticker(ticker)
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
                    print(f"✅ {ticker}: +{len(new_data)} días")
                    return True
                else:
                    print(f"ℹ️ {ticker}: Sin datos nuevos")
                    return True
            else:
                print(f"⚠️ {ticker}: Sin respuesta de Yahoo")
                return True
        else:
            data = regenerate_full_history(ticker)
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
        ticker = os.path.basename(file_path).replace('.csv', '')
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
            
        except Exception as e:
            need_fix.append(ticker)
            print(f"  ❌ {ticker}: Error leyendo")
    
    print(f"\n📊 Resumen:")
    print(f"  ✅ OK: {len(ok_files)}")
    print(f"  ⚠️ Necesitan fix: {len(need_fix)}")
    
    return need_fix

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
    
    print("\n📊 ACTUALIZANDO DATOS")
    print("=" * 60)
    
    all_tickers = get_existing_tickers()
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
    
    print("\n🔍 Verificación:")
    for ticker in priority_found[:5]:
        try:
            df = pd.read_csv(f"data/{ticker}.csv", index_col="Date", parse_dates=True)
            has_adj = 'Adj Close' in df.columns
            status = "⚠️ TIENE ADJ CLOSE" if has_adj else "✅ OK"
            print(f"  {ticker}: {len(df)} registros - {status}")
        except Exception as e:
            print(f"  {ticker}: Error")
    
    print("\n✅ Proceso completado")
