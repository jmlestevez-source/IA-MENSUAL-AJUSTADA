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

# Cache global en memoria
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

def _read_html_with_ua(url, attrs=None):
    """Lee tablas HTML con User-Agent para evitar bloqueos"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        html = resp.text
        return pd.read_html(StringIO(html), attrs=attrs)
    except Exception as e:
        print(f"_read_html_with_ua error ({url}): {e}")
        return []

def parse_wikipedia_date(date_str):
    """Parsea fechas de Wikipedia en distintos formatos (robusto)"""
    if pd.isna(date_str) or not date_str or str(date_str).lower() in ['nan', 'none', '']:
        return None
    date_str = str(date_str).strip()
    try:
        parsed_date = parser.parse(date_str, fuzzy=True)
        return parsed_date.date()
    except Exception:
        try:
            month_map = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12,
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
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
        except Exception:
            return None

# ==============================
# Constituyentes actuales (Wikipedia)
# ==============================

def get_sp500_tickers_from_wikipedia():
    """Obtiene tickers actuales del S&P 500 desde Wikipedia (id=constituents)"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _read_html_with_ua(url, attrs={"id": "constituents"})
    df = tables[0] if tables else None
    if df is None:
        # Fallback: buscar tabla con Symbol/Ticker
        tables = _read_html_with_ua(url)
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if any(c in ("symbol", "ticker") for c in cols):
                df = t
                break
    if df is None:
        return {"tickers": []}
    sym_col = next((c for c in df.columns if str(c).strip().lower() in ("symbol", "ticker")), None)
    if not sym_col:
        return {"tickers": []}
    syms = df[sym_col].astype(str).str.upper().str.replace(".", "-", regex=False).tolist()
    syms = [s.strip() for s in syms if s and s.strip()]
    return {"tickers": list(dict.fromkeys(syms))}

def get_nasdaq100_tickers_from_wikipedia():
    """Obtiene tickers actuales del NASDAQ-100 desde Wikipedia (id=constituents)"""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = _read_html_with_ua(url, attrs={"id": "constituents"})
    df = tables[0] if tables else None
    if df is None:
        # Fallback: buscar Ticker/Symbol
        tables = _read_html_with_ua(url)
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if any(("ticker" in c or "symbol" in c) for c in cols):
                df = t
                break
    if df is None:
        return {"tickers": []}
    sym_col = next((c for c in df.columns if ("ticker" in str(c).lower()) or ("symbol" in str(c).lower())), None)
    if not sym_col:
        return {"tickers": []}
    syms = df[sym_col].astype(str).str.upper().str.replace(".", "-", regex=False).tolist()
    syms = [s.strip() for s in syms if s and s.strip()]
    return {"tickers": list(dict.fromkeys(syms))}

def get_current_constituents(index_name):
    """Obtiene constituyentes actuales con caché (SP500, NDX o Ambos)"""
    idx = str(index_name).strip().upper()
    cache_key = f"current_{idx}"

    cached_data = load_cache(cache_key, prefix="constituents", max_age_days=1)
    if cached_data is not None:
        return cached_data

    if idx in {"SP500", "S&P 500", "S&P500"}:
        result = get_sp500_tickers_from_wikipedia()
    elif idx in {"NDX", "NASDAQ-100", "NASDAQ100"}:
        result = get_nasdaq100_tickers_from_wikipedia()
    elif idx.startswith("AMBOS") or idx in {"BOTH", "SP500+NDX", "ALL", "AMBOS (SP500 + NDX)"}:
        sp = get_sp500_tickers_from_wikipedia().get('tickers', [])
        nd = get_nasdaq100_tickers_from_wikipedia().get('tickers', [])
        tickers = sorted(list(set([t.upper().replace('.', '-') for t in (sp + nd)])))
        result = {'tickers': tickers}
    else:
        raise ValueError(f"Índice {index_name} no soportado")

    save_cache(cache_key, result, prefix="constituents")
    return result

# ==============================
# Cambios históricos (Wikipedia/CSV)
# ==============================

def _normalize_ticker_str(x):
    return str(x).strip().upper().replace('.', '-')

def download_sp500_changes_from_wikipedia():
    """Descarga cambios del S&P 500 desde Wikipedia y normaliza"""
    print("🌐 Descargando cambios del S&P 500 desde Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = _read_html_with_ua(url)
        if not tables:
            print("❌ No se pudieron leer tablas de Wikipedia")
            return pd.DataFrame()
        # Buscar tabla con columnas Date + Added/Removed
        changes_df = None
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if ("date" in cols) and any(("added" in c or "additions" in c) for c in cols) and any(("removed" in c or "removals" in c) for c in cols):
                changes_df = t
                break
        if changes_df is None and len(tables) >= 2:
            changes_df = tables[1]  # fallback: segunda tabla
        if changes_df is None:
            print("⚠️ No se encontró tabla de cambios S&P 500")
            return pd.DataFrame()

        df = changes_df.copy()
        # Normalizar nombres de columnas
        col_map = {}
        for c in df.columns:
            lc = str(c).strip().lower()
            if lc == 'date':
                col_map[c] = 'Date'
            elif 'add' in lc:
                col_map[c] = 'Added'
            elif 'remov' in lc:
                col_map[c] = 'Removed'
        df = df.rename(columns=col_map)
        if 'Date' not in df.columns:
            return pd.DataFrame()

        out = []
        for _, row in df.iterrows():
            d = parse_wikipedia_date(row.get('Date'))
            if not d:
                continue
            for col, action in [('Added', 'Added'), ('Removed', 'Removed')]:
                if col in df.columns and pd.notna(row.get(col)):
                    raw = str(row.get(col))
                    # Splits por saltos de línea, comas, punto y coma
                    parts = re.split(r'[\n,;]+', raw)
                    for p in parts:
                        tk = re.sub(r'```math
.*?```', '', str(p)).strip()  # quitar refs [1]
                        tk = tk.split()[0] if ' ' in tk else tk
                        tk = _normalize_ticker_str(tk)
                        if tk and len(tk) <= 6 and not tk.isdigit():
                            out.append({'Date': pd.to_datetime(d), 'Ticker': tk, 'Action': action})
        if not out:
            return pd.DataFrame()
        res = pd.DataFrame(out).drop_duplicates().sort_values('Date')
        print(f"✅ Descargados {len(res)} cambios del S&P 500")
        return res
    except Exception as e:
        print(f"❌ Error S&P 500 Wikipedia: {e}")
        return pd.DataFrame()

def get_sp500_historical_changes():
    """Obtiene cambios históricos del S&P 500 desde CSV local o Wikipedia"""
    local_csv_path = "sp500_changes.csv"
    fallback_path = os.path.join(DATA_DIR, "sp500_changes.csv")

    changes_df = pd.DataFrame()
    loaded_from_local = False

    for csv_path in [local_csv_path, fallback_path]:
        if os.path.exists(csv_path):
            try:
                print(f"📂 Cargando cambios S&P 500 desde {csv_path}...")
                changes_df = pd.read_csv(csv_path, parse_dates=['Date'])
                # Normalizar
                if 'Ticker' in changes_df.columns:
                    changes_df['Ticker'] = changes_df['Ticker'].astype(str).str.upper().str.replace('.', '-', regex=False)
                loaded_from_local = True
                break
            except Exception as e:
                print(f"⚠️ Error leyendo {csv_path}: {e}")

    try:
        if loaded_from_local:
            last_date = pd.to_datetime(changes_df['Date'], errors='coerce').max()
            if pd.isna(last_date) or (datetime.now() - last_date.to_pydatetime()).days > 7:
                print("🔄 Buscando actualizaciones en Wikipedia...")
                wikipedia_df = download_sp500_changes_from_wikipedia()
                if not wikipedia_df.empty:
                    merged = pd.concat([changes_df, wikipedia_df], ignore_index=True)
                    merged = merged.drop_duplicates(subset=['Date', 'Ticker', 'Action']).sort_values('Date')
                    try:
                        merged.to_csv(local_csv_path, index=False)
                        print(f"💾 CSV actualizado guardado en {local_csv_path}")
                    except Exception:
                        pass
                    return merged
        else:
            print("🌐 No se encontró CSV local. Descargando desde Wikipedia...")
            wikipedia_df = download_sp500_changes_from_wikipedia()
            if not wikipedia_df.empty:
                try:
                    wikipedia_df.to_csv(local_csv_path, index=False)
                    print(f"💾 Guardado en {local_csv_path}")
                except Exception:
                    pass
                return wikipedia_df
    except Exception as e:
        print(f"⚠️ Error actualizando S&P 500 desde Wikipedia: {e}")

    return changes_df if not changes_df.empty else pd.DataFrame()

def download_nasdaq100_changes_from_wikipedia():
    """Descarga cambios del NASDAQ-100 desde Wikipedia (robusto)"""
    print("🌐 Descargando cambios del NASDAQ-100 desde Wikipedia...")
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        tables = _read_html_with_ua(url)
        if not tables:
            print("❌ No se pudieron leer tablas de Wikipedia (NDX)")
            return pd.DataFrame()

        # Buscar tablas con Date + Added/Removed (varían por secciones/años)
        out = []
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if 'date' not in cols:
                continue
            # detectar columnas de añadidos/remociones
            cand_add = [c for c in t.columns if 'add' in str(c).lower() or 'addition' in str(c).lower()]
            cand_rem = [c for c in t.columns if 'remov' in str(c).lower()]
            if not cand_add and not cand_rem:
                continue
            # Normalizar nombres
            df = t.copy()
            col_map = {}
            for c in df.columns:
                lc = str(c).strip().lower()
                if lc == 'date':
                    col_map[c] = 'Date'
                elif 'add' in lc or 'addition' in lc:
                    col_map[c] = 'Added'
                elif 'remov' in lc:
                    col_map[c] = 'Removed'
            df = df.rename(columns=col_map)
            if 'Date' not in df.columns:
                continue

            for _, row in df.iterrows():
                d = parse_wikipedia_date(row.get('Date'))
                if not d:
                    continue
                for col, action in [('Added', 'Added'), ('Removed', 'Removed')]:
                    if col in df.columns and pd.notna(row.get(col)):
                        raw = str(row.get(col))
                        parts = re.split(r'[\n,;]+', raw)
                        for p in parts:
                            tk = re.sub(r'KATEX_INLINE_OPEN.*?KATEX_INLINE_CLOSE|```math
.*?```', '', str(p)).strip()  # quitar paréntesis y refs
                            tk = tk.split()[0] if ' ' in tk else tk
                            tk = _normalize_ticker_str(tk)
                            if tk and len(tk) <= 6 and not tk.isdigit():
                                out.append({'Date': pd.to_datetime(d), 'Ticker': tk, 'Action': action})

        if not out:
            print("⚠️ No se encontraron cambios NDX")
            return pd.DataFrame()
        res = pd.DataFrame(out).drop_duplicates().sort_values('Date')
        print(f"✅ Descargados {len(res)} cambios del NASDAQ-100")
        return res
    except Exception as e:
        print(f"❌ Error NDX Wikipedia: {e}")
        return pd.DataFrame()

def get_nasdaq100_historical_changes():
    """Obtiene cambios históricos del NASDAQ-100 desde CSV local o Wikipedia"""
    local_csv_path = "ndx_changes.csv"
    fallback_path = os.path.join(DATA_DIR, "ndx_changes.csv")

    changes_df = pd.DataFrame()
    loaded_from_local = False

    for csv_path in [local_csv_path, fallback_path]:
        if os.path.exists(csv_path):
            try:
                print(f"📂 Cargando cambios NDX desde {csv_path}...")
                changes_df = pd.read_csv(csv_path, parse_dates=['Date'])
                if 'Ticker' in changes_df.columns:
                    changes_df['Ticker'] = changes_df['Ticker'].astype(str).str.upper().str.replace('.', '-', regex=False)
                loaded_from_local = True
                break
            except Exception as e:
                print(f"⚠️ Error leyendo {csv_path}: {e}")

    try:
        if loaded_from_local:
            last_date = pd.to_datetime(changes_df['Date'], errors='coerce').max()
            if pd.isna(last_date) or (datetime.now() - last_date.to_pydatetime()).days > 7:
                print("🔄 Buscando actualizaciones en Wikipedia (NDX)...")
                wikipedia_df = download_nasdaq100_changes_from_wikipedia()
                if not wikipedia_df.empty:
                    merged = pd.concat([changes_df, wikipedia_df], ignore_index=True)
                    merged = merged.drop_duplicates(subset=['Date', 'Ticker', 'Action']).sort_values('Date')
                    try:
                        merged.to_csv(local_csv_path, index=False)
                        print(f"💾 CSV actualizado guardado en {local_csv_path}")
                    except Exception:
                        pass
                    return merged
        else:
            print("🌐 No se encontró CSV local NDX. Descargando desde Wikipedia...")
            wikipedia_df = download_nasdaq100_changes_from_wikipedia()
            if not wikipedia_df.empty:
                try:
                    wikipedia_df.to_csv(local_csv_path, index=False)
                    print(f"💾 Guardado en {local_csv_path}")
                except Exception:
                    pass
                return wikipedia_df
    except Exception as e:
        print(f"⚠️ Error actualizando NDX desde Wikipedia: {e}")

    return changes_df if not changes_df.empty else pd.DataFrame()

# ==============================
# Descarga de precios
# ==============================

def download_prices_parallel(tickers, start_date, end_date, load_full_data=True, max_workers=10):
    """
    Carga precios desde CSV locales en paralelo
    """
    if isinstance(tickers, dict) and 'tickers' in tickers:
        ticker_list = tickers['tickers']
    elif isinstance(tickers, (list, tuple)):
        ticker_list = list(tickers)
    elif isinstance(tickers, str):
        ticker_list = [tickers]
    else:
        ticker_list = []

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
            cache_key = get_cache_key(ticker, start_date, end_date)
            cached = load_cache(cache_key, prefix="price", max_age_days=1)
            if cached is not None:
                return ticker, cached.get('price'), cached.get('ohlc')

            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)

            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            start_filter = start_date.date() if isinstance(start_date, datetime) else start_date
            end_filter = end_date.date() if isinstance(end_date, datetime) else end_date
            mask = (df.index.date >= start_filter) & (df.index.date <= end_filter)
            df_filtered = df[mask]

            if df_filtered.empty:
                return ticker, None, None

            if 'Adj Close' in df_filtered.columns:
                price_series = df_filtered['Adj Close']
            elif 'Close' in df_filtered.columns:
                price_series = df_filtered['Close']
            else:
                return ticker, None, None

            ohlc = None
            if load_full_data and all(col in df_filtered.columns for col in ['High', 'Low', 'Close']):
                ohlc = {
                    'High': df_filtered['High'],
                    'Low': df_filtered['Low'],
                    'Close': df_filtered['Adj Close'] if 'Adj Close' in df_filtered.columns else df_filtered['Close'],
                    'Volume': df_filtered.get('Volume')
                }

            save_cache(cache_key, {'price': price_series, 'ohlc': ohlc}, prefix="price")
            return ticker, price_series, ohlc

        except Exception:
            return ticker, None, None

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

    if prices_data:
        prices_df = pd.DataFrame(prices_data)
        prices_df = prices_df.interpolate(method='linear', limit_direction='both')
        print(f"✅ Cargados {len(prices_df.columns)} tickers con datos válidos")
        return prices_df, ohlc_data
    else:
        return pd.DataFrame(), {}

def download_prices(tickers, start_date, end_date, load_full_data=True):
    """Wrapper compatibilidad"""
    prices_df, ohlc_data = download_prices_parallel(
        tickers, start_date, end_date,
        load_full_data=load_full_data,
        max_workers=10
    )
    if load_full_data:
        return prices_df, ohlc_data
    else:
        return prices_df

# ==============================
# Universo de constituyentes con validación
# ==============================

def get_constituents_at_date(index_name, start_date, end_date):
    """Obtiene constituyentes con caché mejorado"""
    cache_key = get_cache_key(index_name, start_date, end_date)

    if cache_key in _constituents_cache:
        print(f"⚡ Usando caché en memoria para {index_name}")
        return _constituents_cache[cache_key], None

    cached_data = load_cache(cache_key, prefix="constituents", max_age_days=7)
    if cached_data is not None:
        print(f"📦 Usando caché en disco para {index_name}")
        _constituents_cache[cache_key] = cached_data
        return cached_data, None

    try:
        result, error = get_all_available_tickers_with_historical_validation(
            index_name, start_date, end_date
        )
        if result:
            _constituents_cache[cache_key] = result
            save_cache(cache_key, result, prefix="constituents")
            return result, error
        else:
            current_data = get_current_constituents(index_name)
            fallback_result = {
                'tickers': current_data.get('tickers', []),
                'data': [{'ticker': t, 'added': 'Unknown', 'in_current': True, 'status': 'Current fallback'} for t in current_data.get('tickers', [])],
                'historical_data_available': False,
                'note': 'Fallback to current constituents'
            }
            return fallback_result, "Warning: Using current constituents as fallback"

    except Exception as e:
        error_msg = f"Error obteniendo constituyentes para {index_name}: {e}"
        print(error_msg)
        return None, error_msg

def get_all_available_tickers_with_historical_validation(index_name, start_date, end_date):
    """
    EXACTO a lo pedido:
    - Gate: solo tickers que estén HOY en la tabla 'constituents' de Wikipedia (id=constituents),
            e intersección con los CSV disponibles en data/.
    - Si el ticker aparece en el CSV de changes:
        Incluye si existe algún 'Added' <= fecha de selección y NO hay ningún 'Removed' < fecha de selección.
    - Si NO aparece en el CSV de changes, se asume 'histórico' (siempre estuvo) y se incluye siempre.
    - index_name puede ser 'SP500', 'NDX' o 'Ambos (SP500 + NDX)'.
    """
    try:
        def _norm_t(t):
            return str(t).strip().upper().replace('.', '-')

        def _month_end(d):
            return (pd.Timestamp(d) + pd.offsets.MonthEnd(0)).date()

        selection_date = _month_end(end_date)

        # 1) Tickers con CSV locales
        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        available = set()
        for p in csv_files:
            fn = os.path.basename(p)
            if not fn.endswith(".csv"):
                continue
            tk = _norm_t(fn[:-4])
            if tk and len(tk) <= 6 and not tk.isdigit() and tk not in {'SPY', 'QQQ', 'IEF', 'BIL'}:
                available.add(tk)

        # 2) Constituyentes actuales (hoy)
        idx = str(index_name).strip().upper()
        if idx.startswith("AMBOS") or idx in {"BOTH", "SP500+NDX", "ALL", "AMBOS (SP500 + NDX)"}:
            sp = get_current_constituents("SP500").get('tickers', [])
            nd = get_current_constituents("NDX").get('tickers', [])
            current_tickers = set(map(_norm_t, list(sp) + list(nd)))
            sp_ch = get_sp500_historical_changes()
            nd_ch = get_nasdaq100_historical_changes()
            if sp_ch is None:
                sp_ch = pd.DataFrame()
            if nd_ch is None:
                nd_ch = pd.DataFrame()
            changes = pd.concat([sp_ch, nd_ch], ignore_index=True) if (not sp_ch.empty or not nd_ch.empty) else pd.DataFrame()
        elif idx in {"SP500", "S&P500", "S&P 500"}:
            current_tickers = set(map(_norm_t, get_current_constituents("SP500").get('tickers', [])))
            changes = get_sp500_historical_changes()
        elif idx in {"NDX", "NASDAQ-100", "NASDAQ100"}:
            current_tickers = set(map(_norm_t, get_current_constituents("NDX").get('tickers', [])))
            changes = get_nasdaq100_historical_changes()
        else:
            raise ValueError(f"Índice {index_name} no soportado")

        # Candidatos = intersección con CSV disponibles
        candidates = sorted(current_tickers & available)

        # 3) Si no hay changes válidos, asumir históricos
        if changes is None or changes.empty or not {'Date', 'Ticker', 'Action'}.issubset(set(changes.columns)):
            details = []
            for t in candidates:
                details.append({
                    'ticker': t,
                    'in_current': True,
                    'in_changes': False,
                    'eligible_on': selection_date.isoformat(),
                    'status': 'Incluido (sin registros de cambios: se asume histórico)'
                })
            return {
                'tickers': candidates,
                'data': details,
                'historical_data_available': False,
                'note': f'Sin changes válidos. Selección: {selection_date.isoformat()}'
            }, None

        # Normalizar changes
        ch = changes.copy()
        ch['Ticker'] = ch['Ticker'].astype(str).str.upper().str.replace('.', '-', regex=False)
        ch['Date'] = pd.to_datetime(ch['Date'], errors='coerce').dt.date
        ch = ch.dropna(subset=['Date', 'Ticker', 'Action'])
        ch['Action'] = ch['Action'].astype(str).str.lower()

        grouped = ch.groupby('Ticker')
        included = []
        details = []

        for t in candidates:
            if t not in grouped.groups:
                included.append(t)
                details.append({
                    'ticker': t,
                    'in_current': True,
                    'in_changes': False,
                    'eligible_on': selection_date.isoformat(),
                    'status': 'Incluido (ticker no aparece en changes: se asume histórico)'
                })
                continue

            tdf = grouped.get_group(t)
            add_mask = tdf['Action'].str.contains('add', na=False)
            rem_mask = tdf['Action'].str.contains('remov', na=False)
            added_dates = sorted([d for d in tdf.loc[add_mask, 'Date'] if pd.notna(d)])
            removed_dates = sorted([d for d in tdf.loc[rem_mask, 'Date'] if pd.notna(d)])

            added_before_or_on = any(d <= selection_date for d in added_dates)
            removed_before = any(d < selection_date for d in removed_dates)

            ok = added_before_or_on and not removed_before
            if ok:
                included.append(t)
                status = 'Incluido (Added <= selección y sin Removed antes)'
            else:
                status = 'Excluido (Added > selección o hubo Removed antes)'

            details.append({
                'ticker': t,
                'in_current': True,
                'in_changes': True,
                'added_dates': [d.isoformat() for d in added_dates],
                'removed_dates': [d.isoformat() for d in removed_dates],
                'eligible_on': selection_date.isoformat(),
                'status': status
            })

        return {
            'tickers': included,
            'data': details,
            'historical_data_available': True,
            'note': f'Gate: constituyentes actuales + regla de changes | Selección: {selection_date.isoformat()}'
        }, None

    except Exception as e:
        return None, str(e)

# ==============================
# Utilidades varias de caché
# ==============================

def generate_removed_tickers_summary():
    """Genera resumen de tickers removidos (stub con caché)"""
    cache_key = "removed_tickers_summary"
    cached_data = load_cache(cache_key, max_age_days=30)
    if cached_data is not None:
        return cached_data
    # Implementa si necesitas un reporte; devolvemos vacío para no romper nada.
    df = pd.DataFrame()
    save_cache(cache_key, df)
    return df

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

print(f"💾 Tamaño actual de caché: {get_cache_size():.2f} MB")
