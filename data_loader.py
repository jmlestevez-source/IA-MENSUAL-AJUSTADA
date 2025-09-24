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

# Caché en memoria
_historical_cache = {}
_constituents_cache = {}

def get_cache_key(*args):
    return hashlib.md5(str(args).encode()).hexdigest()

def save_cache(key, data, prefix="cache"):
    try:
        cache_file = os.path.join(CACHE_DIR, f"{prefix}_{key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error guardando caché: {e}")

def load_cache(key, prefix="cache", max_age_days=7):
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

def _normalize_ticker_str(x):
    return str(x).strip().upper().replace('.', '-')

# -------------------------
# Constituyentes actuales
# -------------------------

def get_sp500_tickers_from_wikipedia():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _read_html_with_ua(url, attrs={"id": "constituents"})
    df = tables[0] if tables else None
    if df is None:
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
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = _read_html_with_ua(url, attrs={"id": "constituents"})
    df = tables[0] if tables else None
    if df is None:
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

# -------------------------
# Cambios históricos
# -------------------------

def download_sp500_changes_from_wikipedia():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = _read_html_with_ua(url)
        if not tables:
            return pd.DataFrame()
        changes_df = None
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if ("date" in cols) and any(("add" in c or "addition" in c) for c in cols) and any(("remov" in c) for c in cols):
                changes_df = t
                break
        if changes_df is None and len(tables) >= 2:
            changes_df = tables[1]
        if changes_df is None:
            return pd.DataFrame()

        df = changes_df.copy()
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
            return pd.DataFrame()

        out = []
        for _, row in df.iterrows():
            d = parse_wikipedia_date(row.get('Date'))
            if not d:
                continue
            for col, action in [('Added', 'Added'), ('Removed', 'Removed')]:
                if col in df.columns and pd.notna(row.get(col)):
                    raw = str(row.get(col))
                    parts = re.split(r'[\n,;]+', raw)
                    for p in parts:
                        tk = re.sub(r'\[.*?\]|\(.*?\)', '', str(p)).strip()
                        tk = tk.split()[0] if ' ' in tk else tk
                        tk = _normalize_ticker_str(tk)
                        if tk and len(tk) <= 6 and not tk.isdigit():
                            out.append({'Date': pd.to_datetime(d), 'Ticker': tk, 'Action': action})
        if not out:
            return pd.DataFrame()
        res = pd.DataFrame(out).drop_duplicates().sort_values('Date')
        return res
    except Exception as e:
        print(f"Error S&P 500 Wikipedia: {e}")
        return pd.DataFrame()

def get_sp500_historical_changes():
    local_csv_path = "sp500_changes.csv"
    fallback_path = os.path.join(DATA_DIR, "sp500_changes.csv")

    changes_df = pd.DataFrame()
    loaded_from_local = False

    for csv_path in [local_csv_path, fallback_path]:
        if os.path.exists(csv_path):
            try:
                changes_df = pd.read_csv(csv_path, parse_dates=['Date'])
                if 'Ticker' in changes_df.columns:
                    changes_df['Ticker'] = changes_df['Ticker'].astype(str).str.upper().str.replace('.', '-', regex=False)
                loaded_from_local = True
                break
            except Exception as e:
                print(f"Error leyendo {csv_path}: {e}")

    try:
        if loaded_from_local:
            last_date = pd.to_datetime(changes_df['Date'], errors='coerce').max()
            if pd.isna(last_date) or (datetime.now() - last_date.to_pydatetime()).days > 7:
                wikipedia_df = download_sp500_changes_from_wikipedia()
                if not wikipedia_df.empty:
                    merged = pd.concat([changes_df, wikipedia_df], ignore_index=True)
                    merged = merged.drop_duplicates(subset=['Date', 'Ticker', 'Action']).sort_values('Date')
                    try:
                        merged.to_csv(local_csv_path, index=False)
                    except Exception:
                        pass
                    return merged
        else:
            wikipedia_df = download_sp500_changes_from_wikipedia()
            if not wikipedia_df.empty:
                try:
                    wikipedia_df.to_csv(local_csv_path, index=False)
                except Exception:
                    pass
                return wikipedia_df
    except Exception as e:
        print(f"Error actualizando S&P 500 desde Wikipedia: {e}")

    return changes_df if not changes_df.empty else pd.DataFrame()

def download_nasdaq100_changes_from_wikipedia():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        tables = _read_html_with_ua(url)
        if not tables:
            return pd.DataFrame()

        out = []
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if 'date' not in cols:
                continue
            cand_add = [c for c in t.columns if 'add' in str(c).lower() or 'addition' in str(c).lower()]
            cand_rem = [c for c in t.columns if 'remov' in str(c).lower()]
            if not cand_add and not cand_rem:
                continue
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
                            tk = re.sub(r'\[.*?\]|\(.*?\)', '', str(p)).strip()
                            tk = tk.split()[0] if ' ' in tk else tk
                            tk = _normalize_ticker_str(tk)
                            if tk and len(tk) <= 6 and not tk.isdigit():
                                out.append({'Date': pd.to_datetime(d), 'Ticker': tk, 'Action': action})

        if not out:
            return pd.DataFrame()
        res = pd.DataFrame(out).drop_duplicates().sort_values('Date')
        return res
    except Exception as e:
        print(f"Error NDX Wikipedia: {e}")
        return pd.DataFrame()

def get_nasdaq100_historical_changes():
    local_csv_path = "ndx_changes.csv"
    fallback_path = os.path.join(DATA_DIR, "ndx_changes.csv")

    changes_df = pd.DataFrame()
    loaded_from_local = False

    for csv_path in [local_csv_path, fallback_path]:
        if os.path.exists(csv_path):
            try:
                changes_df = pd.read_csv(csv_path, parse_dates=['Date'])
                if 'Ticker' in changes_df.columns:
                    changes_df['Ticker'] = changes_df['Ticker'].astype(str).str.upper().str.replace('.', '-', regex=False)
                loaded_from_local = True
                break
            except Exception as e:
                print(f"Error leyendo {csv_path}: {e}")

    try:
        if loaded_from_local:
            last_date = pd.to_datetime(changes_df['Date'], errors='coerce').max()
            if pd.isna(last_date) or (datetime.now() - last_date.to_pydatetime()).days > 7:
                wikipedia_df = download_nasdaq100_changes_from_wikipedia()
                if not wikipedia_df.empty:
                    merged = pd.concat([changes_df, wikipedia_df], ignore_index=True)
                    merged = merged.drop_duplicates(subset=['Date', 'Ticker', 'Action']).sort_values('Date')
                    try:
                        merged.to_csv(local_csv_path, index=False)
                    except Exception:
                        pass
                    return merged
        else:
            wikipedia_df = download_nasdaq100_changes_from_wikipedia()
            if not wikipedia_df.empty:
                try:
                    wikipedia_df.to_csv(local_csv_path, index=False)
                except Exception:
                    pass
                return wikipedia_df
    except Exception as e:
        print(f"Error actualizando NDX desde Wikipedia: {e}")

    return changes_df if not changes_df.empty else pd.DataFrame()

# -------------------------
# Descarga de precios
# -------------------------

def download_prices_parallel(tickers, start_date, end_date, load_full_data=True, max_workers=10):
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

    prices_data = {}
    ohlc_data = {}

    def load_single_ticker(ticker):
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
        for future in as_completed(futures):
            ticker, price, ohlc = future.result()
            if price is not None:
                prices_data[ticker] = price
            if ohlc is not None:
                ohlc_data[ticker] = ohlc

    if prices_data:
        prices_df = pd.DataFrame(prices_data)
        prices_df = prices_df.interpolate(method='linear', limit_direction='both')
        return prices_df, ohlc_data
    else:
        return pd.DataFrame(), {}

def download_prices(tickers, start_date, end_date, load_full_data=True):
    prices_df, ohlc_data = download_prices_parallel(
        tickers, start_date, end_date,
        load_full_data=load_full_data,
        max_workers=10
    )
    if load_full_data:
        return prices_df, ohlc_data
    else:
        return prices_df

# -------------------------
# Universo con validación
# -------------------------

def get_constituents_at_date(index_name, start_date, end_date):
    cache_key = get_cache_key(index_name, start_date, end_date)

    if cache_key in _constituents_cache:
        return _constituents_cache[cache_key], None

    cached_data = load_cache(cache_key, prefix="constituents", max_age_days=7)
    if cached_data is not None:
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
    Gate: constituyentes actuales hoy ∩ CSVs en data/.
    Regla: si aparece en changes -> Added <= selección y NO Removed < selección.
           si no aparece en changes -> se asume histórico (incluido).
    index_name: 'SP500', 'NDX' o 'Ambos (SP500 + NDX)'
    """
    try:
        def _norm_t(t):
            return str(t).strip().upper().replace('.', '-')

        def _month_end(d):
            return (pd.Timestamp(d) + pd.offsets.MonthEnd(0)).date()

        selection_date = _month_end(end_date)

        # CSVs disponibles
        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        available = set()
        for p in csv_files:
            fn = os.path.basename(p)
            if not fn.endswith(".csv"):
                continue
            tk = _norm_t(fn[:-4])
            if tk and len(tk) <= 6 and not tk.isdigit() and tk not in {'SPY', 'QQQ', 'IEF', 'BIL'}:
                available.add(tk)

        # Constituyentes actuales
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

        candidates = sorted(current_tickers & available)

        # Sin changes válidos -> todos incluidos como históricos
        if changes is None or changes.empty or not {'Date', 'Ticker', 'Action'}.issubset(set(changes.columns)):
            details = []
            for t in candidates:
                details.append({
                    'ticker': t,
                    'in_current': True,
                    'in_changes': False,
                    'eligible_on': selection_date.isoformat(),
                    'status': 'Incluido (sin changes: se asume histórico)'
                })
            return {
                'tickers': candidates,
                'data': details,
                'historical_data_available': False,
                'note': f'Sin changes válidos. Selección: {selection_date.isoformat()}'
            }, None

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
                    'status': 'Incluido (sin registros en changes: histórico)'
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

# -------------------------
# Utilidades varias
# -------------------------

def generate_removed_tickers_summary():
    cache_key = "removed_tickers_summary"
    cached_data = load_cache(cache_key, max_age_days=30)
    if cached_data is not None:
        return cached_data
    df = pd.DataFrame()
    save_cache(cache_key, df)
    return df

def clear_cache():
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR)
    print("Caché limpiado")

def get_cache_size():
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(CACHE_DIR):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)


# =========================
# ===== backtest.py =====
# =========================

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calcular_atr_optimizado(high, low, close, periods=14):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/periods, adjust=False).mean()
    return atr

def convertir_a_mensual_con_ohlc(ohlc_data):
    monthly_data = {}
    for ticker, data in ohlc_data.items():
        try:
            df = pd.DataFrame({
                'High': data['High'],
                'Low': data['Low'],
                'Close': data['Close']
            })
            df_monthly = df.resample('ME').agg({
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            })
            monthly_data[ticker] = {
                'High': df_monthly['High'],
                'Low': df_monthly['Low'],
                'Close': df_monthly['Close']
            }
        except Exception:
            continue
    return monthly_data

def precalculate_valid_tickers_by_date(monthly_dates, historical_changes_data, current_tickers):
    """
    EXACTO a la regla:
    - Universo SIEMPRE = current_tickers (constituyentes actuales hoy).
    - Para cada fecha y ticker actual:
        * Si aparece en changes: incluir si (existe Added <= fecha) y (NO existe Removed < fecha).
        * Si NO aparece en changes: se asume histórico y se incluye siempre.
    """
    def _norm_t(t):
        return str(t).strip().upper().replace('.', '-')
    current_set = set(_norm_t(t) for t in current_tickers if t)

    if historical_changes_data is None or historical_changes_data.empty:
        return {date: set(current_set) for date in monthly_dates}

    ch = historical_changes_data.copy()
    if 'Date' not in ch.columns or 'Ticker' not in ch.columns or 'Action' not in ch.columns:
        return {date: set(current_set) for date in monthly_dates}

    ch['Ticker'] = ch['Ticker'].astype(str).str.upper().str.replace('.', '-', regex=False)
    ch['Date'] = pd.to_datetime(ch['Date'], errors='coerce')
    ch = ch.dropna(subset=['Date', 'Ticker', 'Action'])
    ch['Action'] = ch['Action'].astype(str).str.lower()

    changes_by_ticker = {}
    for t, tdf in ch.groupby('Ticker'):
        add_dates = sorted(pd.to_datetime(tdf.loc[tdf['Action'].str.contains('add', na=False), 'Date']).dt.normalize())
        rem_dates = sorted(pd.to_datetime(tdf.loc[tdf['Action'].str.contains('remov', na=False), 'Date']).dt.normalize())
        changes_by_ticker[t] = {'add': add_dates, 'rem': rem_dates}

    valid_by_date = {}
    for target_date in monthly_dates:
        td = pd.to_datetime(target_date).normalize()
        valids = set()
        for t in current_set:
            info = changes_by_ticker.get(t)
            if info is None:
                valids.add(t)
                continue
            added_before_or_on = any(d <= td for d in info['add'])
            removed_before = any(d < td for d in info['rem'])
            if added_before_or_on and not removed_before:
                valids.add(t)
        valid_by_date[target_date] = valids

    return valid_by_date

def precalculate_all_indicators(prices_df_m, ohlc_data, corte=680):
    try:
        all_indicators = {}
        monthly_ohlc = convertir_a_mensual_con_ohlc(ohlc_data) if ohlc_data else None

        for ticker in prices_df_m.columns:
            try:
                ticker_data = prices_df_m[ticker].dropna()
                if len(ticker_data) < 15:
                    continue
                if monthly_ohlc and ticker in monthly_ohlc:
                    high = monthly_ohlc[ticker]['High']
                    low = monthly_ohlc[ticker]['Low']
                    close = monthly_ohlc[ticker]['Close']
                else:
                    close = ticker_data
                    if getattr(close.index, 'freq', None) not in ['ME', 'M']:
                        close = close.resample('ME').last()
                    monthly_returns = close.pct_change().dropna()
                    vol = monthly_returns.rolling(3).std().fillna(0.02).clip(0.005, 0.03)
                    high = close * (1 + vol * 0.5)
                    low = close * (1 - vol * 0.5)
                close = close.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                high = high.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                low = low.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                if close.isna().all():
                    continue
                roc_10 = ((close - close.shift(10)) / close.shift(10)) * 100
                f1 = roc_10 * 0.6
                atr_14 = calcular_atr_optimizado(high, low, close, periods=14)
                sma_14 = close.rolling(14).mean()
                volatility_ratio = atr_14 / sma_14
                f2 = volatility_ratio * 0.4
                inercia_alcista = f1 / f2
                score = np.where(inercia_alcista >= corte, inercia_alcista, 0)
                score = pd.Series(score, index=inercia_alcista.index)
                score_adjusted = score / atr_14
                inercia_alcista = inercia_alcista.replace([np.inf, -np.inf], np.nan).fillna(0)
                score = score.replace([np.inf, -np.inf], np.nan).fillna(0)
                score_adjusted = score_adjusted.replace([np.inf, -np.inf], np.nan).fillna(0)
                all_indicators[ticker] = {
                    'InerciaAlcista': inercia_alcista,
                    'Score': score,
                    'ScoreAdjusted': score_adjusted,
                    'ATR14': atr_14,
                    'ROC10': roc_10
                }
            except Exception:
                continue
        return all_indicators
    except Exception:
        return {}

def inertia_score(monthly_prices_df, corte=680, ohlc_data=None):
    try:
        if monthly_prices_df is None or monthly_prices_df.empty:
            return {}
        results = {}
        monthly_ohlc = convertir_a_mensual_con_ohlc(ohlc_data) if ohlc_data else None

        for ticker in monthly_prices_df.columns:
            try:
                ticker_data = monthly_prices_df[ticker].dropna()
                if len(ticker_data) < 15:
                    continue
                if monthly_ohlc and ticker in monthly_ohlc:
                    high = monthly_ohlc[ticker]['High']
                    low = monthly_ohlc[ticker]['Low']
                    close = monthly_ohlc[ticker]['Close']
                else:
                    close = ticker_data
                    if getattr(close.index, 'freq', None) not in ['ME', 'M']:
                        close = close.resample('ME').last()
                    monthly_returns = close.pct_change().dropna()
                    vol = monthly_returns.rolling(3).std().fillna(0.02).clip(0.005, 0.03)
                    high = close * (1 + vol * 0.5)
                    low = close * (1 - vol * 0.5)
                    high = pd.Series(np.maximum(high, close), index=close.index)
                    low = pd.Series(np.minimum(low, close), index=close.index)
                if len(close) < 15:
                    continue
                close = close.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                high = high.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                low = low.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                if close.isna().all() or high.isna().all() or low.isna().all():
                    continue
                roc_10 = ((close - close.shift(10)) / close.shift(10)) * 100
                f1 = roc_10 * 0.6
                atr_14 = calcular_atr_optimizado(high, low, close, periods=14)
                sma_14 = close.rolling(14).mean()
                volatility_ratio = atr_14 / sma_14
                f2 = volatility_ratio * 0.4
                inercia_alcista = f1 / f2
                score = np.where(inercia_alcista >= corte, inercia_alcista, 0)
                score = pd.Series(score, index=inercia_alcista.index)
                score_adjusted = score / atr_14
                inercia_alcista = inercia_alcista.replace([np.inf, -np.inf], np.nan).fillna(0)
                score = score.replace([np.inf, -np.inf], np.nan).fillna(0)
                score_adjusted = score_adjusted.replace([np.inf, -np.inf], np.nan).fillna(0)
                if (inercia_alcista == 0).all() and (score == 0).all() and (score_adjusted == 0).all():
                    continue
                results[ticker] = {
                    "InerciaAlcista": inercia_alcista,
                    "ATR14": atr_14,
                    "Score": score,
                    "ScoreAdjusted": score_adjusted,
                    "F1": f1,
                    "F2": f2,
                    "ROC10": roc_10,
                    "VolatilityRatio": volatility_ratio
                }
            except Exception:
                continue

        if not results:
            return {}

        combined_results = {}
        for metric in ["InerciaAlcista", "ATR14", "Score", "ScoreAdjusted", "F1", "F2", "ROC10", "VolatilityRatio"]:
            metric_data = {}
            for ticker in results.keys():
                if metric in results[ticker] and results[ticker][metric] is not None:
                    metric_data[ticker] = results[ticker][metric]
            if metric_data:
                combined_results[metric] = pd.DataFrame(metric_data)
        return combined_results
    except Exception:
        return {}

def run_backtest_optimized(prices, benchmark, commission=0.003, top_n=10, corte=680, 
                          ohlc_data=None, historical_info=None, fixed_allocation=False,
                          use_roc_filter=False, use_sma_filter=False, spy_data=None,
                          progress_callback=None,
                          use_safety_etfs=False, safety_prices=None, safety_ohlc=None,
                          safety_tickers=('IEF','BIL'),
                          avoid_rebuy_unchanged=True):
    try:
        if prices is None or prices.empty:
            return pd.DataFrame(), pd.DataFrame()

        if isinstance(prices, pd.Series):
            prices_m = prices.resample('ME').last()
            prices_df_m = pd.DataFrame({'Close': prices_m})
        else:
            prices_df_m = prices.resample('ME').last()

        if isinstance(benchmark, pd.Series):
            bench_m = benchmark.resample('ME').last()
        else:
            bench_m = benchmark.resample('ME').last()

        spy_monthly = None
        if (use_roc_filter or use_sma_filter) and spy_data is not None:
            spy_series = spy_data.iloc[:, 0] if isinstance(spy_data, pd.DataFrame) else spy_data
            spy_monthly = spy_series.resample('ME').last()

        safety_prices_m = None
        monthly_safety_ohlc = None
        if use_safety_etfs and safety_prices is not None:
            if isinstance(safety_prices, pd.Series):
                safety_prices = safety_prices.to_frame()
            if isinstance(safety_prices, pd.DataFrame) and not safety_prices.empty:
                safety_prices_m = safety_prices.resample('ME').last()
                monthly_safety_ohlc = convertir_a_mensual_con_ohlc(safety_ohlc) if safety_ohlc else None

        def compute_raw_inercia_and_score_at(prev_dt, close_s, high_s=None, low_s=None):
            try:
                close = close_s.copy().dropna()
                if getattr(close.index, 'freq', None) not in ['ME', 'M']:
                    close = close.resample('ME').last()
                close = close.loc[:prev_dt]
                if len(close) < 15:
                    return np.nan, np.nan
                if high_s is not None and low_s is not None:
                    high = high_s.copy().dropna().resample('ME').max().loc[close.index]
                    low = low_s.copy().dropna().resample('ME').min().loc[close.index]
                else:
                    monthly_returns = close.pct_change().dropna()
                    vol = monthly_returns.rolling(3).std().fillna(0.02).clip(0.005, 0.03)
                    high = close * (1 + vol * 0.5)
                    low = close * (1 - vol * 0.5)
                    high = pd.Series(np.maximum(high, close), index=close.index)
                    low = pd.Series(np.minimum(low, close), index=close.index)
                high = high.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                low = low.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                close = close.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                roc_10 = ((close - close.shift(10)) / close.shift(10)) * 100
                f1 = roc_10 * 0.6
                atr_14 = calcular_atr_optimizado(high, low, close, periods=14)
                sma_14 = close.rolling(14).mean()
                volatility_ratio = atr_14 / sma_14
                f2 = volatility_ratio * 0.4
                inercia_alcista = f1 / f2
                if prev_dt not in inercia_alcista.index or prev_dt not in atr_14.index:
                    return np.nan, np.nan
                inercia_val = float(inercia_alcista.loc[prev_dt])
                atr_val = float(atr_14.loc[prev_dt])
                score_adj_raw = (inercia_val / atr_val) if atr_val and atr_val != 0 else np.nan
                return inercia_val, score_adj_raw
            except Exception:
                return np.nan, np.nan

        all_indicators = precalculate_all_indicators(prices_df_m, ohlc_data, corte)
        current_tickers = list(prices_df_m.columns)
        monthly_dates = prices_df_m.index[1:]

        if historical_info and 'changes_data' in historical_info:
            valid_tickers_by_date = precalculate_valid_tickers_by_date(
                monthly_dates,
                historical_info['changes_data'],
                current_tickers
            )
        else:
            valid_tickers_by_date = {date: set(current_tickers) for date in monthly_dates}

        equity = [10000.0]
        dates = [prices_df_m.index[0]]
        picks_list = []
        total_months = len(prices_df_m) - 1

        prev_selected_set = set()
        last_mode = 'none'
        last_safety_ticker = None

        for i in range(1, len(prices_df_m)):
            try:
                prev_date = prices_df_m.index[i - 1]
                date = prices_df_m.index[i]

                if progress_callback and i % 10 == 0:
                    progress_callback(i / max(1, total_months))

                market_filter_active = False
                if spy_monthly is not None and prev_date in spy_monthly.index:
                    spy_price = spy_monthly.loc[prev_date]
                    if use_roc_filter and len(spy_monthly[:prev_date]) >= 13:
                        spy_12m_ago = spy_monthly[:prev_date].iloc[-13]
                        spy_roc_12m = ((spy_price - spy_12m_ago) / spy_12m_ago) * 100
                        if spy_roc_12m < 0:
                            market_filter_active = True
                    if use_sma_filter and len(spy_monthly[:prev_date]) >= 10:
                        spy_sma_10m = spy_monthly[:prev_date].iloc[-10:].mean()
                        if spy_price < spy_sma_10m:
                            market_filter_active = True

                valid_tickers_for_date = valid_tickers_by_date.get(prev_date, set(current_tickers))
                if use_safety_etfs and safety_tickers:
                    valid_tickers_for_date = set(valid_tickers_for_date) - set(safety_tickers)

                if market_filter_active:
                    if use_safety_etfs and safety_prices_m is not None and not safety_prices_m.empty:
                        candidates = []
                        for st in safety_tickers:
                            if st not in safety_prices_m.columns:
                                continue
                            try:
                                if prev_date in safety_prices_m.index and date in safety_prices_m.index:
                                    pr0 = safety_prices_m.loc[prev_date, st]
                                    pr1 = safety_prices_m.loc[date, st]
                                    if pd.notna(pr0) and pd.notna(pr1) and pr0 != 0:
                                        ret_1m = (pr1 / pr0) - 1
                                        if monthly_safety_ohlc and st in monthly_safety_ohlc:
                                            high_s = monthly_safety_ohlc[st]['High']
                                            low_s = monthly_safety_ohlc[st]['Low']
                                            close_s = monthly_safety_ohlc[st]['Close']
                                        else:
                                            high_s = low_s = None
                                            close_s = safety_prices_m[st]
                                        inercia_val, score_adj_raw = compute_raw_inercia_and_score_at(prev_date, close_s, high_s, low_s)
                                        if pd.isna(score_adj_raw):
                                            score_adj_raw = ret_1m
                                        if pd.isna(inercia_val):
                                            inercia_val = 0.0
                                        candidates.append({
                                            'ticker': st,
                                            'ret': float(ret_1m),
                                            'inercia': float(inercia_val),
                                            'score_adj': float(score_adj_raw)
                                        })
                            except Exception:
                                continue
                        if candidates:
                            candidates = sorted(candidates, key=lambda x: x['score_adj'], reverse=True)
                            best = candidates[0]
                            commission_effect = commission
                            if avoid_rebuy_unchanged and last_mode == 'safety' and last_safety_ticker == best['ticker']:
                                commission_effect = 0.0
                            portfolio_return = best['ret'] - commission_effect
                            new_equity = equity[-1] * (1 + portfolio_return)
                            equity.append(new_equity)
                            dates.append(date)
                            picks_list.append({
                                "Date": prev_date.strftime("%Y-%m-%d"),
                                "Rank": 1,
                                "Ticker": best['ticker'],
                                "Inercia": best['inercia'],
                                "ScoreAdj": best['score_adj'],
                                "HistoricallyValid": True
                            })
                            last_mode = 'safety'
                            last_safety_ticker = best['ticker']
                            prev_selected_set = {best['ticker']}
                            continue
                    equity.append(equity[-1])
                    dates.append(date)
                    last_mode = 'safety'
                    last_safety_ticker = None
                    prev_selected_set = set()
                    continue

                candidates = []
                for ticker in valid_tickers_for_date:
                    if ticker not in all_indicators:
                        continue
                    indicators = all_indicators[ticker]
                    try:
                        if prev_date in indicators['InerciaAlcista'].index:
                            inercia = indicators['InerciaAlcista'].loc[prev_date]
                            score_adj = indicators['ScoreAdjusted'].loc[prev_date]
                            if inercia >= corte and score_adj > 0 and not np.isnan(score_adj):
                                candidates.append({
                                    'ticker': ticker,
                                    'inercia': float(inercia),
                                    'score_adj': float(score_adj)
                                })
                    except Exception:
                        continue
                if not candidates:
                    equity.append(equity[-1])
                    dates.append(date)
                    last_mode = 'normal'
                    continue
                candidates = sorted(candidates, key=lambda x: x['score_adj'], reverse=True)
                selected_picks = candidates[:(10 if fixed_allocation else top_n)]
                selected_tickers = [p['ticker'] for p in selected_picks]

                available_prices = prices_df_m.loc[date]
                prev_prices = prices_df_m.loc[prev_date]
                valid_tickers = []
                ticker_returns = []
                for ticker in selected_tickers:
                    if (ticker in available_prices.index and
                        ticker in prev_prices.index and
                        not pd.isna(available_prices[ticker]) and
                        not pd.isna(prev_prices[ticker]) and
                        prev_prices[ticker] != 0):
                        valid_tickers.append(ticker)
                        ret = (available_prices[ticker] / prev_prices[ticker]) - 1
                        ticker_returns.append(ret)
                if not valid_tickers:
                    equity.append(equity[-1])
                    dates.append(date)
                    last_mode = 'normal'
                    continue

                if fixed_allocation:
                    valid_tickers = valid_tickers[:10]
                    ticker_returns = ticker_returns[:10]
                    weight = 0.1
                else:
                    weight = 1.0 / len(valid_tickers)

                portfolio_return = sum(r * weight for r in ticker_returns)

                commission_effect = commission
                if avoid_rebuy_unchanged:
                    if last_mode != 'normal':
                        commission_effect = commission
                    else:
                        new_set = set(valid_tickers)
                        old_set = set(prev_selected_set)
                        entries = len(new_set - old_set)
                        exits = len(old_set - new_set)
                        turnover_ratio = (entries + exits) / max(1, len(new_set))
                        commission_effect = commission * turnover_ratio

                portfolio_return -= commission_effect
                new_equity = equity[-1] * (1 + portfolio_return)
                equity.append(new_equity)
                dates.append(date)

                for rank, ticker in enumerate(valid_tickers, 1):
                    pick_data = next((p for p in selected_picks if p['ticker'] == ticker), None)
                    if pick_data:
                        picks_list.append({
                            "Date": prev_date.strftime("%Y-%m-%d"),
                            "Rank": rank,
                            "Ticker": ticker,
                            "Inercia": pick_data['inercia'],
                            "ScoreAdj": pick_data['score_adj'],
                            "HistoricallyValid": ticker in valid_tickers_for_date
                        })
                prev_selected_set = set(valid_tickers)
                last_mode = 'normal'
                last_safety_ticker = None
            except Exception:
                equity.append(equity[-1])
                dates.append(date)
                continue

        equity_series = pd.Series(equity, index=dates)
        returns = equity_series.pct_change().fillna(0)
        drawdown = (equity_series / equity_series.cummax() - 1).fillna(0)
        bt_results = pd.DataFrame({
            "Equity": equity_series,
            "Returns": returns,
            "Drawdown": drawdown
        })
        picks_df = pd.DataFrame(picks_list) if picks_list else pd.DataFrame()
        return bt_results, picks_df
    except Exception:
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

def calculate_monthly_returns_by_year(equity_series):
    try:
        if equity_series is None or len(equity_series) < 2:
            return pd.DataFrame()
        monthly_returns = equity_series.pct_change().fillna(0)
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_by_year = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).apply(lambda x: (1 + x).prod() - 1)
        years = sorted(monthly_by_year.index.get_level_values(0).unique())
        months_es = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        table_data = []
        for year in years:
            year_data = {'Año': year}
            year_monthly = monthly_by_year[monthly_by_year.index.get_level_values(0) == year]
            for i, month_abbr in enumerate(months_es, 1):
                if i in year_monthly.index.get_level_values(1):
                    return_value = year_monthly[year_monthly.index.get_level_values(1) == i].iloc[0]
                    year_data[month_abbr] = f"{return_value*100:.1f}%"
                else:
                    year_data[month_abbr] = "-"
            year_equity = equity_series[equity_series.index.year == year]
            if len(year_equity) > 1:
                ytd_return = (year_equity.iloc[-1] / year_equity.iloc[0]) - 1
                year_data['YTD'] = f"{ytd_return*100:.1f}%"
            else:
                year_data['YTD'] = "-"
            table_data.append(year_data)
        if table_data:
            result_df = pd.DataFrame(table_data)
            columns_order = ['Año'] + months_es + ['YTD']
            result_df = result_df[columns_order]
            return result_df
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    risk_free_rate_monthly = (1 + risk_free_rate) ** (1/12) - 1
    excess_returns = returns - risk_free_rate_monthly
    if excess_returns.std() > 0:
        sharpe = (excess_returns.mean() * 12) / (excess_returns.std() * np.sqrt(12))
    else:
        sharpe = 0
    return sharpe

def calcular_atr_amibroker(*args, **kwargs):
    return calcular_atr_optimizado(*args, **kwargs)

def run_backtest(*args, **kwargs):
    return run_backtest_optimized(*args, **kwargs)


# =========================
# ===== app.py =====
# =========================

# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import os
import base64
import pickle
import hashlib
import glob
import requests
from io import StringIO

# Importar módulos propios (imports en líneas separadas para evitar problemas de sintaxis)
from data_loader import get_constituents_at_date
from data_loader import get_sp500_historical_changes
from data_loader import get_nasdaq100_historical_changes
from data_loader import generate_removed_tickers_summary

from backtest import run_backtest_optimized
from backtest import precalculate_all_indicators
from backtest import calculate_monthly_returns_by_year
from backtest import inertia_score
from backtest import calculate_sharpe_ratio

# =========================
# Utilidades y configuración
# =========================

def check_historical_files():
    files_to_check = [
        "sp500_changes.csv",
        "ndx_changes.csv",
        "data/sp500_changes.csv",
        "data/ndx_changes.csv",
    ]
    found_files = []
    for file_path in files_to_check:
        if os.path.exists(file_path):
            found_files.append(file_path)
            try:
                df = pd.read_csv(file_path)
                print(f"✅ Encontrado: {file_path} ({len(df)} registros)")
            except Exception as e:
                print(f"⚠️ Error leyendo {file_path}: {e}")
    if not found_files:
        print("⚠️ No se encontraron archivos de cambios históricos")
        print("📁 Archivos esperados: sp500_changes.csv, ndx_changes.csv")
    return found_files

historical_files = check_historical_files()

st.set_page_config(page_title="IA Mensual Ajustada", page_icon="📈", layout="wide")

# Estado
if "backtest_completed" not in st.session_state:
    st.session_state.backtest_completed = False
if "bt_results" not in st.session_state:
    st.session_state.bt_results = None
if "picks_df" not in st.session_state:
    st.session_state.picks_df = None
if "spy_df" not in st.session_state:
    st.session_state.spy_df = None
if "prices_df" not in st.session_state:
    st.session_state.prices_df = None
if "benchmark_series" not in st.session_state:
    st.session_state.benchmark_series = None
if "ohlc_data" not in st.session_state:
    st.session_state.ohlc_data = None
if "historical_info" not in st.session_state:
    st.session_state.historical_info = None
if "backtest_params" not in st.session_state:
    st.session_state.backtest_params = None
if "universe_tickers" not in st.session_state:
    st.session_state.universe_tickers = set()
if "robust_cache" not in st.session_state:
    st.session_state.robust_cache = {}

# =========================
# Caches
# =========================

@st.cache_data(ttl=3600)
def load_historical_changes_cached(index_name, force_reload=False):
    if force_reload:
        st.cache_data.clear()
    if index_name == "SP500":
        return get_sp500_historical_changes()
    elif index_name == "NDX":
        return get_nasdaq100_historical_changes()
    else:
        sp500 = get_sp500_historical_changes()
        ndx = get_nasdaq100_historical_changes()
        if not sp500.empty and not ndx.empty:
            return pd.concat([sp500, ndx], ignore_index=True)
        return sp500 if not sp500.empty else ndx

@st.cache_data(ttl=86400)
def get_constituents_cached(index_name, start_date, end_date):
    return get_constituents_at_date(index_name, start_date, end_date)

def get_cache_key(params):
    param_str = str(sorted(params.items()))
    return hashlib.md5(param_str.encode()).hexdigest()

@st.cache_data(ttl=3600 * 24 * 7)
def load_prices_from_csv_parallel(tickers, start_date, end_date, load_full_data=True):
    prices_data = {}
    ohlc_data = {}
    def load_single_ticker(ticker):
        clean_ticker = str(ticker).strip().upper().replace(".", "-")
        csv_path = f"data/{clean_ticker}.csv"
        if not os.path.exists(csv_path):
            return ticker, None, None
        try:
            df = pd.read_csv(
                csv_path,
                index_col="Date",
                parse_dates=True,
                date_parser=lambda col: pd.to_datetime(col, utc=True).tz_convert(None),
            )
            start_filter = start_date if not isinstance(start_date, datetime) else start_date.date()
            end_filter = end_date if not isinstance(end_date, datetime) else end_date.date()
            df = df[(df.index.date >= start_filter) & (df.index.date <= end_filter)]
            if df.empty:
                return ticker, None, None
            if "Adj Close" in df.columns:
                price = df["Adj Close"]
            elif "Close" in df.columns:
                price = df["Close"]
            else:
                return ticker, None, None
            ohlc = None
            if load_full_data and all(col in df.columns for col in ["High", "Low", "Close"]):
                ohlc = {
                    "High": df["High"],
                    "Low": df["Low"],
                    "Close": df["Adj Close"] if "Adj Close" in df.columns else df["Close"],
                    "Volume": df["Volume"] if "Volume" in df.columns else None,
                }
            return ticker, price, ohlc
        except Exception as e:
            print(f"⚠️ Error leyendo {ticker}: {e}")
            return ticker, None, None
    from concurrent.futures import ThreadPoolExecutor
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
        prices_df = prices_df.fillna(method="ffill").fillna(method="bfill")
        return prices_df, ohlc_data
    else:
        return pd.DataFrame(), {}

# =========================
# Lectura robusta Wikipedia
# =========================

def _read_html_with_ua(url, attrs=None):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/122.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        html = resp.text
        return pd.read_html(StringIO(html), attrs=attrs)
    except Exception as e:
        print(f"read_html UA error ({url}): {e}")
        return []

@st.cache_data(ttl=3600 * 12)
def get_sp500_name_map():
    tables = _read_html_with_ua("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", attrs={"id": "constituents"})
    df = tables[0] if tables else None
    if df is None:
        tables = _read_html_with_ua("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if any(c in ("symbol", "ticker") for c in cols) and any(c in ("security", "company", "company name") for c in cols):
                df = t
                break
    if df is None:
        return {}
    sym_col = next((c for c in df.columns if str(c).strip().lower() in ("symbol", "ticker")), None)
    sec_col = next((c for c in df.columns if str(c).strip().lower() in ("security", "company", "company name")), None)
    if not sym_col or not sec_col:
        return {}
    s = df[[sym_col, sec_col]].copy()
    s.columns = ["Symbol", "Name"]
    s["Symbol"] = s["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
    return dict(zip(s["Symbol"], s["Name"]))

@st.cache_data(ttl=3600 * 12)
def get_ndx_name_map():
    tables = _read_html_with_ua("https://en.wikipedia.org/wiki/Nasdaq-100", attrs={"id": "constituents"})
    df = tables[0] if tables else None
    if df is None:
        tables = _read_html_with_ua("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if any(("ticker" in c or "symbol" in c) for c in cols) and any("company" in c for c in cols):
                df = t
                break
    if df is None:
        return {}
    sym_col = next((c for c in df.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()), None)
    name_col = next((c for c in df.columns if "company" in str(c).lower()), None)
    if not sym_col or not name_col:
        return {}
    s = df[[sym_col, name_col]].copy()
    s.columns = ["Symbol", "Name"]
    s["Symbol"] = s["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
    return dict(zip(s["Symbol"], s["Name"]))

@st.cache_data(ttl=3600)
def get_name_map_for_index(index_choice):
    if index_choice == "SP500":
        return get_sp500_name_map()
    elif index_choice == "NDX":
        return get_ndx_name_map()
    else:
        m = get_sp500_name_map()
        m2 = get_ndx_name_map()
        m.update(m2)
        return m

def normalize_symbol(t):
    return str(t).strip().upper().replace(".", "-")

# =========================
# Filtros y helpers de UI
# =========================

def is_filter_active_for_next_month(spy_df, use_roc, use_sma):
    try:
        if spy_df is None or spy_df.empty or "SPY" not in spy_df.columns:
            return False
        spy_m = spy_df["SPY"].resample("ME").last().dropna()
        if len(spy_m) < 15:
            return False
        prev_date = spy_m.index[-1]
        price = spy_m.iloc[-1]
        active = False
        if use_roc and len(spy_m) >= 13:
            spy_12m_ago = spy_m.iloc[-13]
            roc_12m = ((price - spy_12m_ago) / spy_12m_ago) * 100 if spy_12m_ago != 0 else 0
            if roc_12m < 0:
                active = True
        if use_sma and len(spy_m) >= 10:
            sma_10 = spy_m.iloc[-10:].mean()
            if price < sma_10:
                active = True
        return active
    except Exception:
        return False

def get_current_constituents_set(index_choice):
    if index_choice == "SP500":
        return set(map(normalize_symbol, get_sp500_name_map().keys()))
    elif index_choice == "NDX":
        return set(map(normalize_symbol, get_ndx_name_map().keys()))
    else:
        s1 = set(map(normalize_symbol, get_sp500_name_map().keys()))
        s2 = set(map(normalize_symbol, get_ndx_name_map().keys()))
        return s1 | s2

# =========================
# Sidebar
# =========================

st.title("📈 Estrategia mensual sobre los componentes del S&P 500 y/o Nasdaq-100")

st.sidebar.header("Parámetros de backtest")
index_choice = st.sidebar.selectbox("Selecciona el índice:", ["SP500", "NDX", "Ambos (SP500 + NDX)"])

try:
    default_end = min(datetime.today().date(), datetime(2030, 12, 31).date())
    default_start = default_end - timedelta(days=365 * 5)
    end_date = st.sidebar.date_input("Fecha final", value=default_end, min_value=datetime(1950, 1, 1).date(), max_value=datetime(2030, 12, 31).date())
    start_date = st.sidebar.date_input("Fecha inicial", value=default_start, min_value=datetime(1950, 1, 1).date(), max_value=datetime(2030, 12, 31).date())
    if start_date >= end_date:
        st.sidebar.warning("⚠️ Fecha inicial debe ser anterior a la fecha final")
        start_date = end_date - timedelta(days=365 * 2)
    st.sidebar.info(f"📅 Rango: {start_date} a {end_date}")
except Exception as e:
    st.sidebar.error(f"❌ Error configurando fechas: {e}")
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=365 * 5)

# top_n por defecto = 5
top_n = st.sidebar.slider("Número de activos", 5, 30, 5)
commission = st.sidebar.number_input("Comisión por operación (%)", 0.0, 1.0, 0.3) / 100
corte = st.sidebar.number_input("Corte de score", 0, 1000, 680)
use_historical_verification = st.sidebar.checkbox("🕐 Usar verificación histórica", value=True)

st.sidebar.subheader("⚙️ Opciones de Estrategia")
fixed_allocation = st.sidebar.checkbox("💰 Asignar 10% capital a cada acción", value=False)
avoid_rebuy_unchanged = st.sidebar.checkbox("⛽ No recomprar picks que se mantienen (evitar comisiones)", value=True)

st.sidebar.subheader("🛡️ Filtros de Mercado")
use_roc_filter = st.sidebar.checkbox("📉 ROC 12 meses del SPY < 0", value=False)
use_sma_filter = st.sidebar.checkbox("📊 Precio SPY < SMA 10 meses", value=False)
use_safety_etfs = st.sidebar.checkbox("🛡️ Usar IEF/BIL cuando el filtro mande a cash", value=False)

run_button = st.sidebar.button("🏃 Ejecutar backtest", type="primary")

# Limpiar
if st.session_state.backtest_completed:
    if st.sidebar.button("🗑️ Limpiar resultados", type="secondary"):
        st.session_state.backtest_completed = False
        st.session_state.bt_results = None
        st.session_state.picks_df = None
        st.session_state.spy_df = None
        st.session_state.prices_df = None
        st.session_state.benchmark_series = None
        st.session_state.ohlc_data = None
        st.session_state.historical_info = None
        st.session_state.backtest_params = None
        st.session_state.universe_tickers = set()
        st.session_state.robust_cache = {}
        st.rerun()

# Constantes
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# =========================
# Main
# =========================

if run_button:
    st.session_state.backtest_completed = False
    try:
        cache_params = {
            "index": index_choice,
            "start": str(start_date),
            "end": str(end_date),
            "top_n": top_n,
            "corte": corte,
            "commission": commission,
            "historical": use_historical_verification,
            "fixed_alloc": fixed_allocation,
            "roc_filter": use_roc_filter,
            "sma_filter": use_sma_filter,
            "use_safety_etfs": use_safety_etfs,
            "avoid_rebuy_unchanged": avoid_rebuy_unchanged,
        }
        cache_key = get_cache_key(cache_params)
        cache_file = os.path.join(CACHE_DIR, f"backtest_{cache_key}.pkl")

        use_cache = False
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                    if st.sidebar.checkbox("🔄 Usar resultados en caché", value=True):
                        use_cache = True
                        st.success("✅ Cargando resultados desde caché...")
                        bt_results = cached_data["bt_results"]
                        picks_df = cached_data["picks_df"]
                        historical_info = cached_data.get("historical_info")
                        prices_df = cached_data.get("prices_df")
                        ohlc_data = cached_data.get("ohlc_data")
                        benchmark_series = cached_data.get("benchmark_series")
                        spy_df = cached_data.get("spy_df")
                        st.session_state.universe_tickers = set(cached_data.get("universe_tickers", []))
            except Exception:
                use_cache = False

        if not use_cache:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("📥 Obteniendo constituyentes...")
            progress_bar.progress(10)
            all_tickers_data, error = get_constituents_cached(index_name=index_choice, start_date=start_date, end_date=end_date)
            if error:
                st.warning(f"Advertencia: {error}")
            if not all_tickers_data or "tickers" not in all_tickers_data:
                st.error("No se encontraron tickers válidos")
                st.stop()
            tickers = list(dict.fromkeys(all_tickers_data["tickers"]))
            st.session_state.universe_tickers = set(tickers)
            st.success(f"✅ Obtenidos {len(tickers)} tickers únicos")

            status_text.text("📊 Cargando precios en paralelo...")
            progress_bar.progress(30)
            prices_df, ohlc_data = load_prices_from_csv_parallel(tickers, start_date, end_date, load_full_data=True)
            if prices_df.empty:
                st.error("❌ No se pudieron cargar precios")
                st.stop()
            st.success(f"✅ Cargados {len(prices_df.columns)} tickers con datos")

            status_text.text("📈 Cargando benchmark...")
            progress_bar.progress(40)
            benchmark_ticker = "SPY" if index_choice != "NDX" else "QQQ"
            benchmark_df, _ = load_prices_from_csv_parallel([benchmark_ticker], start_date, end_date, load_full_data=False)
            if benchmark_df.empty:
                st.warning("Usando promedio como benchmark")
                benchmark_series = prices_df.mean(axis=1)
            else:
                benchmark_series = benchmark_df[benchmark_ticker]

            spy_df = None
            status_text.text("📈 Cargando SPY...")
            spy_result, _ = load_prices_from_csv_parallel(["SPY"], start_date, end_date, load_full_data=False)
            if not spy_result.empty and "SPY" in spy_result.columns:
                spy_df = spy_result
                st.sidebar.success(f"✅ SPY cargado: {len(spy_df)} registros")
            else:
                st.sidebar.warning("⚠️ No se pudo cargar SPY")
                spy_df = None

            historical_info = None
            if use_historical_verification:
                status_text.text("🕐 Cargando datos históricos...")
                progress_bar.progress(50)
                changes_data = load_historical_changes_cached(index_name=index_choice)
                if not changes_data.empty:
                    historical_info = {"changes_data": changes_data, "has_historical_data": True}
                    st.success(f"✅ Cargados {len(changes_data)} cambios históricos")
                else:
                    st.warning("⚠️ No se encontraron datos históricos, continuando sin verificación")
                    historical_info = None

            safety_prices = pd.DataFrame()
            safety_ohlc = {}
            if use_safety_etfs:
                status_text.text("🛡️ Cargando ETFs de refugio (IEF, BIL)...")
                safety_prices, safety_ohlc = load_prices_from_csv_parallel(["IEF", "BIL"], start_date, end_date, load_full_data=True)

            status_text.text("🚀 Ejecutando backtest optimizado...")
            progress_bar.progress(70)
            bt_results, picks_df = run_backtest_optimized(
                prices=prices_df,
                benchmark=benchmark_series,
                commission=commission,
                top_n=top_n,
                corte=corte,
                ohlc_data=ohlc_data,
                historical_info=historical_info,
                fixed_allocation=fixed_allocation,
                use_roc_filter=use_roc_filter,
                use_sma_filter=use_sma_filter,
                spy_data=spy_df,
                progress_callback=lambda p: progress_bar.progress(70 + int(p * 0.3)),
                use_safety_etfs=use_safety_etfs,
                safety_prices=safety_prices,
                safety_ohlc=safety_ohlc,
                avoid_rebuy_unchanged=avoid_rebuy_unchanged,
            )

            st.session_state.bt_results = bt_results
            st.session_state.picks_df = picks_df
            st.session_state.spy_df = spy_df

            prices_df_display = prices_df.copy()
            if use_safety_etfs and not safety_prices.empty:
                prices_df_display = prices_df_display.join(safety_prices, how="outer")

            st.session_state.prices_df = prices_df_display
            st.session_state.benchmark_series = benchmark_series
            st.session_state.ohlc_data = ohlc_data
            st.session_state.historical_info = historical_info
            st.session_state.backtest_params = cache_params
            st.session_state.backtest_completed = True

            status_text.text("💾 Guardando resultados en caché...")
            progress_bar.progress(100)
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(
                        {
                            "bt_results": bt_results,
                            "picks_df": picks_df,
                            "historical_info": historical_info,
                            "prices_df": prices_df_display,
                            "ohlc_data": ohlc_data,
                            "benchmark_series": benchmark_series,
                            "spy_df": spy_df,
                            "universe_tickers": list(st.session_state.universe_tickers),
                            "timestamp": datetime.now(),
                        },
                        f,
                    )
                st.success("✅ Resultados guardados en caché")
            except Exception as e:
                st.warning(f"No se pudo guardar caché: {e}")
            status_text.empty()
            progress_bar.empty()
        else:
            st.session_state.bt_results = bt_results
            st.session_state.picks_df = picks_df
            st.session_state.spy_df = spy_df
            st.session_state.prices_df = prices_df
            st.session_state.benchmark_series = benchmark_series
            st.session_state.ohlc_data = ohlc_data
            st.session_state.historical_info = historical_info
            st.session_state.backtest_params = cache_params
            st.session_state.backtest_completed = True

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

# =========================
# Mostrar resultados
# =========================

if st.session_state.backtest_completed and st.session_state.bt_results is not None:
    bt_results = st.session_state.bt_results
    picks_df = st.session_state.picks_df
    spy_df = st.session_state.spy_df
    prices_df = st.session_state.prices_df
    benchmark_series = st.session_state.benchmark_series
    ohlc_data = st.session_state.ohlc_data
    historical_info = st.session_state.historical_info
    universe_tickers = st.session_state.universe_tickers or set()

    use_roc_filter = st.session_state.backtest_params.get("roc_filter", False) if st.session_state.backtest_params else False
    use_sma_filter = st.session_state.backtest_params.get("sma_filter", False) if st.session_state.backtest_params else False
    index_choice = st.session_state.backtest_params.get("index", "SP500") if st.session_state.backtest_params else "SP500"
    fixed_allocation = st.session_state.backtest_params.get("fixed_alloc", False) if st.session_state.backtest_params else False
    corte = st.session_state.backtest_params.get("corte", 680) if st.session_state.backtest_params else 680
    top_n = st.session_state.backtest_params.get("top_n", 5) if st.session_state.backtest_params else 5
    commission = st.session_state.backtest_params.get("commission", 0.003) if st.session_state.backtest_params else 0.003
    use_safety_etfs = st.session_state.backtest_params.get("use_safety_etfs", False) if st.session_state.backtest_params else False
    avoid_rebuy_unchanged = st.session_state.backtest_params.get("avoid_rebuy_unchanged", True) if st.session_state.backtest_params else True

    st.success("✅ Backtest completado exitosamente")

    final_equity = float(bt_results["Equity"].iloc[-1])
    initial_equity = float(bt_results["Equity"].iloc[0])
    total_return = (final_equity / initial_equity) - 1
    years = (bt_results.index[-1] - bt_results.index[0]).days / 365.25
    cagr = (final_equity / initial_equity) ** (1 / years) - 1 if years > 0 else 0
    max_drawdown = float(bt_results["Drawdown"].min())

    monthly_returns = bt_results["Returns"]
    risk_free_rate_annual = 0.02
    risk_free_rate_monthly = (1 + risk_free_rate_annual) ** (1 / 12) - 1
    excess_returns = monthly_returns - risk_free_rate_monthly
    if len(monthly_returns) > 0 and excess_returns.std() > 0:
        sharpe_ratio = (excess_returns.mean() * 12) / (excess_returns.std() * np.sqrt(12))
    else:
        sharpe_ratio = 0

    st.subheader("📊 Métricas de la Estrategia")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Equity Final", f"${final_equity:,.0f}")
    col2.metric("Retorno Total", f"{total_return:.2%}")
    col3.metric("CAGR", f"{cagr:.2%}")
    col4.metric("Max Drawdown (Mensual)", f"{max_drawdown:.2%}")
    col5.metric("Sharpe Ratio (RF=2%)", f"{sharpe_ratio:.2f}")

    bench_final = initial_equity
    bench_total_return = 0.0
    bench_cagr = 0.0
    bench_max_dd = 0.0
    bench_sharpe = 0.0
    bench_equity_m = None
    bench_drawdown_m = None
    if benchmark_series is not None and not benchmark_series.empty:
        try:
            bench_prices_m = benchmark_series.resample("ME").last()
            bench_prices_aligned = bench_prices_m.reindex(bt_results.index)
            valid_idx = bench_prices_aligned.dropna().index
            if len(valid_idx) > 1:
                bench_returns_valid = bench_prices_aligned.loc[valid_idx].pct_change().fillna(0)
                bench_equity_valid = initial_equity * (1 + bench_returns_valid).cumprod()
                bench_final = float(bench_equity_valid.iloc[-1])
                bench_initial_valid = float(bench_equity_valid.iloc[0])
                bench_total_return = (bench_final / bench_initial_valid) - 1
                bench_years = (valid_idx[-1] - valid_idx[0]).days / 365.25
                bench_cagr = (bench_final / bench_initial_valid) ** (1 / bench_years) - 1 if bench_years > 0 else 0
                bench_drawdown_valid = (bench_equity_valid / bench_equity_valid.cummax()) - 1
                bench_max_dd = float(bench_drawdown_valid.min())
                bench_excess_returns = bench_returns_valid - risk_free_rate_monthly
                bench_sharpe = (bench_excess_returns.mean() * 12) / (bench_excess_returns.std() * np.sqrt(12)) if bench_excess_returns.std() != 0 else 0
                bench_equity_m = bench_equity_valid.reindex(bt_results.index).ffill()
                bench_drawdown_m = bench_drawdown_valid.reindex(bt_results.index).ffill()
        except Exception as e:
            st.warning(f"Error calculando benchmark: {e}")

    benchmark_name = "SPY" if index_choice != "NDX" else "QQQ"
    st.subheader(f"📊 Métricas del Benchmark ({benchmark_name})")
    col1b, col2b, col3b, col4b, col5b = st.columns(5)
    col1b.metric("Equity Final", f"${bench_final:,.0f}")
    col2b.metric("Retorno Total", f"{bench_total_return:.2%}")
    col3b.metric("CAGR", f"{bench_cagr:.2%}")
    col4b.metric("Max Drawdown (Mensual)", f"{bench_max_dd:.2%}")
    col5b.metric("Sharpe Ratio (RF=2%)", f"{bench_sharpe:.2f}")

    st.subheader("📈 Gráficos de Rentabilidad")
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3],
        subplot_titles=("Evolución del Equity", "Drawdown")
    )
    fig.add_trace(
        go.Scatter(x=bt_results.index, y=bt_results["Equity"], mode="lines", name="Estrategia", line=dict(width=3, color="blue")),
        row=1, col=1,
    )
    if bench_equity_m is not None:
        bench_aligned = bench_equity_m.reindex(bt_results.index).ffill()
        bench_line = dict(width=2.5, color="#50C878") if (index_choice != "NDX") else dict(width=2.5, color="#9467BD")
        fig.add_trace(
            go.Scatter(x=bench_aligned.index, y=bench_aligned.values, mode="lines", name=f'Benchmark ({("SPY" if index_choice != "NDX" else "QQQ")})', line=bench_line),
            row=1, col=1,
        )
    if "Drawdown" in bt_results.columns:
        fig.add_trace(
            go.Scatter(x=bt_results.index, y=bt_results["Drawdown"] * 100, mode="lines", name="DD Estrategia", fill="tozeroy", line=dict(color="red", width=2)),
            row=2, col=1,
        )
        if bench_drawdown_m is not None:
            bench_dd_aligned = (bench_drawdown_m.reindex(bt_results.index).ffill() * 100)
            bench_dd_line = dict(color="#FF7F0E", width=2) if (index_choice != "NDX") else dict(color="#9467BD", width=2)
            fig.add_trace(
                go.Scatter(x=bench_dd_aligned.index, y=bench_dd_aligned.values, mode="lines", name=f'DD {("SPY" if index_choice != "NDX" else "QQQ")}', line=bench_dd_line),
                row=2, col=1,
            )
    st.plotly_chart(fig)

    st.subheader("📅 RENDIMIENTOS MENSUALES POR AÑO")
    monthly_table = calculate_monthly_returns_by_year(bt_results["Equity"])
    if not monthly_table.empty:
        def style_returns(val):
            if val == "-" or val == "":
                return ""
            try:
                num = float(val.rstrip("%"))
                if num > 0:
                    return "background-color: #d4edda; color: #155724; font-weight: bold"
                elif num < 0:
                    return "background-color: #f8d7da; color: #721c24; font-weight: bold"
                return ""
            except Exception:
                return ""
        styled_table = monthly_table.style.applymap(style_returns)
        st.dataframe(styled_table)

    if picks_df is not None and not picks_df.empty:
        st.subheader("📊 Picks Históricos")
        col_sidebar, col_main = st.columns([1, 3])

        with col_sidebar:
            st.markdown("### 📅 Navegación por Fechas")
            unique_dates = sorted(picks_df["Date"].unique(), reverse=True)
            selected_date = st.selectbox("Selecciona una fecha:", unique_dates, index=0, key="historical_date_selector")
            date_picks = picks_df[picks_df["Date"] == selected_date]
            st.info(f"🎯 {len(date_picks)} picks seleccionados el {selected_date}")

            monthly_return = 0.0
            try:
                selected_dt = pd.Timestamp(selected_date)
                bt_index = pd.to_datetime(bt_results.index)
                if selected_dt in bt_index:
                    current_idx = bt_index.get_loc(selected_dt)
                    if current_idx < len(bt_index) - 1:
                        monthly_return = bt_results["Returns"].iloc[current_idx + 1]
                        st.metric("📈 Retorno del Mes", f"{monthly_return:.2%}", delta=f"{monthly_return:.2%}" if monthly_return != 0 else None)
                        with st.expander("🔍 Desglose del Retorno"):
                            st.write(f"Fecha picks: {selected_date}")
                            st.write(f"Equity inicial: ${bt_results['Equity'].iloc[current_idx]:,.2f}")
                            st.write(f"Equity final: ${bt_results['Equity'].iloc[current_idx + 1]:,.2f}")
                            st.write(f"Retorno neto: {monthly_return:.4%}")
                            st.write(f"Comisión (parámetro): -{commission:.2%}")
                            st.write(f"Retorno bruto estimado: {monthly_return + commission:.4%}")
                else:
                    closest_idx = bt_index.get_indexer([selected_dt], method="nearest")[0]
                    if closest_idx < len(bt_index) - 1:
                        monthly_return = bt_results["Returns"].iloc[closest_idx + 1]
                        st.metric("📈 Retorno del Mes (aprox)", f"{monthly_return:.2%}", delta=f"{monthly_return:.2%}" if monthly_return != 0 else None)
                        st.caption(f"⚠️ Fecha aproximada: {bt_index[closest_idx].strftime('%Y-%m-%d')}")
            except Exception as e:
                st.warning(f"No se pudo calcular retorno del mes: {e}")

        with col_main:
            st.markdown(f"### 🎯 Picks Seleccionados el {selected_date}")
            date_picks_display = date_picks.copy()
            try:
                selected_dt = pd.Timestamp(selected_date)
                returns_data = []
                for _, row in date_picks.iterrows():
                    ticker = row["Ticker"]
                    try:
                        if ticker in prices_df.columns:
                            ticker_monthly = prices_df[ticker].resample("ME").last()
                            if selected_dt in ticker_monthly.index:
                                idx = ticker_monthly.index.get_loc(selected_dt)
                                if idx < len(ticker_monthly) - 1:
                                    entry_price = ticker_monthly.iloc[idx]
                                    exit_price = ticker_monthly.iloc[idx + 1]
                                    ret = (exit_price / entry_price) - 1 if (pd.notna(entry_price) and pd.notna(exit_price) and entry_price != 0) else None
                                    returns_data.append(ret)
                                else:
                                    returns_data.append(None)
                            else:
                                returns_data.append(None)
                        else:
                            returns_data.append(None)
                    except Exception:
                        returns_data.append(None)
                date_picks_display["Retorno Individual"] = returns_data
                def fmt_ret(x):
                    if pd.isna(x):
                        return "N/A"
                    return f"{x:+.2%}"
                st.dataframe(
                    date_picks_display[["Rank", "Ticker", "Inercia", "ScoreAdj", "Retorno Individual"]].style.format(
                        {"Inercia": "{:.2f}", "ScoreAdj": "{:.2f}", "Retorno Individual": fmt_ret}
                    )
                )
            except Exception as e:
                st.error(f"Error calculando retornos individuales: {e}")

    with st.expander("🔮 Señales Actuales - Vela en Formación", expanded=True):
        st.subheader("📊 Picks Prospectivos para el Próximo Mes")
        try:
            if prices_df is not None and not prices_df.empty:
                if index_choice == "Ambos (SP500 + NDX)":
                    name_map = get_sp500_name_map()
                    tmp = get_ndx_name_map()
                    name_map.update(tmp)
                else:
                    name_map = get_name_map_for_index(index_choice)

                safety_set = {"IEF", "BIL"}
                filter_active = is_filter_active_for_next_month(spy_df, use_roc_filter, use_sma_filter) if (use_roc_filter or use_sma_filter) else False

                current_constituents_set = get_current_constituents_set(index_choice)
                valid_universe = [t for t in prices_df.columns if (t in current_constituents_set) and (t not in safety_set)]

                if use_safety_etfs and filter_active:
                    safety_prices_now, _ = load_prices_from_csv_parallel(list(safety_set), bt_results.index[0].date(), bt_results.index[-1].date(), load_full_data=False)
                    if safety_prices_now.empty:
                        st.warning("⚠️ No hay datos de IEF/BIL para señales actuales.")
                    else:
                        try:
                            safety_ind = inertia_score(safety_prices_now, corte=corte, ohlc_data=None)
                        except Exception:
                            safety_ind = {}
                        score_df = safety_ind.get("ScoreAdjusted")
                        inercia_df = safety_ind.get("InerciaAlcista")
                        last_scores = score_df.iloc[-1] if score_df is not None and not score_df.empty else pd.Series(dtype=float)
                        last_inercia = inercia_df.iloc[-1] if inercia_df is not None and not inercia_df.empty else pd.Series(dtype=float)
                        safety_name_map = {
                            "IEF": "iShares 7-10 Year Treasury Bond ETF",
                            "BIL": "SPDR Bloomberg 1-3 Month T-Bill ETF",
                        }
                        candidates = []
                        for stkr in safety_set:
                            if stkr in safety_prices_now.columns:
                                price_now = float(safety_prices_now[stkr].iloc[-1]) if pd.notna(safety_prices_now[stkr].iloc[-1]) else 0.0
                                sc = float(last_scores.get(stkr)) if stkr in last_scores.index else 0.0
                                ine = float(last_inercia.get(stkr)) if stkr in last_inercia.index else 0.0
                                candidates.append(
                                    {
                                        "Rank": 1,
                                        "Ticker": stkr,
                                        "Nombre": safety_name_map.get(stkr, stkr),
                                        "Inercia Alcista": ine,
                                        "Score Ajustado": sc,
                                        "Precio Actual": price_now,
                                    }
                                )
                        if candidates:
                            candidates = sorted(candidates, key=lambda x: (x["Score Ajustado"], x["Inercia Alcista"]), reverse=True)
                            df_pros = pd.DataFrame([candidates[0]])
                            df_pros["Rank"] = range(1, len(df_pros) + 1)
                            st.dataframe(df_pros)
                            if picks_df is not None and not picks_df.empty:
                                last_date = max(picks_df["Date"])
                                prev_set = set(picks_df[picks_df["Date"] == last_date]["Ticker"].tolist())
                                new_set = set(df_pros["Ticker"].tolist())
                                entran = sorted(new_set - prev_set)
                                salen = sorted(prev_set - new_set)
                                se_mantienen = sorted(prev_set & new_set)
                                c1, c2, c3 = st.columns(3)
                                c1.markdown("### ✅ ENTRAN")
                                c1.write(", ".join(entran) if entran else "-")
                                c2.markdown("### ❌ SALEN")
                                c2.write(", ".join(salen) if salen else "-")
                                c3.markdown("### ♻️ SE MANTIENEN")
                                c3.write(", ".join(se_mantienen) if se_mantienen else "-")
                        else:
                            st.warning("⚠️ No hay candidatos válidos de IEF/BIL ahora mismo.")
                else:
                    if not valid_universe:
                        st.warning("⚠️ No hay tickers válidos para generar señales")
                    else:
                        prices_for_signals = prices_df[valid_universe]
                        ohlc_for_signals = {k: v for k, v in (ohlc_data or {}).items() if k in valid_universe}
                        current_scores = inertia_score(prices_for_signals, corte=corte, ohlc_data=ohlc_for_signals)
                        if current_scores and "ScoreAdjusted" in current_scores and "InerciaAlcista" in current_scores:
                            score_df = current_scores["ScoreAdjusted"]
                            inercia_df = current_scores["InerciaAlcista"]
                            if not score_df.empty and not inercia_df.empty:
                                last_scores = score_df.iloc[-1].dropna()
                                last_inercia = inercia_df.iloc[-1]
                                valid_picks = []
                                for ticker in last_scores.index:
                                    if ticker in last_inercia.index:
                                        inercia_val = last_inercia[ticker]
                                        score_adj = last_scores[ticker]
                                        if inercia_val >= corte and score_adj > 0 and not np.isnan(score_adj):
                                            name = name_map.get(ticker)
                                            if not name and index_choice != "SP500":
                                                name = get_sp500_name_map().get(ticker)
                                            if not name and index_choice != "NDX":
                                                name = get_ndx_name_map().get(ticker)
                                            if not name:
                                                name = ticker
                                            valid_picks.append(
                                                {
                                                    "ticker": ticker,
                                                    "name": name,
                                                    "inercia": float(inercia_val),
                                                    "score_adj": float(score_adj),
                                                    "price": float(prices_for_signals[ticker].iloc[-1]),
                                                }
                                            )
                                if valid_picks:
                                    valid_picks = sorted(valid_picks, key=lambda x: x["score_adj"], reverse=True)
                                    final_picks = valid_picks[: min(top_n, len(valid_picks))]
                                    df_pros = pd.DataFrame(
                                        [
                                            {
                                                "Rank": i + 1,
                                                "Ticker": p["ticker"],
                                                "Nombre": p["name"],
                                                "Inercia Alcista": p["inercia"],
                                                "Score Ajustado": p["score_adj"],
                                                "Precio Actual": p["price"],
                                            }
                                            for i, p in enumerate(final_picks)
                                        ]
                                    )
                                    st.dataframe(df_pros)
                                    if picks_df is not None and not picks_df.empty:
                                        last_date = max(picks_df["Date"])
                                        prev_set = set(picks_df[picks_df["Date"] == last_date]["Ticker"].tolist())
                                        new_set = set(df_pros["Ticker"].tolist())
                                        entran = sorted(new_set - prev_set)
                                        salen = sorted(prev_set - new_set)
                                        se_mantienen = sorted(prev_set & new_set)
                                        c1, c2, c3 = st.columns(3)
                                        c1.markdown("### ✅ ENTRAN")
                                        c1.write(", ".join(entran) if entran else "-")
                                        c2.markdown("### ❌ SALEN")
                                        c2.write(", ".join(salen) if salen else "-")
                                        c3.markdown("### ♻️ SE MANTIENEN")
                                        c3.write(", ".join(se_mantienen) if se_mantienen else "-")
                                else:
                                    st.warning("⚠️ No hay tickers que pasen el corte de inercia actualmente")
                            else:
                                st.warning("⚠️ No hay suficientes datos para calcular señales")
                        else:
                            st.error("❌ No se pudieron calcular indicadores. Verifica los datos.")
            else:
                st.error("❌ No hay datos de precios disponibles para calcular señales actuales")
        except Exception as e:
            st.error(f"Error calculando señales actuales: {str(e)}")

    st.subheader("🧪 Robustez por número de posiciones (3 → 10)")
    do_robust = st.checkbox(
        "Calcular matriz de robustez 3-10 posiciones (CAGR y Max DD)",
        value=False,
        help="Ejecuta backtests para top_n = 3..10 con los mismos parámetros actuales"
    )

    if do_robust:
        with st.spinner("Calculando robustez..."):
            results = []
            safety_prices_k = pd.DataFrame()
            safety_ohlc_k = {}
            if use_safety_etfs:
                sd = pd.to_datetime(st.session_state.backtest_params.get("start")).date()
                ed = pd.to_datetime(st.session_state.backtest_params.get("end")).date()
                safety_prices_k, safety_ohlc_k = load_prices_from_csv_parallel(['IEF','BIL'], sd, ed, load_full_data=True)

            base_params_key = get_cache_key({
                "index": index_choice, "start": str(st.session_state.backtest_params.get("start")),
                "end": str(st.session_state.backtest_params.get("end")),
                "corte": corte, "commission": commission,
                "historical": bool(historical_info),
                "fixed_alloc": fixed_allocation,
                "roc_filter": use_roc_filter, "sma_filter": use_sma_filter,
                "use_safety_etfs": use_safety_etfs,
                "avoid_rebuy_unchanged": avoid_rebuy_unchanged
            })

            for k in range(3, 11):
                key_k = f"{base_params_key}_k{k}"
                if key_k in st.session_state.robust_cache:
                    bt_k = st.session_state.robust_cache[key_k]
                else:
                    bt_k, _ = run_backtest_optimized(
                        prices=prices_df,
                        benchmark=benchmark_series,
                        commission=commission,
                        top_n=k,
                        corte=corte,
                        ohlc_data=ohlc_data,
                        historical_info=historical_info,
                        fixed_allocation=fixed_allocation,
                        use_roc_filter=use_roc_filter,
                        use_sma_filter=use_sma_filter,
                        spy_data=spy_df,
                        progress_callback=None,
                        use_safety_etfs=use_safety_etfs,
                        safety_prices=safety_prices_k if not safety_prices_k.empty else None,
                        safety_ohlc=safety_ohlc_k if safety_ohlc_k else None,
                        avoid_rebuy_unchanged=avoid_rebuy_unchanged
                    )
                    st.session_state.robust_cache[key_k] = bt_k
                if not bt_k.empty:
                    eq = bt_k["Equity"]
                    years_k = (eq.index[-1] - eq.index[0]).days / 365.25 if len(eq) > 1 else 0
                    cagr_k = (eq.iloc[-1] / eq.iloc[0]) ** (1/years_k) - 1 if years_k > 0 else 0
                    maxdd_k = float(bt_k["Drawdown"].min()) if "Drawdown" in bt_k.columns else 0.0
                    results.append({"Posiciones": k, "CAGR": cagr_k, "Max DD": maxdd_k})

            if results:
                rob_df = pd.DataFrame(results).set_index("Posiciones")
                rob_show = rob_df.copy()
                rob_show["CAGR"] = (rob_show["CAGR"]*100).map(lambda x: f"{x:.2f}%")
                rob_show["Max DD"] = (rob_show["Max DD"]*100).map(lambda x: f"{x:.2f}%")
                st.dataframe(rob_show)
            else:
                st.warning("No se pudieron calcular resultados de robustez")

if st.session_state.spy_df is not None:
    st.sidebar.success(f"✅ SPY en memoria: {len(st.session_state.spy_df)} registros")
else:
    if not st.session_state.backtest_completed:
        st.info("👈 Configura los parámetros y haz clic en 'Ejecutar backtest'")
        st.subheader("🔍 Información del Sistema")
        st.info(
            "- ✅ Verificación histórica de constituyentes\n"
            "- ✅ Cálculos optimizados con precálculo y paralelización\n"
            "- ✅ Caché multinivel para carga instantánea\n"
            "- ✅ Comparación completa con benchmark\n"
            "- ✅ Tabla de rendimientos mensuales\n"
            "- ✅ Señales actuales con vela en formación\n"
            "- ✅ Filtros de mercado configurables\n"
            "- ✅ Evitar recompras cuando se mantienen en Top (opcional)\n"
        )
        cache_files = glob.glob(os.path.join(CACHE_DIR, "backtest_*.pkl"))
        if cache_files:
            st.info(f"💾 {len(cache_files)} resultados en caché disponibles para carga instantánea")
