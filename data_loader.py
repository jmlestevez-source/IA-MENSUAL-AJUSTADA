# =========================
# ===== data_loader.py =====
# =========================

# -*- coding: utf-8 -*-
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

# Cachés en memoria
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
        return pd.read_html(StringIO(resp.text), attrs=attrs)
    except Exception as e:
        print(f"_read_html_with_ua error ({url}): {e}")
        return []

def parse_wikipedia_date(date_str):
    if pd.isna(date_str) or not date_str or str(date_str).lower() in ['nan', 'none', '']:
        return None
    s = str(date_str).strip()
    try:
        return parser.parse(s, fuzzy=True).date()
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
            clean = re.sub(r'[^\w\s,]', ' ', s.lower())
            parts = clean.split()
            if len(parts) >= 3:
                month = None
                for part in parts:
                    p = part.replace(',', '')
                    if p in month_map:
                        month = month_map[p]
                        break
                nums = [int(re.findall(r'\d+', p)[0]) for p in parts if re.findall(r'\d+', p)]
                if month and len(nums) >= 2:
                    day = min(nums)
                    year = max(nums)
                    if day <= 31 and year >= 1900:
                        return date(year, month, day)
            return None
        except Exception:
            return None

def _normalize_ticker_str(x):
    return str(x).strip().upper().replace('.', '-')

# -------------------------
# Constituyentes actuales (Wikipedia)
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

    cached = load_cache(cache_key, prefix="constituents", max_age_days=1)
    if cached is not None:
        return cached

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
# Cambios históricos (Wikipedia/CSV)
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
        prices_df = pd.DataFrame(prices_data).interpolate(method='linear', limit_direction='both')
        return prices_df, ohlc_data
    else:
        return pd.DataFrame(), {}

def download_prices(tickers, start_date, end_date, load_full_data=True):
    prices_df, ohlc_data = download_prices_parallel(
        tickers, start_date, end_date,
        load_full_data=load_full_data,
        max_workers=10
    )
    return (prices_df, ohlc_data) if load_full_data else prices_df

# -------------------------
# Universo con validación exacta
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
        return None, f"Error obteniendo constituyentes para {index_name}: {e}"

def _normalize_changes_df(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date', 'Ticker', 'Action'])
    out = df.copy()
    out['Ticker'] = out['Ticker'].astype(str).str.upper().str.replace('.', '-', regex=False)
    out['Date'] = pd.to_datetime(out['Date'], errors='coerce').dt.date
    out = out.dropna(subset=['Date', 'Ticker', 'Action'])
    out['Action'] = out['Action'].astype(str).str.lower()
    return out

def _included_by_last_event(tdf, selection_date):
    tdf_before = tdf[pd.to_datetime(tdf['Date']) <= pd.to_datetime(selection_date)]
    if tdf_before.empty:
        return False
    last_action = str(tdf_before.sort_values('Date').iloc[-1]['Action']).lower()
    if 'add' in last_action:
        return True
    if 'remov' in last_action:
        return False
    return False

def get_all_available_tickers_with_historical_validation(index_name, start_date, end_date):
    """
    Gate: constituyentes actuales hoy ∩ CSVs en data/.
    Regla exacta por fecha de selección (cierre del mes de end_date):
      - Si el ticker NO aparece en changes del índice relevante -> se asume histórico (incluido).
      - Si aparece en changes -> el ÚLTIMO evento con fecha <= selección decide:
            * Added -> incluido
            * Removed -> excluido
      - En 'Ambos': evaluar por índice correspondiente y usar OR (si está incluido en cualquiera de los dos, se incluye).
    """
    try:
        def _norm_t(t):
            return str(t).strip().upper().replace('.', '-')

        selection_date = (pd.Timestamp(end_date) + pd.offsets.MonthEnd(0)).date()

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

        idx = str(index_name).strip().upper()

        if idx.startswith("AMBOS") or idx in {"BOTH", "SP500+NDX", "ALL", "AMBOS (SP500 + NDX)"}:
            sp_list = get_current_constituents("SP500").get('tickers', [])
            nd_list = get_current_constituents("NDX").get('tickers', [])
            sp_curr = set(map(_norm_t, sp_list))
            nd_curr = set(map(_norm_t, nd_list))
            current_tickers = sp_curr | nd_curr

            sp_ch = _normalize_changes_df(get_sp500_historical_changes())
            nd_ch = _normalize_changes_df(get_nasdaq100_historical_changes())
            sp_grp = sp_ch.groupby('Ticker') if not sp_ch.empty else {}
            nd_grp = nd_ch.groupby('Ticker') if not nd_ch.empty else {}

            candidates = sorted(current_tickers & available)

            included = []
            details = []
            for t in candidates:
                in_sp = t in sp_curr
                in_nd = t in nd_curr

                inc_sp = None
                inc_nd = None

                if in_sp:
                    if isinstance(sp_grp, dict) or t not in sp_grp.groups:
                        inc_sp = True  # no hay registros -> histórico
                    else:
                        tdf = sp_grp.get_group(t)
                        inc_sp = _included_by_last_event(tdf, selection_date)

                if in_nd:
                    if isinstance(nd_grp, dict) or t not in nd_grp.groups:
                        inc_nd = True
                    else:
                        tdf = nd_grp.get_group(t)
                        inc_nd = _included_by_last_event(tdf, selection_date)

                is_included = (inc_sp is True) or (inc_nd is True)
                if is_included:
                    included.append(t)

                details.append({
                    'ticker': t,
                    'in_current': True,
                    'indices': ('SP500' if in_sp else '') + (',NDX' if in_nd else ''),
                    'eligible_on': selection_date.isoformat(),
                    'status': 'Incluido (último evento <= selección: Added)' if is_included else 'Excluido (último evento <= selección: Removed o sin Added previo)'
                })

            return {
                'tickers': included,
                'data': details,
                'historical_data_available': True,
                'note': f'Ambos índices | Regla último evento <= selección | Selección: {selection_date.isoformat()}'
            }, None

        elif idx in {"SP500", "S&P500", "S&P 500"}:
            current_tickers = set(map(_norm_t, get_current_constituents("SP500").get('tickers', [])))
            changes = _normalize_changes_df(get_sp500_historical_changes())
            grp = changes.groupby('Ticker') if not changes.empty else {}

            candidates = sorted(current_tickers & available)
            included = []
            details = []

            for t in candidates:
                if isinstance(grp, dict) or t not in grp.groups:
                    included.append(t)
                    status = 'Incluido (sin registros en changes: histórico)'
                else:
                    tdf = grp.get_group(t)
                    ok = _included_by_last_event(tdf, selection_date)
                    if ok:
                        included.append(t)
                        status = 'Incluido (último evento <= selección: Added)'
                    else:
                        status = 'Excluido (último evento <= selección: Removed o sin Added previo)'
                details.append({
                    'ticker': t,
                    'in_current': True,
                    'eligible_on': selection_date.isoformat(),
                    'status': status
                })

            return {
                'tickers': included,
                'data': details,
                'historical_data_available': True,
                'note': f'SP500 | Regla último evento <= selección | Selección: {selection_date.isoformat()}'
            }, None

        elif idx in {"NDX", "NASDAQ-100", "NASDAQ100"}:
            current_tickers = set(map(_norm_t, get_current_constituents("NDX").get('tickers', [])))
            changes = _normalize_changes_df(get_nasdaq100_historical_changes())
            grp = changes.groupby('Ticker') if not changes.empty else {}

            candidates = sorted(current_tickers & available)
            included = []
            details = []

            for t in candidates:
                if isinstance(grp, dict) or t not in grp.groups:
                    included.append(t)
                    status = 'Incluido (sin registros en changes: histórico)'
                else:
                    tdf = grp.get_group(t)
                    ok = _included_by_last_event(tdf, selection_date)
                    if ok:
                        included.append(t)
                        status = 'Incluido (último evento <= selección: Added)'
                    else:
                        status = 'Excluido (último evento <= selección: Removed o sin Added previo)'
                details.append({
                    'ticker': t,
                    'in_current': True,
                    'eligible_on': selection_date.isoformat(),
                    'status': status
                })

            return {
                'tickers': included,
                'data': details,
                'historical_data_available': True,
                'note': f'NDX | Regla último evento <= selección | Selección: {selection_date.isoformat()}'
            }, None

        else:
            raise ValueError(f"Índice {index_name} no soportado")

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

def _adaptive_roc(close, max_period=10, min_period=3):
    close = close.copy().astype(float)
    roc = pd.Series(index=close.index, dtype=float)
    for i in range(len(close)):
        if i >= min_period:
            n = min(max_period, i)
            prev = close.iloc[i - n]
            cur = close.iloc[i]
            if pd.notna(prev) and prev != 0 and pd.notna(cur):
                roc.iloc[i] = ((cur - prev) / prev) * 100.0
            else:
                roc.iloc[i] = np.nan
        else:
            roc.iloc[i] = np.nan
    return roc

def precalculate_valid_tickers_by_date(monthly_dates, historical_changes_data, current_tickers):
    """
    Universo SIEMPRE = current_tickers (constituyentes actuales hoy).
    Para cada fecha (prev_date) y ticker actual:
      - Si NO aparece en changes: se asume histórico y se incluye siempre.
      - Si aparece en changes: el ÚLTIMO evento con fecha <= fecha manda:
            * Added -> incluido
            * Removed -> excluido
      - Si no hay evento <= fecha (solo futuros) -> excluido.
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

    grouped = ch.groupby('Ticker')

    valid_by_date = {}
    for target_date in monthly_dates:
        td = pd.to_datetime(target_date).normalize()
        valids = set()
        for t in current_set:
            if t not in grouped.groups:
                valids.add(t)
                continue
            tdf = grouped.get_group(t)
            tdf_before = tdf[tdf['Date'] <= td]
            if tdf_before.empty:
                continue
            last_action = str(tdf_before.sort_values('Date').iloc[-1]['Action']).lower()
            if 'add' in last_action:
                valids.add(t)
        valid_by_date[target_date] = valids

    return valid_by_date

def precalculate_all_indicators(prices_df_m, ohlc_data, corte=680):
    """
    Precalcula TODOS los indicadores de una vez con ventanas adaptativas para no esperar 14 meses.
    """
    try:
        all_indicators = {}
        monthly_ohlc = convertir_a_mensual_con_ohlc(ohlc_data) if ohlc_data else None

        for ticker in prices_df_m.columns:
            try:
                ticker_data = prices_df_m[ticker].dropna()
                if ticker_data.empty:
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
                    vol = monthly_returns.rolling(3, min_periods=1).std().fillna(0.02).clip(0.005, 0.03)
                    high = close * (1 + vol * 0.5)
                    low = close * (1 - vol * 0.5)

                close = close.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                high = high.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                low = low.replace([np.inf, -np.inf], np.nan).ffill().bfill()

                roc_10 = _adaptive_roc(close, max_period=10, min_period=3)
                f1 = roc_10 * 0.6

                atr_14 = calcular_atr_optimizado(high, low, close, periods=14)
                sma_14 = close.rolling(14, min_periods=3).mean()
                volatility_ratio = atr_14 / sma_14

                f2 = volatility_ratio * 0.4
                inercia_alcista = f1 / f2

                score = pd.Series(np.where(inercia_alcista >= corte, inercia_alcista, 0), index=inercia_alcista.index)
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
    """
    Calcula el score de inercia (adaptativo).
    """
    try:
        if monthly_prices_df is None or monthly_prices_df.empty:
            return {}
        results = {}
        monthly_ohlc = convertir_a_mensual_con_ohlc(ohlc_data) if ohlc_data else None

        for ticker in monthly_prices_df.columns:
            try:
                ticker_data = monthly_prices_df[ticker].dropna()
                if ticker_data.empty:
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
                    vol = monthly_returns.rolling(3, min_periods=1).std().fillna(0.02).clip(0.005, 0.03)
                    high = close * (1 + vol * 0.5)
                    low = close * (1 - vol * 0.5)
                    high = pd.Series(np.maximum(high, close), index=close.index)
                    low = pd.Series(np.minimum(low, close), index=close.index)

                close = close.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                high = high.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                low = low.replace([np.inf, -np.inf], np.nan).ffill().bfill()

                roc_10 = _adaptive_roc(close, max_period=10, min_period=3)
                f1 = roc_10 * 0.6
                atr_14 = calcular_atr_optimizado(high, low, close, periods=14)
                sma_14 = close.rolling(14, min_periods=3).mean()
                volatility_ratio = atr_14 / sma_14
                f2 = volatility_ratio * 0.4
                inercia_alcista = f1 / f2

                score = pd.Series(np.where(inercia_alcista >= corte, inercia_alcista, 0), index=inercia_alcista.index)
                score_adjusted = score / atr_14

                inercia_alcista = inercia_alcista.replace([np.inf, -np.inf], np.nan).fillna(0)
                score = score.replace([np.inf, -np.inf], np.nan).fillna(0)
                score_adjusted = score_adjusted.replace([np.inf, -np.inf], np.nan).fillna(0)

                if (inercia_alcista == 0).all() and (score == 0).all() and (score_adjusted == 0).all():
                    pass

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
    """
    Backtest mensual con:
    - Verificación histórica (regla último evento <= fecha)
    - Refugio IEF/BIL (opcional)
    - Ventanas adaptativas para no esperar 14 meses
    """
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
                if len(close) < 3:
                    return np.nan, np.nan
                if high_s is not None and low_s is not None:
                    high = high_s.copy().dropna().resample('ME').max().loc[close.index]
                    low = low_s.copy().dropna().resample('ME').min().loc[close.index]
                else:
                    monthly_returns = close.pct_change().dropna()
                    vol = monthly_returns.rolling(3, min_periods=1).std().fillna(0.02).clip(0.005, 0.03)
                    high = close * (1 + vol * 0.5)
                    low = close * (1 - vol * 0.5)
                    high = pd.Series(np.maximum(high, close), index=close.index)
                    low = pd.Series(np.minimum(low, close), index=close.index)
                high = high.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                low = low.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                close = close.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                roc_10 = _adaptive_roc(close, max_period=10, min_period=3)
                f1 = roc_10 * 0.6
                atr_14 = calcular_atr_optimizado(high, low, close, periods=14)
                sma_14 = close.rolling(14, min_periods=3).mean()
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
