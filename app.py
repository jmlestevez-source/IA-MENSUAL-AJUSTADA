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

# Importar módulos propios
from data_loader import (
    get_constituents_at_date,
    get_sp500_historical_changes,
    get_nasdaq100_historical_changes,
    generate_removed_tickers_summary,
)
from backtest import (
    run_backtest_optimized,
    precalculate_all_indicators,
    calculate_monthly_returns_by_year,
    inertia_score,
    calculate_sharpe_ratio,
)

# Verificación de archivos históricos
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

# Lectura robusta de Wikipedia con User-Agent para evitar 403
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
    # 1) Intentar id=constituents
    tables = _read_html_with_ua("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", attrs={"id": "constituents"})
    df = tables[0] if tables else None
    if df is None:
        # 2) Fallback: buscar tabla con Symbol/Ticker + Security/Company
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
    # 1) Intentar id=constituents
    tables = _read_html_with_ua("https://en.wikipedia.org/wiki/Nasdaq-100", attrs={"id": "constituents"})
    df = tables[0] if tables else None
    if df is None:
        # 2) Fallback: buscar Ticker/Symbol + Company
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

st.title("📈 Estrategia mensual sobre los componentes del S&P 500 y/o Nasdaq-100")

# Sidebar
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

top_n = st.sidebar.slider("Número de activos", 5, 30, 10)
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

# Main
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

# Mostrar resultados
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
    top_n = st.session_state.backtest_params.get("top_n", 10) if st.session_state.backtest_params else 10
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

    # Benchmark (seguro)
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

    # Gráficos (sin width/use_container_width para evitar errores)
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

    # Tabla retornos mensuales
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

    # Picks históricos (retorno mensual y comisión)
    if picks_df is not None and not picks_df.empty:
        st.subheader("📊 Picks Históricos")
        col_sidebar, col_main = st.columns([1, 3])

        with col_sidebar:
            st.markdown("### 📅 Navegación por Fechas")
            unique_dates = sorted(picks_df["Date"].unique(), reverse=True)
            selected_date = st.selectbox("Selecciona una fecha:", unique_dates, index=0, key="historical_date_selector")
            date_picks = picks_df[picks_df["Date"] == selected_date]
            st.info(f"🎯 {len(date_picks)} picks seleccionados el {selected_date}")

            # Retorno neto del mes
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

    # Señales actuales (vela en formación) con Nombres y transiciones
    with st.expander("🔮 Señales Actuales - Vela en Formación", expanded=True):
        st.subheader("📊 Picks Prospectivos para el Próximo Mes")
        try:
            if prices_df is not None and not prices_df.empty:
                # Mapa de nombres robusto
                if index_choice == "Ambos (SP500 + NDX)":
                    name_map = get_sp500_name_map()
                    tmp = get_ndx_name_map()
                    name_map.update(tmp)
                else:
                    name_map = get_name_map_for_index(index_choice)

                safety_set = {"IEF", "BIL"}
                filter_active = is_filter_active_for_next_month(spy_df, use_roc_filter, use_sma_filter) if (use_roc_filter or use_sma_filter) else False

                # Limitar universo a constituyentes actuales (evita safety y ajenos al índice)
                valid_universe = [t for t in prices_df.columns if t in universe_tickers and t not in safety_set]

                if use_safety_etfs and filter_active:
                    # Refugio IEF/BIL
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
                    # Universo normal
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
                                            # Nombre robusto: intenta en ambos mapas si el actual no lo trae
                                            name = name_map.get(ticker)
                                            if not name and index_choice != "SP500":
                                                name = get_sp500_name_map().get(ticker)
                                            if not name and index_choice != "NDX":
                                                name = get_ndx_name_map().get(ticker)
                                            if not name:
                                                name = ticker  # último recurso
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

    # ==============================
    # Robustez por número de posiciones (3 a 10) - ejecución fiable (re-ejecuta por k)
    # ==============================
    st.subheader("🧪 Robustez por número de posiciones (3 → 10)")
    do_robust = st.checkbox(
        "Calcular matriz de robustez 3-10 posiciones (CAGR y Max DD)",
        value=False,
        help="Ejecuta backtests para top_n = 3..10 con los mismos parámetros actuales"
    )

    if do_robust:
        with st.spinner("Calculando robustez..."):
            results = []
            # Pre-carga opcional de safety para todos los k si procede
            safety_prices_k = pd.DataFrame()
            safety_ohlc_k = {}
            if use_safety_etfs:
                sd = pd.to_datetime(st.session_state.backtest_params.get("start")).date()
                ed = pd.to_datetime(st.session_state.backtest_params.get("end")).date()
                safety_prices_k, safety_ohlc_k = load_prices_from_csv_parallel(['IEF','BIL'], sd, ed, load_full_data=True)

            # Cache por combinación de parámetros+k
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

# Sidebar info
if st.session_state.spy_df is not None:
    st.sidebar.success(f"✅ SPY en memoria: {len(st.session_state.spy_df)} registros")
else:
    if not st.session_state.backtest_completed:
        st.info("👈 Configura los parámetros y haz clic en 'Ejecutar backtest'")
        st.subheader("🔍 Información del Sistema")
        st.info(
            """
        - ✅ Verificación histórica de constituyentes
        - ✅ Cálculos optimizados con precálculo y paralelización
        - ✅ Caché multinivel para carga instantánea
        - ✅ Comparación completa con benchmark
        - ✅ Tabla de rendimientos mensuales
        - ✅ Señales actuales con vela en formación
        - ✅ Filtros de mercado configurables
        - ✅ Evitar recompras cuando se mantienen en Top (opcional)
        """
        )
        cache_files = glob.glob(os.path.join(CACHE_DIR, "backtest_*.pkl"))
        if cache_files:
            st.info(f"💾 {len(cache_files)} resultados en caché disponibles para carga instantánea")
