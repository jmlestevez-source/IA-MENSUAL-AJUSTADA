import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import numpy as np
import os
import requests
import base64
from concurrent.futures import ThreadPoolExecutor
import pickle
import hashlib
import glob

# Importar nuestros módulos - IMPORTANTE
from data_loader import get_constituents_at_date, get_sp500_historical_changes, get_nasdaq100_historical_changes, generate_removed_tickers_summary
from backtest import run_backtest_optimized, precalculate_all_indicators, calculate_monthly_returns_by_year, inertia_score, calculate_sharpe_ratio


# -------------------------------------------------
# Verificación de archivos históricos
# -------------------------------------------------
def check_historical_files():
    files_to_check = [
        "sp500_changes.csv",
        "ndx_changes.csv",
        "data/sp500_changes.csv",
        "data/ndx_changes.csv"
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
    return found_files


historical_files = check_historical_files()

st.set_page_config(page_title="IA Mensual Ajustada", page_icon="📈", layout="wide")


# -------------------------------------------------
# Funciones de caché
# -------------------------------------------------
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


@st.cache_data(ttl=3600*24*7)
def load_prices_from_csv_parallel(tickers, start_date, end_date, load_full_data=True):
    prices_data, ohlc_data = {}, {}

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
            price = df['Close'] if 'Close' in df.columns else None
            ohlc = None
            if load_full_data and all(c in df.columns for c in ['High', 'Low', 'Close']):
                ohlc = {
                    'High': df['High'], 'Low': df['Low'], 'Close': df['Close'],
                    'Volume': df['Volume'] if 'Volume' in df.columns else None
                }
            return ticker, price, ohlc
        except:
            return ticker, None, None

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(load_single_ticker, t) for t in tickers]
        for f in futures:
            t, p, o = f.result()
            if p is not None: prices_data[t] = p
            if o is not None: ohlc_data[t] = o

    if prices_data:
        df = pd.DataFrame(prices_data)
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df, ohlc_data
    return pd.DataFrame(), {}


def create_download_link(df, filename, link_text):
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    except:
        return None


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Parámetros de Backtest")

index_choice = st.sidebar.selectbox("Índice", ["SP500", "NDX", "Ambos (SP500 + NDX)"])
default_end = min(datetime.today().date(), datetime(2030, 12, 31).date())
default_start = default_end - timedelta(days=365*5)
end_date = st.sidebar.date_input("Fecha final", default_end)
start_date = st.sidebar.date_input("Fecha inicial", default_start)
if start_date >= end_date:
    start_date = end_date - timedelta(days=365*2)

top_n = st.sidebar.slider("Número de activos", 5, 30, 10)
commission = st.sidebar.number_input("Comisión (%)", 0.0, 1.0, 0.3) / 100
corte = st.sidebar.number_input("Corte de score", 0, 1000, 680)
use_historical_verification = st.sidebar.checkbox("Usar verificación histórica", True)
fixed_allocation = st.sidebar.checkbox("🚀 10% capital fijo por acción", False)
use_roc_filter = st.sidebar.checkbox("ROC 12m SPY < 0", False)
use_sma_filter = st.sidebar.checkbox("Precio SPY < SMA 10m", False)
run_button = st.sidebar.button("🏃 Ejecutar backtest", type="primary")

# -------------------------------------------------
# Directorios cache
# -------------------------------------------------
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
st.title("📈 Estrategia Mensual SP500 / NDX")


if run_button:
    cache_params = {
        'index': index_choice,
        'start': str(start_date),
        'end': str(end_date),
        'top_n': top_n,
        'corte': corte,
        'commission': commission,
        'historical': use_historical_verification,
        'fixed_alloc': fixed_allocation,
        'roc_filter': use_roc_filter,
        'sma_filter': use_sma_filter
    }
    cache_key = get_cache_key(cache_params)
    cache_file = os.path.join(CACHE_DIR, f"backtest_{cache_key}.pkl")

    use_cache = False
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
            if st.sidebar.checkbox("🔄 Usar caché local", True):
                st.success("✅ Resultados desde caché local")
                use_cache = True
                st.session_state.bt_results = cached["bt_results"]
                st.session_state.picks_df = cached["picks_df"]
                st.session_state.prices_df = cached["prices_df"]
                st.session_state.ohlc_data = cached["ohlc_data"]
                st.session_state.benchmark_series = cached["benchmark_series"]
                st.session_state.spy_df = cached["spy_df"]

    if not use_cache:
        status = st.empty()
        progress = st.progress(0)

        # Constituents
        status.text("📥 Obtener constituyentes...")
        all_tickers_data, _ = get_constituents_cached(index_choice, start_date, end_date)
        tickers = list(dict.fromkeys(all_tickers_data['tickers']))

        # Prices
        prices_df, ohlc_data = load_prices_from_csv_parallel(tickers, start_date, end_date)

        # Benchmark
        benchmark_ticker = "SPY" if index_choice != "NDX" else "QQQ"
        bench_df, _ = load_prices_from_csv_parallel([benchmark_ticker], start_date, end_date, False)
        benchmark_series = bench_df[benchmark_ticker] if not bench_df.empty else prices_df.mean(axis=1)

        # SPY filters
        spy_df = None
        if use_roc_filter or use_sma_filter:
            spy_data, _ = load_prices_from_csv_parallel(["SPY"], start_date, end_date, False)
            spy_df = spy_data if not spy_data.empty else None

        # 🚀 Backtest
        status.text("🚀 Ejecutando backtest...")
        bt_results, picks_df = run_backtest_optimized(
            prices=prices_df,
            benchmark=benchmark_series,
            commission=commission,
            top_n=top_n,
            corte=corte,
            ohlc_data=ohlc_data,
            fixed_allocation=fixed_allocation,
            use_roc_filter=use_roc_filter,
            use_sma_filter=use_sma_filter,
            spy_data=spy_df,
            progress_callback=lambda p: progress.progress(70+int(p*0.3))
        )

        # 👉 Guardamos en session_state para no perderlos en recargas
        st.session_state.bt_results = bt_results
        st.session_state.picks_df = picks_df
        st.session_state.prices_df = prices_df
        st.session_state.ohlc_data = ohlc_data
        st.session_state.benchmark_series = benchmark_series
        st.session_state.spy_df = spy_df

        # Guardar en disco como ya hacías
        with open(cache_file, "wb") as f:
            pickle.dump({
                "bt_results": bt_results,
                "picks_df": picks_df,
                "prices_df": prices_df,
                "ohlc_data": ohlc_data,
                "benchmark_series": benchmark_series,
                "spy_df": spy_df,
                "timestamp": datetime.now()
            }, f)
        status.empty()
        progress.empty()


# -------------------------------------------------
# MOSTRAR RESULTADOS desde session_state (persistentes)
# -------------------------------------------------
if "bt_results" in st.session_state and "picks_df" in st.session_state:
    bt_results = st.session_state.bt_results
    picks_df = st.session_state.picks_df
    prices_df = st.session_state.prices_df

    st.success("✅ Backtest listo — interactúa sin cerrar sesión")

    # Equity curve
    st.subheader("📊 Evolución Equity")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results["Equity"], mode="lines", name="Estrategia"))
    st.plotly_chart(fig, use_container_width=True)

    # Picks históricos
    if not picks_df.empty:
        st.subheader("📅 Picks Históricos")
        unique_dates = sorted(picks_df["Date"].unique(), reverse=True)
        if "selected_date" not in st.session_state:
            st.session_state.selected_date = unique_dates[0]
        selected_date = st.selectbox(
            "Selecciona fecha:",
            unique_dates,
            index=unique_dates.index(st.session_state.selected_date),
            key="date_box"
        )
        st.session_state.selected_date = selected_date

        df_sel = picks_df[picks_df["Date"] == selected_date]
        st.write(f"🎯 Picks en {selected_date}: {len(df_sel)} activos")
        st.dataframe(df_sel)
