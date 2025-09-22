import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

# Importar nuestros módulos - IMPORTANTE: importar inertia_score
from data_loader import get_constituents_at_date, get_sp500_historical_changes, get_nasdaq100_historical_changes, generate_removed_tickers_summary
from backtest import run_backtest_optimized, precalculate_all_indicators, calculate_monthly_returns_by_year, inertia_score, calculate_sharpe_ratio

# Al inicio del script, después de los imports
def check_historical_files():
    """Verifica la existencia de archivos de cambios históricos"""
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
        print("📁 Archivos esperados: sp500_changes.csv, ndx_changes.csv")
    
    return found_files

# Ejecutar verificación
historical_files = check_historical_files()

# -------------------------------------------------
# Configuración de la app
# -------------------------------------------------
st.set_page_config(
    page_title="IA Mensual Ajustada",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------
# INICIALIZAR SESSION STATE PARA MANTENER RESULTADOS
# -------------------------------------------------
if 'backtest_completed' not in st.session_state:
    st.session_state.backtest_completed = False
if 'bt_results' not in st.session_state:
    st.session_state.bt_results = None
if 'picks_df' not in st.session_state:
    st.session_state.picks_df = None
if 'spy_df' not in st.session_state:
    st.session_state.spy_df = None
if 'prices_df' not in st.session_state:
    st.session_state.prices_df = None
if 'benchmark_series' not in st.session_state:
    st.session_state.benchmark_series = None
if 'ohlc_data' not in st.session_state:
    st.session_state.ohlc_data = None
if 'historical_info' not in st.session_state:
    st.session_state.historical_info = None
if 'backtest_params' not in st.session_state:
    st.session_state.backtest_params = None

# -------------------------------------------------
# FUNCIONES DE CACHÉ OPTIMIZADAS
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_historical_changes_cached(index_name, force_reload=False):
    """Carga cambios históricos con caché"""
    
    # Si force_reload, limpiar caché
    if force_reload:
        st.cache_data.clear()
    
    if index_name == "SP500":
        changes = get_sp500_historical_changes()
        if changes.empty:
            st.warning("⚠️ No se pudieron cargar cambios históricos del S&P 500")
        return changes
    elif index_name == "NDX":
        changes = get_nasdaq100_historical_changes()
        if changes.empty:
            st.warning("⚠️ No se pudieron cargar cambios históricos del NASDAQ-100")
        return changes
    else:  # Ambos
        sp500 = get_sp500_historical_changes()
        ndx = get_nasdaq100_historical_changes()
        
        if sp500.empty:
            st.warning("⚠️ No se pudieron cargar cambios del S&P 500")
        if ndx.empty:
            st.warning("⚠️ No se pudieron cargar cambios del NASDAQ-100")
            
        if not sp500.empty and not ndx.empty:
            return pd.concat([sp500, ndx], ignore_index=True)
        return sp500 if not sp500.empty else ndx

@st.cache_data(ttl=86400)
def get_constituents_cached(index_name, start_date, end_date):
    """Obtiene constituyentes con caché"""
    return get_constituents_at_date(index_name, start_date, end_date)

def get_cache_key(params):
    """Genera una clave única para caché basada en parámetros"""
    param_str = str(sorted(params.items()))
    return hashlib.md5(param_str.encode()).hexdigest()

@st.cache_data(ttl=3600*24*7)
def load_prices_from_csv_parallel(tickers, start_date, end_date, load_full_data=True):
    """Carga precios desde CSV en PARALELO (corrigida para manejar zonas horarias)"""
    prices_data = {}
    ohlc_data = {}

    def load_single_ticker(ticker):
        clean_ticker = str(ticker).strip().upper().replace('.', '-')
        csv_path = f"data/{clean_ticker}.csv"
        if not os.path.exists(csv_path):
            return ticker, None, None

        try:
            # Leer CSV eliminando zona horaria directamente
            df = pd.read_csv(
                csv_path,
                index_col="Date",
                parse_dates=True,
                date_parser=lambda col: pd.to_datetime(col, utc=True).tz_convert(None)
            )

            start_filter = start_date.date() if isinstance(start_date, datetime) else start_date
            end_filter = end_date.date() if isinstance(end_date, datetime) else end_date
            df = df[(df.index.date >= start_filter) & (df.index.date <= end_filter)]

            if df.empty:
                return ticker, None, None

            # Usar Close (o Adj Close si existe)
            if 'Adj Close' in df.columns:
                price = df['Adj Close']
            elif 'Close' in df.columns:
                price = df['Close']
            else:
                return ticker, None, None

            ohlc = None
            if load_full_data and all(col in df.columns for col in ['High', 'Low', 'Close']):
                ohlc = {
                    'High': df['High'],
                    'Low': df['Low'],
                    'Close': df['Adj Close'] if 'Adj Close' in df.columns else df['Close'],
                    'Volume': df['Volume'] if 'Volume' in df.columns else None
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
        st.error(f"Error creando enlace de descarga: {e}")
        return None

# -------------------------------------------------
# Título y configuración principal
# -------------------------------------------------
st.title("📈 Estrategia mensual sobre los componentes del S&P 500 y/o Nasdaq-100")

# -------------------------------------------------
# Sidebar - Parámetros
# -------------------------------------------------
st.sidebar.header("Parámetros de backtest")

index_choice = st.sidebar.selectbox("Selecciona el índice:", ["SP500", "NDX", "Ambos (SP500 + NDX)"])

# Fechas
try:
    default_end = min(datetime.today().date(), datetime(2030, 12, 31).date())
    default_start = default_end - timedelta(days=365*5)
    
    end_date = st.sidebar.date_input("Fecha final", value=default_end, min_value=datetime(1950, 1, 1).date(), max_value=datetime(2030, 12, 31).date())
    start_date = st.sidebar.date_input("Fecha inicial", value=default_start, min_value=datetime(1950, 1, 1).date(), max_value=datetime(2030, 12, 31).date())
    
    if start_date >= end_date:
        st.sidebar.warning("⚠️ Fecha inicial debe ser anterior a la fecha final")
        start_date = end_date - timedelta(days=365*2)
        
    st.sidebar.info(f"📅 Rango: {start_date} a {end_date}")
    
except Exception as e:
    st.sidebar.error(f"❌ Error configurando fechas: {e}")
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=365*5)

# Parámetros
top_n = st.sidebar.slider("Número de activos", 5, 30, 10)
commission = st.sidebar.number_input("Comisión por operación (%)", 0.0, 1.0, 0.3) / 100
corte = st.sidebar.number_input("Corte de score", 0, 1000, 680)
use_historical_verification = st.sidebar.checkbox("🕐 Usar verificación histórica", value=True)

st.sidebar.subheader("⚙️ Opciones de Estrategia")
fixed_allocation = st.sidebar.checkbox("💰 Asignar 10% capital a cada acción", value=False)

st.sidebar.subheader("🛡️ Filtros de Mercado")
use_roc_filter = st.sidebar.checkbox("📉 ROC 12 meses del SPY < 0", value=False)
use_sma_filter = st.sidebar.checkbox("📊 Precio SPY < SMA 10 meses", value=False)

run_button = st.sidebar.button("🏃 Ejecutar backtest", type="primary")

# Botón para limpiar resultados
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
        st.rerun()

# -------------------------------------------------
# CONSTANTES
# -------------------------------------------------
GITHUB_RAW_URL = "https://raw.githubusercontent.com/jmlestevez-source/IA-MENSUAL-AJUSTADA/main/"
LOCAL_CHANGES_DIR = "data/historical_changes"
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOCAL_CHANGES_DIR, exist_ok=True)

# -------------------------------------------------
# Main content
# -------------------------------------------------
if run_button:
    # Limpiar session state anterior
    st.session_state.backtest_completed = False
    
    try:
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
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if st.sidebar.checkbox("🔄 Usar resultados en caché", value=True):
                        use_cache = True
                        st.success("✅ Cargando resultados desde caché...")
                        bt_results = cached_data['bt_results']
                        picks_df = cached_data['picks_df']
                        historical_info = cached_data.get('historical_info')
                        prices_df = cached_data.get('prices_df')
                        ohlc_data = cached_data.get('ohlc_data')
                        benchmark_series = cached_data.get('benchmark_series')
                        spy_df = cached_data.get('spy_df')
            except:
                use_cache = False
        
        if not use_cache:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Obtener constituyentes
            status_text.text("📥 Obteniendo constituyentes...")
            progress_bar.progress(10)
            
            all_tickers_data, error = get_constituents_cached(index_choice, start_date, end_date)
            if error:
                st.warning(f"Advertencia: {error}")
            
            if not all_tickers_data or 'tickers' not in all_tickers_data:
                st.error("No se encontraron tickers válidos")
                st.stop()
            
            tickers = list(dict.fromkeys(all_tickers_data['tickers']))
            st.success(f"✅ Obtenidos {len(tickers)} tickers únicos")
            
            # Cargar precios
            status_text.text("📊 Cargando precios en paralelo...")
            progress_bar.progress(30)
            
            prices_df, ohlc_data = load_prices_from_csv_parallel(tickers, start_date, end_date, load_full_data=True)
            
            if prices_df.empty:
                st.error("❌ No se pudieron cargar precios")
                st.stop()
            
            st.success(f"✅ Cargados {len(prices_df.columns)} tickers con datos")
            
            # Cargar benchmark
            status_text.text("📈 Cargando benchmark...")
            progress_bar.progress(40)
            
            benchmark_ticker = "SPY" if index_choice != "NDX" else "QQQ"
            benchmark_df, _ = load_prices_from_csv_parallel([benchmark_ticker], start_date, end_date, load_full_data=False)
            
            if benchmark_df.empty:
                st.warning("Usando promedio como benchmark")
                benchmark_series = prices_df.mean(axis=1)
            else:
                benchmark_series = benchmark_df[benchmark_ticker]
            
            # SIEMPRE cargar SPY para gráficas (independiente de filtros)
            spy_df = None
            status_text.text("📈 Cargando SPY...")
            spy_result, _ = load_prices_from_csv_parallel(["SPY"], start_date, end_date, load_full_data=False)
            if not spy_result.empty and "SPY" in spy_result.columns:
                spy_df = spy_result
                st.sidebar.success(f"✅ SPY cargado: {len(spy_df)} registros")
                
                # Información sobre el uso del SPY
                if use_roc_filter or use_sma_filter:
                    st.sidebar.info("📊 SPY usado para filtros y visualización")
                else:
                    st.sidebar.info("📊 SPY cargado para visualización")
            else:
                st.sidebar.warning("⚠️ No se pudo cargar SPY")
                spy_df = None
            
            # Información histórica
            historical_info = None
            if use_historical_verification:
                status_text.text("🕐 Cargando datos históricos...")
                progress_bar.progress(50)
                
                # Verificar si existen los archivos CSV locales
                sp500_csv_exists = os.path.exists("sp500_changes.csv") or os.path.exists("data/sp500_changes.csv")
                ndx_csv_exists = os.path.exists("ndx_changes.csv") or os.path.exists("data/ndx_changes.csv")
                
                if sp500_csv_exists or ndx_csv_exists:
                    st.info(f"📂 Encontrados archivos CSV locales de cambios históricos")
                
                changes_data = load_historical_changes_cached(index_choice)
                
                if not changes_data.empty:
                    historical_info = {
                        'changes_data': changes_data, 
                        'has_historical_data': True
                    }
                    st.success(f"✅ Cargados {len(changes_data)} cambios históricos")
                    
                    # Mostrar información sobre el origen de los datos
                    if sp500_csv_exists or ndx_csv_exists:
                        st.info("📊 Datos cargados desde archivos CSV locales (más rápido)")
                    else:
                        st.info("🌐 Datos descargados desde Wikipedia")
                else:
                    st.warning("⚠️ No se encontraron datos históricos, continuando sin verificación")
                    st.info("💡 Tip: Asegúrate de que sp500_changes.csv y ndx_changes.csv estén en la raíz del repositorio")
                    historical_info = None
            
            # Ejecutar backtest
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
                progress_callback=lambda p: progress_bar.progress(70 + int(p * 0.3))
            )
            
            # Guardar en session state
            st.session_state.bt_results = bt_results
            st.session_state.picks_df = picks_df
            st.session_state.spy_df = spy_df
            st.session_state.prices_df = prices_df
            st.session_state.benchmark_series = benchmark_series
            st.session_state.ohlc_data = ohlc_data
            st.session_state.historical_info = historical_info
            st.session_state.backtest_params = cache_params
            st.session_state.backtest_completed = True
            
            # Guardar en caché
            status_text.text("💾 Guardando resultados en caché...")
            progress_bar.progress(100)
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'bt_results': bt_results,
                        'picks_df': picks_df,
                        'historical_info': historical_info,
                        'prices_df': prices_df,
                        'ohlc_data': ohlc_data,
                        'benchmark_series': benchmark_series,
                        'spy_df': spy_df,
                        'timestamp': datetime.now()
                    }, f)
                st.success("✅ Resultados guardados en caché")
            except Exception as e:
                st.warning(f"No se pudo guardar caché: {e}")
            
            status_text.empty()
            progress_bar.empty()
        else:
            # Si usamos caché, también guardar en session state
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

# Mostrar resultados si están en session state
if st.session_state.backtest_completed and st.session_state.bt_results is not None:
    # RECUPERAR TODAS LAS VARIABLES DEL SESSION STATE
    bt_results = st.session_state.bt_results
    picks_df = st.session_state.picks_df
    spy_df = st.session_state.spy_df
    prices_df = st.session_state.prices_df
    benchmark_series = st.session_state.benchmark_series
    ohlc_data = st.session_state.ohlc_data
    historical_info = st.session_state.historical_info
    
    # Recuperar parámetros del backtest
    use_roc_filter = False
    use_sma_filter = False
    index_choice = 'SP500'
    fixed_allocation = False
    corte = 680
    top_n = 10
    
    if st.session_state.backtest_params:
        index_choice = st.session_state.backtest_params.get('index', 'SP500')
        use_roc_filter = st.session_state.backtest_params.get('roc_filter', False)
        use_sma_filter = st.session_state.backtest_params.get('sma_filter', False)
        fixed_allocation = st.session_state.backtest_params.get('fixed_alloc', False)
        corte = st.session_state.backtest_params.get('corte', 680)
        top_n = st.session_state.backtest_params.get('top_n', 10)
        start_date = st.session_state.backtest_params.get('start')
        end_date = st.session_state.backtest_params.get('end')
    
    # Debug info
    if spy_df is not None:
        st.sidebar.success(f"✅ Datos SPY disponibles: {len(spy_df)} registros")
    else:
        st.sidebar.warning("⚠️ No hay datos del SPY")
    
    st.sidebar.info(f"🔍 Filtros activos: ROC={use_roc_filter}, SMA={use_sma_filter}")
    
    st.success("✅ Backtest completado exitosamente")
    
    # Calcular métricas
    final_equity = float(bt_results["Equity"].iloc[-1])
    initial_equity = float(bt_results["Equity"].iloc[0])
    total_return = (final_equity / initial_equity) - 1
    years = (bt_results.index[-1] - bt_results.index[0]).days / 365.25
    cagr = (final_equity / initial_equity) ** (1/years) - 1 if years > 0 else 0
    max_drawdown = float(bt_results["Drawdown"].min())
    
    # Sharpe ratio (mensual, RF 2% anual)
    monthly_returns = bt_results["Returns"]
    risk_free_rate_annual = 0.02
    risk_free_rate_monthly = (1 + risk_free_rate_annual) ** (1/12) - 1
    excess_returns = monthly_returns - risk_free_rate_monthly
    
    if len(monthly_returns) > 0 and excess_returns.std() > 0:
        sharpe_ratio = (excess_returns.mean() * 12) / (excess_returns.std() * np.sqrt(12))
    else:
        sharpe_ratio = 0
        
    volatility = float(monthly_returns.std() * np.sqrt(12))
    
    st.subheader("📊 Métricas de la Estrategia")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Equity Final", f"${final_equity:,.0f}")
    col2.metric("Retorno Total", f"{total_return:.2%}")
    col3.metric("CAGR", f"{cagr:.2%}")
    col4.metric("Max Drawdown (Mensual)", f"{max_drawdown:.2%}")
    col5.metric("Sharpe Ratio (RF=2%)", f"{sharpe_ratio:.2f}")
    
    # Max drawdown diario (aprox) de la estrategia, distribuyendo retorno mensual por días hábiles
    strategy_daily_mdd = None
    try:
        m_index = pd.to_datetime(bt_results.index)
        daily_dates = pd.date_range(start=m_index[0], end=m_index[-1], freq='B')
        equity_daily = []
        current_equity = float(bt_results["Equity"].iloc[0])
        equity_daily.append((m_index[0], current_equity))
        
        for i in range(1, len(m_index)):
            prev_date = m_index[i-1]
            date = m_index[i]
            month_ret = float(bt_results["Returns"].iloc[i])  # retorno del tramo prev_date->date
            # Rango diario del mes (excluye el día prev_date, incluye días hábiles hasta date)
            span_days = pd.date_range(start=prev_date + pd.Timedelta(days=1), end=date, freq='B')
            n = len(span_days)
            if n <= 0:
                # sin días (caso raro), aplicar todo al final
                current_equity = current_equity * (1 + month_ret)
                equity_daily.append((date, current_equity))
                continue
            per_day_factor = (1 + month_ret) ** (1 / n)
            for d in span_days:
                current_equity *= per_day_factor
                equity_daily.append((d, current_equity))
        
        equity_daily_series = pd.Series([v for _, v in equity_daily], index=[d for d, _ in equity_daily]).sort_index()
        dd_daily = (equity_daily_series / equity_daily_series.cummax()) - 1
        strategy_daily_mdd = float(dd_daily.min())
    except Exception:
        strategy_daily_mdd = None
    
    col_row = st.columns(2)
    col_row[0].metric("Max DD Diario Estrategia (aprox)", f"{strategy_daily_mdd:.2%}" if strategy_daily_mdd is not None else "N/D")
    
    # Benchmark (usar mensual para coherencia con la estrategia)
    bench_equity = None
    bench_drawdown = None
    bench_sharpe = 0
    bench_final = initial_equity
    bench_total_return = 0
    bench_cagr = 0
    bench_max_dd = 0
    
    # Preparar SPY para métricas diarias (si existe)
    spy_daily_mdd = None
    if spy_df is not None and hasattr(spy_df, 'columns') and 'SPY' in spy_df.columns:
        try:
            spy_daily_returns = spy_df['SPY'].pct_change().fillna(0)
            spy_daily_equity = initial_equity * (1 + spy_daily_returns).cumprod()
            spy_daily_dd = (spy_daily_equity / spy_daily_equity.cummax()) - 1
            spy_daily_mdd = float(spy_daily_dd.min())
        except Exception:
            spy_daily_mdd = None
    
    if benchmark_series is not None and not benchmark_series.empty:
        try:
            # Mensualizar y calcular métricas coherentes con la estrategia
            bench_prices_m = benchmark_series.resample('ME').last()
            bench_returns_m = bench_prices_m.pct_change().fillna(0)
            bench_equity = initial_equity * (1 + bench_returns_m).cumprod()
            bench_drawdown = (bench_equity / bench_equity.cummax() - 1)
            
            bench_final = float(bench_equity.reindex(bt_results.index).ffill().iloc[-1])
            bench_initial = float(bench_equity.reindex(bt_results.index).ffill().iloc[0]) if bench_equity.iloc[0] != 0 else initial_equity
            bench_total_return = (bench_final / bench_initial) - 1
            
            if years > 0:
                bench_cagr = (bench_final / bench_initial) ** (1/years) - 1
            
            bench_max_dd = float(bench_drawdown.reindex(bt_results.index).ffill().min())
            
            # Sharpe mensual con la misma metodología
            bench_excess_returns = bench_returns_m - risk_free_rate_monthly
            bench_sharpe = (bench_excess_returns.mean() * 12) / (bench_excess_returns.std() * np.sqrt(12)) if bench_excess_returns.std() != 0 else 0
                
        except Exception as e:
            st.warning(f"Error calculando benchmark: {e}")
            bench_sharpe = 0
    
    benchmark_name = "SPY" if index_choice != "NDX" else "QQQ"
    
    st.subheader(f"📊 Métricas del Benchmark ({benchmark_name})")
    col1b, col2b, col3b, col4b, col5b = st.columns(5)
    col1b.metric("Equity Final", f"${bench_final:,.0f}")
    col2b.metric("Retorno Total", f"{bench_total_return:.2%}")
    col3b.metric("CAGR", f"{bench_cagr:.2%}")
    col4b.metric("Max Drawdown (Mensual)", f"{bench_max_dd:.2%}")
    col5b.metric("Sharpe Ratio (RF=2%)", f"{bench_sharpe:.2f}")
    
    col_row_b = st.columns(2)
    col_row_b[0].metric("Max DD Diario SPY", f"{spy_daily_mdd:.2%}" if spy_daily_mdd is not None else "N/D")
    
    # Comparación
    st.subheader("⚖️ Comparación Estrategia vs Benchmark")
    col1c, col2c, col3c, col4c = st.columns(4)
    
    alpha = cagr - bench_cagr
    col1c.metric("Alpha (CAGR diff)", f"{alpha:.2%}", delta=f"{alpha:.2%}")
    
    sharpe_diff = sharpe_ratio - bench_sharpe
    col2c.metric("Sharpe Diff", f"{sharpe_diff:.2f}", delta=f"{sharpe_diff:.2f}")
    
    dd_diff = max_drawdown - bench_max_dd
    col3c.metric("DD Difference", f"{dd_diff:.2%}", delta=f"{dd_diff:.2%}")
    
    return_diff = total_return - bench_total_return
    col4c.metric("Return Diff", f"{return_diff:.2%}", delta=f"{return_diff:.2%}")
    
    # Información sobre verificación histórica
    if historical_info and historical_info.get('has_historical_data', False):
        st.info("✅ Este backtest incluye verificación histórica de constituyentes")
    elif st.session_state.backtest_params and st.session_state.backtest_params.get('historical', False):
        st.warning("⚠️ Verificación histórica solicitada pero no se encontraron datos históricos")
    else:
        st.warning("⚠️ Este backtest NO incluye verificación histórica (posible sesgo de supervivencia)")
    
    # ==============================
    # Series mensuales alineadas para gráficas (NUEVO)
    # ==============================
    bench_equity_m = None
    bench_drawdown_m = None
    spy_equity_m = None
    spy_drawdown_m = None
    try:
        if benchmark_series is not None and not benchmark_series.empty:
            bench_prices_m = benchmark_series.resample('ME').last()
            bench_returns_m = bench_prices_m.pct_change().fillna(0)
            bench_equity_m = initial_equity * (1 + bench_returns_m).cumprod()
            bench_drawdown_m = (bench_equity_m / bench_equity_m.cummax()) - 1
        if spy_df is not None and not spy_df.empty and 'SPY' in (spy_df.columns if hasattr(spy_df, 'columns') else []):
            spy_prices_m = spy_df['SPY'].resample('ME').last()
            spy_returns_m = spy_prices_m.pct_change().fillna(0)
            spy_equity_m = initial_equity * (1 + spy_returns_m).cumprod()
            spy_drawdown_m = (spy_equity_m / spy_equity_m.cummax()) - 1
    except Exception as e:
        st.warning(f"No se pudieron preparar series mensuales para gráficas: {e}")
    
    # GRÁFICOS PRINCIPALES CON SUBPLOTS COMPARTIDOS - CORREGIDO PARA MOSTRAR SIEMPRE SPY
    st.subheader("📈 Gráficos de Rentabilidad")
    
    # Mostrar información de debug si hay filtros activos
    if use_roc_filter or use_sma_filter:
        col1_debug, col2_debug = st.columns(2)
        with col1_debug:
            st.info(f"📊 ROC Filter: {'✅ Activo' if use_roc_filter else '❌ Inactivo'}")
        with col2_debug:
            st.info(f"📊 SMA Filter: {'✅ Activo' if use_sma_filter else '❌ Inactivo'}")
    
    # CREAR FIGURA CON SUBPLOTS
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=("Evolución del Equity", "Drawdown")
    )
    
    # SUBPLOT 1: EQUITY
    # Estrategia
    fig.add_trace(
        go.Scatter(
            x=bt_results.index,
            y=bt_results["Equity"],
            mode='lines',
            name='Estrategia',
            line=dict(width=3, color='blue'),
            hovertemplate='<b>%{y:,.0f}</b><br>%{x}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Benchmark mensual y alineado
    if bench_equity_m is not None:
        bench_aligned = bench_equity_m.reindex(bt_results.index).ffill()
        fig.add_trace(
            go.Scatter(
                x=bench_aligned.index,
                y=bench_aligned.values,
                mode='lines',
                name=f'Benchmark ({benchmark_name})',
                line=dict(width=2.5, color='#50C878'), # verde esmeralda
                hovertemplate='<b>%{y:,.0f}</b><br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # SPY mensual y alineado (solo si el benchmark no es SPY para no duplicar)
    if spy_equity_m is not None and benchmark_name != "SPY":
        spy_aligned = spy_equity_m.reindex(bt_results.index).ffill()
        fig.add_trace(
            go.Scatter(
                x=spy_aligned.index,
                y=spy_aligned.values,
                mode='lines',
                name='SPY (Referencia)',
                line=dict(width=2.5, color='#50C878'), # verde esmeralda
                hovertemplate='<b>SPY: %{y:,.0f}</b><br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # SUBPLOT 2: DRAWDOWN
    if "Drawdown" in bt_results.columns:
        # Drawdown estrategia
        fig.add_trace(
            go.Scatter(
                x=bt_results.index,
                y=bt_results["Drawdown"] * 100,
                mode='lines',
                name='DD Estrategia',
                fill='tozeroy',
                line=dict(color='red', width=2),
                hovertemplate='<b>%{y:.2f}%</b><br>%{x}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Drawdown benchmark mensual y alineado
        if bench_drawdown_m is not None:
            bench_dd_aligned = (bench_drawdown_m.reindex(bt_results.index).ffill() * 100)
            fig.add_trace(
                go.Scatter(
                    x=bench_dd_aligned.index,
                    y=bench_dd_aligned.values,
                    mode='lines',
                    name=f'DD {benchmark_name}',
                    line=dict(color='#FF7F0E', width=2), # naranja
                    hovertemplate='<b>%{y:.2f}%</b><br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Drawdown SPY mensual y alineado
        if spy_drawdown_m is not None and benchmark_name != "SPY":
            spy_dd_aligned = (spy_drawdown_m.reindex(bt_results.index).ffill() * 100)
            fig.add_trace(
                go.Scatter(
                    x=spy_dd_aligned.index,
                    y=spy_dd_aligned.values * 100,
                    mode='lines',
                    name='DD SPY (Ref)',
                    line=dict(color='#FF7F0E', width=2), # naranja
                    hovertemplate='<b>%{y:.2f}%</b><br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
    
    # Actualizar layout
    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_yaxes(title_text="Equity ($)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de rendimientos mensuales
    st.subheader("📅 RENDIMIENTOS MENSUALES POR AÑO")
    
    monthly_table = calculate_monthly_returns_by_year(bt_results["Equity"])
    
    if not monthly_table.empty:
        def style_returns(val):
            if val == "-" or val == "":
                return ""
            try:
                num = float(val.rstrip('%'))
                if num > 0:
                    return "background-color: #d4edda; color: #155724; font-weight: bold"
                elif num < 0:
                    return "background-color: #f8d7da; color: #721c24; font-weight: bold"
                else:
                    return ""
            except:
                return ""
        
        styled_table = monthly_table.style.applymap(style_returns)
        st.dataframe(styled_table, use_container_width=True)
        
        # Estadísticas
        total_years = len(monthly_table)
        if total_years > 0:
            positive_years = 0
            ytd_values = []
            
            for _, row in monthly_table.iterrows():
                if row['YTD'] != "-" and row['YTD'] != "":
                    try:
                        ytd_val = float(row['YTD'].rstrip('%'))
                        ytd_values.append(ytd_val)
                        if ytd_val > 0:
                            positive_years += 1
                    except:
                        continue
            
            if ytd_values:
                avg_annual_return = sum(ytd_values) / len(ytd_values)
                win_rate = (positive_years / len(ytd_values)) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Años Totales", total_years)
                col2.metric("Retorno Anual Promedio", f"{avg_annual_return:.1f}%")
                col3.metric("Tasa de Éxito Anual", f"{win_rate:.0f}%")
    
    # PICKS HISTÓRICOS - Con cálculo corregido de retornos
    if picks_df is not None and not picks_df.empty:
        st.subheader("📊 Picks Históricos")
        
        if 'HistoricallyValid' in picks_df.columns:
            total_picks = len(picks_df)
            valid_picks = picks_df['HistoricallyValid'].sum()
            validity_rate = valid_picks / total_picks * 100 if total_picks > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Picks", total_picks)
            col2.metric("Picks Válidos", valid_picks)
            col3.metric("% Validez Histórica", f"{validity_rate:.1f}%")
        
        # Crear columnas para la interfaz
        col_sidebar, col_main = st.columns([1, 3])
        
        # Variable global para el retorno del mes
        monthly_return = 0.0
        
        with col_sidebar:
            st.markdown("### 📅 Navegación por Fechas")
            
            # Obtener fechas únicas ordenadas
            unique_dates = sorted(picks_df['Date'].unique(), reverse=True)
            
            # Selector de fecha con key único para mantener estado
            selected_date = st.selectbox(
                "Selecciona una fecha:",
                unique_dates,
                index=0,
                help="Muestra los picks seleccionados en esta fecha",
                key="historical_date_selector"
            )
            
            # Mostrar información de la fecha seleccionada
            date_picks = picks_df[picks_df['Date'] == selected_date]
            st.info(f"🎯 {len(date_picks)} picks seleccionados el {selected_date}")
            
            # Calcular rentabilidad del mes - VERSIÓN CORREGIDA
            try:
                selected_dt = pd.Timestamp(selected_date)
                bt_index = pd.to_datetime(bt_results.index)
                
                # Buscar la posición exacta de selected_date en el índice
                if selected_dt in bt_index:
                    current_idx = bt_index.get_loc(selected_dt)
                    
                    if current_idx < len(bt_index) - 1:
                        # El retorno del mes es el retorno de la estrategia
                        monthly_return = bt_results['Returns'].iloc[current_idx + 1]
                        
                        # Obtener comisión para mostrar el desglose
                        commission = st.session_state.backtest_params.get('commission', 0.003) if st.session_state.backtest_params else 0.003
                        
                        st.metric(
                            "📈 Retorno del Mes",
                            f"{monthly_return:.2%}",
                            delta=f"{monthly_return:.2%}" if monthly_return != 0 else None
                        )
                        
                        # Mostrar desglose
                        with st.expander("🔍 Desglose del Retorno", expanded=False):
                            st.write(f"Fecha picks: {selected_date}")
                            st.write(f"Equity inicial: ${bt_results['Equity'].iloc[current_idx]:,.2f}")
                            st.write(f"Equity final: ${bt_results['Equity'].iloc[current_idx + 1]:,.2f}")
                            st.write(f"**Retorno neto: {monthly_return:.4%}**")
                            st.write(f"Comisión aplicada: -{commission:.2%}")
                            st.write(f"Retorno bruto estimado: {monthly_return + commission:.4%}")
                    else:
                        st.warning("📅 Último mes del backtest (sin retorno futuro)")
                        monthly_return = 0.0
                else:
                    # Si no hay coincidencia exacta, buscar el más cercano
                    closest_idx = bt_index.get_indexer([selected_dt], method='nearest')[0]
                    if closest_idx < len(bt_index) - 1:
                        monthly_return = bt_results['Returns'].iloc[closest_idx + 1]
                        st.metric(
                            "📈 Retorno del Mes (aprox)",
                            f"{monthly_return:.2%}",
                            delta=f"{monthly_return:.2%}" if monthly_return != 0 else None
                        )
                        st.caption(f"⚠️ Fecha aproximada: {bt_index[closest_idx].strftime('%Y-%m-%d')}")
                    else:
                        monthly_return = 0.0
                        
            except Exception as e:
                st.error(f"Error calculando retorno: {e}")
                monthly_return = 0.0
            
            # Estadísticas rápidas de la fecha
            if not date_picks.empty:
                avg_inercia = date_picks['Inercia'].mean()
                avg_score = date_picks['ScoreAdj'].mean()
                
                st.markdown("### 📊 Estadísticas del Mes")
                st.metric("Inercia Promedio", f"{avg_inercia:.2f}")
                st.metric("Score Ajustado Promedio", f"{avg_score:.2f}")
        
        with col_main:
            # Mostrar picks de la fecha seleccionada
            st.markdown(f"### 🎯 Picks Seleccionados el {selected_date}")
            
            date_picks_display = date_picks.copy()
            
            # Calcular rentabilidad individual - VERSIÓN CORREGIDA
            try:
                # Convertir fechas a datetime para comparación
                selected_dt = pd.Timestamp(selected_date)
                
                # Calcular retorno individual para cada ticker
                # IMPORTANTE: Usar la misma lógica que el backtest
                returns_data = []
                for _, row in date_picks.iterrows():
                    ticker = row['Ticker']
                    
                    try:
                        if ticker in prices_df.columns:
                            # Convertir a mensual como hace el backtest
                            ticker_monthly = prices_df[ticker].resample('ME').last()
                            
                            # Buscar las fechas correctas
                            if selected_dt in ticker_monthly.index:
                                current_idx = ticker_monthly.index.get_loc(selected_dt)
                                
                                if current_idx < len(ticker_monthly) - 1:
                                    # Precio al final del mes de selección
                                    entry_price = ticker_monthly.iloc[current_idx]
                                    # Precio al final del siguiente mes
                                    exit_price = ticker_monthly.iloc[current_idx + 1]
                                    
                                    # Calcular retorno
                                    if pd.notna(entry_price) and pd.notna(exit_price) and entry_price != 0:
                                        individual_return = (exit_price / entry_price) - 1
                                        returns_data.append(individual_return)
                                    else:
                                        returns_data.append(None)
                                else:
                                    returns_data.append(None)
                            else:
                                # Buscar fecha más cercana
                                closest_idx = ticker_monthly.index.get_indexer([selected_dt], method='nearest')[0]
                                if closest_idx >= 0 and closest_idx < len(ticker_monthly) - 1:
                                    entry_price = ticker_monthly.iloc[closest_idx]
                                    exit_price = ticker_monthly.iloc[closest_idx + 1]
                                    
                                    if pd.notna(entry_price) and pd.notna(exit_price) and entry_price != 0:
                                        individual_return = (exit_price / entry_price) - 1
                                        returns_data.append(individual_return)
                                    else:
                                        returns_data.append(None)
                                else:
                                    returns_data.append(None)
                        else:
                            returns_data.append(None)
                    except Exception as e:
                        print(f"Error con {ticker}: {e}")
                        returns_data.append(None)
                
                # Agregar columna de retornos individuales
                date_picks_display['Retorno Individual'] = returns_data
                
                # Formatear retornos
                def format_return(val):
                    if pd.isna(val):
                        return "N/A"
                    elif val >= 0:
                        return f"+{val:.2%}"
                    else:
                        return f"{val:.2%}"
                
                # Aplicar formato condicional
                def color_returns(val):
                    if isinstance(val, str) and val != "N/A":
                        try:
                            num_val = float(val.replace('%', '').replace('+', '')) / 100
                            if num_val > 0:
                                return 'color: green; font-weight: bold'
                            elif num_val < 0:
                                return 'color: red; font-weight: bold'
                        except:
                            pass
                    return ''
                
                # Mostrar tabla con retornos individuales
                display_columns = ['Rank', 'Ticker', 'Inercia', 'ScoreAdj', 'Retorno Individual']
                if 'HistoricallyValid' in date_picks_display.columns:
                    display_columns.append('HistoricallyValid')
                
                styled_df = date_picks_display[display_columns].style.applymap(
                    color_returns, 
                    subset=['Retorno Individual']
                ).format({
                    'Inercia': '{:.2f}',
                    'ScoreAdj': '{:.2f}',
                    'Retorno Individual': format_return
                })
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Resumen de retornos y verificación
                if any(pd.notna(r) for r in returns_data):
                    valid_returns = [r for r in returns_data if pd.notna(r)]
                    if valid_returns:
                        avg_return = sum(valid_returns) / len(valid_returns)
                        positive_count = sum(1 for r in valid_returns if r > 0)
                        win_rate = positive_count / len(valid_returns) * 100
                        
                        st.markdown("### 📈 Resumen de Rentabilidad")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Retorno Promedio Picks", f"{avg_return:.2%}")
                        col2.metric("Tasa de Éxito", f"{win_rate:.1f}%")
                        col3.metric("Mejor Pick", f"{max(valid_returns):.2%}")
                        
                        # Verificación de consistencia mejorada
                        commission = st.session_state.backtest_params.get('commission', 0.003) if st.session_state.backtest_params else 0.003
                        
                        # Calcular retorno teórico con comisión
                        theoretical_return_with_commission = avg_return - commission
                        difference = abs(monthly_return - theoretical_return_with_commission)
                        
                        st.info(f"""
                        📊 **Análisis de Retornos:**
                        - Retorno promedio picks (bruto): {avg_return:.2%}
                        - Menos comisión: -{commission:.2%}
                        - Retorno teórico neto: {theoretical_return_with_commission:.2%}
                        - Retorno real estrategia: {monthly_return:.2%}
                        - Diferencia: {difference:.2%}
                        """)
                        
                        # Explicación de diferencias
                        if difference > 0.005:  # Más de 0.5% de diferencia
                            with st.expander("ℹ️ Análisis de diferencias", expanded=True):
                                st.write(f"""
                                **Diferencia de {difference:.2%} puede deberse a:**
                                
                                1. **Ponderación**: La estrategia usa {len(valid_returns)} de {len(date_picks)} picks
                                2. **Validación histórica**: Algunos tickers podrían no estar disponibles
                                3. **Redondeo y precisión**: Pequeñas diferencias en cálculos
                                
                                **Cálculo detallado:**
                                - Suma de retornos: {sum(valid_returns):.4%}
                                - Dividido entre {len(valid_returns)} = {avg_return:.4%}
                                - Menos comisión {commission:.2%} = {theoretical_return_with_commission:.4%}
                                - Retorno backtest = {monthly_return:.4%}
                                """)
                        
                        # Gráfico de barras de retornos individuales
                        fig_returns = go.Figure()
                        fig_returns.add_trace(go.Bar(
                            x=date_picks_display['Ticker'],
                            y=[r if pd.notna(r) else 0 for r in returns_data],
                            marker_color=['green' if pd.notna(r) and r > 0 else 'red' if pd.notna(r) and r < 0 else 'gray' for r in returns_data],
                            text=[format_return(r) for r in returns_data],
                            textposition='auto'
                        ))
                        fig_returns.update_layout(
                            title="Rentabilidad Individual por Ticker",
                            xaxis_title="Ticker",
                            yaxis_title="Retorno",
                            yaxis_tickformat=".1%",
                            height=400
                        )
                        st.plotly_chart(fig_returns, use_container_width=True)
                else:
                    st.warning("📅 No se pudieron calcular retornos individuales para esta fecha")
                    
            except Exception as e:
                st.error(f"Error calculando retornos individuales: {e}")
                # Mostrar tabla básica sin retornos
                display_columns = ['Rank', 'Ticker', 'Inercia', 'ScoreAdj']
                styled_df = date_picks_display[display_columns].style.format({
                    'Inercia': '{:.2f}',
                    'ScoreAdj': '{:.2f}'
                })
                st.dataframe(styled_df, use_container_width=True)
        
        # Sección adicional: Resumen general de todos los picks
        st.markdown("### 📊 Resumen General de Todos los Picks")
        
        # Tabs para diferentes vistas - CORREGIR DEFINICIÓN
        tabs = st.tabs(["📈 Por Fecha", "🏆 Top Tickers", "📉 Distribución"])
        
        with tabs[0]:  # Tab 1: Por Fecha
            # Picks por fecha
            picks_by_date = picks_df.groupby('Date').size().reset_index(name='Count')
            fig_picks = px.bar(
                picks_by_date, 
                x='Date', 
                y='Count', 
                title="Número de Picks por Fecha",
                labels={'Date': 'Fecha', 'Count': 'Número de Picks'}
            )
            fig_picks.update_layout(height=400)
            st.plotly_chart(fig_picks, use_container_width=True)
        
        with tabs[1]:  # Tab 2: Top Tickers
            # Top tickers más seleccionados
            top_tickers = picks_df['Ticker'].value_counts().head(20).reset_index()
            top_tickers.columns = ['Ticker', 'Count']
            fig_top = px.bar(
                top_tickers, 
                x='Ticker', 
                y='Count', 
                title="Top 20 Tickers Más Seleccionados",
                labels={'Ticker': 'Ticker', 'Count': 'Veces Seleccionado'}
            )
            fig_top.update_layout(height=400)
            st.plotly_chart(fig_top, use_container_width=True)
        
        with tabs[2]:  # Tab 3: Distribución
            # Distribución de Score Adjusted
            fig_score = px.histogram(
                picks_df, 
                x='ScoreAdj', 
                nbins=50, 
                title="Distribución de Score Ajustado",
                labels={'ScoreAdj': 'Score Ajustado'}
            )
            fig_score.update_layout(height=400)
            st.plotly_chart(fig_score, use_container_width=True)
            
            # Distribución de Inercia
            fig_inercia = px.histogram(
                picks_df, 
                x='Inercia', 
                nbins=50, 
                title="Distribución de Inercia Alcista",
                labels={'Inercia': 'Inercia Alcista'}
            )
            fig_inercia.update_layout(height=400)
            st.plotly_chart(fig_inercia, use_container_width=True)
        
        # Botón para descargar todos los picks
        csv = picks_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar todos los picks (CSV)",
            data=csv,
            file_name=f"picks_completos_{start_date}_{end_date}.csv",
            mime="text/csv",
            help="Descarga todos los picks del backtest con sus métricas"
        )
    
    # TOP 5 MEJORES Y PEORES TRADES - NUEVA SECCIÓN
    st.subheader("🏆 Top 5 Mejores y Peores Trades")
    
    try:
        # Calcular todos los retornos individuales de todos los picks
        all_trades = []
        
        for date in picks_df['Date'].unique():
            date_picks = picks_df[picks_df['Date'] == date]
            selected_dt = pd.Timestamp(date)
            
            for _, pick in date_picks.iterrows():
                ticker = pick['Ticker']
                
                if ticker in prices_df.columns:
                    try:
                        # Usar la misma lógica que el backtest
                        ticker_monthly = prices_df[ticker].resample('ME').last()
                        
                        if selected_dt in ticker_monthly.index:
                            current_idx = ticker_monthly.index.get_loc(selected_dt)
                            
                            if current_idx < len(ticker_monthly) - 1:
                                entry_date = ticker_monthly.index[current_idx]
                                exit_date = ticker_monthly.index[current_idx + 1]
                                
                                entry_price = ticker_monthly.iloc[current_idx]
                                exit_price = ticker_monthly.iloc[current_idx + 1]
                                
                                if pd.notna(entry_price) and pd.notna(exit_price) and entry_price != 0:
                                    trade_return = (exit_price / entry_price) - 1
                                    
                                    all_trades.append({
                                        'Date': date,
                                        'Ticker': ticker,
                                        'Entry': entry_date.strftime('%Y-%m-%d'),
                                        'Exit': exit_date.strftime('%Y-%m-%d'),
                                        'Entry Price': entry_price,
                                        'Exit Price': exit_price,
                                        'Return': trade_return,
                                        'Inercia': pick['Inercia'],
                                        'ScoreAdj': pick['ScoreAdj']
                                    })
                    except:
                        continue
        
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df = trades_df.sort_values('Return', ascending=False)
            
            # Top 5 mejores
            top_5_best = trades_df.head(5)
            
            # Top 5 peores
            top_5_worst = trades_df.tail(5).sort_values('Return')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Top 5 Mejores Trades")
                best_display = top_5_best[['Date', 'Ticker', 'Return', 'Entry Price', 'Exit Price']].copy()
                best_display['Return'] = best_display['Return'].apply(lambda x: f"{x:.2%}")
                best_display['Entry Price'] = best_display['Entry Price'].apply(lambda x: f"${x:.2f}")
                best_display['Exit Price'] = best_display['Exit Price'].apply(lambda x: f"${x:.2f}")
                
                st.dataframe(
                    best_display.style.set_properties(**{'background-color': '#d4edda', 'color': '#155724'}),
                    use_container_width=True
                )
                
                avg_best = top_5_best['Return'].mean()
                st.success(f"📈 Retorno promedio Top 5: {avg_best:.2%}")
            
            with col2:
                st.markdown("### 📉 Top 5 Peores Trades")
                worst_display = top_5_worst[['Date', 'Ticker', 'Return', 'Entry Price', 'Exit Price']].copy()
                worst_display['Return'] = worst_display['Return'].apply(lambda x: f"{x:.2%}")
                worst_display['Entry Price'] = worst_display['Entry Price'].apply(lambda x: f"${x:.2f}")
                worst_display['Exit Price'] = worst_display['Exit Price'].apply(lambda x: f"${x:.2f}")
                
                st.dataframe(
                    worst_display.style.set_properties(**{'background-color': '#f8d7da', 'color': '#721c24'}),
                    use_container_width=True
                )
                
                avg_worst = top_5_worst['Return'].mean()
                st.error(f"📉 Retorno promedio Bottom 5: {avg_worst:.2%}")
            
            # Estadísticas generales
            st.markdown("### 📊 Estadísticas Generales de Trades")
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Total Trades", len(trades_df))
            col2.metric("Win Rate", f"{(trades_df['Return'] > 0).mean():.1%}")
            col3.metric("Avg Return", f"{trades_df['Return'].mean():.2%}")
            col4.metric("Max DD Trade", f"{trades_df['Return'].min():.2%}")
            
        else:
            st.warning("No se pudieron calcular trades individuales")
            
    except Exception as e:
        st.error(f"Error calculando top trades: {e}")
    
    # SEÑALES ACTUALES (VELA EN FORMACIÓN)
    with st.expander("🔮 Señales Actuales - Vela en Formación", expanded=True):
        st.subheader("📊 Picks Prospectivos para el Próximo Mes")
        st.warning("""
        ⚠️ **IMPORTANTE**: Estas señales usan datos hasta HOY (vela en formación).
        - Son **preliminares** y pueden cambiar hasta el cierre del mes
        - En un sistema real, tomarías estas posiciones al inicio del próximo mes
        """)
        
        try:
            # Verificar que tenemos los datos necesarios
            if prices_df is not None and not prices_df.empty:
                # Intentar calcular señales actuales
                if ohlc_data is not None:
                    current_scores = inertia_score(prices_df, corte=corte, ohlc_data=ohlc_data)
                else:
                    st.warning("⚠️ Calculando sin datos OHLC (menos preciso)")
                    current_scores = inertia_score(prices_df, corte=corte, ohlc_data=None)
                
                if current_scores and "ScoreAdjusted" in current_scores and "InerciaAlcista" in current_scores:
                    score_df = current_scores["ScoreAdjusted"]
                    inercia_df = current_scores["InerciaAlcista"]
                    
                    if not score_df.empty and not inercia_df.empty:
                        # Obtener últimos valores
                        last_scores = score_df.iloc[-1].dropna()
                        last_inercia = inercia_df.iloc[-1]
                        
                        if len(last_scores) > 0:
                            # Filtrar tickers válidos
                            valid_picks = []
                            for ticker in last_scores.index:
                                if ticker in last_inercia.index:
                                    inercia_val = last_inercia[ticker]
                                    score_adj = last_scores[ticker]
                                    
                                    if inercia_val >= corte and score_adj > 0 and not np.isnan(score_adj):
                                        valid_picks.append({
                                            'ticker': ticker,
                                            'inercia': float(inercia_val),
                                            'score_adj': float(score_adj)
                                        })
                            
                            if valid_picks:
                                valid_picks = sorted(valid_picks, key=lambda x: x['score_adj'], reverse=True)
                                final_picks = valid_picks[:min(top_n, len(valid_picks))]
                                
                                current_picks = []
                                for rank, pick in enumerate(final_picks, 1):
                                    ticker = pick['ticker']
                                    precio_actual = prices_df[ticker].iloc[-1] if ticker in prices_df.columns else 0
                                    
                                    current_picks.append({
                                        'Rank': rank,
                                        'Ticker': ticker,
                                        'Inercia Alcista': pick['inercia'],
                                        'Score Ajustado': pick['score_adj'],
                                        'Precio Actual': precio_actual
                                    })
                                
                                current_picks_df = pd.DataFrame(current_picks)
                                
                                data_date = prices_df.index[-1].strftime('%Y-%m-%d')
                                st.info(f"📅 **Datos hasta**: {data_date}")
                                
                                st.subheader(f"🔥 Top {len(current_picks_df)} Picks Actuales")
                                
                                display_df = current_picks_df.copy()
                                display_df['Precio Actual'] = display_df['Precio Actual'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
                                display_df['Inercia Alcista'] = display_df['Inercia Alcista'].round(2)
                                display_df['Score Ajustado'] = display_df['Score Ajustado'].round(2)
                                
                                st.dataframe(display_df, use_container_width=True)
                                
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Picks Actuales", len(current_picks_df))
                                col2.metric("Inercia Promedio", f"{current_picks_df['Inercia Alcista'].mean():.2f}")
                                col3.metric("Score Promedio", f"{current_picks_df['Score Ajustado'].mean():.2f}")
                                
                                st.subheader("📋 Cómo Usar Estas Señales")
                                
                                # Recuperar fixed_allocation del estado
                                if st.session_state.backtest_params:
                                    fixed_allocation = st.session_state.backtest_params.get('fixed_alloc', False)
                                
                                if fixed_allocation:
                                    capital_info = f"Cada posición: 10% del capital"
                                else:
                                    capital_info = f"Distribución equitativa: {100/len(current_picks_df):.1f}% por posición"
                                
                                st.info(f"""
                                **Para Trading Real:**
                                1. 📅 Espera al cierre del mes para señales definitivas
                                2. 🔄 Recalcula el último día del mes
                                3. 📈 Toma posiciones el primer día del próximo mes
                                4. ⏰ Mantén posiciones todo el mes
                                
                                **{capital_info}**
                                """)
                            else:
                                st.warning("⚠️ No hay tickers que pasen el corte de inercia actualmente")
                        else:
                            st.warning("⚠️ No se encontraron scores válidos")
                    else:
                        st.warning("⚠️ No hay suficientes datos para calcular señales")
                else:
                    st.error("❌ No se pudieron calcular indicadores. Verifica los datos.")
            else:
                st.error("❌ No hay datos de precios disponibles para calcular señales actuales")
                
        except Exception as e:
            st.error(f"Error calculando señales actuales: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Al final del archivo, antes del else final, agrega:
if st.session_state.spy_df is not None:
    st.sidebar.success(f"✅ SPY en memoria: {len(st.session_state.spy_df)} registros")
    
else:
    if not st.session_state.backtest_completed:
        st.info("👈 Configura los parámetros y haz clic en 'Ejecutar backtest'")
        
        st.subheader("🔍 Información del Sistema")
        st.info("""
        **Características principales:**
        - ✅ Verificación histórica de constituyentes
        - ✅ Cálculos optimizados con precálculo y paralelización
        - ✅ Caché multinivel para carga instantánea
        - ✅ Comparación completa con benchmark
        - ✅ Tabla de rendimientos mensuales
        - ✅ Señales actuales con vela en formación
        - ✅ Filtros de mercado configurables
        - ✅ Top 5 mejores y peores trades
        
        **Mejoras de rendimiento:**
        - ⚡ ATR vectorizado con EWM
        - ⚡ Carga paralela de CSVs
        - ⚡ Precálculo de indicadores
        - ⚡ Caché persistente de resultados
        
        **Correcciones aplicadas:**
        - ✅ SPY siempre visible en gráficas
        - ✅ Cálculo correcto de retornos mensuales
        - ✅ Sharpe Ratio calculado correctamente con RF=2% anual
        - ✅ Análisis detallado de diferencias entre retornos
        """)
        
        cache_files = glob.glob(os.path.join(CACHE_DIR, "backtest_*.pkl"))
        if cache_files:
            st.info(f"💾 {len(cache_files)} resultados en caché disponibles para carga instantánea")
