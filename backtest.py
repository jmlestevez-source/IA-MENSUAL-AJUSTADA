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

# Importar nuestros módulos - IMPORTANTE: inertia_score viene de backtest
from data_loader import get_constituents_at_date, get_sp500_historical_changes, get_nasdaq100_historical_changes, generate_removed_tickers_summary

# ---------------------------
# Helper: verificar archivos históricos al inicio
# ---------------------------
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
# FUNCIONES DE CACHÉ OPTIMIZADAS
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_historical_changes_cached(index_name, force_reload=False):
    """Carga cambios históricos con caché"""
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
    else:
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

# ---------------------------
# Corrección crítica: función de carga de CSVs (paralela)
# - maneja tz-aware datetimes al leer CSV (utc -> tz_convert(None))
# - normaliza tickers ('.' -> '-')
# - respeta columnas Close/Adj Close
# ---------------------------
@st.cache_data(ttl=3600*24*7)
def load_prices_from_csv_parallel(tickers, start_date, end_date, load_full_data=True):
    """Carga precios desde CSV en PARALELO (corrigida para manejar zonas horarias)"""
    prices_data = {}
    ohlc_data = {}
    
    def parse_dates_tz_aware(col):
        # convierte a UTC y elimina tz (naive datetime), evita errores con pandas/ numpy datetime64
        try:
            return pd.to_datetime(col, utc=True).tz_convert(None)
        except Exception:
            return pd.to_datetime(col, utc=True).tz_convert(None)
    
    def load_single_ticker(ticker):
        # Normalizar nombre del ticker (igual que en data_loader.py)
        clean_ticker = str(ticker).strip().upper().replace('.', '-')
        csv_path = f"data/{clean_ticker}.csv"
        if not os.path.exists(csv_path):
            return ticker, None, None
        
        try:
            # Leer CSV eliminando zona horaria directamente:
            # Usamos date_parser para forzar utc then tz_convert(None)
            df = pd.read_csv(
                csv_path,
                index_col="Date",
                parse_dates=True,
                date_parser=lambda col: pd.to_datetime(col, utc=True).tz_convert(None)
            )
            
            # Asegurar índice datetime naive
            df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, 'tz', None) else pd.to_datetime(df.index)
            
            # Filtros de fecha
            start_filter = start_date if not isinstance(start_date, datetime) else start_date.date()
            end_filter = end_date if not isinstance(end_date, datetime) else end_date.date()
            # start_filter and end_filter expected as date objects
            if isinstance(start_filter, datetime):
                start_filter = start_filter.date()
            if isinstance(end_filter, datetime):
                end_filter = end_filter.date()
            
            df = df[(df.index.date >= start_filter) & (df.index.date <= end_filter)]
            
            if df.empty:
                return ticker, None, None
            
            # Preferimos Adj Close si existe (seguridad), si no Close
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
            # Mostrar cuál ticker falla y por qué (útil en debugging)
            print(f"⚠️ Error leyendo {ticker}: {e}")
            return ticker, None, None
    
    # Ejecutar en paralelo
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(load_single_ticker, t) for t in tickers]
        for future in futures:
            ticker, price, ohlc = future.result()
            if price is not None:
                prices_data[ticker] = price
            if ohlc is not None:
                ohlc_data[ticker] = ohlc
    
    if prices_data:
        prices_df = pd.DataFrame(prices_data)
        # rellenar gaps
        prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
        return prices_df, ohlc_data
    else:
        return pd.DataFrame(), {}

# -------------------------------------------------
# Util: crear enlace de descarga para DF
# -------------------------------------------------
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

# -------------------------------------------------
# CONSTANTES Y DIRECTORIOS
# -------------------------------------------------
GITHUB_RAW_URL = "https://raw.githubusercontent.com/jmlestevez-source/IA-MENSUAL-AJUSTADA/main/"
LOCAL_CHANGES_DIR = "data/historical_changes"
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOCAL_CHANGES_DIR, exist_ok=True)

# -------------------------------------------------
# MAIN: lógica principal al pulsar el botón
# -------------------------------------------------
if run_button:
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
                        # Cargar en session_state para evitar perdida al interactuar
                        st.session_state['bt_results'] = cached_data['bt_results']
                        st.session_state['picks_df'] = cached_data['picks_df']
                        st.session_state['historical_info'] = cached_data.get('historical_info')
                        st.session_state['prices_df'] = cached_data.get('prices_df')
                        st.session_state['ohlc_data'] = cached_data.get('ohlc_data')
                        st.session_state['benchmark_series'] = cached_data.get('benchmark_series')
                        st.session_state['spy_df'] = cached_data.get('spy_df')
            except Exception:
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
                # benchmark_df contiene columna 'SPY' (clave original)
                # Si fue cargado con clave original, use esa. Si no, el nombre puede ser diferente.
                if benchmark_ticker in benchmark_df.columns:
                    benchmark_series = benchmark_df[benchmark_ticker]
                else:
                    # tomar la primera columna
                    benchmark_series = benchmark_df.iloc[:, 0]
            
            # SPY para filtros
            spy_df = None
            if use_roc_filter or use_sma_filter:
                spy_result, _ = load_prices_from_csv_parallel(["SPY"], start_date, end_date, load_full_data=False)
                spy_df = spy_result if not spy_result.empty else None
            
            # Información histórica
            historical_info = None
            if use_historical_verification:
                status_text.text("🕐 Cargando datos históricos...")
                progress_bar.progress(50)
                
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
            
            # Guardar en caché y en session_state para evitar pérdida al interactuar
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
            
            # Guarda en session_state clave 'last_backtest' con todo lo necesario
            st.session_state['bt_results'] = bt_results
            st.session_state['picks_df'] = picks_df
            st.session_state['historical_info'] = historical_info
            st.session_state['prices_df'] = prices_df
            st.session_state['ohlc_data'] = ohlc_data
            st.session_state['benchmark_series'] = benchmark_series
            st.session_state['spy_df'] = spy_df
            
            status_text.empty()
            progress_bar.empty()
    
    except Exception as e:
        st.error(f"Error ejecutando backtest: {e}")
        st.stop()

# -------------------------------------------------
# Mostrar resultados guardados (tomados preferentemente de session_state)
# -------------------------------------------------
bt_results = st.session_state.get('bt_results', None)
picks_df = st.session_state.get('picks_df', None)
historical_info = st.session_state.get('historical_info', None)
prices_df = st.session_state.get('prices_df', None)
ohlc_data = st.session_state.get('ohlc_data', None)
benchmark_series = st.session_state.get('benchmark_series', None)
spy_df = st.session_state.get('spy_df', None)

# Si no hay resultados aún, mostrar aviso
if bt_results is None or bt_results.empty:
    st.warning("⚠️ Ejecuta un backtest para ver resultados y picks históricos.")
    st.stop()

# -------------------------------------------------
# MOSTRAR RESULTADOS (Métricas, Gráficos, Tabla de Rendimientos)
# -------------------------------------------------
try:
    st.success("✅ Backtest completado exitosamente (mostrando resultados)")
    
    final_equity = float(bt_results["Equity"].iloc[-1])
    initial_equity = float(bt_results["Equity"].iloc[0])
    total_return = (final_equity / initial_equity) - 1
    years = (bt_results.index[-1] - bt_results.index[0]).days / 365.25
    cagr = (final_equity / initial_equity) ** (1/years) - 1 if years > 0 else 0
    max_drawdown = float(bt_results["Drawdown"].min())
    
    monthly_returns = bt_results["Returns"]
    risk_free_rate_monthly = 0.02 / 12
    excess_returns = monthly_returns - risk_free_rate_monthly
    sharpe_ratio = (excess_returns.mean() * 12) / (excess_returns.std() * np.sqrt(12)) if excess_returns.std() > 0 else 0
    volatility = float(monthly_returns.std() * np.sqrt(12))
    
    st.subheader("📊 Métricas de la Estrategia")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Equity Final", f"${final_equity:,.0f}")
    col2.metric("Retorno Total", f"{total_return:.2%}")
    col3.metric("CAGR", f"{cagr:.2%}")
    col4.metric("Max Drawdown", f"{max_drawdown:.2%}")
    col5.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    # Benchmark metrics
    bench_equity = None
    bench_drawdown = None
    bench_sharpe = 0
    bench_final = initial_equity
    bench_total_return = 0
    bench_cagr = 0
    bench_max_dd = 0
    
    if benchmark_series is not None and not benchmark_series.empty:
        try:
            bench_returns = benchmark_series.pct_change().fillna(0)
            bench_equity = initial_equity * (1 + bench_returns).cumprod()
            bench_drawdown = (bench_equity / bench_equity.cummax() - 1)
            
            bench_final = float(bench_equity.iloc[-1])
            bench_initial = float(bench_equity.iloc[0]) if bench_equity.iloc[0] != 0 else initial_equity
            bench_total_return = (bench_final / bench_initial) - 1
            
            if years > 0:
                bench_cagr = (bench_final / bench_initial) ** (1/years) - 1
            
            bench_max_dd = float(bench_drawdown.min())
            
            if len(bench_returns) > len(bt_results) * 15:
                bench_returns_monthly = bench_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            else:
                bench_returns_monthly = bench_returns
            
            bench_excess_returns = bench_returns_monthly - risk_free_rate_monthly
            if bench_excess_returns.std() != 0:
                bench_sharpe = (bench_excess_returns.mean() * 12) / (bench_excess_returns.std() * np.sqrt(12))
                
        except Exception as e:
            st.warning(f"Error calculando benchmark: {e}")
    
    benchmark_name = "SPY" if index_choice != "NDX" else "QQQ"
    
    st.subheader(f"📊 Métricas del Benchmark ({benchmark_name})")
    col1b, col2b, col3b, col4b, col5b = st.columns(5)
    col1b.metric("Equity Final", f"${bench_final:,.0f}")
    col2b.metric("Retorno Total", f"{bench_total_return:.2%}")
    col3b.metric("CAGR", f"{bench_cagr:.2%}")
    col4b.metric("Max Drawdown", f"{bench_max_dd:.2%}")
    col5b.metric("Sharpe Ratio", f"{bench_sharpe:.2f}")
    
    # -------------------------------------------------
    # Gráfico: Equity con benchmark superpuesto (CORRECCIÓN AÑADIDA)
    # -------------------------------------------------
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=bt_results.index,
        y=bt_results["Equity"],
        mode='lines',
        name='Estrategia',
        line=dict(width=3),
        hovertemplate='<b>%{y:,.0f}</b><br>%{x}<extra></extra>'
    ))
    
    # Añadir benchmark si está disponible: usar bench_equity calculado arriba; fallback a SPY en prices_df
    if bench_equity is not None:
        common_index = bt_results.index.intersection(bench_equity.index)
        if len(common_index) > 0:
            bench_aligned = bench_equity.loc[common_index]
            fig_equity.add_trace(go.Scatter(
                x=bench_aligned.index,
                y=bench_aligned.values,
                mode='lines',
                name=f'Benchmark ({benchmark_name})',
                line=dict(width=2, dash='dash'),
                hovertemplate='<b>%{y:,.0f}</b><br>%{x}<extra></extra>'
            ))
    else:
        # Intentar superponer SPY serie si existe en prices_df
        if prices_df is not None and "SPY" in prices_df.columns:
            try:
                spy_series = prices_df["SPY"]
                spy_equity = (spy_series / spy_series.iloc[0]) * initial_equity
                common_index = bt_results.index.intersection(spy_equity.index)
                if len(common_index) > 0:
                    spy_aligned = spy_equity.loc[common_index]
                    fig_equity.add_trace(go.Scatter(
                        x=spy_aligned.index,
                        y=spy_aligned.values,
                        mode='lines',
                        name='SPY (from CSV)',
                        line=dict(width=2, dash='dash'),
                        hovertemplate='<b>%{y:,.0f}</b><br>%{x}<extra></extra>'
                    ))
            except Exception:
                pass
    
    fig_equity.update_layout(
        title="Evolución del Equity",
        xaxis_title="Fecha",
        yaxis_title="Equity ($)",
        hovermode='x unified',
        height=500,
        showlegend=True,
        yaxis_type="log"
    )
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # -------------------------------------------------
    # Gráfico: Drawdown con benchmark superpuesto (CORRECCIÓN AÑADIDA)
    # -------------------------------------------------
    if "Drawdown" in bt_results.columns:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=bt_results.index,
            y=bt_results["Drawdown"] * 100,
            mode='lines',
            name='Drawdown Estrategia',
            fill='tozeroy',
            line=dict(width=2, color='red'),
            hovertemplate='<b>%{y:.2f}%</b><br>%{x}<extra></extra>'
        ))
        
        if bench_equity is not None and bench_drawdown is not None:
            common_index = bt_results.index.intersection(bench_drawdown.index)
            if len(common_index) > 0:
                bench_dd_aligned = bench_drawdown.loc[common_index]
                fig_dd.add_trace(go.Scatter(
                    x=bench_dd_aligned.index,
                    y=bench_dd_aligned.values * 100,
                    mode='lines',
                    name=f'Drawdown {benchmark_name}',
                    line=dict(width=2, dash='dash'),
                    hovertemplate='<b>%{y:.2f}%</b><br>%{x}<extra></extra>'
                ))
        else:
            # fallback: SPY drawdown if available
            if prices_df is not None and "SPY" in prices_df.columns:
                try:
                    spy_series = prices_df["SPY"]
                    spy_equity = (spy_series / spy_series.iloc[0]) * initial_equity
                    spy_dd = spy_equity / spy_equity.cummax() - 1
                    common_index = bt_results.index.intersection(spy_dd.index)
                    if len(common_index) > 0:
                        spy_dd_aligned = spy_dd.loc[common_index]
                        fig_dd.add_trace(go.Scatter(
                            x=spy_dd_aligned.index,
                            y=spy_dd_aligned.values * 100,
                            mode='lines',
                            name='Drawdown SPY (from CSV)',
                            line=dict(width=2, dash='dash'),
                            hovertemplate='<b>%{y:.2f}%</b><br>%{x}<extra></extra>'
                        ))
                except Exception:
                    pass
        
        fig_dd.update_layout(
            title="Drawdown Comparativo",
            xaxis_title="Fecha",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_dd, use_container_width=True)
    
    # -------------------------------------------------
    # Tabla de rendimientos mensuales
    # -------------------------------------------------
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
        
        # Estadísticas agregadas
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
    
    # -------------------------------------------------
    # PICKS HISTÓRICOS
    # Usar session_state para evitar reinicios cuando se interactúa
    # -------------------------------------------------
    if picks_df is not None and not picks_df.empty:
        st.subheader("📊 Picks Históricos")
        
        # Estadísticas de validez histórica si aplica
        if 'HistoricallyValid' in picks_df.columns:
            total_picks = len(picks_df)
            valid_picks = picks_df['HistoricallyValid'].sum()
            validity_rate = valid_picks / total_picks * 100 if total_picks > 0 else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Picks", total_picks)
            col2.metric("Picks Válidos", valid_picks)
            col3.metric("% Validez Histórica", f"{validity_rate:.1f}%")
        
        # Navegación por fechas - cargamos fechas desde picks_df guardado en session_state
        unique_dates = sorted(picks_df['Date'].unique(), reverse=True)
        # Usar session_state para selected_date para evitar perder selección al interactuar
        if 'selected_pick_date' not in st.session_state:
            st.session_state['selected_pick_date'] = unique_dates[0] if unique_dates else None
        
        # Selector de fecha (no desencadena re-ejecución del backtest, solo cambia view)
        selected_date = st.selectbox("Selecciona una fecha:", unique_dates, index=0 if unique_dates else None)
        # Guardar selección en session_state
        st.session_state['selected_pick_date'] = selected_date
        
        # Mostrar picks para la fecha seleccionada sin volver a ejecutar backtest
        date_picks = picks_df[picks_df['Date'] == selected_date]
        st.info(f"🎯 {len(date_picks)} picks seleccionados el {selected_date}")
        
        # Estadísticas rápidas
        if not date_picks.empty:
            avg_inercia = date_picks['Inercia'].mean()
            avg_score = date_picks['ScoreAdj'].mean() if 'ScoreAdj' in date_picks.columns else None
            st.markdown("### 📊 Estadísticas del Mes")
            st.metric("Inercia Promedio", f"{avg_inercia:.2f}")
            if avg_score is not None:
                st.metric("Score Ajustado Promedio", f"{avg_score:.2f}")
        
        # Mostrar tabla y retornos individuales si es posible
        col_sidebar, col_main = st.columns([1, 3])
        with col_sidebar:
            st.markdown("### 📅 Navegación por Fechas (detalle)")
            st.write(f"Seleccionado: {selected_date}")
        
        with col_main:
            st.markdown(f"### 🎯 Picks Seleccionados el {selected_date}")
            date_picks_display = date_picks.copy()
            
            # Intentar calcular retornos individuales: usar prices_df y bt_results indices
            try:
                bt_index = pd.to_datetime(bt_results.index)
                selected_dt = pd.to_datetime(selected_date)
                
                future_dates = bt_index[bt_index > selected_dt]
                if len(future_dates) > 0:
                    next_month = future_dates[0]
                    returns_data = []
                    for _, row in date_picks.iterrows():
                        ticker = row['Ticker']
                        try:
                            if prices_df is not None and ticker in prices_df.columns:
                                entry_price = prices_df.loc[selected_dt, ticker]
                                exit_price = prices_df.loc[next_month, ticker]
                                if entry_price != 0:
                                    individual_return = (exit_price / entry_price) - 1
                                    returns_data.append(individual_return)
                                else:
                                    returns_data.append(None)
                            else:
                                returns_data.append(None)
                        except Exception:
                            returns_data.append(None)
                    
                    date_picks_display['Retorno Individual'] = returns_data
                    
                    def format_return(val):
                        if pd.isna(val):
                            return "N/A"
                        elif val >= 0:
                            return f"+{val:.2%}"
                        else:
                            return f"{val:.2%}"
                    
                    def color_returns(val):
                        if isinstance(val, str) and val != "N/A":
                            try:
                                num_val = float(val.replace('%', '').replace('+', '')) / 100
                                if num_val > 0:
                                    return 'color: green; font-weight: bold'
                                elif num_val < 0:
                                    return 'color: red; font-weight: bold'
                            except:
                                return ''
                        return ''
                    
                    display_columns = ['Rank', 'Ticker', 'Inercia', 'ScoreAdj', 'Retorno Individual'] if 'ScoreAdj' in date_picks_display.columns else ['Rank', 'Ticker', 'Inercia', 'Retorno Individual']
                    styled_df = date_picks_display[display_columns].style.applymap(
                        color_returns, 
                        subset=['Retorno Individual']
                    ).format({
                        'Inercia': '{:.2f}',
                        'ScoreAdj': '{:.2f}',
                        'Retorno Individual': format_return
                    })
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Resumen de retornos
                    valid_returns = [r for r in returns_data if r is not None and not pd.isna(r)]
                    if valid_returns:
                        avg_return = sum(valid_returns) / len(valid_returns)
                        positive_count = sum(1 for r in valid_returns if r > 0)
                        win_rate = positive_count / len(valid_returns) * 100
                        st.markdown("### 📈 Resumen de Rentabilidad")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Retorno Promedio", f"{avg_return:.2%}")
                        col2.metric("Tasa de Éxito", f"{win_rate:.1f}%")
                        col3.metric("Mejor Pick", f"{max(valid_returns):.2%}")
                        
                        fig_returns = go.Figure()
                        fig_returns.add_trace(go.Bar(
                            x=date_picks_display['Ticker'],
                            y=[r if (r is not None and not pd.isna(r)) else 0 for r in returns_data],
                            text=[format_return(r) for r in returns_data],
                            textposition='auto',
                            marker_color=['green' if (r is not None and r > 0) else 'red' if (r is not None and r < 0) else 'gray' for r in returns_data]
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
                    # último mes, no hay retorno futuro
                    display_columns = ['Rank', 'Ticker', 'Inercia', 'ScoreAdj'] if 'ScoreAdj' in date_picks_display.columns else ['Rank', 'Ticker', 'Inercia']
                    styled_df = date_picks_display[display_columns].style.format({
                        'Inercia': '{:.2f}',
                        'ScoreAdj': '{:.2f}'
                    })
                    st.dataframe(styled_df, use_container_width=True)
                    st.warning("📅 Este es el último mes del backtest, no hay datos de retorno futuro")
            except Exception as e:
                st.error(f"Error calculando retornos individuales: {e}")
                display_columns = ['Rank', 'Ticker', 'Inercia', 'ScoreAdj'] if 'ScoreAdj' in date_picks_display.columns else ['Rank', 'Ticker', 'Inercia']
                styled_df = date_picks_display[display_columns].style.format({
                    'Inercia': '{:.2f}',
                    'ScoreAdj': '{:.2f}'
                })
                st.dataframe(styled_df, use_container_width=True)
    
    else:
        st.info("No hay picks históricos para mostrar.")

    # -------------------------------------------------
    # Resumen general de picks (tabs)
    # -------------------------------------------------
    st.markdown("### 📊 Resumen General de Todos los Picks")
    if picks_df is not None and not picks_df.empty:
        tab1, tab2, tab3 = st.tabs(["📈 Por Fecha", "🏆 Top Tickers", "📉 Distribución"])
        
        with tab1:
            picks_by_date = picks_df.groupby('Date').size().reset_index(name='Count')
            fig_picks = px.bar(
                picks_by_date, 
                x='Date', 
                y='Count', 
                title="Número de Picks por Fecha",
                labels={'Date': 'Fecha', 'Count': 'Número de Picks'}
            )
            st.plotly_chart(fig_picks, use_container_width=True)
        
        with tab2:
            top_tickers = picks_df.groupby('Ticker').size().reset_index(name='Count').sort_values('Count', ascending=False).head(50)
            st.dataframe(top_tickers, use_container_width=True)
        
        with tab3:
            fig_dist = px.histogram(picks_df, x='Rank', nbins=20, title="Distribución de Ranks")
            st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("No hay datos de picks para los tabs.")

    # Información histórica sobre verificación
    if historical_info and historical_info.get('has_historical_data', False):
        st.info("✅ Este backtest incluye verificación histórica de constituyentes")
    elif use_historical_verification:
        st.warning("⚠️ Verificación histórica solicitada pero no se encontraron datos históricos")
    else:
        st.warning("⚠️ Este backtest NO incluye verificación histórica (posible sesgo de supervivencia)")
    
except Exception as e:
    st.error(f"Error mostrando resultados: {e}")
    import traceback
    traceback.print_exc()
