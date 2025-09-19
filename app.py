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
import sys
import traceback

# -------------------------------------------------
# Configuración de la app
# -------------------------------------------------
st.set_page_config(
    page_title="IA Mensual Ajustada",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------
# INTENTAR IMPORTS SEGUROS
# -------------------------------------------------
backtest_available = False
try:
    from backtest import run_backtest_optimized, calculate_monthly_returns_by_year
    backtest_available = True
    st.sidebar.success("✅ Módulo backtest cargado")
except Exception as e:
    st.sidebar.error(f"❌ Error cargando backtest: {e}")

data_loader_available = False
try:
    from data_loader import (
        get_constituents_at_date, 
        get_current_constituents,
        load_prices_from_csv_parallel,
        get_sp500_historical_changes,
        get_nasdaq100_historical_changes,
        debug_system_status
    )
    data_loader_available = True
    st.sidebar.success("✅ Módulo data_loader cargado")
except Exception as e:
    st.sidebar.error(f"❌ Error cargando data_loader: {e}")

# -------------------------------------------------
# FUNCIONES AUXILIARES
# -------------------------------------------------
def get_cache_key(params):
    """Genera una clave única para caché basada en parámetros"""
    param_str = str(sorted(params.items()))
    return hashlib.md5(param_str.encode()).hexdigest()

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

# -------------------------------------------------
# FUNCIÓN PARA MOSTRAR RESULTADOS
# -------------------------------------------------
def display_backtest_results(bt_results, picks_df, prices_df, benchmark_series, historical_info, 
                           index_choice, fixed_allocation, corte, top_n, ohlc_data, use_historical_verification):
    """Muestra todos los resultados del backtest"""
    
    st.success("✅ Backtest completado exitosamente")
    
    # Calcular métricas básicas
    final_equity = float(bt_results["Equity"].iloc[-1])
    initial_equity = float(bt_results["Equity"].iloc[0])
    total_return = (final_equity / initial_equity) - 1
    years = (bt_results.index[-1] - bt_results.index[0]).days / 365.25
    cagr = (final_equity / initial_equity) ** (1/years) - 1 if years > 0 else 0
    max_drawdown = float(bt_results["Drawdown"].min())
    
    st.subheader("📊 Métricas de la Estrategia")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Equity Final", f"${final_equity:,.0f}")
    col2.metric("Retorno Total", f"{total_return:.2%}")
    col3.metric("CAGR", f"{cagr:.2%}")
    col4.metric("Max Drawdown", f"{max_drawdown:.2%}")
    
    # Gráfico de equity
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=bt_results.index,
        y=bt_results["Equity"],
        mode='lines',
        name='Estrategia',
        line=dict(width=3, color='blue'),
        hovertemplate='<b>%{y:,.0f}</b><br>%{x}<extra></extra>'
    ))
    
    fig_equity.update_layout(
        title="Evolución del Equity",
        xaxis_title="Fecha",
        yaxis_title="Equity ($)",
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Información sobre verificación histórica
    if historical_info and historical_info.get('has_historical_data', False):
        st.info("✅ Este backtest incluye verificación histórica de constituyentes")
    elif use_historical_verification:
        st.warning("⚠️ Verificación histórica solicitada pero no se encontraron datos históricos")
    else:
        st.warning("⚠️ Este backtest NO incluye verificación histórica (posible sesgo de supervivencia)")

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
    default_end = datetime.today().date()
    default_start = default_end - timedelta(days=365*2)
    
    end_date = st.sidebar.date_input("Fecha final", value=default_end)
    start_date = st.sidebar.date_input("Fecha inicial", value=default_start)
    
    if start_date >= end_date:
        st.sidebar.warning("⚠️ Fecha inicial debe ser anterior a la fecha final")
        start_date = end_date - timedelta(days=365*2)
        
    st.sidebar.info(f"📅 Rango: {start_date} a {end_date}")
    
except Exception as e:
    st.sidebar.error(f"❌ Error configurando fechas: {e}")
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=365*2)

# Parámetros básicos
top_n = st.sidebar.slider("Número de activos", 5, 30, 10)
commission = st.sidebar.number_input("Comisión por operación (%)", 0.0, 1.0, 0.3) / 100
corte = st.sidebar.number_input("Corte de score", 0, 1000, 680)
use_historical_verification = st.sidebar.checkbox("🕐 Usar verificación histórica", value=False)

st.sidebar.subheader("⚙️ Opciones de Estrategia")
fixed_allocation = st.sidebar.checkbox("💰 Asignar 10% capital a cada acción", value=False)

st.sidebar.subheader("🛡️ Filtros de Mercado")
use_roc_filter = st.sidebar.checkbox("📉 ROC 12 meses del SPY < 0", value=False)
use_sma_filter = st.sidebar.checkbox("📊 Precio SPY < SMA 10 meses", value=False)

# Botones
run_button = st.sidebar.button("🏃 Ejecutar backtest", type="primary")

# Botón de debug en sidebar
if st.sidebar.button("🔍 Diagnóstico del Sistema"):
    with st.expander("📊 Diagnóstico Completo", expanded=True):
        if data_loader_available:
            debug_system_status()
        else:
            st.error("❌ data_loader no está disponible")

# -------------------------------------------------
# CONSTANTES
# -------------------------------------------------
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------------------------------
# Main content
# -------------------------------------------------

if run_button:
    if not data_loader_available:
        st.error("❌ data_loader no está disponible. No se puede ejecutar el backtest.")
        st.stop()
    
    if not backtest_available:
        st.error("❌ backtest no está disponible. No se puede ejecutar el backtest.")
        st.stop()
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Obtener constituyentes usando SOLO datos locales
        status_text.text("📥 Obteniendo constituyentes...")
        progress_bar.progress(10)
        
        try:
            all_tickers_data, error = get_constituents_cached(index_choice, start_date, end_date)
            if error:
                st.warning(f"Advertencia: {error}")
            
            if not all_tickers_data or 'tickers' not in all_tickers_data or not all_tickers_data['tickers']:
                st.error("No se encontraron tickers válidos")
                
                # Mostrar debug automáticamente
                with st.expander("🔍 Diagnóstico del Sistema", expanded=True):
                    debug_system_status()
                
                st.stop()
            
            tickers = list(dict.fromkeys(all_tickers_data['tickers']))
            st.success(f"✅ Obtenidos {len(tickers)} tickers únicos")
            
        except Exception as e:
            st.error(f"❌ Error obteniendo constituyentes: {e}")
            
            # Mostrar debug automáticamente
            with st.expander("🔍 Diagnóstico del Sistema", expanded=True):
                debug_system_status()
            
            st.stop()
        
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
            
            changes_data = load_historical_changes_cached(index_choice)
            
            if not changes_data.empty:
                historical_info = {
                    'changes_data': changes_data, 
                    'has_historical_data': True
                }
                st.success(f"✅ Cargados {len(changes_data)} cambios históricos")
            else:
                st.warning("⚠️ No se encontraron datos históricos")
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
        
        # Guardar resultados en session_state
        st.session_state.bt_results = bt_results
        st.session_state.picks_df = picks_df
        st.session_state.prices_df = prices_df
        st.session_state.ohlc_data = ohlc_data
        st.session_state.benchmark_series = benchmark_series
        st.session_state.spy_df = spy_df
        st.session_state.historical_info = historical_info
        st.session_state.index_choice = index_choice
        st.session_state.fixed_allocation = fixed_allocation
        st.session_state.corte = corte
        st.session_state.top_n = top_n
        st.session_state.use_historical_verification = use_historical_verification
        
        status_text.empty()
        progress_bar.empty()
        
        # MOSTRAR RESULTADOS
        if "bt_results" in st.session_state and st.session_state.bt_results is not None:
            display_backtest_results(
                st.session_state.bt_results,
                st.session_state.picks_df,
                st.session_state.prices_df,
                st.session_state.benchmark_series,
                st.session_state.historical_info,
                st.session_state.index_choice,
                st.session_state.fixed_allocation,
                st.session_state.corte,
                st.session_state.top_n,
                st.session_state.ohlc_data,
                st.session_state.get('use_historical_verification', False)
            )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
        
        # Mostrar debug en caso de error
        with st.expander("🔍 Diagnóstico del Sistema", expanded=True):
            if data_loader_available:
                debug_system_status()

# SI NO SE PULSA EL BOTÓN PERO HAY DATOS EN SESSION_STATE
elif "bt_results" in st.session_state and "picks_df" in st.session_state:
    st.info("📊 Mostrando resultados del último backtest.")
    
    # MOSTRAR LOS MISMOS RESULTADOS
    display_backtest_results(
        st.session_state.bt_results,
        st.session_state.picks_df,
        st.session_state.get('prices_df'),
        st.session_state.get('benchmark_series'),
        st.session_state.get('historical_info'),
        st.session_state.get('index_choice', 'SP500'),
        st.session_state.get('fixed_allocation', False),
        st.session_state.get('corte', 680),
        st.session_state.get('top_n', 10),
        st.session_state.get('ohlc_data'),
        st.session_state.get('use_historical_verification', False)
    )

else:
    st.info("👈 Configura los parámetros y haz clic en 'Ejecutar backtest'")
    
    st.subheader("🔧 Estado del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Módulos:**")
        st.write(f"✅ Streamlit: {st.__version__}")
        st.write(f"✅ Pandas: {pd.__version__}")
        st.write(f"✅ Numpy: {np.__version__}")
        st.write(f"{'✅' if backtest_available else '❌'} Backtest")
        st.write(f"{'✅' if data_loader_available else '❌'} Data Loader")
    
    with col2:
        st.write("**Archivos críticos:**")
        st.write(f"{'✅' if os.path.exists('data') else '❌'} Directorio data/")
        st.write(f"{'✅' if os.path.exists('sp500_changes.csv') else '❌'} sp500_changes.csv")
        st.write(f"{'✅' if os.path.exists('ndx_changes.csv') else '❌'} ndx_changes.csv")
        st.write(f"{'✅' if os.path.exists('data/AAPL.csv') else '❌'} data/AAPL.csv")
        st.write(f"{'✅' if os.path.exists('backtest.py') else '❌'} backtest.py")
    
    if data_loader_available:
        st.write("**Prueba rápida:**")
        if st.button("🧪 Probar obtención de datos"):
            try:
                result = get_current_constituents("SP500")
                if result and 'tickers' in result:
                    st.success(f"✅ Obtenidos {len(result['tickers'])} tickers del S&P 500")
                    st.write(f"Primeros 10: {result['tickers'][:10]}")
                else:
                    st.error("❌ No se pudieron obtener tickers")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Información adicional en sidebar
st.sidebar.markdown("---")
st.sidebar.write("**Debug Info:**")
st.sidebar.write(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
st.sidebar.write(f"Archivos CSV: {len(glob.glob('data/*.csv'))}")
