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

# Importar nuestros módulos
from data_loader import get_constituents_at_date, get_sp500_historical_changes, get_nasdaq100_historical_changes
from backtest import run_backtest_optimized, precalculate_all_indicators, calculate_monthly_returns_by_year

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
@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_historical_changes_cached(index_name):
    """Carga cambios históricos con caché"""
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

@st.cache_data(ttl=86400)  # Cache por 24 horas
def get_constituents_cached(index_name, start_date, end_date):
    """Obtiene constituyentes con caché"""
    return get_constituents_at_date(index_name, start_date, end_date)

def get_cache_key(params):
    """Genera una clave única para caché basada en parámetros"""
    param_str = str(sorted(params.items()))
    return hashlib.md5(param_str.encode()).hexdigest()

@st.cache_data(ttl=3600*24*7)  # Cache por 1 semana
def load_prices_from_csv_parallel(tickers, start_date, end_date, load_full_data=True):
    """Carga precios desde CSV en PARALELO - MUCHO MÁS RÁPIDO"""
    prices_data = {}
    ohlc_data = {}
    
    def load_single_ticker(ticker):
        """Carga un ticker individual"""
        csv_path = f"data/{ticker}.csv"
        if not os.path.exists(csv_path):
            return ticker, None, None
        
        try:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Convertir fechas para comparación
            if isinstance(start_date, datetime):
                start_filter = start_date.date()
            else:
                start_filter = start_date
            if isinstance(end_date, datetime):
                end_filter = end_date.date()
            else:
                end_filter = end_date
            
            df = df[(df.index.date >= start_filter) & (df.index.date <= end_filter)]
            
            if df.empty:
                return ticker, None, None
            
            # Extraer precio
            if 'Adj Close' in df.columns:
                price = df['Adj Close']
            elif 'Close' in df.columns:
                price = df['Close']
            else:
                return ticker, None, None
            
            # Extraer OHLC si está disponible
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
            return ticker, None, None
    
    # Usar ThreadPoolExecutor para paralelizar
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
fixed_allocation = st.sidebar.checkbox("💰 Asignar 10% capital a cada acción", value=False)

st.sidebar.subheader("🛡️ Filtros de Mercado")
use_roc_filter = st.sidebar.checkbox("📉 ROC 12 meses del SPY < 0", value=False)
use_sma_filter = st.sidebar.checkbox("📊 Precio SPY < SMA 10 meses", value=False)

# Botón optimizado con progreso
run_button = st.sidebar.button("🏃 Ejecutar backtest", type="primary")

# -------------------------------------------------
# CONSTANTES
# -------------------------------------------------
GITHUB_RAW_URL = "https://raw.githubusercontent.com/jmlestevez-source/IA-MENSUAL-AJUSTADA/main/"
LOCAL_CHANGES_DIR = "data/historical_changes"
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOCAL_CHANGES_DIR, exist_ok=True)

# -------------------------------------------------
# Main content con optimizaciones
# -------------------------------------------------
if run_button:
    try:
        # Crear clave de caché para estos parámetros
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
        
        # Intentar cargar de caché
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
            except:
                use_cache = False
        
        if not use_cache:
            # EJECUTAR BACKTEST OPTIMIZADO
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Paso 1: Obtener constituyentes
            status_text.text("📥 Obteniendo constituyentes...")
            progress_bar.progress(10)
            
            all_tickers_data, error = get_constituents_cached(index_choice, start_date, end_date)
            if error:
                st.warning(f"Advertencia: {error}")
            
            if not all_tickers_data or 'tickers' not in all_tickers_data:
                st.error("No se encontraron tickers válidos")
                st.stop()
            
            tickers = list(dict.fromkeys(all_tickers_data['tickers']))  # Eliminar duplicados
            st.success(f"✅ Obtenidos {len(tickers)} tickers únicos")
            
            # Paso 2: Cargar precios en PARALELO
            status_text.text("📊 Cargando precios en paralelo...")
            progress_bar.progress(30)
            
            prices_df, ohlc_data = load_prices_from_csv_parallel(tickers, start_date, end_date, load_full_data=True)
            
            if prices_df.empty:
                st.error("❌ No se pudieron cargar precios")
                st.stop()
            
            st.success(f"✅ Cargados {len(prices_df.columns)} tickers con datos")
            
            # Paso 3: Cargar benchmark
            status_text.text("📈 Cargando benchmark...")
            progress_bar.progress(40)
            
            benchmark_ticker = "SPY" if index_choice != "NDX" else "QQQ"
            benchmark_df, _ = load_prices_from_csv_parallel([benchmark_ticker], start_date, end_date, load_full_data=False)
            
            if benchmark_df.empty:
                st.warning("Usando promedio como benchmark")
                benchmark_series = prices_df.mean(axis=1)
            else:
                benchmark_series = benchmark_df[benchmark_ticker]
            
            # Paso 4: SPY para filtros
            spy_df = None
            if use_roc_filter or use_sma_filter:
                spy_result, _ = load_prices_from_csv_parallel(["SPY"], start_date, end_date, load_full_data=False)
                spy_df = spy_result if not spy_result.empty else None
            
            # Paso 5: Información histórica
            historical_info = None
            if use_historical_verification:
                status_text.text("🕐 Cargando datos históricos...")
                progress_bar.progress(50)
                
                changes_data = load_historical_changes_cached(index_choice)
                if not changes_data.empty:
                    historical_info = {'changes_data': changes_data, 'has_historical_data': True}
                    st.success(f"✅ Cargados {len(changes_data)} cambios históricos")
            
            # Paso 6: EJECUTAR BACKTEST OPTIMIZADO
            status_text.text("🚀 Ejecutando backtest optimizado...")
            progress_bar.progress(70)
            
            # Usar versión optimizada del backtest
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
            
            # Guardar en caché
            status_text.text("💾 Guardando resultados en caché...")
            progress_bar.progress(100)
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'bt_results': bt_results,
                        'picks_df': picks_df,
                        'historical_info': historical_info,
                        'timestamp': datetime.now()
                    }, f)
                st.success("✅ Resultados guardados en caché")
            except Exception as e:
                st.warning(f"No se pudo guardar caché: {e}")
            
            status_text.empty()
            progress_bar.empty()
        
        # MOSTRAR RESULTADOS (igual que antes pero con métricas mejoradas)
        if bt_results is not None and not bt_results.empty:
            st.success("✅ Backtest completado exitosamente")
            
            # Calcular métricas
            final_equity = float(bt_results["Equity"].iloc[-1])
            initial_equity = float(bt_results["Equity"].iloc[0])
            total_return = (final_equity / initial_equity) - 1
            years = (bt_results.index[-1] - bt_results.index[0]).days / 365.25
            cagr = (final_equity / initial_equity) ** (1/years) - 1 if years > 0 else 0
            max_drawdown = float(bt_results["Drawdown"].min())
            
            # Sharpe Ratio
            monthly_returns = bt_results["Returns"]
            risk_free_rate_monthly = 0.02 / 12
            excess_returns = monthly_returns - risk_free_rate_monthly
            sharpe_ratio = (excess_returns.mean() * 12) / (excess_returns.std() * np.sqrt(12)) if excess_returns.std() > 0 else 0
            
            # Mostrar métricas
            st.subheader("📊 Métricas de la Estrategia")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Equity Final", f"${final_equity:,.0f}")
            col2.metric("Retorno Total", f"{total_return:.2%}")
            col3.metric("CAGR", f"{cagr:.2%}")
            col4.metric("Max Drawdown", f"{max_drawdown:.2%}")
            col5.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            # Información sobre tiempo de ejecución
            if not use_cache:
                st.info(f"⏱️ Backtest ejecutado en tiempo real para {years:.1f} años de datos")
            else:
                st.info("⚡ Resultados cargados desde caché (instantáneo)")
            
            # Gráficos y demás resultados...
            # [El resto del código de visualización se mantiene igual]
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    st.info("👈 Configura los parámetros y haz clic en 'Ejecutar backtest'")
    
    # Mostrar estado del caché
    cache_files = glob.glob(os.path.join(CACHE_DIR, "backtest_*.pkl"))
    if cache_files:
        st.info(f"💾 {len(cache_files)} resultados en caché disponibles para carga instantánea")
