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
# FUNCIONES DE DIAGNÓSTICO INTEGRADAS
# -------------------------------------------------
def diagnose_system():
    """Función de diagnóstico completa integrada"""
    st.write("### 🔍 DIAGNÓSTICO COMPLETO DEL SISTEMA")
    
    # 1. Verificar directorio actual
    st.write("**1. Directorio actual:**")
    current_dir = os.getcwd()
    st.write(f"📁 {current_dir}")
    
    # 2. Verificar estructura de archivos
    st.write("**2. Estructura de archivos:**")
    
    # Verificar directorio data
    data_dir = "data"
    data_exists = os.path.exists(data_dir)
    st.write(f"📂 data/: {'✅' if data_exists else '❌'}")
    
    if data_exists:
        try:
            data_files = os.listdir(data_dir)
            csv_files = [f for f in data_files if f.endswith('.csv')]
            st.write(f"   📊 Archivos CSV en data/: {len(csv_files)}")
            if csv_files:
                st.write(f"   Primeros 10: {csv_files[:10]}")
        except Exception as e:
            st.write(f"   ❌ Error listando data/: {e}")
    
    # 3. Verificar archivos de cambios históricos
    st.write("**3. Archivos de cambios históricos:**")
    change_files = [
        'sp500_changes.csv',
        'ndx_changes.csv',
        'data/sp500_changes.csv',
        'data/ndx_changes.csv'
    ]
    
    for file_path in change_files:
        exists = os.path.exists(file_path)
        st.write(f"📄 {file_path}: {'✅' if exists else '❌'}")
        if exists:
            try:
                df = pd.read_csv(file_path)
                st.write(f"   📊 {len(df)} registros, columnas: {list(df.columns)}")
            except Exception as e:
                st.write(f"   ❌ Error leyendo: {e}")
    
    # 4. Verificar imports
    st.write("**4. Verificación de imports:**")
    try:
        from data_loader import get_sp500_tickers_from_wikipedia
        st.write("✅ get_sp500_tickers_from_wikipedia importado")
    except Exception as e:
        st.write(f"❌ Error importando get_sp500_tickers_from_wikipedia: {e}")
    
    try:
        from data_loader import get_current_constituents
        st.write("✅ get_current_constituents importado")
    except Exception as e:
        st.write(f"❌ Error importando get_current_constituents: {e}")
    
    try:
        from data_loader import get_constituents_at_date
        st.write("✅ get_constituents_at_date importado")
    except Exception as e:
        st.write(f"❌ Error importando get_constituents_at_date: {e}")
    
    # 5. Probar obtención directa de constituyentes
    st.write("**5. Prueba de obtención de constituyentes:**")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].str.replace('.', '-').tolist()
        st.write(f"✅ Wikipedia S&P 500: {len(tickers)} tickers obtenidos directamente")
        st.write(f"   Primeros 5: {tickers[:5]}")
    except Exception as e:
        st.write(f"❌ Error obteniendo S&P 500 de Wikipedia: {e}")
    
    # 6. Verificar archivos CSV específicos
    st.write("**6. Verificación de archivos CSV de muestra:**")
    sample_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY']
    for ticker in sample_tickers:
        file_path = f"data/{ticker}.csv"
        exists = os.path.exists(file_path)
        st.write(f"📈 {ticker}.csv: {'✅' if exists else '❌'}")
        if exists:
            try:
                df = pd.read_csv(file_path, nrows=3)
                st.write(f"   Columnas: {list(df.columns)}")
                st.write(f"   Filas de muestra: {len(df)}")
            except Exception as e:
                st.write(f"   ❌ Error: {e}")

def simple_get_constituents(index_name):
    """Función simplificada para obtener constituyentes"""
    st.write(f"🔍 Intentando obtener constituyentes de {index_name}...")
    
    try:
        if index_name == "SP500" or index_name == "Ambos (SP500 + NDX)":
            st.write("📥 Conectando a Wikipedia S&P 500...")
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            sp500_tickers = df['Symbol'].str.replace('.', '-').tolist()
            st.write(f"✅ S&P 500: {len(sp500_tickers)} tickers")
        else:
            sp500_tickers = []
        
        if index_name == "NDX" or index_name == "Ambos (SP500 + NDX)":
            st.write("📥 Conectando a Wikipedia NASDAQ-100...")
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            tables = pd.read_html(url)
            ndx_tickers = []
            for i in [4, 3, 2, 1]:
                try:
                    df = tables[i]
                    if 'Ticker' in df.columns or 'Symbol' in df.columns:
                        col_name = 'Ticker' if 'Ticker' in df.columns else 'Symbol'
                        ndx_tickers = df[col_name].str.replace('.', '-').tolist()
                        st.write(f"✅ NASDAQ-100: {len(ndx_tickers)} tickers")
                        break
                except:
                    continue
        else:
            ndx_tickers = []
        
        # Combinar si es necesario
        if index_name == "Ambos (SP500 + NDX)":
            all_tickers = list(set(sp500_tickers + ndx_tickers))
        elif index_name == "SP500":
            all_tickers = sp500_tickers
        else:
            all_tickers = ndx_tickers
        
        # Verificar cuáles tienen archivos CSV
        st.write("📁 Verificando archivos CSV disponibles...")
        available_tickers = []
        
        for ticker in all_tickers:
            file_path = f"data/{ticker}.csv"
            if os.path.exists(file_path):
                available_tickers.append(ticker)
        
        st.write(f"✅ Tickers con datos: {len(available_tickers)} de {len(all_tickers)}")
        
        return available_tickers
        
    except Exception as e:
        st.write(f"❌ Error: {e}")
        st.write(f"Traceback: {traceback.format_exc()}")
        return []

# -------------------------------------------------
# FUNCIONES AUXILIARES SIMPLIFICADAS
# -------------------------------------------------
def get_cache_key(params):
    """Genera una clave única para caché basada en parámetros"""
    param_str = str(sorted(params.items()))
    return hashlib.md5(param_str.encode()).hexdigest()

def load_prices_simple(tickers, start_date, end_date):
    """Carga precios de forma simple y directa"""
    prices_data = {}
    
    for ticker in tickers:
        csv_path = f"data/{ticker}.csv"
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
                
                # Filtrar fechas
                start_filter = start_date if isinstance(start_date, str) else start_date.strftime('%Y-%m-%d')
                end_filter = end_date if isinstance(end_date, str) else end_date.strftime('%Y-%m-%d')
                
                df = df.loc[start_filter:end_filter]
                
                if not df.empty and 'Close' in df.columns:
                    prices_data[ticker] = df['Close']
                    
            except Exception as e:
                continue
    
    if prices_data:
        prices_df = pd.DataFrame(prices_data)
        prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
        return prices_df
    else:
        return pd.DataFrame()

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
    from data_loader import get_constituents_at_date, get_current_constituents
    data_loader_available = True
    st.sidebar.success("✅ Módulo data_loader cargado")
except Exception as e:
    st.sidebar.error(f"❌ Error cargando data_loader: {e}")

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

# Botones
run_simple_test = st.sidebar.button("🧪 Prueba Simple", type="primary")
run_full_diagnosis = st.sidebar.button("🔍 Diagnóstico Completo")

# -------------------------------------------------
# CONSTANTES
# -------------------------------------------------
CACHE_DIR = "data/cache"
if not os.path.exists(CACHE_DIR):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except:
        pass

# -------------------------------------------------
# Main content
# -------------------------------------------------

if run_full_diagnosis:
    diagnose_system()

elif run_simple_test:
    st.header("🧪 Prueba Simple del Sistema")
    
    with st.expander("📊 Diagnóstico Básico", expanded=True):
        diagnose_system()
    
    st.header("🔍 Prueba de Obtención de Constituyentes")
    
    # Probar obtención simple
    tickers = simple_get_constituents(index_choice)
    
    if tickers:
        st.success(f"✅ Obtenidos {len(tickers)} tickers con datos")
        
        # Mostrar muestra
        st.write("**Primeros 20 tickers:**")
        st.write(tickers[:20])
        
        # Probar carga de precios
        st.write("**Probando carga de precios...**")
        try:
            sample_tickers = tickers[:5]  # Solo primeros 5 para prueba
            prices_df = load_prices_simple(sample_tickers, start_date, end_date)
            
            if not prices_df.empty:
                st.success(f"✅ Precios cargados: {len(prices_df.columns)} tickers, {len(prices_df)} fechas")
                st.write("**Muestra de datos:**")
                st.dataframe(prices_df.head())
                
                # Estadísticas básicas
                st.write("**Estadísticas básicas:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Tickers", len(prices_df.columns))
                col2.metric("Fechas", len(prices_df))
                col3.metric("Período", f"{prices_df.index[0].date()} a {prices_df.index[-1].date()}")
                
            else:
                st.error("❌ No se pudieron cargar precios")
                
        except Exception as e:
            st.error(f"❌ Error cargando precios: {e}")
            st.write(f"Traceback: {traceback.format_exc()}")
    
    else:
        st.error("❌ No se pudieron obtener tickers")

else:
    st.info("👈 Usa los botones del sidebar para probar el sistema")
    
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
        st.write(f"{'✅' if os.path.exists('data/AAPL.csv') else '❌'} data/AAPL.csv (muestra)")
        st.write(f"{'✅' if os.path.exists('backtest.py') else '❌'} backtest.py")
    
    st.write("**Instrucciones:**")
    st.write("1. Haz clic en '🔍 Diagnóstico Completo' para ver el estado detallado")
    st.write("2. Haz clic en '🧪 Prueba Simple' para probar la obtención básica de datos")
    st.write("3. Si la prueba simple funciona, entonces el problema está en el data_loader.py original")

# Información adicional en sidebar
st.sidebar.markdown("---")
st.sidebar.write("**Debug Info:**")
st.sidebar.write(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
st.sidebar.write(f"Directorio: {os.getcwd()}")
st.sidebar.write(f"Archivos en raíz: {len([f for f in os.listdir('.') if f.endswith('.py')])}")
