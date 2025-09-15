import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import numpy as np
import os

# Importar nuestros módulos
from data_loader import get_constituents_at_date, get_sp500_historical_changes, get_nasdaq100_historical_changes
from backtest import run_backtest, inertia_score, calcular_atr_amibroker

# -------------------------------------------------
# Configuración de la app
# -------------------------------------------------
st.set_page_config(
    page_title="IA Mensual Ajustada",
    page_icon="📈",
    layout="wide"
)
st.title("📈 Estrategia mensual sobre los componentes del S&P 500 y/o Nasdaq-100")

# -------------------------------------------------
# Sidebar - Parámetros
# -------------------------------------------------
st.sidebar.header("Parámetros de backtest")

# Selector de índice
index_choice = st.sidebar.selectbox(
    "Selecciona el índice:",
    ["SP500", "NDX", "Ambos (SP500 + NDX)"]
)

# Fechas
end_date = st.sidebar.date_input("Fecha final", datetime.today())
start_date = st.sidebar.date_input("Fecha inicial", end_date - timedelta(days=365*5))

# Parámetros del backtest
top_n = st.sidebar.slider("Número de activos", 5, 30, 10)
commission = st.sidebar.number_input("Comisión por operación (%)", 0.0, 1.0, 0.3) / 100
corte = st.sidebar.number_input("Corte de score", 0, 1000, 680)

# Nueva opción para verificación histórica
use_historical_verification = st.sidebar.checkbox(
    "🕐 Usar verificación histórica de constituyentes", 
    value=True,
    help="Verifica que los tickers estuvieran realmente en el índice en cada fecha histórica"
)

# ✅ NUEVAS OPCIONES DE ESTRATEGIA
st.sidebar.subheader("⚙️ Opciones de Estrategia")

# Opción de asignación fija de capital
fixed_allocation = st.sidebar.checkbox(
    "💰 Asignar 10% capital a cada acción",
    value=False,
    help="Asigna exactamente 10% a cada posición. El capital no usado queda en efectivo."
)

# Filtros de mercado
st.sidebar.subheader("🛡️ Filtros de Mercado")

use_roc_filter = st.sidebar.checkbox(
    "📉 ROC 10 meses del SPY < 0",
    value=False,
    help="No invierte cuando el ROC de 10 meses del SPY es negativo"
)

use_sma_filter = st.sidebar.checkbox(
    "📊 Precio SPY < SMA 10 meses",
    value=False,
    help="No invierte cuando el SPY está por debajo de su media móvil de 10 meses"
)

# Información sobre filtros activos
if use_roc_filter or use_sma_filter:
    st.sidebar.info("⚠️ Con filtros activos, el sistema venderá todas las posiciones cuando se activen las condiciones")

# Botón de ejecución
run_button = st.sidebar.button("🏃 Ejecutar backtest")

# -------------------------------------------------
# Función para cargar datos desde CSV
# -------------------------------------------------
def load_prices_from_csv(tickers, start_date, end_date, load_full_data=True):
    """Carga precios desde archivos CSV en la carpeta data/ con datos completos OHLC"""
    prices_data = {}
    ohlc_data = {}
    
    for ticker in tickers:
        csv_path = f"data/{ticker}.csv"
        if os.path.exists(csv_path):
            try:
                # Leer CSV
                df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
                
                # ✅ ARREGLO: Manejar fechas con timezone
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    # Convertir a timezone-naive (remover timezone)
                    df.index = df.index.tz_localize(None)
                
                # Asegurar que start_date y end_date sean objetos date
                if isinstance(start_date, datetime):
                    start_date = start_date.date()
                if isinstance(end_date, datetime):
                    end_date = end_date.date()
                
                # Filtrar por rango de fechas
                df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
                
                if not df.empty:
                    # Para el precio de cierre (para compatibilidad)
                    if 'Adj Close' in df.columns:
                        prices_data[ticker] = df['Adj Close']
                    elif 'Close' in df.columns:
                        prices_data[ticker] = df['Close']
                    else:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            prices_data[ticker] = df[numeric_cols[0]]
                    
                    # Cargar datos OHLC completos si están disponibles
                    if load_full_data and all(col in df.columns for col in ['High', 'Low', 'Close']):
                        ohlc_data[ticker] = {
                            'High': df['High'],
                            'Low': df['Low'], 
                            'Close': df['Adj Close'] if 'Adj Close' in df.columns else df['Close'],
                            'Volume': df['Volume'] if 'Volume' in df.columns else None
                        }
                        
            except Exception as e:
                st.warning(f"Error cargando datos de {ticker}: {e}")
                continue
        else:
            # Silencioso para tickers que no existen (común en verificación histórica)
            continue
    
    if prices_data:
        prices_df = pd.DataFrame(prices_data)
        prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
        
        if load_full_data and ohlc_data:
            return prices_df, ohlc_data
        return prices_df
    else:
        return pd.DataFrame()

# -------------------------------------------------
# Main content
# -------------------------------------------------
if run_button:
    # Inicializar variable para evitar errores
    historical_info = None
    
    try:
        with st.spinner("Cargando datos desde CSV..."):
            # Lógica para obtener tickers de uno o ambos índices
            all_tickers_data = {'tickers': [], 'data': []}
            
            indices_to_fetch = []
            if index_choice == "SP500":
                indices_to_fetch = ["SP500"]
            elif index_choice == "NDX":
                indices_to_fetch = ["NDX"]
            else:  # "Ambos (SP500 + NDX)"
                indices_to_fetch = ["SP500", "NDX"]
            
            for idx in indices_to_fetch:
                constituents_data, error = get_constituents_at_date(idx, start_date, end_date)
                if error:
                    st.warning(f"Advertencia obteniendo constituyentes de {idx}: {error}")
                    continue
                if constituents_data and 'tickers' in constituents_data:
                    # Combinar tickers y datos
                    all_tickers_data['tickers'].extend(constituents_data.get('tickers', []))
                    all_tickers_data['data'].extend(constituents_data.get('data', []))
                    # Preservar información histórica
                    if 'historical_data_available' in constituents_data:
                        all_tickers_data['historical_data_available'] = constituents_data['historical_data_available']
                    if 'changes_processed' in constituents_data:
                        all_tickers_data['changes_processed'] = constituents_data.get('changes_processed', 0)
            
            # Verificar fechas de incorporación y datos históricos
            if all_tickers_data and 'data' in all_tickers_data:
                if all_tickers_data.get('historical_data_available', False):
                    st.success(f"✅ Datos históricos disponibles - Verificación de fechas activa")
                    st.info(f"📅 Cambios procesados: {all_tickers_data.get('changes_processed', 0)}")
                    # Mostrar algunos ejemplos de fechas
                    sample_data = all_tickers_data['data'][:5] if all_tickers_data['data'] else []
                    if sample_data:
                        st.text("🔍 Ejemplos de fechas de incorporación:")
                        for item in sample_data:
                            if isinstance(item, dict):
                                ticker = item.get('ticker', 'N/A')
                                added = item.get('added', 'Unknown')
                                status = item.get('status', 'N/A')
                                st.text(f"  • {ticker}: agregado el {added} ({status})")
                else:
                    st.warning("⚠️  Datos históricos no disponibles - Usando constituyentes actuales")
                    st.info("💡 El backtest usará todos los tickers disponibles sin verificación histórica")
            
            # Eliminar duplicados de tickers manteniendo el orden
            if all_tickers_data['tickers']:
                seen = set()
                unique_tickers = []
                for ticker in all_tickers_data['tickers']:
                    if ticker not in seen:
                        unique_tickers.append(ticker)
                        seen.add(ticker)
                all_tickers_data['tickers'] = unique_tickers
            
            tickers_count = len(all_tickers_data.get('tickers', []))
            st.success(f"✅ Obtenidos {tickers_count} constituyentes combinados")
            if tickers_count == 0:
                st.error("No se encontraron tickers válidos")
                st.stop()
            
            # Mostrar algunos tickers de ejemplo
            sample_tickers = all_tickers_data.get('tickers', [])[:10]
            if sample_tickers:
                st.info(f"Tickers de ejemplo: {', '.join(sample_tickers)}")

            # Cargar precios desde CSV con datos OHLC completos
            result = load_prices_from_csv(all_tickers_data['tickers'], start_date, end_date, load_full_data=True)

            if isinstance(result, tuple):
                prices_df, ohlc_data = result
                st.success(f"✅ Cargados precios OHLC completos para {len(prices_df.columns)} tickers")
                st.info(f"Datos OHLC disponibles para: {len(ohlc_data)} tickers")
            else:
                prices_df = result
                ohlc_data = None
                st.warning("⚠️ Solo se cargaron precios de cierre. OHLC no disponible.")
            
            # Validación adicional de precios
            if prices_df is None or prices_df.empty or len(prices_df.columns) == 0:
                st.error("❌ No se pudieron cargar los precios históricos desde los CSV")
                st.info("💡 Consejos para resolver este problema:")
                st.info("1. Verifica que los archivos CSV existan en la carpeta 'data/'")
                st.info("2. Asegúrate de que los archivos tengan la columna 'Date' como índice")
                st.info("3. Verifica que los archivos contengan columnas de precios (Close, Adj Close)")
                st.info("4. Prueba con un rango de fechas más corto")
                
                # Información sobre tickers que no se pudieron cargar
                missing_tickers = set(all_tickers_data['tickers']) - set(prices_df.columns if not prices_df.empty else [])
                if missing_tickers:
                    st.warning(f"🔍 Tickers sin datos CSV disponibles: {len(missing_tickers)}")
                    if len(missing_tickers) <= 20:
                        st.text(f"Faltantes: {', '.join(sorted(missing_tickers))}")
                    else:
                        st.text(f"Primeros 20 faltantes: {', '.join(sorted(list(missing_tickers))[:20])}")
                
                st.stop()
            
            # Información sobre cobertura de datos
            available_tickers = set(prices_df.columns)
            requested_tickers = set(all_tickers_data['tickers'])
            missing_tickers = requested_tickers - available_tickers
            
            coverage_pct = len(available_tickers) / len(requested_tickers) * 100 if requested_tickers else 0
            
            st.success(f"✅ Cargados precios para {len(prices_df.columns)} tickers")
            st.info(f"📊 Cobertura de datos: {coverage_pct:.1f}% ({len(available_tickers)}/{len(requested_tickers)})")
            st.info(f"📅 Rango de fechas: {prices_df.index.min().strftime('%Y-%m-%d')} a {prices_df.index.max().strftime('%Y-%m-%d')}")
            
            if missing_tickers and len(missing_tickers) <= 10:
                st.warning(f"⚠️  Tickers sin datos: {', '.join(sorted(missing_tickers))}")
            elif missing_tickers:
                st.warning(f"⚠️  {len(missing_tickers)} tickers sin datos CSV disponibles")

            # Cargar benchmark desde CSV
            if index_choice == "SP500":
                benchmark_ticker = "SPY"
            elif index_choice == "NDX":
                benchmark_ticker = "QQQ"
            else:  # Ambos
                benchmark_ticker = "SPY"
            
            st.info(f"Cargando benchmark: {benchmark_ticker}")
            benchmark_result = load_prices_from_csv([benchmark_ticker], start_date, end_date, load_full_data=False)
            
            if isinstance(benchmark_result, tuple):
                benchmark_df = benchmark_result[0]
            else:
                benchmark_df = benchmark_result
            
            if benchmark_df is None or benchmark_df.empty:
                st.warning(f"No se pudo cargar el benchmark {benchmark_ticker} desde CSV")
                try:
                    st.info("Usando promedio de constituyentes como benchmark alternativo")
                    if not prices_df.empty:
                        benchmark_series = prices_df.mean(axis=1)
                        benchmark_df = pd.DataFrame({benchmark_ticker: benchmark_series})
                        st.success("✅ Benchmark alternativo creado correctamente")
                    else:
                        st.error("❌ No se pudo crear benchmark alternativo")
                        st.stop()
                except Exception as avg_error:
                    st.error(f"Tampoco se pudo crear benchmark alternativo: {avg_error}")
                    st.stop()
            else:
                st.success(f"✅ Benchmark {benchmark_ticker} cargado correctamente desde CSV")

            # ✅ NUEVO: Cargar SP
