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
import warnings
warnings.filterwarnings('ignore')

# Importar nuestros módulos
from data_loader import get_constituents_at_date, get_sp500_historical_changes, get_nasdaq100_historical_changes
from backtest import run_optimized_backtest, optimized_inertia_score, calcular_atr_amibroker, calculate_monthly_returns_by_year

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
    "📉 ROC 12 meses del SPY < 0",
    value=False,
    help="No invierte cuando el ROC de 12 meses del SPY es negativo"
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
# Constantes para gestión de cambios históricos
# -------------------------------------------------
GITHUB_RAW_URL = "https://raw.githubusercontent.com/jmlestevez-source/IA-MENSUAL-AJUSTADA/main/"
LOCAL_CHANGES_DIR = "data/historical_changes"
SP500_CHANGES_FILE = "sp500_changes.csv"
NDX_CHANGES_FILE = "ndx_changes.csv"

# -------------------------------------------------
# Funciones para gestión de cambios históricos
# -------------------------------------------------
def ensure_local_changes_dir():
    """Asegura que exista el directorio para cambios históricos"""
    if not os.path.exists(LOCAL_CHANGES_DIR):
        os.makedirs(LOCAL_CHANGES_DIR)

def download_csv_from_github(filename):
    """Descarga un CSV desde el repositorio de GitHub"""
    url = GITHUB_RAW_URL + filename
    local_path = os.path.join(LOCAL_CHANGES_DIR, filename)
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Guardar el archivo localmente
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        # Cargar en DataFrame
        df = pd.read_csv(local_path, parse_dates=["Date"])
        return df
    except Exception as e:
        st.warning(f"⚠️ No se pudo descargar {filename} desde GitHub: {e}")
        # Si no se puede descargar, intentar cargar desde local si existe
        if os.path.exists(local_path):
            try:
                df = pd.read_csv(local_path, parse_dates=["Date"])
                return df
            except:
                pass
        return pd.DataFrame()

def get_local_changes_file(filename):
    """Obtiene el archivo de cambios local si existe"""
    local_path = os.path.join(LOCAL_CHANGES_DIR, filename)
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path, parse_dates=["Date"])
            return df
        except:
            pass
    return pd.DataFrame()

def update_changes_with_new_data(index_name, current_changes_df):
    """
    Verifica y actualiza los cambios históricos con nuevos datos
    """
    filename = f"{index_name}_changes.csv"
    local_path = os.path.join(LOCAL_CHANGES_DIR, filename)
    
    # Obtener cambios locales actuales
    local_changes = get_local_changes_file(filename)
    
    if local_changes.empty:
        st.info(f"📥 Descargando {filename} desde GitHub...")
        local_changes = download_csv_from_github(filename)
    
    if local_changes.empty:
        st.warning(f"⚠️  No se encontraron datos históricos para {index_name}")
        # Usar los cambios actuales si no hay datos locales
        if not current_changes_df.empty:
            current_changes_df.to_csv(local_path, index=False)
            st.success(f"💾 Guardado {len(current_changes_df)} cambios iniciales para {index_name}")
        return current_changes_df
    
    # Encontrar la última fecha registrada - CORREGIDO
    if not local_changes.empty:
        # Convertir a datetime.date para comparación consistente
        last_date = pd.to_datetime(local_changes["Date"]).dt.date.max()
        last_date = pd.Timestamp(last_date)  # Convertir a Timestamp para comparación
    else:
        last_date = pd.Timestamp(datetime(1900, 1, 1))
    
    # Filtrar cambios nuevos desde la última fecha - CORREGIDO
    if not current_changes_df.empty:
        # Asegurar que la columna Date sea datetime
        current_changes_df['Date'] = pd.to_datetime(current_changes_df['Date'])
        
        # Comparar fechas correctamente
        new_changes = current_changes_df[current_changes_df["Date"] > last_date]
        
        if not new_changes.empty:
            # Combinar y eliminar duplicados
            updated_changes = pd.concat([local_changes, new_changes], ignore_index=True)
            updated_changes = updated_changes.drop_duplicates(subset=["Date", "Ticker", "Action"])
            
            # Guardar actualización
            updated_changes.to_csv(local_path, index=False)
            st.success(f"✅ Actualizados {len(new_changes)} nuevos cambios en {filename}")
            
            return updated_changes
        else:
            st.info(f"🔍 No hay nuevos cambios en {index_name} desde {last_date.date()}")
            return local_changes
    else:
        return local_changes

# -------------------------------------------------
# Función para crear enlaces de descarga
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

def show_download_buttons(sp500_df, ndx_df):
    """Muestra botones de descarga para los archivos CSV generados"""
    st.subheader("📥 Descargar CSV de Cambios Históricos")
    st.info("💾 Guarda estos archivos y súbelos a la raíz de tu repositorio de GitHub para usarlos en futuras ejecuciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not sp500_df.empty:
            sp500_link = create_download_link(sp500_df, "sp500_changes.csv", "💾 Descargar sp500_changes.csv")
            if sp500_link:
                st.markdown(sp500_link, unsafe_allow_html=True)
                st.text(f"📊 {len(sp500_df)} cambios históricos del S&P 500")
        else:
            st.warning("No hay datos del S&P 500 para descargar")
    
    with col2:
        if not ndx_df.empty:
            ndx_link = create_download_link(ndx_df, "ndx_changes.csv", "💾 Descargar ndx_changes.csv")
            if ndx_link:
                st.markdown(ndx_link, unsafe_allow_html=True)
                st.text(f"📊 {len(ndx_df)} cambios históricos del NASDAQ-100")
        else:
            st.warning("No hay datos del NASDAQ-100 para descargar")
    
    st.info("""
    📋 **Instrucciones para futuras ejecuciones más rápidas:**
    1. Haz clic en los botones de descarga de arriba
    2. Sube los archivos descargados a la raíz de tu repositorio en GitHub
    3. En futuras ejecuciones, el sistema los descargará directamente desde GitHub
    4. Esto evitará tener que scrapear Wikipedia en cada ejecución
    """)

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
# Función para calcular tabla de rendimientos mensuales por año - CORREGIDA
# -------------------------------------------------
def calculate_monthly_returns_by_year(equity_series):
    """
    ✅ CORREGIDA: Calcula tabla de rendimientos mensuales por año con formato consistente
    """
    try:
        if equity_series is None or len(equity_series) < 2:
            return pd.DataFrame()
        
        # Calcular retornos mensuales
        monthly_returns = equity_series.pct_change().fillna(0)
        
        # Agrupar por año y mes
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_by_year = monthly_returns.groupby([
            monthly_returns.index.year, 
            monthly_returns.index.month
        ]).apply(lambda x: (1 + x).prod() - 1)
        
        # Crear DataFrame con estructura de tabla
        years = sorted(monthly_by_year.index.get_level_values(0).unique())
        
        # Crear tabla con meses como columnas (en español)
        months_es = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        
        table_data = []
        
        for year in years:
            year_data = {'Año': year}
            
            # Obtener datos de este año
            year_monthly = monthly_by_year[monthly_by_year.index.get_level_values(0) == year]
            
            # Llenar meses con formato consistente
            for i, month_abbr in enumerate(months_es, 1):
                if i in year_monthly.index.get_level_values(1):
                    return_value = year_monthly[year_monthly.index.get_level_values(1) == i].iloc[0]
                    
                    # ✅ CORREGIDO: Formato consistente para valores cercanos a cero
                    if np.isnan(return_value) or np.isinf(return_value):
                        year_data[month_abbr] = "-"
                    elif abs(return_value) < 0.0001:  # Menos de 0.01%
                        year_data[month_abbr] = "0.00%"
                    else:
                        year_data[month_abbr] = f"{return_value*100:.2f}%"
                else:
                    year_data[month_abbr] = "-"
            
            # Calcular YTD
            year_equity = equity_series[equity_series.index.year == year]
            if len(year_equity) > 1:
                ytd_return = (year_equity.iloc[-1] / year_equity.iloc[0]) - 1
                
                # ✅ CORREGIDO: Formato consistente para YTD
                if np.isnan(ytd_return) or np.isinf(ytd_return):
                    year_data['YTD'] = "-"
                elif abs(ytd_return) < 0.0001:  # Menos de 0.01%
                    year_data['YTD'] = "0.00%"
                else:
                    year_data['YTD'] = f"{ytd_return*100:.2f}%"
            else:
                year_data['YTD'] = "-"
            
            table_data.append(year_data)
        
        # Crear DataFrame final
        if table_data:
            result_df = pd.DataFrame(table_data)
            
            # Asegurar orden correcto de columnas
            columns_order = ['Año'] + months_es + ['YTD']
            result_df = result_df[columns_order]
            
            return result_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error calculando tabla de rendimientos mensuales: {e}")
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
            
                        # ✅ NUEVO: Cargar SPY para filtros si es necesario
            spy_df = None
            if (use_roc_filter or use_sma_filter) and benchmark_ticker != "SPY":
                st.info("📊 Cargando datos del SPY para filtros de mercado...")
                spy_result = load_prices_from_csv(["SPY"], start_date, end_date, load_full_data=False)
                
                if isinstance(spy_result, tuple):
                    spy_df = spy_result[0]
                else:
                    spy_df = spy_result
                
                if spy_df is None or spy_df.empty:
                    st.warning("⚠️ No se pudo cargar SPY para filtros. Los filtros de mercado se desactivarán.")
                    use_roc_filter = False
                    use_sma_filter = False
                else:
                    st.success("✅ SPY cargado para filtros de mercado")

        with st.spinner("Ejecutando backtest optimizado..."):
            # Asegurar que tenemos datos válidos para el benchmark
            if benchmark_df is not None and not benchmark_df.empty:
                benchmark_series = benchmark_df[benchmark_ticker] if benchmark_ticker in benchmark_df.columns else benchmark_df.iloc[:, 0]
            else:
                # Fallback al benchmark alternativo
                benchmark_series = prices_df.mean(axis=1) if not prices_df.empty else pd.Series()
            
            # Validar que tenemos suficientes datos
            if prices_df.empty or len(prices_df) < 20:
                st.error("❌ No hay suficientes datos para ejecutar el backtest (se necesitan al menos 20 períodos)")
                st.stop()
            
            # Preparar información histórica para el backtest si está habilitada
            if use_historical_verification and all_tickers_data.get('historical_data_available', False):
                try:
                    st.info("🔍 Preparando datos históricos para verificación de constituyentes...")
                    
                    changes_data = pd.DataFrame()
                    changes_loaded = []
                    
                    if index_choice in ["SP500", "Ambos (SP500 + NDX)"]:
                        try:
                            sp500_changes = get_sp500_historical_changes()
                            if not sp500_changes.empty:
                                changes_data = pd.concat([changes_data, sp500_changes], ignore_index=True)
                                changes_loaded.append(f"S&P 500: {len(sp500_changes)} cambios")
                                st.success(f"✅ Cargados {len(sp500_changes)} cambios del S&P 500")
                            else:
                                st.warning("⚠️  No se pudieron cargar cambios del S&P 500")
                        except Exception as e:
                            st.error(f"❌ Error cargando S&P 500: {e}")
                    
                    if index_choice in ["NDX", "Ambos (SP500 + NDX)"]:
                        try:
                            ndx_changes = get_nasdaq100_historical_changes()
                            if not ndx_changes.empty:
                                changes_data = pd.concat([changes_data, ndx_changes], ignore_index=True)
                                changes_loaded.append(f"NASDAQ-100: {len(ndx_changes)} cambios")
                                st.success(f"✅ Cargados {len(ndx_changes)} cambios del NASDAQ-100")
                            else:
                                st.warning("⚠️  No se pudieron cargar cambios del NASDAQ-100")
                        except Exception as e:
                            st.error(f"❌ Error cargando NASDAQ-100: {e}")
                    
                    if not changes_data.empty:
                        # Eliminar duplicados y ordenar
                        initial_count = len(changes_data)
                        changes_data = changes_data.drop_duplicates(subset=['Date', 'Ticker', 'Action'])
                        changes_data = changes_data.sort_values('Date', ascending=False)
                        
                        historical_info = {
                            'changes_data': changes_data,
                            'has_historical_data': True
                        }
                        
                        final_count = len(changes_data)
                        duplicate_count = initial_count - final_count
                        
                        st.success(f"✅ Verificación histórica activa con {final_count} cambios únicos")
                        if duplicate_count > 0:
                            st.info(f"📊 Eliminados {duplicate_count} cambios duplicados")
                        
                        # Mostrar resumen de lo cargado
                        st.info("📋 **Datos cargados**: " + " | ".join(changes_loaded))
                        
                        # Mostrar estadísticas de cambios
                        date_range = f"{changes_data['Date'].min()} a {changes_data['Date'].max()}"
                        st.info(f"📊 Rango de cambios históricos: {date_range}")
                        
                    else:
                        st.warning("⚠️  No se pudieron cargar datos históricos, continuando sin verificación")
                        
                except Exception as e:
                    st.warning(f"⚠️  Error cargando datos históricos: {e}")
                    st.info("Continuando backtest sin verificación histórica")
            
            elif use_historical_verification:
                st.warning("⚠️  Verificación histórica solicitada pero no hay datos históricos disponibles")
            
            # Generar y mostrar información sobre tickers removidos
            if historical_info and historical_info.get('has_historical_data', False):
                try:
                    # Generar resumen de tickers removidos
                    from data_loader import generate_removed_tickers_summary
                    removed_summary = generate_removed_tickers_summary()
                    
                    if not removed_summary.empty:
                        with st.expander("📥 Tickers Removidos - Oportunidad de Datos", expanded=False):
                            st.subheader("Tickers removidos de los índices")
                            st.info(f"""
                            💡 **Oportunidad**: Estos {len(removed_summary)} tickers fueron removidos de los índices pero 
                            podrían ser útiles para el backtest en las fechas cuando SÍ estaban incluidos.
                            
                            📥 **CSV generados**:
                            - `data/sp500_removed_tickers.csv`: Solo tickers del S&P 500
                            - `data/all_removed_tickers_summary.csv`: Resumen completo
                            
                            🔍 **Siguiente paso**: Descarga datos históricos de estos tickers y agrégalos a tu carpeta `data/`
                            """)
                            
                            # Mostrar los más recientes
                            st.subheader("Tickers removidos más recientemente")
                            recent_removed = removed_summary.head(20)
                            
                            # Formatear fechas para mostrar
                            display_removed = recent_removed.copy()
                            display_removed['First_Removed'] = pd.to_datetime(display_removed['First_Removed']).dt.strftime('%Y-%m-%d')
                            display_removed['Last_Removed'] = pd.to_datetime(display_removed['Last_Removed']).dt.strftime('%Y-%m-%d')
                            
                            st.dataframe(display_removed, use_container_width=True)
                            
                            # Estadísticas
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                sp500_count = len(removed_summary[removed_summary['Indices'].str.contains('SP500')])
                                st.metric("S&P 500 Removidos", sp500_count)
                            
                            with col2:
                                ndx_count = len(removed_summary[removed_summary['Indices'].str.contains('NDX')])
                                st.metric("NASDAQ-100 Removidos", ndx_count)
                            
                            with col3:
                                both_count = len(removed_summary[removed_summary['Indices'].str.contains('SP500') & 
                                                              removed_summary['Indices'].str.contains('NDX')])
                                st.metric("Removidos de Ambos", both_count)
                            
                            # Lista de tickers para copiar
                            st.subheader("📋 Lista de tickers para descargar")
                            tickers_list = ', '.join(removed_summary['Ticker'].tolist())
                            st.code(tickers_list, language="text")
                            
                            st.warning("""
                            ⚠️ **Importante**: Estos tickers fueron removidos por razones como:
                            - Capitalización de mercado baja
                            - Fusiones/adquisiciones  
                            - Cambio de sector
                            - Delisting
                            
                            Algunos pueden no tener datos disponibles o haber cambiado de ticker.
                            """)
                
                except Exception as e:
                    st.warning(f"Error generando información de removidos: {e}")
            
            # ✅ EJECUTAR BACKTEST OPTIMIZADO
            bt_results, picks_df = run_optimized_backtest(
                prices=prices_df,
                benchmark=benchmark_series,
                commission=commission,
                top_n=top_n,
                corte=corte,
                ohlc_data=ohlc_data,
                historical_info=historical_info,
                fixed_allocation=fixed_allocation,  # ✅ Nueva opción
                use_roc_filter=use_roc_filter,     # ✅ Filtro ROC
                use_sma_filter=use_sma_filter,     # ✅ Filtro SMA
                spy_data=spy_df if spy_df is not None else (benchmark_df if benchmark_ticker == "SPY" else None)  # ✅ Datos del SPY para filtros
            )
            
            if bt_results is None or bt_results.empty or len(bt_results) < 2:
                st.error("❌ El backtest no generó resultados (posiblemente datos insuficientes)")
                st.info("💡 Consejos:")
                st.info("• Prueba con un rango de fechas más largo")
                st.info("• Reduce el número de activos seleccionados")
                st.info("• Verifica que los tickers sean válidos y tengan datos históricos")
                st.stop()
                
            st.success("✅ Backtest completado")

            # -------------------------------------------------
            # Métricas principales CORREGIDAS
            # -------------------------------------------------
            # Calcular métricas de la estrategia
            if "Equity" in bt_results.columns and len(bt_results["Equity"]) > 0:
                final_equity = float(bt_results["Equity"].iloc[-1])
                initial_equity = float(bt_results["Equity"].iloc[0])
                total_return = (final_equity / initial_equity) - 1 if initial_equity != 0 else 0
                
                # Calcular CAGR
                years = (bt_results.index[-1] - bt_results.index[0]).days / 365.25
                if years > 0:
                    cagr = (final_equity / initial_equity) ** (1/years) - 1
                else:
                    cagr = 0
            else:
                final_equity = 10000
                total_return = 0
                cagr = 0
                
            if "Drawdown" in bt_results.columns:
                max_drawdown = float(bt_results["Drawdown"].min())
            else:
                max_drawdown = 0
                
            # ✅ SHARPE RATIO CORREGIDO
            if "Returns" in bt_results.columns and len(bt_results["Returns"]) > 1:
                # Asumir tasa libre de riesgo del 2% anual (0.02/12 mensual)
                risk_free_rate_monthly = 0.02 / 12
                
                monthly_returns = bt_results["Returns"]
                excess_returns = monthly_returns - risk_free_rate_monthly
                
                # Calcular Sharpe ratio anualizado correctamente
                if excess_returns.std() != 0:
                    sharpe_ratio = (excess_returns.mean() * 12) / (excess_returns.std() * (12 ** 0.5))
                else:
                    sharpe_ratio = 0
                    
                # Volatilidad anualizada
                volatility = float(monthly_returns.std() * (12 ** 0.5))
            else:
                volatility = 0
                sharpe_ratio = 0

            # Preparar datos del benchmark ANTES de mostrar métricas
            bench_equity = None
            bench_drawdown = None
            bench_sharpe = 0
            
            if benchmark_df is not None and not benchmark_df.empty:
                try:
                    bench_data = benchmark_df[benchmark_ticker] if benchmark_ticker in benchmark_df.columns else benchmark_df.iloc[:, 0]
                    bench_returns = bench_data.pct_change().fillna(0)
                    bench_equity = 10000 * (1 + bench_returns).cumprod()
                    bench_drawdown = (bench_equity / bench_equity.cummax() - 1)
                    
                    # ✅ SHARPE DEL BENCHMARK CORREGIDO
                    # Convertir a mensual si es necesario
                    if len(bench_returns) > len(bt_results) * 15:  # Si son datos diarios
                        bench_returns_monthly = bench_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
                    else:
                        bench_returns_monthly = bench_returns
                    
                    bench_excess_returns = bench_returns_monthly - risk_free_rate_monthly
                    
                    if bench_excess_returns.std() != 0:
                        bench_sharpe = (bench_excess_returns.mean() * 12) / (bench_excess_returns.std() * (12 ** 0.5))
                    else:
                        bench_sharpe = 0
                        
                except Exception as e:
                    st.warning(f"Error calculando benchmark: {e}")

            # Mostrar métricas de la estrategia
            st.subheader("📊 Métricas de la Estrategia")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Equity Final", f"${final_equity:,.0f}")
            col2.metric("Retorno Total", f"{total_return:.2%}")
            col3.metric("CAGR", f"{cagr:.2%}")
            col4.metric("Máximo Drawdown", f"{max_drawdown:.2%}")
            col5.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

            # Información sobre verificación histórica
            if historical_info and historical_info.get('has_historical_data', False):
                st.info("✅ Este backtest incluye verificación histórica de constituyentes")
            else:
                st.warning("⚠️  Este backtest NO incluye verificación histórica (posible sesgo de supervivencia)")

            # Calcular y mostrar métricas del benchmark CORREGIDAS
            if bench_equity is not None and len(bench_equity) > 0:
                bench_final = float(bench_equity.iloc[-1])
                bench_initial = float(bench_equity.iloc[0])
                bench_total_return = (bench_final / bench_initial) - 1 if bench_initial != 0 else 0
                
                                # CAGR del benchmark
                if years > 0:
                    bench_cagr = (bench_final / bench_initial) ** (1/years) - 1
                else:
                    bench_cagr = 0
                
                # Drawdown del benchmark
                if bench_drawdown is not None:
                    bench_max_dd = float(bench_drawdown.min())
                else:
                    bench_max_dd = 0
                
                # Métricas del benchmark
                st.subheader(f"📊 Métricas del Benchmark ({benchmark_ticker})")
                col1b, col2b, col3b, col4b, col5b = st.columns(5)
                col1b.metric("Equity Final", f"${bench_final:,.0f}")
                col2b.metric("Retorno Total", f"{bench_total_return:.2%}")
                col3b.metric("CAGR", f"{bench_cagr:.2%}")
                col4b.metric("Máximo Drawdown", f"{bench_max_dd:.2%}")
                col5b.metric("Sharpe Ratio", f"{bench_sharpe:.2f}")

            # ✅ NUEVO: Comparación de métricas
            st.subheader("⚖️ Comparación Estrategia vs Benchmark")
            col1c, col2c, col3c = st.columns(3)
            
            with col1c:
                alpha = cagr - bench_cagr if 'bench_cagr' in locals() else cagr
                st.metric("Alpha (CAGR diff)", f"{alpha:.2%}", 
                         delta=f"{alpha:.2%}" if alpha >= 0 else f"{alpha:.2%}")
            
            with col2c:
                sharpe_diff = sharpe_ratio - bench_sharpe
                st.metric("Sharpe Diff", f"{sharpe_diff:.2f}",
                         delta=f"+{sharpe_diff:.2f}" if sharpe_diff >= 0 else f"{sharpe_diff:.2f}")
            
            with col3c:
                dd_diff = max_drawdown - bench_max_dd if 'bench_max_dd' in locals() else max_drawdown
                st.metric("DD Difference", f"{dd_diff:.2%}",
                         delta=f"{dd_diff:.2%}" if dd_diff <= 0 else f"+{dd_diff:.2%}")

            # -------------------------------------------------
            # Gráficos mejorados
            # -------------------------------------------------
            # Gráfico de equity
            try:
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    x=bt_results.index,
                    y=bt_results["Equity"],
                    mode='lines',
                    name='Estrategia',
                    line=dict(width=3, color='blue'),
                    hovertemplate='<b>%{y:,.0f}</b><br>%{x}<extra></extra>'
                ))
                
                # Benchmark
                if bench_equity is not None:
                    # Alinear índices
                    common_index = bt_results.index.intersection(bench_equity.index)
                    if len(common_index) > 0:
                        bench_aligned = bench_equity.loc[common_index]
                        
                        fig_equity.add_trace(go.Scatter(
                            x=bench_aligned.index,
                            y=bench_aligned.values,
                            mode='lines',
                            name=f'Benchmark ({benchmark_ticker})',
                            line=dict(width=2, dash='dash', color='gray'),
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
                
            except Exception as fig_error:
                st.warning(f"Error al crear gráfico de equity: {fig_error}")

            # Gráfico de drawdown combinado
            try:
                if "Drawdown" in bt_results.columns:
                    fig_dd = go.Figure()
                    
                    # Drawdown de la estrategia
                    fig_dd.add_trace(go.Scatter(
                        x=bt_results.index,
                        y=bt_results["Drawdown"] * 100,
                        mode='lines',
                        name='Drawdown Estrategia',
                        fill='tozeroy',
                        line=dict(color='red', width=2),
                        hovertemplate='<b>%{y:.2f}%</b><br>%{x}<extra></extra>'
                    ))
                    
                    # Drawdown del benchmark
                    if bench_drawdown is not None:
                        common_index = bt_results.index.intersection(bench_drawdown.index)
                        if len(common_index) > 0:
                            bench_dd_aligned = bench_drawdown.loc[common_index]
                            
                            fig_dd.add_trace(go.Scatter(
                                x=bench_dd_aligned.index,
                                y=bench_dd_aligned.values * 100,
                                mode='lines',
                                name=f'Drawdown {benchmark_ticker}',
                                line=dict(color='orange', width=2, dash='dash'),
                                hovertemplate='<b>%{y:.2f}%</b><br>%{x}<extra></extra>'
                            ))
                    
                    fig_dd.update_layout(
                        title="Drawdown Comparativo",
                        xaxis_title="Fecha",
                        yaxis_title="Drawdown (%)",
                        hovermode='x unified',
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig_dd, use_container_width=True)
            except Exception as dd_error:
                st.warning(f"Error al crear gráfico de drawdown: {dd_error}")

            # -------------------------------------------------
            # Picks seleccionados
            # -------------------------------------------------
            if picks_df is not None and not picks_df.empty:
                try:
                    st.subheader("Últimos picks seleccionados")
                    latest_date = picks_df["Date"].max()
                    latest_picks = picks_df[picks_df["Date"] == latest_date]
                    if not latest_picks.empty:
                        # Mostrar información de validez histórica si está disponible
                        if 'HistoricallyValid' in latest_picks.columns:
                            valid_count = latest_picks['HistoricallyValid'].sum()
                            total_count = len(latest_picks)
                            st.info(f"📊 Picks históricamente válidos: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
                        
                        display_picks = latest_picks.round(2)
                        st.dataframe(display_picks)
                    else:
                        st.info("No hay picks recientes para mostrar")
                    
                    # Mostrar picks de todos los meses
                    st.subheader("Todos los picks por mes")
                    
                    # Estadísticas de validez histórica si están disponibles
                    if 'HistoricallyValid' in picks_df.columns:
                        total_picks = len(picks_df)
                        valid_picks = picks_df['HistoricallyValid'].sum()
                        validity_rate = valid_picks / total_picks * 100 if total_picks > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total de Picks", total_picks)
                        col2.metric("Picks Válidos", valid_picks)
                        col3.metric("% Validez Histórica", f"{validity_rate:.1f}%")
                        
                        if validity_rate < 90:
                            st.warning(f"⚠️  Solo {validity_rate:.1f}% de los picks fueron históricamente válidos")
                        else:
                            st.success(f"✅ {validity_rate:.1f}% de los picks fueron históricamente válidos")
                    
                    st.dataframe(picks_df.round(2))
                    
                    # Gráfico de picks por fecha
                    try:
                        picks_by_date = picks_df.groupby("Date").size()
                        if len(picks_by_date) > 0:
                            fig_picks = px.bar(
                                x=picks_by_date.index,
                                y=picks_by_date.values,
                                labels={'x': 'Fecha', 'y': 'Número de Picks'},
                                title="Número de Picks por Fecha"
                            )
                            fig_picks.update_layout(height=400)
                            st.plotly_chart(fig_picks, use_container_width=True)
                    except Exception as picks_fig_error:
                        st.warning(f"Error al crear gráfico de picks: 
