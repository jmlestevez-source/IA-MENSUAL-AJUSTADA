Aquí tienes los archivos completos con todas las modificaciones:

## **1. app.py completo:**

```python
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

        with st.spinner("Ejecutando backtest..."):
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
            
            # Ejecutar backtest con datos OHLC e información histórica
            bt_results, picks_df = run_backtest(
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
                        st.warning(f"Error al crear gráfico de picks: {picks_fig_error}")
                        
                except Exception as picks_error:
                    st.warning(f"Error al procesar picks: {picks_error}")
            else:
                st.info("No se generaron picks en este backtest")

            # -------------------------------------------------
            # Señales Actuales (Vela en Formación) - CORREGIDO
            # -------------------------------------------------
            with st.expander("🔮 Señales Actuales - Vela en Formación", expanded=True):
                st.subheader("📊 Picks Prospectivos para el Próximo Mes")
                st.warning("""
                ⚠️ **IMPORTANTE**: Estas señales usan datos hasta HOY (vela en formación).
                - Son **preliminares** y pueden cambiar hasta el cierre del mes
                - En un sistema real, tomarías estas posiciones al inicio del próximo mes
                - Úsalas solo como referencia, NO como señales definitivas
                """)
                
                try:
                    # Usar TODOS los datos disponibles (incluyendo vela en formación)
                    current_scores = inertia_score(prices_df, corte=corte, ohlc_data=ohlc_data)
                    
                    if current_scores and "ScoreAdjusted" in current_scores:
                        score_df = current_scores["ScoreAdjusted"]
                        inercia_df = current_scores["InerciaAlcista"]
                        
                        if not score_df.empty and len(score_df) > 0:
                            # Obtener últimos scores
                            last_scores = score_df.iloc[-1].dropna().sort_values(ascending=False)
                            last_inercia = inercia_df.iloc[-1] if not inercia_df.empty else pd.Series()
                            
                            if len(last_scores) > 0:
                                # ✅ CORREGIDO: Filtrar PRIMERO los que pasan el corte
                                valid_picks = []
                                for ticker in last_scores.index:
                                    inercia_val = last_inercia.get(ticker, 0) if not last_inercia.empty else 0
                                    score_adj = last_scores[ticker]
                                    
                                    # Solo incluir si pasa el corte Y tiene score > 0
                                    if inercia_val >= corte and score_adj > 0:
                                        valid_picks.append({
                                            'ticker': ticker,
                                            'inercia': inercia_val,
                                            'score_adj': score_adj
                                        })
                                
                                # Ordenar por score ajustado y tomar top N válidos
                                valid_picks = sorted(valid_picks, key=lambda x: x['score_adj'], reverse=True)
                                
                                # Tomar solo hasta top_n O todos los válidos si son menos
                                final_picks = valid_picks[:min(top_n, len(valid_picks))]
                                
                                if not final_picks:
                                    st.warning("⚠️ No hay tickers que pasen el corte de inercia actualmente")
                                else:
                                    # Crear DataFrame para mostrar
                                    current_picks = []
                                    for rank, pick in enumerate(final_picks, 1):
                                        ticker = pick['ticker']
                                        current_picks.append({
                                            'Rank': rank,
                                            'Ticker': ticker,
                                            'Inercia Alcista': pick['inercia'],
                                            'Score Ajustado': pick['score_adj'],
                                            'Pasa Corte': '✅',  # Todos pasan porque ya filtramos
                                            'Precio Actual': prices_df[ticker].iloc[-1] if ticker in prices_df.columns else 0
                                        })
                                    
                                    current_picks_df = pd.DataFrame(current_picks)
                                    actual_count = len(current_picks_df)
                                    
                                    # Mostrar fecha de los datos
                                    data_date = prices_df.index[-1].strftime('%Y-%m-%d')
                                    st.info(f"📅 **Datos hasta**: {data_date} (vela en formación)")
                                    
                                    # Información importante sobre filtrado
                                    if actual_count < top_n:
                                        st.warning(f"⚠️ Solo {actual_count} de {top_n} tickers solicitados pasan el corte de inercia ({corte})")
                                    
                                    # Tabla de picks actuales
                                    st.subheader(f"🔥 Top {actual_count} Picks Válidos (de {top_n} solicitados)")
                                    
                                    # Formatear tabla para mostrar
                                    display_df = current_picks_df.copy()
                                    display_df['Precio Actual'] = display_df['Precio Actual'].apply(lambda x: f"${x:.2f}")
                                    display_df['Inercia Alcista'] = display_df['Inercia Alcista'].apply(lambda x: f"{x:.2f}")
                                    display_df['Score Ajustado'] = display_df['Score Ajustado'].apply(lambda x: f"{x:.2f}")
                                    
                                    st.dataframe(display_df, use_container_width=True)
                                    
                                    # Métricas actuales
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Pasan Corte Actual", f"{actual_count}/{top_n}")
                                    
                                    with col2:
                                        avg_inercia_current = current_picks_df['Inercia Alcista'].mean()
                                        st.metric("Inercia Promedio", f"{avg_inercia_current:.2f}")
                                    
                                    with col3:
                                        avg_score_current = current_picks_df['Score Ajustado'].mean()
                                        st.metric("Score Adj Promedio", f"{avg_score_current:.2f}")
                                    
                                    with col4:
                                        max_score_current = current_picks_df['Score Ajustado'].max()
                                        st.metric("Score Adj Máximo", f"{max_score_current:.2f}")
                                    
                                                                        # Comparación con último backtest
                                    if 'picks_df' in locals() and picks_df is not None and not picks_df.empty:
                                        st.subheader("🔄 Comparación con Últimos Picks del Backtest")
                                        
                                        # Obtener últimos picks del backtest
                                        latest_bt_date = picks_df["Date"].max()
                                        latest_bt_picks = picks_df[picks_df["Date"] == latest_bt_date]
                                        
                                        if not latest_bt_picks.empty:
                                            # Comparar tickers
                                            current_tickers = set(current_picks_df['Ticker'].tolist())
                                            bt_tickers = set(latest_bt_picks['Ticker'].tolist())
                                            
                                            # Tickers que se mantienen
                                            mantienen = current_tickers.intersection(bt_tickers)
                                            # Tickers nuevos
                                            nuevos = current_tickers - bt_tickers
                                            # Tickers que salen
                                            salen = bt_tickers - current_tickers
                                            
                                            col1, col2, col3 = st.columns(3)
                                            
                                            with col1:
                                                st.success(f"**Se Mantienen ({len(mantienen)})**")
                                                if mantienen:
                                                    for ticker in sorted(mantienen):
                                                        st.text(f"• {ticker}")
                                            
                                            with col2:
                                                st.info(f"**Nuevos ({len(nuevos)})**")
                                                if nuevos:
                                                    for ticker in sorted(nuevos):
                                                        st.text(f"• {ticker}")
                                            
                                            with col3:
                                                st.warning(f"**Salen ({len(salen)})**")
                                                if salen:
                                                    for ticker in sorted(salen):
                                                        st.text(f"• {ticker}")
                                            
                                            # Estadísticas de rotación
                                            rotacion_pct = (len(nuevos) + len(salen)) / (2 * len(current_tickers)) * 100 if current_tickers else 0
                                            st.metric("% Rotación vs Último Mes", f"{rotacion_pct:.1f}%")
                                    
                                    # Gráfico de comparación Score Ajustado
                                    try:
                                        fig_comparison = go.Figure()
                                        
                                        # Current picks
                                        fig_comparison.add_trace(go.Bar(
                                            x=current_picks_df['Ticker'],
                                            y=current_picks_df['Score Ajustado'],
                                            name='Señales Actuales',
                                            marker_color='lightblue',
                                            text=current_picks_df['Score Ajustado'].round(2),
                                            textposition='auto'
                                        ))
                                        
                                        fig_comparison.update_layout(
                                            title="Score Ajustado - Señales Actuales",
                                            xaxis_title="Ticker",
                                            yaxis_title="Score Ajustado",
                                            height=400,
                                            showlegend=True
                                        )
                                        
                                        st.plotly_chart(fig_comparison, use_container_width=True)
                                        
                                    except Exception as chart_error:
                                        st.warning(f"Error creando gráfico: {chart_error}")
                                    
                                    # Instrucciones para uso práctico
                                    st.subheader("📋 Cómo Usar Estas Señales")
                                    
                                    # Mostrar instrucciones según configuración
                                    if fixed_allocation:
                                        capital_info = f"- Cada posición recibirá exactamente 10% del capital\n- Con {actual_count} picks: {actual_count * 10}% invertido, {100 - actual_count * 10}% en efectivo"
                                    else:
                                        capital_info = f"- El capital se distribuye equitativamente: {100/actual_count:.1f}% por posición\n- 100% del capital invertido en {actual_count} posiciones"
                                    
                                    filter_info = ""
                                    if use_roc_filter or use_sma_filter:
                                        active_filters = []
                                        if use_roc_filter:
                                            active_filters.append("ROC 10M SPY < 0")
                                        if use_sma_filter:
                                            active_filters.append("SPY < SMA 10M")
                                        filter_info = f"\n\n**Filtros de Mercado Activos:**\n- {' y '.join(active_filters)}\n- Si se activan: venta inmediata de todas las posiciones"
                                    
                                    st.info(f"""
                                    **Para Trading Real:**
                                    1. 📅 **Espera al cierre del mes actual** para señales definitivas
                                    2. 🔄 **Recalcula el último día del mes** con datos completos
                                    3. 📈 **Toma posiciones el primer día del próximo mes**
                                    4. ⏰ **Mantén posiciones todo el mes** siguiente
                                    5. 🔁 **Repite el proceso** mensualmente
                                    
                                    **Distribución de Capital:**
                                    {capital_info}
                                    {filter_info}
                                    """)
                                    
                            else:
                                st.warning("No se encontraron señales actuales")
                        else:
                            st.warning("No hay datos suficientes para calcular señales actuales")
                    else:
                        st.error("No se pudieron calcular las señales actuales")
                        
                except Exception as e:
                    st.error(f"Error calculando señales actuales: {e}")
                    st.exception(e)

            # -------------------------------------------------
            # Información adicional sobre verificación histórica
            # -------------------------------------------------
            if historical_info and historical_info.get('has_historical_data', False):
                with st.expander("🕐 Detalles de Verificación Histórica", expanded=False):
                    st.subheader("Información sobre la verificación histórica")
                    
                    changes_data = historical_info['changes_data']
                    
                    st.info(f"""
                    **Datos históricos procesados:**
                    - Total de cambios: {len(changes_data)}
                    - Rango temporal: {changes_data['Date'].min()} a {changes_data['Date'].max()}
                    - Agregaciones: {len(changes_data[changes_data['Action'] == 'Added'])}
                    - Remociones: {len(changes_data[changes_data['Action'] == 'Removed'])}
                    """)
                    
                    # Mostrar algunos cambios recientes
                    st.subheader("Cambios recientes en los índices")
                    recent_changes = changes_data.head(10)
                    st.dataframe(recent_changes)
                    
                    # Debug: Verificar uso de tickers removidos
                    if picks_df is not None and not picks_df.empty:
                        st.subheader("🔍 Debug: Tickers Removidos en Backtest")
                        
                        # Ver cuáles aparecen en los picks
                        try:
                            removed_df = pd.read_csv("data/sp500_removed_tickers.csv")
                            removed_tickers = set(removed_df['Ticker'].tolist())
                            
                            picks_tickers = set(picks_df['Ticker'].tolist())
                            used_removed = picks_tickers.intersection(removed_tickers)
                            
                            st.info(f"✅ Tickers removidos usados en backtest: {len(used_removed)}")
                            if used_removed:
                                st.text(f"Ejemplos: {', '.join(sorted(list(used_removed))[:10])}")
                            
                        except:
                            st.info("No se pudo verificar uso de tickers removidos")
                    
                    # Gráfico de cambios por año
                    try:
                        changes_by_year = changes_data.copy()
                        changes_by_year['Year'] = pd.to_datetime(changes_by_year['Date']).dt.year
                        changes_summary = changes_by_year.groupby(['Year', 'Action']).size().unstack(fill_value=0)
                        
                        if not changes_summary.empty:
                            fig_changes = go.Figure()
                            
                            if 'Added' in changes_summary.columns:
                                fig_changes.add_trace(go.Bar(
                                    x=changes_summary.index,
                                    y=changes_summary['Added'],
                                    name='Agregados',
                                    marker_color='green'
                                ))
                            
                            if 'Removed' in changes_summary.columns:
                                fig_changes.add_trace(go.Bar(
                                    x=changes_summary.index,
                                    y=changes_summary['Removed'],
                                    name='Removidos',
                                    marker_color='red'
                                ))
                            
                            fig_changes.update_layout(
                                title="Cambios en Índices por Año",
                                xaxis_title="Año",
                                yaxis_title="Número de Cambios",
                                barmode='group',
                                height=400
                            )
                            
                            st.plotly_chart(fig_changes, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Error creando gráfico de cambios: {e}")

    except Exception as e:
        st.error(f"❌ Excepción no capturada: {str(e)}")
        st.exception(e)
        st.info("💡 Consejos para resolver este problema:")
        st.info("1. Verifica que los archivos CSV existan en la carpeta 'data/'")
        st.info("2. Asegúrate de que los archivos tengan el formato correcto")
        st.info("3. Prueba con un rango de fechas más corto")
        st.info("4. Verifica que los tickers sean válidos")
        st.info("5. Desactiva la verificación histórica si hay problemas de conectividad")

else:
    st.info("👈 Configura los parámetros en el panel lateral y haz clic en 'Ejecutar backtest'")
    
    # Información sobre el sistema
    st.subheader("🔍 Información del Sistema")
    st.info("""
    **Características principales:**
    - ✅ Verificación histórica de constituyentes de índices (opcional)
    - ✅ Cálculos de inercia compatibles con AmiBroker  
    - ✅ Datos OHLC reales para cálculos precisos
    - ✅ Eliminación del sesgo de supervivencia
    - ✅ Señales actuales filtradas por criterios estrictos
    - ✅ Distribución de capital configurable (fija 10% o equitativa)
    - ✅ Filtros de mercado para protección en bajadas
    
    **Datos requeridos:**
    - Archivos CSV en carpeta 'data/' con formato: TICKER.csv
    - Columnas: Date, High, Low, Close, Adj Close (recomendado)
    - Benchmark: SPY.csv y/o QQQ.csv
    """)
    
    st.subheader("💡 Consejos para mejores resultados")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Configuración recomendada:**
        • Usa un rango de fechas de al menos 2 años
        • Comienza con 10 activos y ajusta según resultados
        • Activa verificación histórica para mayor realismo
        • Considera usar filtros de mercado para protección
        """)
    
    with col2:
        st.info("""
        **Nuevas opciones:**
        • Asignación fija 10%: Mantiene capital en efectivo
        • Filtros de mercado: Protegen en mercados bajistas
        • ROC SPY: Evita invertir con momentum negativo
        • SMA SPY: Evita invertir por debajo de media móvil
        """)
