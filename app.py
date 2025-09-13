import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import yfinance as yf
import numpy as np

# Importar nuestros módulos
from data_loader import download_prices, get_constituents_at_date
from backtest import run_backtest

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

# Selector de índice - MODIFICADO para permitir selección múltiple
index_choice = st.sidebar.selectbox(
    "Selecciona el índice:",
    ["SP500", "NDX", "Ambos (SP500 + NDX)"] # Opción añadida
)

# Fechas
end_date = st.sidebar.date_input("Fecha final", datetime.today())
start_date = st.sidebar.date_input("Fecha inicial", end_date - timedelta(days=365*5))

# Parámetros del backtest
top_n = st.sidebar.slider("Número de activos", 5, 30, 10)
commission = st.sidebar.number_input("Comisión por operación (%)", 0.0, 1.0, 0.3) / 100
corte = st.sidebar.number_input("Corte de score", 0, 1000, 680)

# Botón de ejecución
run_button = st.sidebar.button("🏃 Ejecutar backtest")

# -------------------------------------------------
# Main content
# -------------------------------------------------
if run_button:
    try:
        with st.spinner("Descargando datos..."):
            # Lógica para obtener tickers de uno o ambos índices
            all_tickers_data = {'tickers': [], 'data': []}
            
            indices_to_fetch = []
            if index_choice == "SP500":
                indices_to_fetch = ["SP500"]
            elif index_choice == "NDX":
                indices_to_fetch = ["NDX"]
            else: # "Ambos (SP500 + NDX)"
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

            # Descargar precios de constituyentes
            prices_df = download_prices(all_tickers_data, start_date, end_date)
            
            # Validación adicional de precios
            if prices_df is None or prices_df.empty or len(prices_df.columns) == 0:
                st.error("❌ No se pudieron descargar los precios históricos de los constituyentes")
                st.info("💡 Consejos para resolver este problema:")
                st.info("1. Verifica tu conexión a internet")
                st.info("2. Prueba con un rango de fechas más corto")
                st.info("3. Intenta con menos tickers (un solo índice)")
                st.info("4. Asegúrate de que los tickers sean válidos")
                st.stop()
            
            st.success(f"✅ Descargados precios para {len(prices_df.columns)} tickers")
            st.info(f"Rango de fechas: {prices_df.index.min().strftime('%Y-%m-%d')} a {prices_df.index.max().strftime('%Y-%m-%d')}")
            st.info(f"Muestra de tickers: {', '.join(list(prices_df.columns)[:5])}")

            # Descargar benchmark (SPY para S&P 500, QQQ para Nasdaq-100, SPY para ambos)
            # Se podría mejorar para usar un benchmark ponderado si se seleccionan ambos
            if index_choice == "SP500":
                benchmark_ticker = "SPY"
            elif index_choice == "NDX":
                benchmark_ticker = "QQQ"
            else: # Ambos
                benchmark_ticker = "SPY" # Por simplicidad, usar SPY. Se podría mejorar.
            
            st.info(f"Descargando benchmark: {benchmark_ticker}")
            benchmark_df = download_prices([benchmark_ticker], start_date, end_date)
            
            if benchmark_df is None or benchmark_df.empty:
                st.warning(f"No se pudo descargar el benchmark {benchmark_ticker}")
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
                st.success(f"✅ Benchmark {benchmark_ticker} descargado correctamente")

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
                
            bt_results, picks_df = run_backtest(
                prices=prices_df,
                benchmark=benchmark_series,
                commission=commission,
                top_n=top_n,
                corte=corte
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
            # Métricas principales
            # -------------------------------------------------
            if "Equity" in bt_results.columns and len(bt_results["Equity"]) > 0:
                final_equity = float(bt_results["Equity"].iloc[-1])
                initial_equity = float(bt_results["Equity"].iloc[0])
                total_return = (final_equity / initial_equity) - 1 if initial_equity != 0 else 0
            else:
                final_equity = 10000
                total_return = 0
                
            if "Drawdown" in bt_results.columns:
                max_drawdown = float(bt_results["Drawdown"].min())
            else:
                max_drawdown = 0
                
            if "Returns" in bt_results.columns and len(bt_results["Returns"]) > 1:
                volatility = float(bt_results["Returns"].std() * (12 ** 0.5)) if bt_results["Returns"].std() != 0 else 0
                sharpe_ratio = (float(bt_results["Returns"].mean() * 12) / (volatility + 1e-8)) if volatility != 0 else 0
            else:
                volatility = 0
                sharpe_ratio = 0
                
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Equity Final", f"${final_equity:,.0f}")
            col2.metric("Retorno Total", f"{total_return:.2%}")
            col3.metric("Máximo Drawdown", f"{max_drawdown:.2%}")
            col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

            # -------------------------------------------------
            # Gráficos
            # -------------------------------------------------
            # Gráfico de equity
            try:
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    x=bt_results.index,
                    y=bt_results["Equity"],
                    mode='lines',
                    name='Estrategia',
                    line=dict(width=3),
                    hovertemplate='<b>%{y:,.0f}</b><br>%{x}<extra></extra>'
                ))
                
                # Benchmark
                if benchmark_df is not None and not benchmark_df.empty:
                    try:
                        bench_data = benchmark_df[benchmark_ticker] if benchmark_ticker in benchmark_df.columns else benchmark_df.iloc[:, 0]
                        bench_returns = bench_data.pct_change().fillna(0)
                        bench_equity = 10000 * (1 + bench_returns).cumprod()
                        
                        # Alinear índices
                        common_index = bt_results.index.intersection(bench_equity.index)
                        if len(common_index) > 0:
                            bt_aligned = bt_results.loc[common_index]
                            bench_aligned = bench_equity.loc[common_index]
                            
                            fig_equity.add_trace(go.Scatter(
                                x=bench_aligned.index,
                                y=bench_aligned.values,
                                mode='lines',
                                name=f'Benchmark ({benchmark_ticker})',
                                line=dict(width=2, dash='dash'),
                                hovertemplate='<b>%{y:,.0f}</b><br>%{x}<extra></extra>'
                            ))
                    except Exception as bench_error:
                        st.warning(f"Error al procesar benchmark para gráfico: {bench_error}")
                
                fig_equity.update_layout(
                    title="Evolución del Equity",
                    xaxis_title="Fecha",
                    yaxis_title="Equity ($)",
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig_equity, use_container_width=True)
                
            except Exception as fig_error:
                st.warning(f"Error al crear gráfico de equity: {fig_error}")

            # Gráfico de drawdown
            try:
                if "Drawdown" in bt_results.columns:
                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Scatter(
                        x=bt_results.index,
                        y=bt_results["Drawdown"] * 100,
                        mode='lines',
                        name='Drawdown',
                        fill='tozeroy',
                        line=dict(color='red', width=2),
                        hovertemplate='<b>%{y:.2f}%</b><br>%{x}<extra></extra>'
                    ))
                    fig_dd.update_layout(
                        title="Drawdown",
                        xaxis_title="Fecha",
                        yaxis_title="Drawdown (%)",
                        hovermode='x unified',
                        height=400
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
                        st.dataframe(latest_picks.round(2))
                    else:
                        st.info("No hay picks recientes para mostrar")
                    
                    # Mostrar picks de todos los meses
                    st.subheader("Todos los picks por mes")
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

    except Exception as e:
        st.error(f"❌ Excepción no capturada: {str(e)}")
        st.exception(e)
        st.info("💡 Consejos para resolver este problema:")
        st.info("1. Verifica tu conexión a internet")
        st.info("2. Prueba con un rango de fechas más corto")
        st.info("3. Intenta con menos tickers (un solo índice)")
        st.info("4. Asegúrate de que los tickers sean válidos")
else:
    st.info("👈 Configura los parámetros en el panel lateral y haz clic en 'Ejecutar backtest'")
    st.info("💡 Consejos para mejores resultados:")
    st.info("• Usa un rango de fechas de al menos 2 años")
    st.info("• Comienza con 10 activos y ajusta según los resultados")
    st.info("• Considera usar ambos índices para mayor diversificación")
