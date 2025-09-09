import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf

# Importar nuestros módulos
from data_loader import download_prices, get_constituents_at_date
from backtest import run_backtest
from utils import unify_ticker

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
    ["SP500", "NDX"]
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
            # Obtener constituyentes
            constituents_data, error = get_constituents_at_date(index_choice, start_date, end_date)
            
            if error:
                st.error(f"Error obteniendo constituyentes: {error}")
                st.stop()
            
            if constituents_data is None:
                st.error("No se pudieron obtener los constituyentes del índice")
                st.stop()
            
            tickers_count = len(constituents_data.get('tickers', []))
            st.success(f"✅ Obtenidos {tickers_count} constituyentes")
            
            if tickers_count == 0:
                st.error("No se encontraron tickers válidos")
                st.stop()
            
            # Descargar precios
            prices_df = download_prices(constituents_data, start_date, end_date)
            
            if prices_df is None or prices_df.empty:
                st.error("No se pudieron descargar los precios históricos")
                # Mostrar información de debugging
                if isinstance(constituents_data, dict) and 'tickers' in constituents_data:
                    st.info(f"Tickers intentados: {constituents_data['tickers'][:10]}...")
                st.stop()
            
            st.success(f"✅ Descargados precios para {len(prices_df.columns)} tickers")
            
            # Benchmark (SPY para S&P 500, QQQ para Nasdaq-100)
            benchmark_ticker = "SPY" if index_choice == "SP500" else "QQQ"
            benchmark_df = download_prices([benchmark_ticker], start_date, end_date)
            
            if benchmark_df is None or benchmark_df.empty:
                st.warning(f"No se pudo descargar el benchmark {benchmark_ticker}, usando precios medios")
                benchmark_df = prices_df.mean(axis=1).to_frame(name=benchmark_ticker)
        
        with st.spinner("Ejecutando backtest..."):
            # Ejecutar backtest
            bt_results, picks_df = run_backtest(
                prices=prices_df,
                benchmark=benchmark_df[benchmark_ticker],
                commission=commission,
                top_n=top_n,
                corte=corte
            )
            
            if bt_results.empty:
                st.error("El backtest no generó resultados")
                st.stop()
            
            st.success("✅ Backtest completado")
            
            # -------------------------------------------------
            # Métricas principales
            # -------------------------------------------------
            
            final_equity = bt_results["Equity"].iloc[-1]
            total_return = (final_equity / bt_results["Equity"].iloc[0]) - 1
            max_drawdown = bt_results["Drawdown"].min()
            volatility = bt_results["Returns"].std() * (12 ** 0.5)  # Anualizada
            sharpe_ratio = (bt_results["Returns"].mean() * 12) / (volatility + 1e-8)  # Sharpe anualizado
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Equity Final", f"${final_equity:,.0f}")
            col2.metric("Retorno Total", f"{total_return:.2%}")
            col3.metric("Máximo Drawdown", f"{max_drawdown:.2%}")
            col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            # -------------------------------------------------
            # Gráficos
            # -------------------------------------------------
            
            # Gráfico de equity
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=bt_results.index,
                y=bt_results["Equity"],
                mode='lines',
                name='Estrategia',
                line=dict(width=3)
            ))
            
            # Benchmark
            if not benchmark_df.empty:
                bench_equity = (10000 * (1 + benchmark_df[benchmark_ticker].pct_change().fillna(0))).cumprod()
                bench_equity.index = bt_results.index[:len(bench_equity)]  # Alinear fechas
                fig_equity.add_trace(go.Scatter(
                    x=bench_equity.index,
                    y=bench_equity,
                    mode='lines',
                    name=f'Benchmark ({benchmark_ticker})',
                    line=dict(width=2, dash='dash')
                ))
            
            fig_equity.update_layout(
                title="Evolución del Equity",
                xaxis_title="Fecha",
                yaxis_title="Equity ($)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # Gráfico de drawdown
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=bt_results.index,
                y=bt_results["Drawdown"] * 100,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red', width=2)
            ))
            fig_dd.update_layout(
                title="Drawdown",
                xaxis_title="Fecha",
                yaxis_title="Drawdown (%)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_dd, use_container_width=True)
            
            # -------------------------------------------------
            # Picks seleccionados
            # -------------------------------------------------
            
            if not picks_df.empty:
                st.subheader("Últimos picks seleccionados")
                latest_picks = picks_df[picks_df["Date"] == picks_df["Date"].max()]
                st.dataframe(latest_picks.round(2))
                
                # Gráfico de picks por fecha
                picks_by_date = picks_df.groupby("Date").size()
                fig_picks = px.bar(
                    x=picks_by_date.index,
                    y=picks_by_date.values,
                    labels={'x': 'Fecha', 'y': 'Número de Picks'},
                    title="Número de Picks por Fecha"
                )
                st.plotly_chart(fig_picks, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Excepción no capturada: {str(e)}")
        st.exception(e)

else:
    st.info("👈 Configura los parámetros en el panel lateral y haz clic en 'Ejecutar backtest'")
