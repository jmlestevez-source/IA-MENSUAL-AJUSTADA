import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Importar tus funciones actuales (renombra tu app.py actual a stock_utils.py)
from stock_utils import load_stock_data, get_stock_info, get_available_periods
from data_loader import get_constituents_at_date, download_prices, get_sp500_historical_changes, get_nasdaq100_historical_changes
from backtest import run_backtest, calculate_monthly_returns_by_year, calculate_sharpe_ratio

# Configuración de página
st.set_page_config(
    page_title="IA Mensual Ajustada",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.title("📊 IA Mensual Ajustada - Sistema de Trading Cuantitativo")
st.markdown("Sistema de backtest con estrategia de inercia mensual para índices bursátiles")

# Sidebar
st.sidebar.header("⚙️ Configuración del Backtest")

# Tipo de operación
operation_mode = st.sidebar.radio(
    "Modo de Operación",
    ["🚀 Backtest Rápido", "📈 Análisis Individual", "📊 Backtest Completo"]
)

if operation_mode == "🚀 Backtest Rápido":
    st.header("🚀 Backtest Rápido")
    
    col1, col2 = st.columns(2)
    with col1:
        quick_period = st.selectbox(
            "Período",
            ["1 año", "2 años", "5 años"],
            index=1
        )
    with col2:
        quick_index = st.selectbox(
            "Índice",
            ["S&P 500", "NASDAQ-100", "Ambos"],
            index=0
        )
    
    if st.button("▶️ Ejecutar Backtest Rápido", type="primary"):
        with st.spinner("Ejecutando backtest..."):
            # Mapear opciones a valores
            period_map = {"1 año": 1, "2 años": 2, "5 años": 5}
            years_back = period_map[quick_period]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years_back)
            
            # Mensaje de progreso
            progress_placeholder = st.empty()
            progress_placeholder.info(f"📥 Obteniendo constituyentes del {quick_index}...")
            
            # Obtener tickers
            index_code = "SP500" if quick_index == "S&P 500" else "NDX" if quick_index == "NASDAQ-100" else "AMBOS"
            constituents, error = get_constituents_at_date(index_code, start_date.date(), end_date.date())
            
            if error:
                st.warning(f"Advertencia: {error}")
            
            if constituents and 'tickers' in constituents:
                tickers = constituents['tickers'][:50]  # Limitar para rapidez
                progress_placeholder.info(f"📊 Cargando datos de {len(tickers)} acciones...")
                
                # Descargar datos
                prices_df, ohlc_data = download_prices(tickers, start_date.date(), end_date.date(), load_full_data=True)
                
                if not prices_df.empty:
                    # Benchmark
                    spy_df = download_prices(["SPY"], start_date.date(), end_date.date(), load_full_data=False)
                    spy_series = spy_df["SPY"] if not spy_df.empty else None
                    
                    # Información histórica
                    historical_changes = None
                    if index_code == "SP500":
                        historical_changes = get_sp500_historical_changes()
                    elif index_code == "NDX":
                        historical_changes = get_nasdaq100_historical_changes()
                    
                    historical_info = {'changes_data': historical_changes} if historical_changes is not None else None
                    
                    progress_placeholder.info("🧮 Ejecutando backtest...")
                    
                    # Ejecutar backtest
                    bt, picks_df = run_backtest(
                        prices=prices_df,
                        benchmark=spy_series if spy_series is not None else prices_df.iloc[:, 0],
                        commission=0.003,
                        top_n=10,
                        corte=680,
                        ohlc_data=ohlc_data,
                        historical_info=historical_info,
                        fixed_allocation=False
                    )
                    
                    progress_placeholder.empty()
                    
                    if not bt.empty:
                        # Mostrar resultados
                        st.success(f"✅ Backtest completado exitosamente")
                        
                        # Métricas principales
                        col1, col2, col3, col4 = st.columns(4)
                        
                        initial_capital = 10000
                        final_equity = bt['Equity'].iloc[-1]
                        total_return = (final_equity / initial_capital - 1) * 100
                        max_dd = bt['Drawdown'].min() * 100
                        sharpe = calculate_sharpe_ratio(bt['Returns'])
                        
                        col1.metric("Capital Final", f"${final_equity:,.0f}", f"{total_return:+.1f}%")
                        col2.metric("Retorno Total", f"{total_return:.1f}%")
                        col3.metric("Max Drawdown", f"{max_dd:.1f}%")
                        col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
                        
                        # Gráfico de equity
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=bt.index,
                            y=bt['Equity'],
                            mode='lines',
                            name='Estrategia',
                            line=dict(color='blue', width=2)
                        ))
                        
                        if spy_series is not None:
                            spy_equity = initial_capital * (spy_series / spy_series.iloc[0])
                            fig.add_trace(go.Scatter(
                                x=spy_equity.index,
                                y=spy_equity,
                                mode='lines',
                                name='SPY (Benchmark)',
                                line=dict(color='gray', width=1, dash='dot')
                            ))
                        
                        fig.update_layout(
                            title="Curva de Equity",
                            xaxis_title="Fecha",
                            yaxis_title="Valor ($)",
                            hovermode='x unified',
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Mostrar picks recientes
                        if not picks_df.empty:
                            st.subheader("🎯 Últimas Selecciones")
                            recent_picks = picks_df.tail(20)
                            st.dataframe(
                                recent_picks[['Date', 'Ticker', 'Inercia', 'ScoreAdj']],
                                use_container_width=True
                            )
                    else:
                        st.error("❌ No se generaron resultados del backtest")
                else:
                    st.error("❌ No se pudieron cargar los datos de precios")
            else:
                st.error("❌ No se encontraron constituyentes válidos")

elif operation_mode == "📈 Análisis Individual":
    st.header("📈 Análisis de Acción Individual")
    
    ticker = st.text_input("Símbolo de la acción", value="AAPL").upper()
    period = st.selectbox("Período", list(get_available_periods().keys()), index=3)
    
    if st.button("📊 Analizar", type="primary"):
        with st.spinner(f"Cargando datos de {ticker}..."):
            try:
                # Cargar datos
                period_code = get_available_periods()[period]
                data = load_stock_data(ticker, period_code)
                
                # Información de la acción
                info = get_stock_info(ticker)
                
                # Mostrar información
                col1, col2, col3 = st.columns(3)
                col1.metric("Empresa", info['name'])
                col2.metric("Sector", info['sector'])
                col3.metric("P/E Ratio", f"{info['pe_ratio']:.2f}" if info['pe_ratio'] > 0 else "N/A")
                
                # Gráfico de precios
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data['Date'],
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=ticker
                ))
                fig.update_layout(
                    title=f"Precio de {ticker}",
                    xaxis_title="Fecha",
                    yaxis_title="Precio ($)",
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Validación de datos
                is_valid, message = validate_data_integrity(data)
                if is_valid:
                    st.success(f"✅ {message}")
                else:
                    st.warning(f"⚠️ {message}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif operation_mode == "📊 Backtest Completo":
    st.header("📊 Backtest Completo Personalizado")
    
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.slider("Año de inicio", 1990, 2023, 2019)
        end_year = st.slider("Año de fin", start_year + 1, 2024, 2024)
    
    with col2:
        index_choice = st.selectbox("Índice", ["S&P 500", "NASDAQ-100", "Ambos"])
        top_n = st.slider("Top N acciones", 5, 20, 10)
    
    with st.expander("⚙️ Configuración Avanzada"):
        corte_inercia = st.number_input("Corte de Inercia", 100, 2000, 680, step=10)
        commission = st.number_input("Comisión (%)", 0.0, 1.0, 0.3, step=0.1) / 100
        fixed_allocation = st.checkbox("Asignación fija (10% por posición)", False)
        use_filters = st.checkbox("Usar filtros de mercado", False)
        
        if use_filters:
            col1, col2 = st.columns(2)
            with col1:
                use_roc_filter = st.checkbox("Filtro ROC (12 meses)", True)
            with col2:
                use_sma_filter = st.checkbox("Filtro SMA (10 meses)", False)
        else:
            use_roc_filter = False
            use_sma_filter = False
    
    if st.button("🚀 Ejecutar Backtest Completo", type="primary"):
        st.info("⏳ Este proceso puede tardar varios minutos dependiendo del rango de fechas...")
        
        # Aquí iría la lógica completa similar a la del backtest rápido
        # pero con los parámetros personalizados
        st.warning("🚧 Funcionalidad en desarrollo. Use el Backtest Rápido por ahora.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>IA Mensual Ajustada | Sistema de Trading Cuantitativo | 
        <a href='https://github.com/tu-usuario/tu-repo'>GitHub</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
