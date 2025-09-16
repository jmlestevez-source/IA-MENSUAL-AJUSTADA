import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import ta
import requests
from io import StringIO

# Manejo de importaciones con fallback
try:
    from backtest import run_optimized_backtest, optimized_inertia_score, calcular_atr_amibroker, calculate_monthly_returns_by_year
    BACKTEST_AVAILABLE = True
except ImportError:
    BACKTEST_AVAILABLE = False
    st.warning("Módulo de backtest no disponible. Algunas funcionalidades estarán limitadas.")

# Configuración de la página
st.set_page_config(
    page_title="IA Mensual Ajustada",
    page_icon="📊",
    layout="wide"
)

# Título de la aplicación
st.title("🤖 IA Mensual Ajustada - Análisis Técnico Avanzado")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selección de activo
    symbol = st.text_input("Símbolo del activo", "AAPL").upper()
    
    # Selección de período
    period_options = {
        "1 año": "1y",
        "2 años": "2y", 
        "5 años": "5y",
        "10 años": "10y",
        "Máximo": "max"
    }
    selected_period = st.selectbox("Período de datos", list(period_options.keys()))
    period = period_options[selected_period]
    
    # Parámetros de estrategia
    st.subheader("📈 Parámetros de Estrategia")
    sma_short = st.slider("Media Móvil Corta", 5, 50, 20)
    sma_long = st.slider("Media Móvil Larga", 20, 200, 50)
    rsi_period = st.slider("Período RSI", 5, 30, 14)
    rsi_overbought = st.slider("RSI Sobrecomprado", 50, 90, 70)
    rsi_oversold = st.slider("RSI Sobre vendido", 10, 50, 30)
    
    # Parámetros adicionales
    st.subheader("📊 Parámetros Adicionales")
    atr_period = st.slider("Período ATR", 7, 21, 14)
    volatility_threshold = st.slider("Umbral de Volatilidad (%)", 1.0, 10.0, 2.5, 0.1)

# Función para cargar datos
@st.cache_data
def load_data(symbol, period):
    try:
        data = yf.download(symbol, period=period)
        if data.empty:
            return None
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error al descargar datos: {str(e)}")
        return None

# Función para calcular indicadores técnicos
def calculate_indicators(df, sma_short, sma_long, rsi_period, atr_period):
    df = df.copy()
    
    # Medias móviles
    df[f'SMA_{sma_short}'] = ta.trend.sma_indicator(df['Close'], window=sma_short)
    df[f'SMA_{sma_long}'] = ta.trend.sma_indicator(df['Close'], window=sma_long)
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=rsi_period)
    
    # ATR (Average True Range)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    
    # Volatilidad como porcentaje
    df['Volatility'] = (df['ATR'] / df['Close']) * 100
    
    # Señales de compra/venta mejoradas
    df['Signal'] = 0
    
    # Señal de compra: SMA corta cruza por encima de SMA larga y RSI no sobrecomprado
    buy_condition = (
        (df[f'SMA_{sma_short}'] > df[f'SMA_{sma_long}']) & 
        (df['RSI'] < rsi_overbought) & 
        (df['RSI'] > rsi_oversold) &
        (df['Volatility'] > volatility_threshold)
    )
    
    # Señal de venta: SMA corta cruza por debajo de SMA larga o RSI sobrecomprado
    sell_condition = (
        (df[f'SMA_{sma_short}'] < df[f'SMA_{sma_long}']) | 
        (df['RSI'] > rsi_overbought)
    )
    
    df.loc[buy_condition, 'Signal'] = 1
    df.loc[sell_condition, 'Signal'] = -1
    
    return df

# Función para crear gráfico avanzado
def create_advanced_chart(df, symbol):
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=(f'Precio de {symbol}', 'RSI', 'ATR (Volatilidad)', 'Señales')
    )
    
    # Velas japonesas
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Precio OHLC',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Medias móviles
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df[f'SMA_{sma_short}'], 
                  name=f'SMA {sma_short}', line=dict(color='orange', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df[f'SMA_{sma_long}'], 
                  name=f'SMA {sma_long}', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['RSI'], 
                  name='RSI', line=dict(color='purple', width=2)),
        row=2, col=1
    )
    
    # Líneas de sobrecompra y sobreventa del RSI
    fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
    
    # ATR (Volatilidad)
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Volatility'], 
                  name='Volatilidad %', line=dict(color='orange', width=2)),
        row=3, col=1
    )
    
    # Línea de umbral de volatilidad
    fig.add_hline(y=volatility_threshold, line_dash="dash", line_color="blue", row=3, col=1)
    
    # Señales de compra/venta
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    
    fig.add_trace(
        go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], 
                  mode='markers', name='Compra', 
                  marker=dict(color='green', size=12, symbol='triangle-up')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], 
                  mode='markers', name='Venta', 
                  marker=dict(color='red', size=12, symbol='triangle-down')),
        row=1, col=1
    )
    
    # Señales en el panel inferior
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Signal'], 
                  mode='lines+markers', name='Señales Trading', 
                  line=dict(color='black', width=1),
                  marker=dict(size=4)),
        row=4, col=1
    )
    
    fig.update_layout(
        height=900,
        showlegend=True,
        title_text=f"📊 Análisis Técnico Avanzado de {symbol}",
        title_font_size=20,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Fecha", row=4, col=1)
    fig.update_yaxes(title_text="Precio ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Volatilidad (%)", row=3, col=1)
    fig.update_yaxes(title_text="Señales", row=4, col=1)
    
    return fig

# Función para calcular estadísticas del activo
def calculate_asset_stats(df):
    if df.empty:
        return {}
    
    # Rendimiento total
    total_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
    
    # Volatilidad anualizada
    daily_returns = df['Close'].pct_change().dropna()
    annual_volatility = daily_returns.std() * np.sqrt(252) * 100
    
    # Ratio Sharpe (asumiendo tasa libre de riesgo = 2%)
    risk_free_rate = 0.02
    sharpe_ratio = (daily_returns.mean() * 252 - risk_free_rate) / (daily_returns.std() * np.sqrt(252))
    
    # Máximo drawdown
    rolling_max = df['Close'].expanding().max()
    daily_drawdown = df['Close'] / rolling_max - 1
    max_drawdown = daily_drawdown.min() * 100
    
    return {
        'total_return': total_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'current_price': df['Close'].iloc[-1],
        'price_change': ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100 if len(df) > 1 else 0
    }

# Función para crear tabla de estadísticas
def create_stats_table(stats):
    if not stats:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Precio Actual", f"${stats['current_price']:.2f}", 
                 f"{stats['price_change']:.2f}%" if stats['price_change'] >= 0 else f"{stats['price_change']:.2f}%")
    
    with col2:
        st.metric("Rentabilidad Total", f"{stats['total_return']:.2f}%", 
                 delta=None, delta_color="normal")
    
    with col3:
        st.metric("Volatilidad Anual", f"{stats['annual_volatility']:.2f}%", 
                 delta=None, delta_color="inverse")
    
    with col4:
        st.metric("Ratio Sharpe", f"{stats['sharpe_ratio']:.2f}", 
                 delta=None, delta_color="normal")
    
    # Segunda fila de métricas
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Máximo Drawdown", f"{stats['max_drawdown']:.2f}%", 
                 delta=None, delta_color="inverse")

# Función para simulación de trading simple
def simple_trading_simulation(df):
    if df.empty:
        return None
    
    df = df.copy()
    df['Position'] = 0
    df['Portfolio_Value'] = 10000  # Valor inicial $10,000
    
    # Generar posiciones basadas en señales
    position = 0
    portfolio_value = 10000
    shares = 0
    
    for i in range(len(df)):
        if df['Signal'].iloc[i] == 1 and position == 0:  # Comprar
            position = 1
            shares = portfolio_value / df['Close'].iloc[i]
        elif df['Signal'].iloc[i] == -1 and position == 1:  # Vender
            position = 0
            portfolio_value = shares * df['Close'].iloc[i]
            shares = 0
        
        df.loc[df.index[i], 'Position'] = position
        df.loc[df.index[i], 'Portfolio_Value'] = portfolio_value if position == 0 else shares * df['Close'].iloc[i]
    
    return df

# Función para crear gráfico de simulación
def create_simulation_chart(df):
    fig = go.Figure()
    
    # Valor del portafolio
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Portfolio_Value'], 
                  name='Valor Portafolio', line=dict(color='blue', width=3))
    )
    
    # Precio del activo (normalizado)
    normalized_price = (df['Close'] / df['Close'].iloc[0]) * 10000
    fig.add_trace(
        go.Scatter(x=df['Date'], y=normalized_price, 
                  name='Precio Activo (Normalizado)', line=dict(color='gray', width=2, dash='dot'))
    )
    
    fig.update_layout(
        title="Simulación de Trading",
        xaxis_title="Fecha",
        yaxis_title="Valor ($)",
        height=500,
        showlegend=True
    )
    
    return fig

# Función principal de la aplicación
def main():
    st.header(f"🔍 Análisis de {symbol}")
    
    # Cargar datos
    with st.spinner(f'Cargando datos de {symbol}...'):
        data = load_data(symbol, period)
    
    if data is not None and not data.empty:
        # Calcular indicadores
        with st.spinner('Calculando indicadores técnicos...'):
            data_with_indicators = calculate_indicators(data, sma_short, sma_long, rsi_period, atr_period)
        
        # Calcular estadísticas
        stats = calculate_asset_stats(data_with_indicators)
        
        # Mostrar estadísticas principales
        st.subheader("📈 Estadísticas del Activo")
        create_stats_table(stats)
        
        # Mostrar gráfico avanzado
        st.subheader("📊 Análisis Técnico Completo")
        chart = create_advanced_chart(data_with_indicators, symbol)
        st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
        
        # Simulación de trading
        st.subheader("🎮 Simulación de Trading")
        simulated_data = simple_trading_simulation(data_with_indicators)
        if simulated_data is not None:
            simulation_chart = create_simulation_chart(simulated_data)
            st.plotly_chart(simulation_chart, use_container_width=True)
            
            # Resultados de la simulación
            final_value = simulated_data['Portfolio_Value'].iloc[-1]
            buy_and_hold = (data_with_indicators['Close'].iloc[-1] / data_with_indicators['Close'].iloc[0]) * 10000
            strategy_return = ((final_value / 10000) - 1) * 100
            buy_hold_return = ((buy_and_hold / 10000) - 1) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Valor Final Estrategia", f"${final_value:.2f}", f"{strategy_return:.2f}%")
            col2.metric("Buy & Hold", f"${buy_and_hold:.2f}", f"{buy_hold_return:.2f}%")
            col3.metric("Diferencia", f"${final_value - buy_and_hold:.2f}", 
                       f"{strategy_return - buy_hold_return:.2f}%")
        
        # Datos recientes
        st.subheader("📋 Datos Recientes")
        st.dataframe(data_with_indicators.tail(10)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', f'SMA_{sma_short}', f'SMA_{sma_long}']].round(2))
        
        # Ejecutar backtest si está disponible
        if BACKTEST_AVAILABLE:
            st.subheader("🔬 Backtest Avanzado")
            
            try:
                with st.spinner('Ejecutando backtest optimizado...'):
                    backtest_results = run_optimized_backtest(data_with_indicators)
                    
                    if backtest_results and 'error' not in backtest_results:
                        # Métricas del backtest
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Rentabilidad Total", f"{backtest_results.get('total_return', 0):.2f}%")
                        col2.metric("Ratio Sharpe", f"{backtest_results.get('sharpe_ratio', 0):.2f}")
                        col3.metric("Máximo Drawdown", f"{backtest_results.get('max_drawdown', 0):.2f}%")
                        col4.metric("Número de Operaciones", backtest_results.get('num_trades', 0))
                        
                        # Gráfico de equity curve si está disponible
                        if 'equity_curve' in backtest_results:
                            fig_equity = go.Figure()
                            fig_equity.add_trace(go.Scatter(
                                x=list(range(len(backtest_results['equity_curve']))),
                                y=backtest_results['equity_curve'],
                                name='Curva de Equity',
                                line=dict(color='blue', width=2)
                            ))
                            fig_equity.update_layout(
                                title="Curva de Equity del Backtest",
                                xaxis_title="Período",
                                yaxis_title="Valor del Portafolio",
                                height=400
                            )
                            st.plotly_chart(fig_equity, use_container_width=True)
                            
                    else:
                        st.info("Backtest completado pero sin resultados detallados.")
                        
            except Exception as e:
                st.error(f"Error en el backtest: {str(e)}")
        
        # Análisis de puntuación de inercia
        if BACKTEST_AVAILABLE and 'optimized_inertia_score' in globals():
            try:
                with st.spinner('Calculando puntuación de inercia...'):
                    inertia_score = optimized_inertia_score(data_with_indicators)
                    st.subheader("🎯 Puntuación de Inercia del Mercado")
                    
                    # Visualización de la puntuación
                    fig_inertia = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=inertia_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Inercia del Mercado"},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.3], 'color': "lightcoral"},
                                {'range': [0.3, 0.7], 'color': "gold"},
                                {'range': [0.7, 1], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': inertia_score
                            }
                        }
                    ))
                    
                    fig_inertia.update_layout(height=300)
                    st.plotly_chart(fig_inertia, use_container_width=True)
                    
                    # Interpretación
                    if inertia_score > 0.7:
                        st.success("🟢 Alta inercia: Tendencia fuerte y sostenida - condiciones favorables para momentum trading")
                    elif inertia_score > 0.4:
                        st.info("🟡 Inercia moderada: Tendencia estable pero con posible corrección - estrategia mixta recomendada")
                    else:
                        st.warning("🔴 Baja inercia: Tendencia débil o sin dirección clara - considerar estrategias de rango o espera")
                        
            except Exception as e:
                st.error(f"Error calculando puntuación de inercia: {str(e)}")
        
        # Análisis de ATR
        if BACKTEST_AVAILABLE and 'calcular_atr_amibroker' in globals():
            try:
                with st.spinner('Calculando ATR...'):
                    atr_value = calcular_atr_amibroker(data_with_indicators, atr_period)
                    st.subheader("🛡️ Volatilidad (ATR)")
                    st.metric("ATR Actual", f"{atr_value:.4f}", f"{(atr_value/data_with_indicators['Close'].iloc[-1]*100):.2f}% del precio")
                    
            except Exception as e:
                st.error(f"Error calculando ATR: {str(e)}")
        
        # Análisis de retornos mensuales
        if BACKTEST_AVAILABLE and 'calculate_monthly_returns_by_year' in globals():
            try:
                with st.spinner('Calculando retornos mensuales...'):
                    monthly_returns = calculate_monthly_returns_by_year(data_with_indicators)
                    if monthly_returns:
                        st.subheader("📅 Retornos Mensuales por Año")
                        
                        # Convertir a DataFrame para mejor visualización
                        returns_df = pd.DataFrame(monthly_returns).T
                        # Formatear los valores como porcentajes
                        styled_df = returns_df.style.format("{:.2f}%").background_gradient(cmap='RdYlGn', axis=None)
                        st.dataframe(styled_df, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error en análisis de retornos mensuales: {str(e)}")
    
    else:
        st.error(f"❌ No se pudieron cargar los datos para {symbol}. Por favor, verifica el símbolo del activo.")
        st.info("💡 Algunos símbolos comunes: AAPL (Apple), GOOGL (Google), MSFT (Microsoft), TSLA (Tesla)")

# Información adicional en el sidebar
with st.sidebar:
    st.divider()
    st.subheader("ℹ️ Información")
    st.markdown("""
    **Indicadores utilizados:**
    - Medias Móviles Simple (SMA)
    - Índice de Fuerza Relativa (RSI)
    - Average True Range (ATR)
    
    **Señales de Trading:**
    - Compra: SMA corta > SMA larga + RSI en rango normal + alta volatilidad
    - Venta: SMA corta < SMA larga o RSI sobrecomprado
    
    **Desarrollado por:** Equipo IA Mensual Ajustada
    """)
    
    # Mostrar estado del backtest
    if BACKTEST_AVAILABLE:
        st.success("✅ Módulo de backtest disponible")
    else:
        st.warning("⚠️ Módulo de backtest no disponible")

if __name__ == "__main__":
    main()
