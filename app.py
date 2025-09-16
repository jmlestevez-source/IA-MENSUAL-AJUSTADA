import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import ta

# Importaciones corregidas
try:
    from backtest import run_optimized_backtest, optimized_inertia_score, calcular_atr_amibroker, calculate_monthly_returns_by_year
except ImportError:
    # Si hay error de importación, definimos funciones dummy para evitar errores
    def run_optimized_backtest(*args, **kwargs):
        return {"error": "Función no disponible"}
    
    def optimized_inertia_score(*args, **kwargs):
        return 0
    
    def calcular_atr_amibroker(*args, **kwargs):
        return 0
    
    def calculate_monthly_returns_by_year(*args, **kwargs):
        return {}

# Configuración de la página
st.set_page_config(
    page_title="IA Mensual Ajustada",
    page_icon="📊",
    layout="wide"
)

# Título de la aplicación
st.title("🤖 IA Mensual Ajustada - Análisis Técnico Avanzado")

# Sidebar para configuración
st.sidebar.header("Configuración")

# Selección de activo
symbol = st.sidebar.text_input("Símbolo del activo", "AAPL").upper()

# Selección de período
period_options = {
    "1 año": "1y",
    "2 años": "2y",
    "5 años": "5y",
    "10 años": "10y",
    "Máximo": "max"
}
selected_period = st.sidebar.selectbox("Período de datos", list(period_options.keys()))
period = period_options[selected_period]

# Parámetros de estrategia
st.sidebar.subheader("Parámetros de Estrategia")
sma_short = st.sidebar.slider("Media Móvil Corta", 5, 50, 20)
sma_long = st.sidebar.slider("Media Móvil Larga", 20, 200, 50)
rsi_period = st.sidebar.slider("Período RSI", 5, 30, 14)
rsi_overbought = st.sidebar.slider("RSI Sobrecomprado", 50, 90, 70)
rsi_oversold = st.sidebar.slider("RSI Sobre vendido", 10, 50, 30)

# Descargar datos
@st.cache_data
def load_data(symbol, period):
    try:
        data = yf.download(symbol, period=period)
        if data.empty:
            st.error(f"No se encontraron datos para {symbol}")
            return None
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error al descargar datos: {str(e)}")
        return None

# Calcular indicadores técnicos
def calculate_indicators(df, sma_short, sma_long, rsi_period):
    df = df.copy()
    
    # Medias móviles
    df[f'SMA_{sma_short}'] = ta.trend.sma_indicator(df['Close'], window=sma_short)
    df[f'SMA_{sma_long}'] = ta.trend.sma_indicator(df['Close'], window=sma_long)
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=rsi_period)
    
    # Señales de compra/venta
    df['Signal'] = 0
    df.loc[(df[f'SMA_{sma_short}'] > df[f'SMA_{sma_long}']) & 
           (df['RSI'] < rsi_overbought) & 
           (df['RSI'] > rsi_oversold), 'Signal'] = 1
    
    # Señales de venta
    df.loc[(df[f'SMA_{sma_short}'] < df[f'SMA_{sma_long}']) | 
           (df['RSI'] > rsi_overbought), 'Signal'] = -1
    
    return df

# Crear gráfico
def create_chart(df, symbol):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'Precio de {symbol}', 'RSI', 'Señales')
    )
    
    # Precios y medias móviles
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Close'], name='Precio de Cierre', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Medias móviles
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df[f'SMA_{sma_short}'], name=f'SMA {sma_short}', line=dict(color='orange')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df[f'SMA_{sma_long}'], name=f'SMA {sma_long}', line=dict(color='red')),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    
    # Líneas de sobrecompra y sobreventa del RSI
    fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Señales de compra/venta
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    
    fig.add_trace(
        go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], 
                  mode='markers', name='Compra', 
                  marker=dict(color='green', size=10, symbol='triangle-up')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], 
                  mode='markers', name='Venta', 
                  marker=dict(color='red', size=10, symbol='triangle-down')),
        row=1, col=1
    )
    
    fig.update_layout(height=800, showlegend=True, title_text=f"Análisis Técnico de {symbol}")
    fig.update_xaxes(title_text="Fecha", row=3, col=1)
    fig.update_yaxes(title_text="Precio", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Señales", row=3, col=1)
    
    return fig

# Función principal
def main():
    # Cargar datos
    data = load_data(symbol, period)
    
    if data is not None and not data.empty:
        # Calcular indicadores
        data_with_indicators = calculate_indicators(data, sma_short, sma_long, rsi_period)
        
        # Mostrar gráfico
        st.plotly_chart(create_chart(data_with_indicators, symbol), use_container_width=True)
        
        # Mostrar datos recientes
        st.subheader("Datos Recientes")
        st.dataframe(data_with_indicators.tail(10))
        
        # Ejecutar backtest si las funciones están disponibles
        if 'run_optimized_backtest' in globals() and callable(run_optimized_backtest):
            try:
                with st.spinner('Ejecutando backtest...'):
                    backtest_results = run_optimized_backtest(data_with_indicators)
                    
                    if 'error' not in backtest_results:
                        st.subheader("Resultados del Backtest")
                        
                        # Métricas principales
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Rentabilidad Total", f"{backtest_results.get('total_return', 0):.2f}%")
                        col2.metric("Ratio Sharpe", f"{backtest_results.get('sharpe_ratio', 0):.2f}")
                        col3.metric("Máximo Drawdown", f"{backtest_results.get('max_drawdown', 0):.2f}%")
                        col4.metric("Número de Operaciones", backtest_results.get('num_trades', 0))
                        
                        # Gráfico de equity curve
                        if 'equity_curve' in backtest_results:
                            fig_equity = go.Figure()
                            fig_equity.add_trace(go.Scatter(
                                x=list(range(len(backtest_results['equity_curve']))),
                                y=backtest_results['equity_curve'],
                                name='Curva de Equity',
                                line=dict(color='blue')
                            ))
                            fig_equity.update_layout(title="Curva de Equity del Backtest")
                            st.plotly_chart(fig_equity, use_container_width=True)
                    else:
                        st.warning("Backtest no disponible en esta implementación")
            except Exception as e:
                st.error(f"Error en el backtest: {str(e)}")
        
        # Análisis de puntuación de inercia
        if 'optimized_inertia_score' in globals() and callable(optimized_inertia_score):
            try:
                inertia_score = optimized_inertia_score(data_with_indicators)
                st.subheader("Puntuación de Inercia")
                st.metric("Inercia del Mercado", f"{inertia_score:.2f}")
                
                # Interpretación
                if inertia_score > 0.7:
                    st.success("Alta inercia: Tendencia fuerte y sostenida")
                elif inertia_score > 0.4:
                    st.info("Inercia moderada: Tendencia estable")
                else:
                    st.warning("Baja inercia: Tendencia débil o sin dirección clara")
                    
            except Exception as e:
                st.error(f"Error calculando puntuación de inercia: {str(e)}")
        
        # Análisis de retornos mensuales
        if 'calculate_monthly_returns_by_year' in globals() and callable(calculate_monthly_returns_by_year):
            try:
                monthly_returns = calculate_monthly_returns_by_year(data_with_indicators)
                if monthly_returns:
                    st.subheader("Retornos Mensuales por Año")
                    
                    # Convertir a DataFrame para mejor visualización
                    returns_df = pd.DataFrame(monthly_returns).T
                    st.dataframe(returns_df.style.format("{:.2f}%"))
                    
            except Exception as e:
                st.error(f"Error en análisis de retornos mensuales: {str(e)}")
                
    else:
        st.error("No se pudieron cargar los datos. Por favor, verifica el símbolo del activo.")

if __name__ == "__main__":
    main()
