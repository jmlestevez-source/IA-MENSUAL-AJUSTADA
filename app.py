import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import numpy as np
import os

# Importar nuestros módulos
from data_loader import get_constituents_at_date
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
            st.warning(f"Archivo no encontrado: {csv_path}")
    
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
            
            # Verificar fechas de incorporación
            if all_tickers_data and 'data' in all_tickers_data:
                st.info(f"📅 Verificando fechas de incorporación...")
                # Mostrar algunos ejemplos de fechas
                sample_data = all_tickers_data['data'][:5] if all_tickers_data['data'] else []
                if sample_data:
                    for item in sample_data:
                        if isinstance(item, dict) and 'added' in item:
                            st.text(f"  {item.get('ticker', 'N/A')}: agregado el {item.get('added', 'N/A')}")
            
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
                st.stop()
            
            st.success(f"✅ Cargados precios para {len(prices_df.columns)} tickers")
            st.info(f"Rango de fechas: {prices_df.index.min().strftime('%Y-%m-%d')} a {prices_df.index.max().strftime('%Y-%m-%d')}")
            st.info(f"Muestra de tickers: {', '.join(list(prices_df.columns)[:5])}")

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
                
            # Ejecutar backtest con datos OHLC
            bt_results, picks_df = run_backtest(
                prices=prices_df,
                benchmark=benchmark_series,
                commission=commission,
                top_n=top_n,
                corte=corte,
                ohlc_data=ohlc_data  # Pasar los datos OHLC
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
                
            if "Returns" in bt_results.columns and len(bt_results["Returns"]) > 1:
                volatility = float(bt_results["Returns"].std() * (12 ** 0.5)) if bt_results["Returns"].std() != 0 else 0
                sharpe_ratio = (float(bt_results["Returns"].mean() * 12) / (volatility + 1e-8)) if volatility != 0 else 0
            else:
                volatility = 0
                sharpe_ratio = 0

            # -------------------------------------------------
            # Preparar datos del benchmark ANTES de mostrar métricas
            # -------------------------------------------------
            bench_equity = None
            bench_drawdown = None
            if benchmark_df is not None and not benchmark_df.empty:
                try:
                    bench_data = benchmark_df[benchmark_ticker] if benchmark_ticker in benchmark_df.columns else benchmark_df.iloc[:, 0]
                    bench_returns = bench_data.pct_change().fillna(0)
                    bench_equity = 10000 * (1 + bench_returns).cumprod()
                    bench_drawdown = (bench_equity / bench_equity.cummax() - 1)
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

            # Calcular y mostrar métricas del benchmark
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
                
                # Sharpe del benchmark
                bench_returns = bench_equity.pct_change().fillna(0)
                bench_volatility = float(bench_returns.std() * (12 ** 0.5)) if bench_returns.std() != 0 else 0
                bench_sharpe = (float(bench_returns.mean() * 12) / (bench_volatility + 1e-8)) if bench_volatility != 0 else 0
                
                # Métricas del benchmark
                st.subheader(f"📊 Métricas del Benchmark ({benchmark_ticker})")
                col1b, col2b, col3b, col4b, col5b = st.columns(5)
                col1b.metric("Equity Final", f"${bench_final:,.0f}")
                col2b.metric("Retorno Total", f"{bench_total_return:.2%}")
                col3b.metric("CAGR", f"{bench_cagr:.2%}")
                col4b.metric("Máximo Drawdown", f"{bench_max_dd:.2%}")
                col5b.metric("Sharpe Ratio", f"{bench_sharpe:.2f}")

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

                        # -------------------------------------------------
            # Señales Actuales (Vela en Formación)
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
                                # Top picks actuales
                                current_picks = []
                                for rank, (ticker, score_adj) in enumerate(last_scores.head(top_n).items(), 1):
                                    inercia_val = last_inercia.get(ticker, 0) if not last_inercia.empty else 0
                                    
                                    current_picks.append({
                                        'Rank': rank,
                                        'Ticker': ticker,
                                        'Inercia Alcista': inercia_val,
                                        'Score Ajustado': score_adj,
                                        'Pasa Corte': '✅' if inercia_val >= corte else '❌',
                                        'Precio Actual': prices_df[ticker].iloc[-1] if ticker in prices_df.columns else 0
                                    })
                                
                                current_picks_df = pd.DataFrame(current_picks)
                                
                                # Mostrar fecha de los datos
                                data_date = prices_df.index[-1].strftime('%Y-%m-%d')
                                st.info(f"📅 **Datos hasta**: {data_date} (vela en formación)")
                                
                                # Tabla de picks actuales
                                st.subheader(f"🔥 Top {top_n} Picks Actuales")
                                
                                # Formatear tabla para mostrar
                                display_df = current_picks_df.copy()
                                display_df['Precio Actual'] = display_df['Precio Actual'].apply(lambda x: f"${x:.2f}")
                                display_df['Inercia Alcista'] = display_df['Inercia Alcista'].apply(lambda x: f"{x:.2f}")
                                display_df['Score Ajustado'] = display_df['Score Ajustado'].apply(lambda x: f"{x:.2f}")
                                
                                st.dataframe(display_df, use_container_width=True)
                                
                                # Métricas actuales
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    current_pass_count = (current_picks_df['Inercia Alcista'] >= corte).sum()
                                    st.metric("Pasan Corte Actual", f"{current_pass_count}/{len(current_picks_df)}")
                                
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
                                        rotacion_pct = (len(nuevos) + len(salen)) / (2 * top_n) * 100
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
                                    
                                    # Línea de corte convertida a score ajustado (aproximada)
                                    # fig_comparison.add_hline(y=50, line_dash="dash", line_color="red", 
                                    #                         annotation_text="Referencia Score Adj")
                                    
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
                                st.info("""
                                **Para Trading Real:**
                                1. 📅 **Espera al cierre del mes actual** para señales definitivas
                                2. 🔄 **Recalcula el último día del mes** con datos completos
                                3. 📈 **Toma posiciones el primer día del próximo mes**
                                4. ⏰ **Mantén posiciones todo el mes** siguiente
                                5. 🔁 **Repite el proceso** mensualmente
                                
                                **Monitoreo:**
                                - Estas señales pueden cambiar diariamente
                                - Solo son indicativas de la tendencia actual
                                - Las señales finales se confirman al cierre mensual
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
            # Sección de Debug de Cálculos - MÉTODO AMIBROKER
            # -------------------------------------------------
            with st.expander("🔍 Debug de Cálculos de Inercia (Método AmiBroker)", expanded=False):
                if 'prices_df' in locals() and prices_df is not None and not prices_df.empty:
                    st.subheader("Análisis detallado de cálculos - Réplica exacta de AmiBroker")
                    
                    # Crear una copia de los tickers disponibles
                    available_tickers = sorted(list(prices_df.columns))
                    
                    # Usar session state para mantener el ticker seleccionado
                    if 'debug_ticker' not in st.session_state:
                        st.session_state.debug_ticker = available_tickers[0] if available_tickers else None
                    
                    debug_ticker = st.selectbox(
                        "Selecciona un ticker para analizar:",
                        available_tickers,
                        index=available_tickers.index(st.session_state.debug_ticker) if st.session_state.debug_ticker in available_tickers else 0,
                        key="debug_ticker_select"
                    )
                    
                    if st.button("Analizar Ticker", key="debug_analyze"):
                        st.session_state.debug_ticker = debug_ticker
                        
                        # Usar datos OHLC si están disponibles
                        if ohlc_data and debug_ticker in ohlc_data:
                            st.success("✅ Usando datos OHLC reales del CSV")
                            
                            # Convertir a mensual
                            high_daily = ohlc_data[debug_ticker]['High']
                            low_daily = ohlc_data[debug_ticker]['Low']
                            close_daily = ohlc_data[debug_ticker]['Close']
                            
                            # Crear DataFrame para resample
                            ohlc_df = pd.DataFrame({
                                'High': high_daily,
                                'Low': low_daily,
                                'Close': close_daily
                            })
                            
                            # Convertir a mensual EXACTO como en el código Python
                            monthly_ohlc = ohlc_df.resample('ME').agg({
                                'High': 'max',   # Máximo del mes
                                'Low': 'min',    # Mínimo del mes
                                'Close': 'last'  # Cierre del último día del mes
                            })
                            
                            high = monthly_ohlc['High']
                            low = monthly_ohlc['Low']
                            close = monthly_ohlc['Close']
                            
                        else:
                            st.warning("⚠️ No hay datos OHLC, usando estimación basada en Close")
                            # Fallback a estimación
                            ticker_data = prices_df[[debug_ticker]].dropna()
                            ticker_monthly = ticker_data.resample('ME').last()
                            close = ticker_monthly[debug_ticker]
                            
                            # Estimar High y Low
                            monthly_returns = close.pct_change()
                            monthly_vol = monthly_returns.rolling(3).std()
                            volatility_factor = monthly_vol.fillna(0.02)
                            high = close * (1 + volatility_factor)
                            low = close * (1 - volatility_factor)
                            high = pd.Series(np.maximum(high, close), index=close.index)
                            low = pd.Series(np.minimum(low, close), index=close.index)
                        
                        if len(close) >= 15:
                            # CÁLCULOS EXACTOS COMO EN EL CÓDIGO PYTHON QUE FUNCIONA
                            
                            # Calcular ROC de 10 meses (en porcentaje)
                            roc_10 = ((close - close.shift(10)) / close.shift(10)) * 100
                            
                            # F1 = ROC(10) * 0.6 (0.4 + 0.2)
                            f1 = roc_10 * 0.6
                            
                            # Calcular ATR(14) exactamente como AmiBroker
                            atr_14 = calcular_atr_amibroker(high, low, close, periods=14)
                            
                            # Calcular SMA(14)
                            sma_14 = close.rolling(14).mean()
                            
                            # F2 = (ATR14/SMA14) * 0.4
                            volatility_ratio = atr_14 / sma_14
                            f2 = volatility_ratio * 0.4
                            
                            # Inercia Alcista = F1 / F2
                            inercia_alcista = f1 / f2
                            
                            # Score = Inercia si >= corte, sino 0
                            score = np.where(inercia_alcista >= corte, inercia_alcista, 0)
                            score = pd.Series(score, index=inercia_alcista.index)
                            
                            # Score Adjusted = Score / ATR14
                            score_adjusted = score / atr_14
                            
                            # Limpiar valores infinitos y NaN
                            inercia_alcista = inercia_alcista.replace([np.inf, -np.inf], np.nan).fillna(0)
                            score = score.replace([np.inf, -np.inf], np.nan).fillna(0)
                            score_adjusted = score_adjusted.replace([np.inf, -np.inf], np.nan).fillna(0)
                            
                            # Mostrar últimos valores
                            st.subheader(f"📊 Últimos valores para {debug_ticker}")
                            
                            # Información del método
                            st.info("""
                            **Método AmiBroker:**
                            - Datos mensuales: High=max(mes), Low=min(mes), Close=último día
                            - ROC(10) en porcentaje = ((Close - Close_10) / Close_10) × 100
                            - F1 = ROC(10) × 0.6
                            - ATR(14) método Wilder: primer valor = media simple, luego ATR = ((ATR_prev × 13) + TR) / 14
                            - F2 = (ATR14/SMA14) × 0.4
                            - Inercia = F1/F2
                            - Score Adjusted = Score/ATR14
                            """)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("High del mes", f"${high.iloc[-1]:.2f}")
                                st.metric("Low del mes", f"${low.iloc[-1]:.2f}")
                                st.metric("Close del mes", f"${close.iloc[-1]:.2f}")
                                if len(close) > 10:
                                    st.metric("Close hace 10 meses", f"${close.iloc[-11]:.2f}")
                                st.metric("ROC(10)", f"{roc_10.iloc[-1]:.2f}%")
                                st.metric("F1 (ROC×0.6)", f"{f1.iloc[-1]:.2f}")
                            
                            with col2:
                                # Calcular True Range del último mes para mostrar
                                prev_close = close.shift(1)
                                hl = high - low
                                hc = np.abs(high - prev_close)
                                lc = np.abs(low - prev_close)
                                tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
                                
                                st.metric("True Range", f"${tr.iloc[-1]:.2f}")
                                st.metric("ATR(14) AmiBroker", f"${atr_14.iloc[-1]:.2f}")
                                st.metric("SMA(14)", f"${sma_14.iloc[-1]:.2f}")
                                st.metric("Ratio ATR/SMA", f"{volatility_ratio.iloc[-1]:.4f}")
                                st.metric("F2 (Ratio×0.4)", f"{f2.iloc[-1]:.4f}")
                            
                            with col3:
                                st.metric("Inercia Alcista", f"{inercia_alcista.iloc[-1]:.2f}")
                                st.metric("Score", f"{score.iloc[-1]:.2f}")
                                st.metric("Score Ajustado", f"{score_adjusted.iloc[-1]:.2f}")
                            
                            # Mostrar cálculo detallado
                            idx = -1
                            st.subheader("📝 Verificación paso a paso")
                            st.code(f"""
CÁLCULOS PASO A PASO PARA {debug_ticker} (Método AmiBroker):

1. Datos del mes:
   High: ${high.iloc[idx]:.2f}, Low: ${low.iloc[idx]:.2f}, Close: ${close.iloc[idx]:.2f}

2. ROC(10) = ((Close - Close_10) / Close_10) × 100
   ROC(10) = (({close.iloc[idx]:.2f} - {close.iloc[idx-10]:.2f}) / {close.iloc[idx-10]:.2f}) × 100 = {roc_10.iloc[idx]:.2f}%

3. F1 = ROC(10) × 0.6 = {roc_10.iloc[idx]:.2f} × 0.6 = {f1.iloc[idx]:.2f}

4. True Range = max(H-L, |H-Cprev|, |L-Cprev|) = {tr.iloc[idx]:.2f}

5. ATR(14) método Wilder = {atr_14.iloc[idx]:.2f}

6. SMA(14) = {sma_14.iloc[idx]:.2f}

7. F2 = (ATR14/SMA14) × 0.4 = ({atr_14.iloc[idx]:.2f}/{sma_14.iloc[idx]:.2f}) × 0.4 = {f2.iloc[idx]:.4f}

8. Inercia Alcista = F1/F2 = {f1.iloc[idx]:.2f}/{f2.iloc[idx]:.4f} = {inercia_alcista.iloc[idx]:.2f}

9. Score = {score.iloc[idx]:.2f} (Inercia si >= {corte}, sino 0)

10. Score Ajustado = Score/ATR14 = {score.iloc[idx]:.2f}/{atr_14.iloc[idx]:.2f} = {score_adjusted.iloc[idx]:.2f}
                            """)
                            
                            # Mostrar si pasa el corte
                            if inercia_alcista.iloc[idx] >= corte:
                                st.success(f"✅ Inercia ({inercia_alcista.iloc[idx]:.2f}) >= {corte} - PASA EL CORTE")
                            else:
                                st.warning(f"❌ Inercia ({inercia_alcista.iloc[idx]:.2f}) < {corte} - NO PASA EL CORTE")
                                
                        else:
                            st.error(f"No hay suficientes datos para {debug_ticker} (se necesitan al menos 15 meses)")
                else:
                    st.info("Ejecuta primero el backtest para poder analizar los cálculos")

            # -------------------------------------------------
            # Comparación con últimos picks
            # -------------------------------------------------
            if 'picks_df' in locals() and picks_df is not None and not picks_df.empty:
                with st.expander("📊 Análisis de Consistencia de Picks"):
                    st.subheader("Comparación de valores calculados")
                    
                    # Obtener últimos picks
                    latest_date = picks_df["Date"].max()
                    latest_picks = picks_df[picks_df["Date"] == latest_date].head(10)
                    
                    if not latest_picks.empty:
                        # Crear tabla de análisis
                        analysis_data = []
                        for _, pick in latest_picks.iterrows():
                            analysis_data.append({
                                'Rank': pick['Rank'],
                                'Ticker': pick['Ticker'],
                                'Inercia Alcista': f"{pick['Inercia']:.2f}",
                                'Score Ajustado': f"{pick['ScoreAdj']:.2f}",
                                'Pasa Corte': '✅' if pick['Inercia'] >= corte else '❌'
                            })
                        
                        analysis_df = pd.DataFrame(analysis_data)
                        st.dataframe(analysis_df, use_container_width=True)
                        
                        # Métricas de resumen
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_inercia = latest_picks['Inercia'].mean()
                            st.metric("Promedio Inercia", f"{avg_inercia:.2f}")
                        with col2:
                            avg_score = latest_picks['ScoreAdj'].mean()
                            st.metric("Promedio Score Adj", f"{avg_score:.2f}")
                        with col3:
                            pass_count = (latest_picks['Inercia'] >= corte).sum()
                            st.metric("Tickers que pasan corte", f"{pass_count}/{len(latest_picks)}")
                        
                        # Advertencia si hay inconsistencias
                        if avg_score > 350:
                            st.warning("⚠️ Los valores de Score Ajustado están altos. En AmiBroker suelen estar por debajo de 350.")
                        else:
                            st.success("✅ Los valores de Score Ajustado están en el rango esperado de AmiBroker.")

    except Exception as e:
        st.error(f"❌ Excepción no capturada: {str(e)}")
        st.exception(e)
        st.info("💡 Consejos para resolver este problema:")
        st.info("1. Verifica que los archivos CSV existan en la carpeta 'data/'")
        st.info("2. Asegúrate de que los archivos tengan el formato correcto")
        st.info("3. Prueba con un rango de fechas más corto")
        st.info("4. Verifica que los tickers sean válidos")

else:
    st.info("👈 Configura los parámetros en el panel lateral y haz clic en 'Ejecutar backtest'")
    st.info("💡 Consejos para mejores resultados:")
    st.info("• Usa un rango de fechas de al menos 2 años")
    st.info("• Comienza con 10 activos y ajusta según los resultados")
    st.info("• Considera usar ambos índices para mayor diversificación")
