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

# Importar nuestros módulos - IMPORTANTE: importar inertia_score
from data_loader import get_constituents_at_date, get_sp500_historical_changes, get_nasdaq100_historical_changes, generate_removed_tickers_summary
from backtest import run_backtest_optimized, precalculate_all_indicators, calculate_monthly_returns_by_year, inertia_score, calculate_sharpe_ratio

# Al inicio del script, después de los imports
def check_historical_files():
    """Verifica la existencia de archivos de cambios históricos"""
    files_to_check = [
        "sp500_changes.csv",
        "ndx_changes.csv",
        "data/sp500_changes.csv",
        "data/ndx_changes.csv"
    ]
    
    found_files = []
    for file_path in files_to_check:
        if os.path.exists(file_path):
            found_files.append(file_path)
            try:
                df = pd.read_csv(file_path)
                print(f"✅ Encontrado: {file_path} ({len(df)} registros)")
            except Exception as e:
                print(f"⚠️ Error leyendo {file_path}: {e}")
    
    if not found_files:
        print("⚠️ No se encontraron archivos de cambios históricos")
        print("📁 Archivos esperados: sp500_changes.csv, ndx_changes.csv")
    
    return found_files

# Ejecutar verificación
historical_files = check_historical_files()

# -------------------------------------------------
# Configuración de la app
# -------------------------------------------------
st.set_page_config(
    page_title="IA Mensual Ajustada",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------
# FUNCIONES DE CACHÉ OPTIMIZADAS
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_historical_changes_cached(index_name, force_reload=False):
    """Carga cambios históricos con caché"""
    
    # Si force_reload, limpiar caché
    if force_reload:
        st.cache_data.clear()
    
    if index_name == "SP500":
        changes = get_sp500_historical_changes()
        if changes.empty:
            st.warning("⚠️ No se pudieron cargar cambios históricos del S&P 500")
        return changes
    elif index_name == "NDX":
        changes = get_nasdaq100_historical_changes()
        if changes.empty:
            st.warning("⚠️ No se pudieron cargar cambios históricos del NASDAQ-100")
        return changes
    else:  # Ambos
        sp500 = get_sp500_historical_changes()
        ndx = get_nasdaq100_historical_changes()
        
        if sp500.empty:
            st.warning("⚠️ No se pudieron cargar cambios del S&P 500")
        if ndx.empty:
            st.warning("⚠️ No se pudieron cargar cambios del NASDAQ-100")
            
        if not sp500.empty and not ndx.empty:
            return pd.concat([sp500, ndx], ignore_index=True)
        return sp500 if not sp500.empty else ndx

@st.cache_data(ttl=86400)
def get_constituents_cached(index_name, start_date, end_date):
    """Obtiene constituyentes con caché"""
    return get_constituents_at_date(index_name, start_date, end_date)

def get_cache_key(params):
    """Genera una clave única para caché basada en parámetros"""
    param_str = str(sorted(params.items()))
    return hashlib.md5(param_str.encode()).hexdigest()

@st.cache_data(ttl=3600*24*7)
def load_prices_from_csv_parallel(tickers, start_date, end_date, load_full_data=True):
    """Carga precios desde CSV en PARALELO"""
    prices_data = {}
    ohlc_data = {}
    
    def load_single_ticker(ticker):
        csv_path = f"data/{ticker}.csv"
        if not os.path.exists(csv_path):
            return ticker, None, None
        
        try:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            start_filter = start_date.date() if isinstance(start_date, datetime) else start_date
            end_filter = end_date.date() if isinstance(end_date, datetime) else end_date
            
            df = df[(df.index.date >= start_filter) & (df.index.date <= end_filter)]
            
            if df.empty:
                return ticker, None, None
            
            # Usar Close ya que los datos están ajustados
            if 'Close' in df.columns:
                price = df['Close']
            else:
                return ticker, None, None
            
            ohlc = None
            if load_full_data and all(col in df.columns for col in ['High', 'Low', 'Close']):
                ohlc = {
                    'High': df['High'],
                    'Low': df['Low'],
                    'Close': df['Close'],
                    'Volume': df['Volume'] if 'Volume' in df.columns else None
                }
            
            return ticker, price, ohlc
            
        except Exception as e:
            return ticker, None, None
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(load_single_ticker, ticker) for ticker in tickers]
        
        for future in futures:
            ticker, price, ohlc = future.result()
            if price is not None:
                prices_data[ticker] = price
            if ohlc is not None:
                ohlc_data[ticker] = ohlc
    
    if prices_data:
        prices_df = pd.DataFrame(prices_data)
        prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
        return prices_df, ohlc_data
    else:
        return pd.DataFrame(), {}

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

# FUNCIÓN PARA MOSTRAR TODOS LOS RESULTADOS (para no duplicar código)
def display_backtest_results(bt_results, picks_df, prices_df, benchmark_series, historical_info, index_choice, fixed_allocation, corte, top_n, ohlc_data):
    """Muestra todos los resultados del backtest"""
    
    st.success("✅ Backtest completado exitosamente")
    
    # Calcular métricas
    final_equity = float(bt_results["Equity"].iloc[-1])
    initial_equity = float(bt_results["Equity"].iloc[0])
    total_return = (final_equity / initial_equity) - 1
    years = (bt_results.index[-1] - bt_results.index[0]).days / 365.25
    cagr = (final_equity / initial_equity) ** (1/years) - 1 if years > 0 else 0
    max_drawdown = float(bt_results["Drawdown"].min())
    
    monthly_returns = bt_results["Returns"]
    risk_free_rate_monthly = 0.02 / 12
    excess_returns = monthly_returns - risk_free_rate_monthly
    sharpe_ratio = (excess_returns.mean() * 12) / (excess_returns.std() * np.sqrt(12)) if excess_returns.std() > 0 else 0
    volatility = float(monthly_returns.std() * np.sqrt(12))
    
    st.subheader("📊 Métricas de la Estrategia")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Equity Final", f"${final_equity:,.0f}")
    col2.metric("Retorno Total", f"{total_return:.2%}")
    col3.metric("CAGR", f"{cagr:.2%}")
    col4.metric("Max Drawdown", f"{max_drawdown:.2%}")
    col5.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    # Benchmark
    bench_equity = None
    bench_drawdown = None
    bench_sharpe = 0
    bench_final = initial_equity
    bench_total_return = 0
    bench_cagr = 0
    bench_max_dd = 0
    
    if benchmark_series is not None and not benchmark_series.empty:
        try:
            bench_returns = benchmark_series.pct_change().fillna(0)
            bench_equity = initial_equity * (1 + bench_returns).cumprod()
            bench_drawdown = (bench_equity / bench_equity.cummax() - 1)
            
            bench_final = float(bench_equity.iloc[-1])
            bench_initial = float(bench_equity.iloc[0]) if bench_equity.iloc[0] != 0 else initial_equity
            bench_total_return = (bench_final / bench_initial) - 1
            
            if years > 0:
                bench_cagr = (bench_final / bench_initial) ** (1/years) - 1
            
            bench_max_dd = float(bench_drawdown.min())
            
            if len(bench_returns) > len(bt_results) * 15:
                bench_returns_monthly = bench_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            else:
                bench_returns_monthly = bench_returns
            
            bench_excess_returns = bench_returns_monthly - risk_free_rate_monthly
            if bench_excess_returns.std() != 0:
                bench_sharpe = (bench_excess_returns.mean() * 12) / (bench_excess_returns.std() * np.sqrt(12))
                
        except Exception as e:
            st.warning(f"Error calculando benchmark: {e}")
    
    benchmark_name = "SPY" if index_choice != "NDX" else "QQQ"
    
    st.subheader(f"📊 Métricas del Benchmark ({benchmark_name})")
    col1b, col2b, col3b, col4b, col5b = st.columns(5)
    col1b.metric("Equity Final", f"${bench_final:,.0f}")
    col2b.metric("Retorno Total", f"{bench_total_return:.2%}")
    col3b.metric("CAGR", f"{bench_cagr:.2%}")
    col4b.metric("Max Drawdown", f"{bench_max_dd:.2%}")
    col5b.metric("Sharpe Ratio", f"{bench_sharpe:.2f}")
    
    # Comparación
    st.subheader("⚖️ Comparación Estrategia vs Benchmark")
    col1c, col2c, col3c, col4c = st.columns(4)
    
    alpha = cagr - bench_cagr
    col1c.metric("Alpha (CAGR diff)", f"{alpha:.2%}", delta=f"{alpha:.2%}")
    
    sharpe_diff = sharpe_ratio - bench_sharpe
    col2c.metric("Sharpe Diff", f"{sharpe_diff:.2f}", delta=f"{sharpe_diff:.2f}")
    
    dd_diff = max_drawdown - bench_max_dd
    col3c.metric("DD Difference", f"{dd_diff:.2%}", delta=f"{dd_diff:.2%}")
    
    return_diff = total_return - bench_total_return
    col4c.metric("Return Diff", f"{return_diff:.2%}", delta=f"{return_diff:.2%}")
    
    # Información sobre verificación histórica
    if historical_info and historical_info.get('has_historical_data', False):
        st.info("✅ Este backtest incluye verificación histórica de constituyentes")
    elif use_historical_verification:
        st.warning("⚠️ Verificación histórica solicitada pero no se encontraron datos históricos")
    else:
        st.warning("⚠️ Este backtest NO incluye verificación histórica (posible sesgo de supervivencia)")
    
    # Gráficos
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=bt_results.index,
        y=bt_results["Equity"],
        mode='lines',
        name='Estrategia',
        line=dict(width=3, color='blue'),
        hovertemplate='<b>%{y:,.0f}</b><br>%{x}<extra></extra>'
    ))
    
    if bench_equity is not None:
        common_index = bt_results.index.intersection(bench_equity.index)
        if len(common_index) > 0:
            bench_aligned = bench_equity.loc[common_index]
            
            fig_equity.add_trace(go.Scatter(
                x=bench_aligned.index,
                y=bench_aligned.values,
                mode='lines',
                name=f'Benchmark ({benchmark_name})',
                line=dict(width=2, dash='dash', color='gray'),
                hovertemplate='<b>%{y:,.0f}</b><br>%{x}<extra></extra>'
            ))
    
    fig_equity.update_layout(
        title="Evolución del Equity",
        xaxis_title="Fecha",
        yaxis_title="Equity ($)",
        hovermode='x unified',
        height=500,
        showlegend=True,
        yaxis_type="log"
    )
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Drawdown
    if "Drawdown" in bt_results.columns:
        fig_dd = go.Figure()
        
        fig_dd.add_trace(go.Scatter(
            x=bt_results.index,
            y=bt_results["Drawdown"] * 100,
            mode='lines',
            name='Drawdown Estrategia',
            fill='tozeroy',
            line=dict(color='red', width=2),
            hovertemplate='<b>%{y:.2f}%</b><br>%{x}<extra></extra>'
        ))
        
        if bench_drawdown is not None:
            common_index = bt_results.index.intersection(bench_drawdown.index)
            if len(common_index) > 0:
                bench_dd_aligned = bench_drawdown.loc[common_index]
                
                fig_dd.add_trace(go.Scatter(
                    x=bench_dd_aligned.index,
                    y=bench_dd_aligned.values * 100,
                    mode='lines',
                    name=f'Drawdown {benchmark_name}',
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

    # En app.py, donde obtienes los tickers (alrededor de línea 300-320)
tickers = list(dict.fromkeys(all_tickers_data['tickers']))
st.success(f"✅ Obtenidos {len(tickers)} tickers únicos")

# AÑADE ESTE CÓDIGO DE DEBUG MEJORADO:
if len(tickers) < 200:  # Umbral más razonable
    st.warning(f"⚠️ Solo {len(tickers)} tickers obtenidos. Esto puede ser muy poco.")
    
    # Mostrar algunos tickers para debug
    st.write("Muestra de tickers obtenidos:", tickers[:15])
    
    # Verificar si hay datos históricos
    if use_historical_verification:
        st.info("🔄 Probando sin verificación histórica...")
        from data_loader import get_current_constituents
        current_constituents = get_current_constituents(index_choice)
        if current_constituents and 'tickers' in current_constituents:
            st.write(f"Constituyentes actuales: {len(current_constituents['tickers'])}")
            if len(current_constituents['tickers']) > len(tickers):
                tickers = current_constituents['tickers']
                st.success(f"✅ Usando {len(tickers)} constituyentes actuales como fallback")
                st.write("Muestra de constituyentes actuales:", tickers[:15])

    
    # Tabla de rendimientos mensuales
    st.subheader("📅 RENDIMIENTOS MENSUALES POR AÑO")
    
    monthly_table = calculate_monthly_returns_by_year(bt_results["Equity"])
    
    if not monthly_table.empty:
        def style_returns(val):
            if val == "-" or val == "":
                return ""
            try:
                num = float(val.rstrip('%'))
                if num > 0:
                    return "background-color: #d4edda; color: #155724; font-weight: bold"
                elif num < 0:
                    return "background-color: #f8d7da; color: #721c24; font-weight: bold"
                else:
                    return ""
            except:
                return ""
        
        styled_table = monthly_table.style.applymap(style_returns)
        st.dataframe(styled_table, use_container_width=True)
        
        # Estadísticas
        total_years = len(monthly_table)
        if total_years > 0:
            positive_years = 0
            ytd_values = []
            
            for _, row in monthly_table.iterrows():
                if row['YTD'] != "-" and row['YTD'] != "":
                    try:
                        ytd_val = float(row['YTD'].rstrip('%'))
                        ytd_values.append(ytd_val)
                        if ytd_val > 0:
                            positive_years += 1
                    except:
                        continue
            
            if ytd_values:
                avg_annual_return = sum(ytd_values) / len(ytd_values)
                win_rate = (positive_years / len(ytd_values)) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Años Totales", total_years)
                col2.metric("Retorno Anual Promedio", f"{avg_annual_return:.1f}%")
                col3.metric("Tasa de Éxito Anual", f"{win_rate:.0f}%")
    
    # PICKS HISTÓRICOS - TODO EL BLOQUE
    if picks_df is not None and not picks_df.empty:
        st.subheader("📊 Picks Históricos")
        
        if 'HistoricallyValid' in picks_df.columns:
            total_picks = len(picks_df)
            valid_picks = picks_df['HistoricallyValid'].sum()
            validity_rate = valid_picks / total_picks * 100 if total_picks > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Picks", total_picks)
            col2.metric("Picks Válidos", valid_picks)
            col3.metric("% Validez Histórica", f"{validity_rate:.1f}%")
        
        # Inicializar session state si no existe
        if 'selected_date' not in st.session_state:
            st.session_state.selected_date = None
        
        # Crear columnas para la interfaz
        col_sidebar, col_main = st.columns([1, 3])
        
        with col_sidebar:
            st.markdown("### 📅 Navegación por Fechas")
            
            # Obtener fechas únicas ordenadas
            unique_dates = sorted(picks_df['Date'].unique(), reverse=True)
            
            # Selector de fecha
            selected_date = st.selectbox(
                "Selecciona una fecha:",
                unique_dates,
                index=0 if st.session_state.selected_date is None else 
                      (unique_dates.index(st.session_state.selected_date) if st.session_state.selected_date in unique_dates else 0),
                key="date_selector",
                help="Muestra los picks seleccionados en esta fecha"
            )
            
            # Guardar en session_state
            st.session_state.selected_date = selected_date
            
            # Mostrar información de la fecha seleccionada
            date_picks = picks_df[picks_df['Date'] == selected_date]
            st.info(f"🎯 {len(date_picks)} picks seleccionados el {selected_date}")
            
            # Calcular rentabilidad del mes
            try:
                bt_dates = bt_results.index.strftime('%Y-%m-%d').tolist()
                if selected_date in bt_dates:
                    date_idx = bt_dates.index(selected_date)
                    
                    if date_idx < len(bt_dates) - 1:
                        next_date = bt_dates[date_idx + 1]
                        current_equity = bt_results.iloc[date_idx]['Equity']
                        next_equity = bt_results.iloc[date_idx + 1]['Equity']
                        monthly_return = (next_equity / current_equity) - 1
                        
                        st.metric(
                            "📈 Retorno del Mes",
                            f"{monthly_return:.2%}",
                            delta=f"{monthly_return:.2%}"
                        )
                    else:
                        st.warning("📅 Último mes del backtest")
            except Exception as e:
                st.error(f"Error: {e}")
            
            # Estadísticas rápidas
            if not date_picks.empty:
                avg_inercia = date_picks['Inercia'].mean()
                avg_score = date_picks['ScoreAdj'].mean()
                
                st.markdown("### 📊 Estadísticas del Mes")
                st.metric("Inercia Promedio", f"{avg_inercia:.2f}")
                st.metric("Score Ajustado Promedio", f"{avg_score:.2f}")
        
        with col_main:
            st.markdown(f"### 🎯 Picks Seleccionados el {selected_date}")
            
            date_picks_display = date_picks.copy()
            
            # Mostrar tabla
            display_columns = ['Rank', 'Ticker', 'Inercia', 'ScoreAdj']
            if display_columns[0] in date_picks_display.columns:
                styled_df = date_picks_display[display_columns].style.format({
                    'Inercia': '{:.2f}',
                    'ScoreAdj': '{:.2f}'
                })
                st.dataframe(styled_df, use_container_width=True)

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
    default_end = min(datetime.today().date(), datetime(2030, 12, 31).date())
    default_start = default_end - timedelta(days=365*5)
    
    end_date = st.sidebar.date_input("Fecha final", value=default_end)
    start_date = st.sidebar.date_input("Fecha inicial", value=default_start)
    
    if start_date >= end_date:
        st.sidebar.warning("⚠️ Fecha inicial debe ser anterior a la fecha final")
        start_date = end_date - timedelta(days=365*2)
        
    st.sidebar.info(f"📅 Rango: {start_date} a {end_date}")
    
except Exception as e:
    st.sidebar.error(f"❌ Error configurando fechas: {e}")
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=365*5)

# Parámetros
top_n = st.sidebar.slider("Número de activos", 5, 30, 10)
commission = st.sidebar.number_input("Comisión por operación (%)", 0.0, 1.0, 0.3) / 100
corte = st.sidebar.number_input("Corte de score", 0, 1000, 680)
use_historical_verification = st.sidebar.checkbox("🕐 Usar verificación histórica", value=True)

st.sidebar.subheader("⚙️ Opciones de Estrategia")
fixed_allocation = st.sidebar.checkbox("💰 Asignar 10% capital a cada acción", value=False)

st.sidebar.subheader("🛡️ Filtros de Mercado")
use_roc_filter = st.sidebar.checkbox("📉 ROC 12 meses del SPY < 0", value=False)
use_sma_filter = st.sidebar.checkbox("📊 Precio SPY < SMA 10 meses", value=False)

run_button = st.sidebar.button("🏃 Ejecutar backtest", type="primary")

# -------------------------------------------------
# CONSTANTES
# -------------------------------------------------
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------------------------------
# Main content
# -------------------------------------------------
if run_button:
    try:
        cache_params = {
            'index': index_choice,
            'start': str(start_date),
            'end': str(end_date),
            'top_n': top_n,
            'corte': corte,
            'commission': commission,
            'historical': use_historical_verification,
            'fixed_alloc': fixed_allocation,
            'roc_filter': use_roc_filter,
            'sma_filter': use_sma_filter
        }
        cache_key = get_cache_key(cache_params)
        cache_file = os.path.join(CACHE_DIR, f"backtest_{cache_key}.pkl")
        
        use_cache = False
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if st.sidebar.checkbox("🔄 Usar resultados en caché", value=True):
                        use_cache = True
                        st.success("✅ Cargando resultados desde caché...")
                        
                        # Guardar en session_state cuando se carga desde caché
                        st.session_state.bt_results = cached_data['bt_results']
                        st.session_state.picks_df = cached_data['picks_df']
                        st.session_state.prices_df = cached_data.get('prices_df')
                        st.session_state.ohlc_data = cached_data.get('ohlc_data')
                        st.session_state.benchmark_series = cached_data.get('benchmark_series')
                        st.session_state.spy_df = cached_data.get('spy_df')
                        st.session_state.historical_info = cached_data.get('historical_info')
                        st.session_state.index_choice = index_choice
                        st.session_state.fixed_allocation = fixed_allocation
                        st.session_state.corte = corte
                        st.session_state.top_n = top_n
            except:
                use_cache = False
        
        if not use_cache:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Obtener constituyentes
            status_text.text("📥 Obteniendo constituyentes...")
            progress_bar.progress(10)
            
            all_tickers_data, error = get_constituents_cached(index_choice, start_date, end_date)
            if error:
                st.warning(f"Advertencia: {error}")
            
            if not all_tickers_data or 'tickers' not in all_tickers_data:
                st.error("No se encontraron tickers válidos")
                st.stop()
            
            tickers = list(dict.fromkeys(all_tickers_data['tickers']))
            st.success(f"✅ Obtenidos {len(tickers)} tickers únicos")
            
            # CORRECCIÓN CRÍTICA: Debug y fallback si son muy pocos tickers
            if len(tickers) < 100:
                st.warning(f"⚠️ Solo {len(tickers)} tickers obtenidos. Esto puede ser muy poco.")
                st.write("Primeros tickers:", tickers[:20])  # Mostrar primeros 20
                
                # Fallback automático si son muy pocos
                if use_historical_verification and len(tickers) < 50:
                    st.info("🔄 Probando sin verificación histórica...")
                    from data_loader import get_current_constituents
                    current_constituents = get_current_constituents(index_choice)
                    if current_constituents and len(current_constituents['tickers']) > len(tickers):
                        tickers = current_constituents['tickers']
                        st.success(f"✅ Usando {len(tickers)} constituyentes actuales como fallback")
            
            # Cargar precios
            status_text.text("📊 Cargando precios en paralelo...")
            progress_bar.progress(30)
            
            prices_df, ohlc_data = load_prices_from_csv_parallel(tickers, start_date, end_date, load_full_data=True)
            
            if prices_df.empty:
                st.error("❌ No se pudieron cargar precios")
                st.stop()
            
            st.success(f"✅ Cargados {len(prices_df.columns)} tickers con datos")
            
            # Cargar benchmark
            status_text.text("📈 Cargando benchmark...")
            progress_bar.progress(40)
            
            benchmark_ticker = "SPY" if index_choice != "NDX" else "QQQ"
            benchmark_df, _ = load_prices_from_csv_parallel([benchmark_ticker], start_date, end_date, load_full_data=False)
            
            if benchmark_df.empty:
                st.warning("Usando promedio como benchmark")
                benchmark_series = prices_df.mean(axis=1)
            else:
                benchmark_series = benchmark_df[benchmark_ticker]
            
            # SPY para filtros
            spy_df = None
            if use_roc_filter or use_sma_filter:
                spy_result, _ = load_prices_from_csv_parallel(["SPY"], start_date, end_date, load_full_data=False)
                spy_df = spy_result if not spy_result.empty else None
            
            # Información histórica
            historical_info = None
            if use_historical_verification:
                status_text.text("🕐 Cargando datos históricos...")
                progress_bar.progress(50)
                
                changes_data = load_historical_changes_cached(index_choice)
                
                if not changes_data.empty:
                    historical_info = {
                        'changes_data': changes_data, 
                        'has_historical_data': True
                    }
                    st.success(f"✅ Cargados {len(changes_data)} cambios históricos")
                else:
                    st.warning("⚠️ No se encontraron datos históricos")
                    historical_info = None
            
            # Ejecutar backtest
            status_text.text("🚀 Ejecutando backtest optimizado...")
            progress_bar.progress(70)
            
            bt_results, picks_df = run_backtest_optimized(
                prices=prices_df,
                benchmark=benchmark_series,
                commission=commission,
                top_n=top_n,
                corte=corte,
                ohlc_data=ohlc_data,
                historical_info=historical_info,
                fixed_allocation=fixed_allocation,
                use_roc_filter=use_roc_filter,
                use_sma_filter=use_sma_filter,
                spy_data=spy_df,
                progress_callback=lambda p: progress_bar.progress(70 + int(p * 0.3))
            )
            
            # Guardar resultados en session_state
            st.session_state.bt_results = bt_results
            st.session_state.picks_df = picks_df
            st.session_state.prices_df = prices_df
            st.session_state.ohlc_data = ohlc_data
            st.session_state.benchmark_series = benchmark_series
            st.session_state.spy_df = spy_df
            st.session_state.historical_info = historical_info
            st.session_state.index_choice = index_choice
            st.session_state.fixed_allocation = fixed_allocation
            st.session_state.corte = corte
            st.session_state.top_n = top_n
            
            # Guardar en caché
            status_text.text("💾 Guardando resultados en caché...")
            progress_bar.progress(100)
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'bt_results': bt_results,
                        'picks_df': picks_df,
                        'historical_info': historical_info,
                        'prices_df': prices_df,
                        'ohlc_data': ohlc_data,
                        'benchmark_series': benchmark_series,
                        'spy_df': spy_df,
                        'timestamp': datetime.now()
                    }, f)
                st.success("✅ Resultados guardados en caché")
            except Exception as e:
                st.warning(f"No se pudo guardar caché: {e}")
            
            status_text.empty()
            progress_bar.empty()
        
        # MOSTRAR RESULTADOS (desde session_state)
        if "bt_results" in st.session_state and st.session_state.bt_results is not None:
            display_backtest_results(
                st.session_state.bt_results,
                st.session_state.picks_df,
                st.session_state.prices_df,
                st.session_state.benchmark_series,
                st.session_state.historical_info,
                st.session_state.index_choice,
                st.session_state.fixed_allocation,
                st.session_state.corte,
                st.session_state.top_n,
                st.session_state.ohlc_data
            )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

# SI NO SE PULSA EL BOTÓN PERO HAY DATOS EN SESSION_STATE
elif "bt_results" in st.session_state and "picks_df" in st.session_state:
    st.info("📊 Mostrando resultados del último backtest. Puedes navegar sin perder datos.")
    
    # MOSTRAR LOS MISMOS RESULTADOS
    display_backtest_results(
        st.session_state.bt_results,
        st.session_state.picks_df,
        st.session_state.prices_df,
        st.session_state.benchmark_series,
        st.session_state.get('historical_info'),
        st.session_state.get('index_choice', 'SP500'),
        st.session_state.get('fixed_allocation', False),
        st.session_state.get('corte', 680),
        st.session_state.get('top_n', 10),
        st.session_state.get('ohlc_data')
    )

else:
    st.info("👈 Configura los parámetros y haz clic en 'Ejecutar backtest'")
    
    st.subheader("🔍 Información del Sistema")
    st.info("""
    **Características principales:**
    - ✅ Verificación histórica de constituyentes
    - ✅ Cálculos optimizados con precálculo y paralelización
    - ✅ Caché multinivel para carga instantánea
    - ✅ Comparación completa con benchmark
    - ✅ Tabla de rendimientos mensuales
    - ✅ Señales actuales con vela en formación
    - ✅ Filtros de mercado configurables
    
    **Mejoras de rendimiento:**
    - ⚡ ATR vectorizado con EWM
    - ⚡ Carga paralela de CSVs
    - ⚡ Precálculo de indicadores
    - ⚡ Caché persistente de resultados
    """)
    
    cache_files = glob.glob(os.path.join(CACHE_DIR, "backtest_*.pkl"))
    if cache_files:
        st.info(f"💾 {len(cache_files)} resultados en caché disponibles para carga instantánea")
