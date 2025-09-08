import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from data_loader import download_prices, get_constituents_at_date
from backtest import run_backtest
from utils import unify_ticker, save_cache, load_cache
from datetime import datetime, timedelta

st.set_page_config(page_title="Inercia Alcista – S&P500 & Nasdaq-100", layout="wide")

st.markdown("# 📊 Inercia Alcista – Backtest Rotacional")
st.markdown("**Estrategia mensual** sobre los componentes del **S&P 500** y/o **Nasdaq-100**.")

# Sidebar
idx_choice = st.sidebar.multiselect(
    "Selecciona índice(s)",
    ["S&P 500", "Nasdaq-100"],
    default=["S&P 500"]
)

benchmark = st.sidebar.selectbox("Benchmark", ["SPY", "QQQ", "IWM"], index=0)

start = st.sidebar.date_input("Inicio", value=datetime(2015, 1, 1))
end   = st.sidebar.date_input("Fin", value=datetime.today())

if st.sidebar.button("▶️ Ejecutar backtest"):
    with st.spinner("Descargando precios y ejecutando backtest..."):
        # 1. Constituents históricos
        all_ticks = []
        if "S&P 500" in idx_choice:
            df_sp, chg_sp = get_constituents_at_date("SP500", start, end)
            all_ticks += df_sp["Symbol"].tolist()
        if "Nasdaq-100" in idx_choice:
            df_nq, chg_nq = get_constituents_at_date("NASDAQ100", start, end)
            all_ticks += df_nq["Symbol"].tolist()
        all_ticks = list(set([unify_ticker(t.replace("-", ".")) for t in all_ticks if t]))

        # 2. Descarga precios
        prices = download_prices(all_ticks, start, end)

        # 3. Descarga benchmark
        bench = download_prices([benchmark], start, end)[benchmark].dropna()

        # 4. Backtest
        bt, monthly_picks = run_backtest(prices, bench, comission=0.0030)

    # 5. Métricas
    cagr   = (bt["Equity"][-1] / bt["Equity"][0]) ** (252 / len(bt)) - 1
    vol    = bt["Returns"].std() * (252 ** 0.5)
    sharpe = cagr / vol if vol else 0
    maxdd  = (bt["Drawdown"].min())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("CAGR", f"{cagr:.2%}")
    col2.metric("Volatilidad", f"{vol:.2%}")
    col3.metric("Sharpe", f"{sharpe:.2f}")
    col4.metric("Max Drawdown", f"{maxdd:.2%}")
    col5.metric("Final Equity", f"${bt['Equity'][-1]:,.0f}")

    # 6. Gráficos
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=bt["Equity"], name="Estrategia"))
    fig.add_trace(go.Scatter(x=bench.index, y=(bench / bench[0]) * 10000, name=benchmark))
    fig.update_layout(title="Equity Curve", xaxis_title="Fecha", yaxis_title="USD", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    fig_dd = go.Figure(go.Scatter(x=bt.index, y=bt["Drawdown"], fill='tozeroy', name="Drawdown"))
    fig_dd.update_layout(title="Drawdown", template="plotly_white")
    st.plotly_chart(fig_dd, use_container_width=True)

    # 7. Tabla de selecciones mensuales
    st.markdown("### 📥 Selecciones mensuales (Top 10)")
    st.dataframe(monthly_picks.style.format({"Inercia": "{:.2f}"}))

    csv = monthly_picks.to_csv(index=False)
    st.download_button("Descargar CSV", csv, "selecciones_mensuales.csv", "text/csv")
