import sys
import traceback
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path

# Importar nuestros módulos
from data_loader import download_prices, get_constituents_at_date
from backtest import run_backtest
from utils import unify_ticker

# -------------------------------------------------
# 1. Config de página (SIEMPRE PRIMERO)
# -------------------------------------------------
st.set_page_config(page_title="Inercia Alcista – S&P500 & Nasdaq-100",
                   layout="wide")

# -------------------------------------------------
# 2. Catch-all para que **cualquier** error se muestre
# -------------------------------------------------
try:
    # -------------------------------------------------
    # 3. Título
    # -------------------------------------------------
    st.markdown("# 📊 Inercia Alcista – Backtest Rotacional")
    st.markdown("**Estrategia mensual** sobre los componentes del **S&P 500** y/o **Nasdaq-100**.")

    # -------------------------------------------------
    # 4. Sidebar
    # -------------------------------------------------
    idx_choice = st.sidebar.multiselect("Selecciona índice(s)",
                                        ["S&P 500", "Nasdaq-100"],
                                        default=["S&P 500"])

    benchmark = st.sidebar.selectbox("Benchmark", ["SPY", "QQQ", "IWM"], index=0)

    start = st.sidebar.date_input("Inicio", value=datetime(2015, 1, 1))
    end = st.sidebar.date_input("Fin", value=datetime.today())

    if st.sidebar.button("▶️ Ejecutar backtest"):
        if not idx_choice:
            st.warning("⚠️ Selecciona al menos un índice")
            st.stop()

        with st.spinner("Descargando precios y ejecutando backtest..."):
            # 1. Tickers históricos
            all_ticks = []
            if "S&P 500" in idx_choice:
                df_sp, _ = get_constituents_at_date("SP500", start, end)
                all_ticks += df_sp["Symbol"].tolist()
            if "Nasdaq-100" in idx_choice:
                df_nq, _ = get_constituents_at_date("NASDAQ100", start, end)
                all_ticks += df_nq["Symbol"].tolist()
            all_ticks = list({unify_ticker(t.replace("-", ".")) for t in all_ticks if t})

            # 2. Descargar precios
            prices = download_prices(all_ticks, start, end)
            bench = download_prices([benchmark], start, end)[benchmark].dropna()

            # 3. Backtest
            bt, monthly_picks = run_backtest(prices, bench, comission=0.0030, top_n=10)

        # -------------------------------------------------
        # 5. Métricas
        # -------------------------------------------------
        cagr = (bt["Equity"].iloc[-1] / bt["Equity"].iloc[0]) ** (252 / len(bt)) - 1
        vol = bt["Returns"].std() * np.sqrt(252)
        sharpe = cagr / vol if vol else 0
        maxdd = bt["Drawdown"].min()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("CAGR", f"{cagr:.2%}")
        col2.metric("Volatilidad", f"{vol:.2%}")
        col3.metric("Sharpe", f"{sharpe:.2f}")
        col4.metric("Max Drawdown", f"{maxdd:.2%}")
        col5.metric("Final Equity", f"${bt['Equity'].iloc[-1]:,.0f}")

        # -------------------------------------------------
        # 6. Gráficos
        # -------------------------------------------------
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Equity"], name="Estrategia"))
        fig.add_trace(go.Scatter(x=bench.index, y=(bench/bench.iloc[0])*bt["Equity"].iloc[0], name=benchmark))
        fig.update_layout(title="Equity Curve", xaxis_title="Fecha", yaxis_title="USD", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        fig_dd = go.Figure(go.Scatter(x=bt.index, y=bt["Drawdown"], fill='tozeroy', name="Drawdown"))
        fig_dd.update_layout(title="Drawdown", template="plotly_white")
        st.plotly_chart(fig_dd, use_container_width=True)

        # -------------------------------------------------
        # 7. Tabla de selecciones
        # -------------------------------------------------
        st.markdown("### 📥 Selecciones mensuales (Top 10)")
        st.dataframe(monthly_picks.style.format({"Inercia": "{:.2f}", "ScoreAdj": "{:.2f}"}))

        csv = monthly_picks.to_csv(index=False)
        st.download_button("Descargar CSV", csv, "selecciones_mensuales.csv", "text/csv")

# -------------------------------------------------
# Fin del try global → mostrar cualquier crash
# -------------------------------------------------
except Exception as e:
    st.error("❌ Excepción no capturada:")
    st.code("".join(traceback.format_exception(e, e, e.__traceback__)))
    with open("/tmp/crash.log", "w") as f:
        traceback.print_exc(file=f)
    st.stop()
