import pandas as pd
import numpy as np
from datetime import datetime

def inertia_score(prices, n=10, m=10):
    """
    Calcula Inercia Alcista ajustada por volatilidad (mensual)
    """
    roc1 = prices.pct_change(n) * 0.4
    roc2 = prices.pct_change(m) * 0.2
    f1   = roc1 + roc2

    # ATR sobre precios mensuales (aproximado)
    atr = (prices - prices.shift(1)).abs().rolling(14).mean()
    sma = prices.rolling(14).mean()
    f2  = (atr / sma) * 0.4

    score = (f1 / f2).fillna(0)
    score_adj = score / atr  # penaliza volatilidad
    return score_adj.fillna(0)

def run_backtest(prices, benchmark, comission=0.0030, top_n=10):
    """
    Backtest rotacional mensual
    """
    prices = prices.asfreq('M', method='pad')
    benchmark = benchmark.asfreq('M', method='pad')

    equity = [10000]
    dates  = [prices.index[0]]
    picks_list = []

    for i in range(1, len(prices)):
        prev_date = prices.index[i-1]
        date      = prices.index[i]

        # Score al cierre del mes previo
        score = inertia_score(prices.loc[:prev_date])
        last_scores = score.iloc[-1].sort_values(ascending=False).dropna()
        selected    = last_scores.head(top_n).index.tolist()

        # Ponderación igual
        weight = 1.0 / top_n
        # Retornos del mes
        rets = prices.loc[date, selected].pct_change().fillna(0)
        port_ret = (rets * weight).sum() - comission
        new_eq = equity[-1] * (1 + port_ret)

        equity.append(new_eq)
        dates.append(date)

        # Guardar selección
        for rank, ticker in enumerate(selected, 1):
            picks_list.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Rank": rank,
                "Ticker": ticker,
                "Inercia": last_scores[ticker]
            })

    equity_series = pd.Series(equity, index=dates)
    returns = equity_series.pct_change().fillna(0)
    drawdown = (equity_series / equity_series.cummax() - 1)

    bt = pd.DataFrame({
        "Equity": equity_series,
        "Returns": returns,
        "Drawdown": drawdown
    })

    picks_df = pd.DataFrame(picks_list)
    return bt, picks_df
