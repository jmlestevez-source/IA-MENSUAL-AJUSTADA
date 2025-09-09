
--------------------------------------------------
backtest.py  (completo)
------------------------------------------------```
import pandas as pd
import numpy as np

# ----------  True Range mensual (aprox) ----------
def monthly_true_range(close):
    prev = close.shift(1)
    high = low = close
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev),
                               np.abs(low - prev)))
    return tr

# ----------  Cálculo de Inercia (igual que AFL) ----------
def inertia_score(monthly_close, corte=680):
    """
    monthly_close: DataFrame (cols=tickers) con precios de cierre mensuales
    Devuelve DataFrame con InerciaAlcista, ATR14, Score, ScoreAdjusted
    """
    # ROC mensual
    roc1 = monthly_close.pct_change(10) * 0.4
    roc2 = monthly_close.pct_change(10) * 0.2
    f1 = roc1 + roc2

    # ATR(14) sobre barras mensuales
    tr = monthly_true_range(monthly_close)
    atr14 = tr.rolling(14).mean()

    # Denominador F2
    sma14 = monthly_close.rolling(14).mean()
    f2 = (atr14 / sma14) * 0.4

    inercia_alcista = f1 / f2

    # Corte 680
    score = np.where(inercia_alcista < corte, 0, np.maximum(inercia_alcista, 0))

    # Penalización por volatilidad
    score_adj = score / atr14

    return pd.DataFrame({
        "InerciaAlcista": inercia_alcista,
        "ATR14": atr14,
        "Score": score,
        "ScoreAdjusted": score_adj
    }).fillna(0)

# ----------  Backtest rotacional ----------
def run_backtest(prices, benchmark, comission=0.003, top_n=10, corte=680):
    # 1. Mensualizar
    prices_m = prices.resample('M').last()
    bench_m = benchmark.resample('M').last()

    equity = [10000]
    dates = [prices_m.index[0]]
    picks_list = []

    for i in range(1, len(prices_m)):
        prev_date = prices_m.index[i - 1]
        date = prices_m.index[i]

        # 2. Scores
        df_score = inertia_score(prices_m.loc[:prev_date], corte=corte)
        last_scores = df_score["ScoreAdjusted"].iloc[-1].sort_values(ascending=False).dropna()
        selected = last_scores.head(top_n).index.tolist()

        # 3. Retorno mensual
        weight = 1.0 / top_n
        rets = prices_m.loc[date, selected].pct_change().fillna(0)
        port_ret = (rets * weight).sum() - comission
        new_eq = equity[-1] * (1 + port_ret)

        equity.append(new_eq)
        dates.append(date)

        # 4. Guardar picks
        for rank, ticker in enumerate(selected, 1):
            picks_list.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Rank": rank,
                "Ticker": ticker,
                "Inercia": df_score["InerciaAlcista"].iloc[-1][ticker],
                "ScoreAdj": last_scores[ticker]
            })

    equity_series = pd.Series(equity, index=dates)
    returns = equity_series.pct_change().fillna(0)
    drawdown = equity_series / equity_series.cummax() - 1

    bt = pd.DataFrame({
        "Equity": equity_series,
        "Returns": returns,
        "Drawdown": drawdown
    })
    picks_df = pd.DataFrame(picks_list)
    return bt, picks_df
