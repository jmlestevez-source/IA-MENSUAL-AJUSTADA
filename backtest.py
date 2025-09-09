import pandas as pd
import numpy as np

# ---------- True Range mensual (aproximado) ----------
def monthly_true_range(close):
    prev = close.shift(1)
    high = low = close
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev),
                               np.abs(low - prev)))
    return tr

# ---------- Inercia Alcista (igual que AFL) ----------
def inertia_score(monthly_close, corte=680):
    # ROC
    roc1 = monthly_close.pct_change(10) * 0.4
    roc2 = monthly_close.pct_change(10) * 0.2
    f1 = roc1 + roc2
    
    # ATR(14) sobre barras mensuales
    tr = monthly_true_range(monthly_close)
    atr14 = tr.rolling(14).mean()

    # Denominador
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

# ---------- Backtest rotacional ----------
def run_backtest(prices, benchmark, commission=0.003, top_n=10, corte=680):
    try:
        # Mensualizar
        prices_m = prices.resample('ME').last()
        bench_m = benchmark.resample('ME').last()
        
        if prices_m.empty or bench_m.empty:
            raise ValueError("No hay datos mensuales suficientes para el backtest")
        
        equity = [10000]
        dates = [prices_m.index[0]]
        picks_list = []

        for i in range(1, len(prices_m)):
            prev_date = prices_m.index[i - 1]
            date = prices_m.index[i]

            # Scores
            try:
                df_score = inertia_score(prices_m.loc[:prev_date], corte=corte)
                if df_score.empty or len(df_score) < 14:  # Necesitamos al menos 14 períodos para ATR
                    continue
                    
                last_scores_series = df_score["ScoreAdjusted"].iloc[-1]
                if isinstance(last_scores_series, pd.Series):
                    last_scores = last_scores_series.sort_values(ascending=False).dropna()
                else:
                    continue
                    
                selected = last_scores.head(top_n).index.tolist()
                
                if not selected:
                    continue

                # Retorno mensual
                weight = 1.0 / len(selected)  # Usar len(selected) en caso de que sean menos de top_n
                available_prices = prices_m.loc[date, selected]
                
                if isinstance(available_prices, pd.Series):
                    rets = available_prices.pct_change().fillna(0)
                else:
                    rets = pd.Series([0] * len(selected), index=selected)
                
                port_ret = (rets * weight).sum() - commission
                new_eq = equity[-1] * (1 + port_ret)

                equity.append(new_eq)
                dates.append(date)

                # Guardar picks
                for rank, ticker in enumerate(selected, 1):
                    try:
                        inercia_val = df_score["InerciaAlcista"].iloc[-1]
                        if isinstance(inercia_val, pd.Series):
                            inercia_val = inercia_val[ticker]
                        
                        score_adj_val = last_scores[ticker]
                        
                        picks_list.append({
                            "Date": date.strftime("%Y-%m-%d"),
                            "Rank": rank,
                            "Ticker": ticker,
                            "Inercia": float(inercia_val) if not pd.isna(inercia_val) else 0,
                            "ScoreAdj": float(score_adj_val) if not pd.isna(score_adj_val) else 0
                        })
                    except Exception as e:
                        print(f"Error procesando pick {ticker}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error en iteración {i}: {e}")
                continue

        if len(equity) <= 1:
            raise ValueError("No se generaron resultados de backtest")

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
        
    except Exception as e:
        print(f"Error en run_backtest: {e}")
        # Retornar DataFrames vacíos en caso de error
        empty_bt = pd.DataFrame(columns=["Equity", "Returns", "Drawdown"])
        empty_picks = pd.DataFrame(columns=["Date", "Rank", "Ticker", "Inercia", "ScoreAdj"])
        return empty_bt, empty_picks
