import pandas as pd
import numpy as np

def monthly_true_range(high, low, close):
    """
    Calcula el True Range mensual correctamente
    """
    prev_close = close.shift(1)
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': np.abs(high - prev_close),
        'lc': np.abs(low - prev_close)
    }).max(axis=1)
    return tr

def inertia_score(monthly_prices_df, corte=680):
    """
    Calcula el score de inercia correctamente
    monthly_prices_df debe contener columnas: High, Low, Close
    """
    if monthly_prices_df.empty:
        return pd.DataFrame()

    # Asegurarse de que tenemos las columnas necesarias
    required_cols = ['High', 'Low', 'Close']
    if not all(col in monthly_prices_df.columns for col in required_cols):
        # Si solo tenemos Close, usar Close para todas
        close = monthly_prices_df['Close'] if 'Close' in monthly_prices_df.columns else monthly_prices_df.iloc[:, 0]
        high = close
        low = close
        close = close
    else:
        high = monthly_prices_df['High']
        low = monthly_prices_df['Low']
        close = monthly_prices_df['Close']

    # Calcular ROC (10 períodos)
    roc_10 = close.pct_change(10, fill_method=None)
    f1 = roc_10 * 0.6  # Combinación de los dos ROC

    # Calcular ATR(14)
    tr = monthly_true_range(high, low, close)
    atr14 = tr.rolling(14).mean()

    # Calcular SMA(14)
    sma14 = close.rolling(14).mean()

    # Evitar división por cero - CORREGIDO Y REFORZADO
    try:
        # Paso 1: Calcular la razón ATR/SMA con protección
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_raw = atr14 / sma14
            
        # Paso 2: Reemplazar valores no finitos (inf, -inf, nan) con NaN
        ratio_clean = ratio_raw.replace([np.inf, -np.inf], np.nan)
        
        # Paso 3: Reemplazar ceros con NaN
        ratio_no_zero = ratio_clean.replace(0, np.nan)
        
        # Paso 4: Forward fill para propagar el último valor válido
        ratio_ffilled = ratio_no_zero.ffill()
        
        # Paso 5: Rellenar cualquier NaN restante al inicio con un valor pequeño
        ratio_filled = ratio_ffilled.fillna(1e-10)
        
        # Paso 6: Aplicar el factor de escala
        denominator = ratio_filled * 0.4
        
        # Paso 7: Asegurar que el denominador no sea cero para la división final
        denominator_safe = denominator.replace(0, 1e-10)
        
        # Paso 8: Calcular inercia con protección adicional
        with np.errstate(divide='ignore', invalid='ignore'):
            inercia_alcista = f1 / denominator_safe
            
        # Reemplazar cualquier valor no finito resultante
        inercia_alcista = inercia_alcista.replace([np.inf, -np.inf], np.nan).fillna(0)
        
    except Exception as e:
        print(f"Error en cálculo de inercia: {e}")
        # Fallback seguro
        denominator_safe = pd.Series(1e-10, index=f1.index)
        inercia_alcista = f1 / denominator_safe

    # Aplicar corte
    score = np.where(inercia_alcista < corte, 0, np.maximum(inercia_alcista, 0))

    # Penalización por volatilidad
    atr14_safe = atr14.replace(0, 1e-10).replace([np.inf, -np.inf], 1e-10)
    with np.errstate(divide='ignore', invalid='ignore'):
        score_adj = score / atr14_safe
    score_adj = score_adj.replace([np.inf, -np.inf], np.nan).fillna(0)

    return pd.DataFrame({
        "InerciaAlcista": inercia_alcista,
        "ATR14": atr14,
        "Score": score,
        "ScoreAdjusted": score_adj
    }).fillna(0)

def run_backtest(prices, benchmark, commission=0.003, top_n=10, corte=680):
    try:
        # Mensualizar precios
        if isinstance(prices, pd.Series):
            prices_m = prices.resample('ME').last()
            prices_df_m = pd.DataFrame({'Close': prices_m})
        else:
            prices_m = prices.resample('ME').last()
            prices_df_m = prices_m.copy()

            # Asegurar que tenemos las columnas necesarias
            if len(prices_df_m.columns) == 1:
                col_name = prices_df_m.columns[0]
                prices_df_m = pd.DataFrame({
                    'High': prices_df_m[col_name],
                    'Low': prices_df_m[col_name],
                    'Close': prices_df_m[col_name]
                })
            elif 'Close' not in prices_df_m.columns and len(prices_df_m.columns) > 0:
                # Usar la primera columna como Close
                first_col = prices_df_m.columns[0]
                prices_df_m['Close'] = prices_df_m[first_col]
                if 'High' not in prices_df_m.columns:
                    prices_df_m['High'] = prices_df_m[first_col]
                if 'Low' not in prices_df_m.columns:
                    prices_df_m['Low'] = prices_df_m[first_col]

        # Mensualizar benchmark
        if isinstance(benchmark, pd.Series):
            bench_m = benchmark.resample('ME').last()
        else:
            bench_m = benchmark.resample('ME').last()

        if prices_df_m.empty or bench_m.empty:
            raise ValueError("No hay datos mensuales suficientes para el backtest")

        equity = [10000]
        dates = [prices_df_m.index[0]]
        picks_list = []

        # Crear DataFrame con High, Low, Close para el cálculo de inercia
        if 'Close' not in prices_df_m.columns:
            prices_df_m['Close'] = prices_df_m.iloc[:, 0]
        if 'High' not in prices_df_m.columns:
            prices_df_m['High'] = prices_df_m['Close']
        if 'Low' not in prices_df_m.columns:
            prices_df_m['Low'] = prices_df_m['Close']

        for i in range(1, len(prices_df_m)):
            prev_date = prices_df_m.index[i - 1]
            date = prices_df_m.index[i]

            # Calcular scores usando datos históricos hasta la fecha anterior
            try:
                historical_data = prices_df_m.loc[:prev_date].copy()
                if len(historical_data) < 15:  # Necesitamos suficientes datos
                    continue

                df_score = inertia_score(historical_data, corte=corte)
                if df_score.empty or len(df_score) < 14:
                    continue

                # Obtener el último score ajustado para cada ticker
                if isinstance(df_score["ScoreAdjusted"], pd.DataFrame):
                    # Caso múltiples tickers
                    last_scores = df_score["ScoreAdjusted"].iloc[-1]
                else:
                    # Caso un solo ticker
                    last_scores = pd.Series([df_score["ScoreAdjusted"].iloc[-1]],
                                          index=[prices_df_m.columns[0]])

                last_scores = last_scores.dropna().sort_values(ascending=False)

                if len(last_scores) == 0:
                    continue

                selected = last_scores.head(top_n).index.tolist()

                if not selected:
                    continue

                # Calcular retorno mensual
                weight = 1.0 / len(selected)

                # Obtener precios disponibles para los tickers seleccionados
                available_prices = prices_m.loc[date]
                prev_prices = prices_m.loc[prev_date]

                # Filtrar solo tickers seleccionados que tienen datos
                valid_tickers = [t for t in selected if t in available_prices.index and t in prev_prices.index]

                if len(valid_tickers) == 0:
                    continue

                # Calcular retornos
                rets = pd.Series(index=valid_tickers)
                for ticker in valid_tickers:
                    if prev_prices[ticker] != 0:
                        rets[ticker] = (available_prices[ticker] / prev_prices[ticker]) - 1
                    else:
                        rets[ticker] = 0

                rets = rets.fillna(0)
                port_ret = (rets * weight).sum() - commission
                new_eq = equity[-1] * (1 + port_ret)

                equity.append(new_eq)
                dates.append(date)

                # Guardar picks
                for rank, ticker in enumerate(valid_tickers[:top_n], 1):
                    try:
                        inercia_val = df_score["InerciaAlcista"].iloc[-1]
                        if isinstance(inercia_val, pd.Series):
                            inercia_val = inercia_val.get(ticker, 0)

                        score_adj_val = last_scores.get(ticker, 0)

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
        returns = equity_series.pct_change(fill_method=None).fillna(0)
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
        empty_bt = pd.DataFrame(columns=["Equity", "Returns", "Drawdown"])
        empty_picks = pd.DataFrame(columns=["Date", "Rank", "Ticker", "Inercia", "ScoreAdj"])
        return empty_bt, empty_picks
