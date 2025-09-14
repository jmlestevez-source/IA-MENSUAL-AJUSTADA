import pandas as pd
import numpy as np

def calcular_atr_wilder(high, low, close, periods=14):
    """
    Calcula el ATR exactamente como AmiBroker usando el método de Wilder
    """
    prev_close = close.shift(1)
    
    # True Range: máximo de tres valores
    hl = high - low  # High - Low
    hc = np.abs(high - prev_close)  # |High - PrevClose|
    lc = np.abs(low - prev_close)  # |Low - PrevClose|
    
    # True Range es el máximo de los tres
    tr = pd.DataFrame({'hl': hl, 'hc': hc, 'lc': lc}).max(axis=1)
    
    # ATR usando el método de Wilder
    # Primer valor: media simple de los primeros 'periods' valores
    atr = pd.Series(index=tr.index, dtype=float)
    
    # Calcular la media simple inicial
    if len(tr) >= periods:
        atr.iloc[periods-1] = tr.iloc[:periods].mean()
        
        # Aplicar la fórmula de Wilder para los siguientes valores
        for i in range(periods, len(tr)):
            atr.iloc[i] = (atr.iloc[i-1] * (periods - 1) + tr.iloc[i]) / periods
    
    return atr

def inertia_score(monthly_prices_df, corte=680):
    """
    Calcula el score de inercia correctamente replicando el código AFL
    con el método exacto de AmiBroker para el ATR
    """
    if monthly_prices_df is None or monthly_prices_df.empty:
        return pd.DataFrame()

    try:
        results = {}
        
        for ticker in monthly_prices_df.columns:
            try:
                # Obtener serie de precios para este ticker
                close = monthly_prices_df[ticker].dropna()
                
                if len(close) < 15:
                    continue
                
                # IMPORTANTE: Para datos de CSV que ya están mensualizados,
                # necesitamos estimar High y Low basándonos en la volatilidad histórica
                # O usar Close para los tres si no tenemos datos intradiarios
                
                # Opción 1: Si solo tenemos Close (datos ya mensualizados de CSV)
                # Estimamos High y Low basándonos en la volatilidad típica
                # Esta es una aproximación, pero necesaria sin datos intradiarios
                
                # Calcular volatilidad mensual promedio
                monthly_returns = close.pct_change()
                monthly_vol = monthly_returns.rolling(3).std()
                
                # Estimar High y Low (aproximación)
                # High = Close * (1 + volatilidad/2)
                # Low = Close * (1 - volatilidad/2)
                high = close * (1 + monthly_vol.fillna(0.02))
                low = close * (1 - monthly_vol.fillna(0.02))
                
                # Asegurar que High >= Close >= Low
                high = pd.Series(np.maximum(high, close), index=close.index)
                low = pd.Series(np.minimum(low, close), index=close.index)
                
                # Calcular ROC como porcentaje (AmiBroker style)
                roc_10_percent = ((close - close.shift(10)) / close.shift(10)) * 100
                
                # Aplicar los pesos según el código AFL
                roc_10_w1 = roc_10_percent * 0.4
                roc_10_w2 = roc_10_percent * 0.2
                f1 = roc_10_w1 + roc_10_w2  # Total: ROC * 0.6
                
                # Calcular ATR(14) con el método de Wilder
                atr14 = calcular_atr_wilder(high, low, close, periods=14)
                
                # Calcular SMA(14)
                sma14 = close.rolling(14).mean()
                
                # Calcular F2 = (ATR14/MA(C,14))*0.4
                with np.errstate(divide='ignore', invalid='ignore'):
                    volatility_ratio = atr14 / sma14
                    f2 = volatility_ratio * 0.4
                    
                    # Manejar valores problemáticos
                    f2 = f2.replace([np.inf, -np.inf], np.nan)
                    f2 = f2.fillna(0.01)
                    f2 = f2.apply(lambda x: max(x, 0.0001) if not pd.isna(x) else 0.0001)
                
                # Calcular Inercia Alcista = F1 / F2
                with np.errstate(divide='ignore', invalid='ignore'):
                    inercia_alcista = f1 / f2
                    inercia_alcista = inercia_alcista.replace([np.inf, -np.inf], 0)
                    inercia_alcista = inercia_alcista.fillna(0)
                
                # Aplicar corte
                score = pd.Series(
                    np.where(inercia_alcista < corte, 0, np.maximum(inercia_alcista, 0)),
                    index=inercia_alcista.index
                )
                
                # Score Ajustado = Score / ATR14
                atr14_safe = atr14.copy()
                atr14_safe = atr14_safe.fillna(1.0)
                atr14_safe = atr14_safe.apply(lambda x: max(x, 0.01) if not pd.isna(x) else 0.01)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    score_adj = score / atr14_safe
                    score_adj = score_adj.replace([np.inf, -np.inf], 0)
                    score_adj = score_adj.fillna(0)
                
                # Almacenar resultados
                results[ticker] = {
                    "InerciaAlcista": inercia_alcista,
                    "ATR14": atr14,
                    "Score": score,
                    "ScoreAdjusted": score_adj,
                    "F1": f1,
                    "F2": f2,
                    "ROC10": roc_10_percent,
                    "VolatilityRatio": volatility_ratio
                }
                
            except Exception as e:
                print(f"Error procesando ticker {ticker}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        # Combinar resultados
        combined_results = {}
        for metric in ["InerciaAlcista", "ATR14", "Score", "ScoreAdjusted", "F1", "F2", "ROC10", "VolatilityRatio"]:
            metric_data = {}
            for ticker in results.keys():
                if metric in results[ticker]:
                    metric_data[ticker] = results[ticker][metric]
            if metric_data:
                combined_results[metric] = pd.DataFrame(metric_data)
        
        return combined_results
        
    except Exception as e:
        print(f"Error en cálculo de inercia: {e}")
        return {}

# El resto de las funciones (run_backtest, etc.) permanecen igual

def debug_inertia_calculation(ticker_data, ticker_name, last_date=None):
    """
    Función mejorada para debuggear los cálculos paso a paso
    """
    close = ticker_data
    if last_date:
        close = close[:last_date]
    
    if len(close) < 15:
        print(f"No hay suficientes datos para {ticker_name}")
        return None
    
    # Obtener los últimos valores
    last_close = close.iloc[-1]
    close_10_ago = close.iloc[-11] if len(close) > 10 else np.nan
    
    # Calcular ROC en porcentaje (como AmiBroker)
    roc_10_percent = ((last_close - close_10_ago) / close_10_ago) * 100 if close_10_ago != 0 else 0
    
    # Componentes ponderados
    roc_10_w1 = roc_10_percent * 0.4
    roc_10_w2 = roc_10_percent * 0.2
    f1 = roc_10_w1 + roc_10_w2
    
    # ATR y SMA
    tr = monthly_true_range(close, close, close)
    atr14 = tr.rolling(14).mean().iloc[-1]
    sma14 = close.rolling(14).mean().iloc[-1]
    
    # F2
    volatility_ratio = atr14 / sma14 if sma14 != 0 else 0
    f2 = volatility_ratio * 0.4
    f2 = max(f2, 0.01)  # Evitar división por cero
    
    # Inercia
    inercia = f1 / f2
    
    # Score con corte
    score = max(inercia, 0) if inercia >= 680 else 0
    
    # Score ajustado
    atr14_safe = max(atr14, 0.01)
    score_adj = score / atr14_safe
    
    print(f"\n=== Debug para {ticker_name} ===")
    print(f"Precio actual: ${last_close:.2f}")
    print(f"Precio hace 10 meses: ${close_10_ago:.2f}")
    print(f"ROC(10): {roc_10_percent:.2f}%")
    print(f"ROC10 * 0.4: {roc_10_w1:.2f}")
    print(f"ROC10 * 0.2: {roc_10_w2:.2f}")
    print(f"F1 (suma): {f1:.2f}")
    print(f"ATR(14): ${atr14:.2f}")
    print(f"SMA(14): ${sma14:.2f}")
    print(f"Ratio volatilidad (ATR/SMA): {volatility_ratio:.4f}")
    print(f"F2 (ratio * 0.4): {f2:.4f}")
    print(f"Inercia Alcista (F1/F2): {inercia:.2f}")
    print(f"Score (con corte 680): {score:.2f}")
    print(f"Score Ajustado (Score/ATR): {score_adj:.2f}")
    
    return {
        'ROC10%': roc_10_percent,
        'F1': f1,
        'F2': f2,
        'ATR14': atr14,
        'InerciaAlcista': inercia,
        'Score': score,
        'ScoreAdjusted': score_adj
    }
        
def run_backtest(prices, benchmark, commission=0.003, top_n=10, corte=680):
    try:
        print("Iniciando backtest...")
        print(f"Forma de prices: {prices.shape if hasattr(prices, 'shape') else 'No shape'}")
        print(f"Tipo de prices: {type(prices)}")
        
        # Validar entrada
        if prices is None or (hasattr(prices, 'empty') and prices.empty):
            print("❌ Datos de precios vacíos")
            empty_bt = pd.DataFrame(columns=["Equity", "Returns", "Drawdown"])
            empty_picks = pd.DataFrame(columns=["Date", "Rank", "Ticker", "Inercia", "ScoreAdj"])
            return empty_bt, empty_picks

        # Mensualizar precios
        if isinstance(prices, pd.Series):
            prices_m = prices.resample('ME').last()
            prices_df_m = pd.DataFrame({'Close': prices_m})
        else:
            try:
                prices_m = prices.resample('ME').last()
                prices_df_m = prices_m.copy()
            except Exception as e:
                print(f"Error mensualizando precios: {e}")
                prices_m = prices
                prices_df_m = prices.copy()

        # Mensualizar benchmark
        try:
            if isinstance(benchmark, pd.Series):
                bench_m = benchmark.resample('ME').last()
            else:
                bench_m = benchmark.resample('ME').last()
        except:
            bench_m = benchmark

        if prices_df_m.empty or (hasattr(bench_m, 'empty') and bench_m.empty):
            print("❌ No hay datos mensuales suficientes")
            raise ValueError("No hay datos mensuales suficientes para el backtest")

        equity = [10000]
        dates = [prices_df_m.index[0]] if len(prices_df_m.index) > 0 else []
        picks_list = []

        print(f"Datos preparados. Fechas: {len(prices_df_m)}")

        for i in range(1, len(prices_df_m)):
            try:
                prev_date = prices_df_m.index[i - 1]
                date = prices_df_m.index[i]

                # Calcular scores usando datos históricos hasta la fecha anterior
                historical_data = prices_df_m.loc[:prev_date].copy()
                if len(historical_data) < 15:  # Necesitamos suficientes datos
                    continue

                df_score = inertia_score(historical_data, corte=corte)
                if df_score is None or not df_score or len(historical_data) < 14:
                    continue

                # Obtener el último score ajustado
                try:
                    if "ScoreAdjusted" in df_score:
                        score_adjusted_df = df_score["ScoreAdjusted"]
                        if not score_adjusted_df.empty and len(score_adjusted_df) > 0:
                            last_scores = score_adjusted_df.iloc[-1]
                            # Convertir a Series si es necesario
                            if not isinstance(last_scores, pd.Series):
                                if hasattr(last_scores, 'items'):
                                    last_scores = pd.Series(last_scores)
                                else:
                                    last_scores = pd.Series(dtype=float)
                        else:
                            continue
                    else:
                        continue
                except Exception as e:
                    print(f"Error obteniendo scores: {e}")
                    continue

                # Asegurarse de que last_scores es una Series y eliminar NaN
                if not isinstance(last_scores, pd.Series):
                    continue
                    
                last_scores = last_scores.dropna().sort_values(ascending=False)

                if len(last_scores) == 0:
                    continue

                selected = last_scores.head(top_n).index.tolist()

                if not selected:
                    continue

                # Calcular retorno mensual
                weight = 1.0 / len(selected)

                # Obtener precios disponibles
                try:
                    available_prices = prices_df_m.loc[date]
                    prev_prices = prices_df_m.loc[prev_date]
                except Exception as e:
                    print(f"Error obteniendo precios para {date}: {e}")
                    continue

                # Filtrar tickers válidos
                valid_tickers = []
                for ticker in selected:
                    try:
                        # Verificar que el ticker exista en ambos períodos y tenga valores válidos
                        if (ticker in available_prices.index and 
                            ticker in prev_prices.index and
                            not pd.isna(available_prices[ticker]) and 
                            not pd.isna(prev_prices[ticker]) and
                            prev_prices[ticker] != 0):
                            valid_tickers.append(ticker)
                    except:
                        continue

                if len(valid_tickers) == 0:
                    continue

                # Calcular retornos
                rets = pd.Series(dtype=float)
                for ticker in valid_tickers:
                    try:
                        prev_price = prev_prices[ticker]
                        curr_price = available_prices[ticker]
                        if prev_price != 0 and not pd.isna(prev_price) and not pd.isna(curr_price):
                            ret_value = (curr_price / prev_price) - 1
                            rets[ticker] = ret_value if not np.isinf(ret_value) and not np.isnan(ret_value) else 0
                        else:
                            rets[ticker] = 0
                    except:
                        rets[ticker] = 0

                rets = rets.fillna(0)
                port_ret = (rets * weight).sum() - commission
                new_eq = equity[-1] * (1 + port_ret) if not np.isnan(port_ret) and not np.isinf(port_ret) else equity[-1]

                equity.append(new_eq)
                dates.append(date)

                # Guardar picks
                for rank, ticker in enumerate(valid_tickers[:top_n], 1):
                    try:
                        inercia_val = 0
                        score_adj_val = 0
                        
                        if "InerciaAlcista" in df_score:
                            try:
                                inercia_data = df_score["InerciaAlcista"]
                                if isinstance(inercia_data, pd.DataFrame) and len(inercia_data) > 0 and ticker in inercia_data.columns:
                                    inercia_val = inercia_data.iloc[-1][ticker] if len(inercia_data) > 0 else 0
                            except:
                                inercia_val = 0
                        
                        if ticker in last_scores.index:
                            score_adj_val = last_scores[ticker]

                        picks_list.append({
                            "Date": date.strftime("%Y-%m-%d"),
                            "Rank": rank,
                            "Ticker": str(ticker),
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
            print("❌ No se generaron resultados de backtest")
            raise ValueError("No se generaron resultados de backtest")

        equity_series = pd.Series(equity, index=dates)
        returns = equity_series.pct_change().fillna(0)
        drawdown = (equity_series / equity_series.cummax() - 1).fillna(0)

        bt = pd.DataFrame({
            "Equity": equity_series,
            "Returns": returns,
            "Drawdown": drawdown
        })
        picks_df = pd.DataFrame(picks_list)
        
        print(f"✅ Backtest completado. Equity final: {equity_series.iloc[-1]:.2f}")
        return bt, picks_df

    except Exception as e:
        print(f"❌ Error crítico en run_backtest: {e}")
        import traceback
        traceback.print_exc()
        empty_bt = pd.DataFrame(columns=["Equity", "Returns", "Drawdown"])
        empty_picks = pd.DataFrame(columns=["Date", "Rank", "Ticker", "Inercia", "ScoreAdj"])
        return empty_bt, empty_picks
   
    # Agregar DataFrame de debug
    debug_data = []
    
    # En el loop donde calculas los scores
    for i in range(1, len(prices_df_m)):
        # ... código existente ...
        
        if debug and "InerciaAlcista" in df_score:
            # Guardar datos de debug para los top tickers
            for ticker in selected[:5]:  # Solo los top 5 para no saturar
                if ticker in df_score["InerciaAlcista"].columns:
                    debug_info = {
                        "Date": date.strftime("%Y-%m-%d"),
                        "Ticker": ticker,
                        "InerciaAlcista": df_score["InerciaAlcista"].iloc[-1][ticker],
                        "ATR14": df_score["ATR14"].iloc[-1][ticker],
                        "F1": df_score["F1"].iloc[-1][ticker] if "F1" in df_score else None,
                        "F2": df_score["F2"].iloc[-1][ticker] if "F2" in df_score else None,
                        "ROC10": df_score["ROC10"].iloc[-1][ticker] if "ROC10" in df_score else None,
                        "ScoreAdjusted": last_scores[ticker] if ticker in last_scores else 0
                    }
                    debug_data.append(debug_info)
    
    # Retornar también el DataFrame de debug
    debug_df = pd.DataFrame(debug_data)
    return bt, picks_df, debug_df
