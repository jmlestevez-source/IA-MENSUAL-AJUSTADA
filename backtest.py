import pandas as pd
import numpy as np

def monthly_true_range(high, low, close):
    """
    Calcula el True Range mensual correctamente
    """
    prev_close = close.shift(1)
    tr = pd.DataFrame({
        'hl': np.abs(high - low),
        'hc': np.abs(high - prev_close),
        'lc': np.abs(low - prev_close)
    }).max(axis=1)
    return tr

def inertia_score(monthly_prices_df, corte=680):
    """
    Calcula el score de inercia correctamente replicando el código AFL
    """
    if monthly_prices_df is None or monthly_prices_df.empty:
        return pd.DataFrame()

    try:
        results = {}
        
        for ticker in monthly_prices_df.columns:
            try:
                # Obtener serie de precios para este ticker
                close = monthly_prices_df[ticker].dropna()
                
                if len(close) < 15:  # Necesitamos al menos 15 períodos
                    continue
                
                # Para datos mensuales, usar close como high y low
                high = close
                low = close
                
                # Calcular ROC al estilo AmiBroker (en porcentaje)
                # ROC = ((Close - Close[n]) / Close[n]) * 100
                def amibroker_roc(series, periods):
                    return ((series - series.shift(periods)) / series.shift(periods)) * 100
                
                # Calcular los dos componentes del ROC
                roc_10 = amibroker_roc(close, 10) * 0.4
                roc_10_2 = amibroker_roc(close, 10) * 0.2  # Segundo ROC10
                
                # F1 es la suma de ambos ROCs
                f1 = roc_10 + roc_10_2  # Esto da ROC10 * 0.6 en total
                
                # Calcular ATR(14)
                tr = monthly_true_range(high, low, close)
                atr14 = tr.rolling(14).mean()
                
                # Calcular SMA(14)
                sma14 = close.rolling(14).mean()
                
                # Calcular F2 (denominador) exactamente como en AFL
                # F2 = (ATR14/MA(C,14))*0.4
                with np.errstate(divide='ignore', invalid='ignore'):
                    f2 = (atr14 / sma14) * 0.4
                    f2 = f2.replace([np.inf, -np.inf], np.nan)
                    f2 = f2.fillna(0.0001)  # Evitar división por cero
                
                # Calcular Inercia Alcista = F1 / F2
                inercia_alcista = f1 / f2
                inercia_alcista = inercia_alcista.replace([np.inf, -np.inf], 0)
                inercia_alcista = inercia_alcista.fillna(0)
                
                # Aplicar corte y calcular Score
                score = pd.Series(
                    np.where(inercia_alcista < corte, 0, np.maximum(inercia_alcista, 0)),
                    index=inercia_alcista.index
                )
                
                # Score Ajustado = Score / ATR14
                with np.errstate(divide='ignore', invalid='ignore'):
                    score_adj = score / atr14
                    score_adj = score_adj.replace([np.inf, -np.inf], 0)
                    score_adj = score_adj.fillna(0)
                
                # Almacenar resultados
                results[ticker] = {
                    "InerciaAlcista": inercia_alcista,
                    "ATR14": atr14,
                    "Score": score,
                    "ScoreAdjusted": score_adj
                }
                
            except Exception as e:
                print(f"Error procesando ticker {ticker}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        # Combinar resultados en estructura esperada
        combined_results = {}
        for metric in ["InerciaAlcista", "ATR14", "Score", "ScoreAdjusted"]:
            metric_data = {}
            for ticker in results.keys():
                metric_data[ticker] = results[ticker][metric]
            combined_results[metric] = pd.DataFrame(metric_data)
        
        return combined_results
        
    except Exception as e:
        print(f"Error en cálculo de inercia: {e}")
        return {}
        
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

def debug_calculations(ticker_data, ticker_name):
    """Función para debuggear y comparar cálculos con AmiBroker"""
    close = ticker_data
    
    # Calcular componentes paso a paso
    roc_10_1 = ((close - close.shift(10)) / close.shift(10)) * 100 * 0.4
    roc_10_2 = ((close - close.shift(10)) / close.shift(10)) * 100 * 0.2
    f1 = roc_10_1 + roc_10_2
    
    atr14 = monthly_true_range(close, close, close).rolling(14).mean()
    sma14 = close.rolling(14).mean()
    f2 = (atr14 / sma14) * 0.4
    
    inercia = f1 / f2
    
    print(f"\nDebug para {ticker_name}:")
    print(f"Último ROC10*0.4: {roc_10_1.iloc[-1]:.2f}")
    print(f"Último ROC10*0.2: {roc_10_2.iloc[-1]:.2f}")
    print(f"Último F1: {f1.iloc[-1]:.2f}")
    print(f"Último ATR14: {atr14.iloc[-1]:.2f}")
    print(f"Último SMA14: {sma14.iloc[-1]:.2f}")
    print(f"Último F2: {f2.iloc[-1]:.4f}")
    print(f"Última Inercia: {inercia.iloc[-1]:.2f}")
