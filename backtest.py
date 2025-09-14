import pandas as pd
import numpy as np

def monthly_true_range(high, low, close):
    """
    Calcula el True Range mensual correctamente
    Asumiendo que high, low, close son series mensuales
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
    Calcula el score de inercia siguiendo exactamente la fórmula de AmiBroker
    monthly_prices_df: DataFrame con tickers como columnas y fechas como índice (datos mensuales)
    """
    if monthly_prices_df is None or monthly_prices_df.empty:
        return {}

    try:
        results = {}
        
        for ticker in monthly_prices_df.columns:
            try:
                close = monthly_prices_df[ticker].dropna()
                
                if len(close) < 15:  # Necesitamos al menos 15 períodos
                    continue
                
                # Para datos mensuales, usar close como high y low
                high = close
                low = close
                
                # Calcular ROC(10) - cambio porcentual de 10 meses
                roc_10 = close.pct_change(10)
                
                # Calcular F1 = ROC(C,10)*0.6
                f1 = roc_10 * 0.6
                
                # Calcular True Range mensual
                tr = monthly_true_range(high, low, close)
                
                # Calcular ATR(14) - promedio simple de 14 meses
                atr14 = tr.rolling(14).mean()
                
                # Calcular SMA(14) - promedio simple de 14 meses
                sma14 = close.rolling(14).mean()
                
                # Protección contra divisiones por cero
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Calcular F2 = (ATR14/MA(C,14))*0.4
                    safe_sma14 = np.where(np.abs(sma14) > 1e-10, sma14, 1e-10)
                    f2 = (atr14 / safe_sma14) * 0.4
                    
                    # Calcular InerciaAlcista = F1/F2
                    safe_f2 = np.where(np.abs(f2) > 1e-10, f2, 1e-10)
                    inercia_alcista = f1 / safe_f2
                    
                    # Aplicar corte: Score = IIf(InerciaAlcista<Corte,0,Max(InerciaAlcista,0))
                    score = np.where(inercia_alcista < corte, 0, np.maximum(inercia_alcista, 0))
                    
                    # Score ajustado: ScoreAdjusted = Score / ATR14
                    safe_atr14 = np.where(np.abs(atr14) > 1e-10, atr14, 1e-10)
                    score_adj = score / safe_atr14
                
                # Almacenar resultados para este ticker
                results[ticker] = {
                    "InerciaAlcista": pd.Series(inercia_alcista, index=close.index).fillna(0),
                    "ATR14": pd.Series(atr14, index=close.index).fillna(0),
                    "Score": pd.Series(score, index=close.index).fillna(0),
                    "ScoreAdjusted": pd.Series(score_adj, index=close.index).fillna(0)
                }
                
            except Exception as e:
                print(f"Error procesando ticker {ticker}: {e}")
                continue
        
        if not results:
            return {}
        
        # Combinar resultados
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

        if prices_df_m.empty:
            print("❌ No hay datos mensuales suficientes")
            raise ValueError("No hay datos mensuales suficientes para el backtest")

        equity = [10000]
        dates = [prices_df_m.index[0]] if len(prices_df_m.index) > 0 else []
        picks_list = []

        print(f"Datos mensuales preparados. Fechas: {len(prices_df_m)}")

        # Iterar desde el índice 15 para tener suficientes datos
        for i in range(15, len(prices_df_m)):
            try:
                prev_date = prices_df_m.index[i - 1]
                date = prices_df_m.index[i]

                # Calcular scores usando datos históricos hasta el mes anterior
                historical_data = prices_df_m.loc[:prev_date].copy()
                
                if len(historical_data) < 15:
                    continue

                # Calcular scores de inercia
                df_score = inertia_score(historical_data, corte=corte)
                
                if not df_score or "ScoreAdjusted" not in df_score:
                    continue

                # Obtener el último score ajustado de cada ticker
                try:
                    score_adjusted_df = df_score["ScoreAdjusted"]
                    if score_adjusted_df.empty:
                        continue
                    
                    # Obtener los scores del último período disponible
                    last_scores = score_adjusted_df.iloc[-1]
                    
                    # Convertir a Series si es necesario y eliminar NaN
                    if not isinstance(last_scores, pd.Series):
                        if hasattr(last_scores, 'items'):
                            last_scores = pd.Series(last_scores)
                        else:
                            continue
                    
                    last_scores = last_scores.dropna().sort_values(ascending=False)
                    
                except Exception as e:
                    print(f"Error obteniendo scores para {date}: {e}")
                    continue

                if len(last_scores) == 0:
                    continue

                # Seleccionar top_n activos con mejor ScoreAdjusted
                selected = last_scores.head(top_n).index.tolist()

                if not selected:
                    continue

                # Calcular retorno mensual de la cartera
                weight = 1.0 / len(selected)

                # Obtener precios del mes actual y mes anterior
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

                # Calcular retornos de los activos seleccionados
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

                # Guardar picks seleccionados
                for rank, ticker in enumerate(valid_tickers[:top_n], 1):
                    try:
                        inercia_val = 0
                        score_adj_val = 0
                        
                        # Obtener valores de inercia y score ajustado
                        if "InerciaAlcista" in df_score:
                            try:
                                inercia_data = df_score["InerciaAlcista"]
                                if (isinstance(inercia_data, pd.DataFrame) and 
                                    len(inercia_data) > 0 and 
                                    ticker in inercia_data.columns):
                                    inercia_val = inercia_data.iloc[-1][ticker]
                            except:
                                pass
                        
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
                        print(f"Error procesando pick {ticker} para {date}: {e}")
                        continue

            except Exception as e:
                print(f"Error en iteración {i} para fecha {date if 'date' in locals() else 'desconocida'}: {e}")
                continue

        if len(equity) <= 1:
            print("❌ No se generaron resultados de backtest")
            raise ValueError("No se generaron resultados de backtest")

        # Crear DataFrame de resultados
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
