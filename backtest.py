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
    Calcula el score de inercia correctamente
    monthly_prices_df debe contener columnas: High, Low, Close
    """
    if monthly_prices_df is None or monthly_prices_df.empty:
        return pd.DataFrame()

    # Asegurarse de que tenemos las columnas necesarias
    required_cols = ['High', 'Low', 'Close']
    
    # Si es un DataFrame con múltiples tickers (MultiIndex columns)
    if isinstance(monthly_prices_df.columns, pd.MultiIndex):
        # Para múltiples tickers, procesar cada uno
        results = {}
        for ticker in monthly_prices_df.columns.levels[0]:
            try:
                ticker_data = monthly_prices_df[ticker]
                if all(col in ticker_data.columns for col in required_cols):
                    high = ticker_data['High']
                    low = ticker_data['Low']
                    close = ticker_data['Close']
                else:
                    # Usar Close para todas si no tenemos las columnas separadas
                    close_col = 'Adj Close' if 'Adj Close' in ticker_data.columns else 'Close'
                    if close_col in ticker_data.columns:
                        close = ticker_data[close_col]
                    else:
                        close = ticker_data.iloc[:, 0]
                    high = close
                    low = close
                
                # Calcular para este ticker
                roc_10 = close.pct_change(10)
                f1 = roc_10 * 0.6
                
                tr = monthly_true_range(high, low, close)
                atr14 = tr.rolling(14).mean()
                sma14 = close.rolling(14).mean()
                
                # Protección contra divisiones por cero
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(np.abs(sma14) > 1e-10, atr14 / sma14, 0)
                    denominator = ratio * 0.4
                    denominator = np.where(np.abs(denominator) > 1e-10, denominator, 1e-10)
                    inercia_alcista = np.where(np.abs(denominator) > 1e-10, f1 / denominator, 0)
                
                score = np.where(inercia_alcista < corte, 0, np.maximum(inercia_alcista, 0))
                score_adj = np.where(np.abs(atr14) > 1e-10, score / atr14, 0)
                
                results[ticker] = pd.DataFrame({
                    "InerciaAlcista": inercia_alcista,
                    "ATR14": atr14,
                    "Score": score,
                    "ScoreAdjusted": score_adj
                })
            except Exception as e:
                print(f"Error procesando ticker {ticker}: {e}")
                continue
        
        if results:
            # Combinar resultados
            combined_results = {}
            for metric in ["InerciaAlcista", "ATR14", "Score", "ScoreAdjusted"]:
                metric_data = {ticker: results[ticker][metric] for ticker in results.keys()}
                combined_results[metric] = pd.DataFrame(metric_data)
            return pd.DataFrame(combined_results)
        else:
            return pd.DataFrame()
    
    else:
        # DataFrame simple (un solo ticker o estructura plana)
        try:
            if all(col in monthly_prices_df.columns for col in required_cols):
                high = monthly_prices_df['High']
                low = monthly_prices_df['Low']
                close = monthly_prices_df['Close']
            else:
                # Determinar qué columna usar como precio
                price_col = None
                for col in ['Adj Close', 'Close', 'close', 'CLOSE']:
                    if col in monthly_prices_df.columns:
                        price_col = col
                        break
                if price_col is None and len(monthly_prices_df.columns) > 0:
                    price_col = monthly_prices_df.columns[0]
                
                if price_col is None:
                    return pd.DataFrame()
                
                close = monthly_prices_df[price_col]
                high = close
                low = close

            # Calcular ROC (10 períodos)
            roc_10 = close.pct_change(10)
            f1 = roc_10 * 0.6

            # Calcular ATR(14)
            tr = monthly_true_range(high, low, close)
            atr14 = tr.rolling(14).mean()

            # Calcular SMA(14)
            sma14 = close.rolling(14).mean()

            # Protección contra overflow y divisiones por cero
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                # Evitar división por cero
                safe_sma14 = np.where(np.abs(sma14) > 1e-10, sma14, 1e-10)
                ratio = atr14 / safe_sma14
                
                # Limitar valores extremos
                ratio = np.clip(ratio, -1e6, 1e6)
                
                denominator = ratio * 0.4
                safe_denominator = np.where(np.abs(denominator) > 1e-10, denominator, 1e-10)
                
                # Calcular inercia con protección
                inercia_alcista = np.where(np.abs(safe_denominator) > 1e-10, f1 / safe_denominator, 0)
                inercia_alcista = np.clip(inercia_alcista, -1e6, 1e6)
                
                # Aplicar corte
                score = np.where(inercia_alcista < corte, 0, np.maximum(inercia_alcista, 0))
                
                # Penalización por volatilidad
                safe_atr14 = np.where(np.abs(atr14) > 1e-10, atr14, 1e-10)
                score_adj = score / safe_atr14
                score_adj = np.clip(score_adj, -1e6, 1e6)

            return pd.DataFrame({
                "InerciaAlcista": inercia_alcista,
                "ATR14": atr14,
                "Score": score,
                "ScoreAdjusted": score_adj
            }).fillna(0)

        except Exception as e:
            print(f"Error en cálculo de inercia: {e}")
            return pd.DataFrame()

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

        # Preparar estructura de datos para inercia
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

        # Asegurar columnas necesarias
        required_cols = ['High', 'Low', 'Close']
        for col in required_cols:
            if col not in prices_df_m.columns and len(prices_df_m.columns) > 0:
                prices_df_m[col] = prices_df_m.iloc[:, 0]

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
                if df_score is None or df_score.empty or len(df_score) < 14:
                    continue

                # Obtener el último score ajustado
                try:
                    if "ScoreAdjusted" in df_score.columns:
                        if isinstance(df_score["ScoreAdjusted"], pd.DataFrame):
                            last_scores = df_score["ScoreAdjusted"].iloc[-1]
                        else:
                            last_scores = df_score["ScoreAdjusted"]
                        
                        # Convertir a Series si es necesario
                        if not isinstance(last_scores, pd.Series):
                            if hasattr(last_scores, 'items'):
                                last_scores = pd.Series(last_scores)
                            else:
                                continue
                    else:
                        continue
                except Exception as e:
                    print(f"Error obteniendo scores: {e}")
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
                    available_prices = prices_m.loc[date]
                    prev_prices = prices_m.loc[prev_date]
                except Exception as e:
                    print(f"Error obteniendo precios para {date}: {e}")
                    continue

                # Filtrar tickers válidos
                valid_tickers = []
                for ticker in selected:
                    try:
                        if ticker in available_prices.index and ticker in prev_prices.index:
                            if not pd.isna(available_prices[ticker]) and not pd.isna(prev_prices[ticker]):
                                if prev_prices[ticker] != 0:
                                    valid_tickers.append(ticker)
                    except:
                        continue

                if len(valid_tickers) == 0:
                    continue

                # Calcular retornos
                rets = pd.Series(dtype=float)
                for ticker in valid_tickers:
                    try:
                        if prev_prices[ticker] != 0 and not pd.isna(prev_prices[ticker]) and not pd.isna(available_prices[ticker]):
                            ret_value = (available_prices[ticker] / prev_prices[ticker]) - 1
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
                        
                        if "InerciaAlcista" in df_score.columns:
                            try:
                                inercia_data = df_score["InerciaAlcista"]
                                if isinstance(inercia_data, pd.DataFrame) and len(inercia_data) > 0:
                                    inercia_val = inercia_data.iloc[-1][ticker] if ticker in inercia_data.columns else 0
                                elif isinstance(inercia_data, pd.Series):
                                    inercia_val = inercia_data.iloc[-1] if len(inercia_data) > 0 else 0
                            except:
                                inercia_val = 0
                        
                        score_adj_val = last_scores.get(ticker, 0)

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
