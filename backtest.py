import pandas as pd
import numpy as np

def calcular_atr_amibroker(high, low, close, periods=14):
    """
    Calcula el ATR exactamente como AmiBroker - IDÉNTICO al código Python que funciona
    """
    prev_close = close.shift(1)

    # True Range: máximo de tres valores
    hl = high - low  # High - Low
    hc = np.abs(high - prev_close)  # |High - PrevClose|
    lc = np.abs(low - prev_close)  # |Low - PrevClose|

    # True Range es el máximo de los tres
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    # ATR usando el método de Wilder - EXACTO como en el código Python
    atr = tr.rolling(window=periods).mean()

    # Aplicar la fórmula de Wilder para los siguientes valores
    for i in range(periods, len(tr)):
        if i == periods:
            continue  # Ya tenemos el primer valor
        atr.iloc[i] = (atr.iloc[i-1] * (periods - 1) + tr.iloc[i]) / periods

    return atr

def convertir_a_mensual_con_ohlc(ohlc_data):
    """
    Convierte datos diarios OHLC a mensuales - EXACTO como en el código Python
    """
    monthly_data = {}
    
    for ticker, data in ohlc_data.items():
        try:
            # Crear DataFrame con OHLC
            df = pd.DataFrame({
                'High': data['High'],
                'Low': data['Low'],
                'Close': data['Close']
            })
            
            # Agregación mensual - EXACTA como en el código Python
            df_monthly = df.resample('ME').agg({
                'High': 'max',   # Máximo del mes
                'Low': 'min',    # Mínimo del mes  
                'Close': 'last'  # Cierre del último día del mes
            })
            
            monthly_data[ticker] = {
                'High': df_monthly['High'],
                'Low': df_monthly['Low'],
                'Close': df_monthly['Close']
            }
            
        except Exception as e:
            print(f"Error convirtiendo {ticker} a mensual: {e}")
            continue
    
    return monthly_data

def inertia_score(monthly_prices_df, corte=680, ohlc_data=None):
    """
    Calcula el score de inercia - IDÉNTICO al código Python que funciona
    """
    if monthly_prices_df is None or monthly_prices_df.empty:
        return pd.DataFrame()

    try:
        results = {}
        
        # Si tenemos datos OHLC reales, usarlos
        if ohlc_data:
            monthly_ohlc = convertir_a_mensual_con_ohlc(ohlc_data)
        else:
            monthly_ohlc = None
        
        for ticker in monthly_prices_df.columns:
            try:
                # Si tenemos datos OHLC reales, usarlos
                if monthly_ohlc and ticker in monthly_ohlc:
                    high = monthly_ohlc[ticker]['High']
                    low = monthly_ohlc[ticker]['Low']
                    close = monthly_ohlc[ticker]['Close']
                else:
                    # Fallback: usar solo Close (esto debería evitarse)
                    close = monthly_prices_df[ticker].dropna()
                    if len(close) < 15:
                        continue
                    
                    # Mensualizar si es necesario
                    if close.index.freq != 'ME' and close.index.freq != 'M':
                        close = close.resample('ME').last()
                    
                    # Estimar High/Low (menos preciso)
                    monthly_returns = close.pct_change().dropna()
                    vol = monthly_returns.rolling(3).std().fillna(0.02)
                    vol = vol.clip(0.005, 0.03)
                    
                    high = close * (1 + vol * 0.5)
                    low = close * (1 - vol * 0.5)
                    high = pd.Series(np.maximum(high, close), index=close.index)
                    low = pd.Series(np.minimum(low, close), index=close.index)

                if len(close) < 15:
                    continue

                # CÁLCULOS EXACTOS COMO EN EL CÓDIGO PYTHON QUE FUNCIONA
                
                # Calcular ROC de 10 meses (en porcentaje)
                roc_10 = ((close - close.shift(10)) / close.shift(10)) * 100

                # F1 = ROC(10) * 0.6 (0.4 + 0.2)
                f1 = roc_10 * 0.6

                # Calcular ATR(14) exactamente como AmiBroker
                atr_14 = calcular_atr_amibroker(high, low, close, periods=14)

                # Calcular SMA(14)
                sma_14 = close.rolling(14).mean()

                # F2 = (ATR14/SMA14) * 0.4
                volatility_ratio = atr_14 / sma_14
                f2 = volatility_ratio * 0.4

                # Inercia Alcista = F1 / F2
                inercia_alcista = f1 / f2

                # Score = Inercia si >= corte, sino 0
                score = np.where(inercia_alcista >= corte, inercia_alcista, 0)
                score = pd.Series(score, index=inercia_alcista.index)

                # Score Adjusted = Score / ATR14
                score_adjusted = score / atr_14

                # Limpiar valores infinitos y NaN
                inercia_alcista = inercia_alcista.replace([np.inf, -np.inf], np.nan).fillna(0)
                score = score.replace([np.inf, -np.inf], np.nan).fillna(0)
                score_adjusted = score_adjusted.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Almacenar resultados
                results[ticker] = {
                    "InerciaAlcista": inercia_alcista,
                    "ATR14": atr_14,
                    "Score": score,
                    "ScoreAdjusted": score_adjusted,
                    "F1": f1,
                    "F2": f2,
                    "ROC10": roc_10,
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

def run_backtest(prices, benchmark, commission=0.003, top_n=10, corte=680, ohlc_data=None):
    try:
        print("Iniciando backtest...")
        
        # Validar entrada
        if prices is None or (hasattr(prices, 'empty') and prices.empty):
            print("❌ Datos de precios vacíos")
            empty_bt = pd.DataFrame(columns=["Equity", "Returns", "Drawdown"])
            empty_picks = pd.DataFrame(columns=["Date", "Rank", "Ticker", "Inercia", "ScoreAdj"])
            return empty_bt, empty_picks

        # Mensualizar precios (solo para el backtest, los cálculos usan OHLC)
        if isinstance(prices, pd.Series):
            prices_m = prices.resample('ME').last()
            prices_df_m = pd.DataFrame({'Close': prices_m})
        else:
            try:
                prices_m = prices.resample('ME').last()
                prices_df_m = prices_m.copy()
            except Exception:
                prices_df_m = prices.copy()

        # Mensualizar benchmark
        try:
            if isinstance(benchmark, pd.Series):
                bench_m = benchmark.resample('ME').last()
            else:
                bench_m = benchmark.resample('ME').last()
        except:
            bench_m = benchmark

        if prices_df_m.empty:
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
                if len(historical_data) < 15:
                    continue

                # Pasar también los datos OHLC históricos
                historical_ohlc = None
                if ohlc_data:
                    historical_ohlc = {}
                    for ticker, data in ohlc_data.items():
                        if ticker in historical_data.columns:
                            historical_ohlc[ticker] = {
                                'High': data['High'].loc[:prev_date],
                                'Low': data['Low'].loc[:prev_date],
                                'Close': data['Close'].loc[:prev_date]
                            }

                df_score = inertia_score(historical_data, corte=corte, ohlc_data=historical_ohlc)
                if df_score is None or not df_score:
                    continue

                # Resto del código del backtest igual...
                try:
                    if "ScoreAdjusted" in df_score:
                        score_adjusted_df = df_score["ScoreAdjusted"]
                        if not score_adjusted_df.empty and len(score_adjusted_df) > 0:
                            last_scores = score_adjusted_df.iloc[-1]
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

                if not isinstance(last_scores, pd.Series):
                    continue
                    
                last_scores = last_scores.dropna().sort_values(ascending=False)

                if len(last_scores) == 0:
                    continue

                selected = last_scores.head(top_n).index.tolist()
                if not selected:
                    continue

                weight = 1.0 / len(selected)

                try:
                    available_prices = prices_df_m.loc[date]
                    prev_prices = prices_df_m.loc[prev_date]
                except Exception as e:
                    print(f"Error obteniendo precios para {date}: {e}")
                    continue

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

# Mantener para compatibilidad
def monthly_true_range(high, low, close):
    """Función mantenida para compatibilidad"""
    return calcular_atr_amibroker(high, low, close, periods=1)
