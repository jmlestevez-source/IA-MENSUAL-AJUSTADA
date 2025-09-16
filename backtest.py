import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Cache global para resultados de cálculos
_calculation_cache = {}

def optimized_inertia_score(monthly_prices_df, corte=680, ohlc_data=None, cache_key=None):
    """
    ✅ OPTIMIZADO: Calcula el score de inercia con cache y optimización
    """
    # Usar cache si está disponible
    if cache_key and cache_key in _calculation_cache:
        return _calculation_cache[cache_key]
    
    if monthly_prices_df is None or monthly_prices_df.empty:
        return {}
    
    try:
        results = {}
        processed_count = 0
        total_tickers = len(monthly_prices_df.columns)
        
        print(f"📊 Calculando inercia para {total_tickers} tickers...")
        
        # ✅ OPTIMIZACIÓN: Procesar en lotes
        batch_size = 50
        ticker_batches = [monthly_prices_df.columns[i:i + batch_size] 
                         for i in range(0, len(monthly_prices_df.columns), batch_size)]
        
        for batch_idx, batch in enumerate(ticker_batches):
            print(f"📦 Procesando lote {batch_idx + 1}/{len(ticker_batches)} ({len(batch)} tickers)...")
            
            for ticker in batch:
                try:
                    # ✅ OPTIMIZACIÓN: Verificar cache por ticker
                    ticker_cache_key = f"{ticker}_{corte}"
                    if ticker_cache_key in _calculation_cache:
                        results[ticker] = _calculation_cache[ticker_cache_key]
                        processed_count += 1
                        continue
                    
                    # Obtener datos del ticker
                    close = monthly_prices_df[ticker].dropna()
                    
                    if len(close) < 15:
                        continue
                    
                    # ✅ OPTIMIZACIÓN: Mensualizar si es necesario
                    if close.index.freq != 'ME' and close.index.freq != 'M':
                        close = close.resample('ME').last()
                    
                    if len(close) < 15:
                        continue
                    
                    # ✅ OPTIMIZACIÓN: Usar datos OHLC si están disponibles
                    if ohlc_data and ticker in ohlc_data:
                        high = ohlc_data[ticker]['High']
                        low = ohlc_data[ticker]['Low']
                        close = ohlc_data[ticker]['Close']
                        
                        # Mensualizar si es necesario
                        if high.index.freq != 'ME' and high.index.freq != 'M':
                            high = high.resample('ME').last()
                            low = low.resample('ME').last()
                            close = close.resample('ME').last()
                    else:
                        # ✅ OPTIMIZACIÓN: Estimar OHLC si no están disponibles
                        monthly_returns = close.pct_change().dropna()
                        vol = monthly_returns.rolling(3).std().fillna(0.02)
                        vol = vol.clip(0.005, 0.03)
                        
                        high = close * (1 + vol * 0.5)
                        low = close * (1 - vol * 0.5)
                        high = pd.Series(np.maximum(high, close), index=close.index)
                        low = pd.Series(np.minimum(low, close), index=close.index)
                    
                    # ✅ OPTIMIZACIÓN: Cálculos vectorizados
                    # Calcular ROC de 10 meses
                    roc_10 = ((close - close.shift(10)) / close.shift(10)) * 100
                    
                    # F1 = ROC(10) * 0.6
                    f1 = roc_10 * 0.6
                    
                    # Calcular ATR(14) optimizado
                    atr_14 = optimized_atr_calculation(high, low, close, periods=14)
                    
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
                    
                    # ✅ OPTIMIZACIÓN: Limpiar valores inválidos
                    inercia_alcista = inercia_alcista.replace([np.inf, -np.inf], np.nan).fillna(0)
                    score = score.replace([np.inf, -np.inf], np.nan).fillna(0)
                    score_adjusted = score_adjusted.replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    # Almacenar resultados
                    ticker_result = {
                        "InerciaAlcista": inercia_alcista,
                        "ATR14": atr_14,
                        "Score": score,
                        "ScoreAdjusted": score_adjusted,
                        "F1": f1,
                        "F2": f2,
                        "ROC10": roc_10,
                        "VolatilityRatio": volatility_ratio
                    }
                    
                    results[ticker] = ticker_result
                    
                    # ✅ OPTIMIZACIÓN: Guardar en cache
                    if ticker_cache_key:
                        _calculation_cache[ticker_cache_key] = ticker_result
                    
                    processed_count += 1
                    
                    # ✅ OPTIMIZACIÓN: Mostrar progreso cada 25 tickers
                    if processed_count % 25 == 0:
                        print(f"✅ Procesados {processed_count}/{total_tickers} tickers")
                        
                except Exception as e:
                    print(f"⚠️  Error procesando {ticker}: {e}")
                    continue
            
            # ✅ OPTIMIZACIÓN: Pequeña pausa entre lotes para evitar sobrecarga
            time.sleep(0.1)
        
        print(f"✅ Completados {processed_count}/{total_tickers} tickers")
        
        if not results:
            return {}
        
        # Combinar resultados
        combined_results = {}
        for metric in ["InerciaAlcista", "ATR14", "Score", "ScoreAdjusted", "F1", "F2", "ROC10", "VolatilityRatio"]:
            metric_data = {}
            for ticker in results.keys():
                if metric in results[ticker]:
                    metric_data[ticker] = results[ticker][metric]
            if metric_data:
                combined_results[metric] = pd.DataFrame(metric_data)
        
        # ✅ OPTIMIZACIÓN: Guardar en cache general
        if cache_key:
            _calculation_cache[cache_key] = combined_results
        
        return combined_results
        
    except Exception as e:
        print(f"❌ Error en cálculo de inercia optimizado: {e}")
        return {}

def optimized_atr_calculation(high, low, close, periods=14):
    """
    ✅ OPTIMIZADO: Calcula ATR con métodos vectorizados
    """
    prev_close = close.shift(1)
    
    # True Range vectorizado
    hl = high - low
    hc = np.abs(high - prev_close)
    lc = np.abs(low - prev_close)
    
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    
    # ATR usando método de Wilder optimizado
    atr = tr.rolling(window=periods).mean()
    
    # Aplicar fórmula de Wilder vectorizada
    for i in range(periods, len(tr)):
        if i == periods:
            continue
        atr.iloc[i] = (atr.iloc[i-1] * (periods - 1) + tr.iloc[i]) / periods
    
    return atr

def run_optimized_backtest(prices, benchmark, commission=0.003, top_n=10, corte=680, 
                          ohlc_data=None, historical_info=None, fixed_allocation=False, 
                          use_roc_filter=False, use_sma_filter=False, spy_data=None):
    """
    ✅ OPTIMIZADO: Ejecuta backtest con optimizaciones de rendimiento
    """
    try:
        print("🚀 Iniciando backtest optimizado...")
        
        # Validar entrada
        if prices is None or (hasattr(prices, 'empty') and prices.empty):
            print("❌ Datos de precios vacíos")
            empty_bt = pd.DataFrame(columns=["Equity", "Returns", "Drawdown"])
            empty_picks = pd.DataFrame(columns=["Date", "Rank", "Ticker", "Inercia", "ScoreAdj"])
            return empty_bt, empty_picks

        # ✅ OPTIMIZACIÓN: Mensualizar precios solo si es necesario
        if isinstance(prices, pd.Series):
            if prices.index.freq != 'ME' and prices.index.freq != 'M':
                prices_m = prices.resample('ME').last()
                prices_df_m = pd.DataFrame({'Close': prices_m})
            else:
                prices_df_m = pd.DataFrame({'Close': prices})
        else:
            try:
                if prices.index.freq != 'ME' and prices.index.freq != 'M':
                    prices_m = prices.resample('ME').last()
                    prices_df_m = prices_m.copy()
                else:
                    prices_df_m = prices.copy()
            except Exception:
                prices_df_m = prices.copy()

        # ✅ OPTIMIZACIÓN: Mensualizar benchmark
        try:
            if isinstance(benchmark, pd.Series):
                if benchmark.index.freq != 'ME' and benchmark.index.freq != 'M':
                    bench_m = benchmark.resample('ME').last()
                else:
                    bench_m = benchmark
            else:
                if benchmark.index.freq != 'ME' and benchmark.index.freq != 'M':
                    bench_m = benchmark.resample('ME').last()
                else:
                    bench_m = benchmark
        except:
            bench_m = benchmark

        # ✅ OPTIMIZACIÓN: Preparar datos del SPY para filtros
        spy_monthly = None
        if (use_roc_filter or use_sma_filter) and spy_data is not None:
            try:
                if isinstance(spy_data, pd.DataFrame):
                    spy_series = spy_data.iloc[:, 0]  # Primera columna
                else:
                    spy_series = spy_data
                
                # Mensualizar SPY si es necesario
                if spy_series.index.freq != 'ME' and spy_series.index.freq != 'M':
                    spy_monthly = spy_series.resample('ME').last()
                else:
                    spy_monthly = spy_series
                    
                print("✅ Datos del SPY preparados para filtros de mercado")
            except Exception as e:
                print(f"⚠️ Error preparando datos del SPY: {e}")
                use_roc_filter = False
                use_sma_filter = False

        if prices_df_m.empty:
            print("❌ No hay datos mensuales suficientes")
            raise ValueError("No hay datos mensuales suficientes para el backtest")

        # ✅ OPTIMIZACIÓN: Preparar datos históricos si están disponibles
        historical_changes = None
        current_tickers = list(prices_df_m.columns)
        
        if historical_info and 'changes_data' in historical_info:
            historical_changes = historical_info['changes_data']
            print(f"✅ Usando datos históricos para verificación de constituyentes")
        else:
            print("⚠️  No hay datos históricos, usando todos los tickers disponibles")

        equity = [10000]
        dates = [prices_df_m.index[0]] if len(prices_df_m.index) > 0 else []
        picks_list = []
        
        # ✅ OPTIMIZACIÓN: Variable para rastrear si estamos en efectivo por filtros
        in_cash_by_filter = False

        print(f"Datos preparados. Fechas: {len(prices_df_m)}")
        print(f"Configuración: fixed_allocation={fixed_allocation}, ROC_filter={use_roc_filter}, SMA_filter={use_sma_filter}")

        # ✅ OPTIMIZACIÓN: Procesar en lotes mensuales
        total_months = len(prices_df_m) - 1
        processed_months = 0
        
        for i in range(1, len(prices_df_m)):
            try:
                prev_date = prices_df_m.index[i - 1]
                date = prices_df_m.index[i]

                # ✅ OPTIMIZACIÓN: Mostrar progreso cada 12 meses
                processed_months += 1
                if processed_months % 12 == 0:
                    progress_pct = (processed_months / total_months) * 100 if total_months > 0 else 0
                    print(f"📅 Progreso: {processed_months}/{total_months} meses ({progress_pct:.1f}%)")

                # ✅ OPTIMIZACIÓN: Verificar filtros de mercado
                market_filter_active = False
                filter_reasons = []
                
                if spy_monthly is not None and prev_date in spy_monthly.index:
                    spy_price = spy_monthly.loc[prev_date]
                    
                    # Filtro ROC (12 MESES)
                    if use_roc_filter and len(spy_monthly[:prev_date]) >= 13:
                        spy_12m_ago = spy_monthly[:prev_date].iloc[-13]
                        spy_roc_12m = ((spy_price - spy_12m_ago) / spy_12m_ago) * 100
                        
                        if spy_roc_12m < 0:
                            market_filter_active = True
                            filter_reasons.append(f"ROC 12M SPY: {spy_roc_12m:.2f}% < 0")
                    
                    # Filtro SMA
                    if use_sma_filter and len(spy_monthly[:prev_date]) >= 10:
                        spy_sma_10m = spy_monthly[:prev_date].iloc[-10:].mean()
                        
                        if spy_price < spy_sma_10m:
                            market_filter_active = True
                            filter_reasons.append(f"SPY ${spy_price:.2f} < SMA10 ${spy_sma_10m:.2f}")
                
                # Si los filtros están activos, mantener efectivo
                if market_filter_active:
                    if not in_cash_by_filter:
                        print(f"🛡️ {prev_date.strftime('%Y-%m-%d')}: Filtros de mercado ACTIVADOS - {', '.join(filter_reasons)}")
                        in_cash_by_filter = True
                    
                    # Mantener equity igual (efectivo)
                    equity.append(equity[-1])
                    dates.append(date)
                    continue
                else:
                    if in_cash_by_filter:
                        print(f"✅ {prev_date.strftime('%Y-%m-%d')}: Filtros de mercado DESACTIVADOS - Reanudando inversiones")
                        in_cash_by_filter = False

                # ✅ OPTIMIZACIÓN: VERIFICACIÓN HISTÓRICA
                if historical_changes is not None:
                    valid_tickers_for_date = get_valid_tickers_for_date(
                        prev_date, historical_changes, current_tickers
                    )
                    print(f"📅 {prev_date.strftime('%Y-%m-%d')}: {len(valid_tickers_for_date)} tickers válidos de {len(current_tickers)} disponibles")
                else:
                    valid_tickers_for_date = set(current_tickers)

                # ✅ OPTIMIZACIÓN: Calcular scores usando cache
                cache_key = f"inertia_{prev_date}_{corte}"
                historical_data = prices_df_m.loc[:prev_date].copy()
                
                if len(historical_data) < 15:
                    continue

                # ✅ OPTIMIZACIÓN: Filtrar datos históricos para incluir solo tickers válidos
                available_valid_tickers = list(set(historical_data.columns).intersection(valid_tickers_for_date))
                if not available_valid_tickers:
                    print(f"⚠️  No hay tickers válidos disponibles para {prev_date}")
                    # Mantener equity igual (efectivo)
                    equity.append(equity[-1])
                    dates.append(date)
                    continue
                
                historical_data_filtered = historical_data[available_valid_tickers]

                # ✅ OPTIMIZACIÓN: Pasar también los datos OHLC históricos filtrados
                historical_ohlc = None
                if ohlc_data:
                    historical_ohlc = {}
                    for ticker in available_valid_tickers:
                        if ticker in ohlc_data:
                            historical_ohlc[ticker] = {
                                'High': ohlc_data[ticker]['High'].loc[:prev_date],
                                'Low': ohlc_data[ticker]['Low'].loc[:prev_date],
                                'Close': ohlc_data[ticker]['Close'].loc[:prev_date]
                            }

                # ✅ OPTIMIZACIÓN: Calcular scores solo para tickers válidos con cache
                df_score = optimized_inertia_score(historical_data_filtered, corte=corte, 
                                                 ohlc_data=historical_ohlc, cache_key=cache_key)
                
                if df_score is None or not df_score:
                    # Mantener equity igual (efectivo)
                    equity.append(equity[-1])
                    dates.append(date)
                    continue

                # ✅ OPTIMIZACIÓN: Filtrar por corte ANTES de seleccionar
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
                            # Mantener equity igual (efectivo)
                            equity.append(equity[-1])
                            dates.append(date)
                            continue
                    else:
                        # Mantener equity igual (efectivo)
                        equity.append(equity[-1])
                        dates.append(date)
                        continue
                except Exception as e:
                    print(f"Error obteniendo scores: {e}")
                    # Mantener equity igual (efectivo)
                    equity.append(equity[-1])
                    dates.append(date)
                    continue

                if not isinstance(last_scores, pd.Series):
                    # Mantener equity igual (efectivo)
                    equity.append(equity[-1])
                    dates.append(date)
                    continue
                    
                last_scores = last_scores.dropna()

                if len(last_scores) == 0:
                    # Mantener equity igual (efectivo)
                    equity.append(equity[-1])
                    dates.append(date)
                    continue

                # ✅ OPTIMIZACIÓN: Filtrar PRIMERO por corte de inercia
                inercia_data = df_score.get("InerciaAlcista")
                if inercia_data is not None and not inercia_data.empty:
                    last_inercia = inercia_data.iloc[-1]
                    
                    # Filtrar solo tickers que pasan el corte
                    valid_tickers_scores = []
                    for ticker in last_scores.index:
                        if ticker in last_inercia.index:
                            inercia_val = last_inercia[ticker]
                            score_adj_val = last_scores[ticker]
                            
                            # Solo incluir si pasa el corte Y tiene score ajustado > 0
                            if inercia_val >= corte and score_adj_val > 0:
                                valid_tickers_scores.append({
                                    'ticker': ticker,
                                    'inercia': inercia_val,
                                    'score_adj': score_adj_val
                                })
                    
                    if not valid_tickers_scores:
                        print(f"⚠️ No hay tickers que pasen el corte en {prev_date}")
                        # Mantener equity igual (efectivo)
                        equity.append(equity[-1])
                        dates.append(date)
                        continue
                    
                    # Ordenar por score ajustado y seleccionar top N válidos
                    valid_tickers_scores = sorted(valid_tickers_scores, key=lambda x: x['score_adj'], reverse=True)
                    
                    if fixed_allocation:
                        # ✅ OPTIMIZACIÓN: Asignación fija de 10%
                        selected_picks = valid_tickers_scores[:10]  # Máximo 10 posiciones
                        selected = [pick['ticker'] for pick in selected_picks]
                        weight = 0.1  # 10% por posición
                        print(f"💰 Asignación fija: {len(selected)} posiciones, 10% cada una")
                    else:
                        # Seleccionar hasta top_n de los válidos
                        selected_picks = valid_tickers_scores[:min(top_n, len(valid_tickers_scores))]
                        selected = [pick['ticker'] for pick in selected_picks]
                        weight = 1.0 / len(selected)  # Peso equitativo
                        print(f"💰 Distribución equitativa: {len(selected)} posiciones, {weight*100:.1f}% cada una")
                    
                    print(f"📊 {prev_date.strftime('%Y-%m-%d')}: {len(selected)} tickers válidos seleccionados de {len(last_scores)} disponibles")
                    
                else:
                    # Fallback si no hay datos de inercia
                    selected = last_scores.sort_values(ascending=False).head(top_n).index.tolist()
                    weight = 1.0 / len(selected)
                    print(f"⚠️ Sin datos de inercia para {prev_date}, usando fallback")

                if not selected:
                    # Mantener equity igual (efectivo)
                    equity.append(equity[-1])
                    dates.append(date)
                    continue

                try:
                    available_prices = prices_df_m.loc[date]
                    prev_prices = prices_df_m.loc[prev_date]
                except Exception as e:
                    print(f"Error obteniendo precios para {date}: {e}")
                    # Mantener equity igual (efectivo)
                    equity.append(equity[-1])
                    dates.append(date)
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
                    print(f"⚠️ No hay precios válidos para {date}")
                    # Mantener equity igual (efectivo)
                    equity.append(equity[-1])
                    dates.append(date)
                    continue

                # ✅ OPTIMIZACIÓN: Recalcular peso con tickers que tienen precios válidos
                if fixed_allocation:
                    weight = 0.1
                    valid_tickers = valid_tickers[:10]  # Limitar a máximo 10
                else:
                    weight = 1.0 / len(valid_tickers)

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

                # ✅ OPTIMIZACIÓN: Guardar picks SOLO para tickers válidos que fueron seleccionados
                for rank, ticker in enumerate(valid_tickers, 1):
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

                        # ✅ OPTIMIZACIÓN: Solo guardar si realmente pasa el corte
                        if inercia_val >= corte and score_adj_val > 0:
                            picks_list.append({
                                "Date": date.strftime("%Y-%m-%d"),
                                "Rank": rank,
                                "Ticker": str(ticker),
                                "Inercia": float(inercia_val) if not pd.isna(inercia_val) else 0,
                                "ScoreAdj": float(score_adj_val) if not pd.isna(score_adj_val) else 0,
                                "HistoricallyValid": ticker in valid_tickers_for_date
                            })
                        else:
                            print(f"⚠️ ADVERTENCIA: {ticker} no debería estar seleccionado (Inercia: {inercia_val:.2f})")
                            
                    except Exception as e:
                        print(f"Error procesando pick {ticker}: {e}")
                        continue

            except Exception as e:
                print(f"Error en iteración {i}: {e}")
                # Mantener equity igual en caso de error
                equity.append(equity[-1])
                dates.append(date)
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
        
        print(f"✅ Backtest completado con verificación histórica. Equity final: {equity_series.iloc[-1]:.2f}")
        
        # ✅ OPTIMIZACIÓN: Estadísticas adicionales
        if not picks_df.empty and 'HistoricallyValid' in picks_df.columns:
            total_picks = len(picks_df)
            valid_picks = picks_df['HistoricallyValid'].sum()
            print(f"📊 Picks históricamente válidos: {valid_picks}/{total_picks} ({valid_picks/total_picks*100:.1f}%)")
        
        # ✅ OPTIMIZACIÓN: Estadísticas de selección
        if not picks_df.empty:
            picks_by_month = picks_df.groupby('Date').size()
            avg_picks_per_month = picks_by_month.mean()
            min_picks = picks_by_month.min()
            max_picks = picks_by_month.max()
            print(f"📈 Picks por mes - Promedio: {avg_picks_per_month:.1f}, Min: {min_picks}, Max: {max_picks}")
        
        return bt, picks_df

    except Exception as e:
        print(f"❌ Error crítico en run_optimized_backtest: {e}")
        import traceback
        traceback.print_exc()
        empty_bt = pd.DataFrame(columns=["Equity", "Returns", "Drawdown"])
        empty_picks = pd.DataFrame(columns=["Date", "Rank", "Ticker", "Inercia", "ScoreAdj"])
        return empty_bt, empty_picks

# ✅ CORRECCIÓN: Función para calcular tabla de rendimientos mensuales con consistencia
def calculate_monthly_returns_by_year(equity_series):
    """
    ✅ CORREGIDO: Calcula retornos mensuales distribuidos por año para tabla de dashboard
    Con manejo consistente de "-" vs "0.00%"
    """
    try:
        if equity_series is None or len(equity_series) < 2:
            return pd.DataFrame()
        
        # Calcular retornos mensuales
        monthly_returns = equity_series.pct_change().fillna(0)
        
        # Agrupar por año y mes
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_by_year = monthly_returns.groupby([
            monthly_returns.index.year, 
            monthly_returns.index.month
        ]).apply(lambda x: (1 + x).prod() - 1)
        
        # Crear DataFrame con estructura de tabla
        years = sorted(monthly_by_year.index.get_level_values(0).unique())
        
        # Crear tabla con meses como columnas (en español)
        months_es = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        
        table_data = []
        
        for year in years:
            year_data = {'Año': year}
            
            # Obtener datos de este año
            year_monthly = monthly_by_year[monthly_by_year.index.get_level_values(0) == year]
            
            # Llenar meses con formato consistente
            for i, month_abbr in enumerate(months_es, 1):
                if i in year_monthly.index.get_level_values(1):
                    return_value = year_monthly[year_monthly.index.get_level_values(1) == i].iloc[0]
                    
                    # ✅ CORRECCIÓN: Formato consistente para valores cercanos a cero
                    if np.isnan(return_value) or np.isinf(return_value):
                        year_data[month_abbr] = "-"
                    elif abs(return_value) < 0.0001:  # Menos de 0.01%
                        year_data[month_abbr] = "0.00%"
                    else:
                        year_data[month_abbr] = f"{return_value*100:.2f}%"
                else:
                    year_data[month_abbr] = "-"
            
            # Calcular YTD con formato consistente
            year_equity = equity_series[equity_series.index.year == year]
            if len(year_equity) > 1:
                ytd_return = (year_equity.iloc[-1] / year_equity.iloc[0]) - 1
                
                # ✅ CORRECCIÓN: Formato consistente para YTD
                if np.isnan(ytd_return) or np.isinf(ytd_return):
                    year_data['YTD'] = "-"
                elif abs(ytd_return) < 0.0001:  # Menos de 0.01%
                    year_data['YTD'] = "0.00%"
                else:
                    year_data['YTD'] = f"{ytd_return*100:.2f}%"
            else:
                year_data['YTD'] = "-"
            
            table_data.append(year_data)
        
        # Crear DataFrame final
        if table_data:
            result_df = pd.DataFrame(table_data)
            
            # Asegurar orden correcto de columnas
            columns_order = ['Año'] + months_es + ['YTD']
            result_df = result_df[columns_order]
            
            return result_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error calculando tabla de retornos mensuales: {e}")
        return pd.DataFrame()

# Mantener funciones existentes para compatibilidad
def monthly_true_range(high, low, close):
    """Función mantenida para compatibilidad"""
    return calcular_atr_amibroker(high, low, close, periods=1)

def calcular_atr_amibroker(high, low, close, periods=14):
    """Función mantenida para compatibilidad"""
    return optimized_atr_calculation(high, low, close, periods=periods)

def run_backtest(prices, benchmark, commission=0.003, top_n=10, corte=680, 
                ohlc_data=None, historical_info=None, fixed_allocation=False, 
                use_roc_filter=False, use_sma_filter=False, spy_data=None):
    """Wrapper para backward compatibility"""
    return run_optimized_backtest(prices, benchmark, commission, top_n, corte,
                                 ohlc_data, historical_info, fixed_allocation,
                                 use_roc_filter, use_sma_filter, spy_data)

def inertia_score(monthly_prices_df, corte=680, ohlc_data=None):
    """Wrapper para backward compatibility"""
    return optimized_inertia_score(monthly_prices_df, corte, ohlc_data)
