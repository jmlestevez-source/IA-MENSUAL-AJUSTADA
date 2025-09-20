# backtest.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calcular_atr_optimizado(high, low, close, periods=14):
    """
    Calcula ATR usando EWM (Exponential Weighted Mean)
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/periods, adjust=False).mean()
    return atr

def convertir_a_mensual_con_ohlc(ohlc_data):
    """Convierte datos diarios OHLC a mensuales"""
    monthly_data = {}
    for ticker, data in ohlc_data.items():
        try:
            df = pd.DataFrame({
                'High': data['High'],
                'Low': data['Low'],
                'Close': data['Close']
            })
            df_monthly = df.resample('ME').agg({
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            })
            monthly_data[ticker] = {
                'High': df_monthly['High'],
                'Low': df_monthly['Low'],
                'Close': df_monthly['Close']
            }
        except Exception as e:
            print(f"Error convirtiendo {ticker}: {e}")
            continue
    return monthly_data

def precalculate_valid_tickers_by_date(monthly_dates, historical_changes_data, current_tickers):
    """
    Precalcula tickers válidos para todas las fechas de una vez
    """
    if historical_changes_data is None or historical_changes_data.empty:
        return {date: set(current_tickers) for date in monthly_dates}
    
    # Convertir fechas
    historical_changes_data['Date'] = pd.to_datetime(historical_changes_data['Date']).dt.date
    valid_by_date = {}
    
    for target_date in monthly_dates:
        if isinstance(target_date, pd.Timestamp):
            target_date_clean = target_date.date()
        else:
            target_date_clean = target_date
        valid_tickers = set(current_tickers)
        future_changes = historical_changes_data[historical_changes_data['Date'] > target_date_clean]
        for _, change in future_changes.iterrows():
            ticker = change.get('Ticker')
            action = change.get('Action')
            if action == 'Added':
                valid_tickers.discard(ticker)
            elif action == 'Removed':
                valid_tickers.add(ticker)
        valid_by_date[target_date] = valid_tickers
    return valid_by_date

def precalculate_all_indicators(prices_df_m, ohlc_data, corte=680):
    """
    Precalcula TODOS los indicadores de una vez
    """
    try:
        print("🚀 Precalculando todos los indicadores...")
        all_indicators = {}
        monthly_ohlc = convertir_a_mensual_con_ohlc(ohlc_data) if ohlc_data else None
        total_tickers = len(prices_df_m.columns)
        
        for i, ticker in enumerate(prices_df_m.columns):
            if (i + 1) % 50 == 0:
                print(f"  Procesando ticker {i+1}/{total_tickers}...")
            try:
                ticker_data = prices_df_m[ticker].dropna()
                if len(ticker_data) < 15:
                    continue
                
                # Obtener datos OHLC
                if monthly_ohlc and ticker in monthly_ohlc:
                    high = monthly_ohlc[ticker]['High']
                    low = monthly_ohlc[ticker]['Low']
                    close = monthly_ohlc[ticker]['Close']
                else:
                    close = ticker_data
                    if getattr(close.index, 'freq', None) not in ['ME', 'M']:
                        close = close.resample('ME').last()
                    monthly_returns = close.pct_change().dropna()
                    vol = monthly_returns.rolling(3).std().fillna(0.02).clip(0.005, 0.03)
                    high = close * (1 + vol * 0.5)
                    low = close * (1 - vol * 0.5)
                
                # Limpiar
                close = close.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
                high = high.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
                low = low.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
                if close.isna().all():
                    continue
                
                # Indicadores
                roc_10 = ((close - close.shift(10)) / close.shift(10)) * 100
                f1 = roc_10 * 0.6
                atr_14 = calcular_atr_optimizado(high, low, close, periods=14)
                sma_14 = close.rolling(14).mean()
                volatility_ratio = atr_14 / sma_14
                f2 = volatility_ratio * 0.4
                inercia_alcista = f1 / f2
                score = np.where(inercia_alcista >= corte, inercia_alcista, 0)
                score = pd.Series(score, index=inercia_alcista.index)
                score_adjusted = score / atr_14
                
                inercia_alcista = inercia_alcista.replace([np.inf, -np.inf], np.nan).fillna(0)
                score = score.replace([np.inf, -np.inf], np.nan).fillna(0)
                score_adjusted = score_adjusted.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                all_indicators[ticker] = {
                    'InerciaAlcista': inercia_alcista,
                    'Score': score,
                    'ScoreAdjusted': score_adjusted,
                    'ATR14': atr_14,
                    'ROC10': roc_10
                }
            except Exception as e:
                print(f"Error en {ticker}: {e}")
                continue
        
        print(f"✅ Indicadores precalculados para {len(all_indicators)} tickers")
        return all_indicators
    except Exception as e:
        print(f"Error precalculando indicadores: {e}")
        return {}

def inertia_score(monthly_prices_df, corte=680, ohlc_data=None):
    """
    Calcula el score de inercia - devuelve un dict de DataFrames por métrica
    """
    try:
        if monthly_prices_df is None or monthly_prices_df.empty:
            return {}
        results = {}
        monthly_ohlc = convertir_a_mensual_con_ohlc(ohlc_data) if ohlc_data else None
        
        for ticker in monthly_prices_df.columns:
            try:
                ticker_data = monthly_prices_df[ticker].dropna()
                if len(ticker_data) < 15:
                    continue
                
                if monthly_ohlc and ticker in monthly_ohlc:
                    high = monthly_ohlc[ticker]['High']
                    low = monthly_ohlc[ticker]['Low']
                    close = monthly_ohlc[ticker]['Close']
                else:
                    close = ticker_data
                    if getattr(close.index, 'freq', None) not in ['ME', 'M']:
                        close = close.resample('ME').last()
                    monthly_returns = close.pct_change().dropna()
                    vol = monthly_returns.rolling(3).std().fillna(0.02).clip(0.005, 0.03)
                    high = close * (1 + vol * 0.5)
                    low = close * (1 - vol * 0.5)
                    high = pd.Series(np.maximum(high, close), index=close.index)
                    low = pd.Series(np.minimum(low, close), index=close.index)
                
                if len(close) < 15:
                    continue
                
                close = close.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
                high = high.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
                low = low.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
                
                if close.isna().all() or high.isna().all() or low.isna().all():
                    continue
                
                roc_10 = ((close - close.shift(10)) / close.shift(10)) * 100
                f1 = roc_10 * 0.6
                atr_14 = calcular_atr_optimizado(high, low, close, periods=14)
                sma_14 = close.rolling(14).mean()
                volatility_ratio = atr_14 / sma_14
                f2 = volatility_ratio * 0.4
                inercia_alcista = f1 / f2
                score = np.where(inercia_alcista >= corte, inercia_alcista, 0)
                score = pd.Series(score, index=inercia_alcista.index)
                score_adjusted = score / atr_14
                
                inercia_alcista = inercia_alcista.replace([np.inf, -np.inf], np.nan).fillna(0)
                score = score.replace([np.inf, -np.inf], np.nan).fillna(0)
                score_adjusted = score_adjusted.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                if (inercia_alcista == 0).all() and (score == 0).all() and (score_adjusted == 0).all():
                    continue
                
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
            return {}
        
        combined_results = {}
        for metric in ["InerciaAlcista", "ATR14", "Score", "ScoreAdjusted", "F1", "F2", "ROC10", "VolatilityRatio"]:
            metric_data = {}
            for ticker in results.keys():
                if metric in results[ticker] and results[ticker][metric] is not None:
                    metric_data[ticker] = results[ticker][metric]
            if metric_data:
                combined_results[metric] = pd.DataFrame(metric_data)
        return combined_results
    except Exception as e:
        print(f"Error en cálculo de inercia: {e}")
        return {}

def run_backtest_optimized(prices, benchmark, commission=0.003, top_n=10, corte=680, 
                          ohlc_data=None, historical_info=None, fixed_allocation=False,
                          use_roc_filter=False, use_sma_filter=False, spy_data=None,
                          progress_callback=None):
    """
    VERSIÓN OPTIMIZADA del backtest con precálculo
    Retorna: bt_results (DataFrame), picks_df (DataFrame)
    """
    try:
        print("🚀 Iniciando backtest OPTIMIZADO...")
        if prices is None or prices.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Mensualizar
        if isinstance(prices, pd.Series):
            prices_m = prices.resample('ME').last()
            prices_df_m = pd.DataFrame({'Close': prices_m})
        else:
            prices_df_m = prices.resample('ME').last()
        
        if isinstance(benchmark, pd.Series):
            bench_m = benchmark.resample('ME').last()
        else:
            bench_m = benchmark.resample('ME').last()
        
        spy_monthly = None
        if (use_roc_filter or use_sma_filter) and spy_data is not None:
            spy_series = spy_data.iloc[:, 0] if isinstance(spy_data, pd.DataFrame) else spy_data
            spy_monthly = spy_series.resample('ME').last()
        
        # Precalculos
        all_indicators = precalculate_all_indicators(prices_df_m, ohlc_data, corte)
        current_tickers = list(prices_df_m.columns)
        monthly_dates = prices_df_m.index[1:]  # exclude first
        
        if historical_info and 'changes_data' in historical_info:
            print("📅 Precalculando tickers válidos por fecha...")
            valid_tickers_by_date = precalculate_valid_tickers_by_date(
                monthly_dates,
                historical_info['changes_data'],
                current_tickers
            )
        else:
            valid_tickers_by_date = {date: set(current_tickers) for date in monthly_dates}
        
        # Inicializar
        equity = [10000.0]
        dates = [prices_df_m.index[0]]
        picks_list = []
        total_months = len(prices_df_m) - 1
        
        for i in range(1, len(prices_df_m)):
            try:
                prev_date = prices_df_m.index[i - 1]
                date = prices_df_m.index[i]
                
                if progress_callback and i % 10 == 0:
                    progress_callback(i / max(1, total_months))
                
                # Filtros de mercado
                market_filter_active = False
                if spy_monthly is not None and prev_date in spy_monthly.index:
                    spy_price = spy_monthly.loc[prev_date]
                    if use_roc_filter and len(spy_monthly[:prev_date]) >= 13:
                        spy_12m_ago = spy_monthly[:prev_date].iloc[-13]
                        spy_roc_12m = ((spy_price - spy_12m_ago) / spy_12m_ago) * 100
                        if spy_roc_12m < 0:
                            market_filter_active = True
                    if use_sma_filter and len(spy_monthly[:prev_date]) >= 10:
                        spy_sma_10m = spy_monthly[:prev_date].iloc[-10:].mean()
                        if spy_price < spy_sma_10m:
                            market_filter_active = True
                if market_filter_active:
                    equity.append(equity[-1])
                    dates.append(date)
                    continue
                
                # Tickers válidos
                valid_tickers_for_date = valid_tickers_by_date.get(prev_date, set(current_tickers))
                
                # Seleccionar candidatos
                candidates = []
                for ticker in valid_tickers_for_date:
                    if ticker not in all_indicators:
                        continue
                    indicators = all_indicators[ticker]
                    try:
                        if prev_date in indicators['InerciaAlcista'].index:
                            inercia = indicators['InerciaAlcista'].loc[prev_date]
                            score_adj = indicators['ScoreAdjusted'].loc[prev_date]
                            if inercia >= corte and score_adj > 0 and not np.isnan(score_adj):
                                candidates.append({'ticker': ticker, 'inercia': float(inercia), 'score_adj': float(score_adj)})
                    except Exception:
                        continue
                
                if not candidates:
                    equity.append(equity[-1])
                    dates.append(date)
                    continue
                
                candidates = sorted(candidates, key=lambda x: x['score_adj'], reverse=True)
                if fixed_allocation:
                    selected_picks = candidates[:10]
                else:
                    selected_picks = candidates[:top_n]
                
                selected_tickers = [p['ticker'] for p in selected_picks]
                
                # Precios
                available_prices = prices_df_m.loc[date]
                prev_prices = prices_df_m.loc[prev_date]
                
                valid_tickers = []
                for ticker in selected_tickers:
                    if (ticker in available_prices.index and
                        ticker in prev_prices.index and
                        not pd.isna(available_prices[ticker]) and
                        not pd.isna(prev_prices[ticker]) and
                        prev_prices[ticker] != 0):
                        valid_tickers.append(ticker)
                
                if not valid_tickers:
                    equity.append(equity[-1])
                    dates.append(date)
                    continue
                
                if fixed_allocation:
                    valid_tickers = valid_tickers[:10]
                    weight = 0.1
                else:
                    weight = 1.0 / len(valid_tickers)
                
                portfolio_return = 0.0
                for ticker in valid_tickers:
                    ret = (available_prices[ticker] / prev_prices[ticker]) - 1
                    portfolio_return += ret * weight
                
                portfolio_return -= commission
                new_equity = equity[-1] * (1 + portfolio_return)
                equity.append(new_equity)
                dates.append(date)
                
                # Guardar picks
                for rank, ticker in enumerate(valid_tickers, 1):
                    pick_data = next((p for p in selected_picks if p['ticker'] == ticker), None)
                    if pick_data:
                        picks_list.append({
                            "Date": date.strftime("%Y-%m-%d"),
                            "Rank": rank,
                            "Ticker": ticker,
                            "Inercia": pick_data['inercia'],
                            "ScoreAdj": pick_data['score_adj'],
                            "HistoricallyValid": ticker in valid_tickers_for_date
                        })
            except Exception as e:
                print(f"Error en mes {i}: {e}")
                equity.append(equity[-1])
                dates.append(date)
                continue
        
        # Resultados
        equity_series = pd.Series(equity, index=dates)
        returns = equity_series.pct_change().fillna(0)
        drawdown = (equity_series / equity_series.cummax() - 1).fillna(0)
        
        bt_results = pd.DataFrame({
            "Equity": equity_series,
            "Returns": returns,
            "Drawdown": drawdown
        })
        
        picks_df = pd.DataFrame(picks_list)
        
        print(f"✅ Backtest OPTIMIZADO completado. Equity final: ${equity_series.iloc[-1]:,.2f}")
        if historical_info and historical_info.get('has_historical_data', False):
            print("✅ Backtest ejecutado con verificación histórica de constituyentes")
        else:
            print("⚠️ Backtest ejecutado SIN verificación histórica (posible sesgo de supervivencia)")
        
        return bt_results, picks_df
    except Exception as e:
        print(f"❌ Error en backtest: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

def calculate_monthly_returns_by_year(equity_series):
    """Calcula retornos mensuales por año"""
    try:
        if equity_series is None or len(equity_series) < 2:
            return pd.DataFrame()
        monthly_returns = equity_series.pct_change().fillna(0)
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_by_year = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).apply(lambda x: (1 + x).prod() - 1)
        years = sorted(monthly_by_year.index.get_level_values(0).unique())
        months_es = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        table_data = []
        for year in years:
            year_data = {'Año': year}
            year_monthly = monthly_by_year[monthly_by_year.index.get_level_values(0) == year]
            for i, month_abbr in enumerate(months_es, 1):
                if i in year_monthly.index.get_level_values(1):
                    return_value = year_monthly[year_monthly.index.get_level_values(1) == i].iloc[0]
                    year_data[month_abbr] = f"{return_value*100:.1f}%"
                else:
                    year_data[month_abbr] = "-"
            year_equity = equity_series[equity_series.index.year == year]
            if len(year_equity) > 1:
                ytd_return = (year_equity.iloc[-1] / year_equity.iloc[0]) - 1
                year_data['YTD'] = f"{ytd_return*100:.1f}%"
            else:
                year_data['YTD'] = "-"
            table_data.append(year_data)
        if table_data:
            result_df = pd.DataFrame(table_data)
            columns_order = ['Año'] + months_es + ['YTD']
            result_df = result_df[columns_order]
            return result_df
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error calculando tabla de retornos: {e}")
        return pd.DataFrame()

# En la función calculate_sharpe_ratio (si existe), asegúrate de que use:
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calcula el Sharpe Ratio con tasa libre de riesgo del 2% anual
    """
    # Convertir tasa anual a mensual correctamente
    risk_free_rate_monthly = (1 + risk_free_rate) ** (1/12) - 1
    
    # Calcular exceso de retornos
    excess_returns = returns - risk_free_rate_monthly
    
    # Sharpe ratio anualizado
    if excess_returns.std() > 0:
        sharpe = (excess_returns.mean() * 12) / (excess_returns.std() * np.sqrt(12))
    else:
        sharpe = 0
    
    return sharpe

# Wrappers compatibilidad
def calcular_atr_amibroker(*args, **kwargs):
    return calcular_atr_optimizado(*args, **kwargs)

def run_backtest(*args, **kwargs):
    return run_backtest_optimized(*args, **kwargs)
