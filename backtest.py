# backtest.py
import pandas as pd
import numpy as np
import os
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
    VERSIÓN CORREGIDA: Filtra correctamente los tickers según cambios históricos
    """
    if historical_changes_data is None or historical_changes_data.empty:
        print("⚠️ No hay datos históricos, usando todos los tickers actuales")
        return {date: set(current_tickers) for date in monthly_dates}
    
    print(f"📊 Procesando {len(historical_changes_data)} cambios históricos")
    
    historical_changes_data = historical_changes_data.copy()
    historical_changes_data['Date'] = pd.to_datetime(historical_changes_data['Date'])
    
    all_historical_tickers = set()
    all_historical_tickers.update(current_tickers)
    
    removed_tickers = historical_changes_data[
        historical_changes_data['Action'].str.lower() == 'removed'
    ]['Ticker'].unique()
    all_historical_tickers.update(removed_tickers)
    
    print(f"📊 Total de tickers históricos únicos: {len(all_historical_tickers)}")
    
    valid_by_date = {}
    for target_date in monthly_dates:
        target_dt = target_date if isinstance(target_date, pd.Timestamp) else pd.Timestamp(target_date)
        valid_tickers = set(current_tickers)
        all_changes = historical_changes_data
        for ticker in all_historical_tickers:
            ticker_changes = all_changes[all_changes['Ticker'] == ticker].sort_values('Date')
            if len(ticker_changes) == 0:
                if ticker not in current_tickers:
                    valid_tickers.discard(ticker)
                continue
            changes_before = ticker_changes[ticker_changes['Date'] <= target_dt]
            if len(changes_before) > 0:
                last_action = str(changes_before.iloc[-1]['Action']).lower()
                if last_action == 'added':
                    valid_tickers.add(ticker)
                elif last_action == 'removed':
                    valid_tickers.discard(ticker)
            else:
                changes_after = ticker_changes[ticker_changes['Date'] > target_dt]
                if len(changes_after) > 0:
                    first_future_action = str(changes_after.iloc[0]['Action']).lower()
                    if first_future_action == 'added':
                        valid_tickers.discard(ticker)
                    elif first_future_action == 'removed':
                        valid_tickers.add(ticker)
        valid_by_date[target_date] = valid_tickers
    print(f"✅ Tickers válidos calculados para {len(valid_by_date)} fechas")
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
                close = close.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                high = high.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                low = low.replace([np.inf, -np.inf], np.nan).ffill().bfill()
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
                
                close = close.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                high = high.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                low = low.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                
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

# Helpers NUEVOS para fallback IEF/BIL
def _load_monthly_from_csv(ticker):
    """Carga precio mensual (Adj Close/Close) desde data/{ticker}.csv si existe."""
    path = os.path.join("data", f"{ticker}.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        if 'Adj Close' in df.columns:
            s = df['Adj Close']
        elif 'Close' in df.columns:
            s = df['Close']
        else:
            return pd.Series(dtype=float)
        return s.resample('ME').last().dropna()
    except Exception:
        return pd.Series(dtype=float)

def _momentum_13612w(price_m):
    """Calcula 13612W: (12*r1 + 4*r3 + 2*r6 + 1*r12)/4 con r_t = p0/pt - 1."""
    if price_m is None or price_m.empty:
        return pd.Series(dtype=float)
    r1 = price_m / price_m.shift(1) - 1
    r3 = price_m / price_m.shift(3) - 1
    r6 = price_m / price_m.shift(6) - 1
    r12 = price_m / price_m.shift(12) - 1
    score = (12*r1 + 4*r3 + 2*r6 + 1*r12) / 4.0
    return score

def run_backtest_optimized(prices, benchmark, commission=0.003, top_n=10, corte=680, 
                          ohlc_data=None, historical_info=None, fixed_allocation=False,
                          use_roc_filter=False, use_sma_filter=False, spy_data=None,
                          use_risk_off_fallback=False, event_rebalance=False,
                          progress_callback=None):
    """
    VERSIÓN OPTIMIZADA del backtest con dos mejoras opcionales:
    - use_risk_off_fallback: si filtros activan risk-off, invierte 100% en ganador IEF/BIL (13612W); si no hay datos, cash
    - event_rebalance: mantener ganadoras; comisión por turnover real
    Si ambos están en False, comportamiento original (comisión plana mensual).
    Retorna: bt_results (DataFrame), picks_df (DataFrame)
    """
    try:
        print("🚀 Iniciando backtest OPTIMIZADO...")
        if prices is None or prices.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        prices_df_m = prices.resample('ME').last() if not isinstance(prices, pd.Series) else pd.DataFrame({'Close': prices.resample('ME').last()})
        bench_m = benchmark.resample('ME').last() if not isinstance(benchmark, pd.Series) else benchmark.resample('ME').last()
        
        # SPY mensual (para filtros)
        spy_monthly = None
        if (use_roc_filter or use_sma_filter) and spy_data is not None:
            spy_series = spy_data.iloc[:, 0] if isinstance(spy_data, pd.DataFrame) else spy_data
            spy_monthly = spy_series.resample('ME').last().dropna()
        
        # Fallback IEF/BIL si se activa el toggle
        ief_m = _load_monthly_from_csv("IEF") if use_risk_off_fallback else pd.Series(dtype=float)
        bil_m = _load_monthly_from_csv("BIL") if use_risk_off_fallback else pd.Series(dtype=float)
        ief_score = _momentum_13612w(ief_m) if not ief_m.empty else pd.Series(dtype=float)
        bil_score = _momentum_13612w(bil_m) if not bil_m.empty else pd.Series(dtype=float)
        
        # Indicadores y universo válido
        all_indicators = precalculate_all_indicators(prices_df_m, ohlc_data, corte)
        current_tickers = list(prices_df_m.columns)
        monthly_dates = prices_df_m.index[1:]
        
        if historical_info and 'changes_data' in historical_info:
            valid_tickers_by_date = precalculate_valid_tickers_by_date(
                monthly_dates,
                historical_info['changes_data'],
                current_tickers
            )
        else:
            valid_tickers_by_date = {date: set(current_tickers) for date in monthly_dates}
        
        equity = [10000.0]
        dates = [prices_df_m.index[0]]
        picks_list = []
        total_months = len(prices_df_m) - 1
        
        # Estado para event rebalance
        holdings = {}      # pesos de inicio de mes
        returns_history = []
        
        for i in range(1, len(prices_df_m)):
            try:
                prev_date = prices_df_m.index[i - 1]
                date = prices_df_m.index[i]
                
                if progress_callback and i % 10 == 0:
                    progress_callback(i / max(1, total_months))
                
                # Filtros de mercado (si se activan)
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
                
                # Risk-off fallback
                if market_filter_active and use_risk_off_fallback:
                    choice = None
                    if prev_date in ief_score.index and prev_date in bil_score.index:
                        choice = 'IEF' if ief_score.loc[prev_date] >= bil_score.loc[prev_date] else 'BIL'
                    elif prev_date in ief_score.index:
                        choice = 'IEF'
                    elif prev_date in bil_score.index:
                        choice = 'BIL'
                    
                    if choice in ['IEF', 'BIL']:
                        ser = ief_m if choice == 'IEF' else bil_m
                        if prev_date in ser.index and date in ser.index:
                            ret_fallback = (ser.loc[date] / ser.loc[prev_date]) - 1
                            commission_cost = 0.0
                            if event_rebalance:
                                # si cambiamos totalmente a fallback desde otra cosa, turnover ~ 1
                                if not (len(holdings) == 1 and choice in holdings and abs(holdings[choice]-1.0) < 1e-9):
                                    commission_cost = commission * 1.0
                                holdings = {choice: 1.0}
                            
                            portfolio_return = ret_fallback - commission_cost
                            new_equity = equity[-1] * (1 + portfolio_return)
                            equity.append(new_equity)
                            dates.append(date)
                            returns_history.append(portfolio_return)
                            picks_list.append({
                                "Date": prev_date.strftime("%Y-%m-%d"),
                                "Rank": 1,
                                "Ticker": choice,
                                "Inercia": np.nan,
                                "ScoreAdj": np.nan,
                                "HistoricallyValid": True
                            })
                            continue
                        else:
                            equity.append(equity[-1])
                            dates.append(date)
                            returns_history.append(0.0)
                            continue
                    else:
                        equity.append(equity[-1])
                        dates.append(date)
                        returns_history.append(0.0)
                        continue
                
                # Selección normal de candidatos
                valid_tickers_for_date = valid_tickers_by_date.get(prev_date, set(current_tickers))
                
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
                    returns_history.append(0.0)
                    continue
                
                candidates = sorted(candidates, key=lambda x: x['score_adj'], reverse=True)
                if fixed_allocation:
                    selected_picks = candidates[:10]
                else:
                    selected_picks = candidates[:top_n]
                
                selected_tickers = [p['ticker'] for p in selected_picks]
                
                available_prices = prices_df_m.loc[date]
                prev_prices = prices_df_m.loc[prev_date]
                
                ret_map = {}
                for ticker in selected_tickers:
                    if (ticker in available_prices.index and
                        ticker in prev_prices.index and
                        not pd.isna(available_prices[ticker]) and
                        not pd.isna(prev_prices[ticker]) and
                        prev_prices[ticker] != 0):
                        ret_map[ticker] = (available_prices[ticker] / prev_prices[ticker]) - 1
                
                if not ret_map:
                    equity.append(equity[-1])
                    dates.append(date)
                    returns_history.append(0.0)
                    continue
                
                # Pesos objetivo (equiponderado como original)
                n = len(ret_map)
                weights_target = {t: (0.1 if fixed_allocation else 1.0/n) for t in ret_map.keys()}
                
                if event_rebalance:
                    old_holdings = holdings.copy()
                    keep = {t: old_holdings.get(t, 0.0) for t in ret_map.keys() if t in old_holdings}
                    keep_total = sum(keep.values())
                    freed = 1.0 - keep_total
                    entrants = [t for t in ret_map.keys() if t not in old_holdings]
                    add_w = {}
                    if entrants and freed > 1e-12:
                        add_w = {t: freed/len(entrants) for t in entrants}
                    new_holdings = keep.copy()
                    for t, w in add_w.items():
                        new_holdings[t] = new_holdings.get(t, 0.0) + w
                    ssum = sum(new_holdings.values())
                    if ssum > 0:
                        new_holdings = {t: w/ssum for t, w in new_holdings.items()}
                    else:
                        new_holdings = weights_target.copy()
                    
                    # Turnover y comisión
                    all_t = set(old_holdings.keys()).union(new_holdings.keys())
                    turnover = sum(abs(new_holdings.get(t, 0.0) - old_holdings.get(t, 0.0)) for t in all_t)
                    commission_cost = commission * turnover
                    
                    portfolio_return = sum(new_holdings[t]*ret_map.get(t, 0.0) for t in new_holdings.keys()) - commission_cost
                    new_equity = equity[-1] * (1 + portfolio_return)
                    
                    # Drift de fin de mes
                    post_vals = {t: new_holdings[t]*(1 + ret_map.get(t, 0.0)) for t in new_holdings.keys()}
                    total_val = sum(post_vals.values())
                    holdings = {t: v/total_val for t, v in post_vals.items()} if total_val > 0 else {}
                    
                    equity.append(new_equity)
                    dates.append(date)
                    returns_history.append(portfolio_return)
                else:
                    # Comisión plana como el original
                    portfolio_return = sum(weights_target[t]*ret_map[t] for t in weights_target.keys()) - commission
                    new_equity = equity[-1] * (1 + portfolio_return)
                    equity.append(new_equity)
                    dates.append(date)
                    returns_history.append(portfolio_return)
                
                # Guardar picks
                for rank, t in enumerate(ret_map.keys(), 1):
                    pick_data = next((p for p in selected_picks if p['ticker'] == t), None)
                    if pick_data:
                        picks_list.append({
                            "Date": prev_date.strftime("%Y-%m-%d"),
                            "Rank": rank,
                            "Ticker": t,
                            "Inercia": pick_data['inercia'],
                            "ScoreAdj": pick_data['score_adj'],
                            "HistoricallyValid": t in valid_tickers_for_date
                        })
            except Exception as e:
                print(f"Error en mes {i} ({date}): {e}")
                equity.append(equity[-1])
                dates.append(date)
                returns_history.append(0.0)
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

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calcula el Sharpe Ratio con tasa libre de riesgo del 2% anual
    """
    risk_free_rate_monthly = (1 + risk_free_rate) ** (1/12) - 1
    excess_returns = returns - risk_free_rate_monthly
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
