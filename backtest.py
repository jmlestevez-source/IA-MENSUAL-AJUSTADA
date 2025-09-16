import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calcular_atr_optimizado(high, low, close, periods=14):
    """
    Calcula ATR usando EWM (Exponential Weighted Mean) - MUCHO MÁS RÁPIDO
    """
    # True Range vectorizado
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    # True Range máximo
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR usando EWM (equivalente a Wilder's smoothing)
    # alpha = 1/periods para método de Wilder
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
            
            # Agregación mensual eficiente
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
    OPTIMIZACIÓN: Precalcula tickers válidos para todas las fechas de una vez
    """
    if historical_changes_data is None or historical_changes_data.empty:
        # Sin datos históricos, todos los tickers son válidos siempre
        return {date: set(current_tickers) for date in monthly_dates}
    
    # Convertir fechas
    historical_changes_data['Date'] = pd.to_datetime(historical_changes_data['Date']).dt.date
    
    valid_by_date = {}
    
    for target_date in monthly_dates:
        if isinstance(target_date, pd.Timestamp):
            target_date_clean = target_date.date()
        else:
            target_date_clean = target_date
        
        # Empezar con tickers actuales
        valid_tickers = set(current_tickers)
        
        # Revertir cambios futuros
        future_changes = historical_changes_data[historical_changes_data['Date'] > target_date_clean]
        
        for _, change in future_changes.iterrows():
            ticker = change['Ticker']
            action = change['Action']
            
            if action == 'Added':
                valid_tickers.discard(ticker)
            elif action == 'Removed':
                valid_tickers.add(ticker)
        
        valid_by_date[target_date] = valid_tickers
    
    return valid_by_date

def precalculate_all_indicators(prices_df_m, ohlc_data, corte=680):
    """
    OPTIMIZACIÓN CRÍTICA: Precalcula TODOS los indicadores de una vez
    """
    print("🚀 Precalculando todos los indicadores...")
    
    all_indicators = {}
    
    # Convertir OHLC a mensual si existe
    if ohlc_data:
        monthly_ohlc = convertir_a_mensual_con_ohlc(ohlc_data)
    else:
        monthly_ohlc = None
    
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
                # Estimar si no hay OHLC
                close = ticker_data
                if close.index.freq not in ['ME', 'M']:
                    close = close.resample('ME').last()
                
                monthly_returns = close.pct_change().dropna()
                vol = monthly_returns.rolling(3).std().fillna(0.02).clip(0.005, 0.03)
                high = close * (1 + vol * 0.5)
                low = close * (1 - vol * 0.5)
            
            # Limpiar datos
            close = close.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
            high = high.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
            low = low.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
            
            if close.isna().all():
                continue
            
            # CALCULAR INDICADORES VECTORIZADOS
            
            # ROC 10 meses
            roc_10 = ((close - close.shift(10)) / close.shift(10)) * 100
            
            # F1
            f1 = roc_10 * 0.6
            
            # ATR optimizado
            atr_14 = calcular_atr_optimizado(high, low, close, periods=14)
            
            # SMA
            sma_14 = close.rolling(14).mean()
            
            # F2
            volatility_ratio = atr_14 / sma_14
            f2 = volatility_ratio * 0.4
            
            # Inercia Alcista
            inercia_alcista = f1 / f2
            
            # Score
            score = np.where(inercia_alcista >= corte, inercia_alcista, 0)
            score = pd.Series(score, index=inercia_alcista.index)
            
            # Score Adjusted
            score_adjusted = score / atr_14
            
            # Limpiar infinitos
            inercia_alcista = inercia_alcista.replace([np.inf, -np.inf], np.nan).fillna(0)
            score = score.replace([np.inf, -np.inf], np.nan).fillna(0)
            score_adjusted = score_adjusted.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Almacenar todos los indicadores
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

def run_backtest_optimized(prices, benchmark, commission=0.003, top_n=10, corte=680, 
                          ohlc_data=None, historical_info=None, fixed_allocation=False,
                          use_roc_filter=False, use_sma_filter=False, spy_data=None,
                          progress_callback=None):
    """
    VERSIÓN OPTIMIZADA del backtest con precálculo
    """
    try:
        print("🚀 Iniciando backtest OPTIMIZADO...")
        
        # Validación inicial
        if prices is None or prices.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Mensualizar datos
        if isinstance(prices, pd.Series):
            prices_m = prices.resample('ME').last()
            prices_df_m = pd.DataFrame({'Close': prices_m})
        else:
            prices_df_m = prices.resample('ME').last()
        
        # Mensualizar benchmark
        if isinstance(benchmark, pd.Series):
            bench_m = benchmark.resample('ME').last()
        else:
            bench_m = benchmark.resample('ME').last()
        
        # SPY mensual para filtros
        spy_monthly = None
        if (use_roc_filter or use_sma_filter) and spy_data is not None:
            spy_series = spy_data.iloc[:, 0] if isinstance(spy_data, pd.DataFrame) else spy_data
            spy_monthly = spy_series.resample('ME').last()
        
        # OPTIMIZACIÓN 1: Precalcular indicadores
        all_indicators = precalculate_all_indicators(prices_df_m, ohlc_data, corte)
        
        # OPTIMIZACIÓN 2: Precalcular tickers válidos por fecha
        current_tickers = list(prices_df_m.columns)
        monthly_dates = prices_df_m.index[1:]  # Excluir primera fecha
        
        valid_tickers_by_date = {}
        if historical_info and 'changes_data' in historical_info:
            print("📅 Precalculando tickers válidos por fecha...")
            valid_tickers_by_date = precalculate_valid_tickers_by_date(
                monthly_dates, 
                historical_info['changes_data'], 
                current_tickers
            )
        else:
            # Sin verificación histórica, todos válidos siempre
            valid_tickers_by_date = {date: set(current_tickers) for date in monthly_dates}
        
        # Inicializar backtest
        equity = [10000]
        dates = [prices_df_m.index[0]]
        picks_list = []
        in_cash_by_filter = False
        
        total_months = len(prices_df_m) - 1
        
        # BUCLE PRINCIPAL OPTIMIZADO
        for i in range(1, len(prices_df_m)):
            try:
                prev_date = prices_df_m.index[i - 1]
                date = prices_df_m.index[i]
                
                # Progress callback
                if progress_callback and i % 10 == 0:
                    progress_callback(i / total_months)
                
                # Verificar filtros de mercado
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
                
                # Obtener tickers válidos (precalculados)
                valid_tickers_for_date = valid_tickers_by_date.get(prev_date, set(current_tickers))
                
                # Seleccionar mejores tickers usando indicadores precalculados
                candidates = []
                
                for ticker in valid_tickers_for_date:
                    if ticker not in all_indicators:
                        continue
                    
                    indicators = all_indicators[ticker]
                    
                    # Obtener último valor de cada indicador
                    if prev_date in indicators['InerciaAlcista'].index:
                        inercia = indicators['InerciaAlcista'].loc[prev_date]
                        score_adj = indicators['ScoreAdjusted'].loc[prev_date]
                        
                        if inercia >= corte and score_adj > 0 and not np.isnan(score_adj):
                            candidates.append({
                                'ticker': ticker,
                                'inercia': inercia,
                                'score_adj': score_adj
                            })
                
                if not candidates:
                    equity.append(equity[-1])
                    dates.append(date)
                    continue
                
                # Ordenar y seleccionar top N
                candidates = sorted(candidates, key=lambda x: x['score_adj'], reverse=True)
                
                if fixed_allocation:
                    selected_picks = candidates[:10]
                    weight = 0.1
                else:
                    selected_picks = candidates[:top_n]
                    weight = 1.0 / len(selected_picks) if selected_picks else 0
                
                # Calcular retornos
                selected_tickers = [p['ticker'] for p in selected_picks]
                
                # Obtener precios
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
                
                # Recalcular peso si es necesario
                if fixed_allocation:
                    valid_tickers = valid_tickers[:10]
                    weight = 0.1
                else:
                    weight = 1.0 / len(valid_tickers)
                
                # Calcular retorno del portafolio
                portfolio_return = 0
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
        
        # Crear resultados
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

# Mantener función original para compatibilidad
def run_backtest(*args, **kwargs):
    """Wrapper para compatibilidad"""
    return run_backtest_optimized(*args, **kwargs)

def calculate_monthly_returns_by_year(equity_series):
    """Calcula retornos mensuales por año"""
    try:
        if equity_series is None or len(equity_series) < 2:
            return pd.DataFrame()
        
        monthly_returns = equity_series.pct_change().fillna(0)
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        
        monthly_by_year = monthly_returns.groupby([
            monthly_returns.index.year, 
            monthly_returns.index.month
        ]).apply(lambda x: (1 + x).prod() - 1)
        
        years = sorted(monthly_by_year.index.get_level_values(0).unique())
        months_es = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        
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

# Funciones auxiliares para compatibilidad
def calcular_atr_amibroker(*args, **kwargs):
    """Wrapper para compatibilidad"""
    return calcular_atr_optimizado(*args, **kwargs)

def inertia_score(*args, **kwargs):
    """Esta función ya no se usa, se reemplaza por precalculate_all_indicators"""
    return {}
