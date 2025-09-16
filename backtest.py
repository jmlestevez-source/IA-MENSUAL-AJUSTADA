import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calcular_atr_amibroker(high, low, close, periods=14):
    """
    Calcula el ATR usando EWM (equivalente al método de Wilder) - MUCHO MÁS RÁPIDO
    """
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/periods, adjust=False).mean()
    return atr


def convertir_a_mensual_con_ohlc(ohlc_data):
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
            print(f"Error convirtiendo {ticker} a mensual: {e}")
            continue
    return monthly_data


def get_valid_tickers_for_date(target_date, historical_changes_data, current_tickers):
    if historical_changes_data is None or historical_changes_data.empty:
        return set(current_tickers)
   
    if isinstance(target_date, pd.Timestamp):
        target_date = target_date.date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()

    valid_tickers = set(current_tickers)
    historical_changes_data['Date'] = pd.to_datetime(historical_changes_data['Date']).dt.date
    future_changes = historical_changes_data[historical_changes_data['Date'] > target_date]
    future_changes = future_changes.sort_values('Date', ascending=True)

    for _, change in future_changes.iterrows():
        ticker = change['Ticker']
        action = change['Action']
        if action == 'Added':
            valid_tickers.discard(ticker)
        elif action == 'Removed':
            valid_tickers.add(ticker)

    return valid_tickers


def inertia_score_with_historical_filter(monthly_prices_df, target_date, valid_tickers, corte=680, ohlc_data=None):
    if monthly_prices_df is None or monthly_prices_df.empty:
        return {}

    available_tickers = set(monthly_prices_df.columns)
    tickers_to_use = list(available_tickers.intersection(valid_tickers))
    if not tickers_to_use:
        return {}

    filtered_prices = monthly_prices_df[tickers_to_use]
    filtered_ohlc = None
    if ohlc_data:
        filtered_ohlc = {ticker: data for ticker, data in ohlc_data.items() if ticker in tickers_to_use}

    return inertia_score(filtered_prices, corte=corte, ohlc_data=filtered_ohlc)


def inertia_score(monthly_prices_df, corte=680, ohlc_data=None, end_date=None):
    if monthly_prices_df is None or monthly_prices_df.empty:
        return {}

    if end_date is not None:
        monthly_prices_df = monthly_prices_df.loc[:end_date]
        if ohlc_data:
            ohlc_data = {
                ticker: {k: v.loc[:end_date] for k, v in data.items()}
                for ticker, data in ohlc_data.items()
            }

    try:
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
                    if len(close) < 15:
                        continue
                    if close.index.freq != 'ME' and close.index.freq != 'M':
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
                atr_14 = calcular_atr_amibroker(high, low, close, periods=14)
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
                if metric in results[ticker]:
                    metric_data[ticker] = results[ticker][metric]
            if metric_data:
                combined_results[metric] = pd.DataFrame(metric_data)

        return combined_results

    except Exception as e:
        print(f"Error en cálculo de inercia: {e}")
        return {}


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    if len(returns) < 2:
        return 0.0
    risk_free_rate_monthly = risk_free_rate / 12
    excess_returns = returns - risk_free_rate_monthly
    excess_returns = excess_returns.dropna()
    if len(excess_returns) < 2:
        return 0.0
    std_excess = excess_returns.std()
    if std_excess == 0 or np.isnan(std_excess) or np.isinf(std_excess):
        return 0.0
    mean_excess = excess_returns.mean()
    if np.isnan(mean_excess) or np.isinf(mean_excess):
        return 0.0
    sharpe_ratio = (mean_excess * 12) / (std_excess * (12 ** 0.5))
    if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
        return 0.0
    return sharpe_ratio


def calculate_monthly_returns_by_year(equity_series):
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
        print(f"Error calculando tabla de retornos mensuales: {e}")
        return pd.DataFrame()


def run_backtest(prices, benchmark, commission=0.003, top_n=10, corte=680, ohlc_data=None,
                 historical_info=None, fixed_allocation=False, use_roc_filter=False,
                 use_sma_filter=False, spy_data=None):
    try:
        print("🚀 Iniciando backtest optimizado...")
        if prices is None or (hasattr(prices, 'empty') and prices.empty):
            empty_bt = pd.DataFrame(columns=["Equity", "Returns", "Drawdown"])
            empty_picks = pd.DataFrame(columns=["Date", "Rank", "Ticker", "Inercia", "ScoreAdj"])
            return empty_bt, empty_picks

        if isinstance(prices, pd.Series):
            prices_m = prices.resample('ME').last()
            prices_df_m = pd.DataFrame({'Close': prices_m})
        else:
            try:
                prices_m = prices.resample('ME').last()
                prices_df_m = prices_m.copy()
            except Exception:
                prices_df_m = prices.copy()

        try:
            if isinstance(benchmark, pd.Series):
                bench_m = benchmark.resample('ME').last()
            else:
                bench_m = benchmark.resample('ME').last()
        except:
            bench_m = benchmark

        if prices_df_m.empty:
            raise ValueError("No hay datos mensuales suficientes para el backtest")

        # Preparar SPY
        spy_monthly = None
        if (use_roc_filter or use_sma_filter) and spy_data is not None:
            try:
                spy_series = spy_data.iloc[:, 0] if isinstance(spy_data, pd.DataFrame) else spy_data
                spy_monthly = spy_series.resample('ME').last()
                print("✅ Datos del SPY preparados para filtros")
            except Exception as e:
                print(f"⚠️ Error preparando SPY: {e}")
                use_roc_filter = False
                use_sma_filter = False

        historical_changes = None
        current_tickers = list(prices_df_m.columns)
        if historical_info and 'changes_data' in historical_info:
            historical_changes = historical_info['changes_data']
            print(f"✅ Usando datos históricos ({len(historical_changes)} cambios)")
        else:
            print("⚠️ Sin datos históricos, usando todos los tickers")

        # 🚀 PRECÁLCULO 1: Tickers válidos por fecha
        valid_tickers_by_date = {}
        if historical_changes is not None:
            print("📅 Precalculando tickers válidos por fecha...")
            all_dates = prices_df_m.index
            for target_date in all_dates:
                valid_tickers = get_valid_tickers_for_date(target_date, historical_changes, current_tickers)
                valid_tickers_by_date[target_date] = valid_tickers
            print("✅ Precálculo de tickers válido completado")

        # 🚀 PRECÁLCULO 2: Indicadores para todos los tickers
        print("📊 Precalculando indicadores técnicos...")
        all_scores = {}
        all_ohlc_monthly = convertir_a_mensual_con_ohlc(ohlc_data) if ohlc_data else None

        for ticker in current_tickers:
            try:
                ticker_data = prices_df_m[ticker].dropna()
                if len(ticker_data) < 15:
                    continue
                ohlc_ticker = None
                if all_ohlc_monthly and ticker in all_ohlc_monthly:
                    ohlc_ticker = all_ohlc_monthly[ticker]
                df_ticker = pd.DataFrame({ticker: ticker_data})
                scores = inertia_score(df_ticker, corte=corte, ohlc_data={ticker: ohlc_ticker} if ohlc_ticker else None)
                all_scores[ticker] = scores
            except Exception as e:
                print(f"❌ Error precalculando {ticker}: {e}")
                continue

        print(f"✅ Precálculo completado para {len(all_scores)} tickers")

        # Backtest loop
        equity = [10000]
        dates = [prices_df_m.index[0]] if len(prices_df_m.index) > 0 else []
        picks_list = []
        in_cash_by_filter = False

        print(f"▶️ Iniciando simulación: {len(prices_df_m)} meses | Fixed allocation: {fixed_allocation}")

        for i in range(1, len(prices_df_m)):
            try:
                prev_date = prices_df_m.index[i - 1]
                date = prices_df_m.index[i]

                # Log progresivo
                if i % 24 == 0:  # Cada 2 años
                    print(f"📈 {date.strftime('%Y-%m')} | Equity: ${equity[-1]:,.0f} | Picks activos: {len(picks_list) if picks_list else 0}")

                # Filtros de mercado
                market_filter_active = False
                filter_reasons = []
                if spy_monthly is not None and prev_date in spy_monthly.index:
                    spy_price = spy_monthly.loc[prev_date]
                    if use_roc_filter and len(spy_monthly[:prev_date]) >= 13:
                        spy_12m_ago = spy_monthly[:prev_date].iloc[-13]
                        spy_roc_12m = ((spy_price - spy_12m_ago) / spy_12m_ago) * 100
                        if spy_roc_12m < 0:
                            market_filter_active = True
                            filter_reasons.append(f"ROC 12M SPY: {spy_roc_12m:.2f}% < 0")
                    if use_sma_filter and len(spy_monthly[:prev_date]) >= 10:
                        spy_sma_10m = spy_monthly[:prev_date].iloc[-10:].mean()
                        if spy_price < spy_sma_10m:
                            market_filter_active = True
                            filter_reasons.append(f"SPY ${spy_price:.2f} < SMA10 ${spy_sma_10m:.2f}")

                if market_filter_active:
                    if not in_cash_by_filter:
                        print(f"🛡️ {prev_date.strftime('%Y-%m-%d')}: Filtros activados - {', '.join(filter_reasons)}")
                        in_cash_by_filter = True
                    equity.append(equity[-1])
                    dates.append(date)
                    continue
                else:
                    if in_cash_by_filter:
                        print(f"✅ {prev_date.strftime('%Y-%m-%d')}: Filtros desactivados")
                        in_cash_by_filter = False

                # Validar tickers históricos
                valid_tickers_for_date = valid_tickers_by_date.get(prev_date, set(current_tickers)) if valid_tickers_by_date else set(current_tickers)
                available_valid_tickers = list(set(prices_df_m.columns).intersection(valid_tickers_for_date))
                if not available_valid_tickers:
                    equity.append(equity[-1])
                    dates.append(date)
                    continue

                # Seleccionar picks usando scores precalculados
                selected_picks = []
                for ticker in available_valid_tickers:
                    if ticker in all_scores and "ScoreAdjusted" in all_scores[ticker] and "InerciaAlcista" in all_scores[ticker]:
                        score_adj_series = all_scores[ticker]["ScoreAdjusted"].loc[:prev_date]
                        inercia_series = all_scores[ticker]["InerciaAlcista"].loc[:prev_date]
                        if len(score_adj_series) > 0 and len(inercia_series) > 0:
                            last_score = score_adj_series.iloc[-1]
                            last_inercia = inercia_series.iloc[-1]
                            if last_inercia >= corte and last_score > 0:
                                selected_picks.append({
                                    'ticker': ticker,
                                    'inercia': last_inercia,
                                    'score_adj': last_score
                                })

                if not selected_picks:
                    equity.append(equity[-1])
                    dates.append(date)
                    continue

                # Ordenar y seleccionar top N
                selected_picks = sorted(selected_picks, key=lambda x: x['score_adj'], reverse=True)
                if fixed_allocation:
                    selected_picks = selected_picks[:10]
                    selected = [pick['ticker'] for pick in selected_picks]
                    weight = 0.1
                else:
                    selected_picks = selected_picks[:min(top_n, len(selected_picks))]
                    selected = [pick['ticker'] for pick in selected_picks]
                    weight = 1.0 / len(selected)

                # Calcular retorno
                try:
                    available_prices = prices_df_m.loc[date]
                    prev_prices = prices_df_m.loc[prev_date]
                except Exception as e:
                    equity.append(equity[-1])
                    dates.append(date)
                    continue

                valid_tickers = []
                for ticker in selected:
                    try:
                        if (ticker in available_prices.index and ticker in prev_prices.index and
                            not pd.isna(available_prices[ticker]) and not pd.isna(prev_prices[ticker]) and
                            prev_prices[ticker] != 0):
                            valid_tickers.append(ticker)
                    except:
                        continue

                if len(valid_tickers) == 0:
                    equity.append(equity[-1])
                    dates.append(date)
                    continue

                if fixed_allocation:
                    weight = 0.1
                    valid_tickers = valid_tickers[:10]
                else:
                    weight = 1.0 / len(valid_tickers)

                rets = pd.Series(dtype=float)
                for ticker in valid_tickers:
                    try:
                        prev_price = prev_prices[ticker]
                        curr_price = available_prices[ticker]
                        if prev_price != 0:
                            ret_value = (curr_price / prev_price) - 1
                            rets[ticker] = ret_value if np.isfinite(ret_value) else 0
                        else:
                            rets[ticker] = 0
                    except:
                        rets[ticker] = 0

                rets = rets.fillna(0)
                port_ret = (rets * weight).sum() - commission
                new_eq = equity[-1] * (1 + port_ret) if np.isfinite(port_ret) else equity[-1]

                equity.append(new_eq)
                dates.append(date)

                # Guardar picks
                for rank, ticker in enumerate(valid_tickers, 1):
                    try:
                        inercia_val = next((p['inercia'] for p in selected_picks if p['ticker'] == ticker), 0)
                        score_adj_val = next((p['score_adj'] for p in selected_picks if p['ticker'] == ticker), 0)
                        if inercia_val >= corte and score_adj_val > 0:
                            picks_list.append({
                                "Date": date.strftime("%Y-%m-%d"),
                                "Rank": rank,
                                "Ticker": str(ticker),
                                "Inercia": float(inercia_val),
                                "ScoreAdj": float(score_adj_val),
                                "HistoricallyValid": ticker in valid_tickers_for_date
                            })
                    except Exception as e:
                        continue

            except Exception as e:
                print(f"⚠️ Error en iteración {i} ({date}): {e}")
                equity.append(equity[-1])
                dates.append(date)
                continue

        if len(equity) <= 1:
            raise ValueError("No se generaron resultados")

        equity_series = pd.Series(equity, index=dates)
        returns = equity_series.pct_change().fillna(0)
        drawdown = (equity_series / equity_series.cummax() - 1).fillna(0)

        bt = pd.DataFrame({"Equity": equity_series, "Returns": returns, "Drawdown": drawdown})
        picks_df = pd.DataFrame(picks_list)

        print(f"🎉 Backtest completado! Equity final: ${equity_series.iloc[-1]:,.2f}")
        if not picks_df.empty:
            total_picks = len(picks_df)
            valid_picks = picks_df['HistoricallyValid'].sum()
            print(f"📊 Picks válidos históricamente: {valid_picks}/{total_picks} ({valid_picks/total_picks*100:.1f}%)")

        return bt, picks_df

    except Exception as e:
        print(f"❌ Error crítico en run_backtest: {e}")
        import traceback
        traceback.print_exc()
        empty_bt = pd.DataFrame(columns=["Equity", "Returns", "Drawdown"])
        empty_picks = pd.DataFrame(columns=["Date", "Rank", "Ticker", "Inercia", "ScoreAdj"])
        return empty_bt, empty_picks


def monthly_true_range(high, low, close):
    return calcular_atr_amibroker(high, low, close, periods=1)
