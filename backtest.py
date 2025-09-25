# -*- coding: utf-8 -*-
# backtest.py (con warm-up de 15 meses y recorte desde la fecha de inicio)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings('ignore')

# Importamos para poder recargar precios con warm-up
from data_loader import download_prices

def calcular_atr_optimizado(high, low, close, periods=14):
    """
    ATR suavizado con EWM (no necesita 14 meses completos).
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/periods, adjust=False).mean()
    return atr

def convertir_a_mensual_con_ohlc(ohlc_data):
    """
    Convierte datos diarios OHLC a mensuales (max High, min Low, último Close).
    """
    monthly_data = {}
    for ticker, data in (ohlc_data or {}).items():
        try:
            df = pd.DataFrame({
                'High': data['High'],
                'Low': data['Low'],
                'Close': data['Close']
            }).dropna(how='all')
            if df.empty:
                continue
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
        except Exception:
            continue
    return monthly_data

def _adaptive_roc(close, max_period=10, min_period=3):
    """
    ROC adaptativo: arranca con min_period y escala hasta max_period según haya datos.
    """
    close = close.copy().astype(float)
    roc = pd.Series(index=close.index, dtype=float)
    for i in range(len(close)):
        if i >= min_period:
            n = min(max_period, i)
            prev = close.iloc[i - n]
            cur = close.iloc[i]
            if pd.notna(prev) and prev != 0 and pd.notna(cur):
                roc.iloc[i] = ((cur - prev) / prev) * 100.0
            else:
                roc.iloc[i] = np.nan
        else:
            roc.iloc[i] = np.nan
    return roc

def precalculate_valid_tickers_by_date(monthly_dates, historical_changes_data, current_tickers):
    """
    Validación EXACTA por fecha (regla 'último evento <= fecha manda'):
    - Universo SIEMPRE = current_tickers (constituyentes actuales hoy).
    - Para cada fecha y ticker actual:
        * Si NO aparece en changes: se asume histórico -> incluido.
        * Si aparece en changes: el ÚLTIMO evento con fecha <= fecha manda:
            - Added -> incluido
            - Removed -> excluido
        * Si no hay evento <= fecha (sólo futuros) -> excluido.
    """
    def _norm_t(t):
        return str(t).strip().upper().replace('.', '-')
    current_set = set(_norm_t(t) for t in current_tickers if t)

    if historical_changes_data is None or historical_changes_data.empty:
        return {dt: set(current_set) for dt in monthly_dates}

    ch = historical_changes_data.copy()
    if 'Date' not in ch.columns or 'Ticker' not in ch.columns or 'Action' not in ch.columns:
        return {dt: set(current_set) for dt in monthly_dates}

    ch['Ticker'] = ch['Ticker'].astype(str).str.upper().str.replace('.', '-', regex=False)
    ch['Date'] = pd.to_datetime(ch['Date'], errors='coerce')
    ch = ch.dropna(subset=['Date', 'Ticker', 'Action'])
    ch['Action'] = ch['Action'].astype(str).str.lower()

    grouped = ch.groupby('Ticker')

    valid_by_date = {}
    for target_date in monthly_dates:
        td = pd.to_datetime(target_date).normalize()
        valids = set()
        for t in current_set:
            if t not in grouped.groups:
                valids.add(t)
                continue
            tdf = grouped.get_group(t)
            tdf_before = tdf[tdf['Date'] <= td]
            if tdf_before.empty:
                continue
            last_action = str(tdf_before.sort_values('Date').iloc[-1]['Action']).lower()
            if 'add' in last_action:
                valids.add(t)
        valid_by_date[target_date] = valids

    return valid_by_date

def precalculate_all_indicators(prices_df_m, ohlc_data, corte=680):
    """
    Precalcula indicadores con ventanas adaptativas.
    También guarda RawScoreAdjusted = Inercia/ATR (sin corte) por si quisieras un fallback.
    """
    try:
        all_indicators = {}
        monthly_ohlc = convertir_a_mensual_con_ohlc(ohlc_data) if ohlc_data else None

        for ticker in prices_df_m.columns:
            try:
                ticker_data = prices_df_m[ticker].dropna()
                if ticker_data.empty:
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
                    vol = monthly_returns.rolling(3, min_periods=1).std().fillna(0.02).clip(0.005, 0.03)
                    high = close * (1 + vol * 0.5)
                    low = close * (1 - vol * 0.5)

                close = close.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                high = high.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                low = low.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                if close.isna().all() or high.isna().all() or low.isna().all():
                    continue

                roc_10 = _adaptive_roc(close, max_period=10, min_period=3)
                f1 = roc_10 * 0.6

                atr_14 = calcular_atr_optimizado(high, low, close, periods=14)
                sma_14 = close.rolling(14, min_periods=3).mean()
                volatility_ratio = atr_14 / sma_14

                f2 = volatility_ratio * 0.4
                inercia_alcista = f1 / f2

                raw_score_adj = inercia_alcista / atr_14

                score = pd.Series(np.where(inercia_alcista >= corte, inercia_alcista, 0), index=inercia_alcista.index)
                score_adjusted = score / atr_14

                for s in (inercia_alcista, score, score_adjusted, raw_score_adj, atr_14, roc_10):
                    s.replace([np.inf, -np.inf], np.nan, inplace=True)

                inercia_alcista = inercia_alcista.fillna(0)
                score = score.fillna(0)
                score_adjusted = score_adjusted.fillna(0)
                raw_score_adj = raw_score_adj.fillna(0)

                all_indicators[ticker] = {
                    'InerciaAlcista': inercia_alcista,
                    'Score': score,
                    'ScoreAdjusted': score_adjusted,
                    'RawScoreAdjusted': raw_score_adj,
                    'ATR14': atr_14,
                    'ROC10': roc_10
                }
            except Exception:
                continue
        return all_indicators
    except Exception:
        return {}

def inertia_score(monthly_prices_df, corte=680, ohlc_data=None):
    """
    Score de inercia (adaptativo) por ticker.
    """
    try:
        if monthly_prices_df is None or monthly_prices_df.empty:
            return {}
        results = {}
        monthly_ohlc = convertir_a_mensual_con_ohlc(ohlc_data) if ohlc_data else None

        for ticker in monthly_prices_df.columns:
            try:
                ticker_data = monthly_prices_df[ticker].dropna()
                if ticker_data.empty:
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
                    vol = monthly_returns.rolling(3, min_periods=1).std().fillna(0.02).clip(0.005, 0.03)
                    high = close * (1 + vol * 0.5)
                    low = close * (1 - vol * 0.5)
                    high = pd.Series(np.maximum(high, close), index=close.index)
                    low = pd.Series(np.minimum(low, close), index=close.index)

                close = close.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                high = high.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                low = low.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                if close.isna().all() or high.isna().all() or low.isna().all():
                    continue

                roc_10 = _adaptive_roc(close, max_period=10, min_period=3)
                f1 = roc_10 * 0.6
                atr_14 = calcular_atr_optimizado(high, low, close, periods=14)
                sma_14 = close.rolling(14, min_periods=3).mean()
                volatility_ratio = atr_14 / sma_14
                f2 = volatility_ratio * 0.4
                inercia_alcista = f1 / f2

                raw_score_adj = inercia_alcista / atr_14
                score = pd.Series(np.where(inercia_alcista >= corte, inercia_alcista, 0), index=inercia_alcista.index)
                score_adjusted = score / atr_14

                for s in (inercia_alcista, raw_score_adj, score, score_adjusted, atr_14, roc_10):
                    s.replace([np.inf, -np.inf], np.nan, inplace=True)

                inercia_alcista = inercia_alcista.fillna(0)
                raw_score_adj = raw_score_adj.fillna(0)
                score = score.fillna(0)
                score_adjusted = score_adjusted.fillna(0)

                results[ticker] = {
                    "InerciaAlcista": inercia_alcista,
                    "ATR14": atr_14,
                    "Score": score,
                    "ScoreAdjusted": score_adjusted,
                    "RawScoreAdjusted": raw_score_adj,
                    "F1": f1,
                    "F2": f2,
                    "ROC10": roc_10,
                    "VolatilityRatio": volatility_ratio
                }
            except Exception:
                continue

        if not results:
            return {}

        combined_results = {}
        for metric in ["InerciaAlcista", "ATR14", "Score", "ScoreAdjusted", "RawScoreAdjusted", "F1", "F2", "ROC10", "VolatilityRatio"]:
            metric_data = {}
            for tkr in results.keys():
                if metric in results[tkr] and results[tkr][metric] is not None:
                    metric_data[tkr] = results[tkr][metric]
            if metric_data:
                combined_results[metric] = pd.DataFrame(metric_data)
        return combined_results
    except Exception:
        return {}

def run_backtest_optimized(prices, benchmark, commission=0.003, top_n=10, corte=680, 
                          ohlc_data=None, historical_info=None, fixed_allocation=False,
                          use_roc_filter=False, use_sma_filter=False, spy_data=None,
                          progress_callback=None,
                          use_safety_etfs=False, safety_prices=None, safety_ohlc=None,
                          safety_tickers=('IEF','BIL'),
                          avoid_rebuy_unchanged=True,
                          enable_fallback=False,
                          warmup_months=15,
                          user_start_date=None):
    """
    Backtest mensual con:
    - Regla histórica 'último evento <= fecha manda'.
    - Refugio IEF/BIL (opcional).
    - Ventanas adaptativas (arranca pronto).
    - Warm-up: recarga precios desde (start_date - warmup_months) y recorta resultados desde start_date.
      (start_date se deduce de 'prices' si no se pasa user_start_date)
    - Fallback opcional (por si algún mes no hay candidatos por corte).
    """
    try:
        if prices is None or (isinstance(prices, (pd.DataFrame, pd.Series)) and prices.empty):
            return pd.DataFrame(), pd.DataFrame()

        # Detectar fechas de usuario a partir del DataFrame recibido
        if isinstance(prices, pd.Series):
            idx_all = pd.to_datetime(prices.index)
            user_start = user_start_date or idx_all.min().date()
            user_end = idx_all.max().date()
            tickers_list = [prices.name] if prices.name else []
        else:
            idx_all = pd.to_datetime(prices.index)
            user_start = user_start_date or idx_all.min().date()
            user_end = idx_all.max().date()
            tickers_list = list(prices.columns)

        # Warm-up: recargar precios extendiendo 15 meses hacia atrás desde user_start
        warmup_start_dt = (pd.Timestamp(user_start) - pd.DateOffset(months=warmup_months)).date()
        if tickers_list:
            try:
                prices_ext, ohlc_ext = download_prices(tickers_list, warmup_start_dt, user_end, load_full_data=True)
                if isinstance(prices_ext, pd.DataFrame) and not prices_ext.empty:
                    prices = prices_ext
                    ohlc_data = ohlc_ext
            except Exception:
                # Si fallara, seguimos con lo recibido
                pass

        # Mensualizar precios (ya con warm-up)
        if isinstance(prices, pd.Series):
            prices_m = prices.resample('ME').last()
            prices_df_m = pd.DataFrame({'Close': prices_m})
        else:
            prices_df_m = prices.resample('ME').last()

        # Benchmark mensual (no es crítico aplicar warm-up aquí)
        if isinstance(benchmark, pd.Series):
            bench_m = benchmark.resample('ME').last()
        else:
            bench_m = benchmark.resample('ME').last()

        # SPY mensual para filtros
        spy_monthly = None
        if (use_roc_filter or use_sma_filter) and spy_data is not None:
            spy_series = spy_data.iloc[:, 0] if isinstance(spy_data, pd.DataFrame) else spy_data
            spy_monthly = spy_series.resample('ME').last()

        # ETFs refugio (no es crítico warm-up; se usan si el filtro manda a cash)
        safety_prices_m = None
        monthly_safety_ohlc = None
        if use_safety_etfs and safety_prices is not None:
            if isinstance(safety_prices, pd.Series):
                safety_prices = safety_prices.to_frame()
            if isinstance(safety_prices, pd.DataFrame) and not safety_prices.empty:
                safety_prices_m = safety_prices.resample('ME').last()
                monthly_safety_ohlc = convertir_a_mensual_con_ohlc(safety_ohlc) if safety_ohlc else None

        # Helper refugio sin corte
        def compute_raw_inercia_and_score_at(prev_dt, close_s, high_s=None, low_s=None):
            try:
                close = close_s.copy().dropna()
                if getattr(close.index, 'freq', None) not in ['ME', 'M']:
                    close = close.resample('ME').last()
                close = close.loc[:prev_dt]
                if len(close) < 3:
                    return np.nan, np.nan
                if high_s is not None and low_s is not None:
                    high = high_s.copy().dropna().resample('ME').max().loc[close.index]
                    low = low_s.copy().dropna().resample('ME').min().loc[close.index]
                else:
                    monthly_returns = close.pct_change().dropna()
                    vol = monthly_returns.rolling(3, min_periods=1).std().fillna(0.02).clip(0.005, 0.03)
                    high = close * (1 + vol * 0.5)
                    low = close * (1 - vol * 0.5)
                    high = pd.Series(np.maximum(high, close), index=close.index)
                    low = pd.Series(np.minimum(low, close), index=close.index)
                high = high.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                low = low.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                close = close.replace([np.inf, -np.inf], np.nan).ffill().bfill()

                roc_10 = _adaptive_roc(close, max_period=10, min_period=3)
                f1 = roc_10 * 0.6
                atr_14 = calcular_atr_optimizado(high, low, close, periods=14)
                sma_14 = close.rolling(14, min_periods=3).mean()
                volatility_ratio = atr_14 / sma_14
                f2 = volatility_ratio * 0.4
                inercia_alcista = f1 / f2

                if prev_dt not in inercia_alcista.index or prev_dt not in atr_14.index:
                    return np.nan, np.nan
                inercia_val = float(inercia_alcista.loc[prev_dt])
                atr_val = float(atr_14.loc[prev_dt])
                score_adj_raw = (inercia_val / atr_val) if atr_val and atr_val != 0 else np.nan
                return inercia_val, score_adj_raw
            except Exception:
                return np.nan, np.nan

        # Indicadores universo normal (con warm-up)
        all_indicators = precalculate_all_indicators(prices_df_m, ohlc_data, corte)
        current_tickers = list(prices_df_m.columns)
        monthly_index = prices_df_m.index

        # Elegir el primer índice 'i' tal que prev_date >= fin de mes de user_start
        crop_start_mend = (pd.Timestamp(user_start) + pd.offsets.MonthEnd(0)).normalize()
        # Posición donde prev_date (index[i-1]) cumple >= crop_start_mend
        i_start = 1
        for i in range(1, len(monthly_index)):
            if monthly_index[i - 1] >= crop_start_mend:
                i_start = i
                break

        # Mapa de tickers válidos por fecha (puede incluir warm-up; lo usaremos desde i_start)
        monthly_dates_for_map = monthly_index[1:]  # compat
        if historical_info and 'changes_data' in historical_info:
            valid_tickers_by_date = precalculate_valid_tickers_by_date(
                monthly_dates_for_map, historical_info['changes_data'], current_tickers
            )
        else:
            valid_tickers_by_date = {dt: set(current_tickers) for dt in monthly_dates_for_map}

        # Loop del backtest (arrancamos en i_start)
        base_idx = max(0, i_start - 1)
        equity = [10000.0]
        dates = [monthly_index[base_idx]]
        picks_list = []
        total_months = len(monthly_index) - 1

        prev_selected_set = set()
        last_mode = 'none'   # 'normal' | 'safety' | 'none'
        last_safety_ticker = None

        # Prepara precios mensuales para retornos
        prices_df_m = prices_df_m  # alias

        for i in range(i_start, len(monthly_index)):
            try:
                prev_date = monthly_index[i - 1]
                date_i = monthly_index[i]

                if progress_callback and i % 10 == 0:
                    progress_callback(i / max(1, total_months))

                # Filtros de mercado
                market_filter_active = False
                if spy_monthly is not None and prev_date in spy_monthly.index:
                    spy_price = spy_monthly.loc[prev_date]
                    if use_roc_filter and len(spy_monthly[:prev_date]) >= 13:
                        spy_12m_ago = spy_monthly[:prev_date].iloc[-13]
                        spy_roc_12m = ((spy_price - spy_12m_ago) / spy_12m_ago) * 100 if spy_12m_ago != 0 else 0
                        if spy_roc_12m < 0:
                            market_filter_active = True
                    if use_sma_filter and len(spy_monthly[:prev_date]) >= 10:
                        spy_sma_10m = spy_monthly[:prev_date].iloc[-10:].mean()
                        if spy_price < spy_sma_10m:
                            market_filter_active = True

                valid_tickers_for_date = valid_tickers_by_date.get(prev_date, set(current_tickers))
                if use_safety_etfs and safety_tickers:
                    valid_tickers_for_date = set(valid_tickers_for_date) - set(safety_tickers)

                # Modo refugio
                if market_filter_active:
                    if use_safety_etfs and safety_prices_m is not None and not safety_prices_m.empty:
                        candidates = []
                        for st in safety_tickers:
                            if st not in safety_prices_m.columns:
                                continue
                            try:
                                if prev_date in safety_prices_m.index and date_i in safety_prices_m.index:
                                    pr0 = safety_prices_m.loc[prev_date, st]
                                    pr1 = safety_prices_m.loc[date_i, st]
                                    if pd.notna(pr0) and pd.notna(pr1) and pr0 != 0:
                                        ret_1m = (pr1 / pr0) - 1
                                        if monthly_safety_ohlc and st in monthly_safety_ohlc:
                                            high_s = monthly_safety_ohlc[st]['High']
                                            low_s = monthly_safety_ohlc[st]['Low']
                                            close_s = monthly_safety_ohlc[st]['Close']
                                        else:
                                            high_s = low_s = None
                                            close_s = safety_prices_m[st]
                                        inercia_val, score_adj_raw = compute_raw_inercia_and_score_at(prev_date, close_s, high_s, low_s)
                                        if pd.isna(score_adj_raw):
                                            score_adj_raw = ret_1m
                                        if pd.isna(inercia_val):
                                            inercia_val = 0.0
                                        candidates.append({
                                            'ticker': st,
                                            'ret': float(ret_1m),
                                            'inercia': float(inercia_val),
                                            'score_adj': float(score_adj_raw)
                                        })
                            except Exception:
                                continue
                        if candidates:
                            candidates = sorted(candidates, key=lambda x: x['score_adj'], reverse=True)
                            best = candidates[0]
                            commission_effect = commission
                            if avoid_rebuy_unchanged and last_mode == 'safety' and last_safety_ticker == best['ticker']:
                                commission_effect = 0.0
                            portfolio_return = best['ret'] - commission_effect
                            new_equity = equity[-1] * (1 + portfolio_return)
                            equity.append(new_equity)
                            dates.append(date_i)
                            picks_list.append({
                                "Date": prev_date.strftime("%Y-%m-%d"),
                                "Rank": 1,
                                "Ticker": best['ticker'],
                                "Inercia": best['inercia'],
                                "ScoreAdj": best['score_adj'],
                                "HistoricallyValid": True
                            })
                            last_mode = 'safety'
                            last_safety_ticker = best['ticker']
                            prev_selected_set = {best['ticker']}
                            continue
                    # Sin refugio -> cash
                    equity.append(equity[-1]); dates.append(date_i)
                    last_mode = 'safety'; last_safety_ticker = None; prev_selected_set = set()
                    continue

                # Selección normal (por corte)
                candidates = []
                for ticker in valid_tickers_for_date:
                    indic = all_indicators.get(ticker)
                    if not indic:
                        continue
                    try:
                        if prev_date in indic['InerciaAlcista'].index:
                            inercia = indic['InerciaAlcista'].loc[prev_date]
                            score_adj = indic['ScoreAdjusted'].loc[prev_date]
                            if inercia >= corte and score_adj > 0 and not np.isnan(score_adj):
                                candidates.append({
                                    'ticker': ticker,
                                    'inercia': float(inercia),
                                    'score_adj': float(score_adj)
                                })
                    except Exception:
                        continue

                # Fallback opcional si no hay candidatos por corte (desactivado por defecto)
                used_fallback = False
                if not candidates and enable_fallback:
                    fb = []
                    for ticker in valid_tickers_for_date:
                        indic = all_indicators.get(ticker)
                        if not indic:
                            continue
                        try:
                            if prev_date in indic['RawScoreAdjusted'].index:
                                raw_sa = float(indic['RawScoreAdjusted'].loc[prev_date])
                                ine = float(indic['InerciaAlcista'].loc[prev_date]) if prev_date in indic['InerciaAlcista'].index else 0.0
                                if not np.isnan(raw_sa):
                                    fb.append({'ticker': ticker, 'inercia': ine, 'score_adj': raw_sa})
                        except Exception:
                            continue
                    fb = [x for x in fb if np.isfinite(x['score_adj'])]
                    if fb:
                        fb = sorted(fb, key=lambda x: x['score_adj'], reverse=True)
                        candidates = fb[:min(top_n, len(fb))]
                        used_fallback = True

                if not candidates:
                    equity.append(equity[-1]); dates.append(date_i)
                    last_mode = 'normal'
                    continue

                candidates = sorted(candidates, key=lambda x: x['score_adj'], reverse=True)
                selected_picks = candidates[:(10 if fixed_allocation else top_n)]
                selected_tickers = [p['ticker'] for p in selected_picks]

                # Retornos al siguiente mes
                available_prices = prices_df_m.loc[date_i]
                prev_prices = prices_df_m.loc[prev_date]
                valid_tickers = []
                ticker_returns = []
                for ticker in selected_tickers:
                    if (ticker in available_prices.index and
                        ticker in prev_prices.index and
                        not pd.isna(available_prices[ticker]) and
                        not pd.isna(prev_prices[ticker]) and
                        prev_prices[ticker] != 0):
                        valid_tickers.append(ticker)
                        ret = (available_prices[ticker] / prev_prices[ticker]) - 1
                        ticker_returns.append(ret)

                if not valid_tickers:
                    equity.append(equity[-1]); dates.append(date_i)
                    last_mode = 'normal'
                    continue

                if fixed_allocation:
                    valid_tickers = valid_tickers[:10]
                    ticker_returns = ticker_returns[:10]
                    weight = 0.1
                else:
                    weight = 1.0 / len(valid_tickers)

                portfolio_return = sum(r * weight for r in ticker_returns)

                # Comisión dinámica
                commission_effect = commission
                if avoid_rebuy_unchanged:
                    if last_mode != 'normal':
                        commission_effect = commission
                    else:
                        new_set = set(valid_tickers)
                        old_set = set(prev_selected_set)
                        entries = len(new_set - old_set)
                        exits = len(old_set - new_set)
                        turnover_ratio = (entries + exits) / max(1, len(new_set))
                        commission_effect = commission * turnover_ratio

                portfolio_return -= commission_effect
                new_equity = equity[-1] * (1 + portfolio_return)
                equity.append(new_equity)
                dates.append(date_i)

                for rank, ticker in enumerate(valid_tickers, 1):
                    pdict = next((p for p in selected_picks if p['ticker'] == ticker), None)
                    if pdict:
                        picks_list.append({
                            "Date": prev_date.strftime("%Y-%m-%d"),
                            "Rank": rank,
                            "Ticker": ticker,
                            "Inercia": pdict['inercia'],
                            "ScoreAdj": pdict['score_adj'],
                            "HistoricallyValid": ticker in valid_tickers_for_date,
                            "Fallback": used_fallback
                        })

                prev_selected_set = set(valid_tickers)
                last_mode = 'normal'
                last_safety_ticker = None

            except Exception:
                equity.append(equity[-1])
                dates.append(date_i)
                continue

        # Construir resultados (ya arrancados desde el primer mes válido según warm-up)
        equity_series = pd.Series(equity, index=dates)
        # Recortar por seguridad a >= crop_start_mend
        equity_series = equity_series[equity_series.index >= crop_start_mend]

        returns = equity_series.pct_change().fillna(0)
        drawdown = (equity_series / equity_series.cummax() - 1).fillna(0)
        bt_results = pd.DataFrame({
            "Equity": equity_series,
            "Returns": returns,
            "Drawdown": drawdown
        })

        picks_df = pd.DataFrame(picks_list) if picks_list else pd.DataFrame()
        if not picks_df.empty:
            picks_df['Date'] = pd.to_datetime(picks_df['Date'])
            picks_df = picks_df[picks_df['Date'] >= crop_start_mend]
            # Devolver Date como string (formato YYYY-MM-DD) para mantener compatibilidad UI
            picks_df['Date'] = picks_df['Date'].dt.strftime('%Y-%m-%d')

        return bt_results, picks_df

    except Exception:
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

def calculate_monthly_returns_by_year(equity_series):
    """Calcula retornos mensuales por año."""
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
            for i, m in enumerate(months_es, 1):
                if i in year_monthly.index.get_level_values(1):
                    rv = year_monthly[year_monthly.index.get_level_values(1) == i].iloc[0]
                    year_data[m] = f"{rv*100:.1f}%"
                else:
                    year_data[m] = "-"
            year_equity = equity_series[equity_series.index.year == year]
            if len(year_equity) > 1:
                ytd_return = (year_equity.iloc[-1] / year_equity.iloc[0]) - 1
                year_data['YTD'] = f"{ytd_return*100:.1f}%"
            else:
                year_data['YTD'] = "-"
            table_data.append(year_data)
        if table_data:
            df = pd.DataFrame(table_data)
            return df[['Año'] + months_es + ['YTD']]
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calcula Sharpe Ratio (tasa libre de riesgo anualizada).
    """
    rf_m = (1 + risk_free_rate) ** (1/12) - 1
    excess = returns - rf_m
    if excess.std() > 0:
        return (excess.mean() * 12) / (excess.std() * np.sqrt(12))
    return 0.0

# Wrappers compatibilidad
def calcular_atr_amibroker(*args, **kwargs):
    return calcular_atr_optimizado(*args, **kwargs)

def run_backtest(*args, **kwargs):
    return run_backtest_optimized(*args, **kwargs)
