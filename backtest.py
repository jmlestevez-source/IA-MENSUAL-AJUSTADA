# =========================
# ===== backtest.py =====
# =========================
# -*- coding: utf-8 -*-
# backtest.py (warm-up fuerte, arranque correcto del primer mes real, y fallback robusto)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings('ignore')

from data_loader import download_prices

def calcular_atr_optimizado(high, low, close, periods=14):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/periods, adjust=False).mean()

def convertir_a_mensual_con_ohlc(ohlc_data):
    monthly_data = {}
    for ticker, data in (ohlc_data or {}).items():
        try:
            df = pd.DataFrame({'High': data['High'], 'Low': data['Low'], 'Close': data['Close']}).dropna(how='all')
            if df.empty: 
                continue
            df_m = df.resample('M').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
            monthly_data[ticker] = {'High': df_m['High'], 'Low': df_m['Low'], 'Close': df_m['Close']}
        except Exception:
            continue
    return monthly_data

def _adaptive_roc(close, max_period=10, min_period=3):
    close = close.copy().astype(float)
    roc = pd.Series(index=close.index, dtype=float)
    for i in range(len(close)):
        if i >= min_period:
            n = min(max_period, i)
            prev = close.iloc[i - n]; cur = close.iloc[i]
            roc.iloc[i] = ((cur - prev) / prev) * 100.0 if (pd.notna(prev) and prev != 0 and pd.notna(cur)) else np.nan
        else:
            roc.iloc[i] = np.nan
    return roc

def precalculate_valid_tickers_by_date(monthly_dates, historical_changes_data, current_tickers):
    def _norm_t(t): return str(t).strip().upper().replace('.', '-')
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
                valids.add(t); continue
            tdf = grouped.get_group(t); tdf_before = tdf[tdf['Date'] <= td]
            if tdf_before.empty: 
                continue
            last_action = str(tdf_before.sort_values('Date').iloc[-1]['Action']).lower()
            if 'add' in last_action: 
                valids.add(t)
        valid_by_date[target_date] = valids
    return valid_by_date

def precalculate_all_indicators(prices_df_m, ohlc_data, corte=680):
    try:
        all_indicators = {}
        monthly_ohlc = convertir_a_mensual_con_ohlc(ohlc_data) if ohlc_data else None
        for ticker in prices_df_m.columns:
            try:
                close_series = prices_df_m[ticker].dropna()
                if close_series.empty: 
                    continue
                if monthly_ohlc and ticker in monthly_ohlc:
                    high = monthly_ohlc[ticker]['High']; low = monthly_ohlc[ticker]['Low']; close = monthly_ohlc[ticker]['Close']
                else:
                    close = close_series
                    if getattr(close.index, 'freq', None) not in ['M']:
                        close = close.resample('M').last()
                    monthly_returns = close.pct_change().dropna()
                    vol = monthly_returns.rolling(3, min_periods=1).std().fillna(0.02).clip(0.005, 0.03)
                    high = close * (1 + vol * 0.5); low = close * (1 - vol * 0.5)
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
                all_indicators[ticker] = {
                    'InerciaAlcista': inercia_alcista.fillna(0),
                    'Score': score.fillna(0),
                    'ScoreAdjusted': score_adjusted.fillna(0),
                    'RawScoreAdjusted': raw_score_adj.fillna(0),
                    'ATR14': atr_14, 'ROC10': roc_10
                }
            except Exception:
                continue
        return all_indicators
    except Exception:
        return {}

def inertia_score(monthly_prices_df, corte=680, ohlc_data=None):
    try:
        if monthly_prices_df is None or monthly_prices_df.empty:
            return {}
        results = {}
        monthly_ohlc = convertir_a_mensual_con_ohlc(ohlc_data) if ohlc_data else None
        for ticker in monthly_prices_df.columns:
            try:
                close_series = monthly_prices_df[ticker].dropna()
                if close_series.empty: 
                    continue
                if monthly_ohlc and ticker in monthly_ohlc:
                    high = monthly_ohlc[ticker]['High']; low = monthly_ohlc[ticker]['Low']; close = monthly_ohlc[ticker]['Close']
                else:
                    close = close_series
                    if getattr(close.index, 'freq', None) not in ['M']:
                        close = close.resample('M').last()
                    monthly_returns = close.pct_change().dropna()
                    vol = monthly_returns.rolling(3, min_periods=1).std().fillna(0.02).clip(0.005, 0.03)
                    high = close * (1 + vol * 0.5); low = close * (1 - vol * 0.5)
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
                results[ticker] = {
                    "InerciaAlcista": inercia_alcista.fillna(0),
                    "ATR14": atr_14, "Score": score.fillna(0),
                    "ScoreAdjusted": score_adjusted.fillna(0),
                    "RawScoreAdjusted": raw_score_adj.fillna(0),
                    "F1": f1, "F2": f2, "ROC10": roc_10, "VolatilityRatio": volatility_ratio
                }
            except Exception:
                continue
        if not results: 
            return {}
        out = {}
        for metric in ["InerciaAlcista","ATR14","Score","ScoreAdjusted","RawScoreAdjusted","F1","F2","ROC10","VolatilityRatio"]:
            md = {t: results[t][metric] for t in results if metric in results[t] and results[t][metric] is not None}
            if md: 
                out[metric] = pd.DataFrame(md)
        return out
    except Exception:
        return {}

def run_backtest_optimized(prices, benchmark, commission=0.003, top_n=10, corte=680, 
                          ohlc_data=None, historical_info=None, fixed_allocation=False,
                          use_roc_filter=False, use_sma_filter=False, spy_data=None,
                          progress_callback=None, use_safety_etfs=False,
                          safety_prices=None, safety_ohlc=None, safety_tickers=('IEF','BIL'),
                          avoid_rebuy_unchanged=True, enable_fallback=True,
                          warmup_months=24, user_start_date=None):
    try:
        if prices is None or (isinstance(prices, (pd.DataFrame, pd.Series)) and prices.empty):
            return pd.DataFrame(), pd.DataFrame()

        # Rango recibido y warm-up
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

        warmup_start_dt = (pd.Timestamp(user_start) - pd.DateOffset(months=warmup_months)).date()
        if tickers_list:
            try:
                prices_ext, ohlc_ext = download_prices(tickers_list, warmup_start_dt, user_end, load_full_data=True)
                if isinstance(prices_ext, pd.DataFrame) and not prices_ext.empty:
                    prices = prices_ext; ohlc_data = ohlc_ext
            except Exception:
                pass

        # Mensualizar
        prices_df_m = prices.resample('M').last() if isinstance(prices, pd.DataFrame) else pd.DataFrame({'Close': prices.resample('M').last()})
        prices_df_m = prices_df_m.sort_index()

        # SPY mensual (filtros) con warm-up
        spy_monthly = None
        if use_roc_filter or use_sma_filter:
            try:
                spydf = download_prices(["SPY"], warmup_start_dt, user_end, load_full_data=False)
                spyp = spydf[0] if isinstance(spydf, tuple) else spydf
                if isinstance(spyp, pd.DataFrame) and "SPY" in spyp.columns:
                    spy_monthly = spyp["SPY"].resample("M").last().sort_index()
            except Exception:
                if spy_data is not None:
                    s = spy_data.iloc[:, 0] if isinstance(spy_data, pd.DataFrame) else spy_data
                    spy_monthly = s.resample('M').last().sort_index()

        # Safety mensual (IEF/BIL) con warm-up
        safety_prices_m = None; monthly_safety_ohlc = None
        if use_safety_etfs:
            try:
                safdf = download_prices(list(safety_tickers), warmup_start_dt, user_end, load_full_data=True)
                safp, saf_ohlc = safdf if isinstance(safdf, tuple) else (safdf, None)
                if isinstance(safp, pd.DataFrame) and not safp.empty:
                    safety_prices_m = safp.resample('M').last().sort_index()
                    monthly_safety_ohlc = convertir_a_mensual_con_ohlc(saf_ohlc) if saf_ohlc else None
            except Exception:
                if isinstance(safety_prices, pd.DataFrame) and not safety_prices.empty:
                    safety_prices_m = safety_prices.resample('M').last().sort_index()
                    monthly_safety_ohlc = convertir_a_mensual_con_ohlc(safety_ohlc) if safety_ohlc else None

        # Indicadores
        all_indicators = precalculate_all_indicators(prices_df_m, ohlc_data, corte)
        current_tickers = list(prices_df_m.columns)
        monthly_index = prices_df_m.index

        # Arranque: primera iteración tal que prev_date >= fin de mes del MES ANTERIOR al inicio
        prev_threshold = (pd.Timestamp(user_start) - pd.offsets.MonthEnd(1)).normalize()
        i_start = 1
        for i in range(1, len(monthly_index)):
            if monthly_index[i - 1] >= prev_threshold:
                i_start = i; break

        # Válidos por fecha
        monthly_dates_for_map = monthly_index[1:]
        if historical_info and 'changes_data' in historical_info:
            valid_tickers_by_date = precalculate_valid_tickers_by_date(monthly_dates_for_map, historical_info['changes_data'], current_tickers)
        else:
            valid_tickers_by_date = {dt: set(current_tickers) for dt in monthly_dates_for_map}

        # Loop
        equity_full = [10000.0]; dates_full = [monthly_index[max(0, i_start - 1)]]
        picks_list = []; prev_selected_set = set(); last_mode = 'none'; last_safety_ticker = None
        total_months = len(monthly_index) - 1

        for i in range(i_start, len(monthly_index)):
            try:
                prev_date = monthly_index[i - 1]; date_i = monthly_index[i]
                if progress_callback and i % 10 == 0:
                    progress_callback(i / max(1, total_months))

                # Filtros
                market_filter_active = False
                if (use_roc_filter or use_sma_filter) and spy_monthly is not None and prev_date in spy_monthly.index:
                    spy_price = spy_monthly.loc[prev_date]
                    if use_roc_filter and len(spy_monthly[:prev_date]) >= 13:
                        spy_12m_ago = spy_monthly[:prev_date].iloc[-13]
                        spy_roc_12m = ((spy_price - spy_12m_ago) / spy_12m_ago) * 100 if spy_12m_ago != 0 else 0
                        if spy_roc_12m < 0: market_filter_active = True
                    if use_sma_filter and len(spy_monthly[:prev_date]) >= 10:
                        spy_sma_10m = spy_monthly[:prev_date].iloc[-10:].mean()
                        if spy_price < spy_sma_10m: market_filter_active = True

                valid_tickers_for_date = valid_tickers_by_date.get(prev_date, set(current_tickers))
                if use_safety_etfs and safety_tickers:
                    valid_tickers_for_date = set(valid_tickers_for_date) - set(safety_tickers)

                # Refugio si procede
                if market_filter_active:
                    if use_safety_etfs and safety_prices_m is not None and not safety_prices_m.empty:
                        best_t, best_sc = None, None
                        try:
                            s_scores = inertia_score(safety_prices_m, corte=corte, ohlc_data=None)
                            s_sa = s_scores.get("ScoreAdjusted") if s_scores else None
                            if s_sa is not None and prev_date in s_sa.index:
                                last_sa = s_sa.loc[prev_date].dropna()
                                if not last_sa.empty:
                                    best_t = last_sa.sort_values(ascending=False).index[0]
                                    best_sc = float(last_sa[best_t])
                        except Exception:
                            pass

                        # Fallback a retorno 1m si no hubiera ScoreAdjusted
                        if best_t is None:
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
                                            candidates.append({'ticker': st, 'ret': float(ret_1m)})
                                except Exception:
                                    continue
                            if candidates:
                                best_t = sorted(candidates, key=lambda x: x['ret'], reverse=True)[0]['ticker']
                                best_sc = 0.0

                        if best_t is not None:
                            pr0 = safety_prices_m.loc[prev_date, best_t]
                            pr1 = safety_prices_m.loc[date_i, best_t]
                            ret_1m = ((pr1 / pr0) - 1) if (pd.notna(pr0) and pd.notna(pr1) and pr0 != 0) else 0.0
                            commission_effect = 0.0 if (avoid_rebuy_unchanged and last_mode == 'safety' and last_safety_ticker == best_t) else commission
                            portfolio_return = ret_1m - commission_effect

                            equity_full.append(equity_full[-1] * (1 + portfolio_return)); dates_full.append(date_i)
                            picks_list.append({"Date": prev_date.strftime("%Y-%m-%d"), "Rank": 1, "Ticker": best_t, "Inercia": 0.0, "ScoreAdj": float(best_sc), "HistoricallyValid": True})
                            last_mode = 'safety'; last_safety_ticker = best_t; prev_selected_set = {best_t}
                            continue

                    equity_full.append(equity_full[-1]); dates_full.append(date_i); last_mode = 'safety'; last_safety_ticker = None; prev_selected_set = set()
                    continue

                # Selección normal
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
                                candidates.append({'ticker': ticker, 'inercia': float(inercia), 'score_adj': float(score_adj)})
                    except Exception:
                        continue

                # Fallback por RawScoreAdjusted si no hay candidatos
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
                                if np.isfinite(raw_sa):
                                    fb.append({'ticker': ticker, 'inercia': ine, 'score_adj': raw_sa})
                        except Exception:
                            continue
                    if fb:
                        candidates = sorted(fb, key=lambda x: x['score_adj'], reverse=True)[:min(top_n, len(fb))]
                        used_fallback = True

                if not candidates:
                    equity_full.append(equity_full[-1]); dates_full.append(date_i); last_mode = 'normal'
                    continue

                candidates = sorted(candidates, key=lambda x: x['score_adj'], reverse=True)
                selected_picks = candidates[:(10 if fixed_allocation else top_n)]
                selected_tickers = [p['ticker'] for p in selected_picks]

                # Retornos al siguiente mes
                available_prices = prices_df_m.loc[date_i]; prev_prices_row = prices_df_m.loc[prev_date]
                valid_tickers = []; ticker_returns = []
                for tk in selected_tickers:
                    if (tk in available_prices.index and tk in prev_prices_row.index and
                        pd.notna(available_prices[tk]) and pd.notna(prev_prices_row[tk]) and prev_prices_row[tk] != 0):
                        valid_tickers.append(tk); ticker_returns.append((available_prices[tk] / prev_prices_row[tk]) - 1)

                # Fallback final si nada tiene precio válido (elige por 1m return de todos los válidos disponibles)
                if not valid_tickers:
                    try:
                        prev_row = prices_df_m.loc[prev_date]; next_row = prices_df_m.loc[date_i]
                        one_m_ret = ((next_row / prev_row) - 1).replace([np.inf, -np.inf], np.nan).dropna()
                        one_m_ret = one_m_ret[one_m_ret.index.isin(valid_tickers_for_date)]
                        if not one_m_ret.empty:
                            top_fallback = one_m_ret.sort_values(ascending=False).index[:min(top_n, len(one_m_ret))]
                            valid_tickers = list(top_fallback)
                            ticker_returns = [float(one_m_ret[tk]) for tk in valid_tickers]
                    except Exception:
                        pass

                if not valid_tickers:
                    equity_full.append(equity_full[-1]); dates_full.append(date_i); last_mode = 'normal'
                    continue

                weight = 0.1 if fixed_allocation else (1.0 / len(valid_tickers))
                if fixed_allocation:
                    valid_tickers = valid_tickers[:10]; ticker_returns = ticker_returns[:10]

                portfolio_return = sum(r * weight for r in ticker_returns)
                commission_effect = commission
                if avoid_rebuy_unchanged:
                    if last_mode != 'normal':
                        commission_effect = commission
                    else:
                        new_set = set(valid_tickers); old_set = set(prev_selected_set)
                        turnover_ratio = (len(new_set - old_set) + len(old_set - new_set)) / max(1, len(new_set))
                        commission_effect = commission * turnover_ratio

                portfolio_return -= commission_effect
                equity_full.append(equity_full[-1] * (1 + portfolio_return)); dates_full.append(date_i)

                for rank, tk in enumerate(valid_tickers, 1):
                    pdict = next((p for p in selected_picks if p['ticker'] == tk), None)
                    if pdict is None:
                        pdict = {'inercia': 0.0, 'score_adj': 0.0}
                    picks_list.append({"Date": prev_date.strftime("%Y-%m-%d"), "Rank": rank, "Ticker": tk, "Inercia": float(pdict['inercia']), "ScoreAdj": float(pdict['score_adj']), "HistoricallyValid": tk in valid_tickers_for_date, "Fallback": used_fallback})

                prev_selected_set = set(valid_tickers); last_mode = 'normal'; last_safety_ticker = None

            except Exception:
                equity_full.append(equity_full[-1]); dates_full.append(date_i)
                continue

        # Resultados y recorte a partir del fin de mes del start
        equity_series_full = pd.Series(equity_full, index=dates_full)
        returns_full = equity_series_full.pct_change().fillna(0)
        drawdown_full = (equity_series_full / equity_series_full.cummax() - 1).fillna(0)

        show_start = (pd.Timestamp(user_start) + pd.offsets.MonthEnd(0)).normalize()
        mask = equity_series_full.index >= show_start
        equity_series = equity_series_full[mask]
        returns = returns_full[mask]
        drawdown = drawdown_full[mask]
        bt_results = pd.DataFrame({"Equity": equity_series, "Returns": returns, "Drawdown": drawdown})

        picks_df = pd.DataFrame(picks_list) if picks_list else pd.DataFrame()
        if not picks_df.empty:
            picks_df['Date'] = pd.to_datetime(picks_df['Date'])
            prev_limit = (pd.Timestamp(user_start) - pd.offsets.MonthEnd(1)).normalize()
            picks_df = picks_df[picks_df['Date'] >= prev_limit]
            picks_df['Date'] = picks_df['Date'].dt.strftime('%Y-%m-%d')

        return bt_results, picks_df

    except Exception:
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

def calculate_monthly_returns_by_year(equity_series):
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
    rf_m = (1 + risk_free_rate) ** (1/12) - 1
    excess = returns - rf_m
    if excess.std() > 0:
        return (excess.mean() * 12) / (excess.std() * np.sqrt(12))
    return 0.0

def calcular_atr_amibroker(*args, **kwargs):
    return calcular_atr_optimizado(*args, **kwargs)

def run_backtest(*args, **kwargs):
    return run_backtest_optimized(*args, **kwargs)
