import pandas as pd
import numpy as np
from datetime import datetime

def calcular_atr_optimizado(high, low, close, periods=14):
    """
    Calcula ATR usando EWM (Exponential Weighted Mean) - MUCHO MÁS RÁPIDO
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
    Precalcula tickers válidos para todas las fechas
    """
    if historical_changes_data is None or historical_changes_data.empty:
        return {date: set(current_tickers) for date in monthly_dates}

    historical_changes_data['Date'] = pd.to_datetime(historical_changes_data['Date']).dt.date
    valid_by_date = {}

    for target_date in monthly_dates:
        target_date_clean = target_date.date() if isinstance(target_date, pd.Timestamp) else target_date
        valid_tickers = set(current_tickers)
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
    Precalcula todos los indicadores de una vez
    """
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
            if monthly_ohlc and ticker in monthly_ohlc:
                high = monthly_ohlc[ticker]['High']
                low = monthly_ohlc[ticker]['Low']
                close = monthly_ohlc[ticker]['Close']
            else:
                close = ticker_data
                if close.index.freq not in ['ME', 'M']:
                    close = close.resample('ME').last()
                monthly_returns = close.pct_change().dropna()
                vol = monthly_returns.rolling(3).std().fillna(0.02).clip(0.005, 0.03)
                high = close * (1 + vol * 0.5)
                low = close * (1 - vol * 0.5)
            close = close.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
            high = high.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
            low = low.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
            if close.isna().all():
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


def inertia_score(monthly_prices_df, corte=680, ohlc_data=None):
    """
    Calcula el score de inercia
    """
    # (idéntico a tu versión previa, no lo repito por espacio)
    ...


def run_backtest_optimized(prices, benchmark, commission=0.003, top_n=10, corte=680, 
                          ohlc_data=None, historical_info=None, fixed_allocation=False,
                          use_roc_filter=False, use_sma_filter=False, spy_data=None,
                          progress_callback=None):
    """
    Versión optimizada del backtest con precálculo
    """
    # (idéntico a tu versión previa, sin cambios funcionales)
    ...


def calculate_monthly_returns_by_year(equity_series):
    """Calcula retornos mensuales por año"""
    # (idéntico a tu versión previa, sin cambios)
    ...


def calculate_sharpe_ratio(returns, risk_free_rate=0.02, freq="M"):
    """
    Calcula el Sharpe Ratio anualizado.
    freq = "D" (diario), "M" (mensual), "W" (semanal)
    """
    if len(returns) < 2:
        return 0.0

    if freq == "D":
        periods_per_year = 252
        risk_free_rate_periodic = (1 + risk_free_rate) ** (1/252) - 1
    elif freq == "W":
        periods_per_year = 52
        risk_free_rate_periodic = (1 + risk_free_rate) ** (1/52) - 1
    else:  # mensual
        periods_per_year = 12
        risk_free_rate_periodic = (1 + risk_free_rate) ** (1/12) - 1

    excess_returns = returns - risk_free_rate_periodic
    excess_returns = excess_returns.dropna()

    if len(excess_returns) < 2:
        return 0.0

    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess == 0 or np.isnan(std_excess):
        return 0.0

    sharpe_ratio = (mean_excess * periods_per_year) / (std_excess * np.sqrt(periods_per_year))
    return sharpe_ratio


# Wrappers para compatibilidad
def calcular_atr_amibroker(*args, **kwargs):
    return calcular_atr_optimizado(*args, **kwargs)

def run_backtest(*args, **kwargs):
    return run_backtest_optimized(*args, **kwargs)
