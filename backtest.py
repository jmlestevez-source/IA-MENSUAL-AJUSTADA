import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

def get_valid_tickers_for_date(target_date, historical_changes_data, current_tickers):
    """
    Retorna los tickers que estaban válidos en el índice en una fecha específica
    
    Args:
        target_date: fecha objetivo
        historical_changes_data: DataFrame con cambios históricos
        current_tickers: lista de tickers actuales
    """
    if historical_changes_data is None or historical_changes_data.empty:
        # Si no hay datos históricos, usar todos los tickers actuales
        return set(current_tickers)
    
    # Convertir target_date a date si es datetime
    if isinstance(target_date, pd.Timestamp):
        target_date = target_date.date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()
    
    # Empezar con tickers actuales
    valid_tickers = set(current_tickers)
    
    # Procesar cambios desde target_date hacia adelante (revertir cambios futuros)
    future_changes = historical_changes_data[historical_changes_data['Date'] > target_date]
    future_changes = future_changes.sort_values('Date', ascending=True)  # Más antiguos primero
    
    for _, change in future_changes.iterrows():
        ticker = change['Ticker']
        action = change['Action']
        
        # Revertir cambios futuros
        if action == 'Added':
            # Si fue agregado después de target_date, no debería estar en target_date
            valid_tickers.discard(ticker)
        elif action == 'Removed':
            # Si fue removido después de target_date, debería estar en target_date
            valid_tickers.add(ticker)
    
    return valid_tickers

def inertia_score_with_historical_filter(monthly_prices_df, target_date, valid_tickers, corte=680, ohlc_data=None):
    """
    Calcula el score de inercia solo para tickers válidos en la fecha objetivo
    """
    if monthly_prices_df is None or monthly_prices_df.empty:
        return pd.DataFrame()
    
    # Filtrar solo tickers válidos para la fecha
    available_tickers = set(monthly_prices_df.columns)
    tickers_to_use = list(available_tickers.intersection(valid_tickers))
    
    if not tickers_to_use:
        return {}
    
    # Filtrar DataFrame
    filtered_prices = monthly_prices_df[tickers_to_use]
    
    # Filtrar OHLC data también
    filtered_ohlc = None
    if ohlc_data:
        filtered_ohlc = {ticker: data for ticker, data in ohlc_data.items() if ticker in tickers_to_use}
    
    # Usar la función original de cálculo de inercia
    return inertia_score(filtered_prices, corte=corte, ohlc_data=filtered_ohlc)

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

def run_backtest(prices, benchmark, commission=0.003, top_n=10, corte=680, ohlc_data=None, historical_info=None):
    """
    Ejecuta backtest con verificación histórica de constituyentes
    
    Args:
        historical_info: dict con información histórica de cambios en índices
    """
    try:
        print("Iniciando backtest con verificación histórica...")
        
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

        # Preparar datos históricos si están disponibles
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

        print(f"Datos preparados. Fechas: {len(prices_df_m)}")

        for i in range(1, len(prices_df_m)):
            try:
                prev_date = prices_df_m.index[i - 1]
                date = prices_df_m.index[i]

                # VERIFICACIÓN HISTÓRICA: Obtener tickers válidos para prev_date
                if historical_changes is not None:
                    valid_tickers_for_date = get_valid_tickers_for_date(
                        prev_date, historical_changes, current_tickers
                    )
                    print(f"📅 {prev_date.strftime('%Y-%m-%d')}: {len(valid_tickers_for_date)} tickers válidos de {len(current_tickers)} disponibles")
                else:
                    valid_tickers_for_date = set(current_tickers)

                # Calcular scores usando datos históricos hasta la fecha anterior
                # SOLO para tickers que estaban en el índice en esa fecha
                historical_data = prices_df_m.loc[:prev_date].copy()
                if len(historical_data) < 15:
                    continue

                # Filtrar historical_data para incluir solo tickers válidos
                available_valid_tickers = list(set(historical_data.columns).intersection(valid_tickers_for_date))
                if not available_valid_tickers:
                    print(f"⚠️  No hay tickers válidos disponibles para {prev_date}")
                    # Mantener equity igual (efectivo)
                    equity.append(equity[-1])
                    dates.append(date)
                    continue
                
                historical_data_filtered = historical_data[available_valid_tickers]

                # Pasar también los datos OHLC históricos filtrados
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

                # Calcular scores solo para tickers válidos
                df_score = inertia_score(historical_data_filtered, corte=corte, ohlc_data=historical_ohlc)
                if df_score is None or not df_score:
                    # Mantener equity igual (efectivo)
