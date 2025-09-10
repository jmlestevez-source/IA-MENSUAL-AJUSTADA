import pandas as pd
import yfinance as yf
import requests
from io import StringIO
import os
import pickle
from datetime import datetime, timedelta
import time
import random
import streamlit as st

# Directorio para caché
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_sp500_tickers_cached():
    """Obtiene los tickers del S&P 500 con caché para evitar scraping frecuente"""
    cache_file = os.path.join(CACHE_DIR, "sp500_tickers.pkl")
    
    # Verificar si existe caché válido (menos de 7 días)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cache_time = cached_data.get('timestamp', 0)
                if (datetime.now().timestamp() - cache_time) < (7 * 24 * 3600):  # 7 días
                    print("Usando tickers S&P 500 desde caché")
                    return cached_data
        except Exception as e:
            print(f"Error leyendo caché S&P 500: {e}")
    
    # Headers para evitar bloqueos
    headers_list = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        },
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    ]
    
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    for attempt in range(3):
        try:
            headers = random.choice(headers_list)
            print(f"Intento {attempt + 1} de scraping S&P 500...")
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            tables = pd.read_html(StringIO(response.text))
            
            if not tables:
                raise ValueError("No se encontraron tablas en la página")
            
            # Buscar la tabla principal (usualmente la primera)
            df = tables[0]
            
            # Verificar columnas esperadas
            if 'Symbol' not in df.columns and 'Ticker' in df.columns:
                df = df.rename(columns={'Ticker': 'Symbol'})
            
            if 'Symbol' not in df.columns:
                # Intentar encontrar la columna de tickers
                ticker_columns = [col for col in df.columns if 'symbol' in col.lower() or 'ticker' in col.lower()]
                if ticker_columns:
                    df = df.rename(columns={ticker_columns[0]: 'Symbol'})
                else:
                    raise ValueError("No se encontró columna de símbolos")
            
            tickers = df['Symbol'].tolist()
            
            # Limpiar tickers
            tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
            tickers = [t for t in tickers if t and t != 'nan']
            
            if not tickers:
                raise ValueError("No se encontraron tickers válidos")
            
            print(f"Obtenidos {len(tickers)} tickers S&P 500")
            
            # Guardar en caché
            tickers_data = {
                'tickers': tickers,
                'data': df.to_dict('records'),
                'timestamp': datetime.now().timestamp(),
                'date': datetime.now()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(tickers_data, f)
            
            # Esperar un poco para evitar bloqueos
            time.sleep(random.uniform(1, 2))
            
            return tickers_data
            
        except Exception as e:
            print(f"Intento {attempt + 1} falló: {e}")
            if attempt < 2:  # No esperar después del último intento
                time.sleep(random.uniform(2, 4))
            continue
    
    # Fallback: lista básica de tickers comunes
    print("Usando fallback de tickers S&P 500")
    fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'HD', 'DIS', 'PYPL']
    return {
        'tickers': fallback_tickers, 
        'data': [{'Symbol': t} for t in fallback_tickers], 
        'timestamp': datetime.now().timestamp(),
        'date': datetime.now()
    }

def get_nasdaq100_tickers_cached():
    """Obtiene los tickers del Nasdaq-100 con caché"""
    cache_file = os.path.join(CACHE_DIR, "nasdaq100_tickers.pkl")
    
    # Verificar si existe caché válido (menos de 7 días)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cache_time = cached_data.get('timestamp', 0)
                if (datetime.now().timestamp() - cache_time) < (7 * 24 * 3600):  # 7 días
                    print("Usando tickers Nasdaq-100 desde caché")
                    return cached_data
        except Exception as e:
            print(f"Error leyendo caché Nasdaq-100: {e}")
    
    # Headers para evitar bloqueos
    headers_list = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    ]
    
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    
    for attempt in range(3):
        try:
            headers = random.choice(headers_list)
            print(f"Intento {attempt + 1} de scraping Nasdaq-100...")
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            tables = pd.read_html(StringIO(response.text))
            
            if not tables:
                raise ValueError("No se encontraron tablas en la página")
            
            # Buscar la tabla principal
            df = tables[0]
            
            # Verificar columna de tickers
            ticker_column = None
            for col in df.columns:
                if 'Ticker' in str(col) or 'Symbol' in str(col):
                    ticker_column = col
                    break
            
            if ticker_column is None:
                # Intentar con la primera columna
                ticker_column = df.columns[0]
            
            tickers = df[ticker_column].tolist()
            
            # Limpiar tickers
            tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
            tickers = [t for t in tickers if t and t != 'nan']
            
            if not tickers:
                raise ValueError("No se encontraron tickers válidos")
            
            print(f"Obtenidos {len(tickers)} tickers Nasdaq-100")
            
            # Guardar en caché
            tickers_data = {
                'tickers': tickers,
                'data': df.to_dict('records'),
                'timestamp': datetime.now().timestamp(),
                'date': datetime.now()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(tickers_data, f)
            
            # Esperar un poco
            time.sleep(random.uniform(1, 2))
            
            return tickers_data
            
        except Exception as e:
            print(f"Intento {attempt + 1} falló: {e}")
            if attempt < 2:
                time.sleep(random.uniform(2, 4))
            continue
    
    # Fallback
    print("Usando fallback de tickers Nasdaq-100")
    fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'PEP', 'COST', 'AVGO', 'CMCSA', 'CSCO', 'INTC', 'QCOM']
    return {
        'tickers': fallback_tickers, 
        'data': [{'Ticker': t} for t in fallback_tickers], 
        'timestamp': datetime.now().timestamp(),
        'date': datetime.now()
    }

def get_constituents_at_date(index_name, start_date, end_date):
    """
    Obtiene los constituyentes de un índice en una fecha dada
    """
    if index_name == "SP500":
        tickers_data = get_sp500_tickers_cached()
    elif index_name == "NDX":
        tickers_data = get_nasdaq100_tickers_cached()
    else:
        raise ValueError(f"Índice {index_name} no soportado")
    
    return tickers_data, None

def download_prices_with_retry(tickers, start_date, end_date, max_retries=5):  # Aumentado retries para robustez
    """
    Descarga precios con reintentos y manejo de errores mejorado
    """
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Reintento {attempt} de descarga de precios...")
                time.sleep(random.uniform(5, 10))  # Espera más larga para evitar rate limiting
            
            # Descargar datos con auto_adjust=True y repair=True para fixear issues de 2025
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date, 
                group_by='ticker',
                progress=False,
                threads=True,
                timeout=30,
                auto_adjust=True,  # Agregado para eliminar warning
                repair=True  # Agregado para reparar datos corruptos en versiones recientes
            )
            
            if data.empty:
                raise ValueError("No se recibieron datos")
            
            return data
            
        except Exception as e:
            print(f"Error en intento {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return pd.DataFrame()  # Retornar vacío en lugar de raise, para skip
            continue

def download_prices(tickers, start_date, end_date):
    """
    Descarga precios históricos para una lista de tickers con logging detallado
    """
    try:
        # Extraer tickers del DataFrame o lista
        if isinstance(tickers, dict) and 'tickers' in tickers:
            ticker_list = tickers['tickers']
        elif isinstance(tickers, pd.DataFrame):
            if 'Symbol' in tickers.columns:
                ticker_list = tickers['Symbol'].tolist()
            elif 'Ticker' in tickers.columns:
                ticker_list = tickers['Ticker'].tolist()
            else:
                ticker_list = tickers.iloc[:, 0].tolist()
        elif isinstance(tickers, list):
            ticker_list = tickers
        else:
            ticker_list = [str(tickers)]
        
        # Limpiar tickers
        ticker_list = [str(t).strip().upper() for t in ticker_list if str(t).strip()]
        ticker_list = [t.replace('.', '-') for t in ticker_list]  # Convertir puntos a guiones
        ticker_list = [t for t in ticker_list if t and t != 'nan']
        
        if not ticker_list:
            raise ValueError("No se encontraron tickers válidos")
        
        print(f"Descargando datos para {len(ticker_list)} tickers...")
        print(f"Primeros 10 tickers: {ticker_list[:10]}")
        
        # Probar primero con un ticker individual para diagnosticar
        print("Probando descarga con ticker de prueba...")
        try:
            test_ticker = ticker_list[0] if ticker_list else 'SPY'
            test_data = yf.download(
                test_ticker, 
                start=start_date, 
                end=end_date, 
                progress=False, 
                timeout=30,
                auto_adjust=True,  # Agregado
                repair=True  # Agregado
            )
            if test_data.empty:
                print(f"❌ No se pudo descargar {test_ticker}")
            else:
                print(f"✅ Prueba exitosa con {test_ticker}: {len(test_data)} registros")
        except Exception as test_e:
            print(f"❌ Error en prueba con {test_ticker}: {test_e}")
        
        # Dividir en lotes más pequeños para evitar errores (reducido a 5 para menos rate limiting)
        batch_size = 5
        all_prices = {}
        successful_batches = 0
        failed_batches = 0
        
        for i in range(0, len(ticker_list), batch_size):
            batch = ticker_list[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(ticker_list) + batch_size - 1) // batch_size
            
            print(f"Descargando lote {batch_num}/{total_batches}: {len(batch)} tickers")
            print(f"Tickers: {batch}")
            
            try:
                # Esperar entre lotes (aumentado para 2025 issues)
                if i > 0:
                    wait_time = random.uniform(5, 10)
                    print(f"Esperando {wait_time:.1f} segundos...")
                    time.sleep(wait_time)
                
                batch_data = download_prices_with_retry(batch, start_date, end_date)
                
                if batch_data.empty:
                    print(f"⚠️ Lote {batch_num} vacío, skipping...")
                    failed_batches += 1
                    continue
                
                # Procesar datos del lote
                processed_count = 0
                if len(batch) == 1:
                    ticker = batch[0]
                    if 'Adj Close' in batch_data.columns:
                        all_prices[ticker] = batch_data['Adj Close']
                        processed_count += 1
                    elif 'Close' in batch_data.columns:
                        all_prices[ticker] = batch_data['Close']
                        processed_count += 1
                else:
                    for ticker in batch:
                        try:
                            ticker_data = batch_data[ticker]
                            if 'Adj Close' in ticker_data.columns:
                                all_prices[ticker] = ticker_data['Adj Close']
                                processed_count += 1
                            elif 'Close' in ticker_data.columns:
                                all_prices[ticker] = ticker_data['Close']
                                processed_count += 1
                        except Exception as ticker_e:
                            print(f"⚠️  Error procesando {ticker}: {ticker_e}")
                            continue
                
                successful_batches += 1
                print(f"✅ Lote {batch_num}: {processed_count} tickers procesados")
                
            except Exception as batch_e:
                failed_batches += 1
                print(f"❌ Error en lote {batch_num}: {batch_e}")
                continue
        
        print(f"Resumen: {successful_batches} lotes exitosos, {failed_batches} lotes fallidos")
        print(f"Total tickers procesados: {len(all_prices)}")
        
        if not all_prices:
            print("⚠️ No se descargaron datos, usando vacío")
            return pd.DataFrame()  # No raise, para no detener todo
        
        # Crear DataFrame final
        prices_df = pd.DataFrame(all_prices)
        
        # Eliminar columnas con todos NaN
        prices_df = prices_df.dropna(axis=1, how='all')
        
        if prices_df.empty:
            print("⚠️ DataFrame vacío después de limpieza")
            return pd.DataFrame()
        
        print(f"✅ Descargados datos para {len(prices_df.columns)} tickers")
        return prices_df
        
    except Exception as e:
        print(f"❌ Error crítico en download_prices: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()  # Retornar vacío en lugar de None
```

**Cambios clave en `data_loader.py`**:
- Agregado `auto_adjust=True` y `repair=True` en todas `yf.download()` para eliminar warning y fixear datos corruptos (común en 2025).
- Aumentado waits a 5-10 seg entre lotes y retries a 5 para manejar rate limiting.
- Reducido `batch_size` a 5 para requests más espaciadas.
- En lugar de raise en fallos totales, retornar DataFrame vacío para no detener la app (usa fallback en app.py).
- Mejor logging para debugging.

### Archivo Corregido: `backtest.py`

```python
import pandas as pd
import numpy as np

# ---------- True Range mensual (aproximado) ----------
def monthly_true_range(close):
    prev = close.shift(1)
    high = low = close
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev),
                               np.abs(low - prev)))
    return tr

# ---------- Inercia Alcista (igual que AFL) ----------
def inertia_score(monthly_close, corte=680):
    # ROC con fill_method=None para eliminar warning deprecated
    roc1 = monthly_close.pct_change(10, fill_method=None) * 0.4
    roc2 = monthly_close.pct_change(10, fill_method=None) * 0.2
    f1 = roc1 + roc2
    
    # ATR(14) sobre barras mensuales
    tr = monthly_true_range(monthly_close)
    atr14 = tr.rolling(14).mean()

    # Denominador
    sma14 = monthly_close.rolling(14).mean()
    f2 = (atr14 / sma14) * 0.4

    inercia_alcista = f1 / f2

    # Corte 680
    score = np.where(inercia_alcista < corte, 0, np.maximum(inercia_alcista, 0))

    # Penalización por volatilidad
    score_adj = score / atr14

    return pd.DataFrame({
        "InerciaAlcista": inercia_alcista,
        "ATR14": atr14,
        "Score": score,
        "ScoreAdjusted": score_adj
    }).fillna(0)

# ---------- Backtest rotacional ----------
def run_backtest(prices, benchmark, commission=0.003, top_n=10, corte=680):
    try:
        # Mensualizar
        prices_m = prices.resample('ME').last()
        bench_m = benchmark.resample('ME').last()
        
        if prices_m.empty or bench_m.empty:
            raise ValueError("No hay datos mensuales suficientes para el backtest")
        
        equity = [10000]
        dates = [prices_m.index[0]]
        picks_list = []

        for i in range(1, len(prices_m)):
            prev_date = prices_m.index[i - 1]
            date = prices_m.index[i]

            # Scores
            try:
                df_score = inertia_score(prices_m.loc[:prev_date], corte=corte)
                if df_score.empty or len(df_score) < 14:  # Necesitamos al menos 14 períodos para ATR
                    continue
                    
                last_scores_series = df_score["ScoreAdjusted"].iloc[-1]
                if isinstance(last_scores_series, pd.Series):
                    last_scores = last_scores_series.sort_values(ascending=False).dropna()
                else:
                    continue
                    
                selected = last_scores.head(top_n).index.tolist()
                
                if not selected:
                    continue

                # Retorno mensual
                weight = 1.0 / len(selected)  # Usar len(selected) en caso de que sean menos de top_n
                available_prices = prices_m.loc[date, selected]
                
                if isinstance(available_prices, pd.Series):
                    rets = available_prices.pct_change(fill_method=None).fillna(0)  # Agregado fill_method=None
                else:
                    rets = pd.Series([0] * len(selected), index=selected)
                
                port_ret = (rets * weight).sum() - commission
                new_eq = equity[-1] * (1 + port_ret)

                equity.append(new_eq)
                dates.append(date)

                # Guardar picks
                for rank, ticker in enumerate(selected, 1):
                    try:
                        inercia_val = df_score["InerciaAlcista"].iloc[-1]
                        if isinstance(inercia_val, pd.Series):
                            inercia_val = inercia_val.get(ticker, 0)  # Usar get para evitar KeyError
                        
                        score_adj_val = last_scores.get(ticker, 0)  # Usar get
                        
                        picks_list.append({
                            "Date": date.strftime("%Y-%m-%d"),
                            "Rank": rank,
                            "Ticker": ticker,
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
            raise ValueError("No se generaron resultados de backtest")

        equity_series = pd.Series(equity, index=dates)
        returns = equity_series.pct_change(fill_method=None).fillna(0)  # Agregado fill_method=None
        drawdown = equity_series / equity_series.cummax() - 1

        bt = pd.DataFrame({
            "Equity": equity_series,
            "Returns": returns,
            "Drawdown": drawdown
        })
        picks_df = pd.DataFrame(picks_list)
        return bt, picks_df
        
    except Exception as e:
        print(f"Error en run_backtest: {e}")
        # Retornar DataFrames vacíos en caso de error
        empty_bt = pd.DataFrame(columns=["Equity", "Returns", "Drawdown"])
        empty_picks = pd.DataFrame(columns=["Date", "Rank", "Ticker", "Inercia", "ScoreAdj"])
        return empty_bt, empty_picks
```

**Cambios clave en `backtest.py`**:
- Agregado `fill_method=None` en todos `pct_change` para eliminar warning deprecated y manejar NaNs mejor (evita propagación en datos incompletos).
- Usado `.get(ticker, 0)` en accesos a Series para evitar KeyErrors si ticker falta.
- Mantenido el resto, pero ahora con datos más robustos de data_loader, debería generar resultados incluso con tickers parciales.

### Archivo Corregido: `app.py`

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import yfinance as yf

# Importar nuestros módulos
from data_loader import download_prices, get_constituents_at_date
from backtest import run_backtest
from utils import unify_ticker  # Asumiendo que existe; si no, quítalo

# -------------------------------------------------
# Configuración de la app
# -------------------------------------------------

st.set_page_config(
    page_title="IA Mensual Ajustada",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Estrategia mensual sobre los componentes del S&P 500 y/o Nasdaq-100")

# -------------------------------------------------
# Sidebar - Parámetros
# -------------------------------------------------

st.sidebar.header("Parámetros de backtest")

# Selector de índice
index_choice = st.sidebar.selectbox(
    "Selecciona el índice:",
    ["SP500", "NDX"]
)

# Fechas
end_date = st.sidebar.date_input("Fecha final", datetime.today())
start_date = st.sidebar.date_input("Fecha inicial", end_date - timedelta(days=365*5))

# Parámetros del backtest
top_n = st.sidebar.slider("Número de activos", 5, 30, 10)
commission = st.sidebar.number_input("Comisión por operación (%)", 0.0, 1.0, 0.3) / 100
corte = st.sidebar.number_input("Corte de score", 0, 1000, 680)

# Botón de ejecución
run_button = st.sidebar.button("🏃 Ejecutar backtest")

# -------------------------------------------------
# Main content
# -------------------------------------------------

if run_button:
    try:
        with st.spinner("Descargando datos..."):
            # Obtener constituyentes
            constituents_data, error = get_constituents_at_date(index_choice, start_date, end_date)
            
            if error:
                st.error(f"Error obteniendo constituyentes: {error}")
                st.stop()
            
            if constituents_data is None:
                st.error("No se pudieron obtener los constituyentes del índice")
                st.stop()
            
            tickers_count = len(constituents_data.get('tickers', []))
            st.success(f"✅ Obtenidos {tickers_count} constituyentes")
            
            if tickers_count == 0:
                st.error("No se encontraron tickers válidos")
                st.stop()
            
            # Mostrar algunos tickers de ejemplo
            sample_tickers = constituents_data.get('tickers', [])[:10]
            st.info(f"Tickers de ejemplo: {', '.join(sample_tickers)}")
            
            # Descargar precios de constituyentes
            prices_df = download_prices(constituents_data, start_date, end_date)
            
            if prices_df.empty:
                st.error("No se pudieron descargar los precios históricos de los constituyentes")
                # Mostrar información de debugging
                if isinstance(constituents_data, dict) and 'tickers' in constituents_data:
                    tickers_mostrados = constituents_data['tickers'][:10]
                    st.info(f"Tickers intentados: {', '.join(tickers_mostrados)}...")
                st.stop()
            
            st.success(f"✅ Descargados precios para {len(prices_df.columns)} tickers")
            
            # Descargar benchmark (SPY para S&P 500, QQQ para Nasdaq-100)
            benchmark_ticker = "SPY" if index_choice == "SP500" else "QQQ"
            st.info(f"Descargando benchmark: {benchmark_ticker}")
            
            # Descargar benchmark por separado
            benchmark_df = download_prices([benchmark_ticker], start_date, end_date)
            
            if benchmark_df.empty:
                st.warning(f"No se pudo descargar el benchmark {benchmark_ticker}")
                # Intentar con datos alternativos
                try:
                    # Usar el promedio de los constituyentes como benchmark alternativo
                    st.info("Usando promedio de constituyentes como benchmark alternativo")
                    benchmark_series = prices_df.mean(axis=1)
                    benchmark_df = pd.DataFrame({benchmark_ticker: benchmark_series})
                except Exception as avg_error:
                    st.error(f"Tampoco se pudo crear benchmark alternativo: {avg_error}")
                    st.stop()
            else:
                st.success(f"✅ Benchmark {benchmark_ticker} descargado correctamente")
        
        with st.spinner("Ejecutando backtest..."):
            # Ejecutar backtest (usar benchmark_series si df vacío)
            benchmark_series = benchmark_df[benchmark_ticker] if not benchmark_df.empty else pd.Series()
            bt_results, picks_df = run_backtest(
                prices=prices_df,
                benchmark=benchmark_series,
                commission=commission,
                top_n=top_n,
                corte=corte
            )
            
            if bt_results.empty:
                st.error("El backtest no generó resultados (posiblemente datos insuficientes)")
                st.stop()
            
            st.success("✅ Backtest completado")
            
            # -------------------------------------------------
            # Métricas principales
            # -------------------------------------------------
            
            final_equity = bt_results["Equity"].iloc[-1]
            total_return = (final_equity / bt_results["Equity"].iloc[0]) - 1
            max_drawdown = bt_results["Drawdown"].min()
            volatility = bt_results["Returns"].std() * (12 ** 0.5)  # Anualizada
            sharpe_ratio = (bt_results["Returns"].mean() * 12) / (volatility + 1e-8)  # Sharpe anualizado
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Equity Final", f"${final_equity:,.0f}")
            col2.metric("Retorno Total", f"{total_return:.2%}")
            col3.metric("Máximo Drawdown", f"{max_drawdown:.2%}")
            col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            # -------------------------------------------------
            # Gráficos
            # -------------------------------------------------
            
            # Gráfico de equity
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=bt_results.index,
                y=bt_results["Equity"],
                mode='lines',
                name='Estrategia',
                line=dict(width=3)
            ))
            
            # Benchmark
            if not benchmark_df.empty:
                bench_equity = (10000 * (1 + benchmark_df[benchmark_ticker].pct_change(fill_method=None).fillna(0))).cumprod()  # Agregado fill_method=None
                bench_equity.index = bt_results.index[:len(bench_equity)]  # Alinear fechas
                fig_equity.add_trace(go.Scatter(
                    x=bench_equity.index,
                    y=bench_equity,
                    mode='lines',
                    name=f'Benchmark ({benchmark_ticker})',
                    line=dict(width=2, dash='dash')
                ))
            
            fig_equity.update_layout(
                title="Evolución del Equity",
                xaxis_title="Fecha",
                yaxis_title="Equity ($)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # Gráfico de drawdown
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=bt_results.index,
                y=bt_results["Drawdown"] * 100,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red', width=2)
            ))
            fig_dd.update_layout(
                title="Drawdown",
                xaxis_title="Fecha",
                yaxis_title="Drawdown (%)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_dd, use_container_width=True)
            
            # -------------------------------------------------
            # Picks seleccionados
            # -------------------------------------------------
            
            if not picks_df.empty:
                st.subheader("Últimos picks seleccionados")
                latest_picks = picks_df[picks_df["Date"] == picks_df["Date"].max()]
                st.dataframe(latest_picks.round(2))
                
                # Gráfico de picks por fecha
                picks_by_date = picks_df.groupby("Date").size()
                fig_picks = px.bar(
                    x=picks_by_date.index,
                    y=picks_by_date.values,
                    labels={'x': 'Fecha', 'y': 'Número de Picks'},
                    title="Número de Picks por Fecha"
                )
                st.plotly_chart(fig_picks, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Excepción no capturada: {str(e)}")
        st.exception(e)

else:
    st.info("👈 Configura los parámetros en el panel lateral y haz clic en 'Ejecutar backtest'")
