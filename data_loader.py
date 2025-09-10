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
import numpy as np # Añadido para manejo numérico

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
        },
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0"
        }
    ]
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    for attempt in range(3):
        try:
            headers = random.choice(headers_list)
            print(f"Intento {attempt + 1} de scraping S&P 500...")
            response = requests.get(url, headers=headers, timeout=15) # Timeout aumentado
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
                time.sleep(random.uniform(3, 5)) # Espera aumentada entre reintentos
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
        },
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        },
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0"
        }
    ]
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    for attempt in range(3):
        try:
            headers = random.choice(headers_list)
            print(f"Intento {attempt + 1} de scraping Nasdaq-100...")
            response = requests.get(url, headers=headers, timeout=15) # Timeout aumentado
            response.raise_for_status()
            tables = pd.read_html(StringIO(response.text))
            if not tables or len(tables) < 3:
                raise ValueError("No se encontraron suficientes tablas en la página de Nasdaq-100")

            # Tabla de constituyentes (índice 2 según la estructura de la página)
            df_constituents = tables[2]
            print(f"Tabla de constituyentes tiene {len(df_constituents)} filas y columnas: {df_constituents.columns.tolist()}")

            # Tabla de cambios (índice 3, puede estar vacía o no existir)
            df_changes = None
            if len(tables) > 3:
                 df_changes = tables[3]
                 print(f"Tabla de cambios tiene {len(df_changes)} filas y columnas: {df_changes.columns.tolist()}")
            else:
                 print("Tabla de cambios no encontrada o vacía.")

            # Verificar columna de tickers en la tabla de constituyentes
            ticker_column = None
            # Buscar columnas comunes para tickers
            potential_ticker_cols = ['Ticker', 'Symbol', 'Company']
            for col in potential_ticker_cols:
                if col in df_constituents.columns:
                    ticker_column = col
                    break

            if ticker_column is None:
                # Si no se encuentra, usar la primera columna como fallback
                ticker_column = df_constituents.columns[0]
                print(f"Columna de ticker no encontrada, usando la primera columna: '{ticker_column}'")

            tickers = df_constituents[ticker_column].tolist()
            # Limpiar tickers
            tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
            tickers = [t for t in tickers if t and t != 'nan']
            if not tickers:
                raise ValueError("No se encontraron tickers válidos en la tabla de constituyentes")

            print(f"Obtenidos {len(tickers)} tickers Nasdaq-100")

            # Guardar en caché
            tickers_data = {
                'tickers': tickers,
                'data': df_constituents.to_dict('records'), # Guardar la tabla de constituyentes
                'changes_data': df_changes.to_dict('records') if df_changes is not None else [], # Guardar cambios si existen
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
                time.sleep(random.uniform(3, 5)) # Espera aumentada entre reintentos
            continue
    # Fallback
    print("Usando fallback de tickers Nasdaq-100")
    fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'PEP', 'COST', 'AVGO', 'CMCSA', 'CSCO', 'INTC', 'QCOM']
    return {
        'tickers': fallback_tickers,
        'data': [{'Ticker': t} for t in fallback_tickers],
        'changes_data': [], # No hay datos de cambios en fallback
        'timestamp': datetime.now().timestamp(),
        'date': datetime.now()
    }

def get_constituents_at_date(index_name, start_date, end_date):
    """
    Obtiene los constituyentes de un índice en una fecha dada
    TODO: Implementar lógica para filtrar por fecha de incorporación usando 'changes_data'
    """
    if index_name == "SP500":
        tickers_data = get_sp500_tickers_cached()
    elif index_name == "NDX": # Asegurarse de que el nombre del índice sea consistente
        tickers_data = get_nasdaq100_tickers_cached()
    else:
        raise ValueError(f"Índice {index_name} no soportado")
    return tickers_data, None

def download_prices_with_retry(tickers, start_date, end_date, max_retries=3): # Reducido retries para eficiencia, pero suficiente
    """
    Descarga precios con reintentos y manejo de errores mejorado
    """
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = min(5 * (2 ** attempt) + random.uniform(0, 1), 30) # Espera exponencial con máximo
                print(f"Reintento {attempt} de descarga de precios, esperando {wait_time:.1f}s...")
                time.sleep(wait_time)
            # Descargar datos con auto_adjust=True y repair=True
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                group_by='ticker',
                progress=False,
                threads=True,
                timeout=60, # Timeout aumentado significativamente
                auto_adjust=True,
                repair=True
            )
            if data.empty:
                raise ValueError("No se recibieron datos")
            return data
        except Exception as e:
            print(f"Error en intento {attempt + 1}: {e}")
            if "Multiple" in str(e) and "NoneType" in str(e): # Error específico de yfinance
                 print("Error conocido de yfinance, saltando lote.")
                 return pd.DataFrame() # Devolver DataFrame vacío para saltar
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
                auto_adjust=True,
                repair=True
            )
            if test_data.empty:
                print(f"❌ No se pudo descargar {test_ticker}")
            else:
                print(f"✅ Prueba exitosa con {test_ticker}: {len(test_data)} registros")
        except Exception as test_e:
            print(f"❌ Error en prueba con {test_ticker}: {test_e}")
        # Dividir en lotes más grandes para reducir sobrecarga de llamadas
        # batch_size = 5 # Original
        batch_size = 50 # Aumentado significativamente
        all_prices = {}
        successful_batches = 0
        failed_batches = 0
        total_tickers_processed = 0
        for i in range(0, len(ticker_list), batch_size):
            batch = ticker_list[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(ticker_list) + batch_size - 1) // batch_size
            print(f"Descargando lote {batch_num}/{total_batches}: {len(batch)} tickers")
            # print(f"Tickers: {batch}") # Opcional: comentar para reducir output
            try:
                # Esperar entre lotes (reducido y aleatorizado)
                # if i > 0:
                #     wait_time = random.uniform(5, 10) # Original
                #     print(f"Esperando {wait_time:.1f} segundos...")
                #     time.sleep(wait_time)
                if i > 0:
                    # Espera más corta y aleatoria entre lotes
                    wait_time = random.uniform(1, 3)
                    print(f"Esperando {wait_time:.1f} segundos entre lotes...")
                    time.sleep(wait_time)

                batch_data = download_prices_with_retry(batch, start_date, end_date)
                if batch_data.empty:
                    print(f"⚠️ Lote {batch_num} vacío o fallido, skipping...")
                    failed_batches += 1
                    continue
                # Procesar datos del lote
                processed_count = 0
                if len(batch) == 1:
                    ticker = batch[0]
                    # Manejar estructura de DataFrame para un solo ticker
                    if isinstance(batch_data.columns, pd.MultiIndex):
                        # Estructura MultiIndex (ticker, metric)
                        if ticker in batch_data.columns.levels[0]:
                             ticker_specific_data = batch_data[ticker]
                             if 'Adj Close' in ticker_specific_data.columns:
                                 all_prices[ticker] = ticker_specific_data['Adj Close']
                                 processed_count += 1
                             elif 'Close' in ticker_specific_data.columns:
                                 all_prices[ticker] = ticker_specific_data['Close']
                                 processed_count += 1
                    else:
                        # Estructura plana (metric), típico para un solo ticker
                        if 'Adj Close' in batch_data.columns:
                            all_prices[ticker] = batch_data['Adj Close']
                            processed_count += 1
                        elif 'Close' in batch_data.columns:
                            all_prices[ticker] = batch_data['Close']
                            processed_count += 1
                else:
                    # Múltiples tickers
                    for ticker in batch:
                        try:
                            if ticker in batch_data.columns.levels[0]: # Verificar si el ticker está en los datos
                                ticker_data = batch_data[ticker]
                                if 'Adj Close' in ticker_data.columns:
                                    all_prices[ticker] = ticker_data['Adj Close']
                                    processed_count += 1
                                elif 'Close' in ticker_data.columns:
                                    all_prices[ticker] = ticker_data['Close']
                                    processed_count += 1
                            else:
                                print(f"⚠️  Ticker {ticker} no encontrado en los datos del lote.")
                        except Exception as ticker_e:
                            print(f"⚠️  Error procesando {ticker}: {ticker_e}")
                            continue
                successful_batches += 1
                total_tickers_processed += processed_count
                print(f"✅ Lote {batch_num}: {processed_count} tickers procesados")
            except Exception as batch_e:
                failed_batches += 1
                print(f"❌ Error en lote {batch_num}: {batch_e}")
                # No continuar con 'continue' implícito, dejar que el loop siga
                continue # Explícito

        print(f"Resumen de descarga: {successful_batches} lotes exitosos, {failed_batches} lotes fallidos")
        print(f"Total tickers procesados: {total_tickers_processed}")
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
