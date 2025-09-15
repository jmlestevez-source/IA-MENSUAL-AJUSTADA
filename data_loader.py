import pandas as pd
import requests
from io import StringIO
import os
import random
import time
from datetime import datetime, timedelta
import numpy as np
import re
from dateutil import parser

# Directorio para datos
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Cache para datos históricos
_historical_cache = {}

def parse_wikipedia_date(date_str):
    """Parsea fechas de Wikipedia en diferentes formatos"""
    if pd.isna(date_str) or not date_str or str(date_str).lower() in ['nan', 'none', '']:
        return None
    
    date_str = str(date_str).strip()
    
    try:
        # Intentar parsing directo con dateutil
        parsed_date = parser.parse(date_str, fuzzy=True)
        return parsed_date.date()
    except:
        try:
            # Patrones específicos para fechas del S&P 500
            # Ejemplo: "September 22, 2025"
            month_map = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12,
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
                'oct': 10, 'nov': 11, 'dec': 12
            }
            
            # Limpiar y dividir
            clean_date = re.sub(r'[^\w\s,]', ' ', date_str.lower())
            parts = clean_date.split()
            
            if len(parts) >= 3:
                # Buscar mes
                month = None
                for part in parts:
                    if part.replace(',', '') in month_map:
                        month = month_map[part.replace(',', '')]
                        break
                
                # Buscar día y año
                numbers = [int(re.findall(r'\d+', part)[0]) for part in parts if re.findall(r'\d+', part)]
                
                if month and len(numbers) >= 2:
                    # Determinar cuál es día y cuál es año
                    day = min(numbers)  # El número menor suele ser el día
                    year = max(numbers)  # El número mayor suele ser el año
                    
                    if day <= 31 and year >= 1900:
                        from datetime import date
                        return date(year, month, day)
            
            return None
        except:
            return None

def get_sp500_historical_changes():
    """Obtiene los cambios históricos del S&P 500 desde Wikipedia"""
    print("Descargando cambios históricos del S&P 500...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        
        # Buscar la tabla de cambios históricos (normalmente es la tercera tabla)
        changes_df = None
        
        # Buscar tabla que contenga la estructura específica del S&P 500
        for i, table in enumerate(tables):
            print(f"Analizando tabla {i} con shape: {table.shape}")
            
            # Verificar si es tabla multi-nivel
            if hasattr(table.columns, 'nlevels') and table.columns.nlevels > 1:
                print(f"Tabla {i} tiene {table.columns.nlevels} niveles de columnas")
                # Buscar la estructura específica: Date, Added, Removed
                level_0 = [str(col[0]).lower() for col in table.columns]
                if any('date' in col or 'effective' in col for col in level_0) and 'added' in level_0 and 'removed' in level_0:
                    changes_df = table
                    print(f"✅ Encontrada tabla de cambios S&P 500 en posición {i}")
                    break
            else:
                # Tabla de nivel simple, verificar contenido
                col_text = ' '.join([str(col) for col in table.columns]).lower()
                if ('effective' in col_text or 'date' in col_text) and 'added' in col_text and 'removed' in col_text:
                    changes_df = table
                    print(f"✅ Encontrada tabla de cambios S&P 500 (nivel simple) en posición {i}")
                    break
        
        if changes_df is None:
            print("❌ No se encontró tabla de cambios para S&P 500")
            # Intentar con la tabla por defecto (índice 2)
            if len(tables) >= 3:
                changes_df = tables[2]
                print("Usando tabla 2 por defecto")
            else:
                return pd.DataFrame()
        
        print(f"Procesando tabla con columnas: {changes_df.columns.tolist()}")
        
        # Procesar datos
        changes_clean = []
        removed_tickers_info = []  # Para generar CSV
        
        print(f"Procesando {len(changes_df)} filas de cambios...")
        
        # Identificar columnas correctamente
        if hasattr(changes_df.columns, 'nlevels') and changes_df.columns.nlevels > 1:
            # Tabla multi-nivel
            date_col = changes_df.columns[0]  # Primera columna (fecha)
            added_ticker_col = changes_df.columns[1]  # Segunda columna (Added Ticker)
            removed_ticker_col = changes_df.columns[3] if len(changes_df.columns) > 3 else None  # Cuarta columna (Removed Ticker)
        else:
            # Tabla simple
            if len(changes_df.columns) >= 4:
                date_col = changes_df.columns[0]
                added_ticker_col = changes_df.columns[1]
                removed_ticker_col = changes_df.columns[3]
            elif len(changes_df.columns) >= 3:
                date_col = changes_df.columns[0]
                added_ticker_col = changes_df.columns[1]
                removed_ticker_col = changes_df.columns[2]
            else:
                return pd.DataFrame()
        
        for idx, row in changes_df.iterrows():
            try:
                # Obtener y parsear fecha
                date_value = row[date_col]
                date_str = str(date_value).strip()
                
                if not date_str or date_str.lower() in ['nan', 'none', '']:
                    continue
                
                date_parsed = parse_wikipedia_date(date_str)
                if date_parsed is None:
                    print(f"⚠️  No se pudo parsear fecha en fila {idx}: '{date_str}'")
                    continue
                
                # Obtener tickers agregados y removidos
                added_ticker = str(row[added_ticker_col]).strip().upper() if pd.notna(row[added_ticker_col]) else None
                removed_ticker = str(row[removed_ticker_col]).strip().upper() if removed_ticker_col and pd.notna(row[removed_ticker_col]) else None
                
                # Procesar ticker agregado
                if added_ticker and added_ticker not in ['NAN', 'NONE', '', 'N/A', 'nan']:
                    # Limpiar ticker
                    clean_added = added_ticker.replace('.', '-').replace(' ', '').strip()
                    if len(clean_added) <= 6 and clean_added and not clean_added.isdigit():
                        changes_clean.append({
                            'Date': date_parsed,
                            'Action': 'Added',
                            'Ticker': clean_added
                        })
                        print(f"✅ Agregado: {clean_added} en {date_parsed}")
                
                # Procesar ticker removido
                if removed_ticker and removed_ticker not in ['NAN', 'NONE', '', 'N/A', 'nan']:
                    # Limpiar ticker
                    clean_removed = removed_ticker.replace('.', '-').replace(' ', '').strip()
                    if len(clean_removed) <= 6 and clean_removed and not clean_removed.isdigit():
                        changes_clean.append({
                            'Date': date_parsed,
                            'Action': 'Removed',
                            'Ticker': clean_removed
                        })
                        
                        # Agregar a lista de removidos para CSV
                        removed_tickers_info.append({
                            'Ticker': clean_removed,
                            'Removed_Date': date_parsed.strftime('%Y-%m-%d'),
                            'Year': date_parsed.year
                        })
                        print(f"✅ Removido: {clean_removed} en {date_parsed}")
                        
            except Exception as e:
                print(f"⚠️  Error procesando fila {idx}: {e}")
                continue
        
        # Generar CSV con tickers removidos
        if removed_tickers_info:
            try:
                removed_df = pd.DataFrame(removed_tickers_info)
                
                # Eliminar duplicados y ordenar
                removed_df = removed_df.drop_duplicates(subset=['Ticker'])
                removed_df = removed_df.sort_values('Removed_Date', ascending=False)
                
                # Guardar CSV
                csv_path = os.path.join(DATA_DIR, 'sp500_removed_tickers.csv')
                removed_df.to_csv(csv_path, index=False)
                print(f"✅ Guardado CSV con {len(removed_df)} tickers removidos en: {csv_path}")
                
                # Mostrar algunos ejemplos
                print("📋 Tickers removidos más recientes:")
                for _, row in removed_df.head(10).iterrows():
                    print(f"   {row['Ticker']} (removido el {row['Removed_Date']})")
                    
            except Exception as e:
                print(f"⚠️  Error generando CSV de removidos: {e}")
        
        # Crear DataFrame final
        if changes_clean:
            result_df = pd.DataFrame(changes_clean)
            result_df = result_df.sort_values('Date', ascending=False)  # Más reciente primero
            
            # Eliminar duplicados
            initial_count = len(result_df)
            result_df = result_df.drop_duplicates(subset=['Date', 'Ticker', 'Action'])
            final_count = len(result_df)
            
            print(f"✅ Procesados {final_count} cambios únicos del S&P 500 (de {initial_count} registros)")
            if final_count != initial_count:
                print(f"   Eliminados {initial_count - final_count} duplicados")
            
            # Estadísticas
            added_count = len(result_df[result_df['Action'] == 'Added'])
            removed_count = len(result_df[result_df['Action'] == 'Removed'])
            print(f"📊 Agregados: {added_count}, Removidos: {removed_count}")
            
            return result_df
        else:
            print("❌ No se pudieron procesar cambios históricos del S&P 500")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error obteniendo cambios históricos S&P 500: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def get_nasdaq100_historical_changes():
    """Obtiene los cambios históricos del NASDAQ-100 desde Wikipedia"""
    print("Descargando cambios históricos del NASDAQ-100...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        
        # Buscar tabla de cambios históricos
        changes_df = None
        for i, table in enumerate(tables):
            cols = [str(col).lower() for col in table.columns]
            print(f"Tabla {i} columnas: {cols}")
            if any('date' in col for col in cols) and any('add' in col or 'remove' in col for col in cols):
                changes_df = table
                print(f"Encontrada tabla de cambios NASDAQ-100 en posición {i}")
                break
        
        if changes_df is None:
            print("No se encontró tabla de cambios para NASDAQ-100")
            # Intentar con la tabla por defecto
            if len(tables) >= 4:
                changes_df = tables[3]  # A menudo es la cuarta tabla
                print("Usando tabla 3 por defecto")
            else:
                return pd.DataFrame()
        
        # Procesar similar a S&P 500
        changes_df.columns = [str(col).strip() for col in changes_df.columns]
        
        # Identificar columnas
        date_col = None
        added_col = None
        removed_col = None
        
        for col in changes_df.columns:
            col_lower = col.lower()
            if 'date' in col_lower and date_col is None:
                date_col = col
            elif ('add' in col_lower or 'ticker' in col_lower) and added_col is None:
                added_col = col
            elif ('remov' in col_lower or 'delete' in col_lower) and removed_col is None:
                removed_col = col
        
        # Fallback si no se encuentran las columnas
        if date_col is None and len(changes_df.columns) > 0:
            date_col = changes_df.columns[0]
        if added_col is None and len(changes_df.columns) > 1:
            added_col = changes_df.columns[1]
        if removed_col is None and len(changes_df.columns) > 2:
            removed_col = changes_df.columns[2]
        
        if not all([date_col]):
            print("No se pudieron identificar las columnas necesarias")
            return pd.DataFrame()
        
        changes_clean = []
        for _, row in changes_df.iterrows():
            try:
                date_str = str(row[date_col]).strip()
                date_parsed = parse_wikipedia_date(date_str)
                
                if date_parsed is None:
                    continue
                
                # Procesar tickers agregados
                if added_col and added_col in changes_df.columns:
                    added_ticker = str(row[added_col]).strip().upper() if pd.notna(row[added_col]) else None
                    if added_ticker and added_ticker not in ['NAN', 'NONE', '', 'N/A', 'nan']:
                        added_ticker = added_ticker.replace('.', '-')
                        if len(added_ticker) <= 6 and not added_ticker.isdigit():
                            changes_clean.append({
                                'Date': date_parsed,
                                'Action': 'Added',
                                'Ticker': added_ticker
                            })
                
                # Procesar tickers removidos
                if removed_col and removed_col in changes_df.columns:
                    removed_ticker = str(row[removed_col]).strip().upper() if pd.notna(row[removed_col]) else None
                    if removed_ticker and removed_ticker not in ['NAN', 'NONE', '', 'N/A', 'nan']:
                        removed_ticker = removed_ticker.replace('.', '-')
                        if len(removed_ticker) <= 6 and not removed_ticker.isdigit():
                            changes_clean.append({
                                'Date': date_parsed,
                                'Action': 'Removed',
                                'Ticker': removed_ticker
                            })
                        
            except Exception as e:
                print(f"Error procesando fila: {e}")
                continue
        
        if changes_clean:
            result_df = pd.DataFrame(changes_clean)
            result_df = result_df.sort_values('Date', ascending=False)
            print(f"Procesados {len(result_df)} cambios históricos del NASDAQ-100")
            return result_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error obteniendo cambios históricos NASDAQ-100: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def get_current_constituents(index_name):
    """Obtiene los constituyentes actuales de un índice"""
    if index_name == "SP500":
        return get_sp500_tickers_from_wikipedia()
    elif index_name == "NDX":
        return get_nasdaq100_tickers_from_wikipedia()
    else:
        raise ValueError(f"Índice {index_name} no soportado")

def get_sp500_tickers_from_wikipedia():
    """Obtiene los tickers actuales del S&P 500"""
    print("Obteniendo constituyentes actuales S&P 500...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        
        df = tables[0]  # Primera tabla contiene constituyentes actuales
        
        symbol_column = None
        for col in df.columns:
            if 'symbol' in str(col).lower() or 'ticker' in str(col).lower():
                symbol_column = col
                break
        
        if symbol_column is None:
            symbol_column = df.columns[0]
        
        tickers = df[symbol_column].astype(str).str.strip().str.upper().tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        tickers = [t for t in tickers if t and t != 'nan' and len(t) <= 6 and not t.isdigit()]
        
        print(f"Obtenidos {len(tickers)} constituyentes actuales S&P 500")
        
        return {
            'tickers': tickers,
            'data': df.to_dict('records'),
            'timestamp': datetime.now().timestamp(),
            'date': datetime.now()
        }
        
    except Exception as e:
        print(f"Error obteniendo constituyentes actuales S&P 500: {e}")
        fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
        return {
            'tickers': fallback_tickers,
            'data': [{'Symbol': t} for t in fallback_tickers],
            'timestamp': datetime.now().timestamp(),
            'date': datetime.now()
        }

def get_nasdaq100_tickers_from_wikipedia():
    """Obtiene los tickers actuales del NASDAQ-100"""
    print("Obteniendo constituyentes actuales NASDAQ-100...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        
        df = None
        # Buscar la tabla que contiene los constituyentes
        for table in tables:
            if len(table.columns) >= 2:  # Debe tener al menos 2 columnas
                ticker_cols = [col for col in table.columns if 'Ticker' in str(col) or 'Symbol' in str(col)]
                if ticker_cols:
                    df = table
                    break
        
        if df is None:
            # Fallback: usar la tercera tabla típicamente
            if len(tables) >= 3:
                df = tables[2]
            else:
                raise ValueError("No se encontró tabla de constituyentes")
        
        ticker_column = None
        for col in df.columns:
            if 'Ticker' in str(col) or 'Symbol' in str(col):
                ticker_column = col
                break
        
        if ticker_column is None:
            ticker_column = df.columns[1]  # Usualmente la segunda columna
        
        tickers = df[ticker_column].astype(str).str.strip().str.upper().tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        tickers = [t for t in tickers if t and t != 'nan' and len(t) <= 6 and not t.isdigit()]
        
        print(f"Obtenidos {len(tickers)} constituyentes actuales NASDAQ-100")
        
        return {
            'tickers': tickers,
            'data': df.to_dict('records'),
            'timestamp': datetime.now().timestamp(),
            'date': datetime.now()
        }
        
    except Exception as e:
        print(f"Error obteniendo constituyentes actuales NASDAQ-100: {e}")
        fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'PEP', 'COST']
        return {
            'tickers': fallback_tickers,
            'data': [{'Symbol': t} for t in fallback_tickers],
            'timestamp': datetime.now().timestamp(),
            'date': datetime.now()
        }

def get_valid_constituents_for_period(index_name, start_date, end_date, changes_df):
    """
    Obtiene los tickers que estuvieron en el índice durante un período específico
    """
    try:
        # Obtener constituyentes actuales
        current_data = get_current_constituents(index_name)
        current_tickers = set(current_data['tickers'])
        
        if changes_df.empty:
            print(f"⚠️  No hay datos históricos para {index_name}, usando constituyentes actuales")
            return current_tickers, current_data['data']
        
        # Convertir fechas
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()
        
        # Empezar con tickers actuales
        valid_tickers = set(current_tickers)
        
        # Procesar cambios para identificar tickers válidos en el período
        # Estrategia: revertir cambios futuros para encontrar estado en start_date
        
        # Cambios después de start_date que afectan la composición
        future_changes = changes_df[changes_df['Date'] > start_date].sort_values('Date', ascending=True)
        
        for _, change in future_changes.iterrows():
            ticker = change['Ticker']
            action = change['Action']
            
            # Revertir los cambios futuros para obtener el estado en start_date
            if action == 'Added':
                # Si fue agregado después de start_date, no estaba en start_date
                valid_tickers.discard(ticker)
            elif action == 'Removed':
                # Si fue removido después de start_date, estaba en start_date
                valid_tickers.add(ticker)
        
        # También considerar cambios antes de start_date que afectan el período
        past_changes = changes_df[changes_df['Date'] <= start_date]
        
        # Para cada ticker que fue removido antes de start_date
        for _, change in past_changes.iterrows():
            ticker = change['Ticker']
            action = change['Action']
            
            if action == 'Removed':
                # Verificar si fue re-agregado después
                was_readded = not future_changes[
                    (future_changes['Ticker'] == ticker) & 
                    (future_changes['Action'] == 'Added')
                ].empty
                
                if not was_readded:
                    # Si fue removido y no re-agregado, no estaba en el período
                    valid_tickers.discard(ticker)
        
        # Crear datos históricos
        historical_data = []
        for ticker in valid_tickers:
            # Buscar fecha de incorporación más reciente
            ticker_additions = changes_df[
                (changes_df['Ticker'] == ticker) & 
                (changes_df['Action'] == 'Added') &
                (changes_df['Date'] <= start_date)
            ].sort_values('Date', ascending=False)
            
            added_date = ticker_additions['Date'].iloc[0] if not ticker_additions.empty else None
            
            historical_data.append({
                'ticker': ticker,
                'added': added_date.strftime('%Y-%m-%d') if added_date else 'Unknown',
                'in_current': ticker in current_tickers,
                'status': 'Historical constituent'
            })
        
        return list(valid_tickers), historical_data
        
    except Exception as e:
        print(f"Error en get_valid_constituents_for_period: {e}")
        # Fallback
        current_data = get_current_constituents(index_name)
        return current_data['tickers'], current_data['data']

def get_constituents_at_date(index_name, start_date, end_date):
    """
    Obtiene constituyentes históricos válidos para el rango de fechas
    """
    cache_key = f"{index_name}_{start_date}_{end_date}"
    
    if cache_key in _historical_cache:
        print(f"Usando cache para {index_name}")
        return _historical_cache[cache_key], None
    
    try:
        # Obtener cambios históricos
        if index_name == "SP500":
            changes_df = get_sp500_historical_changes()
        elif index_name == "NDX":
            changes_df = get_nasdaq100_historical_changes()
        else:
            raise ValueError(f"Índice {index_name} no soportado")
        
        # Obtener tickers válidos para el período
        valid_tickers, historical_data = get_valid_constituents_for_period(
            index_name, start_date, end_date, changes_df
        )
        
        # Limpiar tickers
        valid_tickers = [t for t in valid_tickers if t and len(t) <= 6 and not t.isdigit()]
        
        result = {
            'tickers': valid_tickers,
            'data': historical_data,
            'historical_data_available': not changes_df.empty,
            'changes_processed': len(changes_df) if not changes_df.empty else 0,
            'period_start': start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date),
            'period_end': end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date),
            'note': f'Historical constituents for {index_name}'
        }
        
        _historical_cache[cache_key] = result
        return result, None
        
    except Exception as e:
        error_msg = f"Error obteniendo constituyentes históricos para {index_name}: {e}"
        print(error_msg)
        
        # Fallback a constituyentes actuales
        try:
            current_data = get_current_constituents(index_name)
            fallback_result = {
                'tickers': current_data['tickers'],
                'data': current_data['data'],
                'historical_data_available': False,
                'note': f'Fallback to current constituents due to error: {str(e)}'
            }
            return fallback_result, f"Warning: {error_msg}"
        except:
            return None, error_msg

def generate_removed_tickers_summary():
    """Genera resumen completo de tickers removidos de ambos índices"""
    try:
        all_removed = []
        
        # S&P 500
        sp500_changes = get_sp500_historical_changes()
        if not sp500_changes.empty:
            sp500_removed = sp500_changes[sp500_changes['Action'] == 'Removed'].copy()
            sp500_removed['Index'] = 'SP500'
            all_removed.append(sp500_removed)
        
        # NASDAQ-100
        ndx_changes = get_nasdaq100_historical_changes()
        if not ndx_changes.empty:
            ndx_removed = ndx_changes[ndx_changes['Action'] == 'Removed'].copy()
            ndx_removed['Index'] = 'NDX'
            all_removed.append(ndx_removed)
        
        if all_removed:
            # Combinar todos los removidos
            combined_removed = pd.concat(all_removed, ignore_index=True)
            
            # Crear resumen por ticker
            summary = combined_removed.groupby('Ticker').agg({
                'Date': ['min', 'max', 'count'],
                'Index': lambda x: ', '.join(sorted(set(x)))
            }).reset_index()
            
            # Aplanar columnas
            summary.columns = ['Ticker', 'First_Removed', 'Last_Removed', 'Times_Removed', 'Indices']
            
            # Ordenar por fecha más reciente
            summary = summary.sort_values('Last_Removed', ascending=False)
            
            # Guardar CSV
            csv_path = os.path.join(DATA_DIR, 'all_removed_tickers_summary.csv')
            summary.to_csv(csv_path, index=False)
            
            print(f"✅ Generado resumen de tickers removidos: {csv_path}")
            print(f"📊 Total de tickers únicos removidos: {len(summary)}")
            
            return summary
        else:
            return pd.DataFrame()
        
    except Exception as e:
        print(f"❌ Error generando resumen de removidos: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def download_prices(tickers, start_date, end_date):
    """
    Carga precios desde archivos CSV en la carpeta data/
    Formato esperado: data/AAPL.csv, data/MSFT.csv, etc.
    """
    prices_data = {}
    
    # Normalizar entrada de tickers
    if isinstance(tickers, dict) and 'tickers' in tickers:
        ticker_list = tickers['tickers']
    elif isinstance(tickers, (list, tuple)):
        ticker_list = list(tickers)
    elif isinstance(tickers, str):
        ticker_list = [tickers]
    else:
        ticker_list = []
    
    # Limpiar y normalizar tickers
    ticker_list = [str(t).strip().upper().replace('.', '-') for t in ticker_list]
    ticker_list = [t for t in ticker_list if t and not t.isdigit() and len(t) <= 6]
    ticker_list = list(dict.fromkeys(ticker_list))  # Eliminar duplicados
    
    if not ticker_list:
        return pd.DataFrame()
    
    print(f"Cargando {len(ticker_list)} tickers desde CSV...")
    
    for ticker in ticker_list:
        csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if os.path.exists(csv_path):
            try:
                # Leer CSV con Date como índice
                df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
                
                # Filtrar por rango de fechas
                if isinstance(start_date, datetime):
                    start_date = start_date.date()
                if isinstance(end_date, datetime):
                    end_date = end_date.date()
                    
                df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
                
                if not df.empty:
                    # Prioridad: Adj Close → Close → primera columna numérica
                    price_series = None
                    for col in ["Adj Close", "Close"]:
                        if col in df.columns:
                            price_series = df[col]
                            break
                    
                    if price_series is None:
                        # Usar primera columna numérica si no hay Close/Adj Close
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            price_series = df[numeric_cols[0]]
                    
                    if price_series is not None:
                        price_series.name = ticker  # Nombrar la serie con el ticker
                        prices_data[ticker] = price_series
                else:
                    print(f"  No hay datos para {ticker} en el rango de fechas")
            except Exception as e:
                print(f"  Error cargando {ticker}: {e}")
        else:
            print(f"  Archivo no encontrado: {csv_path}")
    
    # Combinar en DataFrame
    if prices_data:
        prices_df = pd.DataFrame(prices_data)
        # Rellenar valores faltantes hacia adelante y hacia atrás
        prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
        return prices_df
    else:
        return pd.DataFrame()
