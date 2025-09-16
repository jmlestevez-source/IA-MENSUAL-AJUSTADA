import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_stock_data(symbol, period="1y"):
    """
    Carga datos de acciones usando yfinance
    Args:
        symbol (str): Símbolo de la acción
        period (str): Período de datos ("1y", "2y", "5y", "10y", "max")
    Returns:
        pandas.DataFrame: Datos de la acción
    """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            raise ValueError(f"No se encontraron datos para {symbol}")
        
        data.reset_index(inplace=True)
        
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Columna requerida {col} no encontrada")
        
        data['Date'] = pd.to_datetime(data['Date'])
        
        return data
        
    except Exception as e:
        raise Exception(f"Error cargando datos de {symbol}: {str(e)}")


def load_multiple_stocks(symbols, period="1y"):
    """
    Carga datos de múltiples acciones
    Args:
        symbols (list): Lista de símbolos
        period (str): Período de datos
    Returns:
        dict: Diccionario con datos de cada acción
    """
    stocks_data = {}
    for symbol in symbols:
        try:
            stocks_data[symbol] = load_stock_data(symbol, period)
        except Exception as e:
            print(f"Error con {symbol}: {str(e)}")
            stocks_data[symbol] = None
    return stocks_data


def get_stock_info(symbol):
    """
    Obtiene información básica de una acción
    Args:
        symbol (str): Símbolo de la acción
    Returns:
        dict: Información de la acción
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'symbol': symbol,
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0),
            '52_week_high': info.get('fiftyTwoWeekHigh', 0),
            '52_week_low': info.get('fiftyTwoWeekLow', 0)
        }
    except Exception as e:
        return {
            'symbol': symbol,
            'name': 'N/A',
            'sector': 'N/A',
            'industry': 'N/A',
            'market_cap': 0,
            'pe_ratio': 0,
            'dividend_yield': 0,
            '52_week_high': 0,
            '52_week_low': 0,
            'error': str(e)
        }


def load_custom_data(file_path):
    """
    Carga datos desde un archivo CSV personalizado
    Args:
        file_path (str): Ruta al archivo CSV
    Returns:
        pandas.DataFrame: Datos cargados
    """
    try:
        data = pd.read_csv(file_path)
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                similar_cols = [c for c in data.columns if col.lower() in c.lower()]
                if similar_cols:
                    data.rename(columns={similar_cols[0]: col}, inplace=True)
                else:
                    raise ValueError(f"Columna requerida {col} no encontrada")
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)
        return data
    except Exception as e:
        raise Exception(f"Error cargando datos personalizados: {str(e)}")


def validate_data_integrity(data):
    """
    Valida la integridad de los datos
    Args:
        data (pandas.DataFrame): Datos a validar
    Returns:
        tuple: (bool, str) - (es_válido, mensaje)
    """
    if data is None:
        return False, "No hay datos"
    if data.empty:
        return False, "Los datos están vacíos"
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in data.columns:
            return False, f"Falta columna requerida: {col}"
    price_columns = ['Open', 'High', 'Low', 'Close']
    for col in price_columns:
        if data[col].isna().any():
            return False, f"Hay valores NaN en la columna {col}"
    for col in price_columns:
        if (data[col] <= 0).any():
            return False, f"Hay valores no positivos en la columna {col}"
    if not (data['High'] >= data['Low']).all():
        return False, "Hay registros donde High < Low"
    if not data['Date'].is_monotonic_increasing:
        return False, "Las fechas no están ordenadas cronológicamente"
    return True, "Datos válidos"


def get_available_periods():
    """
    Retorna los períodos disponibles para descarga de datos
    Returns:
        dict: Diccionario con períodos disponibles
    """
    return {
        "1 mes": "1mo",
        "3 meses": "3mo",
        "6 meses": "6mo",
        "1 año": "1y",
        "2 años": "2y",
        "5 años": "5y",
        "10 años": "10y",
        "Máximo": "max"
    }


if __name__ == "__main__":
    try:
        data = load_stock_data("AAPL", "1y")
        print(f"Datos cargados: {len(data)} registros")
        print(data.head())
        is_valid, message = validate_data_integrity(data)
        print(f"Validación: {is_valid} - {message}")
        info = get_stock_info("AAPL")
        print(f"Información: {info['name']} - Sector: {info['sector']}")
    except Exception as e:
        print(f"Error: {str(e)}")
