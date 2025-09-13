import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Crear directorio data
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_sp500_tickers():
    """Obtener tickers del S&P 500 desde Wikipedia"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    return df['Symbol'].tolist()

def get_nasdaq100_tickers():
    """Obtener tickers del Nasdaq-100 desde Wikipedia"""
    url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
    tables = pd.read_html(url)
    df = tables[4]  # Ajusta el índice si es necesario
    return df['Ticker'].tolist()

def download_ticker_data(ticker):
    """Descargar datos de un ticker"""
    try:
        df = yf.download(ticker, start="2010-01-01", auto_adjust=False)
        if df.empty:
            print(f"⚠️ No hay datos para {ticker}")
            return None
        return df
    except Exception as e:
        print(f"❌ Error descargando {ticker}: {e}")
        return None

def save_to_csv(ticker, df):
    """Guardar datos en CSV con formato específico"""
    # Renombrar columnas
    df = df.rename(
        columns={
            "Adj Close": "Price",
            "Close": "Close",
            "High": "High",
            "Low": "Low",
            "Open": "Open",
            "Volume": "Volume",
        }
    )[["Price", "Close", "High", "Low", "Open", "Volume"]]
    
    # Añadir fila con ticker
    header_df = pd.DataFrame([[ticker]*len(df.columns)], columns=df.columns, index=["Ticker"])
    empty_row = pd.DataFrame([[""]*len(df.columns)], columns=df.columns, index=["Date"])
    df_out = pd.concat([header_df, empty_row, df])
    
    # Guardar
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df_out.to_csv(file_path, index=True)
    return file_path

def main():
    print("🔄 Obteniendo tickers...")
    
    # Obtener tickers
    try:
        sp500_tickers = get_sp500_tickers()
        print(f"✅ S&P 500: {len(sp500_tickers)} tickers")
    except Exception as e:
        print(f"❌ Error S&P 500: {e}")
        sp500_tickers = []
    
    try:
        nasdaq_tickers = get_nasdaq100_tickers()
        print(f"✅ Nasdaq-100: {len(nasdaq_tickers)} tickers")
    except Exception as e:
        print(f"❌ Error Nasdaq-100: {e}")
        nasdaq_tickers = []
    
    # Combinar y eliminar duplicados
    all_tickers = list(set(sp500_tickers + nasdaq_tickers + ["SPY", "QQQ"]))
    print(f"📊 Total tickers únicos: {len(all_tickers)}")
    
    # Descargar datos
    downloaded = 0
    for i, ticker in enumerate(sorted(all_tickers), 1):
        print(f"{i}/{len(all_tickers)} - {ticker}")
        
        df = download_ticker_data(ticker)
        if df is not None:
            file_path = save_to_csv(ticker, df)
            print(f"✅ Guardado {file_path} ({len(df)} filas)")
            downloaded += 1
    
    print(f"\n📈 Actualización completada: {datetime.now()}")
    print(f"📊 Tickers procesados: {downloaded}/{len(all_tickers)}")

if __name__ == "__main__":
    main()
