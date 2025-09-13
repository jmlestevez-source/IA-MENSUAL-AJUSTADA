# scripts/update_data.py

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

DATA_DIR = "data"
TICKERS = [f.replace(".csv", "") for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

def update_ticker(ticker):
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

    # Última fecha registrada
    last_date = df.index[-1]
    today = datetime.today().date()

    # Si ya está actualizado, salir
    if last_date.date() >= today:
        print(f"{ticker}: Ya está actualizado.")
        return

    # Descargar solo el último día
    start_date = last_date + timedelta(days=1)
    end_date = today

    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if data.empty:
        print(f"{ticker}: No hay nuevos datos.")
        return

    # Asegurar formato
    data.index = pd.to_datetime(data.index)
    data = data.rename_axis("Date")

    # Concatenar y guardar
    df = pd.concat([df, data])
    df.to_csv(file_path)
    print(f"{ticker}: Actualizado.")

if __name__ == "__main__":
    for ticker in TICKERS:
        try:
            update_ticker(ticker)
        except Exception as e:
            print(f"Error actualizando {ticker}: {e}")
