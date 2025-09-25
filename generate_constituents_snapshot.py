# -*- coding: utf-8 -*-
# generate_constituents_snapshot.py
# Genera snapshots limpios de constituyentes actuales:
#   data/constituents_sp500.csv  (Symbol,Name)
#   data/constituents_ndx.csv    (Symbol,Name)
import pandas as pd
import requests
from io import StringIO
import os

def _read_html_with_ua(url, attrs=None):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0 Safari/537.36"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text), attrs=attrs)

def save_sp500(path="data/constituents_sp500.csv"):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    dfs = _read_html_with_ua(url, attrs={"id": "constituents"})
    df = dfs[0] if dfs else None
    if df is None:
        dfs = _read_html_with_ua(url)
        for t in dfs:
            cols = [str(c).strip().lower() for c in t.columns]
            if any(c in ("symbol", "ticker") for c in cols) and any(c in ("security", "company", "company name") for c in cols):
                df = t; break
    if df is None:
        raise RuntimeError("No se pudo leer la tabla de SP500")
    sym_col = next((c for c in df.columns if str(c).strip().lower() in ("symbol", "ticker")), None)
    name_col = next((c for c in df.columns if str(c).strip().lower() in ("security", "company", "company name")), None)
    if not sym_col: raise RuntimeError("Tabla SP500 sin columna Symbol/Ticker")
    out = pd.DataFrame({"Symbol": df[sym_col].astype(str).str.upper().str.replace(".", "-", regex=False)})
    if name_col:
        out["Name"] = df[name_col].astype(str)
    out.to_csv(path, index=False)
    print(f"✅ Guardado {path} ({len(out)})")

def save_ndx(path="data/constituents_ndx.csv"):
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    dfs = _read_html_with_ua(url, attrs={"id": "constituents"})
    df = dfs[0] if dfs else None
    if df is None:
        dfs = _read_html_with_ua(url)
        for t in dfs:
            cols = [str(c).strip().lower() for c in t.columns]
            if any(("ticker" in c or "symbol" in c) for c in cols) and any("company" in c for c in cols):
                df = t; break
    if df is None:
        raise RuntimeError("No se pudo leer la tabla de NDX")
    sym_col = next((c for c in df.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()), None)
    name_col = next((c for c in df.columns if "company" in str(c).lower()), None)
    if not sym_col: raise RuntimeError("Tabla NDX sin columna Ticker/Symbol")
    out = pd.DataFrame({"Symbol": df[sym_col].astype(str).str.upper().str.replace(".", "-", regex=False)})
    if name_col:
        out["Name"] = df[name_col].astype(str)
    out.to_csv(path, index=False)
    print(f"✅ Guardado {path} ({len(out)}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    save_sp500()
    save_ndx()
