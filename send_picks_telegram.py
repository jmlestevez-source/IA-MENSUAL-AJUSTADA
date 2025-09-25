# -*- coding: utf-8 -*-
# send_picks_telegram.py
import os
import sys
import json
import html
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from io import StringIO

# Reutilizamos tus módulos
from data_loader import download_prices
from backtest import inertia_score

# --------- Helpers Wikipedia (símbolo -> nombre) ---------
import requests as _req

def _read_html_with_ua(url, attrs=None):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/122.0 Safari/537.36"
        }
        resp = _req.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        html_txt = resp.text
        return pd.read_html(StringIO(html_txt), attrs=attrs)
    except Exception:
        return []

def normalize_symbol(t):
    return str(t).strip().upper().replace(".", "-")

def get_sp500_name_map():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _read_html_with_ua(url, attrs={"id": "constituents"})
    df = tables[0] if tables else None
    if df is None:
        tables = _read_html_with_ua(url)
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if any(c in ("symbol", "ticker") for c in cols) and any(c in ("security", "company", "company name") for c in cols):
                df = t
                break
    if df is None:
        return {}
    sym_col = next((c for c in df.columns if str(c).strip().lower() in ("symbol", "ticker")), None)
    sec_col = next((c for c in df.columns if str(c).strip().lower() in ("security", "company", "company name")), None)
    if not sym_col or not sec_col:
        return {}
    s = df[[sym_col, sec_col]].copy()
    s.columns = ["Symbol", "Name"]
    s["Symbol"] = s["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
    return dict(zip(s["Symbol"], s["Name"]))

def get_ndx_name_map():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = _read_html_with_ua(url, attrs={"id": "constituents"})
    df = tables[0] if tables else None
    if df is None:
        tables = _read_html_with_ua(url)
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if any(("ticker" in c or "symbol" in c) for c in cols) and any("company" in c for c in cols):
                df = t
                break
    if df is None:
        return {}
    sym_col = next((c for c in df.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()), None)
    name_col = next((c for c in df.columns if "company" in str(c).lower()), None)
    if not sym_col or not name_col:
        return {}
    s = df[[sym_col, name_col]].copy()
    s.columns = ["Symbol", "Name"]
    s["Symbol"] = s["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
    return dict(zip(s["Symbol"], s["Name"]))

def get_current_constituents_set(index_choice):
    if index_choice.upper() in ("SP500", "S&P500", "S&P 500"):
        return set(map(normalize_symbol, get_sp500_name_map().keys()))
    elif index_choice.upper() in ("NDX", "NASDAQ-100", "NASDAQ100"):
        return set(map(normalize_symbol, get_ndx_name_map().keys()))
    else:
        s1 = set(map(normalize_symbol, get_sp500_name_map().keys()))
        s2 = set(map(normalize_symbol, get_ndx_name_map().keys()))
        return s1 | s2

def get_name_map(index_choice):
    if index_choice.upper() in ("SP500", "S&P500", "S&P 500"):
        return get_sp500_name_map()
    elif index_choice.upper() in ("NDX", "NASDAQ-100", "NASDAQ100"):
        return get_ndx_name_map()
    else:
        m = get_sp500_name_map()
        m2 = get_ndx_name_map()
        m.update(m2)
        return m

# ---------- Lógica de picks ----------
def compute_prospective_picks(index_choice="BOTH", top_n=5, corte=680, start=None, end=None):
    # Fechas
    if end is None:
        end = date.today()
    if start is None:
        # suficiente histórico, sin exagerar
        start = date(2007, 1, 1)

    # Universo actual
    current_set = get_current_constituents_set(index_choice)
    # Intersección con CSVs locales
    import glob, os
    avail = set(os.path.basename(p)[:-4].upper().replace('.', '-') for p in glob.glob('data/*.csv'))
    universe = sorted(current_set & avail)

    if not universe:
        return [], [], [], {}, pd.DataFrame()

    # Precios
    prices_df = download_prices(universe, start, end, load_full_data=False)
    if isinstance(prices_df, tuple):
        # por compatibilidad si cambia firma
        prices_df = prices_df[0]
    if prices_df is None or prices_df.empty:
        return [], [], [], {}, pd.DataFrame()

    # Filtrar solo universe con precios cargados
    universe = [t for t in universe if t in prices_df.columns]
    if not universe:
        return [], [], [], {}, pd.DataFrame()
    prices_df = prices_df[universe]

    # Scores
    scores = inertia_score(prices_df, corte=corte, ohlc_data=None)
    if not scores or "ScoreAdjusted" not in scores or "InerciaAlcista" not in scores:
        return [], [], [], {}, prices_df

    score_df = scores["ScoreAdjusted"]
    inercia_df = scores["InerciaAlcista"]
    if score_df.empty or inercia_df.empty:
        return [], [], [], {}, prices_df

    last_scores = score_df.iloc[-1].dropna()
    last_inercia = inercia_df.iloc[-1]

    # Construir lista de candidatos
    candidates = []
    last_prices = prices_df.resample("D").last().iloc[-1] if not isinstance(prices_df.index, pd.DatetimeIndex) else prices_df.iloc[-1]
    for t in last_scores.index:
        if t in last_inercia.index:
            ine = float(last_inercia.get(t))
            sca = float(last_scores.get(t))
            if ine >= corte and sca > 0 and not np.isnan(sca):
                pr = float(last_prices.get(t)) if t in last_prices.index else np.nan
                candidates.append({"ticker": t, "inercia": ine, "score_adj": sca, "price": pr})

    if not candidates:
        return [], [], [], {}, prices_df

    candidates = sorted(candidates, key=lambda x: x["score_adj"], reverse=True)
    picks = candidates[:min(top_n, len(candidates))]

    # Cargar último estado para Entradas/Salidas/Mantener
    state_path = f"data/last_picks_{index_choice.upper()}.json"
    prev_set = set()
    try:
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
                prev_set = set(prev.get("tickers", []))
    except Exception:
        prev_set = set()

    new_set = set([p["ticker"] for p in picks])
    comprar = sorted(new_set - prev_set)
    vender = sorted(prev_set - new_set)
    mantener = sorted(prev_set & new_set)

    # Guardar nuevo estado
    try:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.utcnow().isoformat(),
                "index": index_choice,
                "tickers": sorted(list(new_set))
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return picks, comprar, vender, mantener, prices_df

def build_telegram_message(index_choice, picks, comprar, vender, mantener, name_map, as_of_date=None):
    idx_label = "SP500" if index_choice.upper().startswith("SP") else ("NDX" if index_choice.upper().startswith("NDX") else "SP500 + NDX")
    as_of = as_of_date or (datetime.utcnow().date())
    title = f"🧠 Señales prospectivas IA Mensual ({idx_label})\n📅 Cierre: {as_of.isoformat()}"

    # Tabla monoespaciada con HTML <pre>
    header = f"{'#':>2} {'Ticker':<7} {'Nombre':<28} {'Inercia':>8} {'ScoreAdj':>9} {'Precio':>10}"
    rows = []
    for i, p in enumerate(picks, 1):
        nm = name_map.get(p['ticker'], p['ticker'])
        nm = nm if len(nm) <= 28 else (nm[:25] + "...")
        rows.append(f"{i:>2} {p['ticker']:<7} {nm:<28} {p['inercia']:>8.0f} {p['score_adj']:>9.2f} {p['price']:>10.2f}")
    table = "\n".join([header] + rows) if rows else "Sin candidatos"

    def fmt_list(lst, emoji):
        if not lst:
            return f"{emoji} -"
        out = []
        for t in lst:
            nm = name_map.get(t, t)
            out.append(f"{t} — {nm}")
        return f"{emoji} " + "\n".join(out)

    comprar_txt = fmt_list(comprar, "✅ Comprar")
    vender_txt = fmt_list(vender, "❌ Vender")
    mantener_txt = fmt_list(mantener, "♻️ Mantener")

    # Escapar HTML donde toca (nombres, pero dentro de <pre> no hace falta)
    msg = (
        f"<b>{html.escape(title)}</b>\n\n"
        f"<b>Top {len(picks)}</b>\n"
        f"<pre>{html.escape(table)}</pre>\n\n"
        f"{html.escape(comprar_txt)}\n\n"
        f"{html.escape(vender_txt)}\n\n"
        f"{html.escape(mantener_txt)}\n\n"
        f"— Bot IA Mensual"
    )
    return msg

def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, data={
            "chat_id": str(chat_id),
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }, timeout=30)
        resp.raise_for_status()
        return True, resp.json()
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Enviar picks prospectivos a Telegram")
    parser.add_argument("--index", default="BOTH", choices=["SP500", "NDX", "BOTH"], help="Índice: SP500 | NDX | BOTH")
    parser.add_argument("--top", type=int, default=5, help="Número de picks (default 5)")
    parser.add_argument("--corte", type=int, default=680, help="Corte de inercia (default 680)")
    parser.add_argument("--start", type=str, default="2007-01-01", help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Fecha fin (YYYY-MM-DD, default hoy)")
    parser.add_argument("--token", type=str, default=os.getenv("TELEGRAM_BOT_TOKEN"), help="Bot token (o env TELEGRAM_BOT_TOKEN)")
    parser.add_argument("--chat", type=str, default=os.getenv("TELEGRAM_CHAT_ID"), help="Chat ID (o env TELEGRAM_CHAT_ID)")
    args = parser.parse_args()

    if not args.token or not args.chat:
        print("❌ Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID (usa secrets en GitHub Actions).")
        sys.exit(1)

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()

    picks, comprar, vender, mantener, prices_df = compute_prospective_picks(
        index_choice=args.index,
        top_n=args.top,
        corte=args.corte,
        start=start,
        end=end
    )

    name_map = get_name_map(args.index)
    msg = build_telegram_message(args.index, picks, comprar, vender, mantener, name_map, as_of_date=end)

    ok, info = send_telegram(args.token, args.chat, msg)
    if ok:
        print("✅ Mensaje enviado a Telegram")
    else:
        print(f"❌ Error enviando a Telegram: {info}")
        sys.exit(1)

if __name__ == "__main__":
    main()
