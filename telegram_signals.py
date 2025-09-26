# -*- coding: utf-8 -*-
# send_picks_telegram.py
# Señales del sistema para el próximo mes con filtros de mercado ACTIVOS (ROC12 y SMA10)
# y fallback IEF/BIL si el filtro está activo. Coinciden con lo que verías en la app en un día "normal".
# Usa el universo validado por data_loader (regla "último evento <= fecha manda").
#
# Requisitos:
#   - Variables/Secrets: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
#   - data_loader.py y backtest.py en el mismo repo
#
# Ejemplo CLI:
#   python send_picks_telegram.py --index BOTH --top 5 --corte 680
import os
import sys
import json
import html
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime
from io import StringIO
import glob

from data_loader import download_prices, get_constituents_at_date
from backtest import inertia_score  # usa la misma lógica de indicadores que la app

# ------------------- Utilidades Wikipedia (nombres) -------------------
def _read_html_with_ua(url, attrs=None):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/122.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return pd.read_html(StringIO(resp.text), attrs=attrs)
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

def index_to_loader_name(ix):
    k = ix.strip().upper()
    if k in ("SP500", "S&P500", "S&P 500"):
        return "SP500"
    if k in ("NDX", "NASDAQ-100", "NASDAQ100"):
        return "NDX"
    return "Ambos (SP500 + NDX)"

def state_key_from_index(ix):
    k = ix.strip().upper()
    if k in ("SP500", "S&P500", "S&P 500"):
        return "SP500"
    if k in ("NDX", "NASDAQ-100", "NASDAQ100"):
        return "NDX"
    return "BOTH"

# ------------------- Telegram -------------------
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

# ------------------- Señales con filtros + fallback safety -------------------
def compute_signals_with_filters(index_choice="BOTH", top_n=5, corte=680, start=None, end=None, warmup_months=24):
    # Fechas
    if end is None:
        end = date.today()
    if start is None:
        start = date(2007, 1, 1)
    warmup_start = (pd.Timestamp(start) - pd.DateOffset(months=warmup_months)).date()

    # SPY mensual para determinar prev_date y filtros
    spy_obj = download_prices(["SPY"], warmup_start, end, load_full_data=False)
    spy_df = spy_obj[0] if isinstance(spy_obj, tuple) else spy_obj
    if not isinstance(spy_df, pd.DataFrame) or "SPY" not in spy_df.columns:
        return [], {"reason": "No SPY data"}, {}
    spy_m = spy_df["SPY"].resample("ME").last().dropna()
    if spy_m.empty:
        return [], {"reason": "No SPY monthly data"}, {}

    prev_date = spy_m.index[-1]

    # Universo por regla "último evento <= fecha manda" y CSVs disponibles
    loader_name = index_to_loader_name(index_choice)
    res, err = get_constituents_at_date(loader_name, start, prev_date)
    universe = res.get("tickers", []) if isinstance(res, dict) else []
    # Excluir safety del universo normal
    universe = [t for t in universe if t not in ("IEF", "BIL")]
    if not universe:
        return [], {"reason": "Universo vacío"}, {"spy_m": spy_m}

    # Cargar precios del universo (hasta prev_date)
    univ_obj = download_prices(universe, warmup_start, prev_date, load_full_data=False)
    univ_df = univ_obj[0] if isinstance(univ_obj, tuple) else univ_obj
    if not isinstance(univ_df, pd.DataFrame) or univ_df.empty:
        return [], {"reason": "No se pudieron cargar precios del universo"}, {"spy_m": spy_m}
    # Filtrar a tickers que realmente cargaron
    universe = [t for t in universe if t in univ_df.columns]
    if not universe:
        return [], {"reason": "Universo sin precios válidos"}, {"spy_m": spy_m}

    prices_m = univ_df[universe].resample("ME").last().dropna(how="all")
    if prices_m.empty or prev_date not in prices_m.index:
        # Si el último mes cotizado de SPY no existe en el universo, usa el común
        prev_date = prices_m.index[-1] if not prices_m.empty else prev_date

    # Safety mensual (IEF/BIL)
    saf_obj = download_prices(["IEF", "BIL"], warmup_start, prev_date, load_full_data=False)
    saf_df = saf_obj[0] if isinstance(saf_obj, tuple) else saf_obj
    safety_m = saf_df.resample("ME").last() if isinstance(saf_df, pd.DataFrame) and not saf_df.empty else None

    # Scores (idéntico a la app)
    scores = inertia_score(prices_m, corte=corte, ohlc_data=None)
    if not scores or "ScoreAdjusted" not in scores or "InerciaAlcista" not in scores:
        return [], {"reason": "No se pudieron calcular scores", "prev_date": prev_date}, {"spy_m": spy_m, "safety_m": safety_m}
    score_df = scores["ScoreAdjusted"]
    inercia_df = scores["InerciaAlcista"]
    raw_df = scores.get("RawScoreAdjusted", None)

    # Filtros de mercado ACTIVOS por defecto (ROC12 y SMA10)
    market_filter_active = False
    try:
        sp = spy_m.loc[:prev_date]
        if len(sp) >= 13:
            roc12 = ((sp.iloc[-1] - sp.iloc[-13]) / sp.iloc[-13]) * 100 if sp.iloc[-13] != 0 else 0
        else:
            roc12 = 0
        sma10_flag = False
        if len(sp) >= 10:
            sma10 = sp.iloc[-10:].mean()
            sma10_flag = (sp.iloc[-1] < sma10)
        if (roc12 < 0) or sma10_flag:
            market_filter_active = True
    except Exception:
        market_filter_active = False

    # Si filtro activo => fallback safety: elegir el mejor entre IEF y BIL por retorno 1m
    if market_filter_active:
        safety_candidates = []
        if safety_m is not None and prev_date in safety_m.index:
            for st in ("IEF", "BIL"):
                if st in safety_m.columns:
                    try:
                        spx = safety_m[st].dropna()
                        if len(spx.loc[:prev_date]) >= 2:
                            pr0 = spx.loc[:prev_date].iloc[-2]
                            pr1 = spx.loc[:prev_date].iloc[-1]
                            ret_1m = (pr1 / pr0) - 1 if pr0 else 0.0
                        else:
                            ret_1m = 0.0
                        px = float(spx.loc[prev_date]) if prev_date in spx.index else float("nan")
                        safety_candidates.append({"ticker": st, "inercia": 0.0, "score_adj": float(ret_1m), "price": px})
                    except Exception:
                        continue
        if safety_candidates:
            safety_candidates.sort(key=lambda x: x["score_adj"], reverse=True)
            picks = safety_candidates[:1]
            return picks, {"filter": True, "prev_date": prev_date}, {"spy_m": spy_m, "safety_m": safety_m}

        return [], {"filter": True, "reason": "Sin datos de safety", "prev_date": prev_date}, {"spy_m": spy_m, "safety_m": safety_m}

    # Filtro inactivo => picks normales por corte
    if prev_date not in score_df.index or prev_date not in inercia_df.index:
        return [], {"filter": False, "prev_date": prev_date, "reason": "Mes sin scores"}, {"spy_m": spy_m, "safety_m": safety_m}

    last_scores = score_df.loc[prev_date].dropna()
    last_inercia = inercia_df.loc[prev_date]
    last_prices_row = prices_m.loc[prev_date] if prev_date in prices_m.index else pd.Series(dtype=float)

    candidates = []
    for t in last_scores.index:
        ine = float(last_inercia.get(t)) if t in last_inercia.index else 0.0
        sca = float(last_scores.get(t))
        if ine >= corte and sca > 0 and not np.isnan(sca):
            pr = float(last_prices_row.get(t)) if t in last_prices_row.index else float("nan")
            candidates.append({"ticker": t, "inercia": ine, "score_adj": sca, "price": pr})

    # Fallback por RawScoreAdjusted (sin corte) si nadie pasa el umbral
    if not candidates and raw_df is not None and prev_date in raw_df.index:
        raw_last = raw_df.loc[prev_date].dropna().sort_values(ascending=False)
        for t in list(raw_last.index)[:top_n]:
            ine = float(last_inercia.get(t)) if t in last_inercia.index else 0.0
            sca = float(raw_last.get(t))
            pr = float(last_prices_row.get(t)) if t in last_prices_row.index else float("nan")
            candidates.append({"ticker": t, "inercia": ine, "score_adj": sca, "price": pr})

    if candidates:
        candidates.sort(key=lambda x: x["score_adj"], reverse=True)
        candidates = candidates[:top_n]

    return candidates, {"filter": False, "prev_date": prev_date}, {"spy_m": spy_m, "safety_m": safety_m}

def build_telegram_message(index_choice, picks, meta, name_map):
    idx_label = "SP500" if index_choice.upper().startswith("SP") else ("NDX" if index_choice.upper().startswith("NDX") else "SP500 + NDX")
    as_of = meta.get("prev_date")
    if isinstance(as_of, pd.Timestamp):
        as_of = as_of.date()
    as_of_str = as_of.isoformat() if as_of else date.today().isoformat()
    title = f"🧭 Señales del sistema para el próximo mes ({idx_label})\n📅 Cierre: {as_of_str}"

    header = f"{'#':>2} {'Ticker':<7} {'Nombre':<28} {'Inercia':>8} {'ScoreAdj':>9} {'Precio':>10}"
    rows = []
    for i, p in enumerate(picks, 1):
        nm = name_map.get(p['ticker'], p['ticker'])
        nm = nm if len(nm) <= 28 else (nm[:25] + "...")
        price_txt = f"{p['price']:>10.2f}" if (isinstance(p['price'], (float, int)) and np.isfinite(p['price'])) else f"{'N/A':>10}"
        rows.append(f"{i:>2} {p['ticker']:<7} {nm:<28} {p['inercia']:>8.0f} {p['score_adj']:>9.2f} {price_txt}")
    table = "\n".join([header] + rows) if rows else "Sin candidatos"

    # Entradas/Salidas/Mantener comparando con estado previo
    key = state_key_from_index(index_choice)
    state_path = f"data/last_picks_{key}.json"
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
            json.dump({"timestamp": datetime.utcnow().isoformat(), "index": key, "tickers": sorted(list(new_set))}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    def fmt_list(lst, emoji):
        if not lst:
            return f"{emoji} -"
        out = []
        for t in lst:
            nm = name_map.get(t, t)
            out.append(f"{t} — {nm}")
        return f"{emoji} " + "\n".join(out)

    comprar_txt = fmt_list(comprar, "✅ Entrar")
    vender_txt = fmt_list(vender, "❌ Salir")
    mantener_txt = fmt_list(mantener, "♻️ Mantener")

    # Nota de filtro
    filt_note = "🛡️ Filtro de mercado ACTIVO (en refugio IEF/BIL)" if meta.get("filter") else "✅ Filtro INACTIVO (picks normales)"

    msg = (
        f"<b>{html.escape(title)}</b>\n{html.escape(filt_note)}\n\n"
        f"<b>Top {len(picks)}</b>\n"
        f"<pre>{html.escape(table)}</pre>\n\n"
        f"{html.escape(comprar_txt)}\n\n"
        f"{html.escape(vender_txt)}\n\n"
        f"{html.escape(mantener_txt)}\n\n"
        f"— Bot IA Mensual"
    )
    return msg

def main():
    parser = argparse.ArgumentParser(description="Enviar señales del sistema (próximo mes) a Telegram con filtros y safety")
    parser.add_argument("--index", default="BOTH", choices=["SP500", "NDX", "BOTH"], help="Índice: SP500 | NDX | BOTH")
    parser.add_argument("--top", type=int, default=5, help="Número de picks (default 5)")
    parser.add_argument("--corte", type=int, default=680, help="Corte de inercia (default 680)")
    parser.add_argument("--start", type=str, default="2007-01-01", help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Fecha fin (YYYY-MM-DD, default hoy)")
    parser.add_argument("--token", type=str, default=os.getenv("TELEGRAM_BOT_TOKEN"), help="Bot token (o env TELEGRAM_BOT_TOKEN)")
    parser.add_argument("--chat", type=str, default=os.getenv("TELEGRAM_CHAT_ID"), help="Chat ID (o env TELEGRAM_CHAT_ID)")
    args = parser.parse_args()

    if not args.token or not args.chat:
        print("❌ Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID (define en Secrets o Variables del repo).")
        sys.exit(1)

    start = pd.to_datetime(args.start).date()
    end = pd.to_datetime(args.end).date() if args.end else date.today()

    picks, meta, _ctx = compute_signals_with_filters(
        index_choice=args.index,
        top_n=args.top,
        corte=args.corte,
        start=start,
        end=end,
        warmup_months=24
    )
    name_map = get_name_map(args.index)
    msg = build_telegram_message(args.index, picks, meta, name_map)

    ok, info = send_telegram(args.token, args.chat, msg)
    if ok:
        print("✅ Mensaje enviado a Telegram")
    else:
        print(f"❌ Error enviando a Telegram: {info}")
        sys.exit(1)

if __name__ == "__main__":
    main()
