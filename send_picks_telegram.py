# -*- coding: utf-8 -*-
# send_picks_telegram.py
# Señales prospectivas con filtros de mercado activados (ROC12 y SMA10) y fallback IEF/BIL si el filtro está activo.
# Envía a Telegram una tabla con Rank, Ticker, Nombre, Inercia, Score Ajustado y Precio.
# Lee TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID de env (secrets o variables del repo).
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

from data_loader import download_prices
from backtest import inertia_score  # usa la lógica de indicadores del backtest

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
    # Warmup
    warmup_start = (pd.Timestamp(start) - pd.DateOffset(months=warmup_months)).date()

    # Universo actual
    current_set = get_current_constituents_set(index_choice)
    # Intersección con CSVs locales
    avail = set(os.path.basename(p)[:-4].upper().replace('.', '-') for p in glob.glob('data/*.csv'))
    universe = sorted(current_set & avail)
    if not universe:
        return [], {"reason": "No hay universo válido"}, {"prices": None, "spy_m": None, "safety_m": None}

    # Precios universo (warmup)
    prices_df, _ = download_prices(universe, warmup_start, end, load_full_data=False)
    if prices_df is None or prices_df.empty:
        return [], {"reason": "No se pudieron cargar precios del universo"}, {"prices": None, "spy_m": None, "safety_m": None}
    # Filtra por columnas reales
    universe = [t for t in universe if t in prices_df.columns]
    prices_df = prices_df[universe]
    prices_m = prices_df.resample("ME").last().dropna(how="all")
    if prices_m.empty:
        return [], {"reason": "Sin datos mensuales"}, {"prices": None, "spy_m": None, "safety_m": None}

    # SPY y Safety (warmup)
    spy_df = download_prices(["SPY"], warmup_start, end, load_full_data=False)
    spy_prices = spy_df[0] if isinstance(spy_df, tuple) else spy_df
    spy_m = None
    if isinstance(spy_prices, pd.DataFrame) and "SPY" in spy_prices.columns:
        spy_m = spy_prices["SPY"].resample("ME").last().dropna()

    safety_df = download_prices(["IEF", "BIL"], warmup_start, end, load_full_data=False)
    safety_prices = safety_df[0] if isinstance(safety_df, tuple) else safety_df
    safety_m = None
    if isinstance(safety_prices, pd.DataFrame):
        safety_m = safety_prices.resample("ME").last()

    # Calcula scores (mensuales)
    scores = inertia_score(prices_m, corte=corte, ohlc_data=None)
    if not scores or "ScoreAdjusted" not in scores or "InerciaAlcista" not in scores:
        return [], {"reason": "No se pudieron calcular scores"}, {"prices": prices_m, "spy_m": spy_m, "safety_m": safety_m}

    score_df = scores["ScoreAdjusted"]
    inercia_df = scores["InerciaAlcista"]
    if score_df.empty or inercia_df.empty:
        return [], {"reason": "Scores vacíos"}, {"prices": prices_m, "spy_m": spy_m, "safety_m": safety_m}

    # Último mes completo (prev_date)
    prev_date = prices_m.index[-1]
    # Filtro mercado: ROC12<0 o Precio<SMA10
    market_filter_active = False
    if spy_m is not None and prev_date in spy_m.index:
        try:
            sp = spy_m.loc[:prev_date]
            if len(sp) >= 13:
                roc12 = ((sp.iloc[-1] - sp.iloc[-13]) / sp.iloc[-13]) * 100 if sp.iloc[-13] != 0 else 0
            else:
                roc12 = 0
            sma10_ok = False
            if len(sp) >= 10:
                sma10 = sp.iloc[-10:].mean()
                sma10_ok = (sp.iloc[-1] < sma10)
            if (roc12 < 0) or sma10_ok:
                market_filter_active = True
        except Exception:
            market_filter_active = False

    # Si filtro activo => fallback safety (elige mejor entre IEF/BIL)
    if market_filter_active:
        safety_candidates = []
        if safety_m is not None and prev_date in safety_m.index:
            for st in ("IEF", "BIL"):
                if st in safety_m.columns:
                    try:
                        # retorno 1m (para desempate)
                        if len(safety_m.loc[:prev_date, st]) >= 2:
                            pr0 = safety_m.loc[:prev_date, st].iloc[-2]
                            pr1 = safety_m.loc[:prev_date, st].iloc[-1]
                            ret_1m = (pr1 / pr0) - 1 if pr0 else 0.0
                        else:
                            ret_1m = 0.0
                    except Exception:
                        ret_1m = 0.0
                    # score raw usando inertia_score sobre solo safety
                    # más sencillo: aproximar score_adj_raw = ret_1m si no hay suficientes datos
                    # (los safety no siempre tienen OHLC completo)
                    score_adj_raw = ret_1m
                    safety_candidates.append({"ticker": st, "score_adj": float(score_adj_raw), "ret_1m": float(ret_1m)})
        if safety_candidates:
            safety_candidates = sorted(safety_candidates, key=lambda x: x["score_adj"], reverse=True)
            best = safety_candidates[0]
            # devolvemos pick único
            return [{"ticker": best["ticker"], "inercia": 0.0, "score_adj": best["score_adj"], "price": float(safety_m.loc[prev_date, best["ticker"]]) if safety_m is not None and best["ticker"] in safety_m.columns else float("nan")}], {"filter": True, "prev_date": prev_date}, {"prices": prices_m, "spy_m": spy_m, "safety_m": safety_m}

        # No hay datos safety -> sin picks
        return [], {"filter": True, "reason": "Sin datos safety", "prev_date": prev_date}, {"prices": prices_m, "spy_m": spy_m, "safety_m": safety_m}

    # Si NO hay filtro => picks normales por corte
    last_scores = score_df.loc[prev_date].dropna() if prev_date in score_df.index else pd.Series(dtype=float)
    last_inercia = inercia_df.loc[prev_date] if prev_date in inercia_df.index else pd.Series(dtype=float)

    candidates = []
    last_prices_row = prices_m.loc[prev_date] if prev_date in prices_m.index else pd.Series(dtype=float)
    for t in last_scores.index:
        if t in last_inercia.index:
            ine = float(last_inercia.get(t))
            sca = float(last_scores.get(t))
            if ine >= corte and sca > 0 and not np.isnan(sca):
                pr = float(last_prices_row.get(t)) if t in last_prices_row.index else float("nan")
                candidates.append({"ticker": t, "inercia": ine, "score_adj": sca, "price": pr})

    # Fallback blando si no hay ninguno por corte: coge top por ScoreAdjusted (sin corte aplicado)
    if not candidates:
        last_raw = score_df.loc[prev_date].dropna().sort_values(ascending=False)
        for t in list(last_raw.index)[:top_n]:
            ine = float(last_inercia.get(t)) if t in last_inercia.index else 0.0
            sca = float(last_raw.get(t))
            pr = float(last_prices_row.get(t)) if t in last_prices_row.index else float("nan")
            candidates.append({"ticker": t, "inercia": ine, "score_adj": sca, "price": pr})

    if candidates:
        candidates = sorted(candidates, key=lambda x: x["score_adj"], reverse=True)[:top_n]

    return candidates, {"filter": False, "prev_date": prev_date}, {"prices": prices_m, "spy_m": spy_m, "safety_m": safety_m}

def build_telegram_message(index_choice, picks, meta, name_map):
    idx_label = "SP500" if index_choice.upper().startswith("SP") else ("NDX" if index_choice.upper().startswith("NDX") else "SP500 + NDX")
    as_of = meta.get("prev_date")
    if isinstance(as_of, pd.Timestamp):
        as_of = as_of.date()
    as_of_str = as_of.isoformat() if as_of else date.today().isoformat()
    title = f"🧠 Señales prospectivas IA Mensual ({idx_label})\n📅 Cierre: {as_of_str}"

    header = f"{'#':>2} {'Ticker':<7} {'Nombre':<28} {'Inercia':>8} {'ScoreAdj':>9} {'Precio':>10}"
    rows = []
    for i, p in enumerate(picks, 1):
        nm = name_map.get(p['ticker'], p['ticker'])
        nm = nm if len(nm) <= 28 else (nm[:25] + "...")
        rows.append(f"{i:>2} {p['ticker']:<7} {nm:<28} {p['inercia']:>8.0f} {p['score_adj']:>9.2f} {p['price']:>10.2f}" if not np.isnan(p['price']) else f"{i:>2} {p['ticker']:<7} {nm:<28} {p['inercia']:>8.0f} {p['score_adj']:>9.2f} {'N/A':>10}")
    table = "\n".join([header] + rows) if rows else "Sin candidatos"

    # Entradas/Salidas/Mantener comparando con estado previo
    state_key = "BOTH" if idx_label == "SP500 + NDX" else idx_label
    state_path = f"data/last_picks_{state_key}.json"
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
            json.dump({"timestamp": datetime.utcnow().isoformat(), "index": idx_label, "tickers": sorted(list(new_set))}, f, ensure_ascii=False, indent=2)
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

    comprar_txt = fmt_list(comprar, "✅ Comprar")
    vender_txt = fmt_list(vender, "❌ Vender")
    mantener_txt = fmt_list(mantener, "♻️ Mantener")

    # Cabecera informando si hay filtro
    filt_note = "🛡️ Filtro ACTIVO (fallback safety)" if meta.get("filter") else "✅ Filtro INACTIVO (picks normales)"

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
    parser = argparse.ArgumentParser(description="Enviar picks prospectivos a Telegram con filtros y fallback safety")
    parser.add_argument("--index", default="BOTH", choices=["SP500", "NDX", "BOTH"], help="Índice: SP500 | NDX | BOTH")
    parser.add_argument("--top", type=int, default=5, help="Número de picks (default 5)")
    parser.add_argument("--corte", type=int, default=680, help="Corte de inercia (default 680)")
    parser.add_argument("--start", type=str, default="2007-01-01", help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Fecha fin (YYYY-MM-DD, default hoy)")
    parser.add_argument("--token", type=str, default=os.getenv("TELEGRAM_BOT_TOKEN"), help="Bot token (o env TELEGRAM_BOT_TOKEN)")
    parser.add_argument("--chat", type=str, default=os.getenv("TELEGRAM_CHAT_ID"), help="Chat ID (o env TELEGRAM_CHAT_ID)")
    args = parser.parse_args()

    if not args.token or not args.chat:
        print("❌ Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID (usa Secrets o Variables del repo).")
        sys.exit(1)

    start = pd.to_datetime(args.start).date()
    end = pd.to_datetime(args.end).date() if args.end else date.today()

    picks, meta, ctx = compute_signals_with_filters(
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
