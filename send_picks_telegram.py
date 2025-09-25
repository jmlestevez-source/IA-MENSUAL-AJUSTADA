# -*- coding: utf-8 -*-
# send_picks_telegram.py
# Señales prospectivas calculadas igual que el backtest:
# - Universo = constituyentes actuales (SP500/NDX/BOTH) ∩ CSVs locales
# - Warm-up de 24 meses para indicadores
# - Indicadores = backtest.precalculate_all_indicators (ScoreAdjusted / InerciaAlcista)
# - Filtros de mercado ACTIVOS (ROC12<0 o Precio<SMA10). Si se activan -> fallback IEF/BIL
# - Si no hay picks por corte, fallback blando al Top por ScoreAdjusted (sin exigir corte)
# - Envío a Telegram con ENTRAN / SALEN / MANTIENEN
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
from backtest import precalculate_all_indicators  # usamos exactamente el mismo cálculo de indicadores que el backtest

# ===================== Utilidades Wikipedia (nombres) =====================
def _read_html_with_ua(url, attrs=None):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
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
    ic = index_choice.upper()
    if ic in ("SP500", "S&P500", "S&P 500"):
        return set(map(normalize_symbol, get_sp500_name_map().keys()))
    elif ic in ("NDX", "NASDAQ-100", "NASDAQ100"):
        return set(map(normalize_symbol, get_ndx_name_map().keys()))
    else:
        s1 = set(map(normalize_symbol, get_sp500_name_map().keys()))
        s2 = set(map(normalize_symbol, get_ndx_name_map().keys()))
        return s1 | s2

def get_name_map(index_choice):
    ic = index_choice.upper()
    if ic in ("SP500", "S&P500", "S&P 500"):
        return get_sp500_name_map()
    elif ic in ("NDX", "NASDAQ-100", "NASDAQ100"):
        return get_ndx_name_map()
    else:
        m = get_sp500_name_map()
        m2 = get_ndx_name_map()
        m.update(m2)
        return m

# ===================== Telegram =====================
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

# ===================== Utilidades de precios/fechas =====================
def _dl_prices_single(objs, start, end):
    """
    Descarga robusta desde data_loader.download_prices:
    - Si devuelve (df, ohlc), extrae df.
    - Si devuelve df directo, lo retorna.
    """
    out = download_prices(objs, start, end, load_full_data=False)
    if isinstance(out, tuple):
        return out[0]
    return out

def _month_end(d):
    return (pd.Timestamp(d) + pd.offsets.MonthEnd(0)).normalize()

def _latest_month_end_available(df_m, target_me):
    """
    Devuelve el último índice mensual <= target_me. Si no hay, None.
    """
    if df_m is None or df_m.empty:
        return None
    idx = df_m.index
    idx = idx[idx <= target_me]
    return idx[-1] if len(idx) else None

# ===================== Señales con filtros + indicadores del backtest =====================
def compute_signals_with_filters(index_choice="BOTH", top_n=5, corte=680, start=None, end=None, warmup_months=24):
    # Fechas y warm-up
    if end is None:
        end = date.today()
    if start is None:
        start = date(2007, 1, 1)
    warmup_start = (pd.Timestamp(start) - pd.DateOffset(months=warmup_months)).date()
    month_end = _month_end(end)

    # Universo actual ∩ CSVs disponibles
    current_set = get_current_constituents_set(index_choice)
    avail = set(os.path.basename(p)[:-4].upper().replace('.', '-') for p in glob.glob('data/*.csv'))
    universe = sorted(current_set & avail)
    if not universe:
        return [], {"reason": "No hay universo válido", "prev_date": None}, {}

    # Precios universo (warmup)
    prices_df = _dl_prices_single(universe, warmup_start, end)
    if prices_df is None or prices_df.empty:
        return [], {"reason": "No se pudieron cargar precios del universo", "prev_date": None}, {}
    # Filtrar columnas reales
    universe = [t for t in universe if t in prices_df.columns]
    prices_df = prices_df[universe]
    prices_m = prices_df.resample("ME").last()
    prev_date = _latest_month_end_available(prices_m, month_end)
    if prev_date is None:
        return [], {"reason": "Sin datos mensuales", "prev_date": None}, {}

    # SPY y Safety (warmup)
    spy_prices = _dl_prices_single(["SPY"], warmup_start, end)
    spy_m = spy_prices["SPY"].resample("ME").last().dropna() if isinstance(spy_prices, pd.DataFrame) and "SPY" in spy_prices.columns else None
    safety_prices = _dl_prices_single(["IEF", "BIL"], warmup_start, end)
    safety_m = safety_prices.resample("ME").last() if isinstance(safety_prices, pd.DataFrame) else None

    # Indicadores del backtest (exactamente igual)
    indicators = precalculate_all_indicators(prices_m, ohlc_data=None, corte=corte)
    if not indicators:
        return [], {"reason": "No se pudieron calcular indicadores", "prev_date": prev_date}, {}

    # Filtro de mercado (ROC12<0 o Precio<SMA10)
    market_filter_active = False
    if spy_m is not None:
        sp = spy_m.loc[:prev_date]
        if len(sp):
            roc12_bad = False
            sma10_bad = False
            if len(sp) >= 13:
                base = sp.iloc[-13]
                if pd.notna(base) and base != 0:
                    roc12 = ((sp.iloc[-1] - base) / base) * 100
                    roc12_bad = (roc12 < 0)
            if len(sp) >= 10:
                sma10 = sp.iloc[-10:].mean()
                sma10_bad = (sp.iloc[-1] < sma10)
            market_filter_active = bool(roc12_bad or sma10_bad)

    # Si filtro activo => fallback safety (elige mejor entre IEF/BIL por retorno 1m)
    if market_filter_active:
        safety_candidates = []
        if safety_m is not None:
            s_idx = safety_m.index[safety_m.index <= prev_date]
            if len(s_idx):
                s_prev = s_idx[-1]
                for st in ("IEF", "BIL"):
                    if st in safety_m.columns:
                        try:
                            ser = safety_m[st].dropna()
                            if s_prev in ser.index and len(ser.loc[:s_prev]) >= 2:
                                pr1 = ser.loc[s_prev]
                                pr0 = ser.loc[:s_prev].iloc[-2]
                                ret_1m = (pr1 / pr0) - 1 if pr0 else 0.0
                            else:
                                ret_1m = 0.0
                            safety_candidates.append({
                                "ticker": st,
                                "inercia": 0.0,
                                "score_adj": float(ret_1m),
                                "price": float(ser.loc[s_prev]) if s_prev in ser.index else float("nan")
                            })
                        except Exception:
                            continue
        if safety_candidates:
            best = sorted(safety_candidates, key=lambda x: x["score_adj"], reverse=True)[0]
            return [best], {"filter": True, "prev_date": prev_date}, {}

        # Si no hay safety, devolvemos vacío (pero marcando filtro)
        return [], {"filter": True, "reason": "Sin datos safety", "prev_date": prev_date}, {}

    # Sin filtro: picks por corte usando indicadores del backtest
    candidates = []
    for tkr, bundle in indicators.items():
        try:
            if tkr not in universe:
                continue
            if prev_date in bundle['InerciaAlcista'].index:
                ine = float(bundle['InerciaAlcista'].loc[prev_date])
                sca = float(bundle['ScoreAdjusted'].loc[prev_date])
                pr = float(prices_m.loc[prev_date, tkr]) if (tkr in prices_m.columns and prev_date in prices_m.index) else float("nan")
                if ine >= corte and sca > 0 and np.isfinite(sca):
                    candidates.append({"ticker": tkr, "inercia": ine, "score_adj": sca, "price": pr})
        except Exception:
            continue

    # Fallback blando si nadie pasa el corte: usar Top por ScoreAdjusted (sin exigir corte)
    if not candidates:
        pool = []
        for tkr, bundle in indicators.items():
            try:
                if tkr not in universe or prev_date not in bundle['ScoreAdjusted'].index:
                    continue
                sca = float(bundle['ScoreAdjusted'].loc[prev_date])
                ine = float(bundle['InerciaAlcista'].loc[prev_date]) if prev_date in bundle['InerciaAlcista'].index else 0.0
                pr = float(prices_m.loc[prev_date, tkr]) if (tkr in prices_m.columns and prev_date in prices_m.index) else float("nan")
                if np.isfinite(sca):
                    pool.append({"ticker": tkr, "inercia": ine, "score_adj": sca, "price": pr})
            except Exception:
                continue
        if pool:
            candidates = sorted(pool, key=lambda x: x["score_adj"], reverse=True)[:top_n]

    candidates = sorted(candidates, key=lambda x: x["score_adj"], reverse=True)[:top_n] if candidates else []
    return candidates, {"filter": False, "prev_date": prev_date}, {}

# ===================== Mensaje a Telegram =====================
def build_telegram_message(index_choice, picks, meta, name_map):
    idx_label = "SP500" if index_choice.upper().startswith("SP") else ("NDX" if index_choice.upper().startswith("NDX") else "SP500 + NDX")
    as_of = meta.get("prev_date")
    if isinstance(as_of, pd.Timestamp):
        as_of = as_of.date()
    as_of_str = as_of.isoformat() if as_of else date.today().isoformat()
    title = f"🧠 Señales prospectivas IA Mensual ({idx_label})\n📅 Cierre: {as_of_str}"
    filt_note = "🛡️ Filtro ACTIVO (fallback safety)" if meta.get("filter") else "✅ Filtro INACTIVO (picks normales)"

    header = f"{'#':>2} {'Ticker':<7} {'Nombre':<28} {'Inercia':>8} {'ScoreAdj':>9} {'Precio':>10}"
    rows = []
    for i, p in enumerate(picks, 1):
        nm = name_map.get(p['ticker'], p['ticker'])
        nm = nm if len(nm) <= 28 else (nm[:25] + "...")
        price_str = f"{p['price']:>10.2f}" if isinstance(p['price'], (int, float)) and np.isfinite(p['price']) else f"{'N/A':>10}"
        rows.append(f"{i:>2} {p['ticker']:<7} {nm:<28} {p['inercia']:>8.0f} {p['score_adj']:>9.2f} {price_str}")
    table = "\n".join([header] + rows) if rows else "Sin candidatos"

    # ENTRAN/SALEN/MANTIENEN
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
    # guardar estado
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
    parser = argparse.ArgumentParser(description="Enviar picks prospectivos a Telegram (filtros + fallback safety) con indicadores del backtest")
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

    picks, meta, _ = compute_signals_with_filters(
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
