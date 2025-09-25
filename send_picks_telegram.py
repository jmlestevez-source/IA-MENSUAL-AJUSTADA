# -*- coding: utf-8 -*-
# send_picks_telegram.py
# Señales prospectivas EXACTAMENTE como el backtest:
# - Universo = constituyentes actuales (SP500/NDX/BOTH) ∩ CSVs locales
# - Warm-up (24m) para que los indicadores estén listos
# - Usa run_backtest_optimized con filtros ROC/SMA ACTIVOS y fallback IEF/BIL
# - Extrae el último set de picks del backtest y lo envía a Telegram
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
from backtest import run_backtest_optimized  # usamos el mismo motor que la app

# ===================== Utilidades Wikipedia (nombres) =====================
def _read_html_with_ua(url, attrs=None):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0 Safari/537.36"}
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

# ===================== Señales vía backtest =====================
def _dl_prices(objs, start, end, full=True):
    out = download_prices(objs, start, end, load_full_data=full)
    if full:
        return out  # (df, ohlc)
    return out  # df o (df,ohlc) según wrapper

def compute_picks_via_backtest(index_choice="BOTH", top_n=5, corte=680, start=None, end=None, warmup_months=24):
    # Fechas y warm-up
    if end is None:
        end = date.today()
    if start is None:
        start = date(2007, 1, 1)
    warmup_start = (pd.Timestamp(start) - pd.DateOffset(months=warmup_months)).date()
    month_end = (pd.Timestamp(end) + pd.offsets.MonthEnd(0)).normalize()

    # Universo actual ∩ CSVs locales (excluir safety y benchmarks)
    current_set = get_current_constituents_set(index_choice)
    exclude = {"SPY", "QQQ", "IEF", "BIL"}
    avail = set(os.path.basename(p)[:-4].upper().replace('.', '-') for p in glob.glob('data/*.csv'))
    universe = sorted((current_set & avail) - exclude)
    if not universe:
        return [], {"reason": "Universo vacío", "filter": False, "prev_date": None}, {}

    # Precios universo con OHLC (warm-up)
    prices_df, ohlc_data = _dl_prices(universe, warmup_start, end, full=True)
    if prices_df is None or prices_df.empty:
        return [], {"reason": "Sin precios universo", "filter": False, "prev_date": None}, {}

    # Benchmark y SPY (filtros) + safety
    bench_ticker = "QQQ" if index_choice.upper() == "NDX" else "SPY"
    bench_df = _dl_prices([bench_ticker], warmup_start, end, full=False)
    bench = bench_df[bench_ticker] if isinstance(bench_df, pd.DataFrame) and bench_ticker in bench_df.columns else prices_df.mean(axis=1)

    spy_df = _dl_prices(["SPY"], warmup_start, end, full=False)
    spy_series = spy_df["SPY"] if isinstance(spy_df, pd.DataFrame) and "SPY" in spy_df.columns else None

    saf_df, saf_ohlc = _dl_prices(["IEF", "BIL"], warmup_start, end, full=True)
    safety_prices = saf_df if isinstance(saf_df, pd.DataFrame) else None

    # Ejecutar backtest con filtros ACTIVOS + safety fallback + warm-up interno
    bt_results, picks_df = run_backtest_optimized(
        prices=prices_df,
        benchmark=bench,
        commission=0.0,
        top_n=top_n,
        corte=corte,
        ohlc_data=ohlc_data,
        historical_info=None,             # picks prospectivos sobre los actuales
        fixed_allocation=False,
        use_roc_filter=True,              # filtros ACTIVOS
        use_sma_filter=True,
        spy_data=spy_series.to_frame() if isinstance(spy_series, pd.Series) else spy_series,
        progress_callback=None,
        use_safety_etfs=True,             # fallback IEF/BIL
        safety_prices=safety_prices,
        safety_ohlc=saf_ohlc,
        avoid_rebuy_unchanged=True,
        enable_fallback=True,             # por si nadie pasa el corte
        warmup_months=warmup_months,
        user_start_date=start
    )

    if picks_df is None or picks_df.empty:
        # intentar aún así detectar si filtro activo para meta
        spy_m = spy_series.resample("ME").last() if isinstance(spy_series, pd.Series) else None
        filter_active = False
        if spy_m is not None and len(spy_m) >= 13:
            prev = spy_m.index[-1]
            roc12 = ((spy_m.iloc[-1] - spy_m.iloc[-13]) / spy_m.iloc[-13]) * 100 if spy_m.iloc[-13] != 0 else 0
            sma10 = spy_m.iloc[-10:].mean() if len(spy_m) >= 10 else spy_m.iloc[-1]
            filter_active = (roc12 < 0) or (spy_m.iloc[-1] < sma10)
        return [], {"reason": "Sin picks", "filter": filter_active, "prev_date": month_end.date()}, {}

    # Último set de picks del backtest (última fecha)
    last_date = max(pd.to_datetime(picks_df["Date"]))
    last_rows = picks_df[picks_df["Date"] == last_date.strftime("%Y-%m-%d")].copy()
    last_rows.sort_values("Rank", inplace=True)
    # Preparar salida
    prices_m = prices_df.resample("ME").last()
    picks = []
    for _, r in last_rows.iterrows():
        t = r["Ticker"]
        price = float(prices_m.loc[last_date, t]) if (t in prices_m.columns and last_date in prices_m.index) else float("nan")
        picks.append({
            "ticker": t,
            "inercia": float(r.get("Inercia", 0.0)),
            "score_adj": float(r.get("ScoreAdj", 0.0)),
            "price": price
        })

    # Meta: filtro activo si los picks son safety
    filter_active = False
    if len(picks) == 1 and picks[0]["ticker"] in ("IEF", "BIL"):
        filter_active = True

    return picks, {"filter": filter_active, "prev_date": last_date}, {"prices": prices_df, "ohlc": ohlc_data}

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
    parser = argparse.ArgumentParser(description="Enviar picks prospectivos a Telegram (vía backtest, filtros activos y fallback safety)")
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

    picks, meta, _ = compute_picks_via_backtest(
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
