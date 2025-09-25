# -*- coding: utf-8 -*-
# telegram_signals.py
# Señales prospectivas mensuales:
# - Universo = (constituyentes actuales SP500/NDX via Wikipedia, con fallback a CSVs locales) ∩ CSVs de data/
# - Warm-up 24m para indicadores (como el backtest)
# - Indicadores = backtest.precalculate_all_indicators (ScoreAdjusted + InerciaAlcista)
# - Filtros ACTIVOS (ROC12<0 o Precio<SMA10). Si se activan -> fallback IEF/BIL (esa es la señal)
# - Si NO hay picks que pasen corte: “No hay candidatos” (sin fallback extra)
# - COMPRAR / VENDER / MANTENER: calcula picks de mes actual y del mes anterior
import os
import sys
import html
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from io import StringIO
import glob
from pytz import timezone

from data_loader import download_prices
from backtest import precalculate_all_indicators  # mismo cálculo de indicadores que el backtest

# ===================== Helpers: Wikipedia con UA =====================
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

def get_sp500_constituents():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _read_html_with_ua(url, attrs={"id": "constituents"})
    df = tables[0] if tables else None
    if df is None:
        tables = _read_html_with_ua(url)
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if any(c in ("symbol", "ticker") for c in cols):
                df = t
                break
    if df is None:
        return set(), {}
    sym_col = next((c for c in df.columns if str(c).strip().lower() in ("symbol", "ticker")), None)
    sec_col = next((c for c in df.columns if str(c).strip().lower() in ("security", "company", "company name")), None)
    if not sym_col:
        return set(), {}
    syms = df[sym_col].astype(str).str.upper().str.replace(".", "-", regex=False).tolist()
    name_map = {}
    if sec_col:
        s = df[[sym_col, sec_col]].copy()
        s.columns = ["Symbol", "Name"]
        s["Symbol"] = s["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
        name_map = dict(zip(s["Symbol"], s["Name"]))
    return set(s for s in syms if s and s.strip()), name_map

def get_ndx_constituents():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = _read_html_with_ua(url, attrs={"id": "constituents"})
    df = tables[0] if tables else None
    if df is None:
        tables = _read_html_with_ua(url)
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if any(("ticker" in c or "symbol" in c) for c in cols):
                df = t
                break
    if df is None:
        return set(), {}
    sym_col = next((c for c in df.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()), None)
    name_col = next((c for c in df.columns if "company" in str(c).lower()), None)
    if not sym_col:
        return set(), {}
    syms = df[sym_col].astype(str).str.upper().str.replace(".", "-", regex=False).tolist()
    name_map = {}
    if name_col:
        s = df[[sym_col, name_col]].copy()
        s.columns = ["Symbol", "Name"]
        s["Symbol"] = s["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
        name_map = dict(zip(s["Symbol"], s["Name"]))
    return set(s for s in syms if s and s.strip()), name_map

def get_name_map(index_choice):
    ic = index_choice.upper()
    if ic in ("SP500", "S&P500", "S&P 500"):
        sp, m = get_sp500_constituents()
        return m
    elif ic in ("NDX", "NASDAQ-100", "NASDAQ100"):
        nd, m = get_ndx_constituents()
        return m
    else:
        _, m1 = get_sp500_constituents()
        _, m2 = get_ndx_constituents()
        m1.update(m2)
        return m1

# ===================== Helpers: CSV locales, warm-up, filtros =====================
def _dl_prices_single(objs, start, end, full=False):
    out = download_prices(objs, start, end, load_full_data=full)
    if isinstance(out, tuple):
        return out[0]
    return out

def _month_end(d):
    return (pd.Timestamp(d) + pd.offsets.MonthEnd(0)).normalize()

def _latest_month_end_available(df_m, target_me):
    if df_m is None or df_m.empty:
        return None
    idx = df_m.index
    idx = idx[idx <= target_me]
    return idx[-1] if len(idx) else None

def _build_universe(index_choice):
    ic = index_choice.upper()
    wanted = set()
    names = {}
    try:
        if ic in ("SP500", "S&P500", "S&P 500", "BOTH", "SP500 + NDX", "AMBOS", "AMBOS (SP500 + NDX)"):
            sp, nm_sp = get_sp500_constituents()
            wanted |= sp
            names.update(nm_sp)
        if ic in ("NDX", "NASDAQ-100", "NASDAQ100", "BOTH", "SP500 + NDX", "AMBOS", "AMBOS (SP500 + NDX)"):
            nd, nm_nd = get_ndx_constituents()
            wanted |= nd
            names.update(nm_nd)
    except Exception:
        wanted = set()

    csv_tickers = set(os.path.basename(p)[:-4].upper().replace('.', '-') for p in glob.glob('data/*.csv'))
    exclude = {"SPY", "QQQ", "IEF", "BIL"}
    universe = sorted(((wanted & csv_tickers) if wanted else csv_tickers) - exclude)
    return universe, names, (len(wanted) > 0)

def market_filter(spy_m, at_date):
    if spy_m is None or spy_m.empty:
        return False
    sp = spy_m.loc[:at_date]
    if sp.empty:
        return False
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
    return bool(roc12_bad or sma10_bad)

def picks_for_date(prices_m, indicators, at_date, corte, top_n, filter_active, safety_m=None):
    if filter_active:
        # Fallback safety: elegir IEF/BIL con mejor ret_1m al mes at_date
        if safety_m is not None and not safety_m.empty:
            s_idx = safety_m.index[safety_m.index <= at_date]
            if len(s_idx):
                s_prev = s_idx[-1]
                best = None
                for st in ("IEF", "BIL"):
                    if st in safety_m.columns:
                        ser = safety_m[st].dropna()
                        if s_prev in ser.index and len(ser.loc[:s_prev]) >= 2:
                            pr1 = ser.loc[s_prev]
                            pr0 = ser.loc[:s_prev].iloc[-2]
                            ret_1m = (pr1 / pr0) - 1 if pr0 else 0.0
                        else:
                            ret_1m = 0.0
                        cand = {"ticker": st, "inercia": 0.0, "score_adj": float(ret_1m), "price": float(ser.loc[s_prev]) if s_prev in ser.index else float("nan")}
                        if (best is None) or (cand["score_adj"] > best["score_adj"]):
                            best = cand
                return [best] if best else []
        return []

    # Sin filtro: picks estrictos por corte
    candidates = []
    for tkr, bundle in indicators.items():
        try:
            if at_date in bundle['InerciaAlcista'].index:
                ine = float(bundle['InerciaAlcista'].loc[at_date])
                sca = float(bundle['ScoreAdjusted'].loc[at_date]) if at_date in bundle['ScoreAdjusted'].index else 0.0
                pr = float(prices_m.loc[at_date, tkr]) if (tkr in prices_m.columns and at_date in prices_m.index) else float("nan")
                if (ine >= corte) and (sca > 0) and np.isfinite(sca):
                    candidates.append({"ticker": tkr, "inercia": ine, "score_adj": sca, "price": pr})
        except Exception:
            continue
    candidates = sorted(candidates, key=lambda x: x["score_adj"], reverse=True)[:top_n]
    return candidates

# ===================== Señales (actual y mes anterior) =====================
def compute_signals_with_prev(index_choice="BOTH", top_n=5, corte=680, start=None, end=None, warmup_months=24, verbose=True):
    if end is None:
        end = date.today()
    if start is None:
        start = date(2007, 1, 1)
    warmup_start = (pd.Timestamp(start) - pd.DateOffset(months=warmup_months)).date()
    month_end = _month_end(end)

    # Universo y nombres (fallback a CSV si Wikipedia falla)
    universe, name_map_wiki, had_wiki = _build_universe(index_choice)
    if verbose:
        print(f"Universo via Wikipedia={had_wiki}: {len(universe)} tickers")
    if not universe:
        csv_tickers = set(os.path.basename(p)[:-4].upper().replace('.', '-') for p in glob.glob('data/*.csv'))
        universe = sorted(csv_tickers - {"SPY", "QQQ", "IEF", "BIL"})
        if verbose:
            print(f"Fallback a CSV locales: {len(universe)} tickers")
    if not universe:
        return [], [], [], {"prev_date": None, "filter": False, "reason": "Universo vacío"}

    # Precios (warm-up)
    prices_df = _dl_prices_single(universe, warmup_start, end, full=False)
    if prices_df is None or prices_df.empty:
        return [], [], [], {"prev_date": None, "filter": False, "reason": "Sin precios universo"}

    # Ajustar universe a los que realmente cargaron
    universe = [t for t in universe if t in prices_df.columns]
    prices_df = prices_df[universe]
    prices_m = prices_df.resample("ME").last()
    current_me = _latest_month_end_available(prices_m, month_end)
    if current_me is None:
        return [], [], [], {"prev_date": None, "filter": False, "reason": "Sin mensual"}

    # Mes anterior
    idx = prices_m.index
    try:
        pos = idx.get_loc(current_me)
    except KeyError:
        pos = len(idx) - 1
        current_me = idx[pos]
    prev_me = idx[pos - 1] if pos >= 1 else None

    # SPY y Safety
    spy_df = _dl_prices_single(["SPY"], warmup_start, end, full=False)
    spy_m = spy_df["SPY"].resample("ME").last().dropna() if isinstance(spy_df, pd.DataFrame) and "SPY" in spy_df.columns else None
    safety_df = _dl_prices_single(["IEF", "BIL"], warmup_start, end, full=False)
    safety_m = safety_df.resample("ME").last() if isinstance(safety_df, pd.DataFrame) else None

    # Indicadores
    indicators = precalculate_all_indicators(prices_m, ohlc_data=None, corte=corte)
    if not indicators:
        return [], [], [], {"prev_date": current_me, "filter": False, "reason": "Indicadores vacíos"}

    # Filtro actual y anterior
    filt_now = market_filter(spy_m, current_me)
    filt_prev = market_filter(spy_m, prev_me) if prev_me is not None else False

    # Picks ahora y mes anterior
    picks_now = picks_for_date(prices_m, indicators, current_me, corte, top_n, filt_now, safety_m=safety_m)
    picks_prev = picks_for_date(prices_m, indicators, prev_me, corte, top_n, filt_prev, safety_m=safety_m) if prev_me is not None else []

    # Conjuntos para comprar/vender/mantener
    now_set = set(p["ticker"] for p in picks_now)
    prev_set = set(p["ticker"] for p in picks_prev)
    comprar = sorted(now_set - prev_set)
    vender = sorted(prev_set - now_set)
    mantener = sorted(now_set & prev_set)

    # Name map robusto
    name_map = get_name_map(index_choice)
    if not name_map:
        # fallback a mapa vacío (mostraremos ticker como nombre)
        name_map = {}

    return picks_now, comprar, vender, mantener, {"prev_date": current_me, "filter": filt_now, "name_map": name_map}

# ===================== Mensaje y envío =====================
def build_message(index_choice, picks, comprar, vender, mantener, meta):
    name_map = meta.get("name_map", {})
    idx_label = "SP500" if index_choice.upper().startswith("SP") else ("NDX" if index_choice.upper().startswith("NDX") else "SP500 + NDX")
    as_of = meta.get("prev_date")
    if isinstance(as_of, pd.Timestamp):
        as_of = as_of.date()
    as_of_str = as_of.isoformat() if as_of else date.today().isoformat()
    title = f"🧠 Señales prospectivas IA Mensual ({idx_label})\n📅 Cierre: {as_of_str}"
    filt_note = "🛡️ Filtro ACTIVO (fallback safety)" if meta.get("filter") else "✅ Filtro INACTIVO (picks normales)"

    if not picks:
        msg = (
            f"<b>{html.escape(title)}</b>\n{html.escape(filt_note)}\n\n"
            f"<b>No hay candidatos</b>\n\n"
            f"— Bot IA Mensual"
        )
        return msg

    header = f"{'#':>2} {'Ticker':<7} {'Nombre':<28} {'Inercia':>8} {'ScoreAdj':>9} {'Precio':>10}"
    rows = []
    for i, p in enumerate(picks, 1):
        nm = name_map.get(p['ticker'], p['ticker'])
        nm = nm if len(nm) <= 28 else (nm[:25] + "...")
        price_str = f"{p['price']:>10.2f}" if isinstance(p['price'], (int, float)) and np.isfinite(p['price']) else f"{'N/A':>10}"
        rows.append(f"{i:>2} {p['ticker']:<7} {nm:<28} {p['inercia']:>8.0f} {p['score_adj']:>9.2f} {price_str}")
    table = "\n".join([header] + rows)

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

def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, data={
        "chat_id": str(chat_id),
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }, timeout=30)
    resp.raise_for_status()
    return True

# ===================== Control horario (Madrid) =====================
def es_ultimo_dia_mes_y_despues_23_madrid():
    tz = timezone('Europe/Madrid')
    now = datetime.now(tz)
    # último día del mes: mañana es día 1
    manana = now + timedelta(days=1)
    es_ultimo = (manana.month != now.month)
    return es_ultimo and (now.hour >= 23)

# ===================== Main =====================
def main():
    parser = argparse.ArgumentParser(description="Enviar señales prospectivas por Telegram (workflow dedicado)")
    parser.add_argument("--index", default="BOTH", choices=["SP500", "NDX", "BOTH"], help="Índice: SP500 | NDX | BOTH")
    parser.add_argument("--top", type=int, default=5, help="Número de picks (default 5)")
    parser.add_argument("--corte", type=int, default=680, help="Corte de inercia (default 680)")
    parser.add_argument("--start", type=str, default="2007-01-01", help="Fecha inicio datos (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Fecha fin datos (YYYY-MM-DD, default hoy)")
    parser.add_argument("--force", action="store_true", help="Ignorar chequeo de último día a las 23:00 Madrid")
    args = parser.parse_args()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("❌ Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID en Secrets/Variables.")
        sys.exit(1)

    if not args.force and not es_ultimo_dia_mes_y_despues_23_madrid():
        print("⏳ No es último día de mes ≥ 23:00 (Europe/Madrid). Saliendo sin enviar.")
        sys.exit(0)

    start = pd.to_datetime(args.start).date()
    end = pd.to_datetime(args.end).date() if args.end else date.today()

    picks_now, comprar, vender, mantener, meta = compute_signals_with_prev(
        index_choice=args.index,
        top_n=args.top,
        corte=args.corte,
        start=start,
        end=end,
        warmup_months=24,
        verbose=True
    )

    msg = build_message(args.index, picks_now, comprar, vender, mantener, meta)
    try:
        send_telegram(token, chat_id, msg)
        print(f"✅ Señales enviadas. Picks: {len(picks_now)} | Filtro activo: {meta.get('filter')}")
    except Exception as e:
        print(f"❌ Error enviando a Telegram: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
