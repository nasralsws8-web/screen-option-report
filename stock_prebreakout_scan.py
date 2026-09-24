"""
تشغيل ماسح ما قبل الاختراق وكتابة stock_prebreakout.csv.
لا يستبدل ملفاً صالحاً بنتيجة فارغة إذا فشل الجلب.
"""

import os
import time
from datetime import datetime, timezone

import pandas as pd

from cheap_options_screener_v3 import (
    DELAY_BETWEEN,
    _YF_SESSION,
    fetch_premarket,
    fix_ticker,
    load_manual_tickers,
)
from finnhub_premarket import _get, enrich_ticker_premarket, get_api_key
from stock_prebreakout import (
    SIGNAL_IGNORE,
    breakout_level,
    evaluate,
    higher_low_steps,
    is_compressing,
    volume_is_rising,
)

OUT_PATH = os.environ.get("STOCK_PREBREAKOUT_CSV", "stock_prebreakout.csv")
LOG_PATH = os.environ.get("STOCK_PREBREAKOUT_LOG", "stock_prebreakout_log.csv")
MAX_NAMES = int(os.environ.get("STOCK_PREBREAKOUT_MAX", "25"))

COLUMNS = [
    "ticker", "company", "price", "gap_pct", "float_shares",
    "pm_volume", "dollar_volume", "rvol", "rocm20", "rocm200",
    "vwap", "atr_pct", "pmh", "resistance", "distance_to_resistance_pct",
    "trigger", "target", "stop", "risk_reward", "catalyst",
    "pattern", "score", "signal", "signal_label", "reason",
    "notes", "disclaimer", "recommendation", "scanned_at",
]


def _f(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _roc(closes, n):
    if closes is None or len(closes) <= n:
        return None
    past = float(closes.iloc[-(n + 1)])
    if past == 0:
        return None
    return (float(closes.iloc[-1]) - past) / past


def _features_from_hist(ticker, info, hist, pm):
    if hist is None or len(hist) < 25:
        return None
    close = hist["Close"].astype(float)
    high = hist["High"].astype(float)
    low = hist["Low"].astype(float)
    vol = hist["Volume"].astype(float)
    price = _f(pm.get("last")) or float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else price
    gap = ((price - prev_close) / prev_close * 100) if prev_close else None
    avg20 = float(vol.tail(20).mean()) if len(vol) >= 5 else 0
    today_vol = float(vol.iloc[-1])
    rvol = (today_vol / avg20) if avg20 else _f(pm.get("rvol"))
    if pm.get("volume") and avg20:
        rvol = max(rvol or 0, float(pm["volume"]) / avg20)

    tr = (high - low).tail(14)
    atr = float(tr.mean()) if len(tr) else 0
    atr_pct = (atr / price * 100) if price else None

    lows = [float(x) for x in low.tail(6)]
    older = float(high.tail(10).head(5).max() - low.tail(10).head(5).min()) if len(hist) >= 10 else None
    newer = float(high.tail(5).max() - low.tail(5).min()) if len(hist) >= 5 else None
    roc20 = _roc(close, 20)
    roc20_prev = _roc(close.iloc[:-5], 20) if len(close) > 30 else None
    rocm20 = None if roc20 is None or roc20_prev is None else roc20 - roc20_prev
    roc200 = _roc(close, 200) if len(close) > 200 else None
    roc200_prev = _roc(close.iloc[:-10], 200) if len(close) > 220 else None
    rocm200 = None if roc200 is None or roc200_prev is None else roc200 - roc200_prev

    typical = (high + low + close) / 3
    pv = (typical * vol).tail(20).sum()
    vv = vol.tail(20).sum()
    vwap = float(pv / vv) if vv else None

    pmh = _f(pm.get("high")) or float(high.iloc[-1])
    float_shares = _f(info.get("floatShares") or info.get("sharesOutstanding"))
    company = str(info.get("shortName") or info.get("longName") or ticker)

    return {
        "ticker": ticker,
        "company": company,
        "price": round(price, 4),
        "gap_pct": round(gap, 2) if gap is not None else None,
        "float_shares": float_shares,
        "pm_volume": pm.get("volume") or today_vol,
        "dollar_volume": (pm.get("volume") or today_vol) * price,
        "rvol": round(rvol, 2) if rvol else None,
        "rocm20": rocm20,
        "rocm20_rising": bool(rocm20 is not None and roc20_prev is not None and rocm20 > 0),
        "rocm200": rocm200,
        "above_vwap": bool(vwap and price > vwap),
        "vwap": round(vwap, 4) if vwap else None,
        "atr_pct": round(atr_pct, 2) if atr_pct is not None else None,
        "pmh": round(pmh, 4),
        "prev_day_high": float(high.iloc[-2]) if len(high) >= 2 else None,
        "intraday_high": float(high.iloc[-1]),
        "higher_low_steps": higher_low_steps(lows),
        "compression": is_compressing(older, newer),
        "volume_rising": volume_is_rising(vol.tail(4).tolist()),
        "repeated_rejection": False,
        "structure_low": lows[-1] if lows else None,
        "catalyst": "none",
        "catalyst_text": "",
        "rs_vs_spy": None,
        "major_resistance": None,
    }


def _yahoo(ticker):
    import yfinance as yf
    return yf.Ticker(ticker, session=_YF_SESSION)


def _premarket(ticker):
    """نفس جلسة Yahoo ودالة البريماركت المستخدمة في مسح الخيارات."""
    pm_raw = fetch_premarket(ticker)
    pm = {
        "high": pm_raw.get("pm_high"),
        "last": pm_raw.get("pm_last"),
        "volume": pm_raw.get("pm_volume") or 0,
    }
    yt = _yahoo(ticker)
    try:
        info = yt.info or {}
    except Exception:
        info = {}
    try:
        hist = yt.history(period="1y", interval="1d", auto_adjust=False)
    except Exception:
        hist = None
    return info, hist, pm


def _spy_return():
    try:
        hist = _yahoo("SPY").history(period="3mo", interval="1d")
        if hist is None or len(hist) < 21:
            return None
        close = hist["Close"].astype(float)
        return (float(close.iloc[-1]) - float(close.iloc[-21])) / float(close.iloc[-21])
    except Exception:
        return None


def _catalyst(ticker):
    key = get_api_key()
    if not key:
        return "none", ""
    today = datetime.now(timezone.utc).date()
    start = today.fromordinal(today.toordinal() - 5)
    rows = _get(
        "/company-news",
        {"symbol": ticker, "from": start.isoformat(), "to": today.isoformat()},
        key,
    )
    if not isinstance(rows, list) or not rows:
        return "none", ""
    text = " ".join(str(r.get("headline") or "") for r in rows[:8]).lower()
    company_words = (
        "earnings", "fda", "approval", "trial", "contract", "partnership",
        "acquisition", "merger", "sec", "analyst", "upgrade", "offering",
    )
    if any(w in text for w in company_words):
        headline = str(rows[0].get("headline") or "")[:140]
        return "company", headline
    return "sector", str(rows[0].get("headline") or "")[:140]


def _candidates():
    names = []
    try:
        from finvizfinance.screener.technical import Technical
        ft = Technical()
        ft.set_filter(filters_dict={
            "Price": "$1 to $10",
            "Country": "USA",
            "Average Volume": "Over 500K",
            "Relative Volume": "Over 2",
        })
        df = ft.screener_view(order="Relative Volume", ascend=False)
        if df is not None and not df.empty and "Ticker" in df.columns:
            names.extend(fix_ticker(str(t).upper()) for t in df["Ticker"].head(MAX_NAMES).tolist())
    except Exception as exc:
        print(f"Finviz: {exc}")
    for t in load_manual_tickers():
        if t not in names:
            names.append(t)
    manual = os.environ.get("STOCK_TICKERS", "")
    for raw in manual.split(","):
        t = fix_ticker(raw.strip().upper())
        if t and t not in names:
            names.append(t)
    return list(dict.fromkeys(names))[:MAX_NAMES]


def scan():
    spy_ret = _spy_return()
    rows = []
    scanned = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    for ticker in _candidates():
        try:
            info, hist, pm = _premarket(ticker)
            fh = enrich_ticker_premarket(ticker, delay=0.35) if get_api_key() else {}
            if pm.get("last") is None and fh.get("fh_price"):
                pm["last"] = fh["fh_price"]
            feat = _features_from_hist(ticker, info, hist, pm)
            if not feat:
                continue
            if feat.get("gap_pct") is None and fh.get("fh_gap_pct") is not None:
                feat["gap_pct"] = fh["fh_gap_pct"]
            if spy_ret is not None and hist is not None and len(hist) >= 21:
                close = hist["Close"].astype(float)
                stock_ret = (float(close.iloc[-1]) - float(close.iloc[-21])) / float(close.iloc[-21])
                feat["rs_vs_spy"] = stock_ret - spy_ret
            kind, text = _catalyst(ticker)
            feat["catalyst"] = kind
            feat["catalyst_text"] = text
            time.sleep(DELAY_BETWEEN)
            plan = evaluate(feat)
            if plan["signal"] == SIGNAL_IGNORE:
                continue
            level = breakout_level(feat["price"], feat["pmh"], feat.get("prev_day_high"), feat.get("intraday_high"))
            rows.append({
                "ticker": ticker,
                "company": feat["company"],
                "price": feat["price"],
                "gap_pct": feat["gap_pct"],
                "float_shares": feat["float_shares"],
                "pm_volume": feat["pm_volume"],
                "dollar_volume": round(feat["dollar_volume"], 2),
                "rvol": feat["rvol"],
                "rocm20": feat["rocm20"],
                "rocm200": feat["rocm200"],
                "vwap": feat["vwap"],
                "atr_pct": feat["atr_pct"],
                "pmh": feat["pmh"],
                "resistance": level,
                "distance_to_resistance_pct": plan["distance_to_resistance_pct"],
                "trigger": plan["trigger"],
                "target": plan["target"],
                "stop": plan["stop"],
                "risk_reward": plan["risk_reward"],
                "catalyst": text or kind,
                "pattern": plan["pattern"],
                "score": plan["score"],
                "signal": plan["signal"],
                "signal_label": plan["signal_label"],
                "reason": plan["reason"],
                "notes": plan["notes"],
                "disclaimer": plan["disclaimer"],
                "recommendation": plan["recommendation"],
                "scanned_at": scanned,
            })
        except Exception as exc:
            print(f"{ticker}: {type(exc).__name__}: {exc}")
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def _log_key(row):
    ticker = str(row.get("ticker") or "").strip().upper()
    day = str(row.get("scanned_at") or "")[:10]
    if not ticker or len(day) < 10:
        return ""
    return f"{day}|{ticker}"


def _clean_log_value(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return value


def append_log(rows, path=LOG_PATH):
    """سهم واحد لكل يوم. الظهور التالي في نفس اليوم يحدّث الصف ويبقي أول ظهور."""
    log_cols = ["first_seen"] + COLUMNS
    existing = []
    if os.path.exists(path):
        try:
            old = pd.read_csv(path)
            existing = old.to_dict("records")
        except Exception:
            existing = []
    if not rows:
        if not os.path.exists(path):
            pd.DataFrame(columns=log_cols).to_csv(path, index=False)
        return 0
    by_key = {}
    for rec in existing:
        key = _log_key(rec)
        if key:
            by_key[key] = {col: _clean_log_value(rec.get(col)) for col in log_cols}
    added = 0
    for row in rows:
        key = _log_key(row)
        if not key:
            continue
        prev = by_key.get(key)
        merged = {col: _clean_log_value(row.get(col)) for col in COLUMNS}
        merged["first_seen"] = (prev or {}).get("first_seen") or row.get("scanned_at") or ""
        if prev is None:
            added += 1
        by_key[key] = merged
    out = list(by_key.values())
    out.sort(key=lambda rec: (str(rec.get("scanned_at") or ""), float(rec.get("score") or 0)), reverse=True)
    pd.DataFrame(out, columns=log_cols).to_csv(path, index=False)
    print(f"السجل: {len(out)} صفاً ({added} جديداً) → {path}")
    return added


def write_results(rows, path=OUT_PATH, log_path=LOG_PATH):
    append_log(rows, log_path)
    if not rows:
        print("لا صفوف مؤهلة — الإبقاء على الملف السابق إن وُجد")
        if not os.path.exists(path):
            pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)
        return
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    print(f"كتبت {len(rows)} سهماً → {path}")


def main():
    write_results(scan())


if __name__ == "__main__":
    main()
