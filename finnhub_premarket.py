"""
finnhub_premarket.py
--------------------
تأكيد صعود بريماركت عبر Finnhub (مو سكرينر سحري).

يستخدم:
  • /quote  → previous close + السعر الحالي / الافتتاح (gap %)
  • /stock/candle (إن توفر) → High/Vol في نافذة premarket تقريباً

المتغير: FINNHUB_API_KEY
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

try:
    import requests
except ImportError:
    requests = None

FINNHUB_BASE = "https://finnhub.io/api/v1"

# عتبات تأكيد صعود بريماركت
MIN_GAP_PCT = 0.8          # فجوة ≥ 0.8% فوق إغلاق أمس
STRONG_GAP_PCT = 2.0       # فجوة قوية
MIN_REL_RANGE = 0.3        # (high-low)/pc كنسبة نشاط دنيا (إن توفر)


def get_api_key():
    return (os.environ.get("FINNHUB_API_KEY")
            or os.environ.get("FINNHUB_TOKEN")
            or "").strip()


def _get(path, params, key, timeout=8):
    if not requests or not key:
        return None
    params = dict(params or {})
    params["token"] = key
    try:
        r = requests.get(f"{FINNHUB_BASE}{path}", params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def fetch_quote(ticker, key):
    data = _get("/quote", {"symbol": ticker}, key)
    if not data or not isinstance(data, dict):
        return None
    # c=current, o=open, h=high, l=low, pc=prev close
    if not data.get("pc") or data.get("c") in (None, 0):
        return None
    return data


def fetch_premarket_candle_stats(ticker, key):
    """
    محاولة جلب شموع 5 دقائق من ~4:00 إلى 9:30 ET.
    على الخطة المجانية قد تفشل — نرجع None بهدوء.
    """
    # تقريب: ET ≈ UTC-4 (صيفي) — نافذة واسعة تغطي premarket
    now = datetime.now(timezone.utc)
    # آخر 16 ساعة كافية لتغطية جلسة اليوم
    t_to = int(now.timestamp())
    t_from = int((now - timedelta(hours=16)).timestamp())
    data = _get("/stock/candle", {
        "symbol": ticker,
        "resolution": "5",
        "from": t_from,
        "to": t_to,
    }, key)
    if not data or data.get("s") != "ok":
        return None
    highs = data.get("h") or []
    lows = data.get("l") or []
    vols = data.get("v") or []
    if not highs:
        return None
    return {
        "pm_high_est": round(max(highs), 2),
        "pm_low_est": round(min(lows), 2),
        "vol_sum": int(sum(vols)) if vols else 0,
    }


def analyze_bullish_premarket(ticker, key=None):
    """
    يعيد dict:
      gap_pct, bullish, strong, score_bonus, note, quote fields
    """
    key = key or get_api_key()
    out = {
        "fh_gap_pct": None,
        "fh_pm_bullish": False,
        "fh_pm_strong": False,
        "fh_pm_score": 0,
        "fh_pm_note": "",
        "fh_price": None,
        "fh_prev_close": None,
        "fh_open": None,
    }
    if not key:
        out["fh_pm_note"] = "لا يوجد FINNHUB_API_KEY"
        return out

    q = fetch_quote(ticker, key)
    if not q:
        out["fh_pm_note"] = "Finnhub quote فشل"
        return out

    price = float(q.get("c") or 0)
    prev = float(q.get("pc") or 0)
    open_ = float(q.get("o") or 0)
    high = float(q.get("h") or 0)
    low = float(q.get("l") or 0)

    out["fh_price"] = price
    out["fh_prev_close"] = prev
    out["fh_open"] = open_ if open_ else None

    if prev <= 0:
        out["fh_pm_note"] = "prev close ناقص"
        return out

    # فجوة مقابل إغلاق أمس — نستخدم أعلى من (الافتتاح، السعر الحالي)
    ref = max(price, open_ if open_ > 0 else price)
    gap_pct = (ref - prev) / prev * 100.0
    out["fh_gap_pct"] = round(gap_pct, 2)

    candle = fetch_premarket_candle_stats(ticker, key)
    held_above_open = open_ > 0 and price >= open_ * 0.998
    range_ok = True
    if high > 0 and low > 0 and prev > 0:
        rel = (high - low) / prev * 100
        range_ok = rel >= MIN_REL_RANGE or gap_pct >= MIN_GAP_PCT

    bullish = gap_pct >= MIN_GAP_PCT and held_above_open and range_ok
    strong = gap_pct >= STRONG_GAP_PCT and held_above_open

    # إن وُجدت شموع وتجاوز high تقدير البريماركت السابق إغلاق → دعم إضافي
    if candle and candle.get("pm_high_est"):
        if candle["pm_high_est"] > prev * (1 + MIN_GAP_PCT / 100):
            bullish = True
            if candle["pm_high_est"] > prev * (1 + STRONG_GAP_PCT / 100):
                strong = True

    out["fh_pm_bullish"] = bool(bullish)
    out["fh_pm_strong"] = bool(strong)

    bonus = 0
    notes = []
    if strong:
        bonus += 4
        notes.append(f"PM صعود قوي gap {gap_pct:+.1f}%")
    elif bullish:
        bonus += 3
        notes.append(f"PM صعود gap {gap_pct:+.1f}%")
    elif gap_pct > 0:
        notes.append(f"فجوة ضعيفة {gap_pct:+.1f}%")
    else:
        notes.append(f"بدون فجوة صاعدة {gap_pct:+.1f}%")

    if candle and candle.get("vol_sum"):
        notes.append(f"vol≈{candle['vol_sum']:,}")

    out["fh_pm_score"] = bonus
    out["fh_pm_note"] = " | ".join(notes)
    return out


def enrich_ticker_premarket(ticker, key=None, delay=0.25):
    """تحليل سهم واحد + تأخير بسيط لاحترام حدود API."""
    res = analyze_bullish_premarket(ticker, key=key)
    if delay > 0:
        time.sleep(delay)
    return res


def enrich_rows_premarket(rows, key=None, delay=0.2):
    """
    rows: list[dict] فيها مفتاح Ticker
    يضيف حقول fh_* لكل صف.
    """
    key = key or get_api_key()
    if not key:
        for r in rows:
            r["fh_pm_bullish"] = False
            r["fh_pm_strong"] = False
            r["fh_gap_pct"] = None
            r["fh_pm_note"] = "لا يوجد FINNHUB_API_KEY"
            r["fh_pm_score"] = 0
        return rows

    for r in rows:
        t = str(r.get("Ticker") or r.get("ticker") or "").upper().strip()
        if not t:
            continue
        info = enrich_ticker_premarket(t, key=key, delay=delay)
        r.update(info)
    return rows


if __name__ == "__main__":
    key = get_api_key()
    print("FINNHUB key:", "yes" if key else "NO")
    for t in ("SPY", "NVDA", "AAPL"):
        print(t, analyze_bullish_premarket(t, key))
        time.sleep(0.3)
