"""
cheap_options_screener_v3.py  ·  Options Trading Assistant
===========================================================
مصادر البيانات (كلها مجانية 100%):
  Finviz         → فلترة الأسهم (أعلى حجم + تقلب)
  Yahoo Finance  → Option Chain (Bid/Ask/IV/OI/Volume)
  Yahoo Finance  → Premarket (High/Low/Volume/Last)
  Yahoo Finance  → 60 أيام يومي لحساب المؤشرات الفنية

حسابات محلية (لا تحتاج API مدفوع):
  Black-Scholes  → تسعير العقود + Greeks
  EMA20/50/200   → من التاريخ اليومي
  RSI(14)        → من التاريخ اليومي
  VWAP(20d)      → من التاريخ اليومي
  ATR(14)        → من التاريخ اليومي
  Entry/Stop/TP  → من Premarket High + ATR

تقرير كل صفقة يحتوي:
  ★ Rating · Confidence % · Probability %
  BUY / WAIT / AVOID
  Entry (فوق Premarket High)
  Stop Loss (ATR-based + VWAP)
  TP1 / TP2 / TP3 + قيمة العقد عند كل هدف
  Risk:Reward لكل هدف
  جدول سيناريوهات Black-Scholes كامل
  بريكإيفن + نقطة تضاعف العقد

Install (في Colab):
  !pip install finvizfinance yfinance pandas scipy numpy
"""

import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

import math
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from finvizfinance.screener.technical import Technical

# ── curl_cffi session يقلّد Chrome الحقيقي لتجاوز حجب Yahoo Finance ──
try:
    from curl_cffi import requests as _curl_requests
    _YF_SESSION = _curl_requests.Session(impersonate="chrome")
except ImportError:
    import requests as _req
    _YF_SESSION = _req.Session()
    _YF_SESSION.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    })


# ════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════
TARGET_PREM_MIN  = 0.50    # أقل سعر عقد مقبول ($)
TARGET_PREM_MAX  = 3.00    # أعلى سعر عقد مقبول ($)
MIN_OI           = 500     # أقل Open Interest
MAX_SPREAD_PCT = 0.15    # أقصى spread مقبول (15% كـ decimal)
PM_STALE_PCT   = 0.05    # تجاهل pm_high/pm_low إذا ابتعدت عن السعر >5%
ENTRY_MAX_DRIFT = 0.05   # Entry يجب أن يكون ضمن ±5% من السعر الحالي
MAX_STOCK_PRICE = 500.0  # رفض أسعار Yahoo الشاذة (مثل SNDK $1183)
MIN_IV_DISPLAY  = 0.01   # IV أقل من 1% تُعامل كبيانات ناقصة
DTE_TARGET       = 7       # الهدف: 7 أيام للانتهاء
DTE_WINDOW       = 8       # ±8 أيام حول الهدف
MAX_STOCKS_DEEP  = 40      # كم سهم ندرس بعمق
DELAY_BETWEEN    = 0.30    # ثواني بين كل طلب yfinance
MANUAL_TICKERS_FILE = "manual_tickers.txt"
# finviz | manual | both  (both = Finviz + قائمة يدوية، مع fallback)
DEFAULT_SCREENER_SOURCE = "both"

RISK_FREE_RATE   = 0.053   # معدل الفائدة لـ Black-Scholes
MIN_RR_TO_BUY    = 1.3     # أقل R:R لإعطاء BUY
MIN_CONF_TO_BUY  = 55      # أقل Confidence% لإعطاء BUY

FINVIZ_FILTERS = {
    "Option/Short":    "Optionable",
    "Average Volume":  "Over 1M",        # حجم يومي أعلى = spread أضيق
    "Country":         "USA",
    "Price":           "Over $5",        # تجنب penny stocks
    "Relative Volume": "Over 1.5",       # نشاط عالي اليوم (RVOL > 1.5x)
}


# ════════════════════════════════════════════════════════════════════════
#  BLACK-SCHOLES ENGINE
# ════════════════════════════════════════════════════════════════════════

def bs_price(S, K, T, r, sigma, opt):
    """تسعير الأوبشن بـ Black-Scholes. opt = 'call' أو 'put'"""
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0:
        return max(S - K, 0) if opt == "call" else max(K - S, 0)
    if sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "call":
        return max(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2), 0)
    return max(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0)


def bs_greeks(S, K, T, r, sigma, opt):
    """حساب Greeks محليًا بدون API مدفوع."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    sqrt_T = math.sqrt(T)

    if opt == "call":
        delta = norm.cdf(d1)
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
                 - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
                 + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega  = S * pdf_d1 * sqrt_T / 100

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega":  round(vega,  4),
    }


# ════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════

def fix_ticker(t):
    """إصلاح بج finvizfinance: يضاعف الحرف الأول (AARKB → ARKB)."""
    t = str(t).strip()
    return t[1:] if len(t) >= 2 and t[0] == t[1] else t


def parse_num(v):
    """تحويل '1.23M' أو '850K' إلى float."""
    s = str(v).replace(",", "").strip().upper()
    try:
        if s.endswith("B"): return float(s[:-1]) * 1e9
        if s.endswith("M"): return float(s[:-1]) * 1e6
        if s.endswith("K"): return float(s[:-1]) * 1e3
        return float(s.replace("%", "").replace("$", ""))
    except Exception:
        return 0.0


def _safe_int(v):
    if v is None:
        return 0
    try:
        if pd.isna(v):
            return 0
    except (TypeError, ValueError):
        pass
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def effective_oi(oi, opt_vol):
    """Yahoo أحياناً يرجّع OI=0 — استخدم opt_vol كبديل مؤقت."""
    oi = _safe_int(oi)
    ov = _safe_int(opt_vol)
    if oi >= MIN_OI:
        return oi
    if oi == 0 and ov >= MIN_OI:
        return ov
    return oi


def pm_data_fresh(pm_val, price):
    if not pm_val or not price or price <= 0:
        return False
    return abs(float(pm_val) - price) / price <= PM_STALE_PCT


def trade_plan_is_valid(plan, price, is_call):
    entry = plan.get("entry_stock")
    stop  = plan.get("stop_stock")
    tp1   = plan.get("tp1_stock")
    if not all([entry, stop, tp1]) or price <= 0:
        return False
    entry, stop, tp1 = float(entry), float(stop), float(tp1)
    if is_call:
        if entry > price * (1 + ENTRY_MAX_DRIFT):
            return False
        return tp1 > entry > stop
    if entry < price * (1 - ENTRY_MAX_DRIFT):
        return False
    return tp1 < entry < stop


def parse_price_num(val):
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    return parse_num(str(val).replace("$", ""))


def row_passes_save_filters(r):
    """فلترة نهائية موحّدة: OI · spread · premium · سعر منطقي."""
    price = float(r.get("price_num") or parse_price_num(r.get("Price")) or 0)
    prem  = float(r.get("premium") or 0)
    sp    = float(r.get("spread_pct") if r.get("spread_pct") is not None else 99)
    oi    = effective_oi(r.get("oi"), r.get("opt_vol"))
    strike = float(r.get("strike") or 0)

    if price <= 0 or price > MAX_STOCK_PRICE:
        return False
    if prem < TARGET_PREM_MIN or prem > TARGET_PREM_MAX:
        return False
    if oi < MIN_OI or sp > MAX_SPREAD_PCT:
        return False
    if strike > 0 and price > 0 and abs(strike - price) / price > 0.40:
        return False
    return True


def filter_results_df(df):
    if df is None or df.empty:
        return df
    mask = df.apply(row_passes_save_filters, axis=1)
    return df[mask].copy()


def dashboard_fallback_df(df, max_rows=10):
    """أفضل المرشّحين للـ Dashboard عندما الفلتر الصارم يمنع الكل."""
    if df is None or df.empty:
        return df
    work = df.copy()
    prices = work.apply(
        lambda r: float(r.get("price_num") or parse_price_num(r.get("Price")) or 0),
        axis=1,
    )
    work = work[
        (work["premium"].fillna(0) > 0)
        & (prices > 0)
        & (prices <= MAX_STOCK_PRICE)
    ].copy()
    if work.empty:
        return work
    wait_buy = work[work["recommendation"].isin(["BUY", "WAIT"])]
    if not wait_buy.empty:
        return wait_buy.sort_values("Score", ascending=False).head(max_rows)
    return work.sort_values("Score", ascending=False).head(max_rows)


def save_results_csv(filtered_df, path="options_v3_results.csv"):
    """لا تستبدل CSV صالح بملف فاضي عند فشل Yahoo."""
    if filtered_df.empty:
        if os.path.exists(path):
            try:
                prev = pd.read_csv(path)
                if len(prev) > 0:
                    print(f"  ⚠️  0 matches — keeping previous {path} ({len(prev)} rows)")
                    return False
            except Exception:
                pass
        print("  ⚠️  0 matches — no valid previous CSV to keep")
        return False
    filtered_df.to_csv(path, index=False)
    return True


def find_best_expiry(expirations, dte_target=7, window=8):
    """أقرب انتهاء لـ dte_target ضمن ±window أيام."""
    today  = datetime.now().date()
    target = today + timedelta(days=dte_target)
    best, best_diff = None, 9999
    for exp in expirations:
        try:
            d    = datetime.strptime(exp, "%Y-%m-%d").date()
            diff = abs((d - target).days)
            if diff < best_diff and d > today:
                best_diff = diff
                best = exp
        except Exception:
            continue
    return best if best_diff <= window else None


# ════════════════════════════════════════════════════════════════════════
#  PREMARKET DATA
# ════════════════════════════════════════════════════════════════════════

def fetch_premarket(ticker):
    """
    يجلب بيانات الـ Premarket من Yahoo Finance (مجاني).
    يعيد: pm_high, pm_low, pm_last, pm_volume
    """
    out = {"pm_high": None, "pm_low": None, "pm_last": None, "pm_volume": None}
    try:
        yt   = yf.Ticker(ticker, session=_YF_SESSION)
        hist = yt.history(period="1d", interval="1m", prepost=True)
        if hist.empty:
            return out
        # Premarket = 04:00 – 09:29
        pm = hist.between_time("04:00", "09:29")
        if pm.empty:
            return out
        out["pm_high"]   = round(float(pm["High"].max()),  2)
        out["pm_low"]    = round(float(pm["Low"].min()),   2)
        out["pm_last"]   = round(float(pm["Close"].iloc[-1]), 2)
        out["pm_volume"] = int(pm["Volume"].sum())
    except Exception:
        pass
    return out


# ════════════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS  (من التاريخ اليومي – لا يحتاج API مدفوع)
# ════════════════════════════════════════════════════════════════════════

def compute_ema(prices, period):
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    k   = 2.0 / (period + 1)
    ema = float(np.mean(prices[:period]))
    for p in prices[period:]:
        ema = float(p) * k + ema * (1 - k)
    return round(ema, 4)


def compute_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = float(np.mean(gains[:period]))
    avg_l  = float(np.mean(losses[:period]))
    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    if avg_l == 0:
        return 100.0
    return round(100 - (100 / (1 + avg_g / avg_l)), 1)


def compute_technicals(hist, current_price):
    """
    EMA20/50/200 · RSI(14) · VWAP(20d) · ATR(14)
    مستويات الدعم والمقاومة من آخر 20 شمعة
    """
    res = {
        "ema20": None, "ema50": None, "ema200": None,
        "rsi": 50.0,   "vwap": None,  "atr14": None,
        "above_ema20": False, "above_ema50": False, "above_ema200": False,
        "above_vwap":  False,
        "resist1": None, "resist2": None, "resist3": None,
        "support1": None,
        "trend": "UNKNOWN",
        "yesterday_high": None, "yesterday_low": None,
    }
    if hist is None or len(hist) < 5:
        return res

    closes = hist["Close"].astype(float).tolist()
    highs  = hist["High"].astype(float).tolist()
    lows   = hist["Low"].astype(float).tolist()
    vols   = hist["Volume"].astype(float).tolist()
    n      = len(closes)

    # EMA
    if n >= 20:
        res["ema20"] = compute_ema(closes, 20)
        res["above_ema20"] = current_price > res["ema20"]
    if n >= 50:
        res["ema50"] = compute_ema(closes, 50)
        res["above_ema50"] = current_price > res["ema50"]
    elif n >= 20:
        res["ema50"] = compute_ema(closes, n)
        res["above_ema50"] = current_price > res["ema50"]
    res["ema200"] = compute_ema(closes, min(200, n))
    res["above_ema200"] = current_price > res["ema200"]

    # RSI
    res["rsi"] = compute_rsi(closes, 14)

    # VWAP (آخر 20 يوم)
    t20 = [(h + l + c) / 3 for h, l, c in zip(highs[-20:], lows[-20:], closes[-20:])]
    v20 = vols[-20:]
    tv  = sum(t * v for t, v in zip(t20, v20))
    tot = sum(v20)
    if tot > 0:
        res["vwap"] = round(tv / tot, 2)
        res["above_vwap"] = current_price > res["vwap"]

    # ATR(14)
    if n >= 2:
        trs = []
        for i in range(max(1, n - 14), n):
            tr = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i]  - closes[i - 1]))
            trs.append(tr)
        if trs:
            res["atr14"] = round(sum(trs) / len(trs), 4)

    # أعلى وأدنى أمس
    if n >= 2:
        res["yesterday_high"] = round(highs[-2], 2)
        res["yesterday_low"]  = round(lows[-2],  2)

    # مستويات المقاومة (Swing Highs فوق السعر الحالي)
    atr = res["atr14"] or (current_price * 0.025)
    swing_h = sorted(
        set(round(h, 2) for h in highs[-20:] if h > current_price * 1.005),
        reverse=True
    )
    deduped = []
    for h in sorted(swing_h):
        if not deduped or h - deduped[-1] > atr * 0.4:
            deduped.append(h)

    r1 = deduped[0] if len(deduped) >= 1 else round(current_price + atr * 1.0, 2)
    r2 = deduped[1] if len(deduped) >= 2 else round(r1 + atr * 0.8, 2)
    r3 = deduped[2] if len(deduped) >= 3 else round(r2 + atr * 1.0, 2)
    r1 = max(r1, round(current_price + atr * 0.7, 2))
    r2 = max(r2, round(r1 + atr * 0.5, 2))
    r3 = max(r3, round(r2 + atr * 0.7, 2))
    res["resist1"] = round(r1, 2)
    res["resist2"] = round(r2, 2)
    res["resist3"] = round(r3, 2)

    # مستوى الدعم (Swing Low أسفل السعر)
    swing_l = sorted(
        set(round(l, 2) for l in lows[-20:] if l < current_price * 0.995)
    )
    res["support1"] = swing_l[-1] if swing_l else round(current_price - atr * 1.2, 2)

    # الاتجاه
    above_count = sum([res["above_ema20"], res["above_ema50"], res["above_ema200"]])
    rsi_v = res["rsi"]
    if   above_count == 3 and rsi_v > 55: res["trend"] = "STRONG BULLISH 🟢"
    elif above_count >= 2 and rsi_v > 48: res["trend"] = "BULLISH 🟢"
    elif above_count == 1:                res["trend"] = "MIXED ⚪"
    elif above_count == 0 and rsi_v < 45: res["trend"] = "STRONG BEARISH 🔴"
    else:                                  res["trend"] = "BEARISH 🔴"

    return res


# ════════════════════════════════════════════════════════════════════════
#  OPTIONS DATA  (Yahoo Finance – مجاني)
# ════════════════════════════════════════════════════════════════════════

def get_earnings_date(yt_ticker):
    try:
        cal = yt_ticker.calendar
        if cal is None: return None
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if dates: return str(dates[0])[:10]
        if hasattr(cal, "iloc"):
            val = cal.iloc[0, 0]
            return str(val)[:10] if val else None
    except Exception:
        pass
    return None


def fetch_options_data(ticker, stock_price):
    out = {
        "avg_vol": None, "float_shares": None, "earnings": None,
        "premium": None, "iv": None,      "oi": None,
        "opt_vol": None, "spread_pct": None, "expiry": None,
        "has_weekly": False, "direction": "?",
        "strike": None,  "dte_num": 7,
        "hist": None,    "error": None,
        # Premarket
        "pm_high": None, "pm_low": None,
        "pm_last": None, "pm_volume": None,
        "today_vol_yf": None,
    }
    try:
        yt   = yf.Ticker(ticker, session=_YF_SESSION)
        info = yt.info or {}

        out["avg_vol"]      = (info.get("averageVolume")
                               or info.get("averageDailyVolume10Day"))
        out["float_shares"] = info.get("floatShares")
        out["earnings"]     = get_earnings_date(yt)
        out["today_vol_yf"] = int(info.get("volume")
                                  or info.get("regularMarketVolume") or 0)

        # 60 يوم يومي للمؤشرات الفنية
        try:
            hist = yt.history(period="60d", interval="1d")
            if not hist.empty:
                out["hist"] = hist
        except Exception:
            pass

        # Premarket (بيانات اليوم بدقة دقيقة)
        try:
            hist_pm = yt.history(period="1d", interval="1m", prepost=True)
            if not hist_pm.empty:
                pm = hist_pm.between_time("04:00", "09:29")
                if not pm.empty:
                    out["pm_high"]   = round(float(pm["High"].max()),     2)
                    out["pm_low"]    = round(float(pm["Low"].min()),      2)
                    out["pm_last"]   = round(float(pm["Close"].iloc[-1]), 2)
                    out["pm_volume"] = int(pm["Volume"].sum())
        except Exception:
            pass

        # Options chain
        exps = yt.options
        if not exps:
            out["error"] = "no_options"; return out

        exp = find_best_expiry(exps, DTE_TARGET, DTE_WINDOW)
        if exp is None:
            exp = exps[0] if exps else None
        if exp is None:
            out["error"] = "no_expiry"; return out

        out["expiry"]     = exp
        exp_date          = datetime.strptime(exp, "%Y-%m-%d").date()
        dte               = (exp_date - datetime.now().date()).days
        out["dte_num"]    = dte
        out["has_weekly"] = dte <= 14

        chain = yt.option_chain(exp)
        calls = chain.calls.copy() if chain.calls is not None else pd.DataFrame()
        puts  = chain.puts.copy()  if chain.puts  is not None else pd.DataFrame()

        if calls.empty:
            out["error"] = "empty_chain"; return out

        calls["_dist"] = abs(calls["strike"] - stock_price)
        atm_c = calls.loc[calls["_dist"].idxmin()]

        bid  = float(atm_c.get("bid")       or 0)
        ask  = float(atm_c.get("ask")       or 0)
        last = float(atm_c.get("lastPrice") or 0)
        if ask == 0 and bid == 0:
            if last > 0:
                bid = round(last * 0.97, 2)
                ask = round(last * 1.03, 2)
            else:
                out["error"] = "no_market"; return out

        out["premium"]    = round((bid + ask) / 2, 2)
        out["iv"]         = round(float(atm_c.get("impliedVolatility") or 0), 4)
        out["oi"]         = int(atm_c.get("openInterest") or 0)
        out["opt_vol"]    = int(atm_c.get("volume") or 0)
        out["spread_pct"] = round((ask - bid) / ask, 4) if ask > 0 else None
        out["strike"]     = float(atm_c.get("strike") or 0)

        # إشارة الاتجاه: Call OI مقابل Put OI
        if not puts.empty:
            puts["_dist"] = abs(puts["strike"] - stock_price)
            atm_p = puts.loc[puts["_dist"].idxmin()]
            poi   = int(atm_p.get("openInterest") or 0)
            coi   = out["oi"]
            if coi > 0 and poi > 0:
                ratio = coi / poi
                out["direction"] = ("CALL 📈" if ratio > 1.4
                                    else "PUT  📉" if ratio < 0.7
                                    else "NEUTRAL")

    except Exception as e:
        out["error"] = str(e)[:60]
    return out


# ════════════════════════════════════════════════════════════════════════
#  SCORING
# ════════════════════════════════════════════════════════════════════════

def score_stock(row):
    score = 0
    notes = []

    today_vol = float(row.get("today_vol", 0) or 0)
    avg_vol   = float(row.get("avg_vol", 0) or 0)
    opt_vol   = float(row.get("opt_vol", 0) or 0)
    if today_vol <= 0 and opt_vol > 0 and avg_vol > 0:
        # Finviz volume أحياناً 0 — قدّر من حجم العقود
        today_vol = min(opt_vol * 100.0, avg_vol * 3.0)
        row["today_vol"] = today_vol
    rvol = (today_vol / avg_vol) if avg_vol > 0 else 0
    row["RVOL"] = round(rvol, 2)
    if rvol > 2.0:     score += 3; notes.append("RVOL>2 +3")
    elif rvol > 1.5:   score += 2; notes.append("RVOL>1.5 +2")
    elif rvol > 1.2:   score += 1; notes.append("RVOL>1.2 +1")

    if avg_vol and avg_vol > 30e6: score += 2; notes.append("MegaCap +2")
    elif avg_vol and avg_vol > 10e6: score += 1; notes.append("LargeCap +1")

    oi = effective_oi(row.get("oi", 0), row.get("opt_vol", 0))
    row["oi"] = oi
    if oi > 2000:      score += 3; notes.append("OI>2K +3")
    elif oi > 1000:    score += 2; notes.append("OI>1K +2")
    elif oi > MIN_OI:  score += 1; notes.append(f"OI>{MIN_OI} +1")

    ov = row.get("opt_vol", 0) or 0
    if ov > 1000:      score += 3; notes.append("OptVol>1K +3")
    elif ov > 500:     score += 2; notes.append("OptVol>500 +2")
    elif ov > 100:     score += 1; notes.append("OptVol>100 +1")

    sp = row.get("spread_pct")
    if sp is not None:
        if sp < 0.03:   score += 3; notes.append("Spread<3% +3")
        elif sp < 0.05: score += 2; notes.append("Spread<5% +2")
        elif sp < 0.08: score += 1; notes.append("Spread<8% +1")
        else:           score -= 1; notes.append("Spread>8% -1")

    if row.get("has_weekly"): score += 2; notes.append("Weekly✓ +2")

    prem = row.get("premium", 0) or 0
    if TARGET_PREM_MIN <= prem <= TARGET_PREM_MAX:
        score += 2; notes.append("Prem✓ +2")

    atr_pct = row.get("atr_pct", 0) or 0
    if atr_pct > 5:     score += 3; notes.append("ATR%>5 +3")
    elif atr_pct > 3:   score += 2; notes.append("ATR%>3 +2")
    elif atr_pct > 1.5: score += 1; notes.append("ATR%>1.5 +1")

    earnings = row.get("earnings")
    expiry   = row.get("expiry")
    row["earn_before_expiry"] = False
    if earnings and earnings not in ("None", "N/A"):
        try:
            earn_date = datetime.strptime(earnings[:10], "%Y-%m-%d").date()
            today     = datetime.now().date()
            days_to   = (earn_date - today).days

            # أخطر حالة: الإيرنينجز قبل أو عند انتهاء العقد
            if expiry:
                exp_date = datetime.strptime(expiry[:10], "%Y-%m-%d").date()
                if earn_date <= exp_date:
                    score -= 5
                    notes.append("⚠️⚠️ Earn BEFORE Expiry -5")
                    row["earn_before_expiry"] = True
                elif 0 <= days_to <= 5:
                    score -= 2; notes.append("Earnings≤5d ⚠️ -2")
                else:
                    score += 2; notes.append("NoEarnings✓ +2")
            else:
                if 0 <= days_to <= 5:
                    score -= 2; notes.append("Earnings≤5d ⚠️ -2")
                else:
                    score += 2; notes.append("NoEarnings✓ +2")
        except Exception:
            score += 1; notes.append("Earnings? +1")
    else:
        score += 1; notes.append("NoEarnings? +1")

    try:
        rsi_f = float(str(row.get("rsi", 50)).replace("%", ""))
        if 28 <= rsi_f <= 44:    score += 1; notes.append("RSI oversold +1")
        elif 56 <= rsi_f <= 72:  score += 1; notes.append("RSI overbought +1")
    except Exception:
        pass

    fl = row.get("float_shares", 0) or 0
    if 0 < fl < 50e6:   score += 2; notes.append("Float<50M +2")
    elif 0 < fl < 100e6: score += 1; notes.append("Float<100M +1")

    try:
        gap_f = float(str(row.get("gap_pct", 0)).replace("%", ""))
        if abs(gap_f) > 4:   score += 2; notes.append(f"Gap{gap_f:+.1f}% +2")
        elif abs(gap_f) > 2: score += 1; notes.append(f"Gap{gap_f:+.1f}% +1")
    except Exception:
        pass

    # بونص Premarket Volume (نشاط قبل الافتتاح)
    pm_vol = row.get("pm_volume") or 0
    if pm_vol > 500_000: score += 2; notes.append("PM_Vol>500K +2")
    elif pm_vol > 100_000: score += 1; notes.append("PM_Vol>100K +1")

    return score, notes


# ════════════════════════════════════════════════════════════════════════
#  TRADE PLAN
# ════════════════════════════════════════════════════════════════════════

def compute_trade_plan(r, tech):
    """
    يحسب: Entry · Stop · TP1/TP2/TP3 + قيمة العقد عند كل هدف
    Risk/Reward · Confidence% · Probability% · Recommendation
    """
    plan = {
        "entry_stock": None, "entry_note": "",
        "stop_stock":  None, "stop_option": None,
        "tp1_stock": None, "tp1_option": None, "tp1_rr": None,
        "tp2_stock": None, "tp2_option": None, "tp2_rr": None,
        "tp3_stock": None, "tp3_option": None, "tp3_rr": None,
        "risk_per_contract": None,
        "greeks": None,
        "recommendation": "AVOID", "rec_note": "",
        "confidence": 0, "probability": 0, "stars": 0,
    }

    # إصلاح 1: NEUTRAL يُحلّ باستخدام المؤشرات الفنية بدل افتراض PUT
    direction_raw = str(r.get("direction", "")).upper()
    if "CALL" in direction_raw:
        is_call = True
    elif "PUT" in direction_raw:
        is_call = False
    else:
        # NEUTRAL → نقرر بناءً على EMA20 + RSI + VWAP
        rsi_early   = float(tech.get("rsi", 50) or 50)
        above_ema20 = tech.get("above_ema20", False)
        above_vwap  = tech.get("above_vwap",  False)
        bull_signals = sum([above_ema20, above_vwap, rsi_early > 50])
        is_call = bull_signals >= 2

    price    = float(r.get("price_num", 0) or 0)
    strike   = float(r.get("strike",    0) or price)
    iv_raw   = float(r.get("iv", 0) or 0)
    iv_estimated = iv_raw < MIN_IV_DISPLAY
    iv       = iv_raw if not iv_estimated else 0.4
    dte      = int(r.get("dte_num",     7) or 7)
    premium  = float(r.get("premium",   0) or 0)
    atr      = float(tech.get("atr14") or (price * 0.025))
    opt_type = "call" if is_call else "put"

    if price == 0 or premium == 0:
        return plan
    if iv_estimated:
        plan["rec_note"] = "IV من Yahoo ناقص — R:R تقديري"

    T_entry = max(dte / 365, 1 / 365)
    T_half  = max((dte / 2) / 365, 1 / 365)

    # Greeks المحسوبة محليًا
    plan["greeks"] = bs_greeks(price, strike, T_entry, RISK_FREE_RATE, iv, opt_type)

    # ── Entry ─────────────────────────────────────────────────────────────
    # الأولوية: Premarket High → أعلى أمس → السعر الحالي + ATR buffer
    pm_high = r.get("pm_high")
    pm_low  = r.get("pm_low")
    yest_h  = tech.get("yesterday_high") or price
    yest_l  = tech.get("yesterday_low")  or price

    if is_call:
        use_pm   = pm_high and pm_data_fresh(pm_high, price) and pm_high > price * 0.98
        ref_high = pm_high if use_pm else yest_h
        entry    = round(ref_high + atr * 0.05, 2)
        src      = "Premarket" if use_pm else "أعلى أمس"
        plan["entry_note"] = f"فوق {src} High ${ref_high:.2f}"
    else:
        use_pm  = pm_low and pm_data_fresh(pm_low, price) and pm_low < price * 1.02
        ref_low = pm_low if use_pm else yest_l
        entry   = round(ref_low - atr * 0.05, 2)
        src     = "Premarket" if use_pm else "أدنى أمس"
        plan["entry_note"] = f"تحت {src} Low ${ref_low:.2f}"

    plan["entry_stock"] = entry

    # ── Stop Loss ─────────────────────────────────────────────────────────
    if is_call:
        stop_stock = round(entry - atr * 1.1, 2)
        vwap = tech.get("vwap")
        if vwap and vwap < entry * 0.99:
            stop_stock = max(stop_stock, round(vwap - atr * 0.15, 2))
    else:
        stop_stock = round(entry + atr * 1.1, 2)

    stop_opt = bs_price(stop_stock, strike, T_entry, RISK_FREE_RATE, iv, opt_type)
    plan["stop_stock"]  = stop_stock
    plan["stop_option"] = round(stop_opt, 2)

    risk_per = max((premium - stop_opt) * 100, 1.0)
    plan["risk_per_contract"] = round(risk_per, 2)

    # ── Targets ───────────────────────────────────────────────────────────
    if is_call:
        tp1 = tech.get("resist1") or round(entry + atr * 1.1, 2)
        tp2 = tech.get("resist2") or round(entry + atr * 1.9, 2)
        tp3 = tech.get("resist3") or round(entry + atr * 3.0, 2)
    else:
        tp1 = tech.get("support1") or round(entry - atr * 1.1, 2)
        tp2 = round(entry - atr * 1.9, 2)
        tp3 = round(entry - atr * 3.0, 2)

    def opt_at(target_price):
        # نستخدم نصف DTE لأننا نخرج قبل الانتهاء في الغالب
        return round(bs_price(target_price, strike, T_half,
                               RISK_FREE_RATE, iv, opt_type), 2)

    def rr(opt_val):
        reward = (opt_val - premium) * 100
        return round(reward / risk_per, 1) if risk_per > 0 else 0.0

    o1, o2, o3 = opt_at(tp1), opt_at(tp2), opt_at(tp3)
    plan.update({
        "tp1_stock": round(tp1, 2), "tp1_option": o1, "tp1_rr": rr(o1),
        "tp2_stock": round(tp2, 2), "tp2_option": o2, "tp2_rr": rr(o2),
        "tp3_stock": round(tp3, 2), "tp3_option": o3, "tp3_rr": rr(o3),
    })

    # ── Confidence % ──────────────────────────────────────────────────────
    raw = r.get("Score", 0) or 0
    conf = min(int(raw / 30 * 100), 90)
    if tech.get("above_ema20"):  conf += 4
    if tech.get("above_ema50"):  conf += 3
    if tech.get("above_vwap"):   conf += 5
    rsi_v = tech.get("rsi", 50)
    if is_call  and 50 < rsi_v < 75: conf += 3
    if not is_call and 25 < rsi_v < 50: conf += 3
    if plan["tp2_rr"] and plan["tp2_rr"] >= 2: conf += 3
    # Premarket نشاط يزيد الثقة
    pm_vol = r.get("pm_volume") or 0
    if pm_vol > 500_000: conf += 3
    elif pm_vol > 100_000: conf += 1
    plan["confidence"] = max(10, min(conf, 94))

    # ── Probability % ─────────────────────────────────────────────────────
    prob = 48
    if is_call:
        if tech.get("above_ema20"):  prob += 9
        if tech.get("above_ema50"):  prob += 5
        if tech.get("above_vwap"):   prob += 8
        if rsi_v > 55: prob += 5
        if rsi_v > 65: prob += 3
    else:
        if not tech.get("above_ema20"): prob += 9
        if not tech.get("above_ema50"): prob += 5
        if not tech.get("above_vwap"):  prob += 8
        if rsi_v < 45: prob += 5
        if rsi_v < 35: prob += 3

    rvol = r.get("RVOL", 0) or 0
    if rvol > 2.5:   prob += 8
    elif rvol > 1.5: prob += 4

    sp = r.get("spread_pct") or 0.1
    if sp < 0.03: prob += 5
    elif sp < 0.05: prob += 2

    oi = r.get("oi", 0) or 0
    if oi > 2000: prob += 5
    elif oi > 500: prob += 2

    # Premarket Gap Up يدعم الكول
    if pm_high and is_call and pm_high > price * 1.01:
        prob += 5
    if pm_low and not is_call and pm_low < price * 0.99:
        prob += 5

    earn = r.get("earnings")
    if earn and earn not in ("None", "N/A"):
        try:
            d = (datetime.strptime(earn[:10], "%Y-%m-%d").date()
                 - datetime.now().date()).days
            if 0 <= d <= 5: prob -= 15
        except Exception:
            pass

    plan["probability"] = max(25, min(85, prob))

    # ── Recommendation ────────────────────────────────────────────────────
    trend    = tech.get("trend", "")
    best_rr  = max(plan["tp1_rr"] or 0, plan["tp2_rr"] or 0, plan["tp3_rr"] or 0)
    conf     = plan["confidence"]

    # الاتجاه موافق للصفقة
    trend_ok = (is_call and "BULLISH" in trend) or (not is_call and "BEARISH" in trend)
    # إصلاح 4: MIXED مقبول إذا كانت الثقة عالية جداً (≥75%)
    mixed_ok = "MIXED" in trend and conf >= 75

    # إذا كان هناك Earnings قبل الانتهاء → AVOID دائماً
    earn_risk = r.get("earn_before_expiry", False)

    if earn_risk:
        plan["recommendation"] = "AVOID"
        plan["rec_note"]       = "⚠️ Earnings قبل انتهاء العقد — خطر كبير"
    elif not trade_plan_is_valid(plan, price, is_call):
        plan["recommendation"] = "WAIT"
        plan["rec_note"]       = "خطة الدخول غير متسقة مع السعر الحالي — انتظر"
    elif conf >= MIN_CONF_TO_BUY and best_rr >= MIN_RR_TO_BUY and (trend_ok or mixed_ok):
        plan["recommendation"] = "BUY"
        plan["rec_note"]       = ("الاتجاه والزخم والسيولة مناسبة"
                                   if trend_ok else "اختراق قوي رغم الاتجاه المختلط")
    elif conf >= MIN_CONF_TO_BUY and best_rr >= MIN_RR_TO_BUY:
        plan["recommendation"] = "WAIT"
        plan["rec_note"]       = "زخم جيد — راقب تأكيد الاتجاه"
    elif conf >= 45 and best_rr >= 1.0:
        plan["recommendation"] = "WAIT"
        plan["rec_note"]       = "انتظر تأكيد الاختراق"
    else:
        plan["recommendation"] = "AVOID"
        plan["rec_note"]       = "الشروط غير مكتملة"

    c = plan["confidence"]
    plan["stars"] = 5 if c >= 85 else 4 if c >= 75 else 3 if c >= 65 else 2 if c >= 55 else 1

    return plan


# ════════════════════════════════════════════════════════════════════════
#  PRINT REPORT
# ════════════════════════════════════════════════════════════════════════

def print_trade_report(rank, r, tech, plan):
    ticker   = r["Ticker"]
    direction= r.get("direction", "?")
    prem     = r.get("premium",   0) or 0
    strike   = r.get("strike",    0) or 0
    iv       = r.get("iv",        0) or 0
    dte      = r.get("dte_num",   7) or 7
    price_n  = r.get("price_num", 0) or 0
    oi       = int(r.get("oi",    0) or 0)
    opt_vol  = int(r.get("opt_vol",0) or 0)
    rvol     = r.get("RVOL",      0) or 0
    earn_s   = str(r.get("earnings") or "—")[:10]
    sp_s     = (f"{r['spread_pct']*100:.1f}%"
                if r.get("spread_pct") is not None else "N/A")
    iv_s     = f"{iv*100:.1f}%" if iv else "N/A"
    atr_pct  = r.get("atr_pct",  0) or 0

    # Premarket
    pm_h  = r.get("pm_high")
    pm_l  = r.get("pm_low")
    pm_la = r.get("pm_last")
    pm_v  = r.get("pm_volume")
    # إصلاح 3: عرض واضح عندما لا يوجد بيانات Premarket (السوق مغلق / بعد الساعة 9:30)
    if pm_h and pm_v and pm_v > 0:
        pm_str = f"H:${pm_h:.2f}  L:${pm_l:.2f}  Last:${pm_la:.2f}  Vol:{pm_v:,}"
    elif pm_h:
        pm_str = f"H:${pm_h:.2f}  L:${pm_l:.2f}  Last:${pm_la:.2f}  (No PM Volume — السوق مغلق)"
    else:
        pm_str = "لا تتوفر بيانات Premarket (السوق لم يفتح بعد أو بيانات مأخوذة بعد الإغلاق)"

    conf     = plan["confidence"]
    prob     = plan["probability"]
    stars    = plan["stars"]
    rec      = plan["recommendation"]
    rec_note = plan["rec_note"]
    greeks   = plan.get("greeks") or {}

    star_str  = "★" * stars + "☆" * (5 - stars)
    rec_emoji = {"BUY": "🟢 BUY", "WAIT": "🟡 WAIT", "AVOID": "🔴 AVOID"}.get(rec, "⚪")

    trend      = tech.get("trend", "UNKNOWN")
    rsi_v      = tech.get("rsi",   50)
    ema20_s    = f"${tech['ema20']:.2f}"  if tech.get("ema20")  else "N/A"
    ema50_s    = f"${tech['ema50']:.2f}"  if tech.get("ema50")  else "N/A"
    ema200_s   = f"${tech['ema200']:.2f}" if tech.get("ema200") else "N/A"
    vwap_s     = f"${tech['vwap']:.2f}"   if tech.get("vwap")   else "N/A"
    above_e20  = "✅" if tech.get("above_ema20")  else "❌"
    above_e50  = "✅" if tech.get("above_ema50")  else "❌"
    above_e200 = "✅" if tech.get("above_ema200") else "❌"
    above_vwap = "✅" if tech.get("above_vwap")   else "❌"

    W = 72
    earn_warn = "  ⚠️⚠️  EARNINGS BEFORE EXPIRY — HIGH RISK" if r.get("earn_before_expiry") else ""
    print(f"\n{'═'*W}")
    print(f"  #{rank}  {ticker}  │  {direction}  │  Score: {r.get('Score',0)}{earn_warn}")
    print(f"  {star_str}   Confidence: {conf}%  │  Probability: {prob}%")
    print(f"  {rec_emoji}  —  {rec_note}")

    # ── Contract ─────────────────────────────────────────────────────────
    print(f"{'─'*W}")
    print(f"  CONTRACT")
    print(f"    Stock: ${price_n:.2f}    Strike: ${strike:.2f}    Premium: ${prem:.2f}")
    print(f"    IV:    {iv_s}      DTE: {dte}d      Expiry: {r.get('expiry','N/A')}")
    print(f"    OI:    {oi:,}      OptVol: {opt_vol:,}      Spread: {sp_s}")
    print(f"    RVOL:  {rvol:.1f}x      ATR%: {atr_pct:.1f}%      Earnings: {earn_s}")

    # Greeks المحسوبة محليًا
    if greeks:
        print(f"    Greeks → Delta: {greeks.get('delta',0):.3f}  "
              f"Gamma: {greeks.get('gamma',0):.5f}  "
              f"Theta: {greeks.get('theta',0):.4f}  "
              f"Vega: {greeks.get('vega',0):.4f}")

    # ── Premarket ─────────────────────────────────────────────────────────
    print(f"{'─'*W}")
    print(f"  PREMARKET")
    print(f"    {pm_str}")

    # ── Technical ─────────────────────────────────────────────────────────
    print(f"{'─'*W}")
    print(f"  TECHNICAL ANALYSIS")
    print(f"    Trend:   {trend}")
    print(f"    EMA20:   {ema20_s}  {above_e20}    EMA50:  {ema50_s}  {above_e50}")
    print(f"    EMA200:  {ema200_s}  {above_e200}    VWAP:   {vwap_s}  {above_vwap}")
    print(f"    RSI(14): {rsi_v}")

    # ── Trade Plan ────────────────────────────────────────────────────────
    if plan.get("entry_stock"):
        print(f"{'─'*W}")
        print(f"  TRADE PLAN")
        print(f"    Entry:   ${plan['entry_stock']:.2f}    ({plan['entry_note']})")
        print(f"    Stop:    ${plan['stop_stock']:.2f}    → Option ≈ ${plan['stop_option']:.2f}")
        print(f"    Risk:    ${plan['risk_per_contract']:.0f} per contract")
        print(f"{'─'*W}")
        print(f"  TARGETS                         Option Value      Risk : Reward")
        if plan.get("tp1_stock"):
            print(f"    TP1:    ${plan['tp1_stock']:.2f}              "
                  f"${plan['tp1_option']:.2f}            1 : {plan['tp1_rr']:.1f}")
        if plan.get("tp2_stock"):
            print(f"    TP2:    ${plan['tp2_stock']:.2f}              "
                  f"${plan['tp2_option']:.2f}            1 : {plan['tp2_rr']:.1f}")
        if plan.get("tp3_stock"):
            print(f"    TP3:    ${plan['tp3_stock']:.2f}              "
                  f"${plan['tp3_option']:.2f}            1 : {plan['tp3_rr']:.1f}")

    # ── Scenarios ─────────────────────────────────────────────────────────
    if strike and prem and iv and price_n:
        is_call = "CALL" in str(direction).upper()
        opt_t   = "call" if is_call else "put"
        T_full  = max(dte / 365, 1 / 365)
        T_half  = max((dte / 2) / 365, 1 / 365)
        moves   = [-0.15, -0.10, -0.05, 0.0, +0.05, +0.10, +0.15, +0.20]

        print(f"{'─'*W}")
        print(f"  SCENARIOS  (Black-Scholes)")
        print(f"    {'Move':>6}  {'Stock':>8}  {'Now':>7}  {'Half DTE':>9}  {'Expiry':>8}  P&L Now")
        for m in moves:
            ns    = round(price_n * (1 + m), 2)
            vnow  = round(bs_price(ns, strike, T_full, RISK_FREE_RATE, iv, opt_t), 2)
            vhalf = round(bs_price(ns, strike, T_half, RISK_FREE_RATE, iv, opt_t), 2)
            vexp  = round(max(ns-strike,0) if is_call else max(strike-ns,0), 2)
            pnl   = round((vnow - prem) * 100, 0)
            tag   = " ◄ NOW" if m == 0 else ""
            pnl_s = f"+${pnl:.0f}" if pnl >= 0 else f"-${abs(pnl):.0f}"
            print(f"    {m*100:>+5.0f}%   ${ns:>7.2f}  ${vnow:>5.2f}  "
                  f"${vhalf:>8.2f}  ${vexp:>7.2f}  {pnl_s:>8}{tag}")

        # نقاط مرجعية
        fine_moves = [m/100 for m in range(-20, 25, 1)]
        all_sc = []
        for m in fine_moves:
            ns   = round(price_n * (1 + m), 2)
            vnow = bs_price(ns, strike, T_full, RISK_FREE_RATE, iv, opt_t)
            vexp = max(ns-strike,0) if is_call else max(strike-ns,0)
            all_sc.append({"move": m, "stock": ns,
                           "pnl_now": (vnow - prem) * 100,
                           "pnl_exp": (vexp - prem) * 100})

        be_now = next((s for s in all_sc if s["pnl_now"] >= 0), None)
        be_exp = next((s for s in all_sc if s["pnl_exp"] >= 0), None)
        x2     = next((s for s in all_sc if s["pnl_now"] >= prem * 100), None)

        print(f"    {'─'*60}")
        if be_now:
            print(f"    بريكإيفن الآن:         ${be_now['stock']:.2f} ({be_now['move']*100:+.0f}%)")
        if be_exp:
            print(f"    بريكإيفن عند الانتهاء: ${be_exp['stock']:.2f} ({be_exp['move']*100:+.0f}%)")
        if x2:
            print(f"    تضاعف العقد (×2):      ${x2['stock']:.2f} ({x2['move']*100:+.0f}%)")

    print(f"{'─'*W}")
    print(f"  Factors: {r.get('Notes','')}")
    print(f"{'═'*W}")


# ════════════════════════════════════════════════════════════════════════
#  CANDIDATE SOURCES — Finviz + قائمة يدوية
# ════════════════════════════════════════════════════════════════════════

SAVE_COLS = [
    "Ticker", "Price", "premium", "strike", "iv", "oi", "opt_vol", "spread_pct",
    "atr_pct", "gap_pct", "RVOL", "pm_high", "pm_low", "pm_volume",
    "expiry", "direction", "earnings", "float_shares", "avg_vol",
    "entry_stock", "stop_stock", "tp1_stock", "tp2_stock", "tp3_stock",
    "Score", "recommendation", "confidence", "Notes",
]


def get_screener_source():
    return os.environ.get("SCREENER_SOURCE", DEFAULT_SCREENER_SOURCE).strip().lower()


def load_manual_tickers(path=MANUAL_TICKERS_FILE):
    """قراءة manual_tickers.txt — سهم واحد في كل سطر، # للتعليق."""
    if not os.path.exists(path):
        return []
    tickers = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tickers.append(fix_ticker(line.split()[0].upper()))
    return list(dict.fromkeys(tickers))


def fetch_finviz_candidates():
    ft = Technical()
    ft.set_filter(filters_dict=FINVIZ_FILTERS)
    df = ft.screener_view()
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df["Ticker"]     = df["Ticker"].apply(fix_ticker)
    df["Price_Num"]  = df["Price"].apply(parse_num)
    df["Today_Vol"]  = df["Volume"].apply(parse_num)
    df["ATR_Pct"]    = df.apply(
        lambda row: round(parse_num(row["ATR"]) / row["Price_Num"] * 100, 2)
                    if row["Price_Num"] > 0 else 0.0, axis=1)
    df["Gap_Num"]    = df["Gap"].apply(parse_num) if "Gap" in df.columns else 0.0
    df["_FastScore"] = df["ATR_Pct"] * (df["Today_Vol"] / 1e6)
    df["_source"]    = "finviz"
    return (df.sort_values("_FastScore", ascending=False)
              .reset_index(drop=True))


def fetch_manual_candidates(tickers):
    rows = []
    for ticker in tickers:
        try:
            yt = yf.Ticker(ticker, session=_YF_SESSION)
            info = yt.info or {}
            price_num = float(
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
                or 0
            )
            if price_num <= 0:
                print(f"  ⚠ manual {ticker}: no price")
                continue

            hist = yt.history(period="60d", auto_adjust=True)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            tech = compute_technicals(hist, price_num) if hist is not None and len(hist) >= 5 else {}

            today_vol = int(info.get("volume") or info.get("regularMarketVolume") or 0)
            atr = float(tech.get("atr14") or price_num * 0.025)
            atr_pct = round(atr / price_num * 100, 2) if price_num > 0 else 0.0
            prev_close = float(info.get("previousClose") or price_num)
            gap_pct = round((price_num - prev_close) / prev_close * 100, 4) if prev_close > 0 else 0.0

            rows.append({
                "Ticker":     ticker,
                "Price":      f"${price_num:.2f}",
                "Price_Num":  price_num,
                "Today_Vol":  today_vol,
                "ATR_Pct":    atr_pct,
                "Gap_Num":    gap_pct,
                "RSI":        tech.get("rsi", 50),
                "ATR":        atr,
                "_FastScore": atr_pct * (today_vol / 1e6),
                "_source":    "manual",
            })
        except Exception as e:
            print(f"  ⚠ manual {ticker}: {e}")
    return pd.DataFrame(rows)


def fetch_candidates(source=None):
    """
    جلب المرشّحين من Finviz و/أو manual_tickers.txt.
    both: Finviz أولاً + يدوي، مع fallback لليدوي إذا Finviz فشل.
    """
    source = (source or get_screener_source()).lower()
    print(f"  Source mode: {source}")

    finviz_df = pd.DataFrame()
    manual_df = pd.DataFrame()
    manual_tickers = load_manual_tickers()

    if source in ("finviz", "both"):
        print("⏳ Stage 1a: Finviz Technical screener...")
        try:
            finviz_df = fetch_finviz_candidates()
            print(f"  ✓ Finviz: {len(finviz_df)} stocks")
        except Exception as e:
            print(f"  ✗ Finviz error: {e}")

    if source in ("manual", "both"):
        if manual_tickers:
            print(f"⏳ Stage 1b: Manual watchlist ({len(manual_tickers)} tickers)...")
            manual_df = fetch_manual_candidates(manual_tickers)
            print(f"  ✓ Manual: {len(manual_df)} stocks")
        elif source == "manual":
            print("  ✗ manual_tickers.txt فارغ أو غير موجود")

    parts = [df for df in (finviz_df, manual_df) if not df.empty]
    if not parts:
        if source in ("finviz", "both") and manual_tickers:
            print("  ↪ Finviz فارغ — fallback إلى القائمة اليدوية")
            return fetch_candidates("manual")
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Ticker"], keep="first")
    if "_FastScore" in combined.columns:
        combined = combined.sort_values("_FastScore", ascending=False, na_position="last")
    candidates = combined.head(MAX_STOCKS_DEEP).reset_index(drop=True)

    src_counts = candidates.get("_source", pd.Series(dtype=str)).value_counts().to_dict()
    print(f"  → {len(candidates)} candidates ({src_counts})\n")
    return candidates


def process_candidates(candidates_df, show_progress=True):
    """Stage 2: Yahoo options + scoring + trade plan لكل مرشّح."""
    rows = []
    if show_progress:
        print("⏳ Stage 2: Options + Premarket + Technical data...")
        print(f"   {'Ticker':<8} {'Price':>8}  {'Strike':>8}  {'Premium':>8}  "
              f"{'OI':>7}  {'Spread':>7}  {'PM_High':>9}  Score")
        print(f"   {'─'*80}")

    for _, s in candidates_df.iterrows():
        ticker    = s["Ticker"]
        price_num = float(s.get("Price_Num") or parse_price_num(s.get("Price")) or 0)
        if price_num <= 0:
            continue

        opts = fetch_options_data(ticker, price_num)
        hist = opts.get("hist")
        tech_early = compute_technicals(hist, price_num) if hist is not None and len(hist) >= 5 else {}
        today_vol = float(s.get("Today_Vol", 0) or 0)
        if today_vol <= 0:
            today_vol = float(opts.get("today_vol_yf") or 0)

        row = {
            "Ticker":       ticker,
            "Price":        s.get("Price", f"${price_num:.2f}"),
            "price_num":    price_num,
            "rsi":          tech_early.get("rsi", s.get("RSI", 50)),
            "atr":          s.get("ATR", 0),
            "atr_pct":      s.get("ATR_Pct", 0),
            "gap_pct":      s.get("Gap_Num", 0),
            "today_vol":    today_vol,
            "avg_vol":      opts["avg_vol"],
            "float_shares": opts["float_shares"],
            "earnings":     opts["earnings"],
            "premium":      opts["premium"],
            "iv":           opts["iv"],
            "oi":           opts["oi"],
            "opt_vol":      opts["opt_vol"],
            "spread_pct":   opts["spread_pct"],
            "expiry":       opts["expiry"],
            "has_weekly":   opts["has_weekly"],
            "direction":    opts["direction"],
            "strike":       opts["strike"],
            "dte_num":      opts["dte_num"],
            "_hist":        hist,
            "pm_high":      opts["pm_high"],
            "pm_low":       opts["pm_low"],
            "pm_last":      opts["pm_last"],
            "pm_volume":    opts["pm_volume"],
            "_error":       opts["error"],
            "_source":      s.get("_source", ""),
        }

        score, notes = score_stock(row)
        row["Score"] = score
        row["Notes"] = " | ".join(notes)

        tech = compute_technicals(hist, price_num)
        try:
            plan = compute_trade_plan(row, tech)
        except Exception as e:
            row["_error"] = (row.get("_error") or "") + f" plan:{e}"
            plan = {"recommendation": "AVOID", "confidence": 0,
                    "entry_stock": None, "stop_stock": None,
                    "tp1_stock": None, "tp2_stock": None, "tp3_stock": None}
        row["recommendation"] = plan.get("recommendation", "AVOID")
        row["confidence"]     = plan.get("confidence", 0)
        row["entry_stock"]    = plan.get("entry_stock")
        row["stop_stock"]     = plan.get("stop_stock")
        row["tp1_stock"]      = plan.get("tp1_stock")
        row["tp2_stock"]      = plan.get("tp2_stock")
        row["tp3_stock"]      = plan.get("tp3_stock")

        if show_progress:
            stk_s   = f"${row['strike']:.2f}" if row["strike"]  else "   N/A"
            prem_s  = f"${row['premium']:.2f}" if row["premium"] is not None else "   N/A"
            oi_s    = f"{row['oi']:,}"          if row["oi"]     is not None else "  N/A"
            sp_s    = f"{row['spread_pct']*100:.1f}%" if row["spread_pct"] is not None else "  N/A"
            pm_s    = f"${row['pm_high']:.2f}" if row["pm_high"] else "   N/A"
            err_s   = f"  ⚠ {row['_error']}"  if row["_error"] else ""
            print(f"   {ticker:<8} {str(row.get('Price',''))[:8]:>8}  {stk_s:>8}  "
                  f"{prem_s:>8}  {oi_s:>7}  {sp_s:>7}  {pm_s:>9}  {score:>4}{err_s}")

        rows.append(row)
        time.sleep(DELAY_BETWEEN)

    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
              .sort_values("Score", ascending=False)
              .reset_index(drop=True))


def save_screen_results(result_df, path="options_v3_results.csv"):
    save_cols = [c for c in SAVE_COLS if c in result_df.columns]
    strict = filter_results_df(result_df)
    watchlist = dashboard_fallback_df(result_df, max_rows=15)

    parts = []
    if not strict.empty:
        parts.append(strict)
    if not watchlist.empty:
        have = set(strict["Ticker"]) if not strict.empty else set()
        extra = watchlist[~watchlist["Ticker"].isin(have)]
        if not extra.empty:
            parts.append(extra)

    if parts:
        combined = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["Ticker"], keep="first")
    else:
        combined = pd.DataFrame()

    used_fallback = not watchlist.empty and (
        strict.empty or len(combined) > len(strict)
    )
    if used_fallback:
        print(f"  ↪ dashboard: {len(strict)} strict + {len(combined) - len(strict)} WAIT/BUY watchlist")

    ok = save_results_csv(combined[save_cols] if not combined.empty else combined, path)
    return ok, combined, len(result_df), used_fallback


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    start_time = datetime.now()
    print(f"\n{'═'*74}")
    print(f"  🔍 OPTIONS TRADING ASSISTANT v3  (100% Free Data)")
    print(f"  {start_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Premium ${TARGET_PREM_MIN}–${TARGET_PREM_MAX}  │  Min OI: {MIN_OI}  │  DTE ~{DTE_TARGET}d")
    print(f"  Source: {get_screener_source()} (Finviz + manual_tickers.txt)")
    print(f"  Premarket · EMA · VWAP · RSI · ATR · Black-Scholes Greeks")
    print(f"{'═'*74}\n")

    candidates = fetch_candidates()
    if candidates.empty:
        print("  No candidates from Finviz or manual list."); return

    result_df = process_candidates(candidates)
    if result_df.empty:
        print("  No option data returned."); return

    elapsed = (datetime.now() - start_time).seconds
    print(f"\n  ✓ Done in {elapsed}s\n")

    in_range = filter_results_df(result_df)

    # ── Summary Table ─────────────────────────────────────────────────────
    print("═" * 95)
    print(f"  💰 {len(in_range)} matches  │  Premium ${TARGET_PREM_MIN}–${TARGET_PREM_MAX}  │  OI ≥ {MIN_OI}  │  Price ≤ ${MAX_STOCK_PRICE:.0f}")
    print("═" * 95)

    if not in_range.empty:
        disp = in_range[["Ticker","Price","premium","iv","oi","opt_vol",
                          "spread_pct","atr_pct","RVOL","pm_high","direction","Score"]].copy()
        disp.columns = ["Ticker","Price","Premium","IV","OI","OptVol",
                         "Spread%","ATR%","RVOL","PM_High","Direction","Score"]
        disp["Premium"]  = disp["Premium"].apply(lambda x: f"${x:.2f}"       if pd.notna(x)           else "N/A")
        disp["IV"]       = disp["IV"].apply(lambda x: f"{x*100:.1f}%"        if pd.notna(x) and x > 0 else "N/A")
        disp["Spread%"]  = disp["Spread%"].apply(lambda x: f"{x*100:.1f}%"   if pd.notna(x)           else "N/A")
        disp["OI"]       = disp["OI"].apply(lambda x: f"{int(x):,}"          if pd.notna(x)           else "N/A")
        disp["OptVol"]   = disp["OptVol"].apply(lambda x: f"{int(x):,}"      if pd.notna(x)           else "N/A")
        disp["ATR%"]     = disp["ATR%"].apply(lambda x: f"{x:.1f}%"          if pd.notna(x)           else "N/A")
        disp["RVOL"]     = disp["RVOL"].apply(lambda x: f"{x:.1f}x"          if pd.notna(x)           else "N/A")
        disp["PM_High"]  = disp["PM_High"].apply(lambda x: f"${x:.2f}"       if pd.notna(x)           else "N/A")
        print(disp.head(30).to_string(index=False))

    # ── Full Trade Reports ────────────────────────────────────────────────
    top8 = in_range.head(8)
    if not top8.empty:
        print(f"\n\n🎯 FULL TRADE REPORTS — Top {len(top8)} Picks")
        for rank, (_, r) in enumerate(top8.iterrows(), 1):
            hist = r.get("_hist")
            tech = compute_technicals(hist, r.get("price_num", 0))
            plan = compute_trade_plan(dict(r), tech)
            print_trade_report(rank, dict(r), tech, plan)

    # ── Legend ────────────────────────────────────────────────────────────
    print("\n\n📊 SCORE SYSTEM")
    print("  RVOL>2    +3  │  OI>2K     +3  │  OptVol>1K  +3  │  Spread<3%  +3")
    print("  Weekly    +2  │  Prem✓     +2  │  ATR%>5%    +3  │  NoEarnings +2")
    print("  PM_Vol>500K +2│  Float<50M +2  │  Gap>4%     +2")
    print("\n  ⚠️  كل البيانات من Yahoo Finance / Finviz — مجانية 100%")
    print("  ⚠️  Greeks تُحسب محليًا بـ Black-Scholes — لا تحتاج API مدفوع")
    print("  ⚠️  للتعليم فقط — ليس نصيحة مالية — تحقق من بروكرك قبل الدخول")
    print("═" * 95 + "\n")

    # ── Save CSV ──────────────────────────────────────────────────────────
    ok, filtered_df, total, _fb = save_screen_results(result_df)
    if ok:
        print(f"  (فلترة: {total} → {len(filtered_df)} بعد premium/OI/spread/price)")
        print("✅ Saved: options_v3_results.csv\n")
    else:
        print(f"  (فلترة: {total} → {len(filtered_df)} — لم يُحفظ CSV فاضي)\n")


if __name__ == "__main__":
    main()
