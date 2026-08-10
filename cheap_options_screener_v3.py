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
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from finvizfinance.screener.technical import Technical

MARKET_TZ = ZoneInfo("America/New_York")

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
TARGET_PREM_MIN  = 1.00    # أقل سعر عقد للوضع الأساسي ($) — تحت دولار يضر دقة العقد
TARGET_PREM_MAX  = 3.00    # أعلى سعر عقد مقبول ($)
MIN_OI           = 500     # أقل Open Interest
MAX_SPREAD_PCT = 0.10    # أقصى spread مقبول (10% كـ decimal) — سبريد ضيق
PM_STALE_PCT   = 0.05    # تجاهل pm_high/pm_low إذا ابتعدت عن السعر >5%
ENTRY_MAX_DRIFT = 0.05   # Entry يجب أن يكون ضمن ±5% من السعر الحالي
MAX_STOCK_PRICE = 500.0  # رفض أسعار Yahoo الشاذة (مثل SNDK $1183)
MIN_IV_DISPLAY  = 0.01   # IV أقل من 1% تُعامل كبيانات ناقصة
MAX_BS_MID_DIVERGENCE = 0.50  # |BS−mid|/mid فوق 50% → سعر عقد غير موثوق
MAX_SPREAD_FOR_MARKET = 0.25  # سبريد أوسع من 25% يقلّل ثقة سعر السوق
DTE_TARGET       = 5       # تقريباً منتصف الأسبوع → أقرب جمعة
DTE_WINDOW       = 4
MAX_DTE          = 10      # رفض أي عقد أبعد من 10 أيام
PREFER_0DTE      = False    # ثيتا قاتلة — لا نفضّل نفس اليوم
PREFER_FRIDAY    = True     # أولوية: انتهاء يوم الجمعة (weekly)
MAX_STOCKS_DEEP  = 40      # كم سهم ندرس بعمق
DELAY_BETWEEN    = 0.30    # ثواني بين كل طلب yfinance
MANUAL_TICKERS_FILE = "manual_tickers.txt"
# finviz | manual | both  (both = Finviz + قائمة يدوية، مع fallback)
DEFAULT_SCREENER_SOURCE = "both"

RISK_FREE_RATE   = 0.053   # معدل الفائدة لـ Black-Scholes
MIN_RR_TO_BUY    = 1.3     # أقل R:R لإعطاء BUY (من السعر الحي)
MIN_CONF_TO_BUY  = 55      # أقل Confidence% لإعطاء BUY
MAX_ENTRY_CHASE_PCT = 0.008  # إذا تجاوز السعر Entry بأكثر من 0.8% → مطاردة (لا BUY)

# قواعد التشغيل المتفق عليها (2026-07-31)
REQUIRE_SPY_ALIGNMENT = True   # لا PUT والسوق صاعد / لا CALL والسوق هابط
EXEC_AFTER_OPEN_MIN   = 15     # دقائق بعد الافتتاح قبل اعتماد التنفيذ (4:45 السعودية)

# صناديق عكسية — CALL عليها = رهان هبوط السوق (والعكس للـ PUT)
INVERSE_ETFS = frozenset({
    "SQQQ", "SOXS", "SPXU", "SDOW", "FAZ", "TECS",
    "UVIX", "VXX", "UVXY", "SVIX", "VIXY",
})

FINVIZ_FILTERS = {
    "Option/Short":    "Optionable",
    "Average Volume":  "Over 1M",        # حجم يومي أعلى = spread أضيق
    "Country":         "USA",
    "Price":           "Over $5",        # تجنب penny stocks
    "Relative Volume": "Over 2",         # نشاط عالي اليوم (RVOL > 2x)
}

SPY_TICKER = "SPY"
PRIORITY_TICKERS = (SPY_TICKER,)

# بروفايلات tickers خاصة — premium أعلى وحدود مختلفة
TICKER_PROFILES = {
    SPY_TICKER: {
        "label":           "SPY — S&P 500 ETF",
        "prem_min":        1.0,
        "prem_max":        5.0,      # OTM — مو ATM ($6+)
        "target_prem":     2.50,     # الهدف: ~$2.50 للعقد
        "use_otm_strike":  True,     # strike OTM أرخص من ATM
        "max_stock_price": 800.0,
        "max_spread_pct":  0.10,
        "min_oi":          500,
        "max_dte":         10,
        "prefer_0dte":     False,
        "prefer_friday":   True,
    },
}


def ticker_profile(ticker):
    return TICKER_PROFILES.get(str(ticker or "").upper(), {})


def profile_limit(ticker, key, default):
    return ticker_profile(ticker).get(key, default)


def is_spy_ticker(ticker):
    return str(ticker or "").upper() == SPY_TICKER


def classify_spy_regime(tech, price):
    """
    بوصلة السوق من SPY:
      BULL  = فوق VWAP وEMA20 و/أو ترند صاعد
      BEAR  = تحت VWAP وEMA20 و/أو ترند هابط
      NEUTRAL = مختلط
    """
    if not tech or not price:
        return "NEUTRAL"
    trend = str(tech.get("trend") or "")
    above_vwap = bool(tech.get("above_vwap"))
    above_e20 = bool(tech.get("above_ema20"))
    if "BULLISH" in trend and (above_vwap or above_e20):
        return "BULL"
    if "BEARISH" in trend and (not above_vwap or not above_e20):
        return "BEAR"
    if above_vwap and above_e20:
        return "BULL"
    if (not above_vwap) and (not above_e20):
        return "BEAR"
    return "NEUTRAL"


def is_inverse_ticker(ticker):
    return str(ticker or "").upper().strip() in INVERSE_ETFS


def parse_dte(row_or_val, default=7):
    """اقرأ DTE بأمان — 0 يبقى 0 (لا يُحوَّل إلى default بسبب falsy)."""
    if isinstance(row_or_val, dict):
        val = row_or_val.get("dte_num", default)
    else:
        val = row_or_val
    if val is None or val == "":
        return int(default)
    try:
        return int(val)
    except (TypeError, ValueError):
        return int(default)


def spy_alignment_ok(is_call, spy_regime, ticker=None):
    """لا نعكس السوق: PUT مرفوض إذا SPY صاعد، CALL مرفوض إذا SPY هابط.
    الصناديق العكسية تُعامل باتجاه معكوس (CALL على SQQQ = رهان هبوط)."""
    if not REQUIRE_SPY_ALIGNMENT:
        return True, ""
    regime = str(spy_regime or "NEUTRAL").upper()
    inv = is_inverse_ticker(ticker)
    if inv:
        if regime == "BULL" and is_call:
            return False, "CALL على صندوق عكسي مرفوض — SPY صاعد"
        if regime == "BEAR" and not is_call:
            return False, "PUT على صندوق عكسي مرفوض — SPY هابط"
        return True, ""
    if regime == "BULL" and not is_call:
        return False, "PUT مرفوض — SPY صاعد (فلتر اتجاه)"
    if regime == "BEAR" and is_call:
        return False, "CALL مرفوض — SPY هابط (فلتر اتجاه)"
    return True, ""


def inverse_recommendation_ceiling(is_call, spy_regime, ticker=None):
    """
    سقف توصية للصناديق العكسية (SQQQ…).
    أسبوع 3–5 أغسطس: SQQQ خسر 3/3 بعد ما مرّ كفرصة —
    لذلك: مخالفة اتجاه → AVOID، وباقي الحالات لا BUY تلقائي.
    يرجع (ceiling, note) حيث ceiling ∈ BUY|WAIT|AVOID.
    """
    if not is_inverse_ticker(ticker):
        return "BUY", ""
    regime = str(spy_regime or "NEUTRAL").upper()
    if regime == "BULL" and is_call:
        return "AVOID", "CALL على صندوق عكسي مرفوض — SPY صاعد (AVOID)"
    if regime == "BEAR" and not is_call:
        return "AVOID", "PUT على صندوق عكسي مرفوض — SPY هابط (AVOID)"
    if regime == "NEUTRAL":
        return "WAIT", "صندوق عكسي + SPY محايد — لا BUY تلقائي (رافعة/عكس)"
    # محاذاة ظاهرة (مثل PUT على SQQQ والسوق صاعد) — ما زالت رافعة 3x
    return "WAIT", "صندوق عكسي — لا BUY تلقائي حتى مع محاذاة SPY (رافعة)"


def fetch_spy_regime():
    """جلب نظام SPY مرة واحدة قبل مسح الأسهم."""
    try:
        import yfinance as yf
        hist = yf.Ticker(SPY_TICKER).history(period="60d")
        if hist is None or len(hist) < 5:
            return "NEUTRAL", None, None
        price = float(hist["Close"].iloc[-1])
        tech = compute_technicals(hist, price)
        return classify_spy_regime(tech, price), tech, price
    except Exception:
        return "NEUTRAL", None, None


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


def assess_option_price_quality(
    *,
    premium,
    bs_fair,
    iv_raw,
    bid=None,
    ask=None,
    last=None,
    spread_pct=None,
    synthetic_quote=False,
):
    """
    يقيّم موثوقية سعر العقد المعروض (عادة mid من Bid/Ask).
    يرجع: quality ∈ market|estimate|unreliable ، ملاحظة، هل نعرض سيناريوهات BS.
    """
    try:
        mid = float(premium or 0)
    except (TypeError, ValueError):
        mid = 0.0
    try:
        fair = float(bs_fair or 0)
    except (TypeError, ValueError):
        fair = 0.0
    try:
        iv = float(iv_raw or 0)
    except (TypeError, ValueError):
        iv = 0.0
    try:
        sp = float(spread_pct) if spread_pct is not None else None
    except (TypeError, ValueError):
        sp = None

    iv_missing = iv < MIN_IV_DISPLAY
    has_ba = False
    try:
        has_ba = float(bid or 0) > 0 and float(ask or 0) > 0
    except (TypeError, ValueError):
        has_ba = False
    # premium من السلسلة = mid سوق ما لم يكن quote ملفّق من Last فقط
    market_mid = mid > 0 and not bool(synthetic_quote)

    diverg = None
    if mid > 0 and fair >= 0:
        diverg = abs(fair - mid) / mid

    # غير موثوق: لا mid، أو quote ملفّق، أو BS≈0 والعقد له سعر، أو انحراف كبير مع IV ناقص/سيء
    if mid <= 0:
        return {
            "premium_quality": "unreliable",
            "premium_quality_note": "لا يوجد سعر عقد صالح من Yahoo",
            "bs_fair_price": round(fair, 2) if fair else None,
            "show_bs_scenarios": False,
            "premium_ok_for_buy": False,
        }
    if synthetic_quote:
        return {
            "premium_quality": "estimate",
            "premium_quality_note": "Bid/Ask صفر — تقدير من Last فقط",
            "bs_fair_price": round(fair, 2),
            "show_bs_scenarios": (not iv_missing) and (diverg is None or diverg <= MAX_BS_MID_DIVERGENCE),
            "premium_ok_for_buy": False,
        }
    if fair <= 0.02 and mid >= 0.40:
        return {
            "premium_quality": "unreliable",
            "premium_quality_note": (
                f"تسعير BS≈${fair:.2f} يناقض mid السوق ${mid:.2f} — IV/بيانات Yahoo ضعيفة"
            ),
            "bs_fair_price": round(fair, 2),
            "show_bs_scenarios": False,
            "premium_ok_for_buy": False,
        }
    if iv_missing and diverg is not None and diverg > 0.35:
        return {
            "premium_quality": "unreliable",
            "premium_quality_note": (
                f"IV ناقص وانحراف BS عن السوق {diverg*100:.0f}% — لا تعتمد سعر العقد"
            ),
            "bs_fair_price": round(fair, 2),
            "show_bs_scenarios": False,
            "premium_ok_for_buy": False,
        }
    if diverg is not None and diverg > MAX_BS_MID_DIVERGENCE:
        return {
            "premium_quality": "unreliable",
            "premium_quality_note": (
                f"انحراف BS عن mid السوق {diverg*100:.0f}% (> {MAX_BS_MID_DIVERGENCE*100:.0f}%)"
            ),
            "bs_fair_price": round(fair, 2),
            "show_bs_scenarios": False,
            "premium_ok_for_buy": False,
        }
    if sp is not None and sp > MAX_SPREAD_FOR_MARKET:
        return {
            "premium_quality": "estimate",
            "premium_quality_note": f"سبريد واسع {sp*100:.0f}% — mid تقريبي",
            "bs_fair_price": round(fair, 2),
            "show_bs_scenarios": not iv_missing,
            "premium_ok_for_buy": False,
        }
    # mid سوق موثوق: Bid/Ask أو premium من السلسلة (غير synthetic) + IV سليم + انحراف مقبول
    if market_mid and not iv_missing and (diverg is None or diverg <= 0.35):
        src = "Bid/Ask" if has_ba else "سلسلة Yahoo"
        return {
            "premium_quality": "market",
            "premium_quality_note": f"mid سوق ({src}) · BS≈${fair:.2f}",
            "bs_fair_price": round(fair, 2),
            "show_bs_scenarios": True,
            "premium_ok_for_buy": True,
        }
    # mid سوق حقيقي مع IV ضعيف — اعتمد سعر السوق للـ BUY، اخفِ سيناريوهات BS
    if market_mid and iv_missing and (diverg is None or diverg <= 0.35):
        return {
            "premium_quality": "estimate",
            "premium_quality_note": "mid سوق مع IV ناقص — اعتمد السوق لا سيناريوهات BS",
            "bs_fair_price": round(fair, 2),
            "show_bs_scenarios": False,
            "premium_ok_for_buy": True,
        }
    if market_mid:
        return {
            "premium_quality": "estimate",
            "premium_quality_note": "mid سوق مع انحراف متوسط عن BS",
            "bs_fair_price": round(fair, 2),
            "show_bs_scenarios": False,
            "premium_ok_for_buy": True,
        }
    return {
        "premium_quality": "estimate",
        "premium_quality_note": "سعر عقد تقديري — تحقق من البروكر قبل الدخول",
        "bs_fair_price": round(fair, 2) if fair else None,
        "show_bs_scenarios": False,
        "premium_ok_for_buy": False,
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


def effective_oi(oi, opt_vol, min_oi=None):
    """Yahoo أحياناً يرجّع OI=0 — استخدم opt_vol كبديل مؤقت."""
    min_oi = MIN_OI if min_oi is None else min_oi
    oi = _safe_int(oi)
    ov = _safe_int(opt_vol)
    if oi >= min_oi:
        return oi
    if oi == 0 and ov >= min_oi:
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
        if entry < price * (1 - ENTRY_MAX_DRIFT * 0.5):
            return False
        return tp1 > entry > stop
    if entry < price * (1 - ENTRY_MAX_DRIFT):
        return False
    if entry > price * (1 + ENTRY_MAX_DRIFT * 0.5):
        return False
    return tp1 < entry < stop


def _fnum(val):
    try:
        v = float(val)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def compute_entry_level(price, is_call, pm_high, pm_low, yest_h, yest_l, atr, tech=None):
    """
    Entry على هيكل السعر:
    CALL → أقرب دعم تحت السعر (سوينج / VWAP / EMA20)
    PUT  → أقرب مقاومة فوق السعر
    """
    tech = tech or {}
    atr = float(atr or price * 0.025)
    price = float(price)

    if is_call:
        supports = []
        for v in (
            tech.get("support1"), tech.get("support2"),
            tech.get("vwap"), tech.get("ema20"),
            yest_l, pm_low,
        ):
            fv = _fnum(v)
            if fv and fv < price:
                supports.append(fv)
        if supports:
            # أقرب دعم من تحت (الأعلى تحت السعر)
            base = max(supports)
            dist = price - base
            if dist <= atr * 0.35:
                entry = round(price, 2)
                note = f"عند السعر — قريب من دعم ${base:.2f}"
            elif dist <= atr * 2.5:
                entry = round(base + atr * 0.05, 2)
                note = f"عند دعم ${base:.2f}"
            else:
                entry = round(price, 2)
                note = f"دخول حالي — أقرب دعم بعيد ${base:.2f}"
        else:
            entry = round(price, 2)
            note = "دخول عند السعر — لا دعم واضح قريب"
        max_e = round(price * (1 + ENTRY_MAX_DRIFT), 2)
        min_e = round(price * (1 - 0.02), 2)
        entry = min(max(entry, min_e), max_e)
    else:
        resists = []
        for v in (
            tech.get("resist1"), tech.get("resist2"),
            tech.get("vwap"), tech.get("ema20"),
            yest_h, pm_high,
        ):
            fv = _fnum(v)
            if fv and fv > price:
                resists.append(fv)
        if resists:
            base = min(resists)
            dist = base - price
            if dist <= atr * 0.35:
                entry = round(price, 2)
                note = f"عند السعر — قريب من مقاومة ${base:.2f}"
            elif dist <= atr * 2.5:
                entry = round(base - atr * 0.05, 2)
                note = f"عند مقاومة ${base:.2f}"
            else:
                entry = round(price, 2)
                note = f"دخول حالي — أقرب مقاومة بعيدة ${base:.2f}"
        else:
            entry = round(price, 2)
            note = "دخول عند السعر — لا مقاومة واضحة قريبة"
        min_e = round(price * (1 - ENTRY_MAX_DRIFT), 2)
        max_e = round(price * (1 + 0.02), 2)
        entry = min(max(entry, min_e), max_e)

    return round(entry, 2), note


def compute_targets(entry, is_call, atr, tech):
    """TP1/2/3 من دعم/مقاومة حقيقية — CALL فوق / PUT تحت."""
    atr = float(atr or 0)
    entry = float(entry)
    tech = tech or {}

    if is_call:
        r1 = _fnum(tech.get("resist1"))
        r2 = _fnum(tech.get("resist2"))
        r3 = _fnum(tech.get("resist3"))
        tp1 = r1 if r1 and r1 > entry else round(entry + atr * 1.0, 2)
        tp2 = r2 if r2 and r2 > tp1 else round(tp1 + atr * 0.8, 2)
        tp3 = r3 if r3 and r3 > tp2 else round(tp2 + atr * 1.0, 2)
        # فرض ترتيب صحيح
        tp1 = max(tp1, round(entry + atr * 0.35, 2))
        tp2 = max(tp2, round(tp1 + atr * 0.35, 2))
        tp3 = max(tp3, round(tp2 + atr * 0.45, 2))
    else:
        s1 = _fnum(tech.get("support1"))
        s2 = _fnum(tech.get("support2"))
        s3 = _fnum(tech.get("support3"))
        tp1 = s1 if s1 and s1 < entry else round(entry - atr * 1.0, 2)
        tp2 = s2 if s2 and s2 < tp1 else round(tp1 - atr * 0.8, 2)
        tp3 = s3 if s3 and s3 < tp2 else round(tp2 - atr * 1.0, 2)
        tp1 = min(tp1, round(entry - atr * 0.35, 2))
        tp2 = min(tp2, round(tp1 - atr * 0.35, 2))
        tp3 = min(tp3, round(tp2 - atr * 0.45, 2))
        # لا تسمح بأهداف ≤ 0
        floor = max(entry * 0.05, 0.05)
        tp1 = max(tp1, floor)
        tp2 = max(min(tp2, tp1 - atr * 0.2), floor * 0.8)
        tp3 = max(min(tp3, tp2 - atr * 0.2), floor * 0.5)
        if not (tp3 < tp2 < tp1 < entry):
            tp1 = round(entry - atr * 1.0, 2)
            tp2 = round(entry - atr * 1.8, 2)
            tp3 = round(entry - atr * 2.6, 2)
            tp1 = max(tp1, floor)
            tp2 = max(tp2, floor * 0.8)
            tp3 = max(tp3, floor * 0.5)

    return round(tp1, 2), round(tp2, 2), round(tp3, 2)


def compute_structure_stop(entry, is_call, atr, tech):
    """Stop خلف أقرب مستوى هيكل."""
    atr = float(atr or 0)
    entry = float(entry)
    tech = tech or {}
    if is_call:
        floor_candidates = [
            _fnum(tech.get("support2")),
            _fnum(tech.get("support1")),
            _fnum(tech.get("vwap")),
            _fnum(tech.get("ema20")),
        ]
        below = [v for v in floor_candidates if v and v < entry]
        if below:
            stop = max(below) - atr * 0.15
        else:
            stop = entry - atr * 1.0
        stop = min(stop, entry - atr * 0.25)
        return round(stop, 2)
    else:
        ceil_candidates = [
            _fnum(tech.get("resist2")),
            _fnum(tech.get("resist1")),
            _fnum(tech.get("vwap")),
            _fnum(tech.get("ema20")),
        ]
        above = [v for v in ceil_candidates if v and v > entry]
        if above:
            stop = min(above) + atr * 0.15
        else:
            stop = entry + atr * 1.0
        stop = max(stop, entry + atr * 0.25)
        return round(stop, 2)


def stock_rr(entry, target, stop, is_call):
    """R:R على مستويات السهم — مستقر ولا يتأثر بقيم BS الشاذة."""
    entry, target, stop = float(entry), float(target), float(stop)
    if entry <= 0:
        return 0.0
    if is_call:
        risk, reward = entry - stop, target - entry
    else:
        risk, reward = stop - entry, entry - target
    if risk <= 0 or reward <= 0:
        return 0.0
    return round(min(reward / risk, 10.0), 1)


def live_stock_rr(price, tp1, stop, is_call):
    """R:R من السعر الحالي إلى TP1/Stop — يمنع تضخيم R:R بعد تجاوز Entry."""
    try:
        price, tp1, stop = float(price), float(tp1), float(stop)
    except (TypeError, ValueError):
        return 0.0
    if price <= 0 or tp1 <= 0 or stop <= 0:
        return 0.0
    if is_call:
        risk, reward = price - stop, tp1 - price
    else:
        risk, reward = stop - price, price - tp1
    if risk <= 0 or reward <= 0:
        return 0.0
    return round(min(reward / risk, 10.0), 1)


def entry_is_chased(price, entry, is_call, max_pct=None):
    """True إذا السعر تجاوز Entry بأكثر من الحد → مطاردة متأخرة."""
    max_pct = MAX_ENTRY_CHASE_PCT if max_pct is None else max_pct
    try:
        price, entry = float(price), float(entry)
    except (TypeError, ValueError):
        return False
    if price <= 0 or entry <= 0:
        return False
    if is_call:
        return price > entry * (1.0 + max_pct)
    return price < entry * (1.0 - max_pct)


def chase_distance_pct(price, entry, is_call):
    """كم % تجاوز السعر Entry (0 إذا لم يُطارَد)."""
    try:
        price, entry = float(price), float(entry)
    except (TypeError, ValueError):
        return 0.0
    if price <= 0 or entry <= 0:
        return 0.0
    if is_call and price > entry:
        return (price - entry) / entry * 100.0
    if (not is_call) and price < entry:
        return (entry - price) / entry * 100.0
    return 0.0


def grade_wait_setup(
    *,
    conf,
    liquid_ok,
    prem_ok,
    align_ok,
    exec_ok,
    trend_ok,
    mixed_ok,
    plan_valid,
    chased,
    dte,
    is_0dte,
    live_rr,
    tp1_rr,
    price,
    entry,
    stop,
    tp2,
    tp3,
    is_call,
    atr_pct=0,
    score=0,
    rvol=0,
):
    """
    يقيّم اكتمال الشروط حتى لو التوصية WAIT (مطاردة/0DTE/…).
    لا يغيّر بوابة BUY — يعطي setup_pct + wait_tier لزخم محتمل.
    من دراسة أسبوع 3–7 أغسطس: مطاردات خفيفة كثيراً كملت لـ TP بـ MFE قوي.
    """
    checks = [
        (conf >= MIN_CONF_TO_BUY, 18),
        (bool(liquid_ok), 14),
        (bool(prem_ok), 10),
        (bool(align_ok), 14),
        (bool(plan_valid), 8),
        (bool(exec_ok), 6),
        (bool(trend_ok or mixed_ok), 12),
        ((dte or 0) > 0 and not is_0dte, 8),
    ]
    # R:R: الحي المثالي، أو خطة قوية، أو مجال متبقٍ لـ TP2 بعد مطاردة خفيفة
    rr_tp2 = live_stock_rr(price, tp2, stop, is_call) if tp2 else 0.0
    rr_tp3 = live_stock_rr(price, tp3, stop, is_call) if tp3 else 0.0
    rr_ok = (
        live_rr >= MIN_RR_TO_BUY
        or tp1_rr >= MIN_RR_TO_BUY
        or (chased and max(rr_tp2, rr_tp3) >= 1.0)
        or (chased and live_rr >= 0.5 and max(rr_tp2, rr_tp3) >= 0.8)
    )
    checks.append((rr_ok, 10))

    earned = sum(w for ok, w in checks if ok)
    total = sum(w for _, w in checks)
    setup_pct = int(round(100.0 * earned / total)) if total else 0

    chase_pct = chase_distance_pct(price, entry, is_call) if chased else 0.0
    mild_chase = chased and chase_pct <= 1.5  # ≤1.5% فوق/تحت Entry
    try:
        atr = float(atr_pct or 0)
    except (TypeError, ValueError):
        atr = 0.0
    try:
        sc = float(score or 0)
    except (TypeError, ValueError):
        sc = 0.0
    try:
        rv = float(rvol or 0)
    except (TypeError, ValueError):
        rv = 0.0

    # مكافأة زخم استمرار (لا تُحوّل إلى BUY)
    momentum_bonus = 0
    if mild_chase and conf >= MIN_CONF_TO_BUY and align_ok:
        momentum_bonus += 6
    if chased and max(rr_tp2, rr_tp3) >= 1.2:
        momentum_bonus += 8
    elif chased and max(rr_tp2, rr_tp3) >= 0.8:
        momentum_bonus += 4
    if atr > 0 and chase_pct > 0 and chase_pct < atr:
        # المطاردة أصغر من مدى ATR اليومي → غالباً باقي وقود
        momentum_bonus += 5
    if sc >= 20 or rv >= 2.0:
        momentum_bonus += 3

    setup_pct = min(99, setup_pct + momentum_bonus)

    if setup_pct >= 90:
        tier = "FIRE"
    elif setup_pct >= 75:
        tier = "HOT"
    elif setup_pct >= 55:
        tier = "WARM"
    else:
        tier = "COLD"

    edge = ""
    if chased and tier in ("FIRE", "HOT"):
        edge = (
            f"مطاردة {'خفيفة' if mild_chase else ''} {chase_pct:.1f}% — "
            f"الشروط ≈{setup_pct}% · زخم قوي محتمل "
            f"(مجال TP2 R:R حي 1:{rr_tp2:.1f}) — "
            f"جاهز لدخول يدوي عند رجوع Entry (ليست BUY تلقائي)"
        ).replace("  ", " ").strip()
    elif tier == "FIRE":
        edge = (
            f"الشروط ≈{setup_pct}% — جاهز لدخول يدوي عند Entry/تراجع "
            f"(ليست BUY تلقائي؛ شبه مكتملة)"
        )
    elif tier == "HOT":
        edge = (
            f"الشروط ≈{setup_pct}% — جاهز لدخول يدوي عند لمس Entry "
            f"(ليست BUY تلقائي)"
        )
    elif tier == "WARM":
        edge = f"الشروط ≈{setup_pct}% — متوسطة؛ لا تستعجل"
    else:
        edge = f"الشروط ≈{setup_pct}% — ضعيفة حالياً"

    return {
        "setup_pct": setup_pct,
        "wait_tier": tier,
        "wait_edge_note": edge,
        "chase_pct": round(chase_pct, 2),
        "tp2_rr_live": rr_tp2,
    }


def is_exec_window(now=None):
    """
    نافذة اعتماد BUY/تيليجرام: أيام التداول بعد 9:30 ET + EXEC_AFTER_OPEN_MIN
    حتى 16:00 ET. يطابق قاعدة الداشبورد (بعد 9:45 ET).
    IGNORE_EXEC_WINDOW=1 يتجاوز الفحص للاختبار المحلي فقط.
    """
    if os.environ.get("IGNORE_EXEC_WINDOW", "").lower() in ("1", "true", "yes"):
        return True
    if now is None:
        now = datetime.now(MARKET_TZ)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=MARKET_TZ)
    else:
        now = now.astimezone(MARKET_TZ)
    if now.weekday() >= 5:
        return False
    open_mins = 9 * 60 + 30 + int(EXEC_AFTER_OPEN_MIN)
    close_mins = 16 * 60
    mins = now.hour * 60 + now.minute
    return open_mins <= mins < close_mins


def parse_price_num(val):
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    return parse_num(str(val).replace("$", ""))


def row_passes_save_filters(r):
    """فلترة نهائية موحّدة: OI · spread · premium · سعر منطقي."""
    ticker = str(r.get("Ticker", "")).upper()
    max_price  = profile_limit(ticker, "max_stock_price", MAX_STOCK_PRICE)
    prem_min   = profile_limit(ticker, "prem_min", TARGET_PREM_MIN)
    prem_max   = profile_limit(ticker, "prem_max", TARGET_PREM_MAX)
    max_spread = profile_limit(ticker, "max_spread_pct", MAX_SPREAD_PCT)
    min_oi     = profile_limit(ticker, "min_oi", MIN_OI)

    price = float(r.get("price_num") or parse_price_num(r.get("Price")) or 0)
    prem  = float(r.get("premium") or 0)
    sp    = float(r.get("spread_pct") if r.get("spread_pct") is not None else 99)
    oi    = effective_oi(r.get("oi"), r.get("opt_vol"), min_oi)
    strike = float(r.get("strike") or 0)

    if price <= 0 or price > max_price:
        return False
    if prem < prem_min or prem > prem_max:
        return False
    if oi < min_oi or sp > max_spread:
        return False
    if strike > 0 and price > 0 and abs(strike - price) / price > 0.25:
        return False
    # رفض أهداف غير منطقية
    for k in ("tp1_stock", "tp2_stock", "tp3_stock", "stop_stock", "entry_stock"):
        try:
            v = float(r.get(k) or 0)
        except (TypeError, ValueError):
            v = 0
        if k != "entry_stock" and v != 0 and v <= 0:
            return False
    dte = int(r.get("dte_num") if r.get("dte_num") is not None else 99)
    max_dte = profile_limit(ticker, "max_dte", MAX_DTE)
    if dte > max_dte:
        return False
    return True


def filter_results_df(df):
    if df is None or df.empty:
        return df
    mask = df.apply(row_passes_save_filters, axis=1)
    return df[mask].copy()


def row_passes_watchlist_filters(r):
    """فلتر مخفّف للـ watchlist — لا يُحفظ spread/OI ضعيف جداً."""
    ticker = str(r.get("Ticker", "")).upper()
    max_price  = profile_limit(ticker, "max_stock_price", MAX_STOCK_PRICE)
    prem_min   = profile_limit(ticker, "prem_min", TARGET_PREM_MIN)
    prem_max   = profile_limit(ticker, "prem_max", TARGET_PREM_MAX)
    max_spread = profile_limit(ticker, "max_spread_pct", MAX_SPREAD_PCT)

    price = float(r.get("price_num") or parse_price_num(r.get("Price")) or 0)
    prem  = float(r.get("premium") or 0)
    sp    = float(r.get("spread_pct") if r.get("spread_pct") is not None else 99)
    oi    = effective_oi(r.get("oi"), r.get("opt_vol"),
                         profile_limit(ticker, "min_oi", MIN_OI))
    if price <= 0 or price > max_price:
        return False
    if prem < prem_min or prem > prem_max:
        return False
    if sp > max_spread * 1.5:
        return False
    if oi < 100 and not is_spy_ticker(ticker):
        return False
    if is_spy_ticker(ticker) and oi < profile_limit(ticker, "min_oi", 100):
        return False
    dte = int(r.get("dte_num") if r.get("dte_num") is not None else 99)
    if dte > profile_limit(ticker, "max_dte", MAX_DTE):
        return False
    return True


def dashboard_fallback_df(df, max_rows=10):
    """أفضل المرشّحين للـ Dashboard عندما الفلتر الصارم يمنع الكل."""
    if df is None or df.empty:
        return df
    work = df.copy()
    work = work[work.apply(row_passes_watchlist_filters, axis=1)].copy()
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
    out = filtered_df.copy()
    # حافظ على ملاحظات Gemini السابقة عند إعادة المسح بدون المستشار (price_update)
    if "gemini_note" not in out.columns:
        out["gemini_note"] = ""
    if os.path.exists(path):
        try:
            prev = pd.read_csv(path)
            if "gemini_note" in prev.columns and "Ticker" in prev.columns:
                prev_map = {}
                for _, r in prev.iterrows():
                    note = str(r.get("gemini_note") or "").strip()
                    if not note or note.lower() == "nan":
                        continue
                    t = str(r.get("Ticker") or "").upper().strip()
                    try:
                        s = f"{float(r.get('strike') or 0):.4f}"
                    except (TypeError, ValueError):
                        s = ""
                    e = str(r.get("expiry") or "")[:10]
                    prev_map[(t, s, e)] = note
                    prev_map.setdefault((t, "", ""), note)
                for idx, r in out.iterrows():
                    cur = str(r.get("gemini_note") or "").strip()
                    if cur and cur.lower() != "nan":
                        continue
                    t = str(r.get("Ticker") or "").upper().strip()
                    try:
                        s = f"{float(r.get('strike') or 0):.4f}"
                    except (TypeError, ValueError):
                        s = ""
                    e = str(r.get("expiry") or "")[:10]
                    note = prev_map.get((t, s, e)) or prev_map.get((t, "", ""))
                    if note:
                        out.at[idx, "gemini_note"] = note
        except Exception:
            pass
    out["scanned_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    out.to_csv(path, index=False)
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
            if diff < best_diff and d >= today:
                best_diff = diff
                best = exp
        except Exception:
            continue
    return best if best_diff <= window else None


def next_friday_date(today=None):
    """أقرب جمعة قادمة (إذا اليوم جمعة → الجمعة الجاية وليس اليوم)."""
    today = today or datetime.now().date()
    add = (4 - today.weekday()) % 7
    if add == 0:
        add = 7
    return today + timedelta(days=add)


def find_nearest_expiry(expirations, ticker=None, today=None):
    """
    اختيار expiry — أولوية جمعة لاحقة (ليس 0DTE يوم الجمعة).
    يرفض أي expiry أبعد من MAX_DTE.
    """
    today = today or datetime.now().date()
    max_dte = profile_limit(ticker, "max_dte", MAX_DTE)
    prefer_friday = profile_limit(ticker, "prefer_friday", PREFER_FRIDAY)
    prefer_0 = profile_limit(ticker, "prefer_0dte", PREFER_0DTE)

    candidates = []
    for exp in expirations:
        try:
            d = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (d - today).days
            if dte < 0:
                continue
            if dte > max_dte:
                continue
            # Friday = weekday 4
            is_fri = d.weekday() == 4
            candidates.append((dte, exp, is_fri))
        except Exception:
            continue

    if not candidates:
        return None, None

    if prefer_friday:
        fridays = [c for c in candidates if c[2]]
        # تجنّب جمعة اليوم (0DTE): خذ أقرب جمعة لاحقة ضمن النافذة
        non_zero_fri = [c for c in fridays if c[0] > 0]
        if non_zero_fri:
            non_zero_fri.sort(key=lambda x: x[0])
            return non_zero_fri[0][1], non_zero_fri[0][0]
        # لا جمعة لاحقة في القائمة — لا نأخذ 0DTE إلا إذا prefer_0dte صراحة
        if fridays and prefer_0:
            fridays.sort(key=lambda x: x[0])
            return fridays[0][1], fridays[0][0]

    # افتراضياً لا نختار 0DTE (ثيتا قاتلة) ما لم يُطلب صراحة
    pool = candidates if prefer_0 else [c for c in candidates if c[0] > 0]
    if not pool:
        pool = candidates

    if prefer_0:
        pool.sort(key=lambda x: x[0])
        return pool[0][1], pool[0][0]

    target_dte = profile_limit(ticker, "dte_target", DTE_TARGET)
    window = profile_limit(ticker, "dte_window", DTE_WINDOW)
    best = min(pool, key=lambda x: abs(x[0] - target_dte))
    if abs(best[0] - target_dte) <= window:
        return best[1], best[0]
    pool.sort(key=lambda x: x[0])
    return pool[0][1], pool[0][0]


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
        "support1": None, "support2": None, "support3": None,
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
        set(round(h, 2) for h in highs[-20:] if h > current_price * 1.003)
    )
    dedup_r = []
    for h in swing_h:
        if not dedup_r or h - dedup_r[-1] > atr * 0.35:
            dedup_r.append(h)

    r1 = dedup_r[0] if len(dedup_r) >= 1 else round(current_price + atr * 1.0, 2)
    r2 = dedup_r[1] if len(dedup_r) >= 2 else round(r1 + atr * 0.8, 2)
    r3 = dedup_r[2] if len(dedup_r) >= 3 else round(r2 + atr * 1.0, 2)
    res["resist1"] = round(r1, 2)
    res["resist2"] = round(max(r2, r1 + atr * 0.4), 2)
    res["resist3"] = round(max(r3, res["resist2"] + atr * 0.5), 2)

    # مستويات الدعم (Swing Lows تحت السعر)
    swing_l = sorted(
        set(round(l, 2) for l in lows[-20:] if l < current_price * 0.997),
        reverse=True,
    )
    dedup_s = []
    for l in swing_l:
        if not dedup_s or dedup_s[-1] - l > atr * 0.35:
            dedup_s.append(l)

    s1 = dedup_s[0] if len(dedup_s) >= 1 else round(current_price - atr * 1.0, 2)
    s2 = dedup_s[1] if len(dedup_s) >= 2 else round(s1 - atr * 0.8, 2)
    s3 = dedup_s[2] if len(dedup_s) >= 3 else round(s2 - atr * 1.0, 2)
    res["support1"] = round(s1, 2)
    res["support2"] = round(min(s2, s1 - atr * 0.4), 2)
    res["support3"] = round(min(s3, res["support2"] - atr * 0.5), 2)

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

def _atm_row(chain_df, stock_price):
    if chain_df is None or chain_df.empty:
        return None
    work = chain_df.copy()
    work["_dist"] = abs(work["strike"] - stock_price)
    return work.loc[work["_dist"].idxmin()]


def _leg_from_atm(atm_row):
    if atm_row is None:
        return None
    bid  = float(atm_row.get("bid") or 0)
    ask  = float(atm_row.get("ask") or 0)
    last = float(atm_row.get("lastPrice") or 0)
    synthetic = False
    if ask == 0 and bid == 0:
        if last > 0:
            bid = round(last * 0.97, 2)
            ask = round(last * 1.03, 2)
            synthetic = True
        else:
            return None
    return {
        "premium":    round((bid + ask) / 2, 2),
        "bid":        round(bid, 2),
        "ask":        round(ask, 2),
        "last":       round(last, 2) if last else None,
        "synthetic_quote": synthetic,
        "iv":         round(float(atm_row.get("impliedVolatility") or 0), 4),
        "oi":         _safe_int(atm_row.get("openInterest")),
        "opt_vol":    _safe_int(atm_row.get("volume")),
        "spread_pct": round((ask - bid) / ask, 4) if ask > 0 else None,
        "strike":     float(atm_row.get("strike") or 0),
    }


def _apply_leg_fields(target, leg):
    if not leg:
        return False
    target["premium"]    = leg["premium"]
    target["opt_bid"]    = leg.get("bid")
    target["opt_ask"]    = leg.get("ask")
    target["opt_last"]   = leg.get("last")
    target["synthetic_quote"] = bool(leg.get("synthetic_quote"))
    target["iv"]         = leg["iv"]
    target["oi"]         = leg["oi"]
    target["opt_vol"]    = leg["opt_vol"]
    target["spread_pct"] = leg["spread_pct"]
    target["strike"]     = leg["strike"]
    return True


def _pick_best_strike(chain_df, stock_price, is_call, ticker):
    """
    اختيار strike — SPY: OTM ضمن premium $1–$5 (مو ATM ~$6+).
    الباقي: ATM كالسابق.
    """
    if chain_df is None or chain_df.empty:
        return None

    prof = ticker_profile(ticker)
    use_otm = prof.get("use_otm_strike", False)
    prem_min = profile_limit(ticker, "prem_min", TARGET_PREM_MIN)
    prem_max = profile_limit(ticker, "prem_max", TARGET_PREM_MAX)
    target   = prof.get("target_prem", (prem_min + prem_max) / 2)
    max_sp   = profile_limit(ticker, "max_spread_pct", MAX_SPREAD_PCT)
    min_oi   = profile_limit(ticker, "min_oi", MIN_OI)

    if not use_otm:
        return _leg_from_atm(_atm_row(chain_df, stock_price))

    candidates = []
    for _, row in chain_df.iterrows():
        leg = _leg_from_atm(row)
        if not leg:
            continue
        strike = leg["strike"]
        prem   = leg["premium"]
        sp     = leg["spread_pct"] if leg["spread_pct"] is not None else 99
        oi     = effective_oi(leg["oi"], leg["opt_vol"], min_oi)

        if prem < prem_min or prem > prem_max:
            continue
        if sp > max_sp:
            continue
        if oi < min_oi:
            continue
        # OTM فقط
        if is_call and strike <= stock_price * 1.001:
            continue
        if not is_call and strike >= stock_price * 0.999:
            continue
        if abs(strike - stock_price) / stock_price > 0.05:
            continue  # لا نبعد أكثر من 5%

        dist_prem = abs(prem - target)
        candidates.append((dist_prem, leg))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    # fallback: أرخص strike في النطاق (حتى ITM) ثم ATM
    loose = []
    for _, row in chain_df.iterrows():
        leg = _leg_from_atm(row)
        if not leg:
            continue
        prem = leg["premium"]
        sp   = leg["spread_pct"] if leg["spread_pct"] is not None else 99
        if prem_min <= prem <= prem_max and sp <= max_sp:
            loose.append((abs(prem - target), leg))
    if loose:
        loose.sort(key=lambda x: x[0])
        return loose[0][1]

    return _leg_from_atm(_atm_row(chain_df, stock_price))


def resolve_is_call(r, tech=None):
    direction_raw = str(r.get("direction", "")).upper()
    if "CALL" in direction_raw:
        return True
    if "PUT" in direction_raw:
        return False
    tech = tech or {}
    rsi_early   = float(tech.get("rsi", r.get("rsi", 50)) or 50)
    above_ema20 = tech.get("above_ema20", False)
    above_vwap  = tech.get("above_vwap", False)
    bull_signals = sum([above_ema20, above_vwap, rsi_early > 50])
    return bull_signals >= 2


def apply_option_leg(row, is_call):
    leg = row.get("leg_call" if is_call else "leg_put")
    if leg:
        _apply_leg_fields(row, leg)


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
        "strike": None,  "dte_num": None,
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

        dte_tgt = profile_limit(ticker, "dte_target", DTE_TARGET)
        dte_win = profile_limit(ticker, "dte_window", DTE_WINDOW)
        exp, dte = find_nearest_expiry(exps, ticker)
        if exp is None:
            out["error"] = "no_near_expiry"; return out

        out["expiry"]     = exp
        out["dte_num"]    = dte
        out["has_weekly"] = dte <= 7
        out["is_0dte"]    = dte == 0

        chain = yt.option_chain(exp)
        calls = chain.calls.copy() if chain.calls is not None else pd.DataFrame()
        puts  = chain.puts.copy()  if chain.puts  is not None else pd.DataFrame()

        if calls.empty:
            out["error"] = "empty_chain"; return out

        leg_call = _pick_best_strike(calls, stock_price, True, ticker)
        if not leg_call:
            out["error"] = "no_market"; return out

        leg_put = _pick_best_strike(puts, stock_price, False, ticker) if not puts.empty else None
        # اتجاه من ATM OI (إشارة فقط)
        atm_call = _leg_from_atm(_atm_row(calls, stock_price))
        atm_put  = _leg_from_atm(_atm_row(puts, stock_price)) if not puts.empty else None
        out["leg_call"] = leg_call
        out["leg_put"]  = leg_put or leg_call
        _apply_leg_fields(out, leg_call)

        # إشارة الاتجاه: Call OI مقابل Put OI (ATM)
        if atm_put and atm_call and atm_call["oi"] > 0 and atm_put["oi"] > 0:
            ratio = atm_call["oi"] / atm_put["oi"]
            out["direction"] = ("CALL 📈" if ratio > 1.4
                                else "PUT  📉" if ratio < 0.7
                                else "NEUTRAL")
            if "PUT" in out["direction"]:
                _apply_leg_fields(out, leg_put)

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

    if row.get("is_0dte") or parse_dte(row, default=99) == 0:
        score -= 1; notes.append("0DTE ثيتا -1")
    else:
        exp = str(row.get("expiry") or "")
        try:
            is_fri = datetime.strptime(exp[:10], "%Y-%m-%d").weekday() == 4
        except Exception:
            is_fri = False
        if is_fri:
            score += 2; notes.append("Friday✓ +2")
        elif row.get("has_weekly"):
            score += 1; notes.append("Weekly✓ +1")

    dte = int(row.get("dte_num") if row.get("dte_num") is not None else 99)
    max_dte = profile_limit(row.get("Ticker"), "max_dte", MAX_DTE)
    if dte > max_dte:
        score -= 5; notes.append(f"DTE>{max_dte}d -5")
    elif 0 < dte <= 7:
        score += 1; notes.append(f"DTE{dte}d +1")

    prem = row.get("premium", 0) or 0
    ticker = str(row.get("Ticker", "")).upper()
    if is_inverse_ticker(ticker):
        score -= 5
        notes.append("Inverse ETF -5")
    prem_min = profile_limit(ticker, "prem_min", TARGET_PREM_MIN)
    prem_max = profile_limit(ticker, "prem_max", TARGET_PREM_MAX)
    if prem_min <= prem <= prem_max:
        score += 2; notes.append("Prem✓ +2")

    atr_pct = row.get("atr_pct", 0) or 0
    if atr_pct > 5:     score += 3; notes.append("ATR%>5 +3")
    elif atr_pct > 3:   score += 2; notes.append("ATR%>3 +2")
    elif atr_pct > 1.5: score += 1; notes.append("ATR%>1.5 +1")

    # تأكيد صعود بريماركت من Finnhub
    fh_bonus = int(row.get("fh_pm_score") or 0)
    if fh_bonus > 0:
        score += fh_bonus
        if row.get("fh_pm_strong"):
            notes.append(f"FH PM قوي +{fh_bonus}")
        elif row.get("fh_pm_bullish"):
            notes.append(f"FH PM صعود +{fh_bonus}")

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

    is_call  = resolve_is_call(r, tech)
    price    = float(r.get("price_num", 0) or 0)
    strike   = float(r.get("strike",    0) or price)
    iv_raw   = float(r.get("iv", 0) or 0)
    iv_estimated = iv_raw < MIN_IV_DISPLAY
    iv       = iv_raw if not iv_estimated else 0.4
    dte      = parse_dte(r, default=7)
    premium  = float(r.get("premium",   0) or 0)
    atr      = float(tech.get("atr14") or (price * 0.025))
    opt_type = "call" if is_call else "put"

    if price == 0 or premium == 0:
        return plan
    if iv_estimated:
        plan["rec_note"] = "IV من Yahoo ناقص — R:R على السهم"

    T_entry = max(dte / 365, 1 / 365)
    T_half  = max((dte / 2) / 365, 1 / 365)

    # Greeks المحسوبة محليًا
    plan["greeks"] = bs_greeks(price, strike, T_entry, RISK_FREE_RATE, iv, opt_type)

    # جودة سعر العقد: mid سوق مقابل BS (يمنع BUY عند Yahoo/IV فاسد)
    bs_fair = round(bs_price(price, strike, T_entry, RISK_FREE_RATE, iv, opt_type), 2)
    pq = assess_option_price_quality(
        premium=premium,
        bs_fair=bs_fair,
        iv_raw=iv_raw,
        bid=r.get("opt_bid"),
        ask=r.get("opt_ask"),
        last=r.get("opt_last"),
        spread_pct=r.get("spread_pct"),
        synthetic_quote=bool(r.get("synthetic_quote")),
    )
    plan["bs_fair_price"] = pq.get("bs_fair_price")
    plan["premium_quality"] = pq.get("premium_quality", "estimate")
    plan["premium_quality_note"] = pq.get("premium_quality_note", "")
    plan["show_bs_scenarios"] = bool(pq.get("show_bs_scenarios"))
    plan["premium_ok_for_buy"] = bool(pq.get("premium_ok_for_buy"))
    if pq.get("premium_quality") == "unreliable" and not plan.get("rec_note"):
        plan["rec_note"] = pq.get("premium_quality_note") or "سعر عقد غير موثوق"

    # ── Entry ─────────────────────────────────────────────────────────────
    pm_high = r.get("pm_high")
    pm_low  = r.get("pm_low")
    yest_h  = tech.get("yesterday_high") or price
    yest_l  = tech.get("yesterday_low")  or price

    entry, entry_note = compute_entry_level(
        price, is_call, pm_high, pm_low, yest_h, yest_l, atr, tech,
    )
    plan["entry_stock"] = entry
    plan["entry_note"]  = entry_note

    # ── Stop Loss (خلف هيكل) ─────────────────────────────────────────────
    stop_stock = compute_structure_stop(entry, is_call, atr, tech)

    stop_opt = bs_price(stop_stock, strike, T_entry, RISK_FREE_RATE, iv, opt_type)
    plan["stop_stock"]  = stop_stock
    plan["stop_option"] = round(stop_opt, 2)

    risk_per = max(abs(premium - stop_opt) * 100, 1.0)
    plan["risk_per_contract"] = round(risk_per, 2)

    # ── Targets ───────────────────────────────────────────────────────────
    tp1, tp2, tp3 = compute_targets(entry, is_call, atr, tech)

    def opt_at(target_price):
        return round(bs_price(target_price, strike, T_half,
                               RISK_FREE_RATE, iv, opt_type), 2)

    def rr_stock(target_price):
        return stock_rr(entry, target_price, stop_stock, is_call)

    o1, o2, o3 = opt_at(tp1), opt_at(tp2), opt_at(tp3)
    plan.update({
        "tp1_stock": round(tp1, 2), "tp1_option": o1, "tp1_rr": rr_stock(tp1),
        "tp2_stock": round(tp2, 2), "tp2_option": o2, "tp2_rr": rr_stock(tp2),
        "tp3_stock": round(tp3, 2), "tp3_option": o3, "tp3_rr": rr_stock(tp3),
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
    tp1_rr   = plan["tp1_rr"] or 0  # R:R من Entry (مرجع الخطة فقط)
    best_rr  = max(tp1_rr, plan["tp2_rr"] or 0, plan["tp3_rr"] or 0)
    plan["best_rr"] = round(best_rr, 1)
    conf     = plan["confidence"]
    ticker   = r.get("Ticker")

    # R:R حي من السعر الحالي — هذا ما يُستخدم لبوابة BUY (يمنع R:R 9.0 بعد المطاردة)
    live_rr = live_stock_rr(price, plan.get("tp1_stock"), plan.get("stop_stock"), is_call)
    plan["tp1_rr_live"] = live_rr

    # الاتجاه موافق للصفقة
    trend_ok = (is_call and "BULLISH" in trend) or (not is_call and "BEARISH" in trend)
    # إصلاح 4: MIXED مقبول إذا كانت الثقة عالية جداً (≥75%)
    mixed_ok = "MIXED" in trend and conf >= 75

    # إذا كان هناك Earnings قبل الانتهاء → AVOID دائماً
    earn_risk = r.get("earn_before_expiry", False)

    # Entry تحقق = قرب Entry فقط (ليس مطاردة بعد تجاوزه)
    entry_s = float(plan.get("entry_stock") or 0)
    chased = entry_is_chased(price, entry_s, is_call)
    plan["entry_chased"] = chased
    entry_hit_now = False
    if entry_s > 0 and not chased:
        entry_hit_now = (is_call and price >= entry_s) or (not is_call and price <= entry_s)

    # سيولة كافية قبل أي BUY
    min_oi = profile_limit(ticker, "min_oi", MIN_OI)
    oi_ok = effective_oi(r.get("oi"), r.get("opt_vol"), min_oi) >= min_oi
    sp = float(r.get("spread_pct") if r.get("spread_pct") is not None else 99)
    max_sp = profile_limit(ticker, "max_spread_pct", MAX_SPREAD_PCT)
    spread_ok = sp <= max_sp
    liquid_ok = oi_ok and spread_ok

    # قواعد التشغيل: برايميوم ≥ $1 + فلتر اتجاه SPY (+ صناديق عكسية)
    prem_min = profile_limit(ticker, "prem_min", TARGET_PREM_MIN)
    prem_ok = premium >= prem_min
    spy_regime = r.get("spy_regime") or "NEUTRAL"
    align_ok, align_note = spy_alignment_ok(is_call, spy_regime, ticker)
    inv_ceiling, inv_note = inverse_recommendation_ceiling(is_call, spy_regime, ticker)
    plan["spy_regime"] = spy_regime
    plan["spy_align_ok"] = align_ok
    plan["inverse_ceiling"] = inv_ceiling

    # نافذة التنفيذ: لا BUY/تيليجرام قبل 9:45 ET (حتى لو باقي الشروط مكتملة)
    exec_ok = is_exec_window()
    plan["exec_window_ok"] = exec_ok
    # الصناديق العكسية: لا BUY تلقائي أبداً (سقف WAIT حتى مع محاذاة)
    inverse_blocks_buy = is_inverse_ticker(ticker)
    premium_quality_ok = bool(plan.get("premium_ok_for_buy"))
    almost_buy = (
        conf >= MIN_CONF_TO_BUY
        and live_rr >= MIN_RR_TO_BUY
        and liquid_ok
        and prem_ok
        and premium_quality_ok
        and align_ok
        and not chased
        and dte > 0
        and not r.get("is_0dte")
        and not inverse_blocks_buy
    )
    # BUY صارم: الشروط الفنية + داخل نافذة التنفيذ
    buy_ready = almost_buy and exec_ok

    if earn_risk:
        plan["recommendation"] = "AVOID"
        plan["rec_note"]       = "⚠️ Earnings قبل انتهاء العقد — خطر كبير"
    elif inv_ceiling == "AVOID":
        plan["recommendation"] = "AVOID"
        plan["rec_note"]       = inv_note
    elif not align_ok:
        plan["recommendation"] = "WAIT"
        plan["rec_note"]       = align_note + " — مراقبة فقط"
    elif inverse_blocks_buy and inv_ceiling == "WAIT":
        # عكسي محايد أو «محاذي» — مراقبة فقط، لا مسار BUY
        plan["recommendation"] = "WAIT"
        plan["rec_note"]       = inv_note
    elif chased:
        plan["recommendation"] = "WAIT"
        plan["rec_note"]       = (
            f"مطاردة >0.8% — السعر تجاوز Entry (${entry_s:.2f}) "
            f"وR:R الحي 1:{live_rr:.1f} — ادخل عند رجوع Entry"
        )
    elif not prem_ok:
        plan["recommendation"] = "WAIT"
        plan["rec_note"]       = f"Premium ${premium:.2f} تحت الحد ${prem_min:.2f} — الوضع الأساسي ≥$1"
    elif not premium_quality_ok:
        plan["recommendation"] = "WAIT"
        qnote = plan.get("premium_quality_note") or "سعر عقد غير موثوق من Yahoo"
        qtag = plan.get("premium_quality") or "estimate"
        plan["rec_note"] = f"سعر عقد ({qtag}) — {qnote}"
    elif not spread_ok:
        plan["recommendation"] = "WAIT"
        plan["rec_note"]       = f"سبريد مرتفع ({sp*100:.1f}%) — الحد {max_sp*100:.0f}%"
    elif not trade_plan_is_valid(plan, price, is_call):
        plan["recommendation"] = "WAIT"
        plan["rec_note"]       = "خطة الدخول غير متسقة مع السعر الحالي — انتظر"
    elif dte == 0 or r.get("is_0dte"):
        plan["recommendation"] = "WAIT"
        nf = next_friday_date()
        plan["rec_note"] = (
            f"0DTE — مراقبة فقط · يُفضّل عقد الجمعة الجاية {nf.isoformat()} "
            f"(ليس تنفيذ تلقائي اليوم)"
        )
        plan["next_friday_expiry"] = nf.isoformat()
    elif almost_buy and not exec_ok:
        plan["recommendation"] = "WAIT"
        open_h, open_m = divmod(9 * 60 + 30 + int(EXEC_AFTER_OPEN_MIN), 60)
        plan["rec_note"] = (
            f"خارج نافذة التنفيذ — لا BUY قبل "
            f"{open_h}:{open_m:02d} ET (+{EXEC_AFTER_OPEN_MIN}د بعد الافتتاح)"
        )
    elif buy_ready and (trend_ok or mixed_ok):
        plan["recommendation"] = "BUY"
        plan["rec_note"]       = ("الاتجاه والزخم والسيولة مناسبة"
                                   if trend_ok else "اختراق قوي رغم الاتجاه المختلط")
    elif buy_ready and entry_hit_now:
        # Entry تحقق يُسمح به فقط قرب المستوى + شروط BUY (بدون مطاردة)
        plan["recommendation"] = "BUY"
        plan["rec_note"]       = "Entry تحقق + شروط BUY مكتملة"
    elif conf >= MIN_CONF_TO_BUY and (live_rr >= MIN_RR_TO_BUY or tp1_rr >= MIN_RR_TO_BUY):
        plan["recommendation"] = "WAIT"
        if not liquid_ok:
            plan["rec_note"] = "سيولة ضعيفة — راقب"
        elif live_rr < MIN_RR_TO_BUY:
            plan["rec_note"] = f"R:R الحي 1:{live_rr:.1f} ضعيف — انتظر رجوع قرب Entry"
        else:
            plan["rec_note"] = "زخم جيد — راقب تأكيد الاتجاه"
    elif conf >= 50 and (live_rr >= 1.0 or tp1_rr >= 1.0):
        plan["recommendation"] = "WAIT"
        plan["rec_note"]       = "انتظر تأكيد الاختراق"
    else:
        plan["recommendation"] = "AVOID"
        plan["rec_note"]       = "الشروط غير مكتملة"

    # جودة WAIT / اكتمال الشروط (حتى مع مطاردة) — لا يفتح بوابة BUY
    grade = grade_wait_setup(
        conf=conf,
        liquid_ok=liquid_ok,
        prem_ok=prem_ok,
        align_ok=align_ok,
        exec_ok=exec_ok,
        trend_ok=trend_ok,
        mixed_ok=mixed_ok,
        plan_valid=trade_plan_is_valid(plan, price, is_call),
        chased=chased,
        dte=dte,
        is_0dte=bool(r.get("is_0dte") or dte == 0),
        live_rr=live_rr,
        tp1_rr=tp1_rr,
        price=price,
        entry=entry_s,
        stop=stop_stock,
        tp2=plan.get("tp2_stock"),
        tp3=plan.get("tp3_stock"),
        is_call=is_call,
        atr_pct=r.get("atr_pct"),
        score=r.get("Score") or r.get("score"),
        rvol=r.get("RVOL") or r.get("rvol"),
    )
    plan["setup_pct"] = grade["setup_pct"]
    plan["wait_tier"] = grade["wait_tier"]
    plan["wait_edge_note"] = grade["wait_edge_note"]
    plan["chase_pct"] = grade["chase_pct"]
    plan["tp2_rr_live"] = grade["tp2_rr_live"]
    # عكسي: لا تنبيه HOT/FIRE — رافعة عالية وأخطاء مكلفة هذا الأسبوع
    if is_inverse_ticker(ticker) and plan.get("wait_tier") in ("FIRE", "HOT"):
        plan["wait_tier"] = "WARM"
        plan["wait_edge_note"] = (
            (plan.get("wait_edge_note") or "") + " · سقف WARM لصندوق عكسي"
        ).strip(" ·")
        if plan.get("setup_pct", 0) >= 75:
            plan["setup_pct"] = 74
    if plan.get("recommendation") == "WAIT" and plan.get("wait_tier") in ("FIRE", "HOT"):
        base = plan.get("rec_note") or ""
        plan["rec_note"] = (base + " · " + plan.get("wait_edge_note", "")).strip(" ·")

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
    dte      = parse_dte(r, default=7)
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
    pq_label = {
        "market": "سوق",
        "estimate": "تقدير",
        "unreliable": "غير موثوق",
    }.get(str(plan.get("premium_quality") or r.get("premium_quality") or ""), "—")
    print(f"    Stock: ${price_n:.2f}    Strike: ${strike:.2f}    Premium: ${prem:.2f}  [{pq_label}]")
    print(f"    IV:    {iv_s}      DTE: {dte}d      Expiry: {r.get('expiry','N/A')}")
    print(f"    OI:    {oi:,}      OptVol: {opt_vol:,}      Spread: {sp_s}")
    print(f"    RVOL:  {rvol:.1f}x      ATR%: {atr_pct:.1f}%      Earnings: {earn_s}")
    if plan.get("premium_quality_note") or r.get("premium_quality_note"):
        print(f"    جودة السعر: {plan.get('premium_quality_note') or r.get('premium_quality_note')}")
    if plan.get("bs_fair_price") is not None:
        print(f"    BS fair ≈ ${float(plan['bs_fair_price']):.2f}")

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
    show_bs = plan.get("show_bs_scenarios")
    if show_bs is None:
        show_bs = True
    if strike and prem and iv and price_n and show_bs:
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
    elif strike and prem and price_n and not show_bs:
        print(f"{'─'*W}")
        print("  SCENARIOS  — مخفية (سعر عقد/IV غير موثوق)")
        note = plan.get("premium_quality_note") or r.get("premium_quality_note") or ""
        if note:
            print(f"    {note}")

    print(f"{'─'*W}")
    print(f"  Factors: {r.get('Notes','')}")
    print(f"{'═'*W}")


# ════════════════════════════════════════════════════════════════════════
#  CANDIDATE SOURCES — Finviz + قائمة يدوية
# ════════════════════════════════════════════════════════════════════════

SAVE_COLS = [
    "Ticker", "Price", "premium", "strike", "iv", "oi", "opt_vol", "spread_pct",
    "opt_bid", "opt_ask", "opt_last", "synthetic_quote",
    "premium_quality", "premium_quality_note", "bs_fair_price", "show_bs_scenarios",
    "atr_pct", "gap_pct", "RVOL", "pm_high", "pm_low", "pm_volume",
    "expiry", "dte_num", "direction", "earnings", "float_shares", "avg_vol",
    "entry_stock", "stop_stock", "tp1_stock", "tp2_stock", "tp3_stock", "tp1_rr",
    "tp1_rr_live", "tp2_rr_live", "entry_chased", "chase_pct", "exec_window_ok",
    "setup_pct", "wait_tier", "wait_edge_note", "next_friday_expiry",
    "entry_note", "fh_gap_pct", "fh_pm_bullish", "fh_pm_strong", "fh_pm_note",
    "spy_regime", "rec_note", "gemini_note",
    "Score", "recommendation", "confidence", "Notes", "scanned_at",
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
    candidates = pin_priority_tickers(combined, PRIORITY_TICKERS).reset_index(drop=True)

    src_counts = candidates.get("_source", pd.Series(dtype=str)).value_counts().to_dict()
    print(f"  → {len(candidates)} candidates ({src_counts})\n")
    return candidates


def pin_priority_tickers(df, priority=PRIORITY_TICKERS):
    """تثبيت tickers مهمة (SPY) حتى لو خرجت من top N."""
    if df is None or df.empty:
        return df
    pinned_parts = []
    rest = df.copy()
    for t in priority:
        hit = rest[rest["Ticker"].str.upper() == str(t).upper()]
        if not hit.empty:
            pinned_parts.append(hit.iloc[[0]])
            rest = rest[rest["Ticker"].str.upper() != str(t).upper()]
    if not pinned_parts:
        return df.head(MAX_STOCKS_DEEP).reset_index(drop=True)
    pinned = pd.concat(pinned_parts, ignore_index=True)
    room = max(0, MAX_STOCKS_DEEP - len(pinned))
    return pd.concat([pinned, rest.head(room)], ignore_index=True)


def process_candidates(candidates_df, show_progress=True):
    """Stage 2: Yahoo options + Finnhub PM + scoring + trade plan لكل مرشّح."""
    from finnhub_premarket import get_api_key, enrich_ticker_premarket

    rows = []
    fh_key = get_api_key()
    spy_regime, spy_tech, spy_px = fetch_spy_regime()
    if show_progress:
        print("⏳ Stage 2: Options + Premarket + Technical data...")
        print(f"   🧭 SPY regime: {spy_regime}"
              + (f" @ ${spy_px:.2f}" if spy_px else ""))
        print(f"   قواعد: Prem≥${TARGET_PREM_MIN:.2f} · Spread≤{MAX_SPREAD_PCT*100:.0f}%"
              f" · فلتر اتجاه SPY={'ON' if REQUIRE_SPY_ALIGNMENT else 'OFF'}"
              f" · تنفيذ بعد +{EXEC_AFTER_OPEN_MIN}د من الافتتاح")
        if fh_key:
            print("   Finnhub: تأكيد صعود بريماركت مفعّل")
        else:
            print("   Finnhub: بدون FINNHUB_API_KEY — تخطي تأكيد PM")
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

        fh = enrich_ticker_premarket(ticker, key=fh_key, delay=0.35) if fh_key else {
            "fh_gap_pct": None, "fh_pm_bullish": False, "fh_pm_strong": False,
            "fh_pm_score": 0, "fh_pm_note": "لا يوجد FINNHUB_API_KEY",
        }

        # إذا الصف SPY نفسه — حدّث البوصلة من بياناته الحية في المسح
        row_regime = spy_regime
        if is_spy_ticker(ticker) and hist is not None and len(hist) >= 5:
            row_regime = classify_spy_regime(
                compute_technicals(hist, price_num), price_num
            )
            spy_regime = row_regime

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
            "opt_bid":      opts.get("opt_bid"),
            "opt_ask":      opts.get("opt_ask"),
            "opt_last":     opts.get("opt_last"),
            "synthetic_quote": bool(opts.get("synthetic_quote")),
            "expiry":       opts["expiry"],
            "has_weekly":   opts["has_weekly"],
            "direction":    opts["direction"],
            "strike":       opts["strike"],
            "dte_num":      opts["dte_num"],
            "is_0dte":      opts.get("is_0dte", False),
            "_hist":        hist,
            "pm_high":      opts["pm_high"],
            "pm_low":       opts["pm_low"],
            "pm_last":      opts["pm_last"],
            "pm_volume":    opts["pm_volume"],
            "leg_call":     opts.get("leg_call"),
            "leg_put":      opts.get("leg_put"),
            "_error":       opts["error"],
            "_source":      s.get("_source", ""),
            "fh_gap_pct":   fh.get("fh_gap_pct"),
            "fh_pm_bullish": bool(fh.get("fh_pm_bullish")),
            "fh_pm_strong": bool(fh.get("fh_pm_strong")),
            "fh_pm_note":   fh.get("fh_pm_note", ""),
            "fh_pm_score":  int(fh.get("fh_pm_score") or 0),
            "spy_regime":   row_regime,
        }

        tech = compute_technicals(hist, price_num)
        is_call = resolve_is_call(row, tech)
        trend = str(tech.get("trend") or "")
        strong_bear = "BEARISH" in trend and "BULLISH" not in trend
        # تأكيد صعود بريماركت من Finnhub → فضّل CALL (سهم/عقد صاعد)
        if row["fh_pm_bullish"] and not strong_bear:
            is_call = True
        # لا نقلب الاتجاه صمتاً: التعارض مع SPY يُحوَّل إلى WAIT داخل compute_trade_plan
        apply_option_leg(row, is_call)
        row["direction"] = "CALL 📈" if is_call else "PUT  📉"

        score, notes = score_stock(row)
        row["Score"] = score
        row["Notes"] = " | ".join(notes)

        try:
            plan = compute_trade_plan(row, tech)
        except Exception as e:
            row["_error"] = (row.get("_error") or "") + f" plan:{e}"
            plan = {"recommendation": "AVOID", "confidence": 0,
                    "entry_stock": None, "stop_stock": None,
                    "tp1_stock": None, "tp2_stock": None, "tp3_stock": None,
                    "rec_note": "", "spy_regime": row_regime}
        row["recommendation"] = plan.get("recommendation", "AVOID")
        row["confidence"]     = plan.get("confidence", 0)
        row["entry_stock"]    = plan.get("entry_stock")
        row["stop_stock"]     = plan.get("stop_stock")
        row["tp1_stock"]      = plan.get("tp1_stock")
        row["tp2_stock"]      = plan.get("tp2_stock")
        row["tp3_stock"]      = plan.get("tp3_stock")
        row["tp1_rr"]         = plan.get("tp1_rr")
        row["tp1_rr_live"]    = plan.get("tp1_rr_live")
        row["tp2_rr_live"]    = plan.get("tp2_rr_live")
        row["entry_chased"]   = bool(plan.get("entry_chased", False))
        row["chase_pct"]      = plan.get("chase_pct")
        row["exec_window_ok"] = bool(plan.get("exec_window_ok", False))
        row["setup_pct"]      = plan.get("setup_pct")
        row["wait_tier"]      = plan.get("wait_tier")
        row["wait_edge_note"] = plan.get("wait_edge_note", "")
        row["next_friday_expiry"] = plan.get("next_friday_expiry") or ""
        row["entry_note"]     = plan.get("entry_note", "")
        row["rec_note"]       = plan.get("rec_note", "")
        row["spy_regime"]     = plan.get("spy_regime", row_regime)
        row["premium_quality"] = plan.get("premium_quality", "estimate")
        row["premium_quality_note"] = plan.get("premium_quality_note", "")
        row["bs_fair_price"] = plan.get("bs_fair_price")
        row["show_bs_scenarios"] = bool(plan.get("show_bs_scenarios", False))
        if is_spy_ticker(ticker):
            row["profile"] = "spy"

        if show_progress:
            stk_s   = f"${row['strike']:.2f}" if row["strike"]  else "   N/A"
            prem_s  = f"${row['premium']:.2f}" if row["premium"] is not None else "   N/A"
            oi_s    = f"{row['oi']:,}"          if row["oi"]     is not None else "  N/A"
            sp_s    = f"{row['spread_pct']*100:.1f}%" if row["spread_pct"] is not None else "  N/A"
            pm_s    = f"${row['pm_high']:.2f}" if row["pm_high"] else "   N/A"
            fh_s    = " FH↑" if row["fh_pm_bullish"] else ""
            err_s   = f"  ⚠ {row['_error']}"  if row["_error"] else ""
            print(f"   {ticker:<8} {str(row.get('Price',''))[:8]:>8}  {stk_s:>8}  "
                  f"{prem_s:>8}  {oi_s:>7}  {sp_s:>7}  {pm_s:>9}  {score:>4}{fh_s}{err_s}")

        rows.append(row)
        time.sleep(DELAY_BETWEEN)

    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
              .sort_values("Score", ascending=False)
              .reset_index(drop=True))


def ensure_spy_saved(combined, result_df, save_cols):
    """SPY يُحفظ دائماً في CSV إذا وُجدت بيانات عقد."""
    if result_df is None or result_df.empty:
        return combined
    spy_hit = result_df[result_df["Ticker"].str.upper() == SPY_TICKER]
    if spy_hit.empty:
        return combined
    spy = spy_hit.iloc[[0]].copy()
    if not float(spy.iloc[0].get("premium") or 0):
        return combined
    cols = [c for c in save_cols if c in spy.columns]
    spy = spy[cols]
    if combined is None or combined.empty:
        return spy
    combined = combined[combined["Ticker"].str.upper() != SPY_TICKER]
    return pd.concat([spy, combined], ignore_index=True)


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

    combined = ensure_spy_saved(combined, result_df, save_cols)

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
