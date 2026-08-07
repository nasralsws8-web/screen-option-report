"""
outcome_tracker.py
------------------
يسجّل توصيات BUY الجديدة في outcomes.csv
ويتحقق من النتائج للتوصيات المفتوحة (هل وصل Entry/TP1/TP2/TP3 أو ضرب Stop؟)

كل عقد يُحفظ مستقلاً بمفتاح: Ticker + Strike + Expiry
(نفس السهم بعقد مختلف = صف جديد في السجل)

result_pct      = حركة السهم % (للمرجع)
option_pnl_pct  = ربح/خسارة العقد % من premium الدخول → exit_premium
exit_premium    = سعر العقد عند الخروج (يفضّل إغلاق Yahoo اليومي الحقيقي)
exit_premium_source = market | intrinsic | estimate

مقاييس فترة المراقبة:
  days_to_entry   = أيام تقويمية من التوصية → لمس Entry
  days_held       = أيام تقويمية من Entry → الخروج (أو حتى اليوم إن مفتوحة)
  mfe_pct         = أقصى حركة مواتية % من Entry أثناء المسك (Max Favorable)
  mae_pct         = أقصى تراجع معاكس % من Entry أثناء المسك (Max Adverse)
  hold_expiry_pct = حركة السهم % لو انمسك من Entry حتى إغلاق يوم الانتهاء
  expiry_premium  = سعر العقد عند/قرب يوم الانتهاء (حتى لو أُغلقت الصفقة عند TP/Stop)
  expiry_premium_source = market | intrinsic | estimate
  option_pnl_expiry_pct = P&L العقد % لو انمسك من premium الدخول حتى الانتهاء
  exit_date       = تاريخ إغلاق الصفقة المدارة (TP/Stop/Expiry)
"""

import math
import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from scipy.stats import norm

from data_quality import classify_data_quality

OUTCOMES_FILE = "outcomes.csv"
RESULTS_FILE  = "options_v3_results.csv"

RISK_FREE_RATE = 0.053
MIN_IV         = 0.01
DEFAULT_IV     = 0.40

OUTCOMES_COLS = [
    "date", "ticker", "direction", "score", "confidence",
    "price_at_rec", "entry_stock", "stop_stock",
    "tp1_stock", "tp2_stock", "tp3_stock", "expiry",
    "premium", "strike", "iv", "dte_num",
    "rec_kind",       # BUY | WAIT_HOT
    "entry_chased", "setup_pct", "wait_tier", "scanned_at",
    "entry_hit", "entry_hit_date", "exit_date", "exit_stock",
    "exit_premium", "exit_premium_source",
    "status",   # open / tp1_hit / tp2_hit / tp3_hit / stop_hit / expired
    "result_pct",
    "option_pnl_pct",
    "days_to_entry", "days_held",
    "mfe_pct", "mae_pct", "hold_expiry_pct",
    "expiry_premium", "expiry_premium_source", "option_pnl_expiry_pct",
    "data_quality",       # reliable | partial | unreliable | open
    "data_quality_note",
]

PATH_METRIC_COLS = [
    "exit_date", "exit_stock", "exit_premium", "exit_premium_source",
    "days_to_entry", "days_held",
    "mfe_pct", "mae_pct", "hold_expiry_pct",
]


def bs_price(S, K, T, r, sigma, opt):
    """Black-Scholes — نفس منطق cheap_options_screener_v3."""
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


def parse_date(val):
    try:
        if val is None or val == "" or (isinstance(val, float) and math.isnan(val)):
            return None
        ts = pd.to_datetime(val, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


def valid_target(price):
    try:
        return float(price) > 0
    except (TypeError, ValueError):
        return False


def parse_price(val):
    """سعر قد يأتي كرقم أو نص فيه $."""
    if val is None or val == "":
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    try:
        f = float(str(val).replace("$", "").replace(",", "").strip())
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def resolve_is_call(direction, entry, tp1, stop):
    d = str(direction or "").upper()
    if "PUT" in d:
        return False
    if "CALL" in d:
        return True
    entry = float(entry or 0)
    tp1   = float(tp1 or 0)
    stop  = float(stop or 0)
    if entry > 0 and tp1 > 0:
        if tp1 > entry:
            return True
        if tp1 < entry:
            return False
    if entry > 0 and stop > 0:
        return stop < entry
    return True


def move_pct(entry, target, is_call):
    if entry <= 0:
        return 0.0
    if is_call:
        return round((target - entry) / entry * 100, 2)
    return round((entry - target) / entry * 100, 2)


def occ_symbol(ticker, expiry, strike, is_call):
    """رمز OCC مثل AMZN260807C00272500."""
    t = str(ticker or "").strip().upper()
    exp = parse_date(expiry)
    if not t or not exp:
        return None
    try:
        k = int(round(float(strike) * 1000))
    except (TypeError, ValueError):
        return None
    if k <= 0:
        return None
    return f"{t}{exp.strftime('%y%m%d')}{'C' if is_call else 'P'}{k:08d}"


def fetch_option_daily_hist(symbol, start_date, end_date, cache):
    """تاريخ يومي لسعر العقد من Yahoo (إغلاق حقيقي)."""
    if not symbol:
        return pd.DataFrame()
    key = (symbol, str(start_date), str(end_date))
    if key in cache:
        return cache[key]
    try:
        hist = yf.download(
            symbol,
            start=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
            end=(pd.Timestamp(end_date) + timedelta(days=2)).strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        if hist is None:
            hist = pd.DataFrame()
    except Exception:
        hist = pd.DataFrame()
    cache[key] = hist
    return hist


def market_option_close(ticker, expiry, strike, is_call, on_date, cache):
    """إغلاق العقد اليومي في/قبل تاريخ الخروج — سعر سوق حقيقي إن وُجد."""
    if on_date is None:
        return None
    sym = occ_symbol(ticker, expiry, strike, is_call)
    if not sym:
        return None
    start = on_date - timedelta(days=14)
    hist = fetch_option_daily_hist(sym, start, on_date, cache)
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    close = hist_close_on_or_before(hist, on_date)
    if close is None or close < 0:
        return None
    return round(float(close), 4)


def calc_option_exit_price(row, stock_at_exit, exit_date):
    """
    تقدير سعر العقد ($) — احتياطي فقط إذا فشل سعر السوق.
    - يوم الانتهاء أو بعده: intrinsic
    - قبل الانتهاء: Black-Scholes
    """
    try:
        strike = float(row.get("strike") or 0)
        stock_at_exit = float(stock_at_exit)
    except (TypeError, ValueError):
        return None
    if strike <= 0 or stock_at_exit <= 0:
        return None

    iv_raw = row.get("iv")
    try:
        iv = float(iv_raw) if iv_raw is not None else DEFAULT_IV
    except (TypeError, ValueError):
        iv = DEFAULT_IV
    if iv < MIN_IV:
        iv = DEFAULT_IV

    expiry = parse_date(row.get("expiry"))
    if exit_date is None:
        exit_date = datetime.now().date()
    elif not isinstance(exit_date, datetime):
        exit_date = parse_date(exit_date) or datetime.now().date()

    if expiry:
        days_left = (expiry - exit_date).days
        T = 0.0 if days_left <= 0 else days_left / 365
    else:
        raw_dte = row.get("dte_num")
        if raw_dte is None or raw_dte == "":
            dte = 7
        else:
            try:
                dte = int(float(raw_dte))
            except (TypeError, ValueError):
                dte = 7
        T = max(dte / 365, 0.0)

    is_call = resolve_is_call(
        row.get("direction"), row.get("entry_stock"),
        row.get("tp1_stock"), row.get("stop_stock"),
    )
    opt_type = "call" if is_call else "put"
    return round(bs_price(stock_at_exit, strike, T, RISK_FREE_RATE, iv, opt_type), 4)


def calc_option_pnl_pct(row, stock_at_exit, exit_date):
    """نسبة ربح/خسارة العقد عند سعر سهم معيّن وتاريخ خروج."""
    try:
        premium = float(row.get("premium") or 0)
    except (TypeError, ValueError):
        return None
    if premium <= 0:
        return None
    exit_val = calc_option_exit_price(row, stock_at_exit, exit_date)
    if exit_val is None:
        return None
    return round((exit_val - premium) / premium * 100, 2)


def first_hit_date(post, level, is_call, hit_type="tp"):
    """أول يوم في post حيث تحقق TP أو Stop."""
    level = float(level)
    for dt, bar in post.iterrows():
        h, l = float(bar["High"]), float(bar["Low"])
        if hit_type == "tp":
            if is_call and h >= level:
                return dt.date()
            if not is_call and l <= level:
                return dt.date()
        else:
            if is_call and l <= level:
                return dt.date()
            if not is_call and h >= level:
                return dt.date()
    return None


def _level_preexisting_at_open(is_call, open_px, high, low, entry, level, kind):
    """
    True إذا مستوى Stop/TP كان متحققاً على الافتتاح قبل مسار دخول نظيف
    في نفس الشمعة اليومية (لمس Entry لاحقاً في اليوم).
    kind: 'stop' | 'tp'
    """
    if open_px is None or entry is None or level is None:
        return False
    try:
        o, h, l, e, lvl = float(open_px), float(high), float(low), float(entry), float(level)
    except (TypeError, ValueError):
        return False
    if e <= 0 or lvl <= 0:
        return False
    if kind == "tp":
        return (is_call and o >= lvl and l <= e) or ((not is_call) and o <= lvl and h >= e)
    # stop
    return (is_call and o <= lvl and h >= e) or ((not is_call) and o >= lvl and l <= e)


def _level_already_at_rec(is_call, price_at_rec, level, kind):
    """
    True إذا المستوى كان متحققاً أصلاً عند سعر التوصية —
    لا يُحسب كإنجاز بعد طرح التوصية.
    """
    if price_at_rec is None or level is None:
        return False
    try:
        px, lvl = float(price_at_rec), float(level)
    except (TypeError, ValueError):
        return False
    if px <= 0 or lvl <= 0:
        return False
    if kind == "tp":
        return (is_call and px >= lvl) or ((not is_call) and px <= lvl)
    return (is_call and px <= lvl) or ((not is_call) and px >= lvl)


def resolve_first_touch(post, is_call, stop, tp1, tp2, tp3, entry=None,
                        rec_date=None, price_at_rec=None):
    """
    أول لمسة زمنية: يمشي يوم بيوم.
    إذا Stop و TP في نفس الشمعة اليومية → Stop يفوز (محافظة)،
    ما لم يكن Stop مُسبقاً على الافتتاح قبل الدخول.
    كذلك لا تُحتسب TP إذا كانت على الافتتاح قبل مسار الدخول.

    يوم التوصية فقط (rec_date): الشمعة اليومية لا تعرف ترتيب High/Low،
    فقمة الصباح قد تسبق طرح التوصية. لذلك:
      - لا هدف/وقف إذا كان متحققاً عند price_at_rec
      - التأكيد يكون بإغلاق اليوم (Close) لا بـ High/Low وحدها
    بعد يوم التوصية: High/Low كالسابق.
    يرجع (status, exit_price, exit_date) أو None.
    """
    levels = []
    if valid_target(stop):
        levels.append(("stop_hit", float(stop), "stop"))
    if valid_target(tp1):
        levels.append(("tp1_hit", float(tp1), "tp"))
    if valid_target(tp2):
        levels.append(("tp2_hit", float(tp2), "tp"))
    if valid_target(tp3):
        levels.append(("tp3_hit", float(tp3), "tp"))
    if not levels:
        return None

    try:
        entry_f = float(entry) if entry is not None else None
        if entry_f is not None and entry_f <= 0:
            entry_f = None
    except (TypeError, ValueError):
        entry_f = None

    px_rec = parse_price(price_at_rec)
    rec_d = parse_date(rec_date) if rec_date is not None else None

    for dt, bar in post.iterrows():
        bar_date = dt.date() if hasattr(dt, "date") else pd.Timestamp(dt).date()
        is_rec_day = rec_d is not None and bar_date == rec_d
        o = float(bar["Open"]) if "Open" in bar.index and pd.notna(bar["Open"]) else None
        h, l = float(bar["High"]), float(bar["Low"])
        c = float(bar["Close"]) if "Close" in bar.index and pd.notna(bar["Close"]) else None
        stop_hit = False
        tp_hits = []
        for status, level, kind in levels:
            if is_rec_day and _level_already_at_rec(is_call, px_rec, level, kind):
                continue
            if kind == "stop":
                if is_rec_day:
                    # بعد التوصية فقط: إغلاق اليوم يؤكد الوقف (تجنّب قاع قبل الطرح)
                    hit = c is not None and ((c <= level) if is_call else (c >= level))
                else:
                    hit = (l <= level) if is_call else (h >= level)
                if hit:
                    stop_hit = True
            else:
                if is_rec_day:
                    hit = c is not None and ((c >= level) if is_call else (c <= level))
                else:
                    hit = (h >= level) if is_call else (l <= level)
                if hit:
                    tp_hits.append((status, level))
        if stop_hit:
            stop_lvl = next(lvl for st, lvl, k in levels if k == "stop")
            if not _level_preexisting_at_open(is_call, o, h, l, entry_f, stop_lvl, "stop"):
                return "stop_hit", stop_lvl, bar_date
            # Stop مُسبق على الافتتاح → تجاهله هذا اليوم وكمّل لـ TP / اليوم التالي
        if tp_hits:
            # أقرب هدف تحقق في هذا اليوم (TP1 قبل TP3)
            order = {"tp1_hit": 1, "tp2_hit": 2, "tp3_hit": 3}
            tp_hits.sort(key=lambda x: order.get(x[0], 9))
            status, level = tp_hits[0]
            if _level_preexisting_at_open(is_call, o, h, l, entry_f, level, "tp"):
                continue
            return status, level, bar_date
    return None


def _as_ts(d):
    if d is None:
        return None
    return pd.Timestamp(d).normalize()


def hist_close_on_or_before(hist, target_date):
    """آخر إغلاق في/قبل تاريخ معيّن."""
    if hist is None or hist.empty or target_date is None:
        return None
    cutoff = _as_ts(target_date)
    sub = hist[hist.index.normalize() <= cutoff]
    if sub.empty:
        return None
    return float(sub["Close"].iloc[-1])


def calc_mfe_mae(post, entry, is_call):
    """أقصى حركة مواتية/معاكسة % من Entry على شموع يومية."""
    if post is None or post.empty or entry <= 0:
        return None, None
    mfe = 0.0
    mae = 0.0
    for _, bar in post.iterrows():
        h, l = float(bar["High"]), float(bar["Low"])
        if is_call:
            fav = (h - entry) / entry * 100.0
            adv = (entry - l) / entry * 100.0
        else:
            fav = (entry - l) / entry * 100.0
            adv = (h - entry) / entry * 100.0
        if fav > mfe:
            mfe = fav
        if adv > mae:
            mae = adv
    return round(mfe, 2), round(mae, 2)


def calendar_days(start, end):
    if start is None or end is None:
        return None
    try:
        return int((pd.Timestamp(end).date() - pd.Timestamp(start).date()).days)
    except Exception:
        return None


def build_hist_cache(outcomes_df):
    """تحميل يومي مرة واحدة لكل تيكر."""
    today = datetime.now().date()
    cache = {}
    tickers = sorted({
        str(t).strip()
        for t in outcomes_df.get("ticker", pd.Series(dtype=str)).tolist()
        if str(t).strip()
    })
    for ticker in tickers:
        grp = outcomes_df[outcomes_df["ticker"].astype(str).str.strip() == ticker]
        starts = [parse_date(d) for d in grp["date"].tolist()]
        starts = [d for d in starts if d]
        if not starts:
            continue
        start = min(starts) - timedelta(days=5)
        end = today + timedelta(days=3)
        try:
            hist = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            cache[ticker] = hist if hist is not None else pd.DataFrame()
        except Exception as e:
            print(f"⚠️  {ticker}: فشل جلب التاريخ — {e}")
            cache[ticker] = pd.DataFrame()
    return cache


def resolve_exit_stock(row, hist=None):
    """
    سعر السهم عند الخروج:
    - TP/Stop → مستوى الهدف/الوقف
    - expired → إغلاق يوم الانتهاء (من hist إن وُجد)
    """
    status = str(row.get("status") or "")
    if status == "open" or not status:
        return None

    level_map = {
        "tp1_hit": float(row.get("tp1_stock") or 0),
        "tp2_hit": float(row.get("tp2_stock") or 0),
        "tp3_hit": float(row.get("tp3_stock") or 0),
        "stop_hit": float(row.get("stop_stock") or 0),
    }
    if status in level_map and level_map[status] > 0:
        return round(level_map[status], 4)

    if status == "expired":
        expiry_date = parse_date(row.get("expiry")) or parse_date(row.get("exit_date"))
        if hist is not None and not hist.empty and expiry_date:
            close = hist_close_on_or_before(hist, expiry_date)
            if close is not None:
                return round(close, 4)
        # احتياطي: استنتج من entry + result_pct إن أمكن
        try:
            entry = float(row.get("entry_stock") or 0)
            res = float(row.get("result_pct"))
            is_call = resolve_is_call(
                row.get("direction"), entry,
                row.get("tp1_stock"), row.get("stop_stock"),
            )
            if entry > 0 and res == res:  # not NaN
                if is_call:
                    return round(entry * (1 + res / 100.0), 4)
                return round(entry * (1 - res / 100.0), 4)
        except (TypeError, ValueError):
            pass
    return None


def compute_path_metrics(row, hist, today=None):
    """
    يحسب مقاييس المراقبة لصف واحد.
    يرجع dict بالحقول الجديدة (قد تكون None).
    """
    today = today or datetime.now().date()
    out = {c: None for c in PATH_METRIC_COLS}

    rec_date = parse_date(row.get("date"))
    entry = float(row.get("entry_stock") or 0)
    stop = float(row.get("stop_stock") or 0)
    tp1 = float(row.get("tp1_stock") or 0)
    tp2 = float(row.get("tp2_stock") or 0)
    tp3 = float(row.get("tp3_stock") or 0)
    expiry_date = parse_date(row.get("expiry"))
    status = str(row.get("status") or "open")
    is_call = resolve_is_call(row.get("direction"), entry, tp1, stop)

    if hist is None or hist.empty or rec_date is None:
        # حتى بدون hist: TP/Stop معروفان من المستويات
        if status != "open":
            out["exit_stock"] = resolve_exit_stock(row, None)
            if status == "expired" and expiry_date:
                out["exit_date"] = expiry_date.strftime("%Y-%m-%d")
        return out

    # سعر/تاريخ الخروج متاح حتى بدون Entry (حالة expired)
    if status != "open":
        if status == "expired" and expiry_date:
            out["exit_date"] = expiry_date.strftime("%Y-%m-%d")
        out["exit_stock"] = resolve_exit_stock(row, hist)
        # exit_premium يُملأ لاحقاً من سوق Yahoo في enrich_market_exit_premiums

    if entry <= 0:
        return out

    # Entry hit من السجل أو إعادة اكتشاف — فقط داخل نافذة العقد
    entry_hit = bool(row.get("entry_hit", False))
    entry_hit_date = parse_date(row.get("entry_hit_date"))

    def _entry_window(h):
        w = h[h.index.normalize() >= _as_ts(rec_date)]
        if expiry_date:
            w = w[w.index.normalize() <= _as_ts(expiry_date)]
        return w

    # تجاهل دخول مسجّل بعد الانتهاء (بيانات قديمة خاطئة)
    if entry_hit_date and expiry_date and entry_hit_date > expiry_date:
        entry_hit = False
        entry_hit_date = None

    if not entry_hit or entry_hit_date is None:
        post_rec = _entry_window(hist)
        if is_call:
            hit_days = post_rec[post_rec["High"] >= entry]
        else:
            hit_days = post_rec[post_rec["Low"] <= entry]
        if not hit_days.empty:
            entry_hit = True
            entry_hit_date = hit_days.index[0].date()

    if not entry_hit or entry_hit_date is None:
        # بدون دخول: نبقي exit_stock لـ expired إن وُجد
        return out

    # إذا لُمس Entry قبل/نفس يوم التوصية → 0 (جاهز فوراً)
    dte = calendar_days(rec_date, entry_hit_date)
    out["days_to_entry"] = 0 if dte is not None and dte < 0 else dte

    # تاريخ الخروج
    exit_date = parse_date(row.get("exit_date")) or parse_date(out.get("exit_date"))
    if exit_date and expiry_date and exit_date > expiry_date:
        exit_date = expiry_date
    if exit_date is None and status != "open":
        if status == "expired" and expiry_date:
            exit_date = expiry_date
        else:
            post = hist[hist.index.normalize() >= _as_ts(entry_hit_date)]
            if expiry_date:
                post = post[post.index.normalize() <= _as_ts(expiry_date)]
            touch = resolve_first_touch(
                post, is_call, stop, tp1, tp2, tp3, entry=entry,
                rec_date=rec_date, price_at_rec=row.get("price_at_rec"),
            )
            if touch:
                exit_date = touch[2]
                # سعر اللمس الفعلي من الشمعة إن أمكن أدق من المستوى
                if status.startswith("tp") or status == "stop_hit":
                    out["exit_stock"] = round(float(touch[1]), 4)
            elif status == "expired" and expiry_date:
                exit_date = expiry_date

    if status == "open":
        hold_end = today
        if expiry_date and today > expiry_date:
            hold_end = expiry_date
    else:
        hold_end = exit_date or today

    out["exit_date"] = exit_date.strftime("%Y-%m-%d") if exit_date else out.get("exit_date")
    held = calendar_days(entry_hit_date, hold_end)
    out["days_held"] = 0 if held is not None and held < 0 else held

    # نافذة المسك لمقاييس MFE/MAE
    window = hist[hist.index.normalize() >= _as_ts(entry_hit_date)]
    if hold_end:
        window = window[window.index.normalize() <= _as_ts(hold_end)]
    mfe, mae = calc_mfe_mae(window, entry, is_call)
    out["mfe_pct"] = mfe
    out["mae_pct"] = mae

    # نتيجة لو انمسك حتى إغلاق يوم الانتهاء
    if expiry_date and entry > 0:
        expiry_close = hist_close_on_or_before(hist, expiry_date)
        if expiry_close is not None and today >= expiry_date:
            out["hold_expiry_pct"] = move_pct(entry, expiry_close, is_call)
            if status == "expired" and out.get("exit_stock") is None:
                out["exit_stock"] = round(expiry_close, 4)

    # تأكيد exit_stock / exit_premium للحالات المغلقة
    if status != "open" and out.get("exit_stock") is None:
        out["exit_stock"] = resolve_exit_stock(row, hist)
    if status != "open" and out.get("exit_stock") is not None:
        # لا تستبدل سعر سوق سبق حفظه — يُحدَّث لاحقاً في enrich_market_exit_premiums
        prev_src = str(row.get("exit_premium_source") or "")
        if prev_src != "market":
            exit_d = parse_date(out.get("exit_date")) or parse_date(row.get("exit_date"))
            if status == "expired" and expiry_date:
                exit_d = expiry_date
            out["exit_premium"] = calc_option_exit_price(row, out["exit_stock"], exit_d)
            out["exit_premium_source"] = (
                "intrinsic" if (expiry_date and exit_d and exit_d >= expiry_date) else "estimate"
            )

    return out


def apply_path_metrics(outcomes_df, idx, metrics):
    for col, val in metrics.items():
        outcomes_df.at[idx, col] = val


_OPTION_PRICE_CACHE = {}


def apply_close(outcomes_df, idx, row, status, entry, stock_price, is_call, exit_date=None):
    """يحدّث status + result_pct + option_pnl_pct + exit_date + exit_stock + exit_premium."""
    outcomes_df.at[idx, "status"] = status
    outcomes_df.at[idx, "result_pct"] = move_pct(entry, stock_price, is_call)
    try:
        px = float(stock_price)
        if px > 0:
            outcomes_df.at[idx, "exit_stock"] = round(px, 4)
    except (TypeError, ValueError):
        pass
    if exit_date is not None:
        if hasattr(exit_date, "strftime"):
            outcomes_df.at[idx, "exit_date"] = exit_date.strftime("%Y-%m-%d")
        else:
            outcomes_df.at[idx, "exit_date"] = str(exit_date)[:10]
    exit_d = exit_date
    if exit_d is None:
        exit_d = parse_date(outcomes_df.at[idx, "exit_date"]) if "exit_date" in outcomes_df.columns else None
    if isinstance(exit_d, datetime):
        exit_d = exit_d.date()

    # 1) سعر سوق حقيقي من Yahoo (إغلاق يوم الخروج)
    mkt = market_option_close(
        row.get("ticker"), row.get("expiry"), row.get("strike"),
        is_call, exit_d, _OPTION_PRICE_CACHE,
    )
    if mkt is not None:
        outcomes_df.at[idx, "exit_premium"] = mkt
        outcomes_df.at[idx, "exit_premium_source"] = "market"
        try:
            prem = float(row.get("premium") or 0)
            if prem > 0:
                outcomes_df.at[idx, "option_pnl_pct"] = round((mkt - prem) / prem * 100, 2)
        except (TypeError, ValueError):
            pass
        return

    # 2) احتياطي: intrinsic / BS
    exit_prem = calc_option_exit_price(row, stock_price, exit_d)
    if exit_prem is not None:
        outcomes_df.at[idx, "exit_premium"] = exit_prem
        exp = parse_date(row.get("expiry"))
        src = "intrinsic" if (exp and exit_d and exit_d >= exp) else "estimate"
        outcomes_df.at[idx, "exit_premium_source"] = src
    opt_pnl = calc_option_pnl_pct(row, stock_price, exit_d)
    if opt_pnl is not None:
        outcomes_df.at[idx, "option_pnl_pct"] = opt_pnl


def recalculate_result_pct(row):
    entry = float(row.get("entry_stock") or 0)
    if entry <= 0:
        return row.get("result_pct")
    is_call = resolve_is_call(
        row.get("direction"), entry,
        row.get("tp1_stock"), row.get("stop_stock"),
    )
    status = str(row.get("status") or "")
    targets = {
        "tp1_hit":  float(row.get("tp1_stock") or 0),
        "tp2_hit":  float(row.get("tp2_stock") or 0),
        "tp3_hit":  float(row.get("tp3_stock") or 0),
        "stop_hit": float(row.get("stop_stock") or 0),
    }
    tgt = targets.get(status, 0)
    if tgt > 0:
        return move_pct(entry, tgt, is_call)
    return row.get("result_pct")


def recalculate_option_pnl(row):
    status = str(row.get("status") or "")
    if status == "open":
        return row.get("option_pnl_pct")
    entry = float(row.get("entry_stock") or 0)
    if entry <= 0:
        return None
    is_call = resolve_is_call(
        row.get("direction"), entry,
        row.get("tp1_stock"), row.get("stop_stock"),
    )
    stock_map = {
        "tp1_hit":  float(row.get("tp1_stock") or 0),
        "tp2_hit":  float(row.get("tp2_stock") or 0),
        "tp3_hit":  float(row.get("tp3_stock") or 0),
        "stop_hit": float(row.get("stop_stock") or 0),
    }
    if status in stock_map and stock_map[status] > 0:
        exit_date = parse_date(row.get("entry_hit_date")) or parse_date(row.get("date"))
        return calc_option_pnl_pct(row, stock_map[status], exit_date)
    return row.get("option_pnl_pct")


def recalculate_all_outcomes(outcomes_df):
    fixed_stock = fixed_opt = 0
    for idx, row in outcomes_df.iterrows():
        if row.get("status") == "open":
            continue
        old = row.get("result_pct")
        new = recalculate_result_pct(row)
        if new is not None and old != new:
            outcomes_df.at[idx, "result_pct"] = new
            fixed_stock += 1
        if has_option_data(row):
            old_opt = row.get("option_pnl_pct")
            new_opt = recalculate_option_pnl(row)
            if new_opt is not None and old_opt != new_opt:
                outcomes_df.at[idx, "option_pnl_pct"] = new_opt
                fixed_opt += 1
    if fixed_stock:
        print(f"✅ أعيد حساب result_pct لـ {fixed_stock} صف")
    if fixed_opt:
        print(f"✅ أعيد حساب option_pnl_pct لـ {fixed_opt} صف")
    return outcomes_df


def repair_preexisting_level_hits(outcomes_df, hist_cache=None):
    """
    يلغي إغلاق TP/Stop إذا:
      1) المستوى متحققاً على افتتاح يوم الدخول مع لمس Entry (قبل مسار نظيف)، أو
      2) الإغلاق كان يوم التوصية بدون تأكيد Close بعد الطرح
         (قمة/قاع اليوم قد تكون قبل التوصية على شمعة يومية).
    يعيد الصف إلى open ليعاد تقييمه.
    """
    if outcomes_df is None or outcomes_df.empty:
        return outcomes_df
    if hist_cache is None:
        hist_cache = build_hist_cache(outcomes_df)

    fixed = 0
    for idx, row in outcomes_df.iterrows():
        status = str(row.get("status") or "")
        if status != "stop_hit" and not status.startswith("tp"):
            continue
        entry_d = parse_date(row.get("entry_hit_date"))
        exit_d = parse_date(row.get("exit_date"))
        rec_d = parse_date(row.get("date"))
        if entry_d is None or exit_d is None or entry_d != exit_d:
            continue

        entry = float(row.get("entry_stock") or 0)
        tp1 = float(row.get("tp1_stock") or 0)
        tp2 = float(row.get("tp2_stock") or 0)
        tp3 = float(row.get("tp3_stock") or 0)
        stop = float(row.get("stop_stock") or 0)
        is_call = resolve_is_call(row.get("direction"), entry, tp1, stop)
        if status == "stop_hit":
            level, kind = stop, "stop"
        else:
            level = {"tp1_hit": tp1, "tp2_hit": tp2, "tp3_hit": tp3}.get(status, 0)
            kind = "tp"
        if entry <= 0 or level <= 0:
            continue

        ticker = str(row.get("ticker") or "").strip()
        hist = hist_cache.get(ticker)
        if hist is None or hist.empty:
            continue
        day = hist[hist.index.normalize() == _as_ts(entry_d)]
        if day.empty:
            continue
        bar = day.iloc[0]
        try:
            o = float(bar["Open"])
            h = float(bar["High"])
            l = float(bar["Low"])
            c = float(bar["Close"])
        except (TypeError, ValueError, KeyError):
            continue

        px_rec = parse_price(row.get("price_at_rec"))
        reason = None
        if _level_preexisting_at_open(is_call, o, h, l, entry, level, kind):
            reason = "على الافتتاح قبل الدخول"
        elif rec_d is not None and exit_d == rec_d:
            if _level_already_at_rec(is_call, px_rec, level, kind):
                reason = "متحقق عند سعر التوصية (قبل/مع الطرح)"
            else:
                # يوم التوصية: بدون تأكيد الإغلاق لا نثق بـ High/Low
                close_ok = (
                    (c >= level) if (is_call and kind == "tp") else
                    (c <= level) if (is_call and kind == "stop") else
                    (c <= level) if ((not is_call) and kind == "tp") else
                    (c >= level)
                )
                if not close_ok:
                    reason = "يوم التوصية بدون تأكيد إغلاق بعد الطرح"

        if not reason:
            continue

        outcomes_df.at[idx, "status"] = "open"
        outcomes_df.at[idx, "exit_date"] = None
        outcomes_df.at[idx, "exit_stock"] = None
        outcomes_df.at[idx, "exit_premium"] = None
        outcomes_df.at[idx, "exit_premium_source"] = None
        outcomes_df.at[idx, "result_pct"] = None
        outcomes_df.at[idx, "option_pnl_pct"] = None
        fixed += 1
        label = "Stop" if kind == "stop" else "الهدف"
        print(f"  ↺ {ticker}: أُلغي {status} — {label} {reason} ({entry_d})")

    if fixed:
        print(f"✅ إصلاح مستويات مُسبقة/قبل التوصية (TP/Stop): {fixed} صف أُعيد فتحه")
    return outcomes_df


# توافق مع الاسم القديم
repair_preexisting_tp_hits = repair_preexisting_level_hits


def has_option_data(row):
    try:
        return float(row.get("premium") or 0) > 0 and float(row.get("strike") or 0) > 0
    except (TypeError, ValueError):
        return False


def backfill_option_fields(outcomes_df):
    """يملأ premium/strike/iv للصفوف القديمة من options_v3_results.csv."""
    if not os.path.exists(RESULTS_FILE):
        return outcomes_df
    try:
        results = pd.read_csv(RESULTS_FILE)
    except Exception:
        return outcomes_df
    if results.empty:
        return outcomes_df

    def _contract_key(ticker, strike, expiry):
        t = str(ticker or "").upper().strip()
        try:
            s = f"{float(strike):.4f}"
        except (TypeError, ValueError):
            s = ""
        e = str(expiry or "")[:10]
        return t, s, e

    lookup = {}
    by_ticker = {}
    for _, r in results.iterrows():
        t = str(r.get("Ticker", "")).upper().strip()
        if not t:
            continue
        key = _contract_key(t, r.get("strike"), r.get("expiry"))
        lookup[key] = r
        by_ticker[t] = r  # آخر صف للسهم — احتياطي فقط إذا ما فيه سترايك

    filled = 0
    for idx, row in outcomes_df.iterrows():
        if str(row.get("status") or "") != "open":
            continue
        if has_option_data(row):
            continue
        t = str(row.get("ticker", "")).upper().strip()
        key = _contract_key(t, row.get("strike"), row.get("expiry"))
        src = lookup.get(key)
        # لا تملأ من سهم آخر بعقد مختلف إذا عندنا سترايك معروف ومخالف
        if src is None:
            try:
                row_strike = float(row.get("strike") or 0)
            except (TypeError, ValueError):
                row_strike = 0
            if row_strike > 0:
                continue  # سترايك معروف لكن ما طابق نتائج اليوم — لا تخلط عقود
            src = by_ticker.get(t)
        if src is None:
            continue
        try:
            prem = float(src.get("premium") or 0)
            strike = float(src.get("strike") or 0)
        except (TypeError, ValueError):
            continue
        if prem <= 0 or strike <= 0:
            continue
        outcomes_df.at[idx, "premium"] = prem
        outcomes_df.at[idx, "strike"]  = strike
        outcomes_df.at[idx, "iv"]      = src.get("iv")
        outcomes_df.at[idx, "dte_num"] = src.get("dte_num")
        filled += 1

    if filled:
        print(f"✅ backfill: premium/strike لـ {filled} صف من CSV الحالي")
    return outcomes_df


def repair_closed_option_estimates(outcomes_df):
    """صفوف مغلقة: premium الحالي من CSV ≠ عقد وقت الدخول — لا نقدّر P&L."""
    cleared = 0
    for idx, row in outcomes_df.iterrows():
        if str(row.get("status") or "") == "open":
            continue
        if not has_option_data(row) and pd.isna(row.get("option_pnl_pct")):
            continue
        for col in ("premium", "strike", "iv", "dte_num", "option_pnl_pct"):
            outcomes_df.at[idx, col] = None
        cleared += 1
    if cleared:
        print(f"✅ repair: مسح تقديرات P&L لـ {cleared} صف مغلق (بدون premium وقت الدخول)")
    return outcomes_df


def load_outcomes():
    STR_COLS = [
        "date", "ticker", "direction", "expiry",
        "entry_hit_date", "exit_date", "status",
    ]
    if os.path.exists(OUTCOMES_FILE):
        df = pd.read_csv(OUTCOMES_FILE)
        for col in OUTCOMES_COLS:
            if col not in df.columns:
                df[col] = None
        for col in STR_COLS:
            if col in df.columns:
                df[col] = df[col].astype(object).where(df[col].notna(), None)
        ordered = [c for c in OUTCOMES_COLS if c in df.columns]
        extra = [c for c in df.columns if c not in ordered]
        return df[ordered + extra]
    return pd.DataFrame(columns=OUTCOMES_COLS)


def _norm_strike(val):
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        return round(float(val), 4)
    except (TypeError, ValueError):
        return None


def _norm_expiry(val):
    d = parse_date(val)
    return d.isoformat() if d else str(val or "").strip()


def _same_contract(df, ticker, strike_f, expiry_key):
    """صفوف نفس العقد: Ticker + Strike + Expiry."""
    if df is None or df.empty:
        return df.iloc[0:0] if df is not None else pd.DataFrame()
    mask = df["ticker"].astype(str).str.strip() == ticker
    if strike_f is None:
        mask = mask & df["strike"].isna()
    else:
        strikes = pd.to_numeric(df["strike"], errors="coerce")
        mask = mask & strikes.notna() & (strikes.sub(strike_f).abs() < 1e-6)
    exp = df["expiry"].map(_norm_expiry)
    mask = mask & (exp == expiry_key)
    return df[mask]


def _row_is_hot_wait(r) -> bool:
    if str(r.get("recommendation") or "").strip().upper() != "WAIT":
        return False
    tier = str(r.get("wait_tier") or "").strip().upper()
    if tier in ("HOT", "FIRE"):
        return True
    try:
        return float(r.get("setup_pct") or 0) >= 75
    except (TypeError, ValueError):
        return False


def _truthy_chase(val) -> bool:
    s = str(val).strip().lower()
    return s in ("1", "true", "yes", "y")


def add_new_recommendations(outcomes_df):
    if not os.path.exists(RESULTS_FILE):
        print("⚠️  options_v3_results.csv غير موجود — تخطي إضافة توصيات جديدة")
        return outcomes_df

    results = pd.read_csv(RESULTS_FILE)
    if results.empty or "recommendation" not in results.columns:
        print("⚠️  options_v3_results.csv فاضي — تخطي إضافة توصيات جديدة")
        return outcomes_df

    buy_rows = results[results["recommendation"] == "BUY"].copy()
    hot_rows = results[results.apply(_row_is_hot_wait, axis=1)].copy()
    today = datetime.now().strftime("%Y-%m-%d")
    added = 0

    def _append_row(r, rec_kind):
        nonlocal outcomes_df, added
        ticker = str(r.get("Ticker", "")).strip()
        if not ticker:
            return

        entry = float(r.get("entry_stock") or 0)
        if entry <= 0:
            return

        prem = r.get("premium")
        strike = r.get("strike")
        try:
            prem_f = float(prem) if prem is not None else None
        except (TypeError, ValueError):
            prem_f = None
        strike_f = _norm_strike(strike)
        expiry_key = _norm_expiry(r.get("expiry", ""))

        open_same = _same_contract(
            outcomes_df[outcomes_df["status"] == "open"],
            ticker, strike_f, expiry_key,
        )
        if not open_same.empty:
            return

        already = _same_contract(
            outcomes_df[outcomes_df["date"].astype(str) == today],
            ticker, strike_f, expiry_key,
        )
        if not already.empty:
            return

        try:
            setup_pct = float(r.get("setup_pct")) if r.get("setup_pct") is not None else None
        except (TypeError, ValueError):
            setup_pct = None

        new_row = {
            "date":           today,
            "ticker":         ticker,
            "direction":      str(r.get("direction", "")).replace("📈", "").replace("📉", "").strip(),
            "score":          r.get("Score", 0),
            "confidence":     r.get("confidence", 0),
            "price_at_rec":   r.get("Price", 0),
            "entry_stock":    entry,
            "stop_stock":     float(r.get("stop_stock") or 0),
            "tp1_stock":      float(r.get("tp1_stock") or 0),
            "tp2_stock":      float(r.get("tp2_stock") or 0),
            "tp3_stock":      float(r.get("tp3_stock") or 0),
            "expiry":         r.get("expiry", ""),
            "premium":        prem_f,
            "strike":         strike_f,
            "iv":             r.get("iv"),
            "dte_num":        r.get("dte_num"),
            "rec_kind":       rec_kind,
            "entry_chased":   _truthy_chase(r.get("entry_chased")),
            "setup_pct":      setup_pct,
            "wait_tier":      str(r.get("wait_tier") or ""),
            "scanned_at":     str(r.get("scanned_at") or ""),
            "entry_hit":      False,
            "entry_hit_date": "",
            "exit_date":      "",
            "exit_stock":     None,
            "exit_premium":   None,
            "exit_premium_source": "",
            "status":         "open",
            "result_pct":     None,
            "option_pnl_pct": None,
            "days_to_entry":  None,
            "days_held":      None,
            "mfe_pct":        None,
            "mae_pct":        None,
            "hold_expiry_pct": None,
            "data_quality":      "open",
            "data_quality_note": "صفقة مفتوحة" if rec_kind == "BUY" else "WAIT HOT — دخول يدوي / تتبع",
        }
        outcomes_df = pd.concat(
            [outcomes_df, pd.DataFrame([new_row])],
            ignore_index=True,
        )
        added += 1
        print(f"  + [{rec_kind}] {ticker} strike={strike_f} expiry={expiry_key}")

    for _, r in buy_rows.iterrows():
        _append_row(r, "BUY")
    for _, r in hot_rows.iterrows():
        _append_row(r, "WAIT_HOT")

    print(f"✅ أضفت {added} توصية جديدة (BUY + WAIT HOT)")
    return outcomes_df


def update_open_outcomes(outcomes_df, hist_cache=None):
    open_mask = outcomes_df["status"] == "open"
    open_rows = outcomes_df[open_mask]

    if open_rows.empty:
        print("لا توجد توصيات مفتوحة للتحقق")
        return outcomes_df

    today = datetime.now().date()
    if hist_cache is None:
        hist_cache = build_hist_cache(outcomes_df)

    for idx, row in open_rows.iterrows():
        ticker     = str(row["ticker"]).strip()
        rec_date   = parse_date(row["date"])
        entry      = float(row["entry_stock"] or 0)
        stop       = float(row["stop_stock"] or 0)
        tp1        = float(row["tp1_stock"] or 0)
        tp2        = float(row["tp2_stock"] or 0)
        tp3        = float(row["tp3_stock"] or 0)
        expiry_str = str(row.get("expiry", ""))
        is_call    = resolve_is_call(row.get("direction"), entry, tp1, stop)

        if not ticker or entry <= 0 or rec_date is None:
            continue

        expiry_date = parse_date(expiry_str) or (today + timedelta(days=7))
        hist = hist_cache.get(ticker)
        if hist is None or hist.empty:
            print(f"⚠️  {ticker}: لا بيانات سعر")
            continue

        # قص التاريخ من يوم التوصية فما بعد
        hist = hist[hist.index.normalize() >= _as_ts(rec_date)]
        if hist.empty:
            continue

        entry_hit      = bool(row.get("entry_hit", False))
        entry_hit_date = row.get("entry_hit_date")

        if not entry_hit:
            win = hist[hist.index.normalize() <= _as_ts(expiry_date)] if expiry_date else hist
            if is_call:
                hit_days = win[win["High"] >= entry]
            else:
                hit_days = win[win["Low"] <= entry]
            if not hit_days.empty:
                entry_hit      = True
                entry_hit_date = hit_days.index[0].strftime("%Y-%m-%d")
                outcomes_df.at[idx, "entry_hit"]      = True
                outcomes_df.at[idx, "entry_hit_date"] = entry_hit_date

        if entry_hit and entry_hit_date:
            post = hist[hist.index.normalize() >= _as_ts(entry_hit_date)]
            if expiry_date:
                post = post[post.index.normalize() <= _as_ts(expiry_date)]
            if not post.empty:
                touch = resolve_first_touch(
                    post, is_call, stop, tp1, tp2, tp3, entry=entry,
                    rec_date=rec_date, price_at_rec=row.get("price_at_rec"),
                )
                if touch:
                    status, exit_price, exit_d = touch
                    apply_close(outcomes_df, idx, row, status, entry, exit_price, is_call, exit_d)

        if today > expiry_date and outcomes_df.at[idx, "status"] == "open":
            # إغلاق بانتهاء العقد — استخدم إغلاق يوم الانتهاء إن وُجد
            exp_close = hist_close_on_or_before(hist, expiry_date)
            last_close = exp_close if exp_close is not None else float(hist["Close"].iloc[-1])
            apply_close(
                outcomes_df, idx, row, "expired", entry, last_close, is_call, expiry_date,
            )

        # حدّث مقاييس المسار للصف (مفتوح أو أُغلق الآن)
        row_now = outcomes_df.loc[idx]
        metrics = compute_path_metrics(row_now, hist_cache.get(ticker), today=today)
        apply_path_metrics(outcomes_df, idx, metrics)

        opt_s = outcomes_df.at[idx, "option_pnl_pct"]
        opt_str = f" opt={opt_s:+.1f}%" if opt_s is not None and not pd.isna(opt_s) else ""
        mfe = outcomes_df.at[idx, "mfe_pct"]
        mae = outcomes_df.at[idx, "mae_pct"]
        path_str = ""
        if mfe is not None and not pd.isna(mfe):
            path_str = f" MFE={mfe:+.1f}% MAE={mae:+.1f}%"
        print(f"  {ticker}: {outcomes_df.at[idx, 'status']}{opt_str}{path_str}")

    return outcomes_df


def enrich_path_metrics(outcomes_df, hist_cache=None):
    """يملأ/يحدّث مقاييس المراقبة لكل الصفوف (مغلق ومفتوح)."""
    if outcomes_df is None or outcomes_df.empty:
        return outcomes_df

    today = datetime.now().date()
    if hist_cache is None:
        print("📥 جلب أسعار يومية لمقاييس المسار...")
        hist_cache = build_hist_cache(outcomes_df)

    filled = 0
    for idx, row in outcomes_df.iterrows():
        ticker = str(row.get("ticker") or "").strip()
        hist = hist_cache.get(ticker)
        if hist is None or hist.empty:
            continue
        metrics = compute_path_metrics(row, hist, today=today)
        # زامن entry_hit_date من المقاييس/النافذة الصحيحة
        metrics_probe = compute_path_metrics(row, hist, today=today)
        # أعد اكتشاف الدخول داخل نافذة العقد فقط
        rec_date = parse_date(row.get("date"))
        expiry_date = parse_date(row.get("expiry"))
        entry = float(row.get("entry_stock") or 0)
        tp1 = float(row.get("tp1_stock") or 0)
        stop = float(row.get("stop_stock") or 0)
        is_call = resolve_is_call(row.get("direction"), entry, tp1, stop)
        if rec_date and entry > 0:
            post_rec = hist[hist.index.normalize() >= _as_ts(rec_date)]
            if expiry_date:
                post_rec = post_rec[post_rec.index.normalize() <= _as_ts(expiry_date)]
            hit_days = (
                post_rec[post_rec["High"] >= entry]
                if is_call else
                post_rec[post_rec["Low"] <= entry]
            )
            old_hit = parse_date(row.get("entry_hit_date"))
            if not hit_days.empty:
                new_hit = hit_days.index[0].date()
                outcomes_df.at[idx, "entry_hit"] = True
                outcomes_df.at[idx, "entry_hit_date"] = new_hit.strftime("%Y-%m-%d")
            elif old_hit and expiry_date and old_hit > expiry_date:
                outcomes_df.at[idx, "entry_hit"] = False
                outcomes_df.at[idx, "entry_hit_date"] = ""
            row = outcomes_df.loc[idx]
            metrics = compute_path_metrics(row, hist, today=today)
        else:
            metrics = metrics_probe

        apply_path_metrics(outcomes_df, idx, metrics)
        if metrics.get("days_to_entry") is not None or metrics.get("mfe_pct") is not None:
            filled += 1

    print(f"✅ مقاييس المسار: {filled}/{len(outcomes_df)} صف")
    return outcomes_df


def enrich_market_exit_premiums(outcomes_df, cache=None):
    """
    يستبدل تقدير BS بسعر إغلاق العقد اليومي الحقيقي من Yahoo (OCC).
    يعيد حساب option_pnl_pct من السعر الحقيقي.
    """
    if outcomes_df is None or outcomes_df.empty:
        return outcomes_df

    if cache is None:
        cache = _OPTION_PRICE_CACHE

    market_n = intrinsic_n = estimate_n = fail_n = 0
    print("📥 جلب أسعار العقود الحقيقية من Yahoo...")

    for idx, row in outcomes_df.iterrows():
        status = str(row.get("status") or "")
        if status == "open":
            continue

        ticker = str(row.get("ticker") or "").strip()
        expiry = row.get("expiry")
        strike = row.get("strike")
        entry = float(row.get("entry_stock") or 0)
        tp1 = float(row.get("tp1_stock") or 0)
        stop = float(row.get("stop_stock") or 0)
        is_call = resolve_is_call(row.get("direction"), entry, tp1, stop)
        exit_d = parse_date(row.get("exit_date"))
        if status == "expired":
            exit_d = parse_date(expiry) or exit_d
        if exit_d is None:
            fail_n += 1
            continue

        mkt = market_option_close(ticker, expiry, strike, is_call, exit_d, cache)
        if mkt is not None:
            outcomes_df.at[idx, "exit_premium"] = mkt
            outcomes_df.at[idx, "exit_premium_source"] = "market"
            try:
                prem = float(row.get("premium") or 0)
                if prem > 0:
                    outcomes_df.at[idx, "option_pnl_pct"] = round((mkt - prem) / prem * 100, 2)
            except (TypeError, ValueError):
                pass
            market_n += 1
            continue

        # احتياطي
        exit_stock = row.get("exit_stock")
        try:
            exit_stock = float(exit_stock) if exit_stock is not None else None
        except (TypeError, ValueError):
            exit_stock = None
        if exit_stock is None or exit_stock <= 0:
            fail_n += 1
            continue
        est = calc_option_exit_price(row, exit_stock, exit_d)
        if est is None:
            fail_n += 1
            continue
        exp = parse_date(expiry)
        src = "intrinsic" if (exp and exit_d >= exp) else "estimate"
        outcomes_df.at[idx, "exit_premium"] = est
        outcomes_df.at[idx, "exit_premium_source"] = src
        opt_pnl = calc_option_pnl_pct(row, exit_stock, exit_d)
        if opt_pnl is not None:
            outcomes_df.at[idx, "option_pnl_pct"] = opt_pnl
        if src == "intrinsic":
            intrinsic_n += 1
        else:
            estimate_n += 1

    print(
        f"✅ أسعار العقود: سوق={market_n} | ذاتي={intrinsic_n} | "
        f"تقدير={estimate_n} | فشل={fail_n}"
    )
    return outcomes_df


def enrich_expiry_option_premiums(outcomes_df, hist_cache=None, cache=None):
    """
    يتابع سعر العقد يومياً حتى يوم الانتهاء — حتى لو أُغلقت الصفقة مبكراً عند TP/Stop.
    asof = min(اليوم، expiry):
      - قبل الانتهاء: آخر إغلاق سوق متاح (mark) → المصدر market
      - يوم الانتهاء أو بعده: إغلاق الانتهاء أو intrinsic
    يملأ: expiry_premium / expiry_premium_source / option_pnl_expiry_pct
    ويحدّث hold_expiry_pct (سهم) بعد اكتمال يوم الانتهاء.
    """
    if outcomes_df is None or outcomes_df.empty:
        return outcomes_df

    for col in ("expiry_premium", "expiry_premium_source", "option_pnl_expiry_pct"):
        if col not in outcomes_df.columns:
            outcomes_df[col] = None

    if cache is None:
        cache = _OPTION_PRICE_CACHE
    if hist_cache is None:
        hist_cache = {}

    today = datetime.now().date()
    market_n = intrinsic_n = estimate_n = skip_n = fail_n = 0
    print("📥 متابعة أسعار العقود حتى الانتهاء (يومياً)...")

    for idx, row in outcomes_df.iterrows():
        expiry_d = parse_date(row.get("expiry"))
        if expiry_d is None:
            fail_n += 1
            continue

        # لا نحدّث قبل يوم التوصية / بدون بيانات عقد
        rec_d = parse_date(row.get("date"))
        asof = min(today, expiry_d)
        if rec_d and asof < rec_d:
            skip_n += 1
            continue

        ticker = str(row.get("ticker") or "").strip()
        strike = row.get("strike")
        entry = float(row.get("entry_stock") or 0)
        tp1 = float(row.get("tp1_stock") or 0)
        stop = float(row.get("stop_stock") or 0)
        is_call = resolve_is_call(row.get("direction"), entry, tp1, stop)
        expired = today >= expiry_d

        try:
            prem = float(row.get("premium") or 0)
        except (TypeError, ValueError):
            prem = 0.0

        hist = hist_cache.get(ticker)
        stock_asof = (
            hist_close_on_or_before(hist, asof)
            if hist is not None and not getattr(hist, "empty", True)
            else None
        )

        # حركة السهم لين الانتهاء — فقط بعد اكتمال يوم الـ expiry
        if expired and stock_asof is not None and entry > 0:
            outcomes_df.at[idx, "hold_expiry_pct"] = move_pct(entry, stock_asof, is_call)

        mkt = market_option_close(ticker, row.get("expiry"), strike, is_call, asof, cache)
        if mkt is not None:
            outcomes_df.at[idx, "expiry_premium"] = mkt
            outcomes_df.at[idx, "expiry_premium_source"] = "market"
            if prem > 0:
                outcomes_df.at[idx, "option_pnl_expiry_pct"] = round((mkt - prem) / prem * 100, 2)
            market_n += 1
            continue

        if stock_asof is None or stock_asof <= 0:
            fail_n += 1
            continue

        est = calc_option_exit_price(row, stock_asof, asof)
        if est is None:
            fail_n += 1
            continue
        # عند/بعد الانتهاء بدون سوق → intrinsic؛ قبلها تقدير BS
        src = "intrinsic" if expired else "estimate"
        outcomes_df.at[idx, "expiry_premium"] = est
        outcomes_df.at[idx, "expiry_premium_source"] = src
        if prem > 0:
            outcomes_df.at[idx, "option_pnl_expiry_pct"] = round((est - prem) / prem * 100, 2)
        if src == "intrinsic":
            intrinsic_n += 1
        else:
            estimate_n += 1

    print(
        f"✅ عقد→انتهاء: سوق={market_n} | ذاتي={intrinsic_n} | "
        f"تقدير={estimate_n} | تخطي={skip_n} | فشل={fail_n}"
    )
    return outcomes_df


def apply_data_quality(outcomes_df):
    """يملأ data_quality / data_quality_note لكل الصفوف."""
    if outcomes_df is None or outcomes_df.empty:
        return outcomes_df
    if "data_quality" not in outcomes_df.columns:
        outcomes_df["data_quality"] = ""
    if "data_quality_note" not in outcomes_df.columns:
        outcomes_df["data_quality_note"] = ""

    counts = {"reliable": 0, "partial": 0, "unreliable": 0, "open": 0}
    for idx, row in outcomes_df.iterrows():
        q, note = classify_data_quality(row)
        outcomes_df.at[idx, "data_quality"] = q
        outcomes_df.at[idx, "data_quality_note"] = note
        counts[q] = counts.get(q, 0) + 1

    print(
        f"✅ جودة البيانات: reliable={counts.get('reliable',0)} | "
        f"partial={counts.get('partial',0)} | "
        f"unreliable={counts.get('unreliable',0)} | "
        f"open={counts.get('open',0)}"
    )
    return outcomes_df


def print_summary(outcomes_df):
    closed = outcomes_df[outcomes_df["status"] != "open"]
    if closed.empty:
        print("\nلا توجد نتائج مغلقة بعد")
        return

    reliable = closed
    if "data_quality" in closed.columns:
        reliable = closed[closed["data_quality"].astype(str).str.lower() == "reliable"]

    print(f"\n📊 ملخص الاعتماد (reliable فقط): {len(reliable)}/{len(closed)} مغلق")
    if reliable.empty:
        print("   لا صفوف reliable بعد — راجع data_quality_note")
    else:
        with_opt = reliable[reliable["option_pnl_pct"].notna()]
        if not with_opt.empty:
            opt_wins = with_opt[with_opt["option_pnl_pct"] > 0]
            opt_loss = with_opt[with_opt["option_pnl_pct"] <= 0]
            opt_wr = len(opt_wins) / len(with_opt) * 100
            avg_opt_win = float(opt_wins["option_pnl_pct"].mean()) if not opt_wins.empty else 0
            avg_opt_loss = float(opt_loss["option_pnl_pct"].mean()) if not opt_loss.empty else 0
            print(f"   Win Rate عقد : {opt_wr:.1f}%")
            print(f"   Avg Win عقد  : {avg_opt_win:+.1f}%")
            print(f"   Avg Loss عقد : {avg_opt_loss:+.1f}%")
        else:
            wins = reliable[reliable["status"].isin(["tp1_hit", "tp2_hit", "tp3_hit"])]
            stops = reliable[reliable["status"] == "stop_hit"]
            win_rate = len(wins) / len(reliable) * 100
            avg_win = float(wins["result_pct"].mean()) if not wins.empty else 0
            avg_loss = float(stops["result_pct"].mean()) if not stops.empty else 0
            print(f"   Win Rate سهم : {win_rate:.1f}%")
            print(f"   Avg Win سهم  : {avg_win:+.2f}%")
            print(f"   Avg Loss سهم : {avg_loss:+.2f}%")

    path = outcomes_df.dropna(subset=["mfe_pct"], how="all") if "mfe_pct" in outcomes_df.columns else pd.DataFrame()
    if not path.empty:
        dte = pd.to_numeric(path.get("days_to_entry"), errors="coerce")
        held = pd.to_numeric(path.get("days_held"), errors="coerce")
        mfe = pd.to_numeric(path.get("mfe_pct"), errors="coerce")
        mae = pd.to_numeric(path.get("mae_pct"), errors="coerce")
        hx = pd.to_numeric(path.get("hold_expiry_pct"), errors="coerce")
        print(f"\n⏱ مقاييس فترة المراقبة ({len(path)} صف):")
        if dte.notna().any():
            print(f"   Avg days→Entry : {dte.mean():.1f}")
        if held.notna().any():
            print(f"   Avg days held  : {held.mean():.1f}")
        if mfe.notna().any():
            print(f"   Avg MFE        : {mfe.mean():+.2f}%")
        if mae.notna().any():
            print(f"   Avg MAE        : {mae.mean():+.2f}%")
        if hx.notna().any():
            print(f"   Avg hold→expiry سهم: {hx.mean():+.2f}%  ({hx.notna().sum()} صف)")
    if "option_pnl_expiry_pct" in outcomes_df.columns:
        ox = pd.to_numeric(outcomes_df["option_pnl_expiry_pct"], errors="coerce")
        if ox.notna().any():
            print(f"   Avg عقد→انتهاء : {ox.mean():+.1f}%  ({ox.notna().sum()} صف)")


if __name__ == "__main__":
    eod = os.environ.get("EOD_RUN", "").lower() in ("1", "true", "yes")
    label = "EOD — بعد إغلاق السوق" if eod else "تحديث سجل النتائج"
    print(f"🔍 {label}\n")

    outcomes = load_outcomes()
    if os.environ.get("REPAIR_CLOSED_OPTIONS", "").lower() in ("1", "true", "yes"):
        outcomes = repair_closed_option_estimates(outcomes)
    outcomes = backfill_option_fields(outcomes)
    outcomes = recalculate_all_outcomes(outcomes)
    if not eod:
        outcomes = add_new_recommendations(outcomes)
    else:
        print("  (EOD: تحديث النتائج فقط — بدون إضافة BUY جديدة)")

    print("📥 جلب أسعار يومية...")
    hist_cache = build_hist_cache(outcomes)
    outcomes = repair_preexisting_level_hits(outcomes, hist_cache=hist_cache)
    outcomes = update_open_outcomes(outcomes, hist_cache=hist_cache)
    outcomes = enrich_path_metrics(outcomes, hist_cache=hist_cache)
    outcomes = enrich_market_exit_premiums(outcomes)
    outcomes = enrich_expiry_option_premiums(outcomes, hist_cache=hist_cache)
    outcomes = apply_data_quality(outcomes)

    # حفظ بترتيب الأعمدة القياسي
    for col in OUTCOMES_COLS:
        if col not in outcomes.columns:
            outcomes[col] = None
    outcomes = outcomes[[c for c in OUTCOMES_COLS if c in outcomes.columns]]
    outcomes.to_csv(OUTCOMES_FILE, index=False)
    print_summary(outcomes)

    print("\n✅ تم حفظ outcomes.csv")
