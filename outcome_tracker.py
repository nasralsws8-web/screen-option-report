"""
outcome_tracker.py
------------------
يسجّل توصيات BUY الجديدة في outcomes.csv
ويتحقق من النتائج للتوصيات المفتوحة (هل وصل Entry/TP1/TP2/TP3 أو ضرب Stop؟)
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

OUTCOMES_FILE = "outcomes.csv"
RESULTS_FILE  = "options_v3_results.csv"

OUTCOMES_COLS = [
    "date", "ticker", "direction", "score", "confidence",
    "price_at_rec", "entry_stock", "stop_stock",
    "tp1_stock", "tp2_stock", "tp3_stock", "expiry",
    "entry_hit", "entry_hit_date",
    "status",   # open / tp1_hit / tp2_hit / tp3_hit / stop_hit / expired
    "result_pct"
]


def valid_target(price):
    """رفض أهداف سالبة أو صفر (مثل SNXX tp3=-3.78)."""
    try:
        return float(price) > 0
    except (TypeError, ValueError):
        return False


def resolve_is_call(direction, entry, tp1, stop):
    """CALL/PUT/NEUTRAL — NEUTRAL يُستنتج من entry/tp1/stop."""
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


def recalculate_all_outcomes(outcomes_df):
    fixed = 0
    for idx, row in outcomes_df.iterrows():
        if row.get("status") == "open":
            continue
        old = row.get("result_pct")
        new = recalculate_result_pct(row)
        if new is not None and old != new:
            outcomes_df.at[idx, "result_pct"] = new
            fixed += 1
    if fixed:
        print(f"✅ أعيد حساب result_pct لـ {fixed} صف تاريخي")
    return outcomes_df


def move_pct(entry, target, is_call):
    """نسبة ربح/خسارة موجّهة حسب اتجاه الصفقة."""
    if entry <= 0:
        return 0.0
    if is_call:
        return round((target - entry) / entry * 100, 2)
    return round((entry - target) / entry * 100, 2)


# ─────────────────────────────────────────────
# 1) تحميل أو إنشاء outcomes.csv
# ─────────────────────────────────────────────
def load_outcomes():
    STR_COLS = ["date", "ticker", "direction", "expiry", "entry_hit_date", "status"]
    if os.path.exists(OUTCOMES_FILE):
        df = pd.read_csv(OUTCOMES_FILE)
        for col in OUTCOMES_COLS:
            if col not in df.columns:
                df[col] = None
        for col in STR_COLS:
            if col in df.columns:
                df[col] = df[col].astype(object).where(df[col].notna(), None)
        return df
    return pd.DataFrame(columns=OUTCOMES_COLS)


# ─────────────────────────────────────────────
# 2) إضافة توصيات BUY الجديدة
# ─────────────────────────────────────────────
def add_new_recommendations(outcomes_df):
    if not os.path.exists(RESULTS_FILE):
        print("⚠️  options_v3_results.csv غير موجود — تخطي إضافة توصيات جديدة")
        return outcomes_df

    results = pd.read_csv(RESULTS_FILE)
    if results.empty or "recommendation" not in results.columns:
        print("⚠️  options_v3_results.csv فاضي — تخطي إضافة توصيات جديدة")
        return outcomes_df

    buy_rows = results[results["recommendation"] == "BUY"].copy()
    today = datetime.now().strftime("%Y-%m-%d")
    added = 0

    for _, r in buy_rows.iterrows():
        ticker = str(r.get("Ticker", "")).strip()
        if not ticker:
            continue

        open_same = outcomes_df[
            (outcomes_df["ticker"] == ticker) &
            (outcomes_df["status"] == "open")
        ]
        if not open_same.empty:
            continue

        already = outcomes_df[
            (outcomes_df["ticker"] == ticker) &
            (outcomes_df["date"] == today)
        ]
        if not already.empty:
            continue

        entry = float(r.get("entry_stock") or 0)
        if entry <= 0:
            continue

        new_row = {
            "date":          today,
            "ticker":        ticker,
            "direction":     str(r.get("direction", "")).replace("📈", "").replace("📉", "").strip(),
            "score":         r.get("Score", 0),
            "confidence":    r.get("confidence", 0),
            "price_at_rec":  r.get("Price", 0),
            "entry_stock":   entry,
            "stop_stock":    float(r.get("stop_stock") or 0),
            "tp1_stock":     float(r.get("tp1_stock") or 0),
            "tp2_stock":     float(r.get("tp2_stock") or 0),
            "tp3_stock":     float(r.get("tp3_stock") or 0),
            "expiry":        r.get("expiry", ""),
            "entry_hit":     False,
            "entry_hit_date": "",
            "status":        "open",
            "result_pct":    None,
        }
        outcomes_df = pd.concat(
            [outcomes_df, pd.DataFrame([new_row])],
            ignore_index=True
        )
        added += 1

    print(f"✅ أضفت {added} توصية جديدة")
    return outcomes_df


# ─────────────────────────────────────────────
# 3) تحديث نتائج التوصيات المفتوحة
# ─────────────────────────────────────────────
def update_open_outcomes(outcomes_df):
    open_mask = outcomes_df["status"] == "open"
    open_rows = outcomes_df[open_mask]

    if open_rows.empty:
        print("لا توجد توصيات مفتوحة للتحقق")
        return outcomes_df

    today = datetime.now().date()

    for idx, row in open_rows.iterrows():
        ticker     = str(row["ticker"]).strip()
        rec_date   = pd.to_datetime(row["date"]).date()
        entry      = float(row["entry_stock"] or 0)
        stop       = float(row["stop_stock"] or 0)
        tp1        = float(row["tp1_stock"] or 0)
        tp2        = float(row["tp2_stock"] or 0)
        tp3        = float(row["tp3_stock"] or 0)
        expiry_str = str(row.get("expiry", ""))
        is_call    = resolve_is_call(
            row.get("direction"), entry, tp1, stop
        )

        if not ticker or entry <= 0:
            continue

        try:
            expiry_date = pd.to_datetime(expiry_str).date()
        except Exception:
            expiry_date = today + timedelta(days=7)

        try:
            start = rec_date.strftime("%Y-%m-%d")
            # +2 أيام لضمان اكتمال شمعة اليوم بعد إغلاق السوق
            end   = (today + timedelta(days=2)).strftime("%Y-%m-%d")
            hist  = yf.download(ticker, start=start, end=end,
                                interval="1d", progress=False, auto_adjust=True)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
        except Exception as e:
            print(f"⚠️  {ticker}: خطأ في جلب البيانات — {e}")
            continue

        if hist.empty:
            continue

        entry_hit      = bool(row.get("entry_hit", False))
        entry_hit_date = row.get("entry_hit_date")

        if not entry_hit:
            if is_call:
                hit_days = hist[hist["High"] >= entry]
            else:
                hit_days = hist[hist["Low"] <= entry]
            if not hit_days.empty:
                entry_hit      = True
                entry_hit_date = hit_days.index[0].strftime("%Y-%m-%d")
                outcomes_df.at[idx, "entry_hit"]      = True
                outcomes_df.at[idx, "entry_hit_date"] = entry_hit_date

        if entry_hit and entry_hit_date:
            post = hist[hist.index >= pd.to_datetime(entry_hit_date)]
            if not post.empty:
                max_high = float(post["High"].max())
                min_low  = float(post["Low"].min())

                if is_call:
                    if valid_target(tp3) and max_high >= tp3:
                        outcomes_df.at[idx, "status"]     = "tp3_hit"
                        outcomes_df.at[idx, "result_pct"] = move_pct(entry, tp3, True)
                    elif valid_target(tp2) and max_high >= tp2:
                        outcomes_df.at[idx, "status"]     = "tp2_hit"
                        outcomes_df.at[idx, "result_pct"] = move_pct(entry, tp2, True)
                    elif valid_target(tp1) and max_high >= tp1:
                        outcomes_df.at[idx, "status"]     = "tp1_hit"
                        outcomes_df.at[idx, "result_pct"] = move_pct(entry, tp1, True)
                    elif valid_target(stop) and min_low <= stop:
                        outcomes_df.at[idx, "status"]     = "stop_hit"
                        outcomes_df.at[idx, "result_pct"] = move_pct(entry, stop, True)
                else:
                    if valid_target(tp3) and min_low <= tp3:
                        outcomes_df.at[idx, "status"]     = "tp3_hit"
                        outcomes_df.at[idx, "result_pct"] = move_pct(entry, tp3, False)
                    elif valid_target(tp2) and min_low <= tp2:
                        outcomes_df.at[idx, "status"]     = "tp2_hit"
                        outcomes_df.at[idx, "result_pct"] = move_pct(entry, tp2, False)
                    elif valid_target(tp1) and min_low <= tp1:
                        outcomes_df.at[idx, "status"]     = "tp1_hit"
                        outcomes_df.at[idx, "result_pct"] = move_pct(entry, tp1, False)
                    elif valid_target(stop) and max_high >= stop:
                        outcomes_df.at[idx, "status"]     = "stop_hit"
                        outcomes_df.at[idx, "result_pct"] = move_pct(entry, stop, False)

        if today > expiry_date and outcomes_df.at[idx, "status"] == "open":
            last_close = float(hist["Close"].iloc[-1])
            outcomes_df.at[idx, "status"]     = "expired"
            outcomes_df.at[idx, "result_pct"] = move_pct(entry, last_close, is_call)

        print(f"  {ticker}: {outcomes_df.at[idx, 'status']}")

    return outcomes_df


# ─────────────────────────────────────────────
# 4) طباعة ملخص
# ─────────────────────────────────────────────
def print_summary(outcomes_df):
    closed = outcomes_df[outcomes_df["status"] != "open"]
    if closed.empty:
        print("\nلا توجد نتائج مغلقة بعد")
        return

    wins  = closed[closed["status"].isin(["tp1_hit", "tp2_hit", "tp3_hit"])]
    stops = closed[closed["status"] == "stop_hit"]

    win_rate = len(wins) / len(closed) * 100 if len(closed) > 0 else 0
    avg_win  = float(wins["result_pct"].mean())  if not wins.empty  else 0
    avg_loss = float(stops["result_pct"].mean()) if not stops.empty else 0

    print(f"\n📊 ملخص النتائج:")
    print(f"   إجمالي مغلق : {len(closed)}")
    print(f"   Win Rate    : {win_rate:.1f}%")
    print(f"   Avg Win     : {avg_win:+.2f}%")
    print(f"   Avg Loss    : {avg_loss:+.2f}%")


if __name__ == "__main__":
    eod = os.environ.get("EOD_RUN", "").lower() in ("1", "true", "yes")
    label = "EOD — بعد إغلاق السوق" if eod else "تحديث سجل النتائج"
    print(f"🔍 {label}\n")

    outcomes = load_outcomes()
    outcomes = recalculate_all_outcomes(outcomes)
    if not eod:
        outcomes = add_new_recommendations(outcomes)
    else:
        print("  (EOD: تحديث النتائج فقط — بدون إضافة BUY جديدة)")
    outcomes = update_open_outcomes(outcomes)
    outcomes.to_csv(OUTCOMES_FILE, index=False)
    print_summary(outcomes)

    print("\n✅ تم حفظ outcomes.csv")
