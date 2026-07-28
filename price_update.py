"""
price_update.py — تحديث سريع كل 5 دقايق خلال ساعات السوق
======================================================
يقرأ options_v3_results.csv ويعيد تحليل نفس الأسهم بدون Finviz
أسرع بكثير من السكريبت الكامل (2-4 دقائق بدل 10-15)
"""

import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

import time
import pandas as pd
import yfinance as yf
from datetime import datetime

# ── استيراد كل المنطق من السكريبت الرئيسي (main() لن يشتغل بسبب __name__ guard) ──
from cheap_options_screener_v3 import (
    _YF_SESSION,
    fetch_options_data,
    score_stock,
    compute_technicals,
    compute_trade_plan,
    parse_num,
    save_results_csv,
    filter_results_df,
    DTE_TARGET, DTE_WINDOW,
    TARGET_PREM_MIN, TARGET_PREM_MAX,
    MIN_OI, MAX_SPREAD_PCT, DELAY_BETWEEN,
)

CSV_FILE = "options_v3_results.csv"


def get_live_price_and_vol(ticker):
    """جيب السعر الحالي + حجم اليوم من Yahoo Finance"""
    try:
        yt   = yf.Ticker(ticker, session=_YF_SESSION)
        info = yt.info or {}
        price = (info.get("currentPrice")
                 or info.get("regularMarketPrice")
                 or info.get("previousClose"))
        vol   = (info.get("volume")
                 or info.get("regularMarketVolume")
                 or 0)
        return float(price) if price else None, int(vol) if vol else 0
    except Exception:
        return None, 0


def compute_atr_pct(hist, price):
    """احسب ATR% من التاريخ اليومي"""
    if hist is None or hist.empty or price <= 0:
        return 0.0
    highs  = hist["High"].astype(float).tolist()
    lows   = hist["Low"].astype(float).tolist()
    closes = hist["Close"].astype(float).tolist()
    n = len(closes)
    if n < 2:
        return 0.0
    trs = []
    for i in range(max(1, n - 14), n):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i]  - closes[i - 1]))
        trs.append(tr)
    atr14 = sum(trs) / len(trs) if trs else 0
    return round(atr14 / price * 100, 2)


def main():
    start = datetime.now()
    print(f"\n{'='*65}")
    print(f"  PRICE UPDATE -- {start.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*65}\n")

    # اقرأ الـ CSV الحالي
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"  ERROR: {CSV_FILE} not found -- run main screener first")
        return

    tickers = df["Ticker"].tolist()
    print(f"  Updating {len(tickers)} tickers...\n")
    print(f"   {'Ticker':<8} {'Price':>8}  {'Prem':>7}  {'OI':>7}  {'Rec':<6}  {'Conf':>5}  {'Score':>5}")
    print(f"   {'-'*65}")

    rows = []
    for ticker in tickers:

        # السعر الحالي + حجم اليوم
        price_num, today_vol = get_live_price_and_vol(ticker)

        if not price_num or price_num <= 0:
            old = df[df["Ticker"] == ticker]
            if not old.empty:
                price_str = str(old["Price"].values[0]).replace("$", "")
                price_num = parse_num(price_str) or 0
            if price_num <= 0:
                print(f"   {ticker:<8}  -- no price")
                continue

        opts = fetch_options_data(ticker, price_num)

        if today_vol <= 0:
            today_vol = int(opts.get("today_vol_yf") or 0)

        old         = df[df["Ticker"] == ticker]
        atr_pct_old = float(old["atr_pct"].values[0]) if not old.empty and "atr_pct" in old.columns else 0
        gap_pct_old = float(old["gap_pct"].values[0]) if not old.empty and "gap_pct" in old.columns else 0
        avg_vol_old = float(old["avg_vol"].values[0]) if not old.empty and "avg_vol" in old.columns else 0
        rsi_old     = float(old["rsi"].values[0]) if not old.empty and "rsi" in old.columns else 50

        hist    = opts.get("hist")
        atr_pct = compute_atr_pct(hist, price_num) or atr_pct_old
        tech_early = compute_technicals(hist, price_num) if hist is not None and not hist.empty else {}

        row = {
            "Ticker":       ticker,
            "Price":        f"${price_num:.2f}",
            "price_num":    price_num,
            "rsi":          tech_early.get("rsi", rsi_old),
            "atr_pct":      atr_pct,
            "gap_pct":      gap_pct_old,
            "today_vol":    today_vol,
            "avg_vol":      opts["avg_vol"] or avg_vol_old,
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
        }

        score, notes = score_stock(row)
        row["Score"] = score
        row["Notes"] = " | ".join(notes)

        tech = compute_technicals(hist, price_num)
        plan = compute_trade_plan(row, tech)

        row["recommendation"] = plan.get("recommendation", "AVOID")
        row["confidence"]     = plan.get("confidence", 0)
        row["entry_stock"]    = plan.get("entry_stock")
        row["stop_stock"]     = plan.get("stop_stock")
        row["tp1_stock"]      = plan.get("tp1_stock")
        row["tp2_stock"]      = plan.get("tp2_stock")
        row["tp3_stock"]      = plan.get("tp3_stock")

        rec    = row["recommendation"]
        conf   = row["confidence"]
        prem_s = f"${row['premium']:.2f}" if row["premium"] is not None else "  N/A"
        oi_s   = f"{row['oi']:,}"         if row["oi"]     is not None else "  N/A"
        err_s  = f"  [{row['_error']}]"   if row["_error"] else ""
        tag    = {"BUY": "[BUY]", "WAIT": "[WAIT]", "AVOID": "[AVOID]"}.get(rec, "[ ? ]")

        print(f"   {ticker:<8} ${price_num:>7.2f}  {prem_s:>7}  {oi_s:>7}  "
              f"{tag:<7}  {conf:>4}%  {score:>5}{err_s}")

        rows.append(row)
        time.sleep(DELAY_BETWEEN)

    if not rows:
        print("\n  No data to save.")
        return

    # حفظ CSV
    result_df = (pd.DataFrame(rows)
                   .sort_values("Score", ascending=False)
                   .reset_index(drop=True))

    save_cols = [
        "Ticker", "Price", "premium", "strike", "iv", "oi", "opt_vol",
        "spread_pct", "atr_pct", "gap_pct", "RVOL", "pm_high", "pm_low",
        "pm_volume", "expiry", "direction", "earnings", "float_shares",
        "avg_vol", "Score", "recommendation", "confidence", "Notes",
        "entry_stock", "stop_stock", "tp1_stock", "tp2_stock", "tp3_stock"
    ]
    save_cols = [c for c in save_cols if c in result_df.columns]
    filtered_df = filter_results_df(result_df)
    if save_results_csv(filtered_df[save_cols], CSV_FILE):
        print(f"  Saved: {CSV_FILE} ({len(result_df)} → {len(filtered_df)} rows)")
    else:
        print(f"  ⚠️  No rows passed filter — kept previous {CSV_FILE}")

    elapsed     = (datetime.now() - start).seconds
    buy_count   = sum(1 for r in rows if r["recommendation"] == "BUY")
    wait_count  = sum(1 for r in rows if r["recommendation"] == "WAIT")
    avoid_count = len(rows) - buy_count - wait_count

    print(f"\n{'-'*65}")
    print(f"  Done: {len(rows)} tickers updated in {elapsed}s")
    print(f"  BUY: {buy_count}  |  WAIT: {wait_count}  |  AVOID: {avoid_count}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
