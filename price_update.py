import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
import time, pandas as pd, yfinance as yf
from datetime import datetime
from cheap_options_screener_v3 import (
    _YF_SESSION, fetch_options_data, score_stock,
    compute_technicals, compute_trade_plan, parse_num,
    DELAY_BETWEEN,
)
CSV_FILE = "options_v3_results.csv"

def get_live_price(ticker):
    try:
        info = yf.Ticker(ticker, session=_YF_SESSION).info or {}
        p = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        v = info.get("volume") or info.get("regularMarketVolume") or 0
        return float(p) if p else None, int(v)
    except: return None, 0

def atr_pct(hist, price):
    if hist is None or hist.empty or price <= 0: return 0.0
    H,L,C = hist["High"].tolist(), hist["Low"].tolist(), hist["Close"].tolist()
    n = len(C)
    trs = [max(H[i]-L[i], abs(H[i]-C[i-1]), abs(L[i]-C[i-1])) for i in range(max(1,n-14),n)]
    return round((sum(trs)/len(trs))/price*100, 2) if trs else 0.0

def main():
    start = datetime.now()
    print(f"\n{'='*60}\n  PRICE UPDATE -- {start.strftime('%Y-%m-%d %H:%M UTC')}\n{'='*60}\n")
    try: df = pd.read_csv(CSV_FILE)
    except FileNotFoundError: print("ERROR: run main screener first"); return
    rows = []
    for ticker in df["Ticker"].tolist():
        price, today_vol = get_live_price(ticker)
        if not price:
            old = df[df["Ticker"]==ticker]
            price = parse_num(str(old["Price"].values[0]).replace("$","")) if not old.empty else 0
        if not price: continue
        opts = fetch_options_data(ticker, price)
        old = df[df["Ticker"]==ticker]
        def ov(col): return float(old[col].values[0]) if not old.empty and col in old.columns else 0
        hist = opts.get("hist")
        row = {
            "Ticker": ticker, "Price": f"${price:.2f}", "price_num": price,
            "atr_pct": atr_pct(hist, price) or ov("atr_pct"),
            "gap_pct": ov("gap_pct"), "today_vol": today_vol,
            "avg_vol": opts["avg_vol"] or ov("avg_vol"),
            "float_shares": opts["float_shares"], "earnings": opts["earnings"],
            "premium": opts["premium"], "iv": opts["iv"], "oi": opts["oi"],
            "opt_vol": opts["opt_vol"], "spread_pct": opts["spread_pct"],
            "expiry": opts["expiry"], "has_weekly": opts["has_weekly"],
            "direction": opts["direction"], "strike": opts["strike"],
            "dte_num": opts["dte_num"], "_hist": hist,
            "pm_high": opts["pm_high"], "pm_low": opts["pm_low"],
            "pm_last": opts["pm_last"], "pm_volume": opts["pm_volume"],
            "_error": opts["error"],
        }
        score, notes = score_stock(row)
        row["Score"] = score; row["Notes"] = " | ".join(notes)
        tech = compute_technicals(hist, price)
        plan = compute_trade_plan(row, tech)
        row["recommendation"] = plan.get("recommendation","AVOID")
        row["confidence"] = plan.get("confidence",0)
        tag = {"BUY":"[BUY]","WAIT":"[WAIT]","AVOID":"[AVOID]"}.get(row["recommendation"],"[?]")
        prem = f"${row['premium']:.2f}" if row["premium"] else "N/A"
        print(f"   {ticker:<8} ${price:>7.2f}  {prem:>7}  {tag:<7}  {row['confidence']}%  {score}")
        rows.append(row)
        time.sleep(DELAY_BETWEEN)
    if not rows: return
    result = pd.DataFrame(rows).sort_values("Score",ascending=False).reset_index(drop=True)
    cols = ["Ticker","Price","premium","strike","iv","oi","opt_vol","spread_pct",
            "atr_pct","gap_pct","RVOL","pm_high","pm_low","pm_volume","expiry",
            "direction","earnings","float_shares","avg_vol","Score","recommendation","confidence","Notes"]
    result[[c for c in cols if c in result.columns]].to_csv(CSV_FILE, index=False)
    elapsed = (datetime.now()-start).seconds
    b = sum(1 for r in rows if r["recommendation"]=="BUY")
    w = sum(1 for r in rows if r["recommendation"]=="WAIT")
    print(f"\n  Done in {elapsed}s | BUY:{b} WAIT:{w} AVOID:{len(rows)-b-w}")
    print(f"  Saved: {CSV_FILE}\n{'='*60}\n")

if __name__ == "__main__": main()
