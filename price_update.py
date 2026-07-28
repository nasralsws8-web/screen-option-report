"""
price_update.py — فحص دوري خلال ساعات السوق
============================================
كل run يبحث من جديد:
  • Finviz (فلترة تلقائية)
  • manual_tickers.txt (قائمة يدوية)
  • أو الاثنين معاً (افتراضي: both)

لا يعتمد على CSV القديم — يكتشف أسهم جديدة كل مرة.
"""

import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

from datetime import datetime

from cheap_options_screener_v3 import (
    fetch_candidates,
    process_candidates,
    save_screen_results,
    get_screener_source,
)

CSV_FILE = "options_v3_results.csv"


def main():
    start = datetime.now()
    print(f"\n{'='*65}")
    print(f"  MARKET SCAN -- {start.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Source: {get_screener_source()} (Finviz + manual)")
    print(f"{'='*65}\n")

    candidates = fetch_candidates()
    if candidates.empty:
        print("  No candidates from Finviz or manual list.")
        return

    result_df = process_candidates(candidates, show_progress=True)
    if result_df.empty:
        print("\n  No option data returned.")
        return

    ok, filtered_df, total, _fb = save_screen_results(result_df, CSV_FILE)
    if ok:
        print(f"\n  Saved: {CSV_FILE} ({total} → {len(filtered_df)} rows)")
    else:
        print(f"\n  ⚠️  No rows passed filter — kept previous {CSV_FILE}")

    elapsed     = (datetime.now() - start).seconds
    buy_count   = int((result_df["recommendation"] == "BUY").sum())
    wait_count  = int((result_df["recommendation"] == "WAIT").sum())
    avoid_count = len(result_df) - buy_count - wait_count

    print(f"\n{'-'*65}")
    print(f"  Done: {len(result_df)} tickers scanned in {elapsed}s")
    print(f"  BUY: {buy_count}  |  WAIT: {wait_count}  |  AVOID: {avoid_count}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
