"""
telegram_notify.py
------------------
يرسل إشعار Telegram لكل توصية BUY جديدة في options_v3_results.csv
"""

import os
import urllib.request
import urllib.parse
import json
import pandas as pd

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
CSV_FILE         = "options_v3_results.csv"


def send_message(text: str):
    """يرسل رسالة نصية لـ Telegram"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  TELEGRAM_TOKEN أو TELEGRAM_CHAT_ID غير موجود")
        return False

    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "HTML",
    }).encode()

    try:
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except Exception as e:
        print(f"❌ خطأ في إرسال Telegram: {e}")
        return False


def format_buy_message(row) -> str:
    """شكّل رسالة BUY"""
    ticker    = row.get("Ticker", "N/A")
    price     = row.get("Price", "N/A")
    direction = str(row.get("direction", "")).replace("📈","").replace("📉","").strip()
    entry     = row.get("entry_stock", "N/A")
    stop      = row.get("stop_stock",  "N/A")
    tp1       = row.get("tp1_stock",   "N/A")
    tp2       = row.get("tp2_stock",   "N/A")
    tp3       = row.get("tp3_stock",   "N/A")
    premium   = row.get("premium",     "N/A")
    oi        = row.get("oi",          "N/A")
    expiry    = row.get("expiry",      "N/A")
    conf      = row.get("confidence",  "N/A")
    score     = row.get("Score",       "N/A")

    def fmt(v):
        try:    return f"${float(v):.2f}"
        except: return str(v)

    def fmt_oi(v):
        try:    return f"{int(float(v)):,}"
        except: return str(v)

    arrow = "📈" if str(direction).upper() == "CALL" else "📉"

    msg = (
        f"🟢 <b>BUY — {ticker}</b> {arrow}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💰 السعر الحالي: <b>{price}</b>\n"
        f"🎯 Entry:  <b>{fmt(entry)}</b>\n"
        f"🛑 Stop:   <b>{fmt(stop)}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"✅ TP1: <b>{fmt(tp1)}</b>\n"
        f"✅ TP2: <b>{fmt(tp2)}</b>\n"
        f"✅ TP3: <b>{fmt(tp3)}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📊 Premium: {fmt(premium)}  |  OI: {fmt_oi(oi)}\n"
        f"📅 Expiry: {expiry}\n"
        f"🔥 Score: {score}  |  Confidence: {conf}%\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"⚠️ <i>للأغراض التعليمية فقط — ليس توصية مالية</i>"
    )
    return msg


def main():
    if not os.path.exists(CSV_FILE):
        print(f"⚠️  {CSV_FILE} غير موجود")
        return

    df = pd.read_csv(CSV_FILE)

    buy_rows = df[df["recommendation"] == "BUY"]

    if buy_rows.empty:
        print("لا توجد توصيات BUY اليوم — لم يُرسل شيء")
        return

    print(f"📤 إرسال {len(buy_rows)} توصية BUY لـ Telegram...")

    # رسالة ملخص أولاً
    summary = (
        f"📋 <b>Options Screener — تقرير اليوم</b>\n"
        f"🟢 BUY: {len(buy_rows)}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"التفاصيل تليها ↓"
    )
    send_message(summary)

    # رسالة لكل BUY
    for _, row in buy_rows.iterrows():
        msg = format_buy_message(row)
        ok  = send_message(msg)
        ticker = row.get("Ticker", "?")
        print(f"  {'✅' if ok else '❌'} {ticker}")


if __name__ == "__main__":
    main()
