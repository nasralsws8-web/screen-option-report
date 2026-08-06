"""
telegram_notify.py
------------------
Sends each BUY signal as a styled PNG card + text caption to Telegram.

Dedupe: عقد + يوم تقويم ET — لا إعادة إرسال نفس الإشارة في نفس اليوم.
الحالة تُحفظ في telegram_sent.json وتُرفع مع نتائج الـ workflow.
"""

import os
import io
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
CSV_FILE         = "options_v3_results.csv"
SENT_FILE        = os.environ.get("TELEGRAM_SENT_FILE", "telegram_sent.json")
SENT_KEEP_DAYS   = 21
MARKET_TZ        = ZoneInfo("America/New_York")

# ── Palette ──────────────────────────────────────────────────────
BG          = (10,  13,  20)
CARD        = (18,  22,  30)

# Header colors per direction
HDR_CALL    = (8,   45,  22)     # dark green
HDR_PUT     = (55,  12,  12)     # dark red
HDR_DEFAULT = (8,   22,  55)     # dark blue (fallback)

# Accent colors
GREEN_BRIGHT = (0,   220, 80)
RED_BRIGHT   = (255, 60,  70)
BLUE_BRIGHT  = (60,  160, 255)
GOLD         = (255, 200, 0)
WHITE        = (235, 240, 255)
LGRAY        = (110, 125, 145)
DGRAY        = (26,  32,  44)
BORDER       = (38,  48,  65)
DIVIDER      = (32,  42,  56)


def _fonts():
    bold = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    reg = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    def load(paths, size):
        for p in paths:
            try: return ImageFont.truetype(p, size)
            except: pass
        return ImageFont.load_default()
    return {
        "h1":  load(bold, 38),
        "h2":  load(bold, 26),
        "h3":  load(bold, 21),
        "dir": load(bold, 22),
        "sub": load(reg,  15),
        "xs":  load(reg,  12),
    }

F = _fonts()


def fmt(v):
    try:    return f"${float(v):.2f}"
    except: return str(v)

def fmt_plain(v):
    try:    return f"{float(v):.2f}"
    except: return str(v)

def fmt_oi(v):
    try:    return f"{int(float(v)):,}"
    except: return str(v)


def normalize_direction(direction) -> str:
    d = str(direction or "").replace("📈", "").replace("📉", "").strip().upper()
    if "PUT" in d:
        return "PUT"
    if "CALL" in d:
        return "CALL"
    return d


def trading_day_et(now=None) -> str:
    now = now or datetime.now(MARKET_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=MARKET_TZ)
    else:
        now = now.astimezone(MARKET_TZ)
    return now.strftime("%Y-%m-%d")


def alert_key(row, day=None) -> str:
    """مفتاح منع التكرار: يوم ET + Ticker + اتجاه + Strike + Expiry."""
    day = day or trading_day_et()
    ticker = str(row.get("Ticker") or row.get("ticker") or "?").upper().strip()
    direction = normalize_direction(row.get("direction"))
    expiry = str(row.get("expiry") or "")[:10]
    try:
        strike = f"{float(row.get('strike', row.get('Strike', 0))):.2f}"
    except (TypeError, ValueError):
        strike = str(row.get("strike") or row.get("Strike") or "")
    return f"{day}|{ticker}|{direction}|{strike}|{expiry}"


def load_sent(path=SENT_FILE) -> dict:
    if not os.path.exists(path):
        return {"version": 1, "sent": {}}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"version": 1, "sent": {}}
        sent = data.get("sent") or {}
        if not isinstance(sent, dict):
            sent = {}
        return {"version": 1, "sent": sent}
    except Exception as e:
        print(f"⚠️  قراءة {path} فشلت ({e}) — نبدأ فارغ")
        return {"version": 1, "sent": {}}


def prune_sent(sent: dict, keep_days=SENT_KEEP_DAYS, today=None) -> dict:
    today = today or trading_day_et()
    try:
        cutoff = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=keep_days)).strftime("%Y-%m-%d")
    except Exception:
        return sent
    return {k: v for k, v in sent.items() if str(k).split("|", 1)[0] >= cutoff}


def save_sent(data: dict, path=SENT_FILE):
    payload = {
        "version": 1,
        "sent": prune_sent(data.get("sent") or {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def row_allows_telegram(row) -> bool:
    """لا ترسل إن وُجد exec_window_ok=False صراحة (حماية إضافية)."""
    try:
        val = row["exec_window_ok"]
    except Exception:
        return True
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    s = str(val).strip().lower()
    if s in ("", "nan", "none"):
        return True
    return s in ("1", "true", "yes", "y")


def select_buy_rows(df: pd.DataFrame, sent_keys=None, force=False, day=None):
    """يرجع (للإرسال، المتخطى بسبب dedupe، المتخطى بسبب النافذة)."""
    sent_keys = set(sent_keys or [])
    day = day or trading_day_et()
    force = force or os.environ.get("TELEGRAM_FORCE", "").lower() in ("1", "true", "yes")

    if "recommendation" not in df.columns:
        return df.iloc[0:0], [], []

    buys = df[df["recommendation"].astype(str).str.upper() == "BUY"]
    to_send = []
    skipped_dup = []
    skipped_window = []
    for _, row in buys.iterrows():
        if not row_allows_telegram(row):
            skipped_window.append(alert_key(row, day=day))
            continue
        key = alert_key(row, day=day)
        if not force and key in sent_keys:
            skipped_dup.append(key)
            continue
        to_send.append((key, row))
    return to_send, skipped_dup, skipped_window


def contract_symbol(ticker, direction, expiry_str, strike) -> str:
    """بناء رمز العقد مثل #CLF260807C12.00"""
    try:
        dt = datetime.strptime(str(expiry_str).strip(), "%Y-%m-%d")
        date_part = dt.strftime("%y%m%d")
    except Exception:
        date_part = str(expiry_str).replace("-", "")[-6:]

    letter = "C" if normalize_direction(direction) == "CALL" else "P"

    try:
        strike_fmt = f"{float(strike):.2f}"
    except Exception:
        strike_fmt = str(strike)

    return f"#{ticker}{date_part}{letter}{strike_fmt}"


def create_card(row) -> bytes:
    ticker    = str(row.get("Ticker",      "N/A"))
    price     = row.get("Price",           "N/A")
    direction = normalize_direction(row.get("direction", ""))
    entry     = row.get("entry_stock",     "N/A")
    stop_v    = row.get("stop_stock",      "N/A")
    strike    = row.get("strike",          row.get("Strike", "N/A"))
    tp1       = row.get("tp1_stock",       "N/A")
    tp2       = row.get("tp2_stock",       "N/A")
    tp3       = row.get("tp3_stock",       "N/A")
    expiry    = str(row.get("expiry",      ""))

    is_call   = direction == "CALL"
    hdr_bg    = HDR_CALL if is_call else HDR_PUT
    dir_color = GREEN_BRIGHT if is_call else RED_BRIGHT
    dir_label = "▲  CALL" if is_call else "▼  PUT"

    W, H  = 560, 490
    PAD   = 22

    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # ── outer card ──
    draw.rounded_rectangle([0, 0, W-1, H-1], radius=16, fill=CARD, outline=BORDER, width=1)

    # ── header ──────────────────────────────────────────────────
    HDR_H = 76
    draw.rounded_rectangle([0, 0, W-1, HDR_H], radius=16, fill=hdr_bg)
    draw.rectangle([0, HDR_H-16, W-1, HDR_H], fill=hdr_bg)

    # ticker (big, left)
    draw.text((PAD, 12), ticker, font=F["h1"], fill=WHITE)

    # direction label (right, bold, colored)
    draw.text((W - PAD, 14), dir_label, font=F["dir"], fill=dir_color, anchor="ra")

    # expiry (right, below direction, white)
    draw.text((W - PAD, 46), expiry, font=F["sub"], fill=WHITE, anchor="ra")

    y = HDR_H + 14

    # ── current price ────────────────────────────────────────────
    draw.rounded_rectangle([PAD, y, W-PAD, y+54], radius=10, fill=DGRAY)
    draw.text((PAD+16,   y+10), "Current Price", font=F["sub"], fill=LGRAY)
    draw.text((W-PAD-16, y+10), str(price),       font=F["h2"],  fill=WHITE, anchor="ra")
    y += 66

    # ── entry | stop ─────────────────────────────────────────────
    half = (W - PAD*2 - 10) // 2

    draw.rounded_rectangle([PAD,         y, PAD+half,  y+62], radius=10, fill=DGRAY)
    draw.text((PAD+16,      y+8),  "Entry",    font=F["sub"], fill=LGRAY)
    draw.text((PAD+16,      y+28), fmt(entry), font=F["h3"],  fill=GOLD)

    draw.rounded_rectangle([PAD+half+10, y, W-PAD,    y+62], radius=10, fill=DGRAY)
    draw.text((PAD+half+26, y+8),  "Stop",     font=F["sub"], fill=LGRAY)
    draw.text((PAD+half+26, y+28), fmt(stop_v),font=F["h3"],  fill=RED_BRIGHT)
    y += 76

    # ── strike ───────────────────────────────────────────────────
    draw.rounded_rectangle([PAD, y, W-PAD, y+50], radius=10, fill=DGRAY)
    draw.text((PAD+16,   y+8),  "Strike",    font=F["sub"], fill=LGRAY)
    draw.text((W-PAD-16, y+8),  fmt(strike), font=F["h3"],  fill=BLUE_BRIGHT, anchor="ra")
    y += 64

    # ── divider ──────────────────────────────────────────────────
    draw.line([PAD, y, W-PAD, y], fill=DIVIDER, width=1)
    y += 12

    # ── targets ──────────────────────────────────────────────────
    tp_w = (W - PAD*2 - 20) // 3
    for i, (lbl, val) in enumerate([("TP 1", tp1), ("TP 2", tp2), ("TP 3", tp3)]):
        x0 = PAD + i*(tp_w+10)
        draw.rounded_rectangle([x0, y, x0+tp_w, y+60], radius=10, fill=DGRAY)
        draw.text((x0+tp_w//2, y+8),  lbl,      font=F["xs"],  fill=LGRAY,       anchor="ma")
        draw.text((x0+tp_w//2, y+28), fmt(val), font=F["h3"],  fill=GREEN_BRIGHT, anchor="ma")
    y += 74

    # ── footer ───────────────────────────────────────────────────
    draw.line([PAD, y, W-PAD, y], fill=DIVIDER, width=1)
    y += 10
    draw.text((W//2, y+5), "For educational purposes only — not financial advice",
              font=F["xs"], fill=LGRAY, anchor="ma")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def build_caption(row) -> str:
    ticker    = str(row.get("Ticker",      "N/A"))
    direction = normalize_direction(row.get("direction", ""))
    entry     = row.get("entry_stock",     "N/A")
    stop_v    = row.get("stop_stock",      "N/A")
    strike    = row.get("strike",          row.get("Strike", "N/A"))
    tp1       = row.get("tp1_stock",       "N/A")
    tp2       = row.get("tp2_stock",       "N/A")
    tp3       = row.get("tp3_stock",       "N/A")
    expiry    = str(row.get("expiry",      ""))
    premium   = row.get("premium",         "N/A")
    oi        = row.get("oi",              "N/A")
    score     = row.get("Score",           "N/A")
    conf      = row.get("confidence",      "N/A")

    symbol = contract_symbol(ticker, direction, expiry, strike)
    gemini = str(row.get("gemini_note") or "").strip()
    gemini_block = f"\n🤖 <b>مستشار Gemini</b>\n{gemini}\n" if gemini else ""

    return (
        f"<code>{symbol}</code>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"💵 سعر الدخول: <b>{fmt(entry)}</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 الهدف الأول:  <b>{fmt(tp1)}</b>\n"
        f"🎯 الهدف الثاني: <b>{fmt(tp2)}</b>\n"
        f"🎯 الهدف الثالث: <b>{fmt(tp3)}</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🛑 وقف الخسارة: <b>{fmt(stop_v)}</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 Score: <b>{score}</b>  |  Confidence: <b>{conf}%</b>\n"
        f"💰 Premium: <b>{fmt(premium)}</b>  |  OI: <b>{fmt_oi(oi)}</b>\n"
        f"{gemini_block}"
        f"⚠️ <i>للأغراض التعليمية فقط</i>"
    )


def send_photo(image_bytes: bytes, caption: str = "") -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Missing TELEGRAM credentials")
        return False

    boundary = "----TGBound9x"
    parts = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
        f"{TELEGRAM_CHAT_ID}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="parse_mode"\r\n\r\n'
        f"HTML\r\n"
    )
    if caption:
        parts += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="caption"\r\n\r\n'
            f"{caption}\r\n"
        )
    header = parts.encode()
    file_part = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="photo"; filename="signal.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode()
    footer = f"\r\n--{boundary}--\r\n".encode()

    body = header + file_part + image_bytes + footer

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type":   f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read()).get("ok", False)
    except Exception as e:
        print(f"❌ sendPhoto: {e}")
        return False


def send_message(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
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
            return json.loads(resp.read()).get("ok", False)
    except Exception as e:
        print(f"❌ sendMessage: {e}")
        return False


def main():
    if not os.path.exists(CSV_FILE):
        print(f"⚠️  {CSV_FILE} not found")
        return

    df = pd.read_csv(CSV_FILE)
    store = load_sent(SENT_FILE)
    sent_map = store.get("sent") or {}
    to_send, skipped_dup, skipped_window = select_buy_rows(df, sent_keys=sent_map.keys())

    if skipped_dup:
        print(f"⏭️  تخطي {len(skipped_dup)} مكررة (نفس العقد اليوم)")
    if skipped_window:
        print(f"⏭️  تخطي {len(skipped_window)} خارج نافذة التنفيذ")

    if not to_send:
        print("No new BUY signals to send (deduped or empty)")
        # حافظ على ملف الحالة مُقلَّماً حتى لو ما فيه إرسال
        save_sent(store, SENT_FILE)
        return

    print(f"📤 Sending {len(to_send)} new signal(s)...")

    send_message(
        f"📋 <b>Options Screener</b>\n"
        f"🟢 <b>{len(to_send)}</b> new signal(s) ↓"
    )

    dirty = False
    for key, row in to_send:
        ticker = row.get("Ticker", "?")
        try:
            card    = create_card(row)
            caption = build_caption(row)
            ok      = send_photo(card, caption)
            print(f"  {'✅' if ok else '❌'} {ticker} [{key}]")
            if ok:
                sent_map[key] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                dirty = True
        except Exception as e:
            print(f"  ❌ {ticker}: {e}")

    if dirty:
        store["sent"] = sent_map
        save_sent(store, SENT_FILE)
        print(f"💾 حدّث {SENT_FILE} ({len(sent_map)} مفتاح)")
    else:
        save_sent(store, SENT_FILE)


if __name__ == "__main__":
    main()
