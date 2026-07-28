"""
telegram_notify.py
------------------
Sends each BUY recommendation as a styled PNG card to Telegram
"""

import os
import io
import json
import urllib.request
import urllib.parse
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
CSV_FILE         = "options_v3_results.csv"

# ── Color palette ───────────────────────────────────────────────
BG        = (13,  17,  23)
CARD      = (22,  27,  34)
HEADER_G  = (0,   200, 83)
RED       = (255, 23,  68)
GOLD      = (255, 214, 0)
BLUE      = (33,  150, 243)
WHITE     = (255, 255, 255)
LGRAY     = (139, 148, 158)
DGRAY     = (33,  38,  45)
BORDER    = (48,  54,  61)
GREEN_TXT = (0,   200, 83)


def _load_fonts():
    bold_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    reg_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]

    def try_load(paths, size):
        for p in paths:
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
        return ImageFont.load_default()

    return {
        "xl": try_load(bold_paths, 30),
        "lg": try_load(bold_paths, 22),
        "md": try_load(bold_paths, 17),
        "sm": try_load(reg_paths,  14),
        "xs": try_load(reg_paths,  12),
    }


FONTS = _load_fonts()


def fmt(v):
    try:
        return f"${float(v):.2f}"
    except Exception:
        return str(v)


def fmt_oi(v):
    try:
        return f"{int(float(v)):,}"
    except Exception:
        return str(v)


def create_card(row) -> bytes:
    ticker    = str(row.get("Ticker",      "N/A"))
    price     = row.get("Price",           "N/A")
    direction = str(row.get("direction",   "")).replace("📈","").replace("📉","").strip().upper()
    entry     = row.get("entry_stock",     "N/A")
    stop_v    = row.get("stop_stock",      "N/A")
    tp1       = row.get("tp1_stock",       "N/A")
    tp2       = row.get("tp2_stock",       "N/A")
    tp3       = row.get("tp3_stock",       "N/A")
    premium   = row.get("premium",         "N/A")
    oi        = row.get("oi",              "N/A")
    expiry    = row.get("expiry",          "N/A")
    conf      = row.get("confidence",      "N/A")
    score     = row.get("Score",           "N/A")

    dir_label = "CALL  📈" if direction == "CALL" else "PUT  📉"

    W, H  = 540, 640
    PAD   = 24
    R     = 14

    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # outer card
    draw.rounded_rectangle([0, 0, W-1, H-1], radius=R, fill=CARD, outline=BORDER, width=1)

    # header
    draw.rounded_rectangle([0, 0, W-1, 76], radius=R, fill=HEADER_G)
    draw.rectangle([0, 60, W-1, 76], fill=HEADER_G)

    draw.text((PAD, 14),    f"BUY — {ticker}", font=FONTS["xl"], fill=BG)
    draw.text((PAD, 48),    dir_label,          font=FONTS["sm"], fill=BG)
    draw.text((W-PAD, 18),  "Options Screener", font=FONTS["xs"], fill=BG, anchor="ra")

    y = 94

    # current price
    draw.rounded_rectangle([PAD, y, W-PAD, y+52], radius=8, fill=DGRAY)
    draw.text((PAD+14,   y+8),  "Current Price", font=FONTS["sm"], fill=LGRAY)
    draw.text((W-PAD-14, y+8),  str(price),       font=FONTS["lg"], fill=WHITE, anchor="ra")
    y += 68

    # entry | stop
    half = (W - PAD*2 - 10) // 2

    draw.rounded_rectangle([PAD,          y, PAD+half,    y+58], radius=8, fill=DGRAY)
    draw.text((PAD+14,         y+8),  "Entry",    font=FONTS["sm"], fill=LGRAY)
    draw.text((PAD+14,         y+28), fmt(entry), font=FONTS["lg"], fill=GOLD)

    draw.rounded_rectangle([PAD+half+10, y, W-PAD,       y+58], radius=8, fill=DGRAY)
    draw.text((PAD+half+24,    y+8),  "Stop Loss", font=FONTS["sm"], fill=LGRAY)
    draw.text((PAD+half+24,    y+28), fmt(stop_v), font=FONTS["lg"], fill=RED)
    y += 74

    # divider + targets
    draw.line([PAD, y, W-PAD, y], fill=BORDER, width=1)
    y += 12
    draw.text((PAD, y), "Targets", font=FONTS["sm"], fill=LGRAY)
    y += 22

    # TP1 | TP2 | TP3
    tp_w = (W - PAD*2 - 20) // 3
    for i, (lbl, val) in enumerate([("TP1", tp1), ("TP2", tp2), ("TP3", tp3)]):
        x0 = PAD + i * (tp_w + 10)
        draw.rounded_rectangle([x0, y, x0+tp_w, y+52], radius=8, fill=DGRAY)
        draw.text((x0 + tp_w//2, y+8),  lbl,      font=FONTS["sm"], fill=LGRAY,    anchor="ma")
        draw.text((x0 + tp_w//2, y+26), fmt(val), font=FONTS["md"], fill=GREEN_TXT, anchor="ma")
    y += 68

    # divider
    draw.line([PAD, y, W-PAD, y], fill=BORDER, width=1)
    y += 14

    # premium | OI
    draw.rounded_rectangle([PAD,          y, PAD+half,    y+52], radius=8, fill=DGRAY)
    draw.text((PAD+14,         y+8),  "Premium",       font=FONTS["sm"], fill=LGRAY)
    draw.text((PAD+14,         y+28), fmt(premium),    font=FONTS["md"], fill=WHITE)

    draw.rounded_rectangle([PAD+half+10, y, W-PAD,       y+52], radius=8, fill=DGRAY)
    draw.text((PAD+half+24,    y+8),  "Open Interest", font=FONTS["sm"], fill=LGRAY)
    draw.text((PAD+half+24,    y+28), fmt_oi(oi),      font=FONTS["md"], fill=WHITE)
    y += 68

    # expiry
    draw.rounded_rectangle([PAD, y, W-PAD, y+48], radius=8, fill=DGRAY)
    draw.text((PAD+14, y+8),   "Expiry",    font=FONTS["sm"], fill=LGRAY)
    draw.text((PAD+14, y+26),  str(expiry), font=FONTS["md"], fill=WHITE)
    y += 64

    # score | confidence
    draw.rounded_rectangle([PAD,          y, PAD+half,    y+52], radius=8, fill=DGRAY)
    draw.text((PAD+14,         y+8),  "Score",       font=FONTS["sm"], fill=LGRAY)
    draw.text((PAD+14,         y+28), str(score),    font=FONTS["lg"], fill=GOLD)

    draw.rounded_rectangle([PAD+half+10, y, W-PAD,       y+52], radius=8, fill=DGRAY)
    draw.text((PAD+half+24,    y+8),  "Confidence",  font=FONTS["sm"], fill=LGRAY)
    draw.text((PAD+half+24,    y+28), f"{conf}%",    font=FONTS["lg"], fill=BLUE)
    y += 68

    # disclaimer
    draw.text((W//2, y+4), "For educational purposes only — not financial advice",
              font=FONTS["xs"], fill=LGRAY, anchor="ma")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def send_photo(image_bytes: bytes) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Missing TELEGRAM credentials")
        return False

    boundary = "----TGBoundary7x"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
        f"{TELEGRAM_CHAT_ID}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="photo"; filename="signal.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type":   f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read()).get("ok", False)
    except Exception as e:
        print(f"❌ sendPhoto error: {e}")
        return False


def send_message(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text":    text,
        "parse_mode": "HTML",
    }).encode()
    try:
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read()).get("ok", False)
    except Exception as e:
        print(f"❌ sendMessage error: {e}")
        return False


def main():
    if not os.path.exists(CSV_FILE):
        print(f"⚠️  {CSV_FILE} not found")
        return

    df       = pd.read_csv(CSV_FILE)
    buy_rows = df[df["recommendation"] == "BUY"]

    if buy_rows.empty:
        print("No BUY signals today — nothing sent")
        return

    print(f"📤 Sending {len(buy_rows)} BUY cards...")

    send_message(
        f"📋 <b>Options Screener — Daily Report</b>\n"
        f"🟢 BUY Signals: <b>{len(buy_rows)}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Cards below ↓"
    )

    for _, row in buy_rows.iterrows():
        ticker = row.get("Ticker", "?")
        try:
            card = create_card(row)
            ok   = send_photo(card)
            print(f"  {'✅' if ok else '❌'} {ticker}")
        except Exception as e:
            print(f"  ❌ {ticker}: {e}")


if __name__ == "__main__":
    main()
