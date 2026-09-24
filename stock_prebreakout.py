"""
ماسح أسهم قبل الاختراق.
لا يُصدر BUY. التنبيه مراقبة حتى يلمس السعر التفعيل ويؤكَّد.
الهدف = التفعيل × 1.07، وليس وعداً بأن السهم سيصعد.
"""

from __future__ import annotations

# أوزان القسم 22 — مجموعها 100
WEIGHTS = {
    "catalyst": 15,
    "dollar_volume": 15,
    "rvol": 15,
    "float": 10,
    "gap": 10,
    "higher_lows": 10,
    "compression": 10,
    "dist_pmh": 5,
    "vwap": 5,
    "momentum": 5,
}

TARGET_MULT = 1.07
TRIGGER_BUFFER = 1.01
MIN_RR = 3.0

SIGNAL_A = "A_PLUS"
SIGNAL_STRONG = "STRONG"
SIGNAL_WATCH = "WATCH"
SIGNAL_IGNORE = "IGNORE"

DISCLAIMER = (
    "ليست ضماناً لصعود 7%. الإعداد يجمع الأدلة فقط، والدخول يدوي بعد تأكيد التفعيل."
)


def volume_is_rising(volumes) -> bool:
    """الحجم يتصاعد نحو المقاومة، لا يتراجع."""
    vals = []
    for v in volumes or []:
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if len(vals) < 3:
        return False
    ups = sum(1 for a, b in zip(vals, vals[1:]) if b > a)
    return ups >= len(vals) - 1


def higher_low_steps(lows) -> int:
    """أطول سلسلة قيعان صاعدة متتالية."""
    vals = []
    for v in lows or []:
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    best = 1 if vals else 0
    run = 1
    for a, b in zip(vals, vals[1:]):
        if b > a:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def is_compressing(older_range, newer_range) -> bool:
    try:
        older_range = float(older_range)
        newer_range = float(newer_range)
    except (TypeError, ValueError):
        return False
    if older_range <= 0:
        return False
    return newer_range < older_range * 0.80


def _num(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def breakout_level(price, pmh, prev_day_high=None, intraday_high=None):
    """أقرب مقاومة فوق السعر. إن لم توجد، قمة البريماركت."""
    ceilings = []
    for raw in (pmh, prev_day_high, intraday_high):
        level = _num(raw)
        px = _num(price)
        if level is None:
            continue
        if px is None or level >= px * 0.999:
            ceilings.append(level)
    if ceilings:
        return min(ceilings)
    return _num(pmh)


def trigger_price(level):
    """تفعيل فوق المقاومة بقليل. 4.85 → 4.90."""
    level = _num(level)
    if not level or level <= 0:
        return None
    return round(level * TRIGGER_BUFFER + 1e-9, 2)


def target_price(entry):
    """هدف عملي: الدخول × 1.07 مقرباً لخانتين."""
    entry = _num(entry)
    if not entry or entry <= 0:
        return None
    return round(entry * TARGET_MULT + 1e-9, 2)


def structure_stop(trigger, structure_low):
    """وقف تحت آخر قاع صاعد، لا نسبة ثابتة لكل سهم."""
    trigger = _num(trigger)
    low = _num(structure_low)
    if not trigger or trigger <= 0:
        return None
    if low and low < trigger:
        stop = round(low - max(0.01, low * 0.005) + 1e-9, 2)
    else:
        stop = round(trigger * 0.96 + 1e-9, 2)
    if stop >= trigger:
        stop = round(trigger - 0.01, 2)
    return stop


def risk_reward(entry, stop, target):
    entry, stop, target = _num(entry), _num(stop), _num(target)
    if None in (entry, stop, target):
        return 0.0
    risk = entry - stop
    reward = target - entry
    if risk <= 0 or reward <= 0:
        return 0.0
    return round(reward / risk, 2)


def _points_catalyst(kind):
    kind = str(kind or "none").lower()
    if kind == "company":
        return WEIGHTS["catalyst"]
    if kind == "sector":
        return 6
    return 0


def _points_dollar(pm_volume, dollar_volume):
    vol = _num(pm_volume) or 0
    dollar = _num(dollar_volume) or 0
    if vol < 500_000:
        return 0
    if vol >= 1_000_000 and dollar >= 5_000_000:
        return WEIGHTS["dollar_volume"]
    if vol >= 1_000_000 and dollar >= 2_000_000:
        return 12
    if vol >= 500_000 and dollar >= 1_000_000:
        return 8
    if vol >= 500_000:
        return 4
    return 0


def _points_rvol(rvol):
    rvol = _num(rvol)
    if rvol is None or rvol < 2:
        return 0
    if rvol >= 5:
        return WEIGHTS["rvol"]
    if rvol >= 3:
        return 12
    return 8


def _points_float(float_shares):
    shares = _num(float_shares)
    if shares is None or shares <= 0 or shares >= 100_000_000:
        return 0
    if shares < 20_000_000:
        return WEIGHTS["float"]
    if shares < 50_000_000:
        return 7
    return 4


def _points_gap(gap_pct, catalyst_points):
    gap = _num(gap_pct)
    if gap is None or gap < 5:
        return 0
    pts = WEIGHTS["gap"] if gap >= 10 else 6
    if gap >= 10 and catalyst_points <= 0:
        pts = min(pts, 4)
    return pts


def _points_higher_lows(steps, rejected):
    if rejected:
        return 0
    steps = int(steps or 0)
    if steps >= 4:
        return WEIGHTS["higher_lows"]
    if steps >= 3:
        return 6
    return 0


def _points_compression(compressed, volume_rising, rejected):
    if rejected:
        return 0
    if compressed and volume_rising:
        return WEIGHTS["compression"]
    if compressed or volume_rising:
        return 5
    return 0


def _points_distance(price, pmh):
    price, pmh = _num(price), _num(pmh)
    if not price or not pmh or price <= 0:
        return 0
    if price > pmh:
        return 0
    dist = (pmh - price) / price * 100
    if dist <= 2:
        return WEIGHTS["dist_pmh"]
    if dist <= 4:
        return 3
    if dist <= 8:
        return 1
    return 0


def _points_vwap(above_vwap, reclaim):
    if above_vwap or reclaim:
        return WEIGHTS["vwap"]
    return 0


def _points_momentum(rocm20, rocm20_rising, rocm200, rs_vs_spy):
    rocm20 = _num(rocm20)
    rocm200 = _num(rocm200)
    rs = _num(rs_vs_spy)
    if rocm200 is not None and rocm200 <= -0.05:
        base = 0
    elif rocm20 is not None and rocm20 > 0 and rocm20_rising:
        base = 3
    elif rocm20 is not None and rocm20 > 0:
        base = 2
    else:
        base = 0
    rs_pts = 2 if rs is not None and rs > 0 else 0
    return min(WEIGHTS["momentum"], base + rs_pts)


def _price_penalty(price, pm_volume, dollar_volume):
    price = _num(price)
    if price is None or price <= 0 or price > 10:
        return 100
    vol = _num(pm_volume) or 0
    dollar = _num(dollar_volume) or 0
    exceptional = vol >= 1_000_000 and dollar >= 5_000_000
    if price < 1:
        return 8 if exceptional else 40
    if price < 2:
        return 5
    return 0


def signal_from_score(score):
    if score >= 85:
        return SIGNAL_A, "🔥 A+ PRE-BREAKOUT"
    if score >= 75:
        return SIGNAL_STRONG, "🟢 STRONG PRE-BREAKOUT"
    if score >= 65:
        return SIGNAL_WATCH, "🟡 WATCHLIST"
    return SIGNAL_IGNORE, "❌ IGNORE"


def reason_for(code, trigger):
    trigger_txt = f"${trigger:.2f}" if trigger else "$—"
    if code == SIGNAL_A:
        return (
            "The stock is approaching a major resistance with Higher Lows, "
            "Compression, increasing volume, strong RVOL, and positive momentum. "
            f"Trigger = {trigger_txt}. Wait for confirmation before entering."
        )
    if code == SIGNAL_STRONG:
        return "Strong setup, but one or more confirmations are still developing."
    if code == SIGNAL_WATCH:
        return "Potential setup, but important conditions are missing."
    return "Insufficient pre-breakout evidence."


def evaluate(setup: dict) -> dict:
    """
    درجة 0–100 من عدة تأكيدات.
    مؤشر واحد لا يكفي للوصول إلى 65.
    """
    setup = setup or {}
    price = _num(setup.get("price"))
    pm_volume = _num(setup.get("pm_volume")) or 0
    dollar = _num(setup.get("dollar_volume"))
    if dollar is None and price:
        dollar = pm_volume * price
    dollar = dollar or 0

    catalyst_pts = _points_catalyst(setup.get("catalyst"))
    parts = {
        "catalyst": catalyst_pts,
        "dollar_volume": _points_dollar(pm_volume, dollar),
        "rvol": _points_rvol(setup.get("rvol")),
        "float": _points_float(setup.get("float_shares")),
        "gap": _points_gap(setup.get("gap_pct"), catalyst_pts),
        "higher_lows": _points_higher_lows(
            setup.get("higher_low_steps"), bool(setup.get("repeated_rejection"))
        ),
        "compression": _points_compression(
            bool(setup.get("compression")),
            bool(setup.get("volume_rising")),
            bool(setup.get("repeated_rejection")),
        ),
        "dist_pmh": _points_distance(price, setup.get("pmh")),
        "vwap": _points_vwap(bool(setup.get("above_vwap")), bool(setup.get("vwap_reclaim"))),
        "momentum": _points_momentum(
            setup.get("rocm20"),
            bool(setup.get("rocm20_rising")),
            setup.get("rocm200"),
            setup.get("rs_vs_spy"),
        ),
    }
    raw = sum(parts.values())
    penalty = _price_penalty(price, pm_volume, dollar)
    notes = []

    atr = _num(setup.get("atr_pct"))
    if atr is None or atr < 2:
        penalty += 12
        notes.append("ATR% لا يكفي تاريخياً لهدف +7%")
    elif atr < 3:
        penalty += 4

    level = breakout_level(
        price, setup.get("pmh"), setup.get("prev_day_high"), setup.get("intraday_high")
    )
    trigger = trigger_price(level)
    target = target_price(trigger)
    stop = structure_stop(trigger, setup.get("structure_low"))
    rr = risk_reward(trigger, stop, target)

    major = _num(setup.get("major_resistance"))
    resistance_blocks = bool(
        major and trigger and target and trigger < major < target
    )
    if resistance_blocks:
        penalty += 10
        notes.append(f"مقاومة ${major:.2f} قبل هدف +7%")

    if rr < MIN_RR:
        penalty += 8
        notes.append(f"R:R 1:{rr:.2f} أقل من 1:3")

    post_breakout = bool(price and trigger and price >= trigger)
    if post_breakout:
        notes.append("السعر تجاوز التفعيل — ليس إعداداً قبل الاختراق")

    score = max(0, min(100, int(round(raw - penalty))))
    if post_breakout:
        score = min(score, 74)

    code, label = signal_from_score(score)
    if code == SIGNAL_A and rr < MIN_RR:
        code, label = SIGNAL_STRONG, "🟢 STRONG PRE-BREAKOUT"
        notes.append("A+ يحتاج R:R لا يقل عن 1:3")
    if code == SIGNAL_A and (atr is None or atr < 2):
        code, label = SIGNAL_STRONG, "🟢 STRONG PRE-BREAKOUT"

    dist = None
    pmh = _num(setup.get("pmh"))
    if price and pmh and price > 0:
        dist = round((pmh - price) / price * 100, 2)

    pattern_bits = []
    if parts["higher_lows"] >= 6:
        pattern_bits.append("Higher Lows")
    if setup.get("compression"):
        pattern_bits.append("Compression")
    if setup.get("volume_rising"):
        pattern_bits.append("Increasing Volume")
    if parts["dist_pmh"] == WEIGHTS["dist_pmh"]:
        pattern_bits.append("Near Premarket High")
    pattern = " + ".join(pattern_bits) if pattern_bits else "لا نمط مكتمل"

    return {
        "score": score,
        "parts": parts,
        "raw": raw,
        "penalty": penalty,
        "signal": code,
        "signal_label": label,
        "recommendation": "WATCH" if code != SIGNAL_IGNORE else "IGNORE",
        "reason": reason_for(code, trigger),
        "disclaimer": DISCLAIMER,
        "trigger": trigger,
        "target": target,
        "stop": stop,
        "risk_reward": rr,
        "resistance": level,
        "distance_to_resistance_pct": dist,
        "resistance_before_target": major if resistance_blocks else None,
        "pattern": pattern,
        "notes": " · ".join(notes),
        "post_breakout": post_breakout,
    }
