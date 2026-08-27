"""
إغلاق عقد يومي من Theta Data — طبقة المتتبع فقط.

لا يُستخدم في المسح الحي ولا في بوابات BUY.
المجاني: تقرير EOD بعد 17:15 ET عبر Theta Terminal على المنفذ 25503.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime

try:
    import requests
except ImportError:
    requests = None

DEFAULT_BASE = "http://127.0.0.1:25503/v3"
_LAST_CALL = 0.0
_READY = None


def theta_base_url() -> str:
    return (os.environ.get("THETA_BASE_URL") or DEFAULT_BASE).rstrip("/")


def theta_disabled() -> bool:
    return os.environ.get("THETA_DISABLE", "").strip().lower() in ("1", "true", "yes")


def theta_min_interval() -> float:
    try:
        return max(0.0, float(os.environ.get("THETA_MIN_INTERVAL", "3.1")))
    except (TypeError, ValueError):
        return 3.1


def theta_wanted() -> bool:
    """لا نفحص المنفذ في كل تحديث سعر — فقط إن طُلب صراحة أو Terminal محلي مع EOD."""
    if theta_disabled() or requests is None:
        return False
    if os.environ.get("THETA_BASE_URL", "").strip():
        return True
    if os.environ.get("THETA_EMAIL", "").strip():
        return True
    if os.environ.get("EOD_RUN", "").strip().lower() in ("1", "true", "yes"):
        return True
    return False


def reset_theta_ready() -> None:
    global _READY
    _READY = None


def theta_ready(timeout=0.6) -> bool:
    """هل Terminal يرد؟ النتيجة تُحفظ حتى reset_theta_ready()."""
    global _READY
    if _READY is not None:
        return _READY
    if not theta_wanted():
        _READY = False
        return False
    url = f"{theta_base_url()}/option/history/eod"
    try:
        r = requests.get(url, params={"format": "json"}, timeout=timeout)
        _READY = r.status_code < 500
    except Exception:
        _READY = False
    return _READY


def _as_date(val):
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    try:
        return datetime.strptime(str(val)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _ymd(d) -> str:
    d = _as_date(d)
    return d.strftime("%Y%m%d") if d else ""


def _strike(val) -> str:
    return f"{float(val):.3f}"


def _right(is_call) -> str:
    return "call" if is_call else "put"


def parse_eod_close(payload) -> float | None:
    """
    سعر الخروج من تقرير EOD:
    إغلاق صفقة إن وُجد >0، وإلا منتصف NBBO، وإلا 0 إن وُجد عرض/طلب صفري (منتهٍ بلا قيمة).
    """
    rows = payload
    if isinstance(payload, dict):
        rows = payload.get("response") or payload.get("data") or payload.get("rows") or []
    if not isinstance(rows, list) or not rows:
        return None
    row = rows[-1] if isinstance(rows[-1], dict) else None
    if not row:
        return None

    def _f(key):
        try:
            v = row.get(key)
            if v is None or v == "":
                return None
            f = float(v)
            return f if f == f else None
        except (TypeError, ValueError):
            return None

    close = _f("close")
    if close is not None and close > 0:
        return round(close, 4)
    bid, ask = _f("bid"), _f("ask")
    if bid is not None and ask is not None and bid >= 0 and ask >= 0 and ask >= bid:
        if ask > 0 or bid > 0:
            return round((bid + ask) / 2.0, 4)
        return 0.0
    if close == 0:
        return 0.0
    return None


def _throttle():
    global _LAST_CALL
    wait = theta_min_interval()
    if wait <= 0:
        _LAST_CALL = time.monotonic()
        return
    now = time.monotonic()
    gap = wait - (now - _LAST_CALL)
    if gap > 0:
        time.sleep(gap)
    _LAST_CALL = time.monotonic()


def fetch_option_eod(symbol, expiration, strike, is_call, on_date, session=None):
    """طلب واحد لتقرير EOD. session اختياري للاختبارات."""
    if requests is None or theta_disabled():
        return None
    exp_d = _as_date(expiration)
    on_d = _as_date(on_date)
    if not symbol or exp_d is None or on_d is None:
        return None
    qday = min(on_d, exp_d)
    params = {
        "symbol": str(symbol).strip().upper(),
        "expiration": _ymd(exp_d),
        "strike": _strike(strike),
        "right": _right(is_call),
        "start_date": _ymd(qday),
        "end_date": _ymd(qday),
        "format": "json",
    }
    url = f"{theta_base_url()}/option/history/eod"
    _throttle()
    get = session.get if session is not None else requests.get
    try:
        r = get(url, params=params, timeout=20)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def theta_option_close(ticker, expiry, strike, is_call, on_date, cache=None, session=None):
    """إغلاق/منتصف NBBO ليوم الخروج. None إن تعطّل Theta أو لا بيانات."""
    if not theta_wanted() or not theta_ready():
        return None
    try:
        k = float(strike)
    except (TypeError, ValueError):
        return None
    key = (
        str(ticker or "").strip().upper(),
        str(_as_date(expiry)),
        round(k, 4),
        bool(is_call),
        str(_as_date(on_date)),
    )
    if cache is not None and key in cache:
        return cache[key]
    payload = fetch_option_eod(ticker, expiry, strike, is_call, on_date, session=session)
    px = parse_eod_close(payload)
    if cache is not None:
        cache[key] = px
    return px
