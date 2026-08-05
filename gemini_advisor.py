"""
gemini_advisor.py
-----------------
مستشار Gemini بعد قواعد السكرينر فقط.
لا يغيّر BUY/WAIT/AVOID — يضيف ملاحظة تحليلية قصيرة (عربي).

المتغير: GEMINI_API_KEY من Google AI Studio
اختياري: GEMINI_MODEL (افتراضي gemini-2.0-flash)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping, Optional

import pandas as pd
import requests

RESULTS_FILE = "options_v3_results.csv"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"
API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def get_api_key() -> str:
    return (os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or "").strip()


def _num(val: Any, default: float = 0.0) -> float:
    try:
        if val is None or val == "":
            return default
        return float(str(val).replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return default


def _row_payload(row: Mapping[str, Any]) -> dict:
    return {
        "ticker": str(row.get("Ticker") or row.get("ticker") or "?"),
        "recommendation": str(row.get("recommendation") or "AVOID"),
        "direction": str(row.get("direction") or ""),
        "confidence": _num(row.get("confidence")),
        "score": _num(row.get("Score") or row.get("score")),
        "price": _num(row.get("Price") or row.get("price_num")),
        "entry": _num(row.get("entry_stock")),
        "stop": _num(row.get("stop_stock")),
        "tp1": _num(row.get("tp1_stock")),
        "tp2": _num(row.get("tp2_stock")),
        "tp3": _num(row.get("tp3_stock")),
        "tp1_rr": _num(row.get("tp1_rr")),
        "tp1_rr_live": _num(row.get("tp1_rr_live")),
        "entry_chased": str(row.get("entry_chased") or ""),
        "premium": _num(row.get("premium")),
        "strike": _num(row.get("strike")),
        "expiry": str(row.get("expiry") or ""),
        "dte": _num(row.get("dte_num")),
        "spread_pct": _num(row.get("spread_pct")),
        "iv": _num(row.get("iv")),
        "oi": _num(row.get("oi")),
        "spy_regime": str(row.get("spy_regime") or ""),
        "rec_note": str(row.get("rec_note") or ""),
        "entry_note": str(row.get("entry_note") or ""),
        "earnings": str(row.get("earnings") or ""),
        "fh_pm_note": str(row.get("fh_pm_note") or ""),
    }


def _build_prompt(payload: dict) -> str:
    return f"""أنت مستشار تعليمي لسكربت خيارات أمريكية. القواعد الحاسوبية قررت مسبقاً التوصية ولا يجوز تغييرها.

التوصية النهائية الثابتة: {payload['recommendation']}
البيانات:
{json.dumps(payload, ensure_ascii=False, indent=2)}

اكتب بالعربية الفصحى المبسطة فقط:
1) سطر واحد: تقييم المخاطر (منخفض/متوسط/مرتفع) مع سبب مختصر.
2) سطر واحد: نقطة مراقبة مهمة قبل التنفيذ (مطاردة Entry، أرباح، سبريد، R:R حي، اتجاه SPY...).
3) سطر واحد: تذكير أن هذا تعليمي وليس نصيحة استثمارية.

ممنوع:
- تغيير التوصية أو اقتراح BUY إذا كانت WAIT/AVOID
- أرقام أسعار جديدة من عندك
- وعود ربح

الحد الأقصى: 3 أسطر قصيرة جداً."""


def advise_row(row: Mapping[str, Any], api_key: Optional[str] = None,
               model: Optional[str] = None, timeout: int = 25) -> str:
    """يرجع نص المستشار أو سلسلة فارغة عند الفشل/غياب المفتاح."""
    key = (api_key or get_api_key()).strip()
    if not key:
        return ""

    model_id = (model or DEFAULT_MODEL).strip()
    payload = _row_payload(row)
    url = f"{API_BASE}/models/{model_id}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [{"text": _build_prompt(payload)}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 220,
        },
    }
    try:
        resp = requests.post(
            url,
            params={"key": key},
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
        if resp.status_code >= 400:
            print(f"  ⚠ Gemini HTTP {resp.status_code}: {resp.text[:180]}")
            return ""
        data = resp.json()
        parts = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
        )
        text = " ".join(str(p.get("text") or "").strip() for p in parts).strip()
        # تنظيف خفيف
        text = " ".join(text.split())
        if len(text) > 500:
            text = text[:497] + "…"
        return text
    except Exception as e:
        print(f"  ⚠ Gemini error: {e}")
        return ""


def enrich_dataframe(df: pd.DataFrame, only_recs=("BUY", "WAIT"),
                     sleep_s: float = 0.4) -> pd.DataFrame:
    """يضيف عمود gemini_note بدون تعديل recommendation."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "gemini_note" not in out.columns:
        out["gemini_note"] = ""
    if not get_api_key():
        print("Gemini: لا يوجد GEMINI_API_KEY — تخطي المستشار")
        return out

    mask = out["recommendation"].isin(list(only_recs))
    idxs = list(out.index[mask])
    print(f"Gemini advisor: {len(idxs)} صف ({', '.join(only_recs)})")
    for i, idx in enumerate(idxs, 1):
        row = out.loc[idx]
        note = advise_row(row)
        out.at[idx, "gemini_note"] = note
        ticker = row.get("Ticker", "?")
        print(f"  [{i}/{len(idxs)}] {ticker}: {'✓' if note else '—'}")
        if sleep_s > 0 and i < len(idxs):
            time.sleep(sleep_s)
    return out


def enrich_results_csv(path: str = RESULTS_FILE) -> bool:
    if not os.path.exists(path):
        print(f"⚠️  {path} غير موجود")
        return False
    df = pd.read_csv(path)
    enriched = enrich_dataframe(df)
    enriched.to_csv(path, index=False)
    print(f"✓ حُفظ {path} مع gemini_note")
    return True


def main():
    ok = enrich_results_csv(RESULTS_FILE)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
