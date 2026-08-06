"""
تصنيف جودة صفوف outcomes — مصدر الحقيقة الواحد.
الداشبورد يكرّر نفس القواعد في classifyDataQuality (يجب إبقاؤها متطابقة).
"""


def truthy_entry_hit(val):
    if isinstance(val, bool):
        return val
    s = str(val or "").strip().lower()
    return s in ("true", "1", "yes", "y")


def classify_data_quality(row):
    """
    reliable   = صالح لنسب الاعتماد (دخول + سوق/ذاتي + بدون تناقض)
    partial    = مفيد للمراجعة لكن لا يدخل KPI الافتراضي
    unreliable = ضعيف / ناقص / متناقض
    open       = صفقة مفتوحة
    """
    status = str(row.get("status") or "").strip()
    if status == "open":
        return "open", "صفقة مفتوحة — خارج نسب الإغلاق"

    if not truthy_entry_hit(row.get("entry_hit")):
        return "unreliable", "Entry لم يتحقق — لا تُحسب كتنفيذ"

    try:
        prem = float(row.get("premium") or 0)
    except (TypeError, ValueError):
        prem = 0.0
    try:
        strike = float(row.get("strike") or 0)
    except (TypeError, ValueError):
        strike = 0.0
    if prem <= 0 or strike <= 0:
        return "unreliable", "ناقص premium أو strike"

    src = str(row.get("exit_premium_source") or "").strip().lower()
    if src not in ("market", "intrinsic", "estimate"):
        return "unreliable", "لا مصدر موثوق لسعر خروج العقد"

    try:
        opt_pnl = float(row.get("option_pnl_pct"))
        if opt_pnl != opt_pnl:  # NaN
            opt_pnl = None
    except (TypeError, ValueError):
        opt_pnl = None

    # تناقض: وقف خسارة مع ربح عقد — غالباً إغلاق يومي مضلل
    if status == "stop_hit" and opt_pnl is not None and opt_pnl > 5:
        return "unreliable", "تناقض Stop مع ربح عقد (إغلاق يومي مضلل)"

    # هدف مع خسارة عقد ضخمة من تقدير فقط → ضعيف
    if (
        status in ("tp1_hit", "tp2_hit", "tp3_hit")
        and src == "estimate"
        and opt_pnl is not None
        and opt_pnl < -30
    ):
        return "partial", "هدف سهم مع خسارة عقد تقديرية كبيرة"

    if src == "estimate":
        return "partial", "سعر عقد تقديري — للمراجعة فقط"

    if src == "intrinsic" and status != "expired":
        return "partial", "قيمة ذاتية بدون حالة انتهاء"

    if opt_pnl is None:
        return "partial", "لا P&L عقد محسوب رغم بيانات الدخول"

    return "reliable", "دخول + سعر سوق/ذاتي + بدون تناقض"
