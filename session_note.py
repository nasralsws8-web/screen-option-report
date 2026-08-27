"""
ملفات أوبسيديان من outcomes.csv فقط.

الأيام → [[السجل]] → تتفرع [[تحققت]] [[فشلت]] [[انتظار]] وغيرها.
لا يلمس المحرك، لا يرقّي WAIT→BUY، لا يخمن أسعاراً.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
OUTCOMES_FILE = ROOT / "outcomes.csv"
SESSIONS_DIR = ROOT / "sessions"
DEFAULT_VAULT = Path.home() / "Documents" / "Second-Brain" / "جلسات"
STUDY_N = 30
LEDGER_NOTE = "السجل.md"
BRANCH_NOTES = {
    "تحققت": "تحققت.md",
    "فشلت": "فشلت.md",
    "انتظار": "انتظار.md",
}
FAIL_NOTES = {
    "وقف": "وقف.md",
    "انتهاء": "انتهاء.md",
    "هدف بلا اعتماد": "بلا اعتماد.md",
}
STALE_NOTES = ("محققة.md",)

GROUP_BLURB = {
    "تحققت": "أُغلق على هدف، والتدقيق اعتمد الصف (جودة موثوق). القياس فقط من BUY هنا.",
    "فشلت": "وقف، أو انتهاء، أو هدف لكن التدقيق لم يعتمدها (جزئي/ضعيف).",
    "انتظار": "ما زال مفتوحاً في السجل. ليس قائمة تنفيذ اليوم.",
}

STATUS_AR = {
    "tp1_hit": "هدف 1",
    "tp2_hit": "هدف 2",
    "tp3_hit": "هدف 3",
    "stop_hit": "وقف",
    "expired": "انتهاء",
    "open": "مفتوح",
}

KIND_AR = {
    "BUY": "BUY",
    "WAIT_HOT": "WAIT_HOT",
    "LEGACY": "بلا نوع",
}

QUALITY_AR = {
    "reliable": "معتمد",
    "partial": "جزئي",
    "unreliable": "ضعيف",
    "open": "لم يُغلق",
}

KIND_HEADING = {
    "BUY": "تنفيذ BUY — هذا وحده يدخل القياس",
    "WAIT_HOT": "مراقبة WAIT_HOT — خارج القياس",
    "LEGACY": "بلا نوع (يوليو) — خارج القياس",
}

FAIL_ORDER = ("وقف", "انتهاء", "هدف بلا اعتماد", "أخرى")


def default_session_date(now: datetime | None = None) -> str:
    """تاريخ جلسة ET. السبت/الأحد → الجمعة السابقة (آخر إغلاق)."""
    now = now or datetime.now(ET)
    d = now.date()
    if d.weekday() >= 5:
        d = d - timedelta(days=d.weekday() - 4)
    return d.isoformat()


def _cell(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "none", "nat"):
        return "—"
    return s.replace("|", "/")


def _num(val, digits=2):
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        s = str(val).replace("$", "").replace(",", "").strip()
        if s == "" or s.lower() in ("nan", "none"):
            return None
        return round(float(s), digits)
    except (TypeError, ValueError):
        return None


def _fmt_px(val) -> str:
    n = _num(val)
    return f"{n:g}" if n is not None else "—"


def _fmt_pct(val) -> str:
    n = _num(val, 1)
    if n is None:
        return "—"
    return f"{n:+.1f}%"


def _ymd(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    try:
        return pd.to_datetime(val).date().isoformat()
    except (TypeError, ValueError):
        return ""


def rec_kind(row) -> str:
    k = str(row.get("rec_kind") or "").upper().strip()
    if k in ("WAIT_HOT", "HOT", "FIRE"):
        return "WAIT_HOT"
    if k == "BUY":
        return "BUY"
    tier = str(row.get("wait_tier") or "").upper().strip()
    if tier in ("HOT", "FIRE"):
        return "WAIT_HOT"
    return "LEGACY"


def is_closed(row) -> bool:
    return str(row.get("status") or "").strip() not in ("", "open")


def _quality(row) -> str:
    return str(row.get("data_quality") or "").strip().lower()


def _status(row) -> str:
    return str(row.get("status") or "").strip()


def audit_group(row) -> str:
    """بعد التدقيق: تحققت = هدف + جودة معتمدة. انتظار = لم يُغلق. الباقي فشلت."""
    closed = row["_closed"] if "_closed" in row else is_closed(row)
    if not closed:
        return "انتظار"
    if _quality(row) == "reliable" and _status(row).startswith("tp"):
        return "تحققت"
    return "فشلت"


def fail_reason(row) -> str:
    st = _status(row)
    if st == "stop_hit":
        return "وقف"
    if st == "expired":
        return "انتهاء"
    if st.startswith("tp") and _quality(row) != "reliable":
        return "هدف بلا اعتماد"
    return "أخرى"


def load_outcomes(path: Path | None = None) -> pd.DataFrame:
    p = path or OUTCOMES_FILE
    if not p.exists():
        raise FileNotFoundError(f"لا يوجد سجل: {p}")
    df = pd.read_csv(p)
    if df.empty:
        return df
    return df


def _rows(df: pd.DataFrame) -> list[dict]:
    return df.fillna("").to_dict("records") if not df.empty else []


def annotate(df: pd.DataFrame) -> list[dict]:
    rows = _rows(df)
    for r in rows:
        r["_kind"] = rec_kind(r)
        r["_date"] = _ymd(r.get("date"))
        r["_exit"] = _ymd(r.get("exit_date"))
        r["_closed"] = is_closed(r)
        r["_group"] = audit_group(r)
        r["_fail"] = fail_reason(r) if r["_group"] == "فشلت" else ""
    return rows


def _sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (
            r.get("_date") or "9999",
            str(r.get("ticker") or ""),
            r.get("_exit") or "",
        ),
    )


def session_dates(df: pd.DataFrame) -> list[str]:
    dates = set()
    for r in annotate(df):
        if r["_date"]:
            dates.add(r["_date"])
        if r["_exit"]:
            dates.add(r["_exit"])
    return sorted(dates)


def last_close_date(df: pd.DataFrame, now: datetime | None = None) -> str:
    cap = default_session_date(now)
    past = [d for d in session_dates(df) if d <= cap]
    return past[-1] if past else cap


def _table(rows: list[dict], with_dates: bool = True) -> str:
    rows = _sort_rows(rows)
    if not rows:
        return "_لا صفوف في هذا القسم._\n"
    lines = [
        "| سهم | النوع | النتيجة | سعر | دخول | خروج | العقد | التدقيق | سُجّل | أُغلق |",
        "|-----|--------|---------|-----|------|------|-------|---------|------|------|",
    ]
    for r in rows:
        kind = rec_kind(r)
        status = _status(r) or "—"
        q = _quality(r)
        closed = r["_closed"] if "_closed" in r else is_closed(r)
        lines.append(
            "| {ticker} | {kind} | {status} | {px} | {entry} | {exit} | {pnl} | {q} | {dt} | {xd} |".format(
                ticker=_cell(r.get("ticker")),
                kind=KIND_AR.get(kind, kind),
                status=STATUS_AR.get(status, status),
                px=_fmt_px(r.get("price_at_rec")),
                entry=_fmt_px(r.get("entry_stock")),
                exit=_fmt_px(r.get("exit_stock")) if closed else "—",
                pnl=_fmt_pct(r.get("option_pnl_pct")) if closed else "—",
                q=QUALITY_AR.get(q, _cell(r.get("data_quality")) or "—"),
                dt=_cell(r.get("_date") or r.get("date")),
                xd=_cell(r.get("_exit") or r.get("exit_date")) if closed else "—",
            )
        )
    return "\n".join(lines) + "\n"


def _kind_blocks(rows: list[dict]) -> list[str]:
    lines: list[str] = []
    for kind in ("BUY", "WAIT_HOT", "LEGACY"):
        chunk = [r for r in rows if r.get("_kind") == kind]
        if not chunk:
            continue
        lines += [
            f"### {KIND_HEADING[kind]} ({len(chunk)})",
            "",
            _table(chunk),
        ]
    if not lines:
        return ["_لا صفوف في هذا القسم._", ""]
    return lines


def _fail_blocks(rows: list[dict]) -> list[str]:
    lines: list[str] = []
    for reason in FAIL_ORDER:
        chunk = [r for r in rows if r.get("_fail") == reason]
        if not chunk:
            continue
        lines += [
            f"### {reason} ({len(chunk)})",
            "",
            _table(chunk),
        ]
    if not lines:
        return ["_لا صفوف في هذا القسم._", ""]
    return lines


def _group_rows(rows: list[dict], name: str) -> list[dict]:
    return [r for r in rows if r.get("_group") == name]


def _group_body(name: str, rows: list[dict]) -> list[str]:
    chunk = _group_rows(rows, name)
    lines = [
        GROUP_BLURB[name],
        "",
        f"**{len(chunk)}** صف. المصدر: السجل بعد الإغلاق. ليست توصية.",
        "",
    ]
    if name == "فشلت":
        lines += _fail_blocks(chunk)
    else:
        lines += _kind_blocks(chunk)
    return lines


def _count_quality(rows: list[dict]) -> dict[str, int]:
    out = {"reliable": 0, "partial": 0, "unreliable": 0, "open": 0}
    for r in rows:
        q = str(r.get("data_quality") or "").strip().lower()
        if q in out:
            out[q] += 1
    return out


def _kpi(rows: list[dict]) -> dict:
    buys = [r for r in rows if r["_kind"] == "BUY"]
    waits = [r for r in rows if r["_kind"] == "WAIT_HOT"]
    legacy = [r for r in rows if r["_kind"] == "LEGACY"]
    buy_closed = [r for r in buys if r["_closed"]]
    buy_open = [r for r in buys if not r["_closed"]]
    buy_rel = [
        r for r in buy_closed
        if str(r.get("data_quality") or "").strip().lower() == "reliable"
    ]
    buy_rel_tp = [r for r in buy_rel if str(r.get("status") or "").startswith("tp")]
    buy_rel_stop = [r for r in buy_rel if str(r.get("status") or "") == "stop_hit"]
    buy_rel_exp = [r for r in buy_rel if str(r.get("status") or "") == "expired"]
    wr = None
    scored = [r for r in buy_rel if _num(r.get("option_pnl_pct")) is not None]
    if scored:
        wins = [r for r in scored if _num(r.get("option_pnl_pct")) > 0]
        wr = 100.0 * len(wins) / len(scored)
    return {
        "buys": buys,
        "waits": waits,
        "legacy": legacy,
        "buy_closed": buy_closed,
        "buy_open": buy_open,
        "buy_rel": buy_rel,
        "buy_rel_tp": buy_rel_tp,
        "buy_rel_stop": buy_rel_stop,
        "buy_rel_exp": buy_rel_exp,
        "wr": wr,
        "n_rel": len(buy_rel),
        "ready": len(buy_rel) >= STUDY_N,
        "quality": _count_quality(rows),
        "open_rows": [r for r in rows if not r["_closed"]],
        "wait_open": [r for r in waits if not r["_closed"]],
    }


def _pnl_wr_suffix(rows: list[dict]) -> str:
    scored = [r for r in rows if _num(r.get("option_pnl_pct")) is not None]
    if not scored:
        return ""
    wins = [r for r in scored if _num(r.get("option_pnl_pct")) > 0]
    return f" (ربح عقد {100.0 * len(wins) / len(scored):.0f}%)"


def _dte_val(row: dict):
    return _num(row.get("dte_num"), digits=1)


def _kpi_block(k: dict, heading: str = "## قياس النظام") -> list[str]:
    n_rel = k["n_rel"]
    rel = k["buy_rel"]
    dte_short = [r for r in rel if (_dte_val(r) is not None and _dte_val(r) <= 1)]
    spy = [r for r in rel if str(r.get("ticker") or "").strip().upper() == "SPY"]
    names = [r for r in rel if str(r.get("ticker") or "").strip().upper() != "SPY"]
    lines = [
        heading,
        "",
        f"- العينة: **{n_rel} / {STUDY_N}**"
        + (" — جاهزة للدراسة." if k["ready"] else " — تحت العتبة، لا حكم على النظام."),
        f"- هدف: {len(k['buy_rel_tp'])} · وقف: {len(k['buy_rel_stop'])} · BUY مفتوح: {len(k['buy_open'])} · BUY مغلق الكل: {len(k['buy_closed'])}",
    ]
    if k["wr"] is not None and n_rel:
        lines.append(
            f"- ربح عقد (موثوق فقط): {k['wr']:.0f}% — لا يُخلط مع يوليو بلا نوع ولا WAIT_HOT."
        )
    else:
        lines.append("- ربح عقد: — حتى تتوفر صفقات BUY موثوقة.")
    lines += [
        f"- WAIT_HOT في السجل: {len(k['waits'])} (دفتر مراقبة، خارج KPI).",
        f"- يوليو/بلا نوع: {len(k['legacy'])} — خارج نسب BUY.",
        "",
        "تشخيص العينة — BUY موثوق مغلق فقط، ليست توصية وليست بطاقة:",
        "",
        f"- DTE ≤ 1 يوم: {len(dte_short)} صف{_pnl_wr_suffix(dte_short)}.",
        f"- SPY: {len(spy)} صف{_pnl_wr_suffix(spy)} · باقي الأسماء: {len(names)} صف{_pnl_wr_suffix(names)}.",
        f"- وقف: {len(k['buy_rel_stop'])}.",
        f"- انتهاء: {len(k.get('buy_rel_exp') or [])}.",
        "",
    ]
    return lines


def _front_matter(**fields) -> list[str]:
    """YAML أمام الملاحظة — الوسوم تثبّت ألوان رسم أوبسيديان حسب الاختصاص."""
    lines = ["---"]
    for key, val in fields.items():
        if isinstance(val, (list, tuple)):
            inner = ", ".join(str(x) for x in val)
            lines.append(f"{key}: [{inner}]")
        else:
            lines.append(f"{key}: {val}")
    lines += ["---", ""]
    return lines


def _day_nav(session_date: str, all_dates: list[str]) -> str:
    prev_d = next_d = ""
    if session_date in all_dates:
        i = all_dates.index(session_date)
        if i > 0:
            prev_d = all_dates[i - 1]
        if i + 1 < len(all_dates):
            next_d = all_dates[i + 1]
    # تواريخ المجاورة نصاً بلا ويكي — حتى لا تتشابك خطوط الأيام في رسم أوبسيديان
    parts = []
    if prev_d:
        parts.append(f"← {prev_d}")
    parts.append("[[السجل]]")
    if next_d:
        parts.append(f"{next_d} →")
    return " · ".join(parts)


def build_day_markdown(
    df: pd.DataFrame,
    session_date: str,
    all_dates: list[str] | None = None,
) -> str:
    """يوم واحد يذهب إلى السجل. لا يربط فروع التدقيق حتى يبقى الرسم نجمة حول السجل."""
    rows = annotate(df)
    dates = all_dates if all_dates is not None else session_dates(df)
    day = [
        r for r in rows
        if r["_date"] == session_date or r["_exit"] == session_date
    ]
    n_new = len([r for r in day if r["_date"] == session_date])
    n_closed = len([r for r in day if r["_exit"] == session_date])
    lines = _front_matter(
        date=session_date,
        source="outcomes.csv",
        engine="none",
        tags=["يوم"],
    )
    lines += [
        f"# {session_date}",
        "",
        _day_nav(session_date, dates),
        "",
        f"يذهب إلى [[السجل]]. سُجّل **{n_new}** · أُغلق **{n_closed}**.",
        "",
        _table(day),
        "[[السجل]]",
        "",
    ]
    return "\n".join(lines)


def build_fail_markdown(
    df: pd.DataFrame,
    reason: str,
    last_close: str,
) -> str:
    rows = annotate(df)
    dates = [d for d in session_dates(df) if d <= last_close]
    first = dates[0] if dates else last_close
    chunk = [r for r in rows if r.get("_fail") == reason]
    title = "بلا اعتماد" if reason == "هدف بلا اعتماد" else reason
    fail_tag = {"وقف": "وقف", "انتهاء": "انتهاء", "هدف بلا اعتماد": "بلا-اعتماد"}[reason]
    lines = _front_matter(
        **{
            "branch": title,
            "from": first,
            "to": last_close,
            "source": "outcomes.csv",
            "engine": "none",
            "tags": ["فشل-فرع", fail_tag],
        }
    )
    lines += [
        f"# {title}",
        "",
        f"متفرع من [[فشلت]]. من **{first}** إلى **{last_close}** — {len(chunk)} صف.",
        "",
        _table(chunk),
        "",
    ]
    return "\n".join(lines)


def build_branch_markdown(
    df: pd.DataFrame,
    name: str,
    last_close: str,
) -> str:
    rows = annotate(df)
    dates = [d for d in session_dates(df) if d <= last_close]
    first = dates[0] if dates else last_close
    n = len(_group_rows(rows, name))
    extra = ""
    if name == "فشلت":
        extra = "يتفرع أيضاً إلى [[وقف]] · [[انتهاء]] · [[بلا اعتماد]]."
    branch_tag = {"تحققت": "تحقق", "فشلت": "فشل", "انتظار": "انتظار"}[name]
    lines = _front_matter(
        **{
            "branch": name,
            "from": first,
            "to": last_close,
            "source": "outcomes.csv",
            "engine": "none",
            "tags": [branch_tag],
        }
    )
    lines += [
        f"# {name}",
        "",
        f"متفرع من [[السجل]]. من **{first}** إلى **{last_close}** — {n} صف.",
        "",
    ]
    if extra:
        lines += [extra, ""]
    lines += [
        "يتحدّث تلقائياً بعد إغلاق السوق. لا يتحدّث أثناء التداول.",
        "",
    ]
    lines += _group_body(name, rows)
    lines += [
        "[[السجل]]",
        "",
    ]
    return "\n".join(lines)


def build_ledger_markdown(df: pd.DataFrame, last_close: str) -> str:
    rows = annotate(df)
    k = _kpi(rows)
    dates = [d for d in session_dates(df) if d <= last_close]
    first = dates[0] if dates else last_close
    counts = {name: len(_group_rows(rows, name)) for name in BRANCH_NOTES}
    fail_counts = {
        reason: len([r for r in rows if r.get("_fail") == reason])
        for reason in FAIL_NOTES
    }
    day_links = " · ".join(f"[[{d}]]" for d in dates)

    lines = _front_matter(
        **{
            "from": first,
            "to": last_close,
            "source": "outcomes.csv",
            "engine": "none",
            "tags": ["سجل"],
        }
    )
    lines += [
        "# السجل",
        "",
        f"من **{first}** إلى آخر إغلاق **{last_close}**. المصدر: `outcomes.csv` فقط.",
        "",
        "ليست توصية. ليست شاشة تنفيذ. لا ترقية WAIT→BUY.",
        "",
        "## الأيام",
        "",
        day_links if day_links else "_لا أيام._",
        "",
        "## يتفرع بعد التدقيق",
        "",
        f"- [[تحققت]] — {counts['تحققت']}",
        f"- [[فشلت]] — {counts['فشلت']} → وقف ({fail_counts['وقف']}) · انتهاء ({fail_counts['انتهاء']}) · بلا اعتماد ({fail_counts['هدف بلا اعتماد']})",
        f"- [[انتظار]] — {counts['انتظار']}",
        "",
        "**التحديث:** تلقائي بعد إغلاق السوق. الأيام تُضاف هنا، وما أُغلق ينتقل من [[انتظار]] إلى [[تحققت]] أو [[فشلت]]. لا يتحدّث أثناء التداول.",
        "",
    ]
    lines += _kpi_block(k)
    lines += [
        "[[القاعدة]] · [[الشركاء]]",
        "",
    ]
    return "\n".join(lines)


def vault_dir() -> Path | None:
    raw = os.environ.get("SESSION_NOTE_VAULT", str(DEFAULT_VAULT))
    p = Path(raw).expanduser()
    if p.parent.exists():
        return p
    return None


def _dest_dirs(out_dir: Path, copy_vault: bool) -> list[Path]:
    dests = [out_dir]
    if not copy_vault:
        return dests
    vault = vault_dir()
    if vault is None:
        return dests
    try:
        vault.mkdir(parents=True, exist_ok=True)
        if vault.resolve() != out_dir.resolve():
            dests.append(vault)
    except OSError as exc:
        print(f"⚠ لم يُنسخ إلى أوبسيديان ({vault}): {exc}")
    return dests


def _write_text(dests: list[Path], name: str, body: str) -> list[Path]:
    written = []
    for d in dests:
        path = d / name
        path.write_text(body, encoding="utf-8")
        written.append(path)
    return written


def write_ledger_note(
    df: pd.DataFrame,
    last_close: str,
    out_dir: Path | None = None,
    copy_vault: bool = True,
) -> list[Path]:
    out_dir = Path(out_dir) if out_dir else SESSIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    body = build_ledger_markdown(df, last_close)
    return _write_text(_dest_dirs(out_dir, copy_vault), LEDGER_NOTE, body)


def write_branch_note(
    df: pd.DataFrame,
    name: str,
    last_close: str,
    out_dir: Path | None = None,
    copy_vault: bool = True,
) -> list[Path]:
    out_dir = Path(out_dir) if out_dir else SESSIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    body = build_branch_markdown(df, name, last_close)
    return _write_text(_dest_dirs(out_dir, copy_vault), BRANCH_NOTES[name], body)


def write_day_note(
    df: pd.DataFrame,
    session_date: str,
    out_dir: Path | None = None,
    copy_vault: bool = True,
    all_dates: list[str] | None = None,
) -> list[Path]:
    out_dir = Path(out_dir) if out_dir else SESSIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    body = build_day_markdown(df, session_date, all_dates=all_dates)
    return _write_text(_dest_dirs(out_dir, copy_vault), f"{session_date}.md", body)


def write_fail_note(
    df: pd.DataFrame,
    reason: str,
    last_close: str,
    out_dir: Path | None = None,
    copy_vault: bool = True,
) -> list[Path]:
    out_dir = Path(out_dir) if out_dir else SESSIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    body = build_fail_markdown(df, reason, last_close)
    return _write_text(_dest_dirs(out_dir, copy_vault), FAIL_NOTES[reason], body)


def prune_stale_notes(out_dir: Path, keep_dates: list[str]) -> None:
    if not out_dir.exists():
        return
    keep = set(keep_dates)
    for name in STALE_NOTES:
        path = out_dir / name
        if path.exists():
            path.unlink()
    for path in out_dir.glob("*.md"):
        stem = path.stem
        if len(stem) == 10 and stem[4] == "-" and stem[7] == "-" and stem[:4].isdigit():
            if stem not in keep:
                path.unlink()


def write_all_notes(
    df: pd.DataFrame,
    out_dir: Path | None = None,
    copy_vault: bool = True,
    now: datetime | None = None,
) -> list[Path]:
    out_dir = Path(out_dir) if out_dir else SESSIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    last = last_close_date(df, now=now)
    dates = [d for d in session_dates(df) if d <= last]
    written: list[Path] = []
    written.extend(write_ledger_note(df, last, out_dir=out_dir, copy_vault=copy_vault))
    for name in BRANCH_NOTES:
        written.extend(
            write_branch_note(df, name, last, out_dir=out_dir, copy_vault=copy_vault)
        )
    for reason in FAIL_NOTES:
        written.extend(
            write_fail_note(df, reason, last, out_dir=out_dir, copy_vault=copy_vault)
        )
    for d in dates:
        written.extend(
            write_day_note(
                df, d, out_dir=out_dir, copy_vault=copy_vault, all_dates=dates
            )
        )
    prune_stale_notes(out_dir, dates)
    if copy_vault:
        vault = vault_dir()
        if vault is not None and vault.resolve() != out_dir.resolve():
            prune_stale_notes(vault, dates)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="أيام → السجل → تحققت / فشلت / انتظار")
    p.add_argument("--csv", default=str(OUTCOMES_FILE))
    p.add_argument("--out", default=str(SESSIONS_DIR))
    p.add_argument("--no-vault", action="store_true")
    args = p.parse_args(argv)

    df = load_outcomes(Path(args.csv))
    paths = write_all_notes(
        df, out_dir=Path(args.out), copy_vault=not args.no_vault
    )
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        print(f"✅ {path.name} → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
