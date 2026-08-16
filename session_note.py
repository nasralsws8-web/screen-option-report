"""
ملفات أوبسيديان من outcomes.csv فقط: جلسة لكل يوم + السجل الكامل.

لا يلمس المحرك، لا يرقّي WAIT→BUY، لا يخمن أسعاراً.
يُشغَّل بعد EOD حين يكون السجل قد حُدِّث.
"""

from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta
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
    """بعد التدقيق: محققة = هدف + جودة معتمدة. انتظار = لم يُغلق. الباقي فشلت."""
    closed = row["_closed"] if "_closed" in row else is_closed(row)
    if not closed:
        return "انتظار"
    if _quality(row) == "reliable" and _status(row).startswith("tp"):
        return "محققة"
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


def _audit_sections(rows: list[dict]) -> list[str]:
    groups = {
        "محققة": [r for r in rows if r["_group"] == "محققة"],
        "فشلت": [r for r in rows if r["_group"] == "فشلت"],
        "انتظار": [r for r in rows if r["_group"] == "انتظار"],
    }
    blurbs = {
        "محققة": "أُغلق على هدف، والتدقيق اعتمد الصف (جودة موثوق). القياس فقط من BUY هنا.",
        "فشلت": "وقف، أو انتهاء، أو هدف لكن التدقيق لم يعتمدها (جزئي/ضعيف).",
        "انتظار": "ما زال مفتوحاً في السجل. ليس قائمة تنفيذ اليوم.",
    }
    lines: list[str] = []
    for name in ("محققة", "فشلت", "انتظار"):
        chunk = groups[name]
        lines += [
            f"## {name} ({len(chunk)})",
            "",
            blurbs[name],
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
        "wr": wr,
        "n_rel": len(buy_rel),
        "ready": len(buy_rel) >= STUDY_N,
        "quality": _count_quality(rows),
        "open_rows": [r for r in rows if not r["_closed"]],
        "wait_open": [r for r in waits if not r["_closed"]],
    }


def _kpi_block(k: dict, heading: str = "## قياس النظام (BUY مغلق موثوق)") -> list[str]:
    n_rel = k["n_rel"]
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
    ]
    return lines


def _nav(session_date: str, all_dates: list[str] | None) -> str:
    dates = all_dates or []
    prev_d = next_d = ""
    if session_date in dates:
        i = dates.index(session_date)
        if i > 0:
            prev_d = dates[i - 1]
        if i + 1 < len(dates):
            next_d = dates[i + 1]
    parts = []
    if prev_d:
        parts.append(f"← [[{prev_d}]]")
    parts.append("[[السجل]]")
    if next_d:
        parts.append(f"[[{next_d}]] →")
    return " · ".join(parts)


def build_markdown(
    df: pd.DataFrame,
    session_date: str,
    all_dates: list[str] | None = None,
) -> str:
    """يوم واحد بعد التدقيق: محققة / فشلت / انتظار. القياس الكامل في [[السجل]]."""
    rows = annotate(df)
    dates = all_dates if all_dates is not None else session_dates(df)
    day = [
        r for r in rows
        if r["_date"] == session_date or r["_exit"] == session_date
    ]
    n_new = len([r for r in day if r["_date"] == session_date])
    n_closed = len([r for r in day if r["_exit"] == session_date])

    lines = [
        "---",
        f"date: {session_date}",
        "source: outcomes.csv",
        "engine: none",
        "---",
        "",
        f"# جلسة {session_date}",
        "",
        _nav(session_date, dates),
        "",
        "مصدر: السجل بعد الإغلاق فقط. ليست توصية وليست شاشة تنفيذ. لا ترقية WAIT→BUY.",
        "",
        "هذا اليوم فقط. السجل الكامل: [[السجل]].",
        "",
        f"سُجّل اليوم **{n_new}** · أُغلق اليوم **{n_closed}**.",
        "",
        "يتحدّث تلقائياً بعد إغلاق السوق من السجل. لا يتحدّث أثناء التداول.",
        "",
    ]
    lines += _audit_sections(day)
    lines += [
        "[[القاعدة]] · [[الشركاء]] · [[السجل]]",
        "",
    ]
    return "\n".join(lines)


def build_ledger_markdown(df: pd.DataFrame, last_close: str) -> str:
    rows = annotate(df)
    k = _kpi(rows)
    dates = [d for d in session_dates(df) if d <= last_close]
    first = dates[0] if dates else last_close
    day_links = " · ".join(f"[[{d}]]" for d in dates)

    lines = [
        "---",
        f"from: {first}",
        f"to: {last_close}",
        "source: outcomes.csv",
        "engine: none",
        "---",
        "",
        "# السجل",
        "",
        f"من **{first}** إلى آخر إغلاق **{last_close}**. المصدر: `outcomes.csv` فقط.",
        "",
        "ليست توصية. ليست شاشة تنفيذ. لا ترقية WAIT→BUY.",
        "",
        "**التحديث:** تلقائي بعد إغلاق السوق (4:30 و 5:30 ET) من السجل. يُضاف ما أُغلق ويُنقل من انتظار. لا يتحدّث أثناء التداول. على الجهاز يظهر بعد سحب GitHub.",
        "",
    ]
    lines += _kpi_block(k, heading="## قياس النظام")
    lines += _audit_sections(rows)
    lines += [
        "## الأيام",
        "",
        day_links if day_links else "_لا أيام._",
        "",
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


def write_session_note(
    df: pd.DataFrame,
    session_date: str,
    out_dir: Path | None = None,
    copy_vault: bool = True,
    all_dates: list[str] | None = None,
) -> list[Path]:
    out_dir = Path(out_dir) if out_dir else SESSIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    body = build_markdown(df, session_date, all_dates=all_dates)
    return _write_text(_dest_dirs(out_dir, copy_vault), f"{session_date}.md", body)


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


def write_all_notes(
    df: pd.DataFrame,
    out_dir: Path | None = None,
    copy_vault: bool = True,
    now: datetime | None = None,
) -> list[Path]:
    last = last_close_date(df, now=now)
    dates = [d for d in session_dates(df) if d <= last]
    written: list[Path] = []
    written.extend(write_ledger_note(df, last, out_dir=out_dir, copy_vault=copy_vault))
    for d in dates:
        written.extend(
            write_session_note(
                df, d, out_dir=out_dir, copy_vault=copy_vault, all_dates=dates
            )
        )
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ملفات جلسة وسجل من outcomes.csv فقط")
    p.add_argument("--date", help="YYYY-MM-DD ليوم واحد. بدونها: كل الأيام حتى آخر إغلاق")
    p.add_argument("--csv", default=str(OUTCOMES_FILE))
    p.add_argument("--out", default=str(SESSIONS_DIR))
    p.add_argument("--no-vault", action="store_true")
    p.add_argument("--all", action="store_true", help="كل الأيام + السجل الكامل (افتراضي إذا لم يُمرَّر --date)")
    args = p.parse_args(argv)

    df = load_outcomes(Path(args.csv))
    copy_vault = not args.no_vault
    out = Path(args.out)

    if args.date:
        date.fromisoformat(args.date)
        last = last_close_date(df)
        dates = [d for d in session_dates(df) if d <= last]
        paths = write_session_note(
            df, args.date, out_dir=out, copy_vault=copy_vault, all_dates=dates
        )
        paths.extend(write_ledger_note(df, last, out_dir=out, copy_vault=copy_vault))
    else:
        paths = write_all_notes(df, out_dir=out, copy_vault=copy_vault)

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
