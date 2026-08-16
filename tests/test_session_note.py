"""السجل يتفرع إلى محققة / فشلت / انتظار — من outcomes.csv فقط."""

import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from session_note import (  # noqa: E402
    STUDY_N,
    build_branch_markdown,
    build_ledger_markdown,
    default_session_date,
    last_close_date,
    rec_kind,
    session_dates,
    write_all_notes,
)

ET = ZoneInfo("America/New_York")


def _row(**kw):
    base = {
        "date": "2026-08-13",
        "ticker": "AAA",
        "direction": "CALL",
        "rec_kind": "BUY",
        "wait_tier": "",
        "price_at_rec": 10.0,
        "entry_stock": 10.0,
        "exit_stock": 11.0,
        "premium": 1.5,
        "option_pnl_pct": 20.0,
        "status": "tp1_hit",
        "exit_date": "2026-08-14",
        "data_quality": "reliable",
    }
    base.update(kw)
    return base


def _df():
    return pd.DataFrame([
        _row(ticker="JUL", rec_kind="", date="2026-07-24", exit_date="2026-07-24",
             status="tp1_hit", data_quality="partial"),
        _row(ticker="HOT", rec_kind="WAIT_HOT", option_pnl_pct=80, data_quality="reliable"),
        _row(ticker="BUY1", rec_kind="BUY", option_pnl_pct=20, data_quality="reliable"),
        _row(ticker="BUY2", rec_kind="BUY", status="stop_hit",
             option_pnl_pct=-40, data_quality="reliable"),
        _row(ticker="OPEN", rec_kind="BUY", status="open", exit_date="",
             option_pnl_pct="", data_quality="open"),
    ])


class TestSessionNote(unittest.TestCase):
    def test_weekend_falls_back_to_friday(self):
        sun = datetime(2026, 8, 16, 17, 0, tzinfo=ET)
        self.assertEqual(default_session_date(sun), "2026-08-14")

    def test_weekday_keeps_et_date(self):
        fri = datetime(2026, 8, 14, 16, 40, tzinfo=ET)
        self.assertEqual(default_session_date(fri), "2026-08-14")

    def test_july_empty_rec_kind_is_legacy_not_buy(self):
        self.assertEqual(rec_kind({"rec_kind": "", "wait_tier": ""}), "LEGACY")
        self.assertEqual(rec_kind({"rec_kind": "BUY"}), "BUY")
        self.assertEqual(rec_kind({"rec_kind": "WAIT_HOT"}), "WAIT_HOT")
        self.assertEqual(rec_kind({"rec_kind": "", "wait_tier": "FIRE"}), "WAIT_HOT")

    def test_session_dates_span_rec_and_exit(self):
        df = pd.DataFrame([
            _row(date="2026-07-24", ticker="JUL", rec_kind="", exit_date="2026-07-24"),
            _row(date="2026-08-13", ticker="AAA", exit_date="2026-08-14"),
        ])
        self.assertEqual(session_dates(df), ["2026-07-24", "2026-08-13", "2026-08-14"])
        sun = datetime(2026, 8, 16, 17, 0, tzinfo=ET)
        self.assertEqual(last_close_date(df, now=sun), "2026-08-14")

    def test_hub_has_no_row_dump_and_links_branches(self):
        md = build_ledger_markdown(_df(), "2026-08-14")
        self.assertIn("[[محققة]]", md)
        self.assertIn("[[فشلت]]", md)
        self.assertIn("[[انتظار]]", md)
        self.assertIn("2 / 30", md)
        self.assertIn("هدف: 1 · وقف: 1", md)
        self.assertNotIn("| BUY1 |", md)
        self.assertNotIn("| JUL |", md)
        self.assertNotIn("## كل الصفوف", md)
        self.assertNotIn("[[2026-07-24]]", md)
        self.assertIn("لا ترقية WAIT→BUY", md)
        self.assertIn("تلقائي بعد إغلاق السوق", md)

    def test_branches_hold_the_rows(self):
        df = _df()
        verified = build_branch_markdown(df, "محققة", "2026-08-14")
        failed = build_branch_markdown(df, "فشلت", "2026-08-14")
        waiting = build_branch_markdown(df, "انتظار", "2026-08-14")
        self.assertIn("| BUY1 |", verified)
        self.assertIn("| HOT |", verified)
        self.assertNotIn("| BUY2 |", verified)
        self.assertIn("متفرع من [[السجل]]", verified)
        self.assertIn("| BUY2 |", failed)
        self.assertIn("| JUL |", failed)
        self.assertIn("هدف بلا اعتماد", failed)
        self.assertIn("| OPEN |", waiting)
        self.assertNotIn("| OPEN |", verified)

    def test_write_all_is_hub_plus_three_branches(self):
        df = pd.DataFrame([
            _row(ticker="JUL", rec_kind="", date="2026-07-24", exit_date="2026-07-24",
                 status="tp1_hit", data_quality="partial"),
            _row(ticker="BUY1", date="2026-08-13", exit_date="2026-08-14"),
        ])
        sun = datetime(2026, 8, 16, 17, 0, tzinfo=ET)
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "2026-07-24.md"
            old.write_text("stale", encoding="utf-8")
            paths = write_all_notes(
                df, out_dir=Path(tmp), copy_vault=False, now=sun
            )
            names = sorted({p.name for p in paths})
            self.assertEqual(
                set(names),
                {"السجل.md", "محققة.md", "فشلت.md", "انتظار.md"},
            )
            self.assertFalse(old.exists())
            hub = (Path(tmp) / "السجل.md").read_text(encoding="utf-8")
            self.assertIn(f"{STUDY_N}", hub)
            self.assertIn("[[محققة]]", hub)
            self.assertNotIn("| JUL |", hub)
            verified = (Path(tmp) / "محققة.md").read_text(encoding="utf-8")
            self.assertIn("| BUY1 |", verified)
            failed = (Path(tmp) / "فشلت.md").read_text(encoding="utf-8")
            self.assertIn("| JUL |", failed)


if __name__ == "__main__":
    unittest.main()
