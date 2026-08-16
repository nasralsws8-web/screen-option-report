"""ملف الجلسة والسجل يُبنيان من outcomes.csv فقط — بلا ترقية توصية."""

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
    build_ledger_markdown,
    build_markdown,
    default_session_date,
    last_close_date,
    rec_kind,
    session_dates,
    write_all_notes,
    write_session_note,
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

    def test_daily_note_is_that_day_only_and_links_ledger(self):
        df = pd.DataFrame([
            _row(ticker="JUL", rec_kind="", date="2026-07-24", exit_date="2026-07-24",
                 status="tp1_hit", data_quality="partial"),
            _row(ticker="BUY1", rec_kind="BUY", date="2026-08-13", exit_date="2026-08-14"),
        ])
        md = build_markdown(df, "2026-07-24", all_dates=["2026-07-24", "2026-08-13", "2026-08-14"])
        self.assertIn("| JUL |", md)
        self.assertNotIn("| BUY1 |", md)
        self.assertIn("[[السجل]]", md)
        self.assertIn("[[2026-08-13]]", md)
        self.assertIn("هذا اليوم فقط", md)
        self.assertIn("## فشلت", md)
        self.assertIn("هدف بلا اعتماد", md)
        self.assertNotIn("## كل الصفوف", md)

    def test_ledger_splits_after_audit(self):
        df = pd.DataFrame([
            _row(ticker="JUL", rec_kind="", date="2026-07-24", exit_date="2026-07-24",
                 status="tp1_hit", data_quality="partial"),
            _row(ticker="HOT", rec_kind="WAIT_HOT", option_pnl_pct=80, data_quality="reliable"),
            _row(ticker="BUY1", rec_kind="BUY", option_pnl_pct=20, data_quality="reliable"),
            _row(ticker="BUY2", rec_kind="BUY", status="stop_hit",
                 option_pnl_pct=-40, data_quality="reliable"),
            _row(ticker="OPEN", rec_kind="BUY", status="open", exit_date="",
                 option_pnl_pct="", data_quality="open"),
        ])
        md = build_ledger_markdown(df, "2026-08-14")
        verified = md[md.index("## محققة"): md.index("## فشلت")]
        failed = md[md.index("## فشلت"): md.index("## انتظار")]
        waiting = md[md.index("## انتظار"):]
        self.assertIn("| BUY1 |", verified)
        self.assertIn("| HOT |", verified)
        self.assertNotIn("| BUY2 |", verified)
        self.assertIn("| BUY2 |", failed)
        self.assertIn("| JUL |", failed)
        self.assertIn("هدف بلا اعتماد", failed)
        self.assertIn("| OPEN |", waiting)
        self.assertNotIn("| OPEN |", verified)
        self.assertIn("2 / 30", md)
        self.assertIn("هدف: 1 · وقف: 1", md)
        self.assertIn("WAIT_HOT في السجل: 1", md)
        self.assertIn("يوليو/بلا نوع: 1", md)
        self.assertIn("[[2026-07-24]]", md)
        self.assertIn("[[2026-08-14]]", md)
        self.assertIn("from: 2026-07-24", md)
        self.assertIn("to: 2026-08-14", md)
        self.assertIn("لا ترقية WAIT→BUY", md)
        self.assertIn("تلقائي بعد إغلاق السوق", md)
        self.assertNotIn("## كل الصفوف", md)

    def test_write_all_covers_start_to_last_close(self):
        df = pd.DataFrame([
            _row(ticker="JUL", rec_kind="", date="2026-07-24", exit_date="2026-07-24"),
            _row(ticker="BUY1", date="2026-08-13", exit_date="2026-08-14"),
        ])
        sun = datetime(2026, 8, 16, 17, 0, tzinfo=ET)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_all_notes(
                df, out_dir=Path(tmp), copy_vault=False, now=sun
            )
            names = sorted({p.name for p in paths})
            self.assertEqual(
                names,
                ["2026-07-24.md", "2026-08-13.md", "2026-08-14.md", "السجل.md"],
            )
            ledger = (Path(tmp) / "السجل.md").read_text(encoding="utf-8")
            self.assertIn(f"{STUDY_N}", ledger)
            self.assertIn("| JUL |", ledger)
            self.assertIn("| BUY1 |", ledger)

    def test_writes_file_named_by_session_date(self):
        df = pd.DataFrame([_row()])
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_session_note(
                df, "2026-08-14", out_dir=Path(tmp), copy_vault=False
            )
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].name, "2026-08-14.md")
            body = paths[0].read_text(encoding="utf-8")
            self.assertIn("source: outcomes.csv", body)
            self.assertIn("engine: none", body)
            self.assertIn("[[السجل]]", body)
            self.assertIn("[[القاعدة]]", body)


if __name__ == "__main__":
    unittest.main()
