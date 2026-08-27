"""الأيام تذهب إلى السجل، ومنه تتفرع تحققت / فشلت / انتظار."""

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
    build_fail_markdown,
    build_branch_markdown,
    build_day_markdown,
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

    def test_session_dates_span_rec_and_exit(self):
        df = pd.DataFrame([
            _row(date="2026-07-24", ticker="JUL", rec_kind="", exit_date="2026-07-24"),
            _row(date="2026-08-13", ticker="AAA", exit_date="2026-08-14"),
        ])
        self.assertEqual(session_dates(df), ["2026-07-24", "2026-08-13", "2026-08-14"])
        sun = datetime(2026, 8, 16, 17, 0, tzinfo=ET)
        self.assertEqual(last_close_date(df, now=sun), "2026-08-14")

    def test_days_go_to_ledger_not_to_branches(self):
        md = build_day_markdown(
            _df(), "2026-07-24", all_dates=["2026-07-24", "2026-08-13", "2026-08-14"]
        )
        self.assertIn("| JUL |", md)
        self.assertNotIn("| BUY1 |", md)
        self.assertIn("[[السجل]]", md)
        self.assertNotIn("[[تحققت]]", md)
        self.assertNotIn("[[القاعدة]]", md)
        self.assertNotIn("[[2026-08-13]]", md)
        self.assertIn("2026-08-13 →", md)

    def test_hub_links_days_and_branches(self):
        md = build_ledger_markdown(_df(), "2026-08-14")
        self.assertIn("[[تحققت]]", md)
        self.assertIn("[[فشلت]]", md)
        self.assertIn("[[انتظار]]", md)
        self.assertNotIn("[[وقف]]", md)
        self.assertIn("وقف (", md)
        self.assertIn("[[2026-07-24]]", md)
        self.assertIn("[[2026-08-14]]", md)
        self.assertNotIn("| BUY1 |", md)
        self.assertIn("2 / 30", md)
        self.assertIn("[[القاعدة]]", md)
        self.assertIn("tags: [سجل]", md)
        self.assertIn("تشخيص العينة", md)
        self.assertIn("DTE ≤ 1 يوم:", md)
        self.assertIn("انتهاء:", md)

    def test_study_diag_is_buy_only(self):
        df = pd.DataFrame([
            _row(ticker="SPY", dte_num=1, status="expired", option_pnl_pct=-100,
                 rec_kind="BUY", data_quality="reliable"),
            _row(ticker="HOT", rec_kind="WAIT_HOT", dte_num=1, status="tp1_hit",
                 option_pnl_pct=80, data_quality="reliable"),
            _row(ticker="AAA", dte_num=7, status="stop_hit", option_pnl_pct=-50,
                 rec_kind="BUY", data_quality="reliable"),
        ])
        md = build_ledger_markdown(df, "2026-08-14")
        self.assertIn("DTE ≤ 1 يوم: 1 صف (ربح عقد 0%).", md)
        self.assertIn("SPY: 1 صف (ربح عقد 0%) · باقي الأسماء: 1 صف (ربح عقد 0%).", md)
        self.assertIn("- وقف: 1.", md)
        self.assertIn("- انتهاء: 1.", md)
        self.assertIn("WAIT_HOT في السجل: 1", md)

    def test_days_tagged_for_graph(self):
        md = build_day_markdown(
            _df(), "2026-07-24", all_dates=["2026-07-24", "2026-08-13"]
        )
        self.assertIn("tags: [يوم]", md)

    def test_branches_hold_the_rows(self):
        df = _df()
        verified = build_branch_markdown(df, "تحققت", "2026-08-14")
        failed = build_branch_markdown(df, "فشلت", "2026-08-14")
        waiting = build_branch_markdown(df, "انتظار", "2026-08-14")
        self.assertIn("| BUY1 |", verified)
        self.assertIn("متفرع من [[السجل]]", verified)
        self.assertIn("| BUY2 |", failed)
        self.assertIn("| JUL |", failed)
        self.assertIn("| OPEN |", waiting)
        self.assertNotIn("| OPEN |", verified)
        self.assertIn("[[وقف]]", failed)
        self.assertIn("tags: [تحقق]", verified)
        self.assertIn("tags: [فشل]", failed)
        self.assertIn("tags: [انتظار]", waiting)
        stop = build_fail_markdown(_df(), "وقف", "2026-08-14")
        self.assertIn("tags: [فشل-فرع, وقف]", stop)
        self.assertIn("[[فشلت]]", stop)
        self.assertNotIn("[[السجل]]", stop)
        self.assertNotIn("[[تحققت]]", stop)

    def test_write_all_restores_days_and_graph_hub(self):
        df = pd.DataFrame([
            _row(ticker="JUL", rec_kind="", date="2026-07-24", exit_date="2026-07-24",
                 status="tp1_hit", data_quality="partial"),
            _row(ticker="BUY1", date="2026-08-13", exit_date="2026-08-14"),
        ])
        sun = datetime(2026, 8, 16, 17, 0, tzinfo=ET)
        with tempfile.TemporaryDirectory() as tmp:
            stale = Path(tmp) / "محققة.md"
            stale.write_text("old", encoding="utf-8")
            paths = write_all_notes(
                df, out_dir=Path(tmp), copy_vault=False, now=sun
            )
            names = {p.name for p in paths}
            self.assertIn("السجل.md", names)
            self.assertIn("تحققت.md", names)
            self.assertIn("فشلت.md", names)
            self.assertIn("انتظار.md", names)
            self.assertIn("2026-07-24.md", names)
            self.assertIn("2026-08-14.md", names)
            self.assertIn("وقف.md", names)
            self.assertFalse(stale.exists())
            hub = (Path(tmp) / "السجل.md").read_text(encoding="utf-8")
            self.assertIn(f"{STUDY_N}", hub)
            self.assertIn("[[2026-07-24]]", hub)
            self.assertIn("[[تحققت]]", hub)
            day = (Path(tmp) / "2026-07-24.md").read_text(encoding="utf-8")
            self.assertIn("[[السجل]]", day)
            self.assertNotIn("[[تحققت]]", day)


if __name__ == "__main__":
    unittest.main()
