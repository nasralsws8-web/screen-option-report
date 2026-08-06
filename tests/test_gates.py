"""اختبارات بوابات التشغيل: مطاردة، نافذة تنفيذ، جودة بيانات، dedupe تيليجرام."""

import os
import sys
import tempfile
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cheap_options_screener_v3 import entry_is_chased, is_exec_window  # noqa: E402
from data_quality import classify_data_quality  # noqa: E402
from outcome_tracker import resolve_first_touch  # noqa: E402
from telegram_notify import (  # noqa: E402
    alert_key,
    load_sent,
    prune_sent,
    save_sent,
    select_buy_rows,
)

ET = ZoneInfo("America/New_York")


class TestChase(unittest.TestCase):
    def test_call_not_chased_at_entry(self):
        self.assertFalse(entry_is_chased(100.0, 100.0, True))

    def test_call_chased_beyond_limit(self):
        self.assertTrue(entry_is_chased(100.5, 100.0, True))  # +0.5% > 0.3%

    def test_call_within_tolerance(self):
        self.assertFalse(entry_is_chased(100.2, 100.0, True))  # +0.2% < 0.3%

    def test_put_chased_below_entry(self):
        self.assertTrue(entry_is_chased(99.5, 100.0, False))


class TestExecWindow(unittest.TestCase):
    def setUp(self):
        os.environ.pop("IGNORE_EXEC_WINDOW", None)

    def test_before_945_closed(self):
        self.assertFalse(is_exec_window(datetime(2026, 8, 3, 9, 44, tzinfo=ET)))

    def test_at_945_open(self):
        self.assertTrue(is_exec_window(datetime(2026, 8, 3, 9, 45, tzinfo=ET)))

    def test_at_close_closed(self):
        self.assertFalse(is_exec_window(datetime(2026, 8, 3, 16, 0, tzinfo=ET)))

    def test_weekend_closed(self):
        self.assertFalse(is_exec_window(datetime(2026, 8, 1, 12, 0, tzinfo=ET)))

    def test_ignore_env(self):
        os.environ["IGNORE_EXEC_WINDOW"] = "1"
        try:
            self.assertTrue(is_exec_window(datetime(2026, 8, 1, 12, 0, tzinfo=ET)))
        finally:
            os.environ.pop("IGNORE_EXEC_WINDOW", None)


class TestDataQuality(unittest.TestCase):
    def test_open(self):
        q, _ = classify_data_quality({"status": "open"})
        self.assertEqual(q, "open")

    def test_no_entry(self):
        q, _ = classify_data_quality({
            "status": "tp1_hit", "entry_hit": False,
            "premium": 1.5, "strike": 100, "exit_premium_source": "market",
            "option_pnl_pct": 10,
        })
        self.assertEqual(q, "unreliable")

    def test_estimate_partial(self):
        q, _ = classify_data_quality({
            "status": "tp1_hit", "entry_hit": True,
            "premium": 1.5, "strike": 100, "exit_premium_source": "estimate",
            "option_pnl_pct": 5,
        })
        self.assertEqual(q, "partial")

    def test_stop_with_option_profit_unreliable(self):
        q, note = classify_data_quality({
            "status": "stop_hit", "entry_hit": True,
            "premium": 1.5, "strike": 100, "exit_premium_source": "market",
            "option_pnl_pct": 40,
        })
        self.assertEqual(q, "unreliable")
        self.assertIn("Stop", note)

    def test_reliable_market(self):
        q, _ = classify_data_quality({
            "status": "tp1_hit", "entry_hit": True,
            "premium": 1.5, "strike": 100, "exit_premium_source": "market",
            "option_pnl_pct": 20,
        })
        self.assertEqual(q, "reliable")

    def test_tp_estimate_big_loss_partial(self):
        q, _ = classify_data_quality({
            "status": "tp1_hit", "entry_hit": True,
            "premium": 2.0, "strike": 50, "exit_premium_source": "estimate",
            "option_pnl_pct": -55,
        })
        self.assertEqual(q, "partial")


class TestPreexistingLevels(unittest.TestCase):
    def _bar(self, o, h, l, c):
        return pd.DataFrame(
            {"Open": [o], "High": [h], "Low": [l], "Close": [c]},
            index=pd.to_datetime(["2026-08-05"]),
        )

    def test_skips_tp_when_open_already_above_target(self):
        # مثل AMZN: افتتاح 281 فوق TP1 275 مع لمس Entry 271 في نفس اليوم
        post = self._bar(281.59, 282.79, 270.74, 272.65)
        touch = resolve_first_touch(
            post, True, stop=269.08, tp1=275.15, tp2=281.07, tp3=287.2, entry=271.61,
        )
        self.assertIsNone(touch)

    def test_allows_tp_when_open_below_target(self):
        post = self._bar(272.0, 276.0, 271.0, 275.5)
        touch = resolve_first_touch(
            post, True, stop=269.08, tp1=275.15, tp2=281.07, tp3=287.2, entry=271.61,
        )
        self.assertIsNotNone(touch)
        self.assertEqual(touch[0], "tp1_hit")

    def test_skips_stop_when_open_already_through_stop(self):
        # CALL: افتتاح تحت Stop ثم ارتد ولمس Entry — لا Stop نظيف
        post = self._bar(268.50, 272.50, 268.00, 271.80)
        touch = resolve_first_touch(
            post, True, stop=269.08, tp1=275.15, tp2=281.07, tp3=287.2, entry=271.61,
        )
        self.assertIsNone(touch)

    def test_allows_stop_when_open_above_stop(self):
        # افتتاح فوق Stop ثم كسره — Stop صالح
        post = self._bar(272.0, 273.0, 268.5, 269.0)
        touch = resolve_first_touch(
            post, True, stop=269.08, tp1=275.15, tp2=281.07, tp3=287.2, entry=271.61,
        )
        self.assertIsNotNone(touch)
        self.assertEqual(touch[0], "stop_hit")


class TestTelegramDedupe(unittest.TestCase):
    def test_alert_key_stable(self):
        row = {
            "Ticker": "SPY", "direction": "CALL 📈",
            "strike": 773.0, "expiry": "2026-08-07",
        }
        k1 = alert_key(row, day="2026-08-05")
        k2 = alert_key(row, day="2026-08-05")
        self.assertEqual(k1, k2)
        self.assertEqual(k1, "2026-08-05|SPY|CALL|773.00|2026-08-07")

    def test_select_skips_duplicates(self):
        df = pd.DataFrame([
            {
                "Ticker": "SPY", "recommendation": "BUY", "direction": "CALL",
                "strike": 773, "expiry": "2026-08-07", "exec_window_ok": True,
            },
            {
                "Ticker": "AMZN", "recommendation": "BUY", "direction": "CALL",
                "strike": 272.5, "expiry": "2026-08-07", "exec_window_ok": True,
            },
            {
                "Ticker": "X", "recommendation": "WAIT", "direction": "CALL",
                "strike": 1, "expiry": "2026-08-07", "exec_window_ok": True,
            },
        ])
        day = "2026-08-05"
        spy_key = alert_key(df.iloc[0], day=day)
        to_send, skipped, _ = select_buy_rows(
            df, sent_keys={spy_key}, force=False, day=day,
        )
        tickers = [r.get("Ticker") for _, r in to_send]
        self.assertEqual(tickers, ["AMZN"])
        self.assertEqual(len(skipped), 1)

    def test_select_skips_closed_window(self):
        df = pd.DataFrame([{
            "Ticker": "SPY", "recommendation": "BUY", "direction": "CALL",
            "strike": 773, "expiry": "2026-08-07", "exec_window_ok": False,
        }])
        to_send, _, skipped_w = select_buy_rows(df, sent_keys=set(), day="2026-08-05")
        self.assertEqual(len(to_send), 0)
        self.assertEqual(len(skipped_w), 1)

    def test_sent_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "telegram_sent.json")
            save_sent({"sent": {"2026-08-05|SPY|CALL|1.00|2026-08-07": "t"}}, path)
            loaded = load_sent(path)
            self.assertIn("2026-08-05|SPY|CALL|1.00|2026-08-07", loaded["sent"])

    def test_prune_old_keys(self):
        sent = {
            "2026-01-01|A|CALL|1.00|2026-01-10": "old",
            "2026-08-05|B|CALL|1.00|2026-08-07": "new",
        }
        pruned = prune_sent(sent, keep_days=21, today="2026-08-05")
        self.assertNotIn("2026-01-01|A|CALL|1.00|2026-01-10", pruned)
        self.assertIn("2026-08-05|B|CALL|1.00|2026-08-07", pruned)


if __name__ == "__main__":
    unittest.main()
