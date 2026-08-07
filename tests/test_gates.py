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

from cheap_options_screener_v3 import (  # noqa: E402
    entry_is_chased,
    find_nearest_expiry,
    grade_wait_setup,
    inverse_recommendation_ceiling,
    is_exec_window,
    next_friday_date,
)
from data_quality import classify_data_quality  # noqa: E402
from outcome_tracker import resolve_first_touch  # noqa: E402
from telegram_notify import (  # noqa: E402
    alert_key,
    is_active_watch,
    is_hot_wait_row,
    load_sent,
    prune_sent,
    save_sent,
    select_buy_rows,
    select_hot_wait_rows,
)

ET = ZoneInfo("America/New_York")


class TestExpiryFriday(unittest.TestCase):
    def test_next_friday_skips_today_when_friday(self):
        fri = __import__("datetime").date(2026, 8, 7)  # Friday
        self.assertEqual(next_friday_date(fri).isoformat(), "2026-08-14")

    def test_on_friday_picks_next_friday_not_0dte(self):
        fri = __import__("datetime").date(2026, 8, 7)
        exps = ["2026-08-07", "2026-08-14", "2026-08-21"]
        exp, dte = find_nearest_expiry(exps, ticker="NVDA", today=fri)
        self.assertEqual(exp, "2026-08-14")
        self.assertEqual(dte, 7)

    def test_thursday_picks_tomorrow_friday(self):
        thu = __import__("datetime").date(2026, 8, 6)
        exps = ["2026-08-07", "2026-08-14"]
        exp, dte = find_nearest_expiry(exps, ticker="SPY", today=thu)
        self.assertEqual(exp, "2026-08-07")
        self.assertEqual(dte, 1)


class TestInverseGate(unittest.TestCase):
    def test_sqqq_call_in_bull_is_avoid(self):
        ceiling, note = inverse_recommendation_ceiling(True, "BULL", "SQQQ")
        self.assertEqual(ceiling, "AVOID")
        self.assertIn("صاعد", note)

    def test_sqqq_put_in_bull_is_wait_not_buy(self):
        ceiling, note = inverse_recommendation_ceiling(False, "BULL", "SQQQ")
        self.assertEqual(ceiling, "WAIT")
        self.assertNotEqual(ceiling, "BUY")

    def test_sqqq_neutral_no_buy(self):
        ceiling, _ = inverse_recommendation_ceiling(True, "NEUTRAL", "SQQQ")
        self.assertEqual(ceiling, "WAIT")

    def test_normal_stock_unchanged(self):
        ceiling, note = inverse_recommendation_ceiling(True, "BULL", "NVDA")
        self.assertEqual(ceiling, "BUY")
        self.assertEqual(note, "")


class TestWaitGrade(unittest.TestCase):
    def test_mild_chase_can_be_hot(self):
        # مثل NVDA/TQQQ هذا الأسبوع: مطاردة بسيطة مع مجال لـ TP2
        g = grade_wait_setup(
            conf=76, liquid_ok=True, prem_ok=True, align_ok=True, exec_ok=True,
            trend_ok=True, mixed_ok=False, plan_valid=True, chased=True,
            dte=3, is_0dte=False, live_rr=0.5, tp1_rr=2.0,
            price=211.85, entry=209.43, stop=207.5, tp2=218.7, tp3=226.0,
            is_call=True, atr_pct=2.5, score=22, rvol=2.2,
        )
        self.assertGreaterEqual(g["setup_pct"], 75)
        self.assertIn(g["wait_tier"], ("HOT", "FIRE"))

    def test_cold_when_misaligned(self):
        g = grade_wait_setup(
            conf=40, liquid_ok=False, prem_ok=False, align_ok=False, exec_ok=False,
            trend_ok=False, mixed_ok=False, plan_valid=False, chased=False,
            dte=0, is_0dte=True, live_rr=0.2, tp1_rr=0.5,
            price=100, entry=100, stop=99, tp2=102, tp3=104,
            is_call=True, atr_pct=1.0, score=5, rvol=0.5,
        )
        self.assertLess(g["setup_pct"], 55)
        self.assertEqual(g["wait_tier"], "COLD")


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
    def _bar(self, o, h, l, c, day="2026-08-05"):
        return pd.DataFrame(
            {"Open": [o], "High": [h], "Low": [l], "Close": [c]},
            index=pd.to_datetime([day]),
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

    def test_rec_day_ignores_high_without_close_confirm(self):
        # مثل NVDA: قمة لمست TP1 لكن الإغلاق تحت الهدف يوم التوصية
        post = self._bar(221.53, 223.63, 217.27, 218.92, day="2026-08-06")
        touch = resolve_first_touch(
            post, True, stop=215.24, tp1=222.22, tp2=226.0, tp3=230.0, entry=217.90,
            rec_date="2026-08-06", price_at_rec=217.90,
        )
        self.assertIsNone(touch)

    def test_rec_day_tp_when_close_confirms(self):
        post = self._bar(218.0, 223.5, 217.5, 222.50, day="2026-08-06")
        touch = resolve_first_touch(
            post, True, stop=215.24, tp1=222.22, tp2=226.0, tp3=230.0, entry=217.90,
            rec_date="2026-08-06", price_at_rec=217.90,
        )
        self.assertIsNotNone(touch)
        self.assertEqual(touch[0], "tp1_hit")

    def test_after_rec_day_high_still_counts(self):
        # بعد يوم التوصية: High يكفي حتى لو الإغلاق تحت الهدف
        post = self._bar(220.0, 223.5, 219.0, 220.5, day="2026-08-07")
        touch = resolve_first_touch(
            post, True, stop=215.24, tp1=222.22, tp2=226.0, tp3=230.0, entry=217.90,
            rec_date="2026-08-06", price_at_rec=217.90,
        )
        self.assertIsNotNone(touch)
        self.assertEqual(touch[0], "tp1_hit")

    def test_skips_tp_already_at_price_at_rec(self):
        post = self._bar(223.0, 224.0, 222.5, 223.2, day="2026-08-06")
        touch = resolve_first_touch(
            post, True, stop=215.24, tp1=222.22, tp2=226.0, tp3=230.0, entry=217.90,
            rec_date="2026-08-06", price_at_rec=222.50,
        )
        self.assertIsNone(touch)


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
        self.assertEqual(
            alert_key(row, day="2026-08-05", kind="WAIT_HOT"),
            "WAIT_HOT|2026-08-05|SPY|CALL|773.00|2026-08-07",
        )

    def test_select_hot_wait_rows(self):
        df = pd.DataFrame([
            {
                "Ticker": "NVDA", "recommendation": "WAIT", "direction": "CALL",
                "strike": 220, "expiry": "2026-08-14", "exec_window_ok": True,
                "wait_tier": "HOT", "setup_pct": 82, "chase_pct": 0.8, "tp2_rr_live": 1.4,
            },
            {
                "Ticker": "AMD", "recommendation": "WAIT", "direction": "CALL",
                "strike": 100, "expiry": "2026-08-14", "exec_window_ok": True,
                "wait_tier": "COLD", "setup_pct": 40,
            },
            {
                "Ticker": "SPY", "recommendation": "BUY", "direction": "CALL",
                "strike": 770, "expiry": "2026-08-14", "exec_window_ok": True,
                "wait_tier": "FIRE", "setup_pct": 95,
            },
        ])
        day = "2026-08-07"
        to_send, _, _ = select_hot_wait_rows(df, sent_keys=set(), day=day)
        tickers = [r.get("Ticker") for _, r in to_send]
        self.assertEqual(tickers, ["NVDA"])
        self.assertTrue(is_hot_wait_row(df.iloc[0]))
        self.assertTrue(is_active_watch(df.iloc[0]))
        self.assertFalse(is_hot_wait_row(df.iloc[1]))

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
