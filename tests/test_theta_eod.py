"""Theta EOD للمتتبع: إغلاق عقد منتهٍ — لا يلمس بوابات BUY."""

import os
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from outcome_tracker import enrich_market_exit_premiums, market_option_close
from theta_eod import (
    parse_eod_close,
    reset_theta_ready,
    theta_option_close,
    theta_wanted,
)


class TestParseEodClose(unittest.TestCase):
    def test_prefers_trade_close(self):
        self.assertEqual(
            parse_eod_close([{"close": 1.25, "bid": 1.2, "ask": 1.3}]),
            1.25,
        )

    def test_mid_when_close_zero(self):
        self.assertEqual(
            parse_eod_close([{"close": 0, "bid": 0.04, "ask": 0.06}]),
            0.05,
        )

    def test_worthless_expired(self):
        self.assertEqual(
            parse_eod_close([{"close": 0, "bid": 0, "ask": 0}]),
            0.0,
        )

    def test_empty(self):
        self.assertIsNone(parse_eod_close([]))
        self.assertIsNone(parse_eod_close(None))


class TestThetaWanted(unittest.TestCase):
    def setUp(self):
        reset_theta_ready()

    def tearDown(self):
        reset_theta_ready()
        os.environ.pop("THETA_DISABLE", None)
        os.environ.pop("THETA_BASE_URL", None)
        os.environ.pop("EOD_RUN", None)
        os.environ.pop("THETA_EMAIL", None)

    def test_off_by_default(self):
        os.environ.pop("THETA_BASE_URL", None)
        os.environ.pop("EOD_RUN", None)
        os.environ.pop("THETA_EMAIL", None)
        os.environ.pop("THETA_DISABLE", None)
        self.assertFalse(theta_wanted())

    def test_eod_run_wants_theta(self):
        os.environ["EOD_RUN"] = "true"
        self.assertTrue(theta_wanted())

    def test_disable_wins(self):
        os.environ["EOD_RUN"] = "true"
        os.environ["THETA_DISABLE"] = "1"
        self.assertFalse(theta_wanted())


class TestThetaThenYahoo(unittest.TestCase):
    def setUp(self):
        reset_theta_ready()
        os.environ["THETA_MIN_INTERVAL"] = "0"
        os.environ["EOD_RUN"] = "true"

    def tearDown(self):
        reset_theta_ready()
        for k in ("THETA_MIN_INTERVAL", "EOD_RUN", "THETA_DISABLE", "THETA_BASE_URL"):
            os.environ.pop(k, None)

    def test_theta_hit_skips_yahoo(self):
        with patch("theta_eod.theta_ready", return_value=True), patch(
            "theta_eod.fetch_option_eod",
            return_value=[{"close": 0.42, "bid": 0.4, "ask": 0.44}],
        ), patch("outcome_tracker.fetch_option_daily_hist") as yf:
            px = market_option_close(
                "SPY", "2026-08-14", 769, True, date(2026, 8, 14), cache={},
            )
        self.assertEqual(px, 0.42)
        yf.assert_not_called()

    def test_yahoo_when_theta_down(self):
        hist = pd.DataFrame(
            {"Close": [0.18]},
            index=pd.to_datetime(["2026-08-14"]),
        )
        with patch("theta_eod.theta_ready", return_value=False), patch(
            "outcome_tracker.fetch_option_daily_hist", return_value=hist,
        ):
            px = market_option_close(
                "SPY", "2026-08-14", 769, True, date(2026, 8, 14), cache={},
            )
        self.assertEqual(px, 0.18)

    def test_theta_option_close_uses_cache(self):
        fetch = MagicMock(return_value=[{"close": 2.0}])
        cache = {}
        with patch("theta_eod.theta_wanted", return_value=True), patch(
            "theta_eod.theta_ready", return_value=True,
        ), patch("theta_eod.fetch_option_eod", fetch):
            a = theta_option_close("AAPL", "2026-08-14", 170, True, date(2026, 8, 14), cache)
            b = theta_option_close("AAPL", "2026-08-14", 170, True, date(2026, 8, 14), cache)
        self.assertEqual(a, 2.0)
        self.assertEqual(b, 2.0)
        self.assertEqual(fetch.call_count, 1)


class TestEodJavaMatchesThetaJar(unittest.TestCase):
    def test_workflow_uses_java_21(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, ".github", "workflows", "eod_outcome.yml")
        with open(path, encoding="utf-8") as f:
            yml = f.read()
        self.assertIn("java-version: '21'", yml)
        self.assertNotIn("java-version: '17'", yml)
        self.assertIn("group: eod-csv-writes", yml)
        self.assertNotIn("group: repo-csv-writes", yml)
        self.assertIn("timeout-minutes: 60", yml)
        self.assertNotIn("timeout-minutes: 15", yml)


class TestEnrichTagsTheta(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("EOD_RUN", None)

    def test_eod_enrich_writes_theta_source(self):
        os.environ["EOD_RUN"] = "true"
        df = pd.DataFrame([{
            "status": "stop_hit",
            "ticker": "SPY",
            "expiry": "2026-08-14",
            "strike": 769,
            "direction": "CALL",
            "entry_stock": 770,
            "tp1_stock": 772,
            "stop_stock": 766,
            "exit_date": "2026-08-14",
            "premium": 2.45,
            "exit_premium": 0.12,
            "exit_premium_source": "market",
            "option_pnl_pct": -95.1,
        }])
        with patch("outcome_tracker.market_option_quote", return_value=(0.42, "theta")):
            out = enrich_market_exit_premiums(df, cache={})
        self.assertEqual(str(out.at[0, "exit_premium_source"]), "theta")
        self.assertEqual(float(out.at[0, "exit_premium"]), 0.42)

    def test_eod_skips_completed_theta_rows(self):
        os.environ["EOD_RUN"] = "true"
        df = pd.DataFrame([{
            "status": "stop_hit",
            "ticker": "SPY",
            "expiry": "2026-08-14",
            "strike": 769,
            "direction": "CALL",
            "entry_stock": 770,
            "tp1_stock": 772,
            "stop_stock": 766,
            "exit_date": "2026-08-14",
            "premium": 2.45,
            "exit_premium": 0.42,
            "exit_premium_source": "theta",
            "option_pnl_pct": -82.86,
        }])
        with patch("outcome_tracker.market_option_quote") as q:
            out = enrich_market_exit_premiums(df, cache={})
        q.assert_not_called()
        self.assertEqual(str(out.at[0, "exit_premium_source"]), "theta")
        self.assertEqual(float(out.at[0, "exit_premium"]), 0.42)


if __name__ == "__main__":
    unittest.main()
