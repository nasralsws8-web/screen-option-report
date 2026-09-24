"""قواعد ماسح ما قبل الاختراق كما أُرسلت: درجة متعددة، لا BUY، هدف +7%."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stock_prebreakout import (  # noqa: E402
    DISCLAIMER,
    SIGNAL_A,
    SIGNAL_IGNORE,
    SIGNAL_STRONG,
    SIGNAL_WATCH,
    WEIGHTS,
    evaluate,
    higher_low_steps,
    is_compressing,
    risk_reward,
    target_price,
    trigger_price,
    volume_is_rising,
)


def _full(**overrides):
    setup = {
        "price": 4.76,
        "pmh": 4.85,
        "prev_day_high": 5.40,
        "gap_pct": 12,
        "float_shares": 18_000_000,
        "pm_volume": 1_200_000,
        "dollar_volume": 6_000_000,
        "rvol": 5.2,
        "catalyst": "company",
        "higher_low_steps": 5,
        "compression": True,
        "volume_rising": True,
        "repeated_rejection": False,
        "above_vwap": True,
        "rocm20": 0.04,
        "rocm20_rising": True,
        "rocm200": 0.01,
        "rs_vs_spy": 0.03,
        "atr_pct": 6.5,
        "structure_low": 4.82,
        "major_resistance": 6.50,
    }
    setup.update(overrides)
    return setup


class TestWeights(unittest.TestCase):
    def test_weights_sum_to_100(self):
        self.assertEqual(sum(WEIGHTS.values()), 100)


class TestNoSingleIndicator(unittest.TestCase):
    def test_gap_alone_is_ignore(self):
        out = evaluate({"price": 4.70, "gap_pct": 15, "atr_pct": 5, "pmh": 6})
        self.assertLess(out["score"], 65)
        self.assertEqual(out["signal"], SIGNAL_IGNORE)
        self.assertNotEqual(out["recommendation"], "BUY")

    def test_rvol_alone_is_ignore(self):
        out = evaluate({"price": 4.70, "rvol": 8, "atr_pct": 5, "pmh": 6})
        self.assertLess(out["score"], 65)
        self.assertEqual(out["signal"], SIGNAL_IGNORE)

    def test_float_alone_is_ignore(self):
        out = evaluate({"price": 4.70, "float_shares": 10_000_000, "atr_pct": 5, "pmh": 6})
        self.assertLess(out["score"], 65)
        self.assertEqual(out["signal"], SIGNAL_IGNORE)

    def test_three_pillars_still_below_watch(self):
        out = evaluate({
            "price": 4.70,
            "catalyst": "company",
            "rvol": 6,
            "pm_volume": 2_000_000,
            "dollar_volume": 10_000_000,
            "atr_pct": 5,
            "pmh": 6,
        })
        self.assertLess(out["score"], 65)


class TestBands(unittest.TestCase):
    def test_full_setup_is_prebreakout_not_buy(self):
        out = evaluate(_full())
        self.assertGreaterEqual(out["score"], 85)
        self.assertEqual(out["signal"], SIGNAL_A)
        self.assertEqual(out["recommendation"], "WATCH")
        self.assertNotIn("BUY", out["recommendation"])
        self.assertIn("Wait for confirmation", out["reason"])
        self.assertIn("Trigger = $4.90", out["reason"])
        self.assertIn("7%", DISCLAIMER)

    def test_watch_and_strong_thresholds(self):
        watch = evaluate(_full(
            catalyst="none", above_vwap=False, rocm20=None,
            rocm20_rising=False, rs_vs_spy=None, gap_pct=6, rvol=2.1,
        ))
        self.assertLess(watch["score"], evaluate(_full())["score"])
        self.assertNotEqual(watch["signal"], SIGNAL_A)
        self.assertNotEqual(watch["recommendation"], "BUY")


class TestLevels(unittest.TestCase):
    def test_trigger_and_target_match_example(self):
        self.assertEqual(trigger_price(4.85), 4.90)
        self.assertEqual(target_price(4.90), 5.24)
        out = evaluate(_full())
        self.assertEqual(out["trigger"], 4.90)
        self.assertEqual(out["target"], 5.24)
        self.assertLess(out["stop"], 4.90)
        self.assertGreaterEqual(out["risk_reward"], 3)

    def test_risk_reward_formula(self):
        self.assertEqual(risk_reward(4.90, 4.70, 5.24), round((5.24 - 4.90) / (4.90 - 4.70), 2))


class TestPenalties(unittest.TestCase):
    def test_large_gap_without_catalyst_scores_less(self):
        with_cat = evaluate(_full())
        no_cat = evaluate(_full(catalyst="none"))
        self.assertLess(no_cat["parts"]["gap"], with_cat["parts"]["gap"])
        self.assertLess(no_cat["score"], with_cat["score"])

    def test_resistance_before_target_is_reported(self):
        out = evaluate(_full(major_resistance=5.05))
        self.assertEqual(out["resistance_before_target"], 5.05)
        self.assertIn("مقاومة", out["notes"])
        self.assertLess(out["score"], evaluate(_full())["score"])

    def test_low_atr_cannot_be_a_plus(self):
        out = evaluate(_full(atr_pct=1.0))
        self.assertNotEqual(out["signal"], SIGNAL_A)
        self.assertIn("ATR%", out["notes"])

    def test_penny_without_liquidity_is_ignore(self):
        out = evaluate(_full(price=0.40, dollar_volume=100_000, pm_volume=200_000))
        self.assertEqual(out["signal"], SIGNAL_IGNORE)

    def test_above_ten_dollars_is_ignore(self):
        out = evaluate(_full(price=14))
        self.assertEqual(out["signal"], SIGNAL_IGNORE)

    def test_already_broken_is_not_a_plus(self):
        out = evaluate(_full(price=4.95, prev_day_high=4.80))
        self.assertTrue(out["post_breakout"])
        self.assertNotEqual(out["signal"], SIGNAL_A)
        self.assertLess(out["score"], 75)

    def test_rr_below_three_blocks_a_plus(self):
        out = evaluate(_full(structure_low=4.40))
        self.assertLess(out["risk_reward"], 3)
        self.assertNotEqual(out["signal"], SIGNAL_A)


class TestStructure(unittest.TestCase):
    def test_rising_volume_versus_fading(self):
        self.assertTrue(volume_is_rising([500_000, 650_000, 800_000, 1_100_000]))
        self.assertFalse(volume_is_rising([2_000_000, 1_500_000, 1_000_000, 700_000]))

    def test_higher_lows_and_compression(self):
        self.assertGreaterEqual(higher_low_steps([29.20, 29.45, 29.65, 29.80, 29.90]), 4)
        self.assertTrue(is_compressing(1.20, 0.40))
        self.assertFalse(is_compressing(0.40, 1.20))

    def test_near_premarket_high_gets_distance_points(self):
        near = evaluate(_full())
        far = evaluate(_full(price=4.20, pmh=4.85))
        self.assertEqual(near["parts"]["dist_pmh"], WEIGHTS["dist_pmh"])
        self.assertLess(far["parts"]["dist_pmh"], near["parts"]["dist_pmh"])
        self.assertLessEqual(near["distance_to_resistance_pct"], 2)


class TestRoc(unittest.TestCase):
    def test_roc_reads_by_position_on_date_index(self):
        import pandas as pd
        from stock_prebreakout_scan import _roc

        idx = pd.date_range("2024-01-01", periods=30, freq="B")
        closes = pd.Series(range(100, 130), index=idx, dtype=float)
        self.assertAlmostEqual(_roc(closes, 20), (129 - 109) / 109)


class TestStockLog(unittest.TestCase):
    def test_same_day_updates_and_keeps_first_seen(self):
        import os
        import tempfile
        import pandas as pd
        from stock_prebreakout_scan import append_log

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "log.csv")
            first = {
                "ticker": "GLND", "company": "Greenland", "price": 3.10,
                "score": 70, "signal": "WATCH", "scanned_at": "2026-09-24 14:00 UTC",
                "trigger": 3.20, "target": 3.42, "stop": 3.00,
            }
            later = dict(first, price=3.21, score=72, scanned_at="2026-09-24 15:00 UTC")
            self.assertEqual(append_log([first], path), 1)
            self.assertEqual(append_log([later], path), 0)
            rows = pd.read_csv(path).to_dict("records")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["first_seen"], "2026-09-24 14:00 UTC")
            self.assertEqual(float(rows[0]["price"]), 3.21)
            self.assertEqual(append_log([], path), 0)
            self.assertEqual(len(pd.read_csv(path)), 1)
            nxt = dict(first, scanned_at="2026-09-25 14:00 UTC", price=3.40)
            append_log([nxt], path)
            self.assertEqual(len(pd.read_csv(path)), 2)


class TestSources(unittest.TestCase):
    def test_scan_uses_options_clients(self):
        import cheap_options_screener_v3 as opt
        import finnhub_premarket as fh
        import stock_prebreakout_scan as scan

        self.assertIs(scan.fetch_premarket, opt.fetch_premarket)
        self.assertIs(scan._YF_SESSION, opt._YF_SESSION)
        self.assertIs(scan.load_manual_tickers, opt.load_manual_tickers)
        self.assertIs(scan.fix_ticker, opt.fix_ticker)
        self.assertIs(scan.get_api_key, fh.get_api_key)
        self.assertIs(scan._get, fh._get)
        self.assertIs(scan.enrich_ticker_premarket, fh.enrich_ticker_premarket)
        self.assertEqual(scan.DELAY_BETWEEN, opt.DELAY_BETWEEN)


if __name__ == "__main__":
    unittest.main()
