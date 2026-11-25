from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from assetlab.trading import (TradingConfig, backtest, check_panel, fit_relationships,
                              formation_folds, performance, simulate_trades)
from assetlab.trading_io import append_observation, run_trading, signal_preview


def synthetic_relationships(length=400):
    rng = np.random.default_rng(22)
    idx = pd.bdate_range("2020-01-01", periods=length)
    common = np.cumsum(rng.normal(0.0003, 0.008, length))
    residuals = np.zeros((length, 3))
    for i in range(1, length):
        residuals[i] = 0.87 * residuals[i - 1] + rng.normal(0, 0.003, 3)
    values = {ticker: 100 * np.exp(common + residuals[:, j]) for j, ticker in enumerate(["A", "B", "C"])}
    values["SPY"] = 100 * np.exp(common)
    return pd.DataFrame(values, index=idx)


def known_fold():
    dates = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"A": [100.0, 80.0, 120.0, 120.0], "B": [100.0] * 4,
                           "SPY": [100.0] * 4}, index=dates)
    model = {"ticker": "A", "peers": ["B"], "intercept": 0.0, "beta": 1.0,
             "residual_mean": 0.0, "residual_std": 0.1}
    fold = {"id": 1, "start": 2, "stop": 4, "models": {"A": model}, "audit": [],
            "train_end": dates[1].date().isoformat(), "test_start": dates[2].date().isoformat(),
            "test_end": dates[3].date().isoformat()}
    return prices, fold


class ExecutionTests(unittest.TestCase):
    def test_signal_cannot_buy_at_observed_cheap_close_and_costs_hit_both_sides(self):
        prices, fold = known_fold()
        cfg = TradingConfig(initial_capital=1000, max_positions=1, trend_filter=False, cost_bps=100)
        result = simulate_trades(prices, [fold], cfg)
        self.assertEqual(len(result["trades"]), 1)
        trade = result["trades"][0]
        self.assertEqual(trade["entry_signal_date"], "2024-01-02")
        self.assertEqual(trade["entry_date"], "2024-01-03")
        self.assertEqual(trade["entry_price"], 120)
        self.assertEqual(trade["exit_price"], 120)
        self.assertAlmostEqual(trade["gross_pnl"], 0)
        self.assertAlmostEqual(trade["entry_cost"], 1000 / 1.01 * .01)
        self.assertAlmostEqual(trade["net_pnl"], -trade["entry_cost"] - trade["exit_cost"])
        self.assertAlmostEqual(result["daily"][-1]["equity"], 1000 + trade["net_pnl"])

    def test_zero_cost_case_does_not_invent_a_profit_from_next_session_rebound(self):
        prices, fold = known_fold()
        cfg = TradingConfig(initial_capital=1000, max_positions=1, trend_filter=False)
        result = simulate_trades(prices, [fold], cfg, cost_bps=0)
        self.assertAlmostEqual(result["daily"][-1]["equity"], 1000)

    def test_no_new_entry_on_last_fold_close_and_empty_models_mean_cash(self):
        prices, fold = known_fold()
        cfg = TradingConfig(initial_capital=1000, max_positions=1, trend_filter=False)
        fold["models"] = {}
        result = simulate_trades(prices, [fold], cfg)
        self.assertFalse(result["trades"])
        self.assertTrue(all(row["equity"] == 1000 and row["positions"] == 0 for row in result["daily"]))
        self.assertIsNone(performance(result["daily"], 1000)["sharpe_zero_rf"])

    def test_missing_quotes_are_rejected_instead_of_bridged(self):
        prices = synthetic_relationships()
        prices.iloc[200, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing quotes"):
            check_panel(prices, ["A", "B", "C"], TradingConfig(train_sessions=126))


class WalkForwardTests(unittest.TestCase):
    def setUp(self):
        self.prices = synthetic_relationships()
        self.cfg = TradingConfig(train_sessions=126, test_sessions=63, trend_filter=False)

    def test_future_prices_cannot_change_earlier_relationships_or_trades(self):
        original = backtest(self.prices, ["A", "B", "C"], self.cfg)
        changed = self.prices.copy()
        cutoff = 126 + 63
        changed.iloc[cutoff:, 0] *= np.exp(np.linspace(.5, 2, len(changed) - cutoff))
        altered = backtest(changed, ["A", "B", "C"], self.cfg)
        self.assertEqual(original["strategy"]["folds"][0], altered["strategy"]["folds"][0])
        self.assertEqual(original["strategy"]["daily"][:63], altered["strategy"]["daily"][:63])
        self.assertEqual(original["model_audit"][:3], altered["model_audit"][:3])

    def test_all_fill_dates_follow_signals_and_account_reconciles(self):
        result = backtest(self.prices, ["A", "B", "C"], self.cfg)
        self.assertTrue(result["strategy"]["trades"])
        for trade in result["strategy"]["trades"]:
            self.assertLess(trade["entry_signal_date"], trade["entry_date"])
            self.assertLess(trade["exit_signal_date"], trade["exit_date"])
        pnl = sum(t["net_pnl"] for t in result["strategy"]["trades"])
        self.assertAlmostEqual(self.cfg.initial_capital + pnl, result["metrics"]["ending_equity"], places=6)
        self.assertTrue(all(r["cash"] >= -1e-7 and r["positions"] <= self.cfg.max_positions for r in result["strategy"]["daily"]))
        self.assertEqual(len(result["dates"]), len(self.prices) - self.cfg.train_sessions)

    def test_forward_preview_uses_the_same_fixed_recalibration_schedule(self):
        snapshot = signal_preview(self.prices, ["A", "B", "C"], self.cfg)
        end = 126 + ((len(self.prices) - 126) // 63) * 63
        models, _ = fit_relationships(self.prices.iloc[end - 126:end], ["A", "B", "C"], self.cfg)
        self.assertEqual(snapshot["models"], models)
        self.assertEqual(snapshot["model_train_end"], self.prices.index[end - 1].date().isoformat())


class ForwardJournalTests(unittest.TestCase):
    def test_observations_are_append_only_deduplicated_and_rules_are_frozen(self):
        prices = synthetic_relationships()
        cfg = TradingConfig(train_sessions=126, test_sessions=63, trend_filter=False)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.csv"
            prices.to_csv(path)
            folder, _ = run_trading(["A", "B", "C"], cfg, start="2020-01-01", end="2022-01-01", csv_path=path, out=temp)
            self.assertFalse((folder / "signals.jsonl").exists())
            now = datetime(2026, 11, 25, 12, tzinfo=timezone.utc)
            first = append_observation(folder, prices.iloc[:-1], {"source": "csv"}, observation_type="uploaded_snapshot", now=now)
            prefix = (folder / "signals.jsonl").read_bytes()
            self.assertIsNone(append_observation(folder, prices.iloc[:-1], {"source": "csv"}, observation_type="uploaded_snapshot", now=now))
            append_observation(folder, prices, {"source": "csv"}, observation_type="uploaded_snapshot", now=now)
            self.assertTrue((folder / "signals.jsonl").read_bytes().startswith(prefix))
            self.assertEqual(first["recorded_at_utc"], now.isoformat())
            self.assertEqual(first["observation_type"], "uploaded_snapshot")
            lines = (folder / "signals.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            manifest = json.loads((folder / "manifest.json").read_text())
            for name, digest in manifest["files_sha256"].items():
                self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), digest)
            rules = json.loads((folder / "rules.json").read_text())
            rules["config"]["entry_z"] = 1.5
            (folder / "rules.json").write_text(json.dumps(rules))
            with self.assertRaisesRegex(ValueError, "Frozen rules were modified"):
                append_observation(folder, prices, {"source": "csv"}, now=now)


if __name__ == "__main__":
    unittest.main()
