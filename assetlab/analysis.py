"""Chronological evaluation with training-only optimization inputs."""

from itertools import combinations

import numpy as np
import pandas as pd

from .data import align_returns
from .quantum import simulate
from .selection import build_problem, solve_exact


def portfolio_metrics(returns, assets):
    daily = returns[assets].mean(axis=1)
    wealth = np.r_[1.0, (1 + daily).cumprod().to_numpy()]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    return {
        "assets": list(assets), "annual_volatility": float(daily.std(ddof=1) * np.sqrt(252)),
        "total_return": float(wealth[-1] - 1), "max_drawdown": float(drawdown.min()),
        "equity": (wealth[1:] * 100).tolist(),
        "assumptions": "Fixed equal weights, frictionless daily rebalancing; no costs or cash flows",
    }


def analyze(prices, provenance, config):
    returns, quality = align_returns(prices)
    cut = int(len(returns) * config["train_fraction"])
    train, test = returns.iloc[:cut], returns.iloc[cut:]
    if len(train) < 126 or len(test) < 60:
        raise ValueError("Need at least 126 training and 60 holdout return rows.")
    flat = train.columns[train.std() < 1e-12].tolist()
    if flat:
        raise ValueError("Constant training-period returns for: " + ", ".join(flat))
    candidates = config["candidates"]
    covariance = train[candidates].cov().to_numpy() * 252
    problem = build_problem(covariance, candidates, config["choose"])
    exact = solve_exact(problem)
    quantum = simulate(problem, layers=config["layers"], shots=config["shots"], seed=config["seed"])
    matrices = {
        period: {method: frame.corr(method=method).to_numpy().tolist()
                 for method in ("pearson", "spearman")}
        for period, frame in (("train", train), ("holdout", test))
    }
    train_corr = train.corr()
    test_corr = test.corr()
    pairs, rolling = [], []
    for left, right in combinations(returns.columns, 2):
        pairs.append({"asset_a": left, "asset_b": right,
                      "train_pearson": train_corr.loc[left, right],
                      "holdout_pearson": test_corr.loc[left, right],
                      "change": test_corr.loc[left, right] - train_corr.loc[left, right]})
        for window in config["windows"]:
            value = returns[left].iloc[-window:].corr(returns[right].iloc[-window:]) if len(returns) >= window else np.nan
            rolling.append({"asset_a": left, "asset_b": right, "window": window,
                            "last_date": returns.index[-1].date().isoformat(), "pearson": value})
    portfolios = {
        "Exact selection": portfolio_metrics(test, exact[0]["assets"]),
        "All candidates": portfolio_metrics(test, candidates),
    }
    if quantum["selected_assets"]:
        portfolios["QAOA sampled selection"] = portfolio_metrics(test, quantum["selected_assets"])
    payload = {
        "schema_version": 1, "config": config, "provenance": provenance, "quality": quality,
        "split": {"train_start": train.index[0].date().isoformat(),
                  "train_end": train.index[-1].date().isoformat(), "train_rows": len(train),
                  "holdout_start": test.index[0].date().isoformat(),
                  "holdout_end": test.index[-1].date().isoformat(), "holdout_rows": len(test)},
        "tickers": returns.columns.tolist(), "correlations": matrices,
        "pairs": sorted(pairs, key=lambda row: -abs(row["train_pearson"])),
        "rolling_latest": rolling, "problem": problem, "exact": exact, "quantum": quantum,
        "portfolios": portfolios, "dates": [d.date().isoformat() for d in returns.index],
        "returns": returns.to_numpy().tolist(),
        "holdout_dates": [d.date().isoformat() for d in test.index],
        "limitations": [
            "Correlations describe relationships; they do not establish causation or predict gains.",
            "One chronological holdout is a first check, not a completed strategy validation.",
            "Sample covariance is not regularized. Selection may be unstable across periods.",
            "The candidate order is fixed by configuration, never ranked on holdout data.",
            "Ideal QAOA simulation excludes hardware noise, queues, and calibration error.",
            "Optimization and exact enumeration are deliberately limited to ten candidate assets.",
            "Local simulation uses a classical optimizer; no quantum speedup is claimed.",
            "Portfolio evaluation uses frictionless daily rebalancing; trading costs are excluded.",
        ],
    }
    return payload, returns
