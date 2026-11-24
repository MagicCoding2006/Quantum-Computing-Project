#!/usr/bin/env python3
"""
checklist_scanner.py - find past setups that came before big gains

Every week, for every stock, each CONDITION (checklist item) is either met or
not. A CHECKLIST is a set of conditions that must all be met at once. For
every possible checklist the scanner measures, on past data:

  - how often it fired
  - how often a big gain followed (hit rate), vs. how often one followed on
    any random week (base rate)
  - how confident we can be in that hit rate, given how few times it fired

Checklists are ranked on the TRAINING years and then re-checked on later TEST
years they never saw. The summary also shows which stocks match the top
checklists right now. No alerts are sent.

Setup (once):
    python3 -m pip install -r requirements.txt

Run:
    python3 checklist_scanner.py                                   # 10 default stocks
    python3 checklist_scanner.py --universe sp500                  # every current S&P 500 stock
    python3 checklist_scanner.py --universe sp500 --sector "information technology"
    python3 checklist_scanner.py --tickers AAPL MSFT NVDA --horizon 26 --target 0.25
    python3 checklist_scanner.py --vs-market      # "big gain" = beat the S&P 500 by the target
    python3 checklist_scanner.py --demo           # fake data, no internet

Sectors (GICS, as listed on Wikipedia's S&P 500 page; partial names work, e.g. "health"):
    Communication Services, Consumer Discretionary, Consumer Staples, Energy,
    Financials, Health Care, Industrials, Information Technology, Materials,
    Real Estate, Utilities
"""

import argparse
import io
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

MARKET = "^GSPC"  # S&P 500 index
DEFAULT_TICKERS = ["AAPL", "MSFT", "IBM", "JPM", "XOM", "JNJ", "PG", "KO", "WMT", "GE"]
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
def load_yahoo(ticker, cache_dir):
    """Full available daily history from Yahoo (split/dividend adjusted), cached to CSV."""
    cache = cache_dir / f"{ticker}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    import yfinance as yf

    try:
        df = yf.download(ticker.replace(".", "-"), period="max", auto_adjust=True, progress=False)
    except Exception as e:
        print(f"  {ticker}: download failed ({e})")
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close", "Volume"]].dropna()
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    time.sleep(0.3)  # be polite to Yahoo on big runs
    return df


def load_demo(ticker):
    """Random-walk prices. Nothing here is predictable, so any 'edge' found is pure luck."""
    rng = np.random.default_rng(zlib.crc32(ticker.encode()))
    idx = pd.bdate_range("1990-01-02", "2025-12-31")
    r = rng.normal(rng.normal(0.0003, 0.0002), rng.uniform(0.012, 0.03), len(idx))
    close = 50 * np.exp(np.cumsum(np.log1p(r)))
    vol = rng.lognormal(15, 0.4, len(idx))
    return pd.DataFrame({"Close": close, "Volume": vol}, index=idx)


def resample_last(obj, freq="W"):
    return obj.resample("W-FRI").last()


def sp500_list(cache_dir, sector=None):
    """Current S&P 500 members and their GICS sectors, from Wikipedia (cached)."""
    cache = cache_dir / "sp500_list.csv"
    if cache.exists():
        table = pd.read_csv(cache)
    else:
        req = urllib.request.Request(SP500_URL, headers={"User-Agent": "Mozilla/5.0 (quant study)"})
        html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
        table = pd.read_html(io.StringIO(html))[0][["Symbol", "GICS Sector"]]
        cache_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache, index=False)
    if sector:
        table = table[table["GICS Sector"].str.lower().str.contains(sector.lower())]
        if table.empty:
            raise SystemExit(f"No sector matches '{sector}'. See the sector list at the top of this file.")
        print(f"Sector filter '{sector}' -> {', '.join(sorted(table['GICS Sector'].unique()))}")
    return [s.replace(".", "-") for s in table.Symbol]


# --------------------------------------------------------------------------
# CHECKLIST ITEMS
# Each takes a daily DataFrame with Close, Volume and Mkt (S&P 500 close) and
# returns True/False for each day, using only data up to that day.
# To add an item, add one entry. Keep the total around 8-11 so it fits the lab's
# 11 qubits (one qubit per item).
# --------------------------------------------------------------------------
def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def _golden_cross_recent(d):
    above = d.Close.rolling(50).mean() > d.Close.rolling(200).mean()
    crossed = above & ~above.shift(1, fill_value=False)
    return above & (crossed.rolling(20).max() > 0)


def _vol_squeeze(d):
    r = d.Close.pct_change()
    return r.rolling(21).std() < 0.7 * r.rolling(126).std()


CONDITIONS = {
    "strong_momentum":   lambda d: d.Close.shift(21) / d.Close.shift(252) - 1 > 0.20,  # up 20%+ over the past year (skipping last month)
    "pullback":          lambda d: d.Close / d.Close.shift(21) - 1 < -0.05,            # down 5%+ over the last month
    "volume_surge":      lambda d: d.Volume.rolling(5).mean() > 1.5 * d.Volume.rolling(63).mean(),  # last week's volume 1.5x normal
    "above_200dma":      lambda d: d.Close > d.Close.rolling(200).mean(),               # above 200-day average
    "near_52w_high":     lambda d: d.Close >= 0.95 * d.Close.rolling(252).max(),        # within 5% of 1-year high
    "vol_squeeze":       _vol_squeeze,                                                   # unusually calm lately
    "golden_cross":      _golden_cross_recent,                                           # 50-day crossed above 200-day in last month
    "oversold_rsi":      lambda d: _rsi(d.Close) < 30,                                   # RSI below 30
    "market_uptrend":    lambda d: d.Mkt > d.Mkt.rolling(200).mean(),                   # S&P 500 above its 200-day average
    "beating_market_3m": lambda d: d.Close / d.Close.shift(63) > d.Mkt / d.Mkt.shift(63),  # outran the S&P 500 over 3 months
}

WARMUP_DAYS = 260  # longest look-back above; earlier rows are dropped


def load(ticker, a):
    return load_demo(ticker) if a.demo else load_yahoo(ticker, Path(a.cache))


def build_panel(a):
    """Long table: one row per (stock, week) with each condition, the forward return, and hit flag."""
    mkt = load(MARKET, a)
    if mkt is None or mkt.empty:
        raise SystemExit("Could not load S&P 500 index data.")
    mkt_close = mkt.Close.rename("Mkt")
    mkt_weekly = resample_last(mkt_close, "W")
    mkt_fwd = mkt_weekly.shift(-a.horizon) / mkt_weekly - 1

    frames = []
    for t in a.tickers:
        daily = load(t, a)
        if daily is None or daily.empty:
            print(f"  {t}: no data, skipping")
            continue
        d = daily.join(mkt_close, how="inner").dropna(subset=["Close", "Mkt"])
        if len(d) < WARMUP_DAYS + 52 * 5:
            print(f"  {t}: under 5 years of usable history, skipping")
            continue
        cond = pd.DataFrame({name: fn(d).fillna(False).astype(bool) for name, fn in CONDITIONS.items()})
        cond = cond.iloc[WARMUP_DAYS:]
        weekly = resample_last(cond, "W").dropna().astype(bool)
        close_w = resample_last(d.Close, "W").reindex(weekly.index)
        weekly["fwd"] = close_w.shift(-a.horizon) / close_w - 1
        weekly["mkt_fwd"] = mkt_fwd.reindex(weekly.index)
        weekly["ticker"] = t
        frames.append(weekly)
        print(f"  {t}: {weekly.index[0].date()} -> {weekly.index[-1].date()}")
    if not frames:
        raise SystemExit("No stocks loaded.")
    panel = pd.concat(frames).rename_axis("date").reset_index()
    gain = panel.fwd - panel.mkt_fwd if a.vs_market else panel.fwd
    panel["hit"] = gain >= a.target
    return panel


# --------------------------------------------------------------------------
# SCORING EVERY CHECKLIST AT ONCE
# Each row's conditions are packed into one number (bit i = condition i met).
# A checklist S fires on a row when all of S's bits are set. Summing over all
# "supersets" gives counts for all 2^n checklists in one pass.
# --------------------------------------------------------------------------
def superset_sum(a, n):
    a = a.astype(np.float64).copy()
    idx = np.arange(len(a))
    for i in range(n):
        lo = idx[(idx & (1 << i)) == 0]
        a[lo] += a[lo | (1 << i)]
    return a


def wilson_low(hits, n, z=1.645):
    """Lower edge of the 95% one-sided confidence range for a hit rate. Punishes small samples."""
    n = np.maximum(n, 1)
    p = hits / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / (1 + z * z / n)


def score_checklists(codes, hit, fwd, n):
    size = 1 << n
    cnt = superset_sum(np.bincount(codes, minlength=size), n)
    hits = superset_sum(np.bincount(codes, weights=hit.astype(float), minlength=size), n)
    fsum = superset_sum(np.bincount(codes, weights=fwd, minlength=size), n)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(cnt > 0, hits / cnt, np.nan)
        avg = np.where(cnt > 0, fsum / cnt, np.nan)
    return cnt, hits, rate, avg


def count_episodes(panel, fired, horizon):
    """Separate occurrences: a firing counts as new only if the same stock hadn't fired in the previous `horizon` weeks."""
    s = pd.Series(fired, index=panel.index)
    total = 0
    for _, g in s.groupby(panel.ticker):
        recent = g.shift(1, fill_value=False).astype(float).rolling(horizon, min_periods=1).max().astype(bool)
        total += int((g & ~recent).sum())
    return total


def md(df, index=False):
    """Markdown table if the 'tabulate' package is installed, plain text otherwise."""
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return "```\n" + df.to_string(index=index) + "\n```"


def items_of(S, names):
    return [names[i] for i in range(len(names)) if S & (1 << i)]


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=None, help="your own list of stocks")
    p.add_argument("--universe", choices=["default", "sp500"], default="default",
                   help="sp500 = every current S&P 500 member (list from Wikipedia)")
    p.add_argument("--sector", default=None,
                   help="only stocks in this S&P 500 sector, e.g. 'energy' or 'information technology'")
    p.add_argument("--horizon", type=int, default=13, help="weeks ahead to measure the gain (13 = about 3 months)")
    p.add_argument("--target", type=float, default=0.15, help="gain that counts as 'big' (0.15 = +15%%)")
    p.add_argument("--vs-market", action="store_true", help="big gain = beat the S&P 500 by --target")
    p.add_argument("--max-items", type=int, default=4, help="most items allowed in one checklist")
    p.add_argument("--min-signals", type=int, default=50, help="fewest training firings for a checklist to be ranked")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--out", default="results_checklist")
    p.add_argument("--cache", default="data_cache")
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()

    if a.tickers is None:
        if a.universe == "sp500" or a.sector:
            a.tickers = ["DEMO%03d" % i for i in range(60)] if a.demo else sp500_list(Path(a.cache), a.sector)
        else:
            a.tickers = DEFAULT_TICKERS

    names = list(CONDITIONS)
    n = len(names)
    print(f"Loading {len(a.tickers)} stocks + S&P 500...")
    panel = build_panel(a)

    # Latest week per stock, kept before dropping rows with no future return yet
    current = panel.sort_values("date").groupby("ticker").tail(1).set_index("ticker")

    data = panel.dropna(subset=["fwd", "mkt_fwd"]).reset_index(drop=True)
    codes = np.zeros(len(data), dtype=np.int64)
    for i, nm in enumerate(names):
        codes |= data[nm].to_numpy().astype(np.int64) << i

    dates = np.sort(data.date.unique())
    split_date = pd.Timestamp(dates[int(len(dates) * a.train_frac)])
    embargo = split_date - pd.Timedelta(weeks=a.horizon)  # stop training early so train gains don't overlap test weeks
    tr = (data.date <= embargo).to_numpy()
    te = (data.date > split_date).to_numpy()

    hit, fwd = data.hit.to_numpy(), data.fwd.to_numpy()
    c_tr, h_tr, r_tr, f_tr = score_checklists(codes[tr], hit[tr], fwd[tr], n)
    c_te, h_te, r_te, f_te = score_checklists(codes[te], hit[te], fwd[te], n)
    base_tr, base_te = hit[tr].mean(), hit[te].mean()

    S = np.arange(1 << n)
    size = np.array([bin(s).count("1") for s in S])
    eligible = (S > 0) & (size <= a.max_items) & (c_tr >= a.min_signals)
    conf = np.where(eligible, wilson_low(h_tr, c_tr), -1)
    order = np.argsort(-conf)
    top = [s for s in order[: a.top] if eligible[s]]

    rows = []
    for s in top:
        fired = (codes & s) == s
        rows.append({
            "checklist": " + ".join(items_of(s, names)),
            "items": int(size[s]),
            "train_fired": int(c_tr[s]),
            "train_episodes": count_episodes(data[tr], fired[tr], a.horizon),
            "train_hit_rate": r_tr[s],
            "train_confidence_floor": conf[s],
            "train_lift": r_tr[s] / base_tr,
            "train_avg_gain": f_tr[s],
            "test_fired": int(c_te[s]),
            "test_episodes": count_episodes(data[te], fired[te], a.horizon),
            "test_hit_rate": r_te[s],
            "test_lift": r_te[s] / base_te if base_te > 0 else np.nan,
            "test_avg_gain": f_te[s],
        })
    table = pd.DataFrame(rows)

    # Overfitting check across all eligible checklists that also fired enough in the test years
    both = eligible & (c_te >= max(10, a.min_signals // 5))
    rank_corr = pd.Series(r_tr[both]).rank().corr(pd.Series(r_te[both]).rank()) if both.sum() > 2 else np.nan
    beat_base = int((eligible & (conf > base_tr)).sum())

    # "Some of the items" view for the #1 checklist
    partial = []
    if top:
        best = top[0]
        k = int(size[best])
        met = np.zeros(len(data), dtype=int)
        for i in range(n):
            if best & (1 << i):
                met += (codes >> i) & 1
        for j in range(k, -1, -1):
            m_tr, m_te = tr & (met == j), te & (met == j)
            partial.append({
                "items_met": f"{j} of {k}",
                "train_weeks": int(m_tr.sum()),
                "train_hit_rate": hit[m_tr].mean() if m_tr.any() else np.nan,
                "test_weeks": int(m_te.sum()),
                "test_hit_rate": hit[m_te].mean() if m_te.any() else np.nan,
            })
    partial = pd.DataFrame(partial)

    # Who matches the top checklists right now
    cur_rows = []
    for t, r in current.iterrows():
        on = [nm for nm in names if r[nm]]
        code = sum(1 << i for i, nm in enumerate(names) if r[nm])
        full = [str(rank + 1) for rank, s in enumerate(top) if (code & s) == s]
        cur_rows.append({
            "ticker": t,
            "week_of": r.date.date(),
            "items_met_now": ", ".join(on) if on else "(none)",
            "top_checklists_fully_met": ", ".join(full) if full else "-",
        })
    cur = pd.DataFrame(cur_rows)

    # ---------------- write outputs ----------------
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "top_checklists.csv", index=False)
    partial.to_csv(out / "partial_matches.csv", index=False)
    cur.to_csv(out / "current_matches.csv", index=False)
    all_rows = pd.DataFrame({
        "checklist": [" + ".join(items_of(s, names)) for s in S[1:]],
        "items": size[1:], "train_fired": c_tr[1:].astype(int), "train_hit_rate": r_tr[1:],
        "test_fired": c_te[1:].astype(int), "test_hit_rate": r_te[1:],
    })
    all_rows.to_csv(out / "all_checklists.csv", index=False)

    gain_word = f"beat the S&P 500 by {a.target:.0%}+" if a.vs_market else f"gained {a.target:.0%}+"
    pct = lambda x: "-" if pd.isna(x) else f"{x:.0%}"
    lines = [
        "# Checklist scan summary\n",
        f"- Stocks: {data.ticker.nunique()} | weeks scanned: {len(data):,}",
        f"- 'Big gain' = {gain_word} over the next {a.horizon} weeks",
        f"- Training: {pd.Timestamp(dates[0]).date()} -> {embargo.date()} | "
        f"Test: {(split_date + pd.Timedelta(days=1)).date()} -> {pd.Timestamp(dates[-1]).date()}",
        f"- Base rate (any random week): train {pct(base_tr)}, test {pct(base_te)}",
        f"- {n} checklist items -> {(1 << n) - 1:,} possible checklists; "
        f"{int(eligible.sum()):,} have <= {a.max_items} items and fired {a.min_signals}+ times in training\n",
        "## Top checklists (ranked on training years only)\n",
        "Confidence floor = the hit rate we're 95% sure it beats, given how often it fired. "
        "Lift = hit rate / base rate (1.0 = no better than random). Episodes = separate occurrences.\n",
    ]
    if len(table):
        show = table.copy()
        for c in ["train_hit_rate", "train_confidence_floor", "test_hit_rate", "train_avg_gain", "test_avg_gain"]:
            show[c] = show[c].map(pct)
        for c in ["train_lift", "test_lift"]:
            show[c] = show[c].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}x")
        show.index = range(1, len(show) + 1)
        lines.append(md(show, index=True))
    else:
        lines.append("No checklist fired often enough. Add stocks or lower --min-signals.")
    lines += [
        "\n## Overfitting check\n",
        f"- Checklists whose confidence floor beats the base rate in training: {beat_base:,} "
        "(the number quantum counting would estimate)",
        f"- Do training winners stay winners? Rank correlation train vs. test hit rate: "
        f"{'-' if pd.isna(rank_corr) else f'{rank_corr:.2f}'} (near 0 = mostly luck)",
    ]
    if len(partial):
        ps = partial.copy()
        ps["train_hit_rate"] = ps.train_hit_rate.map(pct)
        ps["test_hit_rate"] = ps.test_hit_rate.map(pct)
        lines += ["\n## Does hitting only some items still work? (#1 checklist)\n", md(ps)]
    lines += ["\n## Right now\n", md(cur)]
    report = "\n".join(lines)
    (out / "summary.md").write_text(report)
    print("\n" + report)
    print(f"\nSaved to {out}/: summary.md, top_checklists.csv, partial_matches.csv, current_matches.csv, all_checklists.csv")


if __name__ == "__main__":
    main()
