# Asset Lab

A first working asset research pipeline: return correlations, rolling relationship
charts, and a small asset-selection experiment compared across an exact solver
and an ideal QAOA quantum simulation. Each run produces a self-contained,
interactive HTML report and a reproducible data snapshot.

The original checklist scanner remains available below. It answers a different
question: which combinations of conditions preceded large stock gains?

The research goal is to discover a reproducible market signal, profitable after
costs, using quantum methods, and measure what those methods contribute beyond
classical alternatives. Finding a profitable classical strategy first is not a
prerequisite. Classical methods are comparison controls throughout discovery.

The current implementation provides `trade_lab.py`: a long-only relative-value
backtester, cost comparisons, a timestamped forward candidate journal, and a
small quantum trade-selection experiment. `quantum_search.py` now adds an
offline Grover rule-search feasibility prototype. Validated signal discovery
remains a research goal; the trade selector does not implement signal learning.

## Known problems and research gaps

These findings describe the saved experiments as of November 25, 2026. They
do not establish that quantum methods generally fail at market research.

### Latest hardware experiment

The historical example uses the June 6, 2014 signal for EFA and XLV. Two asset
decision bits and two unused-slot bits make four qubits, representing only four
valid economic choices: cash, EFA, XLV, or both. The model optimum is both.
IBM job `dc37e06q4uuc73a6e330` ran on `ibm_kingston` with 4,096 shots.

| Method | Result on the specified model |
| --- | --- |
| Exact classical enumeration | Checks all four valid choices and returns the optimum |
| Uniform random sampling of valid choices | 25% optimum probability per draw, analytically |
| Ideal QAOA simulated on a CPU | 12.87% optimum samples; 48.14% valid samples, out of 4,096 |
| IBM QAOA hardware | 9.69% optimum shots; 46.02% valid shots, out of 4,096 |

The percentages count all generated samples, including invalid ones. Both QAOA
runs contained the optimum, so selecting their best measured valid result
returns the same answer as enumeration. These per-draw rates do not establish
an advantage in final answers, runtime, or money made. Comparing complete runs
requires budgets and overhead as well as sample quality. The ideal simulator
executes the quantum algorithm on a CPU; it is not the classical optimizer.

Evidence is saved locally in
[`analysis.json`](results_asset_lab/trading/20261125T052641Z-bda72df5/quantum/20261125T054410Z-e67f5956/analysis.json)
and
[`hardware_results.json`](results_asset_lab/trading/20261125T052641Z-bda72df5/quantum/20261125T054410Z-e67f5956/ibm/hardware_results.json).
Generated results are ignored by Git, so these links require the original run.

### Problems to solve

1. **The current quantum task does not discover signals.** Classical code picks
   eligible mean-reversion signals and supplies a covariance matrix and an
   assumed convergence score. QAOA chooses a subset using that supplied
   objective. The earlier correlation demo also computes correlations locally.
   Neither experiment asks the QPU to learn an unknown predictive relationship.
2. **The encoding wastes samples before hardware noise is introduced.** Over
   half the ideal samples violate the unused-slot constraint. Its unconditional
   optimum rate already trails uniform sampling of valid choices. Circuit
   design, constraint handling, angle fitting, and hardware effects need separate
   investigation; noise alone cannot explain the current shortfall.
3. **The benchmark is too small to demonstrate a computational need for QC.**
   Four choices are trivial to enumerate. Ten qubits have 1,024 state amplitudes,
   so a small ideal circuit can also be simulated classically. The current local
   implementation caps experiments at 16 total qubits and uses enumeration and
   statevectors. More assets or a larger backend alone do not establish value.
4. **Optimizing this score is not evidence of trading profit.** The convergence
   score assumes a stationary peer basket and is not a calibrated return
   forecast. The hardware example does not evaluate subsequent realized P&L or
   alter the historical trading ledger.
5. **The initial trading hypothesis has no demonstrated edge.** The saved
   walk-forward preset returned **-5.45% after 10 bps per side**, across 74
   completed trades from January 3, 2012 through November 24, 2026. Its zero-cost
   result was also negative. Average invested exposure was only 3.12%, so raw
   comparisons with fully invested buy-and-hold are not alpha estimates.
   See the saved
   [`trading.json`](results_asset_lab/trading/20261125T052641Z-bda72df5/trading.json).
6. **We have not measured the full cost of a discovery.** Angle optimization,
   simulation, duplicate proposals, invalid samples, data preparation, QPU
   execution, and classical evaluation all consume resources. A fair comparison
   needs repeated trials and declared budgets, not one favorable bitstring or
   equal shot counts alone.
7. **Novel-looking patterns can be false discoveries.** Selected-peer p-values
   are nominal filters, the universe is fixed using present-day assets, and
   repeatedly inspected historical periods are exploratory. Every feature,
   strategy, and parameter search is another trial. New rules need validation
   that search decisions cannot inspect, followed by timestamped forward tests.
8. **The current journal is not a running trading account.** It records candidate
   observations, without persistent account positions, paper fills, scheduling,
   or forward realized P&L. Those are still required to test an eventual signal
   as a trading system.

## Revised objective: quantum-assisted signal discovery

The question is: **Can a quantum-assisted model or search discover a useful
market relationship that the specified classical methods miss under declared
research budgets, and does that relationship remain useful after costs on
unseen data?** Both approaches start before a profitable edge is known.

Classical data preparation and backtesting remain necessary. The quantum
component must influence which relationships are learned or proposed, rather
than only selecting trades after our existing rules have defined the signals.
All models see the same information available at each decision time.

### First scaling experiment: account for the cost of Grover search

The search path now has an offline feasibility prototype:

```bash
python -m pip install -r requirements-quantum.txt
python quantum_search.py --demo
```

To construct its inputs from an existing adjusted-price snapshot:

```bash
python quantum_search.py \
  --prices results_asset_lab/trading/YOUR_TRADING_RUN/prices.csv \
  --training-end 2021-12-31 \
  --episodes 8 \
  --min-support 2 \
  --min-score 1
```

This changes the quantum task from choosing prefiltered mean-reversion trades
to searching conjunctions of observed market conditions. The prototype uses
six binary conditions for SPY with QQQ as a market input: positive trailing
1-, 5-, and 20-session returns, price above its 20-session mean, 5-session
volatility above 20-session volatility, and positive 5-session market return.
The target and market are configurable. A selected condition must hold for a
rule to enter; selecting no conditions represents always entering.

Inputs are the first chronological, nonoverlapping episodes after indicator
warmup, with entry at the next session's close and a ten-session holding period.
Only episodes ending on or before `--training-end` are included. The score is
the **sum** of episode returns after ten-basis-point costs per side, rounded to
25-basis-point units by default. `--min-score 1` means at least one such unit
in aggregate, with at least `--min-support` trades. This is a deliberately
simple training predicate, not a compounded portfolio return or proof of an
edge. Rounding can change which rules pass; each episode's score differs from
its unrounded return by at most half a score unit.

The oracle computes eligibility, support, signed score arithmetic, threshold
checks, and uncomputation from these observations. It does **not** receive a
classically precomputed list of winning rules. There is no assumed QRAM or
free call to a Python backtester inside the quantum circuit. Different binary
feature sets can be supplied with `--episodes-csv`: one condition per column,
with one integer `score_units` outcome per row. That input must already account
for execution, costs, and training boundaries; price-specific flags apply only
to `--prices`.

Every run saves `episodes.csv`, `problem.json`, `audit.json`, provenance, and,
for bounded cases, a complete `grover.qpy` / `grover.qasm` circuit under
`results_asset_lab/search/`. The audit reports:

- Search qubits **plus** support, score, and temporary workspace qubits.
- Actual reference-compiled CX gates and depth for the oracle and exported
  circuit. Compilation assumes all-to-all connectivity; device routing,
  calibration, and error correction are not modeled.
- A classical superset-sum baseline exploiting the conjunction structure,
  rather than comparing only against independent brute-force checks.
- A classical search merging rules that fire on the same historical episodes.
  With `E` episodes there are at most `2^E` distinct firing sets, regardless of
  how many nominal rule encodings exist. A huge bit count with very little data
  can therefore remain easy classically and statistically underdetermined.
- Ideal success versus iterations and black-box scaling up to 256 search bits.
  These are mathematical diagnostics, not hardware measurements or market
  forecasts. Small-case exhaustive scoring provides the marked count only for
  validation; an unknown-solution search must account for learning or handling
  an unknown marked fraction.

Compilation/export is capped at 12 conditions, 16 episodes, and 12 score bits;
exhaustive classical diagnostics stop above 20 conditions. Larger allowed
inputs still get a register-size plan without an exponential statevector or
precomputed answer table. This command submits no job, spends no QPU allocation,
and does not change the trading ledger. Its files use a distinct experiment
schema; the existing QAOA `ibm_quantum.py submit` command does not accept them.

The first saved-price audit used six conditions and eight historical episodes:
**20 total qubits, 4,043 CX gates, and depth 9,090 per oracle** in the reference
compilation. Classical scoring found 17 passing rules among 64 in about 34
microseconds; merging equivalent rules left just 20 distinct firing sets. This
is the cost of this straightforward oracle, not a minimum achievable cost.
The ideal formula gives 99.71% passing probability after one Grover iteration
but 17.70% after two: amplification can overshoot even without hardware noise.
These probabilities are mathematical diagnostics using the classically known
marked count, not QPU results. Evidence:
[`audit.json`](results_asset_lab/search/20261125T180633Z-827b39dd/audit.json).

**Scaling has to follow an algorithmic advantage.** Grover reduces unstructured
search from order `N/M` classical checks to order `sqrt(N/M)` coherent oracle
calls for `M` marked candidates among `N` candidates. For a single
marked item, `2^256 -> 2^128` is a quadratic improvement in `N = 2^256`, while
both costs remain exponential in the 256-bit description. Even `2^128` calls
would be infeasible here. Increasing QAOA shots does not implement this
algorithm, and increasing a timeout only permits a longer job. Repetitions
improve sampling precision; they do not remove circuit bias or hardware noise.
More coherent depth can amplify a signal with the right algorithm, but it also
accumulates errors. See [IBM's Grover explanation](https://quantum.cloud.ibm.com/learning/en/modules/computer-science/grovers)
and its [circuit-depth discussion](https://eu-de.quantum.cloud.ibm.com/docs/en/tutorials/grovers-algorithm).

The next gate is to reduce the measured oracle/workspace cost, test increasing
rule sizes against strong classical methods, and measure the success curve on
hardware where depth permits. A practical advantage requires saved search
queries to outweigh oracle construction, coherent arithmetic, noise handling,
and the best classical use of the same problem structure. Adding candidates
alone does not establish that crossover. Broader search also creates more
opportunities for historical overfitting, so any discovered rule still needs
the unseen-data and forward checks below.

### Alternative discovery experiment: learn a signal with quantum features

This is a research plan, **not an implemented command**. Keep the current QAOA
selector as a hardware and optimization benchmark while adding one discovery
experiment with the following protocol:

1. **Freeze one prediction task.** Start with a declared ETF universe and a
   ten-session holding period, entering at the next session's close. Predict
   forward returns and apply identical transaction costs, position limits, and
   execution timing to every model. Do not require signals to pass the existing
   mean-reversion filter; that would restrict discovery to its hypothesis.
2. **Supply a small, shared set of observations.** Begin with six features such
   as trailing 1-, 5-, and 20-session returns, 20-session volatility, the market's
   trailing 5-session return, and trailing 60-session correlation with the
   market. Fit scaling on training data only. This tests nonlinear combinations
   and market regimes rather than only unconditional pairwise correlations.
3. **Let the quantum computation affect the learned signal.** Encode those
   features with parameterized rotations and entangling gates. Estimate a
   quantum similarity kernel from the circuits and fit a classical predictor
   using that kernel. The hypothesis is that this representation makes useful
   relationships easier to learn. It is not assumed to contain extra market
   information or to outperform classical representations.
4. **Run credible controls alongside it.** Use the same features and temporal
   splits for regularized linear models, polynomial and RBF kernels, boosted
   trees, and a classical random-feature model. Include the ideal CPU simulation
   of the identical quantum kernel and an unentangled circuit variant. Use
   declared tuning budgets and record actual resource use. This separates the
   feature representation's effect from hardware noise and classical fitting.
5. **Bound the hardware work before submitting it.** Choose a small, fixed set
   of kernel reference examples using training data only. Declare circuit,
   shot, and retry limits; full pairwise kernel estimation grows quadratically
   with the number of examples. Record data encoding, classical fitting,
   compilation, QPU execution, and total elapsed time, with queue time separate.
6. **Keep discovery separate from evaluation.** Tune on past training and
   validation folds, purge labels crossing fold boundaries, and allow for the
   holding period and execution delay. Freeze the full procedure before a final
   test. Previously inspected historical data stays exploratory; timestamped
   future observations provide new evidence. Record all trials, including failed
   models, and assess uncertainty with methods that respect temporal dependence.
7. **Judge economic usefulness and the quantum contribution separately.**
   Report after-cost return, drawdown, turnover, exposure, calibration, and
   stability across periods. Use comparable risk and exposure when assessing
   excess performance. Check whether a combined classical-plus-quantum forecast
   adds value over classical forecasts alone on validation, then freeze it for
   testing. A different prediction or one successful trade is insufficient.

Quantum feature maps and kernels are established methods, so using one is not
itself a novelty claim; see
[Havlicek et al., *Supervised learning with quantum enhanced feature spaces*](https://arxiv.org/abs/1804.11326).
Classical models can also learn effectively from data even when a quantum
representation is involved; see
[Huang et al., *Power of data in quantum machine learning*](https://arxiv.org/abs/2011.01938).
Any contribution would need to come from the tested market hypothesis,
evaluation, or measured improvement.

A separate heuristic search experiment could reuse QAOA to propose combinations of signal
features or trading rules, evaluate them with the cost-aware backtester, and
update a training-only surrogate score. QAOA requires an explicit objective; it
cannot sample an unspecified notion of future profitability. Compare that loop
with valid random sampling, local search, simulated annealing, and suitable
classical black-box optimizers using matched evaluation budgets and reported
overhead. QAOA has no general Grover-style quadratic search guarantee.
Compare quantum and classical proposals from the **same** surrogate
to isolate the sampler's contribution. A
[2026 preprint on surrogate-guided quantum discovery](https://arxiv.org/abs/2602.09374)
offers a conceptual precedent in document-processing risk discovery, not
evidence of a market edge.

### What a successful result could establish

- **Useful discovery:** a quantum-assisted procedure finds a stable, after-cost
  signal that the tested classical searches miss under the reported protocol.
- **Added predictive value:** its forecasts improve unseen-data results beyond
  strong classical controls, with uncertainty and all research trials reported.
- **Computational advantage:** the full quantum-assisted workflow achieves a
  specified quality with fewer relevant resources than strong classical
  alternatives, including classical simulation where practical. This is a
  separate, stronger claim and has not been demonstrated here.

"The tested classical methods missed this signal under these budgets" is a
testable claim. "No classical computer could find it" does not follow from
that result. A six-to-ten-qubit discovery experiment can test a research
hypothesis, but its small circuits can be simulated classically. The signal
could still be economically useful without establishing that quantum hardware
is necessary. A negative result should also be retained and reported.

## Trading research

Update the dependencies in your existing virtual environment, then run:

```bash
python -m pip install -r requirements.txt
python trade_lab.py run --open
```

For an offline workflow check:

```bash
python trade_lab.py run --demo --open
```

Results are saved under `results_asset_lab/trading/`. To add a summary and link
to an existing Asset Lab dashboard, supply
`--attach-to results_asset_lab/YOUR_CORRELATION_RUN`.

The initial preset is a hypothesis, not a selected winning strategy:

- Each formation period contains 504 common daily price observations.
- A target asset's peers are up to three other universe members with training
  return correlation of at least 0.60. A log-price regression fits the target
  against the geometric peer basket.
- Relationships must pass a nominal Engle-Granger cointegration filter at 0.05,
  have a fitted mean-reversion half-life of 2-60 sessions, and retain sufficient
  return correlation in both halves of formation. Because peers were selected
  on the same training data, the p-values are descriptive filters, not corrected
  evidence of an investable edge.
- Models are frozen for the next 63-session test fold. There is no automatic
  search over entry parameters or selection of the best test period.
- Entries require a residual z-score of -2 or lower, but above -4, and SPY above
  its trailing 200-session mean. At most five long-only positions are permitted.
- Exits trigger at a recovered z-score of -0.5 or higher, deterioration to -4,
  a 5% observed price loss, or a ten-session holding limit. Stops fill later and
  cannot guarantee a maximum realized loss.
- A close at session `t` generates a decision that fills at session `t+1`'s
  close. The backtester never buys at the same cheap close used for the signal.
- Costs default to ten basis points of notional on each buy and each sell.
  Zero-, five-, and 25-basis-point scenarios are also reported without tuning
  the strategy. Position cash, units, costs, and closed-trade P&L reconcile.
- Positions are liquidated at predetermined fold boundaries and the end of
  evaluation, including costs. This keeps relationships fixed for every trade.

The report shows equity and drawdown charts, benchmark returns, turnover,
cash exposure, cost sensitivity, results by fold, and a complete trade ledger.
Benchmarks are equal-weight buy and hold and SPY buy and hold over the same
evaluation dates, with entry/exit costs. Their invested exposure differs from
the strategy's; return differences alone are not an alpha estimate.

The model uses adjusted-close total-return proxies and fractional units rather
than executable share quotes. Cash earns zero, and the Sharpe ratio uses a
zero risk-free rate. Complete common-session prices are required; missing
quotes are rejected rather than filled or silently bridged. The default is a
fixed present-day ETF universe, and results are exploratory historical research.
This long-only implementation does not hedge the peer basket and is not a
market-neutral pairs strategy.

Useful controls include `--tickers`, `--prices`, `--start`, `--end` (exclusive),
`--train-sessions`, `--test-sessions`, `--cost-bps`, `--entry-z`, `--exit-z`,
`--stop-z`, `--max-hold`, `--max-positions`, and `--no-trend-filter`. Every new
run saves its own rule identifier, configuration, strategy-source hash, input
snapshot, model acceptance/rejection audit, and ledger. Treat manually changed
presets as additional research trials; do not present the best trial as an
untouched test result.

### Forward candidate observations

After obtaining new completed-session data, record a snapshot using a saved run:

```bash
python trade_lab.py signals --run results_asset_lab/trading/YOUR_TRADING_RUN
```

This retains the run's frozen rules and its anchored recalibration schedule.
It appends an actual capture timestamp and archives the input prices and fitted
models. Repeated identical snapshots are not duplicated; historical observations
are not backfilled into a live journal. Edited rules or changed strategy code
require a new research run. CSV inputs require an explicitly updated `--prices`
file and are labeled `uploaded_snapshot`; demo inputs remain synthetic.

`signals.jsonl` is the append-only candidate journal. `forward_snapshots/`
retains the quote and model evidence, and the report displays the latest
observation. The initial historical preview is separate from this journal.
These are research candidates, not an account of executed trades: no paper
fills, broker orders, or forward P&L are inferred. No scheduler is installed;
run the command manually to capture each new observation.

### Quantum selection of trade candidates

Build a local QAOA experiment from a saved signal snapshot:

```bash
python trade_lab.py quantum --run results_asset_lab/trading/YOUR_TRADING_RUN --open
```

If no frozen entry signals qualify, the command reports that fact and creates
no circuit. To exercise the hardware path on a clearly labeled historical
example instead:

```bash
python trade_lab.py quantum --run results_asset_lab/trading/YOUR_TRADING_RUN --example --open
```

The example is the first chronological session with at least two eligible
signals; it is not chosen using later profits. `--as-of YYYY-MM-DD` selects an
explicit historical session instead. Each experiment is saved under that
trading run's `quantum/` folder with an exact classical solution, ideal QAOA
simulation, an Ising model, the prices available at the signal, and a report.
The parent trading report links to the latest experiment.

The objective is risk aversion times holding-horizon variance minus a
model-implied convergence score net of assumed round-trip costs. Each asset's
score assumes its peer basket remains stationary. It is not a calibrated
forecast or demonstrated alpha. Formation covariance uses only earlier data.
The horizon defaults to ten sessions and risk aversion to five; these are
explicit research settings, not settings selected from test-period profits.

The model chooses **at most** the configured number of new positions. Binary
unused-slot variables enforce `sum(asset_bits) + unused_slots = capacity`.
Every allowed real-asset subset has one encoding, including the all-cash
choice. Cash variables are never decoded as asset names. Each selected asset
uses `1 / max_positions` of a hypothetical account, preserving the frozen
unlevered position limit. Existing account holdings are not inferred.

Controls include `--max-candidates` (1-12 real decisions), `--capacity`,
`--horizon`, `--risk-aversion`, `--layers`, `--shots`, and `--seed`. Real
decisions plus unused-slot bits must fit within 16 total qubits.

The command prints a complete IBM submission command for its experiment.
Use the saved experiment folder with the existing IBM commands:

```bash
python ibm_quantum.py prepare --run YOUR_QUANTUM_EXPERIMENT_FOLDER
python ibm_quantum.py submit --run YOUR_QUANTUM_EXPERIMENT_FOLDER --backend YOUR_AVAILABLE_BACKEND --shots 4096 --max-execution-seconds 60
python ibm_quantum.py collect --run YOUR_QUANTUM_EXPERIMENT_FOLDER
```

Preparation is offline and submission is explicit. Collection scores the
weighted unused-slot constraint, updates the experiment report with hardware
counts, and updates its parent trading dashboard. It never places trades or
rewrites the original backtest. The experiment does not establish improved
trading P&L or quantum advantage.

Method references: [relative-value trading research](https://www.nber.org/papers/w7032),
[Engle-Granger implementation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html),
and [backtest overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

## Start here

Requires Python 3.11 or newer. In this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python asset_lab.py --demo --open
```

On Windows, use `py -m venv .venv` and `.venv\Scripts\activate` instead of
the first two commands. Once installed, only the activation and run commands
are needed. The demo uses clearly labeled synthetic prices and works offline.
The report opens in your browser; it needs no server, login, or external scripts.

For real adjusted historical prices from Yahoo:

```bash
python asset_lab.py --open
python asset_lab.py --tickers SPY QQQ IWM TLT IEF GLD --choose 3 --open
python asset_lab.py --tickers AAPL MSFT IBM JPM XOM JNJ PG KO --candidates AAPL MSFT IBM JPM XOM JNJ --choose 3 --open
```

These are example research universes. The default universe contains 20 ETFs;
the default optimization candidates are its first six, in that fixed order.
Choose your universe and candidates before examining the holdout results.
Yahoo snapshots expire after 24 hours; `--refresh` forces a new download.
No source failure silently switches to demo data.

You can also provide adjusted prices from another source:

```bash
python asset_lab.py --prices my_prices.csv --tickers SPY QQQ TLT GLD --choose 2 --open
```

The CSV's first column must contain unique dates; subsequent columns must be
named for the tickers and contain positive adjusted prices. There must be at
least 252 common daily return observations. Missing prices are not forward-filled;
both the missing day's return and its following return are excluded. Supply
consistent daily sessions and a common currency; currency conversion and
cross-market close-time alignment are not implemented.

Other controls: `--start 2018-01-01`, `--end 2026-11-24` (exclusive),
`--train-fraction 0.7`, `--windows 60 120 252`, `--layers 1`, `--shots 4096`,
`--seed 42`, and `--out results_asset_lab`. Use `--help` for all options.

## What a run does

1. Loads adjusted prices and computes simple daily returns without filling gaps.
2. Splits common return rows chronologically: 70% training, 30% holdout by default.
3. Computes Pearson and Spearman correlations for each period, plus recent
   rolling Pearson correlations. The report lets you inspect any asset pair.
4. Uses only training covariance to choose exactly `k` candidate assets, with
   equal weights, to minimize estimated portfolio variance. Every feasible
   subset is enumerated as the exact reference answer.
5. Builds an equivalent QUBO and Ising model, optimizes QAOA angles on training
   costs, simulates cost phases and X rotations, and samples measurements.
6. Evaluates the exact selection, best measured feasible selection, and
   equal-weight candidate baseline on the untouched later period.

The quantum experiment supports 2-16 candidates and 1-3 QAOA layers. The wider
correlation universe can contain up to 100 assets. These are deliberate prototype
limits. Simulated measurements are not hardware results. A low optimum frequency
is a valid outcome; the report includes uniform-sampling baselines and counts
infeasible measurements. Picking the best observed feasible result also performs
classical scoring; this experiment makes no speedup claim.

For binary inclusion variables `x`, the objective is:

```text
E(x) = (xᵀ Σ_train x) / (k² * covariance_scale) + penalty * (sum(x) - k)²
x_i = (1 - z_i) / 2
```

The penalty exceeds an upper bound on the risk term, so an infeasible state
cannot be the ground state. `problem.json` includes the upper-triangular QUBO,
all Ising fields and pair couplings, constant offsets, normalization, and bit
conventions. Ticker `i` is qubit `i` and bit `i` of the integer state; displayed
bitstrings show the highest-index qubit on the left.

Each timestamped folder under `results_asset_lab/` contains:

| File | Purpose |
|---|---|
| `report.html` | Interactive, offline research report |
| `manifest.json` | Configuration, source hashes, software versions, artifact hashes |
| `prices.csv`, `returns.csv` | Actual run inputs and aligned return panel |
| `train_pearson.csv`, `train_spearman.csv` | Training correlation matrices |
| `holdout_pearson.csv`, `holdout_spearman.csv` | Later-period matrices |
| `pairs.csv`, `rolling_latest.csv` | Pair comparisons and latest rolling correlations |
| `exact_selections.csv` | Every feasible candidate selection, ranked on training risk |
| `problem.json` | Portable QUBO / Ising model for hardware mapping |
| `analysis.json` | Full results, angles, probabilities, counts, and evaluation |

The holdout is a first check, not a complete strategy backtest. Sample covariance
is not regularized; relationships and risk can change. Portfolio evaluation
assumes frictionless daily rebalancing, with no fees or market impact. There is
no automatic trading or alerting.

## IBM quantum path

Install the optional dependencies and prepare a circuit entirely offline:

```bash
python -m pip install -r requirements-quantum.txt
python ibm_quantum.py prepare --run results_asset_lab/YOUR_RUN_FOLDER
```

This writes `ibm/circuit.qasm`, `ibm/circuit.qpy`, and preparation metadata. The
Qiskit circuit implements the same Ising cost phases and mixer as the NumPy
simulator; the tests compare their probabilities for one and multiple layers.

Actual hardware requires an IBM Quantum account and an intended instance and
backend. To save your credentials locally, run:

```bash
python ibm_quantum.py login
```

The API-key prompt is hidden. Paste the full Open Plan instance CRN, beginning
with `crn:`, at the second prompt. Login saves a default Qiskit Runtime account
without connecting or submitting a workload. Existing saved credentials are
kept; login does not overwrite them. For alternative per-session credentials,
provide `IBM_QUANTUM_API_KEY` and `IBM_QUANTUM_INSTANCE` as environment variables.
Environment values take precedence over saved credentials; use
`unset IBM_QUANTUM_API_KEY IBM_QUANTUM_INSTANCE` in a Mac/Linux Terminal to
clear old overrides before using the saved account. Keep credentials outside
source control. Submission is a separate, explicit command:

```bash
python ibm_quantum.py submit --run results_asset_lab/YOUR_RUN_FOLDER --backend YOUR_BACKEND --shots 4096 --max-execution-seconds 60
python ibm_quantum.py collect --run results_asset_lab/YOUR_RUN_FOLDER
```

Submission uses Runtime SamplerV2 job mode with locally fitted angles; it does
not start a hardware optimization loop or require a persistent session. It saves
the transpiled circuit and job receipt, then exits. Collect polls once and saves
`ibm/hardware_results.json` when the job is done and updates the HTML report
with hardware measurements, exact ideal probabilities, and classical random
baselines. Hardware runs consume your instance's
allocation or incur charges according to its plan. An execution-time limit is
not a dollar spending limit.

The script refuses a second submission attempt for the same run, including when
the first response was uncertain. Reconcile uncertain attempts in the IBM
dashboard rather than automatically resubmitting. No hardware is submitted by
`asset_lab.py`, `prepare`, or the automated tests.

To refresh an existing report from saved results without contacting IBM:

```bash
python ibm_quantum.py report --run results_asset_lab/YOUR_RUN_FOLDER
```

The hardware panel includes individual outcomes, the distribution of selected
asset counts, compiled depth and gate counts, and total variation distance
from the exact ideal distribution. A separate uniformly random **valid**
portfolio baseline enforces the asset-count constraint before sampling; it is
stronger than unrestricted random bitstrings. The original analysis remains
unchanged. Report refreshes record updated hashes and a revision trail in the
manifest, retaining the original data and analysis provenance.

## Local scaling benchmarks

Run a sweep against a saved study's fixed training returns:

```bash
python benchmark_asset_lab.py --run results_asset_lab/YOUR_RUN_FOLDER
```

Defaults: 6, 10, and 14 candidate assets; choose half; one and two QAOA layers;
seeds 42, 43, and 44; 4096 measurements each. This creates 18 cases and compares
QAOA with exact enumeration, equally many uniformly random valid portfolios,
and a classical greedy search followed by improving swaps. All angle fitting
and sampling are local; the sweep submits no IBM jobs. Change the sweep with
`--sizes 6 10 14 16 --layers 1 2 --seeds 42 43 44 --shots 4096`.

Candidate sets use the first `n` assets of the original configured universe,
never a ranking based on the holdout. The benchmark holds its training split
fixed and saves every case as a complete run under the original run's
`benchmarks/` folder. Its summary appears on the original report, with links
to individual cases. Each case can be prepared and submitted using
`ibm_quantum.py` with that case's folder as `--run`.

At 14 assets choosing seven there are 3,432 legal portfolios; at 16 choosing
eight there are 12,870. Both remain feasible classical benchmarks. Simulation
and exhaustive evaluation grow exponentially; raising the cap is not a route
to unlimited scale. Larger quantum experiments need different training and
evaluation methods as well as circuits that survive hardware noise.

The comparison reports excess training variance (gap), valid/optimal shot
rates, and measured local wall times. It does not claim quantum speedup.
Reusing the original holdout for exploration requires a new final unseen
period before making a predictive claim.

IBM references: [SamplerV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/server-side-sampler-sampler-v2),
[optimization tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm),
[access plans](https://quantum.cloud.ibm.com/docs/en/ibm-quantum-platform/plans-overview).

## IU lab and cloud hosting

`problem.json` is the common model boundary for a future IU adapter. The lab
must confirm available qubits, fields, coupling signs/ranges, programmable
interaction graphs, readout, and scheduling before that adapter can be written.
The system does not assume that an IBM circuit can run unchanged on the lab
apparatus, or that the apparatus supports arbitrary Ising couplings.

A Dockerfile packages the classical worker and ideal simulator:

```bash
docker build -t asset-lab .
docker run --rm -v "$PWD/results_asset_lab:/output" asset-lab --demo --out /output
```

This is a batch worker, not a hosted web application. It can form the basis of
an IBM Cloud Code Engine job. Cloud deployment still needs persistent object
storage for results, credentials, and job configuration. The local reports
are not publicly published. The container excludes the optional IBM packages;
install those separately if building a hardware-submission worker.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover missing-price gaps, holdout isolation, direct objective/QUBO/Ising
equivalence, exact solutions, simulation reproducibility, run snapshots,
optional Qiskit circuit equivalence, and mocked IBM submission/collection.

## Original checklist scanner

Finds combinations of market conditions ("checklists") that came before big
stock gains, scores how reliable each one was, re-checks the winners on later
years, and lists which stocks match today. Read
`Quantum_Checklist_Study_Overview.pdf` first for what it does and why.

Setup takes about 5 minutes. You only do steps 1-3 once.


## Mac

1. Unzip and move the `Quant_Checklist_Scanner` folder to your Desktop.

2. Open Terminal (press Cmd + Space, type "Terminal", press Enter).

3. Paste these lines one at a time, pressing Enter after each:

       cd ~/Desktop/Quant_Checklist_Scanner
       python3 -m pip install -r requirements.txt

4. Run it:

       python3 checklist_scanner.py

Next time, open Terminal and run only the `cd` line and step 4.


## Windows

1. Unzip and move the `Quant_Checklist_Scanner` folder to your Desktop.

2. Install Python from https://www.python.org/downloads/ if you don't have it.
   On the first installer screen, CHECK the box "Add python.exe to PATH".

3. Open Command Prompt (press the Windows key, type "cmd", press Enter) and
   paste these lines one at a time:

       cd /d "%USERPROFILE%\Desktop\Quant_Checklist_Scanner"
       py -m pip install -r requirements.txt

4. Run it:

       py checklist_scanner.py

If your Desktop is synced to OneDrive and the `cd` line fails, use:

       cd /d "%USERPROFILE%\OneDrive\Desktop\Quant_Checklist_Scanner"


## If something goes wrong

| Error | Fix |
|---|---|
| `command not found: python3` (Mac) | Run `xcode-select --install`, accept the popup, try again |
| `'py' is not recognized` (Windows) | Reinstall Python and check "Add python.exe to PATH" |
| `No such file or directory` on the `cd` line | Type `cd ` (with a space), drag the folder from Finder/File Explorer into the window, press Enter |
| `externally-managed-environment` on the pip line | Add `--user` to the end of the pip line |

The first run downloads price history, so give it a minute. The full S&P 500
(~500 stocks) takes much longer the first time; after that everything is
saved in `data_cache/` and runs fast.


## Ways to run it

Mac uses `python3`, Windows uses `py`. Everything after that is the same.

    python3 checklist_scanner.py                                   # 10 default stocks
    python3 checklist_scanner.py --universe sp500                  # whole S&P 500
    python3 checklist_scanner.py --sector energy                   # one S&P 500 sector
    python3 checklist_scanner.py --tickers AAPL MSFT NVDA          # your own list
    python3 checklist_scanner.py --vs-market                       # big gain = beat the S&P 500
    python3 checklist_scanner.py --horizon 26 --target 0.25        # +25% over 26 weeks
    python3 checklist_scanner.py --demo                            # fake data, no internet

Sectors: Communication Services, Consumer Discretionary, Consumer Staples,
Energy, Financials, Health Care, Industrials, Information Technology,
Materials, Real Estate, Utilities. Partial names work, e.g. "health" or "tech".


## Reading the results

Everything is saved in the `results_checklist` folder:

- `summary.md` - the full report (opens in any text editor)
- `top_checklists.csv` - the top checklists; opens in Excel or Google Sheets
- `all_checklists.csv` - every checklist the scanner scored
- `partial_matches.csv` - whether meeting only some items of the #1 checklist still worked
- `current_matches.csv` - which stocks meet the top checklists this week

Key columns:
- hit rate - how often a big gain followed when the checklist fired
- base rate - how often a big gain followed on any random week
- lift - hit rate / base rate. 1.0x = no better than random
- confidence floor - the hit rate we're 95% sure it beats, given how often it fired
- test_ columns - the same numbers on later years the checklist was never ranked
  on. These are the ones to trust.


## Adding a checklist item

Add one line to the `CONDITIONS` list near the top of `checklist_scanner.py`.
In a proposed Grover encoding, each item is one search-register qubit; oracle
implementation may require additional qubits and substantial circuit depth.
There is no Grover implementation in the original scanner. Lab compatibility
and usable system size must be confirmed with the lab.

The original scanner's Wilson confidence scores treat stock-week firings as
independent; overlapping forward returns and correlated assets violate that
assumption. Episode counts are descriptive and do not correct those confidence
scores. Treat the original confidence wording as exploratory, not as a verified
95% guarantee. Its legacy price cache also persists until manually refreshed;
the new Asset Lab uses a separate cache with a 24-hour lifetime.
