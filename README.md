# Solana Meme-Coin Trading Bot

An end-to-end machine-learning trading system for Solana meme coins. It collects 1-minute OHLCV candles from GeckoTerminal, builds a 169-feature dataset, trains two gradient-boosted models (a **move** model and a **direction** model), tunes a trading strategy on them in a causality-tested simulator, and runs that strategy live: many independent bots behind a web dashboard, in **paper** mode (simulated fills) or **shadow** mode (real Jupiter swap quotes, never sent).

The repository has three parts:

* **Development notebooks:** the five-step pipeline from raw candles to a tuned strategy, run in order.
* **Research notebook:** a measurement study for the live pipeline (when is a 1-minute candle final?).
* **Live pipeline:** the multi-bot application (Python back end and a single-page dashboard) that runs on a Raspberry Pi.

[`FINDINGS.md`](FINDINGS.md) is the project's running log: every experiment, its result and the decision taken, from the first label search to the current pipeline. Its §0 is the project guide.

> **Headline (current notebook output, first 30 of 50 strategy runs):** from $20 per coin, the run that was best on validation ends at **$43.76 per coin on 100 unseen test coins** (2.2×). The mean over all runs is $47.74 on test. These are simulated results with 1.5% slippage per side; no real money has been traded.

---

## Repository layout

```
├── README.md
├── FINDINGS.md                    the research log (project guide in §0)
├── development_notebooks/         the pipeline, run in this order
│   ├── meme_coin_pipeline.ipynb   1  data collection
│   ├── data_cleaning.ipynb        2  fill rule, death points, short coins
│   ├── Dataset_creationV9.ipynb   3  features, labels, splits
│   ├── model_search.ipynb         4  move + direction model search
│   └── strategy.ipynb             5  strategy optimisation, test evaluation, live export
├── research_notebooks/
│   └── finalization_study.ipynb   when a GeckoTerminal candle stops changing
└── live_pipeline/                 the trading application
    ├── app.py  bot_manager.py  bot_loop.py
    ├── data_collection.py  features.py  scoring.py  strategy.py
    ├── execution.py  dex.py  database.py
    ├── templates/dashboard.html
    └── artifacts/                 models, reference distributions, parameters (from notebook 5)
```

The notebooks create their data folders (`raw_data/`, `processed_data/`, `dataset_v9/`, `model_search/`, `model_results/`, `strategy_results/`) next to themselves. The data and the trained artifacts are not part of the repository.

---

## Terminology

**Solana (SOL):** a fast, low-fee blockchain that hosts most meme-coin trading. All money in the live system is tracked in SOL; the dashboard converts it to USD with a live SOL price. The simulations count in USD, with $20 of starting capital per coin.

**Meme coin:** a highly speculative token whose price is driven by attention and short-term trading rather than fundamentals. Extremely volatile and often short-lived.

**Candle / OHLCV:** one minute of trading: open, high, low, close and volume. Every minute without volume is a flat candle at the last real close, so every minute has a row and *row number = minutes since the coin's first candle*.

**Move:** a rise of at least 10% within the next 30 minutes. The *move model* estimates its probability.

**Direction:** whether the price first reaches +15% or −15% within the next 2 hours. The *direction model* estimates the probability that +15% comes first.

**Score / percentile:** a model output converted to its rank within the model's predictions on the *tune* split (0 = lowest, 1 = highest). All strategy thresholds are on these percentiles, so they mean the same in research and live.

**Candidate minute:** a minute that passes the entry gate (combo score and price-position filters). Buys and rebases only happen on candidate minutes.

**Rebase / race:** the loss floor and the stale time never sell. They start a *race*: the position stays open and is re-anchored (rebased) at the next acceptable candidate minute, unless the take-profit comes first.

**Paper / shadow / live:** the same decisions with different fills. *Paper* simulates fills with a fixed fee. *Shadow* asks Jupiter for a real swap quote at decision time and records that fill, without signing or sending anything. *Live* (real swaps) is not implemented.

---

## Development notebooks

The pipeline that produced the dataset, the models and the strategy parameters. Every notebook is resumable (finished work is skipped) and writes into its own output folder.

| # | Notebook | Reads | Writes | What it does |
|---|---|---|---|---|
| 1 | [`meme_coin_pipeline`](development_notebooks/meme_coin_pipeline.ipynb) | GeckoTerminal API, `COIN_LIST` | `raw_data/` | Collects 1-minute OHLCV per pool, applies the fill rule, stops collecting a coin at its death point (48 h window < 1,000 SOL volume, after a 48 h grace) and marks it dead. |
| 2 | [`data_cleaning`](development_notebooks/data_cleaning.ipynb) | `raw_data/` (never modified) | `processed_data/` | The fill rule again, a volume death mark and a pump-rate (label) death mark, cut at the earlier one; leaves out coins with less than 12 h of data. |
| 3 | [`Dataset_creationV9`](development_notebooks/Dataset_creationV9.ipynb) | `processed_data/` | `dataset_v9/` | 169 causal features and the labels for every minute, a **leakage test** on all features, and a stratified coin-level split into 5 splits. |
| 4 | [`model_search`](development_notebooks/model_search.ipynb) | `dataset_v9/splits`, `processed_data/` | `model_search/` | Optuna searches for the move and the direction model (validation metric minus a model-size penalty); the top 10 of each are then judged in strategy simulations, and the pair is picked there. |
| 5 | [`strategy`](development_notebooks/strategy.ipynb) | `model_results/` (the chosen pair), `dataset_v9/splits`, `processed_data/` | `strategy_results/` | 50 strategy optimisations (optimised on tune, selected on val_tune), each evaluated **once** on test; the last cell exports the live artifacts. |

**Data.** 402 coins, collected by hand-picking pools that met a fixed screen when added: Solana, at least $20k liquidity, GeckoTerminal security score ≥ 50, between 12 hours and 3 months old. Prices and volume are in SOL. FINDINGS §0.4 documents the data's provenance and every alteration.

**Splits (by coin).** Coins are stratified by length × pump rate into 15 / 10 / 25 / 25 / 25%:

| Split | Coins | Used for |
|---|---|---|
| `train` / `val_train` | 59 / 39 | training the models / early stopping |
| `tune` / `val_tune` | 102 / 102 | optimising the strategy / selecting and reporting; the percentile references come from `tune` |
| `test` | 100 | read **once** per strategy run, after everything is fixed |

**Features.** 169 features, every window on a fixed set from 1 minute to 7 days:
* **Lifecycle and time:** minutes since launch, launch characteristics, hour of day (UTC).
* **Volatility:** several horizons, its dynamics (change, clustering, trend) and context (percentile, intraday pattern).
* **Volume:** ratios, z-scores, dispersion, autocorrelation, skewness, concentration, volume × volatility.
* **Momentum:** returns from 1 minute to 3 days, all-time-high distance, price positions within the last 15 minutes to 24 hours.
* **Pump dynamics:** two pump definitions (one per model), each with clustering, velocity, density, failure rate and time between pumps.
* **Liquidity:** a causal estimate of the SOL needed for a 1% price impact (`estimated_sol_1pct_impact`), its age and uncertainty, liquidity trends.

**Labels.** `label_10pct_30m` (move model), `label_fpclass_15pct_2h` (direction model: up first / down first / neither) and `label_meta_minutes_to_end` (training rows need ≥ 240 minutes of future; never a feature).

**The models.**
* The **move** model predicts P(close rises ≥ 10% within 30 minutes).
* The **direction** model predicts P(+15% is reached before −15% within 2 hours). It is trained **only on candidate minutes**: tradable, move score in its top 50%, and an outcome within 2 hours. This "A50" recipe was the single biggest model-side improvement found (FINDINGS §21–22).
* The buy score (**combo**) is the tune percentile of P(move) × P(direction).

### Evaluation rules

The rules every experiment followed (FINDINGS §0.7):
* a **leakage test** for features: values must be identical with and without future data;
* a **causality test** for every simulator change: a coin simulated on its full data and on data cut at minute C must give identical trades before C;
* **controls** (shuffled scores, random blocks) for any rule that skips or changes trades;
* a decision rule **fixed before** looking at results, and confirmation with **new model seeds**;
* the test split read **once**.

Two pitfalls shaped these rules. First, a simulator rule applied after the fact leaked the future and produced a fake doubling of the score (§35). Second, model-level AUC does not predict strategy results; only the simulations do (§20).

---

## Research notebook

| Notebook | Purpose |
|---|---|
| [`finalization_study`](research_notebooks/finalization_study.ipynb) | **When is a live candle final?** The dataset's candles were collected long after the fact, but the live bot reads a minute seconds after it ends. The study reads the just-ended candle at 1.0, 1.5, … 40 s after its close (a sweep of 79 offsets within GeckoTerminal's rate limit) and compares each read with the value 5 minutes later. It checks the computer's clock against the server's `Date` header on every call, survives restarts, and logs every rate-limit hit. A standalone statistics cell reports, per field and for the whole candle, the earliest second at which 95 / 99 / 99.9% of candles are final. |

**First reading** (one coin, about one day; FINDINGS §40.8): candles become final much later than expected. At 1–3 s almost no candle is final; at 38–40 s about half are. More coins and a longer sweep are needed before the live read time is set from it.

---

## Results

All from the saved notebook outputs on the rebuilt data (FINDINGS §42).

**Model search** (val_tune, 10 strategy runs per model):

| | Mean final capital per coin |
|---|---|
| best move model alone (`move_trial057`, 0.36 MB) | $48.32 ± 0.48 |
| with the best direction model (`dir_for_move_trial057_trial002`, 0.07 MB) | $50.62 ± 0.73 |
| value of the direction model | **+$2.30** (better in 9/10 runs, paired p 0.003); median over all 10 direction models +$1.47, 10/10 better |

**Strategy** (first 30 of 50 runs):

| | val_tune | test |
|---|---|---|
| mean over runs | $50.03 | $47.74 |
| run that is best on val (the honest result) | $54.03 | **$43.76** |

Picking the best of many runs on validation selects for validation luck: that run drops $10 from val to test, against $2 on average. The test number of the best-val run is therefore the one to quote, and it agrees with an earlier hold-out estimate on other coins (≈ $44, §35.2).

---

## Live pipeline

### Overview

What a bot does, end to end:

1. **History.** On its first start it fetches the coin's full minute history from GeckoTerminal (1000 candles per request, up to `historical_days` back) and applies the fill rule. It refuses to start if the history does not reach the coin's first minute, because several features (minutes since launch, launch and all-time-high features, the 12-hour trading start) depend on it.
2. **Every minute** it fetches the newest candles, re-checks the previous minute for late revisions, and appends the new minutes to the history.
3. **Features.** It computes the 169 features for the new minute(s) with the dataset notebook's own code (`features.py`).
4. **Scores.** The move and direction models predict; the predictions become three percentile scores: *combo*, *direction* and *move* (`scoring.py`).
5. **Strategy.** The A50 state machine (`strategy.py`) decides: buy, sell, rebase or nothing.
6. **Execution.** Paper or shadow fills (`execution.py`, `dex.py`).
7. **Records.** Candles, scores, trades, capital and the full strategy state go to the SQLite database; the dashboard reads from there.

### Architecture

The system is **process-oriented**: the web app and one worker process per bot coordinate only through a shared SQLite database. `bot_manager.py` launches each bot as its own `bot_loop.py` subprocess (`--bot-id N`). Up to 10 bots are supported.

```
┌───────────────────────────────────────────────────────────────┐
│  Web process                                                  │
│  ┌────────────┐    ┌──────────────┐   ┌────────────────────┐  │
│  │  app.py    │    │ bot_manager  │   │ SOL price collector│  │
│  │  (Flask)   │◄──►│ (BotManager) │   │ (background thread)│  │
│  └────────────┘    └──────┬───────┘   └────────────────────┘  │
└────────────────────────────┼──────────────────────────────────┘
                             │ subprocess.Popen --bot-id N
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ bot_loop │   │ bot_loop │   │ bot_loop │   … one per bot
        │  (bot 1) │   │  (bot 2) │   │  (bot N) │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             └──────────────┼──────────────┘
                            ▼
                  ┌────────────────────┐
                  │   SQLite database  │  ← single source of truth
                  └────────────────────┘
```

A bot's strategy state (open position, phase, races, re-entry bookkeeping) is saved as JSON after **every minute**. A stopped or crashed bot therefore resumes exactly where it was: it reloads the stored history, fetches the missing minutes, and replays them through the state machine. It does not buy on minutes in the past; a sell that became due while it was down is executed at once.

### Files

| File | Responsibility |
| --- | --- |
| `app.py` | Flask server: the dashboard and its API, plus the SOL-price background thread |
| `bot_manager.py` | Bot process lifecycle (initialise / start / pause / resume / stop / deallocate) and the dashboard's bot data |
| `bot_loop.py` | Per-bot worker process: the only module that wires everything together |
| `data_collection.py` | GeckoTerminal OHLCV fetching (full history, gap filling, live minutes), rate-limit handling, the fill rule |
| `features.py` | The feature code of `Dataset_creationV9`, copied verbatim, plus a live section that computes only the newest minutes |
| `scoring.py` | Loads the two models and turns their predictions into the three percentile scores |
| `strategy.py` | The A50 strategy as a minute-by-minute state machine, its parameters and its fill prices |
| `execution.py` | Buy and sell execution: paper and shadow (live stubbed) |
| `dex.py` | Jupiter swap quotes and network-fee estimates for shadow mode (never signs or sends) |
| `database.py` | SQLite schema, migrations, queries and all persistence |
| `templates/dashboard.html` | Single-page frontend served by Flask |

### From the notebooks to the bot

The last cell of `strategy.ipynb` writes everything the bot needs to `strategy_results/live_artifacts/`. Copy those files into `live_pipeline/artifacts/`:

```
artifacts/
├── move.ubj, direction.ubj           the chosen models (each file carries its feature list)
├── ref_move.npy, ref_direction.npy   the sorted tune predictions: the reference
├── ref_combo.npy                     distributions of the percentile scores
└── best_params.json                  the parameters of the best val run (used as a whole)
```

The models and the parameters must come from the same run of the notebook, because the parameters were tuned on these models' scores.

**Alignment with the notebooks.** Before publishing, the live code was checked against the notebooks with their own code (FINDINGS §41):
* the features are bit-identical to the dataset, including the incremental computation;
* the strategy produces the same trades as the research simulator on 600 random coins and parameter sets;
* the scores equal the notebook's predictions and percentiles;
* the fill rule equals the cleaning notebook's.

### The models and scores

Two XGBoost models score every minute; they are loaded once when a bot starts (from `global_settings.model_dir`, default `artifacts`). Predictions use `iteration_range=(0, best_iteration + 1)`, as in the research. The raw predictions become three **percentile scores**, each the share of the saved tune predictions at or below the value:

* **combo** = percentile of P(move) × P(direction): the buy gate;
* **direction** = percentile of P(direction);
* **move** = percentile of P(move).

The direction and move scores are shown on the dashboard and used only by the (disabled) signal exit.

### The trading strategy: A50, rebase mode

One position per coin, evaluated once per closed minute. All parameters come from `artifacts/best_params.json`.

**Entry.** Buy at the close (plus slippage) when all of these hold:

1. the coin is at least **720 minutes** (12 h) past its first candle;
2. `combo ≥ confidence_threshold`;
3. `price_position_15m ≤ max_price_position_15m` and `price_position_1h ≤ max_price_position_60m` (where the price sits within the last 15 / 60 minutes' range; this avoids buying on a local peak);
4. after a sale, the **re-entry rule**: the price must be at least `reentry_threshold_pct` below the last sale price. After `reentry_decay_start_minutes` the required drop shrinks by `reentry_decay_rate` per minute (it can become a premium), until the rule no longer applies.

The buy size is the bot's capital, capped so that the buy stays within the price-impact cap (`max_price_impact_pct`, default 0.333%, using the causal 1%-impact liquidity estimate from the features).

**Exits.**

* **Take-profit:** sell once the peak close since the segment started reaches `reference × (1 + activation_threshold)`. The reference is the buy price, or the close of the last rebase.
* **Loss floor:** after `loss_floor_consecutive_minutes` minutes in a row at or below `reference × (1 − loss_floor_pct)`, the position enters a *loss-floor race*. The first close back above the floor returns it to normal.
* **Stale time:** after `stale_trade_minutes` in the current segment, the position enters a *stale race*.
* **Races never sell.** During a race, the first candidate minute that passes the re-entry rule *rebases* the position: the reference becomes that minute's close, the peak resets, and the shares stay.

Within a minute the order is: rebase check → take-profit → (signal exit) → loss floor / stale transitions. `strategy.py` documents the details; the research simulator in `strategy.ipynb` is the authority on the logic.

**Signal exit: off.** The research also tried a *signal exit* (sell when the direction or move score drops below a threshold). Its gains came from a look-ahead in the simulator (FINDINGS §35); with the simulator fixed it adds nothing. It stays behind the `signal_exit_enabled` switch for research only.

**Slippage** of 1.5% per side in paper mode (`static_trading_fee_pct`), as in the simulations: on the high side on purpose, to keep paper results pessimistic.

**Parameters.** On startup, `database.py` loads `best_params.json` into the `strategy_config` table and stores a hash of the file. Replacing the file loads the new parameters on the next restart. Edits through `/api/config/strategy` are validated and last until the file changes.

### Features in the live bot

* `features.compute_live_features(all_data, last_n)` computes the newest `last_n` minutes from the coin's full history; timestamps are converted to UTC inside, and the last 30 minutes are padded internally (the features are causal, so the padding does not change them).
* The first **120 minutes** of a coin have no features.
* **Exact optimisation:** the notebook code recomputes the whole history on every call (~3 s at 30 days of history). The live copy computes only the requested rows and caches the finished blocks of the liquidity estimate. The result is bit-identical and takes ~0.1 s per minute. These are the only edits to the notebook code, marked `LIVE OPTIMISATION` in `features.py`.
* After a change to the feature code in `Dataset_creationV9`, copy it into `features.py` again.

### Data and timing

* **Full history.** A bot only trades a coin whose history it has from the first candle. With `historical_days = 30`, coins up to 30 days old can be initialised; older coins are refused with an explanation.
* **Minute grid and fill rule.** All history is kept on a gap-free one-row-per-minute grid. Every zero-volume minute, whether skipped by the API or returned as an empty interval, is a flat candle at the last real close: exactly the rule of the research data. After a restart the gap is fetched page by page, so long outages are filled completely.
* **When a minute is read.** The bot waits until second `grace_period_seconds` (paper mode, default 5) after each minute, fetches, and takes candles up to `now − finalization_lag_minutes`. GeckoTerminal may still revise a candle after its minute ends; the previous minute is re-checked on every fetch and corrected if it changed. The finalization study measures how late candles become final; its first reading suggests the read time needs to be much later than 5 s.
* **Clock.** The read time is measured with the host's clock, so keep it synchronised (see *Installation*).
* **Rate limit.** GeckoTerminal's limit is per IP. Each bot makes one OHLCV request per minute; initialising a 30-day coin takes ~45 requests, spaced 15 s apart.

### Trading modes

Chosen when the wallet is initialised (dashboard).

| Mode | Fills | Settings |
| --- | --- | --- |
| **Paper** | Buy at close × (1 + fee), sell at close × (1 − fee) | static fee (default 1.5%), grace period, finalization lag |
| **Shadow** | A real Jupiter quote at decision time (AMM fees, price impact, routing) plus the estimated network fee; nothing is signed or sent. If no quote can be had, the paper fill is used and flagged. | a Solana wallet **public key** (needed for the quotes) |
| **Live** | — | not implemented; the dashboard refuses it |

### Dashboard

Served by `app.py` at `http://<host>:8050`. Per bot:

* **Price chart** with buy and sell markers. While a position is open: the take-profit and loss-floor lines. After a sale: the re-entry line, which relaxes over time.
* **Combo score** under the price, with its buy threshold, and a ribbon that marks every **candidate minute**; hovering shows all three scores.
* **Buy readiness:** the entry conditions (combo, price position 15m / 1h, re-entry) and the take-profit, each against its threshold.
* **Bot card:** current scores, take-profit, loss floor, stale time, phase and rebases; P&L, win rate, holding times, trade counts and fees; the CPU and RAM of the bot's process.
* **Capital** over time (cash, open position, fees), trades, logs and a per-bot debug toggle.

Wallet-wide: total capital, SOL price, system stats (CPU, RAM, uptime) and all trades. Controls: pause, resume, deallocate, force sell immediately, force sell on the next sell signal, and stop all.

---

## Installation

### Notebooks

Written with **Python 3.9.20**. Packages: pandas, numpy, numba, xgboost (≥ 2.0, `.ubj` model files), optuna, cmaes, scikit-learn, scipy, pyarrow, matplotlib, requests.

```bash
pip install pandas numpy numba "xgboost>=2.0" optuna cmaes scikit-learn scipy pyarrow matplotlib requests
```

* The XGBoost settings use `device: 'cuda'` (an NVIDIA GPU). Without one, set it to `'cpu'` in `model_search` (`PARAMS_FIXED`); training is then much slower.
* Run the notebooks from `development_notebooks/` in the order above, each with **Run All**. Rough runtimes for ~400 coins: cleaning minutes, dataset tens of minutes, `model_search` many hours, `strategy` hours.
* `model_search` names the chosen pair at the end; copy the two `.ubj` files (and their `_params.json`) from `model_search/` to `model_results/` before running `strategy`.

### Live pipeline

Requires **Python 3.10+**. Built for a Linux host such as a Raspberry Pi (`app.py` reads `/proc/uptime` for the uptime display).

```bash
cd live_pipeline
python3 -m venv venv && source venv/bin/activate      # optional
pip install flask pandas numpy scipy xgboost numba requests psutil
```

> On a Raspberry Pi, `scipy`, `xgboost` or `numba` may lack a prebuilt wheel for your Python version. If `pip` falls back to a slow source build or fails, install them with the system package manager: `sudo apt install python3-scipy python3-xgboost python3-numba`.

Put the files of `strategy_results/live_artifacts/` into `live_pipeline/artifacts/`. The app creates `data/` (the database and per-bot folders) on its first start.

**Clock synchronisation.** The bots time their reads with the system clock. Raspberry Pi OS syncs over NTP by default (`systemd-timesyncd`); check with `timedatectl` that `System clock synchronized: yes`. The Pi has no battery-backed clock, so its time is wrong after a boot until the first sync, and a VPN with a kill switch can delay that sync. `chrony` (`sudo apt install chrony`) syncs faster and keeps a tighter clock. If the app runs as a systemd service, let it wait for the first sync (`After=time-sync.target` and `systemd-time-wait-sync.service`).

The application runs on an 8 GB Raspberry Pi 5 with Raspberry Pi OS Lite (64-bit), without a desktop environment.

## Usage

Start the dashboard server from `live_pipeline/` (this also initialises or migrates the database, loads the strategy parameters and starts the SOL-price collector):

```bash
python app.py
```

Then open `http://<host>:8050` and:

1. **Initialise the wallet:** a starting SOL amount and the mode (paper, or shadow with a wallet public key).
2. **Add a bot** for a Solana pool (symbol + pool address). The manager starts a `bot_loop.py` process for it, which first fetches the coin's history (a few minutes for an older coin) and then runs every minute. It trades once the coin is 12 hours old.
3. **Manage bots** from the dashboard: pause, resume, deallocate, force sell immediately, force sell on the next sell signal, or stop all.

You never run `bot_loop.py` by hand; `bot_manager.py` launches it.

**New models or parameters:** run the export cell of `strategy.ipynb`, replace the files in `live_pipeline/artifacts/` and restart the app. Running bots keep their parameters until they are restarted.

**Coin selection:** the results describe coins that passed the screen above (liquidity, security score, age). Apply the same screen when choosing coins for the bots, or the results may not transfer.
