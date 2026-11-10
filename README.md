# Solana Meme-Coin Trading Bot

An end-to-end machine-learning trading system for Solana meme coins. It collects 1-minute OHLCV candles from GeckoTerminal, builds a 169-feature dataset, trains two gradient-boosted models on 146 of those features each (a **move** model and a **direction** model), tunes a trading strategy on them in a causality-tested simulator, and runs that strategy live: many independent bots behind a web dashboard, in **paper** mode (simulated fills) or **shadow** mode (real Jupiter swap quotes, never sent).

The repository has three parts:

* **Development notebooks:** the five-step pipeline from raw candles to a tuned strategy, run in order.
* **Research notebook:** a measurement study for the live pipeline (when is a 1-minute candle final?).
* **Live pipeline:** the multi-bot application (Python back end and a single-page dashboard) that runs on a Raspberry Pi.

[`development_log.md`](development_log.md) is the project's running log: every experiment, its result and the decision taken, from the first label search to the current pipeline. Its §0 is the project guide.

> **Headline (current notebook output, first 30 of 50 strategy runs):** from $20 per coin, the run that was best on validation ends at **$43.76 per coin on 100 unseen test coins** (2.2×). The mean over all runs is $47.74 on test. These are simulated results with 1.5% slippage per side; no real money has been traded.

---

## Repository layout

```
├── README.md
├── development_log.md
├── .gitignore
├── development_notebooks/         the pipeline, run in this order
│   ├── meme_coin_pipeline.ipynb
│   ├── data_cleaning.ipynb
│   ├── Dataset_creationV9.ipynb
│   ├── model_search.ipynb
│   └── strategy.ipynb
├── research_notebooks/            studies outside the pipeline
│   ├── readme_visuals/            the README's images
│   ├── finalization_study.ipynb
│   └── profitability_visuals.ipynb
├── live_pipeline/                 the trading application
│   ├── app.py
│   ├── bot_manager.py
│   ├── bot_loop.py
│   ├── data_collection.py
│   ├── features.py
│   ├── scoring.py
│   ├── strategy.py
│   ├── execution.py
│   ├── dex.py
│   ├── database.py
│   ├── templates/                 the dashboard page
│   │   └── dashboard.html
│   └── artifacts/                 the chosen models and strategy files
├── data/                          the data (not committed)
│   ├── raw_data/
│   ├── processed_data/
│   └── dataset_v9/
└── research_development/          the notebooks' outputs (not committed)
    ├── model_search/
    ├── strategy_results/
    ├── finalization_data/
    └── readme_visuals/
```

---

## Terminology

**Solana (SOL):** a fast, low-fee blockchain that hosts most meme-coin trading. All money in the live system is tracked in SOL; the dashboard converts it to USD with a live SOL price. The simulations count in USD, with $20 of starting capital per coin.

**Meme coin:** a highly speculative token whose price is driven by attention and short-term trading rather than fundamentals. Extremely volatile and often short-lived.

**Candle / OHLCV:** one minute of trading: open, high, low, close and volume. Every minute without volume is a flat candle at the last real close, so every minute has a row and *row number = minutes since the coin's first candle*.

**Move:** a rise of at least 10% within the next 30 minutes. The *move model* estimates its probability.

**Direction:** whether the price first reaches +15% or −15% within the next 2 hours. The *direction model* estimates the probability that +15% comes first.

**Score / percentile:** in the research, a model output converted to its rank within the model's predictions on the *tune* split (0 = lowest, 1 = highest). All strategy thresholds are on these percentiles, so they mean the same for any model. The live bot uses the equivalent raw cutoff (see *The models and scores*), so it needs no tune predictions. In research and live.

**Candidate minute:** a minute that passes the entry gate (combo score and price-position filters). Buys and rebases only happen on candidate minutes.

**Rebase / race:** the loss floor and the stale time never sell. They start a *race*: the position stays open and is re-anchored (rebased) at the next acceptable candidate minute, unless the take-profit comes first.

**Paper / shadow / live:** the same decisions with different fills. *Paper* simulates fills with a fixed fee. *Shadow* asks Jupiter for a real swap quote at decision time and records that fill, without signing or sending anything. *Live* (real swaps) is not implemented.

---

## Development notebooks

The pipeline that produced the dataset, the models and the strategy parameters. Every notebook is resumable (finished work is skipped).

| # | Notebook | Reads | Writes | What it does |
|---|---|---|---|---|
| 1 | [`meme_coin_pipeline`](development_notebooks/meme_coin_pipeline.ipynb) | GeckoTerminal API, `COIN_LIST` | `data/raw_data/` | Collects 1-minute OHLCV per pool, applies the fill rule, stops collecting a coin at its death point (48 h window < 1,000 SOL volume, after a 48 h grace) and marks it dead. |
| 2 | [`data_cleaning`](development_notebooks/data_cleaning.ipynb) | `data/raw_data/` (never modified) | `data/processed_data/` | The fill rule again, a volume death mark and a pump-rate (label) death mark, cut at the earlier one; leaves out coins with less than 12 h of data. |
| 3 | [`Dataset_creationV9`](development_notebooks/Dataset_creationV9.ipynb) | `data/processed_data/` | `data/dataset_v9/` | 169 causal features and the labels for every minute, a **leakage test** on all features, and a stratified coin-level split into 5 splits. |
| 4 | [`model_search`](development_notebooks/model_search.ipynb) | `data/dataset_v9/splits`, `data/processed_data/` | `research_development/model_search/` | Optuna searches for the move and the direction model (validation metric minus a model-size penalty); the top 10 of each are then judged in strategy simulations, and the pair is picked there. |
| 5 | [`strategy`](development_notebooks/strategy.ipynb) | `live_pipeline/artifacts/` (the chosen pair), `data/dataset_v9/splits`, `data/processed_data/` | `research_development/strategy_results/` | `N_RUNS` strategy optimisations (optimised on tune, selected on val_tune), each evaluated **once** on test; the run with the best `SELECT_BY` score (test) is exported as a named strategy file. |

**Folders.** Every notebook finds the repository's top folder by itself, so it runs from wherever Jupyter starts. It reads and writes its data in `data/` (`raw_data/`, `processed_data/`, `dataset_v9/`) and its other outputs in `research_development/`, in the output folders it always used (`model_search/`, `strategy_results/`, `finalization_data/`, `readme_visuals/`); neither folder is committed. The notebooks that use the chosen models (`strategy`, `profitability_visuals`) read them from `live_pipeline/artifacts/`, together with the strategy files; what goes into that folder is copied there by hand (see *From the notebooks to the bot*).

**Data.** 402 coins, collected by hand-picking pools that met a fixed screen when added: Solana, at least $20k liquidity, GeckoTerminal security score ≥ 50, between 12 hours and 3 months old. Prices and volume are in SOL. development log §0.4 documents the data's provenance and every alteration.

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
* **Pump dynamics:** two pump definitions (one per model: +10% within 30 minutes for the move model, +15% within 2 hours for the direction model), 23 features each: clustering, velocity, density, failure rate and time between pumps.
* **Liquidity:** a causal estimate of the SOL needed for a 1% price impact (`estimated_sol_1pct_impact`), its age and uncertainty, liquidity trends.

Each model uses **146** of the 169 features: the 123 that both share, plus its own pump set of 23. The other model's pump set is left out.

**Labels.** `label_10pct_30m` (move model), `label_fpclass_15pct_2h` (direction model: up first / down first / neither) and `label_meta_minutes_to_end` (training rows need ≥ 240 minutes of future; never a feature).

**The models.**
* The **move** model predicts P(close rises ≥ 10% within 30 minutes), from 146 features (the shared ones and the +10% / 30 min pump set).
* The **direction** model predicts P(+15% is reached before −15% within 2 hours), from 146 features (the shared ones and the +15% / 2 h pump set). It is trained **only on candidate minutes**: tradable, move score in its top 50%, and an outcome within 2 hours. This "A50" recipe was the single biggest model-side improvement found (development log §21–22).
* The buy score (**combo**) is the tune percentile of P(move) × P(direction).

### Evaluation rules

The rules every experiment followed (development log §0.7):
* a **leakage test** for features: values must be identical with and without future data;
* a **causality test** for every simulator change: a coin simulated on its full data and on data cut at minute C must give identical trades before C;
* **controls** (shuffled scores, random blocks) for any rule that skips or changes trades;
* a decision rule **fixed before** looking at results, and confirmation with **new model seeds**;
* the test split read **once**.

Two pitfalls shaped these rules. First, a simulator rule applied after the fact leaked the future and produced a fake doubling of the score (§35). Second, model-level AUC does not predict strategy results; only the simulations do (§20).

---

## Research notebooks

| Notebook | Purpose |
|---|---|
| [`profitability_visuals`](research_notebooks/profitability_visuals.ipynb) | Simulates a strategy file from `live_pipeline/artifacts/` on the test coins with the live pipeline's decision code and draws the README's charts: final capital per coin, and the capital of one coin over time. Writes to `research_development/readme_visuals/`; copy the charts for the README into `research_notebooks/readme_visuals/`, which is committed. |
| [`finalization_study`](research_notebooks/finalization_study.ipynb) | **When is a live candle final?** The dataset's candles were collected long after the fact, but the live bot reads a minute seconds after it ends. The study reads the just-ended candle at 1.0, 1.5, … 40 s after its close (a sweep of 79 offsets within GeckoTerminal's rate limit) and compares each read with the value 5 minutes later. It checks the computer's clock against the server's `Date` header on every call, survives restarts, and logs every rate-limit hit. A standalone statistics cell reports, per field and for the whole candle, the earliest second at which 95 / 99 / 99.9% of candles are final. Writes to `research_development/finalization_data/`. |

**First reading** (one coin, about one day; development log §40.8): candles become final much later than expected. At 1–3 s almost no candle is final; at 38–40 s about half are. More coins and a longer sweep are needed before the live read time is set from it.

---

## Results

All from the saved notebook outputs on the rebuilt data (development log §42).

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

1. **History.** On its first start it fetches the coin's full minute history from GeckoTerminal, paging backwards 1000 candles per request (15 s apart) until the API has nothing older, and applies the fill rule. The history must start at the coin's first candle, because several features (minutes since launch, the launch and all-time-high features, the liquidity estimate) are anchored to it, so a page that fails after its retries stops the bot with an error instead of running on a partial history.
2. **Every minute** it fetches the newest candles, re-checks the previous minute for late revisions, and appends the new minutes to the history with the same fill rule.
3. **Features.** It computes the 169 features for the new minute(s) with the dataset notebook's own code (`features.py`); each model reads its own 146 of them.
4. **Scores.** The move and direction models predict; their product is the *combo* score (`scoring.py`).
5. **Strategy.** The A50 state machine (`strategy.py`) decides: buy, sell, rebase or nothing.
6. **Execution.** Paper or shadow fills (`execution.py`, `dex.py`).
7. **Records.** Candles, scores, trades, capital and the strategy state go to the SQLite database; the dashboard reads from there.

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

A bot's strategy state (open position, phase, races, peak, re-entry bookkeeping) is stored in the `bot_trading_state` table and written whenever it changes. A stopped or crashed bot resumes from that state: it reloads its full stored history, fetches every minute it missed (page by page, however long it was down) and fills them in, so the features stay exact. The missed minutes are not scored or traded; trading continues with the next new minute.

The dashboard follows the database through a live update stream: `app.py` checks a set of change markers once a second and pushes them over `/api/stream` (Server-Sent Events) only when something changed.

### Files

| File | Responsibility |
| --- | --- |
| `app.py` | Flask server: the dashboard, its API and the live update stream (`/api/stream`), plus the SOL-price background thread |
| `bot_manager.py` | Bot process lifecycle (initialise / start / pause / resume / stop / deallocate) and the dashboard's bot data |
| `bot_loop.py` | Per-bot worker process: the only module that wires everything together |
| `data_collection.py` | GeckoTerminal OHLCV fetching (full history, the gap after a restart, live minutes), rate-limit handling, the fill rule |
| `features.py` | The feature code of `Dataset_creationV9`, copied verbatim without the labels, plus an incremental version of the liquidity estimate and the per-bot `LiveFeatureEngine` |
| `scoring.py` | Loads the two models and computes P(move), P(direction) and the combo score |
| `strategy.py` | The A50 strategy as a minute-by-minute state machine, the strategy files (parameters) and the fill prices |
| `execution.py` | Buy and sell execution: paper and shadow (live stubbed) |
| `dex.py` | Jupiter swap quotes and network-fee estimates for shadow mode (never signs or sends) |
| `database.py` | SQLite schema, queries, all persistence, and the change markers behind the live update stream |
| `templates/dashboard.html` | Single-page frontend served by Flask (Plotly charts) |

### From the notebooks to the bot

`live_pipeline/artifacts/` holds what the live pipeline and the notebooks after `model_search` use. It is filled by hand:

```
artifacts/
├── move.ubj, move.json              the chosen move model and its _params.json
├── direction.ubj, direction.json    the chosen direction model and its _params.json
└── strategy_<name>.json             one file per strategy
```

1. `model_search` names the chosen pair at the end. Copy the two models from `research_development/model_search/` into `artifacts/` as `move.ubj` + `move.json` and `direction.ubj` + `direction.json` (the `.json` files are the models' `_params.json`).
2. `strategy` loads the models from `artifacts/`. Its last cell writes the chosen run as `strategy_<STRATEGY_NAME>.json` into `research_development/strategy_results/`: the parameters, the raw `combo_cutoff`, the simulation settings and fingerprints (sha256) of the two models. Copy that file into `artifacts/`.

A strategy is only valid with the models it was tuned on, because `combo_cutoff` is a score of those models. The live pipeline and `profitability_visuals` check the fingerprints and refuse a strategy file whose models don't match; several strategies can sit side by side in `artifacts/` as long as they were tuned on the same two models.

**Alignment with the notebooks.** The live code was checked against the notebooks' own code:
* **Features:** all 169 values are identical to `compute_coin_features_v9` on a synthetic 12,000-minute coin with gaps and zero-volume minutes, both on the full history and minute by minute on the history up to each minute, with the bot's Europe/Amsterdam timestamps across a daylight-saving change. The incremental liquidity estimate gives the same values as the notebook's function, also after a revised minute.
* **Strategy:** the bot's decision logic, replayed minute by minute on 30 random synthetic coins, makes the same trades and ends with the same capital as the notebook's simulator (with the simulator's 12-hour trading start set to 0, because the live bot has none; see *Entry*).
* **Fill rule:** identical to the cleaning notebook's, also when new minutes are appended to a stored history.
* **Scores:** the models are loaded with their own feature lists and predict with `iteration_range=(0, best_iteration + 1)`, as in the research. The live scores have not yet been compared with the notebook's saved predictions from the real models.

### The models and scores

Two XGBoost models score every minute; they are loaded once when a bot starts (from `global_settings.model_dir`, default `artifacts`). Predictions use `iteration_range=(0, best_iteration + 1)`, as in the research. The buy score is the **combo**, P(move) × P(direction).

In the research the buy threshold is a percentile: `confidence_threshold = 0.8619` means "a combo higher than 86.19% of the tune minutes". A percentile only grows with the raw value, so the same gate is exactly "combo ≥ `combo_cutoff`", where `combo_cutoff` is the tune combo at that rank. The export cell computes it once (and checks that it selects the same tune minutes), so the live bot needs only the two models. The dashboard shows the raw combo, P(direction) and P(move).

### The trading strategy: A50, rebase mode

One position per coin, evaluated once per closed minute. The tuned parameters come from the bot's strategy file in `live_pipeline/artifacts/`.

**Entry.** Buy at the close (plus slippage) when all of these hold:

1. `combo ≥ combo_cutoff` (the raw form of the tuned `confidence_threshold` percentile);
2. `price_position_15m ≤ max_price_position_15m` and `price_position_1h ≤ max_price_position_60m` (where the price sits within the last 15 / 60 minutes' range; this avoids buying on a local peak);
3. after a sale, the **re-entry rule**: the price must be at least `reentry_threshold_pct` below the last sale price. After `reentry_decay_start_minutes` the required drop shrinks by `reentry_decay_rate` per minute (it can become a premium), until the rule no longer applies.

The buy size is the bot's capital, capped so that the buy stays within the price-impact cap (`max_price_impact_pct`, default 0.333%): at most `estimated_sol_1pct_impact × max_price_impact_pct × 100` SOL, using the causal 1%-impact liquidity estimate from the features. While no estimate exists yet, the whole capital is used, as in the simulator.

**No 12-hour start.** The research simulator never buys in a coin's first 720 minutes (12 h); the live bot can buy from the first minute with features (minute 120). The tuned parameters and the results above come from simulations with the 12-hour start, so early trades on young coins are outside what was tested.

**Exits.**

* **Take-profit:** sell once the peak close since the segment started reaches `reference × (1 + activation_threshold)`. The reference is the buy price, or the close of the last rebase.
* **Loss floor:** after `loss_floor_consecutive_minutes` minutes in a row at or below `reference × (1 − loss_floor_pct)`, the position enters a *loss-floor race*. The first close back above the floor returns it to normal.
* **Stale time:** after `stale_trade_minutes` in the current segment, the position enters a *stale race*.
* **Races never sell.** During a race, the first candidate minute that passes the re-entry rule *rebases* the position: the reference becomes that minute's close, the peak resets, and the shares stay.

Within a minute the order is: rebase check → take-profit → loss floor / stale transitions. `strategy.py` documents the details; the research simulator in `strategy.ipynb` is the authority on the logic.

**No signal exit.** The research also tried a *signal exit* (sell when the direction or move score drops below a threshold). Its gains came from a look-ahead in the simulator (development log §35); with the simulator fixed it added nothing, so the live bot has none.

**Slippage** of 1.5% per side in paper mode (`static_trading_fee_pct`), as in the simulations: on the high side on purpose, to keep paper results pessimistic.

**Parameters.** Each strategy file in `live_pipeline/artifacts/` holds one strategy's parameters. When a bot is set up, a dropdown picks its strategy (default: the first file by name); the bot keeps it until it is deallocated. Strategy files can't be edited from the dashboard. The table shows the values of `strategy_max_capital.json`:

| Parameter | Value | Search range |
| --- | --- | --- |
| `confidence_threshold` (tune percentile) | 0.861851 | 0 – 1 |
| `combo_cutoff` (its raw combo) | 0.3475198447704315 | — |
| `max_price_position_15m` | 0.435592 | 0.05 – 0.95 |
| `max_price_position_60m` | 0.354948 | 0.05 – 0.95 |
| `activation_threshold` (take-profit) | 0.236134 | 0.1 – 0.35 |
| `stale_trade_minutes` | 841 | 60 – 1440 |
| `reentry_threshold_pct` | 0.191757 | 0.005 – 0.2 |
| `reentry_decay_rate` | 0.00924998 | 0.0005 – 0.01 |
| `reentry_decay_start_minutes` | 74 | 20 – 180 |
| `loss_floor_pct` | 0.260773 | 0.15 – 0.4 |
| `loss_floor_consecutive_minutes` | 14 | 2 – 15 |

### Features in the live bot

* Each bot keeps a `LiveFeatureEngine`. At startup it builds the liquidity estimate over the coin's full history; after that, `sync` processes each new minute and `revise` re-processes a minute that GeckoTerminal revised.
* `engine.features(all_data, rows)` calls `compute_live_features(df, rows, liquidity)`, the notebook's per-minute feature code for the requested rows only, without the labels and without the notebook's 30-minute cut at the end of a coin.
* The hour-of-day features use **UTC** hours, as the dataset does (its loader parses timestamps as UTC), whatever timezone the bot keeps its data in.
* The first **120 minutes** of a coin have no features, as in the dataset.
* **Cost.** The rolling statistics are recomputed over the full history on every call, because their values depend on the history from the coin's first row; this is what keeps them identical to the dataset. In testing (not on the Pi) a new minute took about 45 ms for a 12,000-minute coin. The liquidity estimate is the only incremental part: the notebook version recomputes the coin's whole history; `CausalLiquidity` keeps its state between minutes and gives the same values.
* After a change to the feature code in `Dataset_creationV9`, copy it into `features.py` again.

### Data and timing

* **Full history.** A bot fetches as much history as GeckoTerminal has, from the coin's first candle. GeckoTerminal's minute data reaches back roughly six months, so for an older coin the history would start late and the launch-anchored features would differ from the dataset's; the bots are meant for young coins.
* **Minute grid and fill rule.** All history is kept on a gap-free one-row-per-minute grid. Every zero-volume minute, whether skipped by the API or returned as an empty interval, is a flat candle at the last real close: exactly the rule of the research data. After a restart the gap is fetched page by page, so long outages are filled completely; if the gap can't be fetched, the bot stops with an error rather than filling it with flat candles.
* **When a minute is read.** The bot waits until second `grace_period_seconds` (paper mode, default 5) after each minute, fetches, and takes candles up to `now − finalization_lag_minutes`. GeckoTerminal may still revise a candle after its minute ends; the previous minute is re-checked on every fetch and, if it changed, corrected and re-processed by the feature engine. The finalization study measures how late candles become final; its first reading suggests the read time needs to be much later than 5 s.
* **Clock.** The read time is measured with the host's clock, so keep it synchronised (see *Installation*).
* **Rate limit.** GeckoTerminal's limit is per IP. Each bot makes one OHLCV request per minute; initialising a coin takes one request per 1000 minutes of history, spaced 15 s apart (~11 minutes for a 30-day coin).

### Trading modes

Chosen when the wallet is initialised (dashboard).

| Mode | Fills | Settings |
| --- | --- | --- |
| **Paper** | Buy at close × (1 + fee), sell at close × (1 − fee) | static fee (default 1.5%), grace period, finalization lag |
| **Shadow** | A real Jupiter quote at decision time (AMM fees, price impact, routing) plus the estimated network fee; nothing is signed or sent. If no quote can be had, the paper fill is used and flagged. | a Solana wallet **public key** (needed for the quotes) |
| **Live** | — | not implemented; the dashboard refuses it |

### Dashboard

Served by `app.py` at `http://<host>:8050`. A **System Overview** row shows the Pi's uptime, CPU and RAM, the SOL price, the number of active bots with **Pause all** / **Resume all** buttons (each acts only on the bots that aren't paused or running yet), and the wallet mode. Below it are three tabs:

* **Overview:** a coloured status card per bot (status, coin, initial and current capital with P&L, position, completed trades, assigned time and the bot's controls), the **portfolio capital composition** chart (unallocated, initial capital, realized profit, open positions, and cumulative fees in their own pane), all bots' trade history and the pipeline terminal.
* **Manage Bots:** per bot,
  * a **price chart** in three panes: the model scores on top (P(move) red, P(direction) blue, and the combo orange and thicker, as the main score); the candles with buy and sell markers, the take-profit and loss-floor lines while a position is open, and the re-entry line after a sale, which relaxes over time; and the volume. Hovering shows all values of a minute;
  * **buy readiness:** combo, price position 15m / 60m, re-entry and take-profit, each against its threshold;
  * the **bot info card** (strategy, position, phase and rebases, capital, P&L, win rate, holding times, trade counts, fees, assigned time, the CPU and RAM of the bot's process);
  * **capital composition & fees** over time, the bot's trades and its terminal, and a debug-logging toggle.

  An unassigned bot shows the setup form instead: token address (symbol and image fetched from GeckoTerminal), pool address, allocation (percentage of the wallet or a SOL amount), the strategy (a dropdown of the strategy files in `artifacts/`) and the price-impact cap.
* **Analytics:** the one-time wallet setup (paper or shadow), the wallet status, the SOL distribution and a capital overview per bot.

Controls per bot: pause, resume, deallocate, sell now, and sell on the next sell signal (both pause the bot after the sale).

**Live updates.** The page refreshes when the live update stream reports a change, plus every 30 seconds for the elapsed-time fields; if the stream is down it polls every 5 seconds. Text and panels are only rewritten when their content changed, and charts are only redrawn when they have new data. The charts use Plotly, which the browser loads from its CDN.

---

## Installation

### Notebooks

Written with **Python 3.9.20**. Packages: pandas, numpy, numba, xgboost (≥ 2.0, `.ubj` model files), optuna, cmaes, scikit-learn, scipy, pyarrow, matplotlib, requests.

```bash
pip install pandas numpy numba "xgboost>=2.0" optuna cmaes scikit-learn scipy pyarrow matplotlib requests
```

* The XGBoost settings use `device: 'cuda'` (an NVIDIA GPU). Without one, set it to `'cpu'` in `model_search` (`PARAMS_FIXED`); training is then much slower.
* Run the notebooks from `development_notebooks/` in the order above, each with **Run All**. Rough runtimes for ~400 coins: cleaning minutes, dataset tens of minutes, `model_search` many hours, `strategy` hours.
* Put the data in `data/` (`raw_data/`, or `processed_data/` and `dataset_v9/` if you skip steps); every other output appears in `research_development/`.
* After `model_search` and after `strategy`, copy the chosen models and the strategy file into `live_pipeline/artifacts/` (see *From the notebooks to the bot*).

### Live pipeline

Requires **Python 3.10+**. Built for a Linux host such as a Raspberry Pi (`app.py` reads `/proc/uptime` for the uptime display).

```bash
cd live_pipeline
python3 -m venv venv && source venv/bin/activate      # optional
pip install flask pandas numpy scipy xgboost requests psutil
```

> On a Raspberry Pi, `scipy` or `xgboost` may lack a prebuilt wheel for your Python version. If `pip` falls back to a slow source build or fails, install them with the system package manager: `sudo apt install python3-scipy python3-xgboost`.

The app uses the models and strategy files in `live_pipeline/artifacts/`. It creates `live_pipeline/data/` (the database and per-bot folders, not committed) on its first start. There are no database migrations: after a change to the schema, delete `data/trading_bots.db` and set the wallet and bots up again.

**Clock synchronisation.** The bots time their reads with the system clock. Raspberry Pi OS syncs over NTP by default (`systemd-timesyncd`); check with `timedatectl` that `System clock synchronized: yes`. The Pi has no battery-backed clock, so its time is wrong after a boot until the first sync, and a VPN with a kill switch can delay that sync. `chrony` (`sudo apt install chrony`) syncs faster and keeps a tighter clock. If the app runs as a systemd service, let it wait for the first sync (`After=time-sync.target` and `systemd-time-wait-sync.service`).

The application runs on an 8 GB Raspberry Pi 5 with Raspberry Pi OS Lite (64-bit), without a desktop environment.

## Usage

Start the dashboard server from `live_pipeline/` (this also creates the database if it doesn't exist and starts the SOL-price collector):

```bash
python app.py
```

Then open `http://<host>:8050` and:

1. **Initialise the wallet:** a starting SOL amount and the mode (paper, or shadow with a wallet public key).
2. **Set up a bot** for a Solana pool on the Manage Bots tab (token address, pool address, allocation, strategy), then start it. The manager starts a `bot_loop.py` process for it, which first fetches the coin's full history (about 15 s per 1000 minutes) and then runs every minute. It can trade from the coin's first minute with features (minute 120).
3. **Manage bots** from the dashboard: pause, resume, deallocate, sell now, or sell on the next sell signal; or pause or resume all bots at once.

You never run `bot_loop.py` by hand; `bot_manager.py` launches it.

**New models or parameters:** copy the new files into `live_pipeline/artifacts/` (see *From the notebooks to the bot*). New strategy files appear in the setup dropdown at once; running bots keep their strategy until they are restarted. Strategy files tuned on other models than the ones in `artifacts/` show as unusable.

**Coin selection:** the results describe coins that passed the screen above (liquidity, security score, age). Apply the same screen when choosing coins for the bots, or the results may not transfer.
