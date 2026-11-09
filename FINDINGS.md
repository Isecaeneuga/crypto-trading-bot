# Pump-prediction trading system: findings log

Running record of what has been tested, what was found, and what was decided.
Newest findings are added at the bottom of each section. All money figures are per-coin capital with $20 starting capital per coin, unless stated otherwise.

*Last updated: 2026-11-17 (§0 project guide: presentable pipeline, raw-data provenance, file formats; §40 candle finalization study); repository clean-up: §0.12 live artifacts, §40.8 first reading, §41 live-pipeline alignment check, §42 rebuild results*

---

## 0. Project guide — START HERE

This file is the project's memory. §0 describes the project **as it is now**: its notebooks, data, file formats and rules. §1–38 are the **research log** in the order it happened. §39 lists known issues. §40 documents the live pipeline's **candle finalization study** (`finalization_study.ipynb`). §41 checks the live pipeline against the notebooks; §42 holds the results of the rebuilt pipeline. In the repository, the five pipeline notebooks are in `development_notebooks/`, the finalization study in `research_notebooks/` and the live application in `live_pipeline/`. Many notebooks named there are older research notebooks that are not part of the presentable project; their results are summarised in the log.

### 0.1 What the system does
* It predicts short-term pumps of Solana meme coins from 1-minute OHLCV data and trades them in a simulator.
* **Two XGBoost models:**
  * the **move model** (label `label_10pct_30m`: the close rises ≥ 10% within 30 min);
  * the **direction model** (label `label_fpclass_15pct_2h`: +15% is reached before −15% within 2 h). It is trained **only on candidate minutes**: tradable, move score in its top 50%, outcome decided. This is the "A50" recipe (§21–22).
* **Buy score:** the percentile, within the tune distribution, of P(move) × P(direction).
* **Strategy:**
  * buy when the score ≥ the confidence and both price positions are below their caps;
  * sell at the take-profit;
  * the loss floor and the stale time only re-anchor the reference price ("rebase" exit mode) and **never sell**;
  * a re-entry rule applies after each sale.
* **Score:** the mean over coins of the final capital, starting at $20 per coin and capped at 10×.
* **Honest level so far** (old split, models from the research notebooks): about **$44 per coin on unseen coins** (hold-out, §35.2). Val scores are about $51 and include a selection bonus. The presentable rebuild re-measures this on the test split.

### 0.2 The presentable pipeline (run in this order)

| # | Notebook | Reads | Writes | What it does |
|---|---|---|---|---|
| 1 | `meme_coin_pipeline.ipynb` | GeckoTerminal API, `COIN_LIST` | `raw_data/` | collects 1-minute OHLCV per pool; fills minutes; stops collecting a coin at its death point and marks it dead; only updates coins in `COIN_LIST` that are not marked dead |
| 2 | `data_cleaning.ipynb` | `raw_data/` (never modified) | `processed_data/` | fill rule; volume and label death marks; leaves out coins with < 12 h (§0.5) |
| 3 | `Dataset_creationV9.ipynb` | `processed_data/` | `dataset_v9/` | 169 features + labels per minute; 5 coin-level splits; leakage test |
| 4 | `model_search.ipynb` | `dataset_v9/splits`, `processed_data/` | `model_search/` | Optuna for both models (validation metric − α × size); the top N in strategy sims; picks the pair |
| 5 | `strategy.ipynb` (`strategy_tighten.ipynb`: the tightening re-entry rule, tested worse; kept as a record) | `model_results/` (the chosen pair, copied by hand), `dataset_v9/splits`, `processed_data/` | `strategy_results/` (incl. `live_artifacts/`) | N strategy optimisations (tune → val-selected); each run's parameters evaluated **once** on test; the last cell exports the live artifacts (§0.12) |

**One-time helpers.** Both have already been run on the author's data; they're only needed for data from before 2026-11:

| Notebook | What it did |
|---|---|
| `fix_pipeline_output_fill.ipynb` | rewrote the zero-volume minutes in the old `pipeline_output/` folder (§0.4) |
| `fix_raw_data_once.ipynb` | orphan metadata; zero-volume minutes; marking archived coins dead, in `raw_data/` (§0.4) |

**Live trading** is a separate code base in `live_pipeline/`: `app.py`, `bot_loop.py`, `features.py`, `scoring.py`, `strategy.py`, `execution.py`, `dex.py`, and others.
* `features.py` contains a **verbatim copy** of the dataset notebook's feature code plus a live section. It must stay identical apart from the lines marked `LIVE OPTIMISATION` (checked in §41).
* The live gap filling (`data_collection.py`) follows the same fill rule (§0.5).
* The live bot reads its models, reference distributions and parameters from `live_pipeline/artifacts/`, written by the last cell of `strategy.ipynb` (§0.12).
* `finalization_study.ipynb` (`research_notebooks/`) measures how many seconds after a minute ends its candle stops changing on GeckoTerminal; its result sets the live bot's read time (§40).

### 0.3 Folders
The notebooks create these data folders next to themselves (in `development_notebooks/`); they are not part of the repository.


| Folder | Content |
|---|---|
| `raw_data/` | what the pipeline collected (see §0.4 for how it was altered) |
| `processed_data/` | cleaned series, rebuilt from scratch on every cleaning run |
| `dataset_v9/` | `per_coin/` (parquet per coin), `splits/` (`train`, `val_train`, `tune`, `val_tune`, `test`, `coin_assignments.csv`), `reports/` |
| `model_search/` | every trial's model (`.ubj`) + hyperparameters (`_params.json`), study logs, caches, sim runs |
| `model_results/` | the chosen move + direction model (+ their `_params.json`), copied by hand |
| `strategy_results/` | the strategy runs (incl. each run's test result), caches; `live_artifacts/` for the live pipeline (§0.12) |

### 0.4 Raw data: provenance and every alteration
`raw_data/` is **not** a pristine API dump. In order:
1. **Coin universe.** There was no automated rule; coins were picked by hand, but every coin met these criteria when it was **added** to `COIN_LIST`:
   * Solana network;
   * at least **$20k liquidity**;
   * GeckoTerminal **security score ≥ 50**;
   * at least **12 hours old**;
   * at most **3 months old**. Coins kept being updated afterwards, so some later passed 3 months.

   Results therefore describe coins that passed this screen. A live bot should apply the same screen when choosing coins, or results may not transfer.
2. **Collection.**
   * The pipeline fetches GeckoTerminal 1-minute candles with `include_empty_intervals` (prices and volume in SOL). The API still sometimes skips minutes; the pipeline adds them.
   * The API only reaches **~6 months** back, so data older than that can't be re-fetched.
   * The pipeline stops collecting a coin at its death point (48 h window < 1,000 SOL volume, windows starting after a 48 h grace) and **crops the file there**.
3. **Older archive coins** (collected before 2026-11, previously in a `data/` folder) were cropped **in place** by the old death-point notebook (`data_death_point_detection`). Its rules: a volume window scaled from minute 1 without a grace period, the label mark, and deletion of coins with < 24 h. **Data cut away then is gone.** Some coins also have no metadata from that time (see 5).
4. **Gap filling, old style.** Until 2026-11, a minute the API skipped was filled by **copying the previous candle's open / high / low / close** with volume 0, and the `backfilled` flag was dropped on every update. Fixed in 6–7.
5. **`fix_raw_data_once.ipynb`** (once, 2026-11, in place, on `raw_data/`):
   * (a) CSVs without metadata got a metadata file flagged `metadata_incomplete` (only symbol and pool prefix known);
   * (b) every zero-volume minute became a flat candle at the last real close (the fill rule, §0.5), with `backfilled` set from the volume;
   * (c) every metadata file got `death_point.is_dead = true` (reason in `marked_dead_reason`), so the pipeline never updates these coins again. Updating would leave a gap, because of the 6-month API limit. Coins still being collected were excluded.
6. **`fix_pipeline_output_fill.ipynb`** (once, 2026-11, in place, on the old `pipeline_output/` before it was copied into `raw_data/`): the same zero-volume rewrite. Typically only a few minutes per coin changed: most zero-volume minutes are API empty intervals that were already flat.
7. **Since then** the pipeline applies the fill rule itself on every run, and `data_cleaning` applies it again, so the processed data is correct regardless of a raw file's history.

### 0.5 Data rules (as implemented)
* **Fill rule (everywhere: pipeline, cleaning, live bot):**
  * the series is a continuous 1-minute grid;
  * **every zero-volume minute is a flat candle at the last real close** (open = high = low = close), volume 0;
  * `backfilled` = 1 for zero-volume minutes, derived from the volume.
  * A real candle always has volume > 0.
* **Cleaning (`data_cleaning`):**
  * **volume mark:** the pipeline's rule, cut at the window **end** (knowable live);
  * **label mark:** a 7-day window of `label_10pct_30m` averaging < 1% → cut at the window **start**. This is determined in hindsight: it removes up to 7 days before a coin's death that a live bot would still face;
  * cut at whichever mark keeps fewer rows;
  * coins with < **12 h** left are left out (trading starts at 12 h, so they could never be traded).
* **Timestamps:** stored with the Europe/Amsterdam offset; the features use **UTC** hours.
* **Prices and volume:** in SOL (GeckoTerminal `currency=token`); prices are SOL per token.

### 0.6 File formats
**OHLCV CSV:**
* `{symbol}_{first 5 chars of the pool address}_ohlcv_cleaned.csv`; some archive files are `…_ohlcv.csv`.
* Columns: `timestamp, open, high, low, close, volume, backfilled`.
* In the dataset, a coin's `coin_id` is the file stem without `_ohlcv`, i.e. `{base}_cleaned`.

**Metadata `{base}_metadata.json`:**

| Block / field | Meaning |
|---|---|
| `project_info` | `created_at`, `description`, `timezone` |
| `api_config` | `base_url`, `network` (pipeline files) |
| `token_info` | `token_address`, `token_symbol`, `token_name` |
| `pool_info` | `pool_address`, `pair`; orphans: `pool_address = null`, `pool_address_prefix` |
| `data_files` | paths of the data files (`cleaned_data`, `raw_data`) |
| `notes` | free-text notes written by the creating notebook |
| `metadata_incomplete` | `true` only for orphan files reconstructed from the file name |
| **raw** `death_point` | `is_dead`, `death_timestamp`, `window_hours` (48), `volume_threshold_sol` (1000), optional `marked_dead_reason`. **`is_dead = true` means "never update this coin"**: either a death point was found, or the coin was archived by the one-time fix |
| **processed** `death_point` (written by `data_cleaning`) | `is_dead` (the cleaning cut the coin), `death_timestamp` (last kept minute), `cropped_by` (`volume` / `label` / null), `rows_full`, `rows_removed_combined`, `volume_mark` {`death_timestamp`, `window_hours`, `volume_threshold_sol`, `grace_hours`, `rows_removed`}, `label_mark` {`death_timestamp`, `window_days`, `min_avg_label`, `forward_minutes`, `threshold_pct`, `rows_removed`} |

Note that **`is_dead` means different things in raw and processed files.** In raw files it's an instruction to the pipeline; in processed files it's the cleaning result.

### 0.7 Dataset, models and evaluation rules
* **Features:** 169, computed in `Dataset_creationV9` only (and copied verbatim into the live `features.py`):
  * the V9 features;
  * two pump-feature sets: no suffix for the move model, `_p15_2h` for the direction model;
  * the 22 window features (§18);
  * no `days_since_first`.
  * The move model uses every feature except the `_p15_2h` set; the direction model uses every feature except the move model's pump set.
* **Labels:** `label_10pct_30m`, `label_fpclass_15pct_2h` (2 = up first, 1 = down first, 0 = neither), `label_meta_minutes_to_end` (training rows need ≥ 240; never a feature).
* **Splits:** coin-level, stratified by coin length × pump rate; 15 / 10 / 25 / 25 / 25% (`SPLIT_PCTS`).

  | Split | Used for |
  |---|---|
  | train / val_train | model training / early stopping |
  | tune | strategy optimisation |
  | val_tune | selection and reporting |
  | **test** | read **once**, at the very end |

* **Rules every experiment followed** (keep them):
  * a **leakage test** for features (values identical with and without future data);
  * a **causality test** for every simulator change (a coin simulated on full and on cut data must give identical past trades; §35);
  * **controls** (shuffled scores, random blocks) for any rule that skips or changes trades;
  * a decision rule **fixed before** looking at results;
  * a confirmation with **new model seeds**;
  * the test split used **once**.
* **The two biggest pitfalls found:**
  1. a simulator rule applied after the fact leaked the future and produced a fake doubling of the score (§35);
  2. model-level AUC does not predict strategy results; only sims do (§20).

### 0.8 Current status and next steps (presentable project)
* **Status:**
  * the data pipeline is rebuilt (§0.4–0.5);
  * `Dataset_creationV9` is rebuilt from `processed_data/`: 402 coins, 5 splits (§42);
  * `model_search` on the rebuilt data chose `move_trial057` + `dir_for_move_trial057_trial002` (§42);
  * `strategy` is running; its saved results cover the first 30 of 50 runs (§42);
  * the live pipeline matches the notebooks: features bit-identical, strategy identical trade by trade (§41).
* **Results in §1–35 come from the OLD data and splits.** The rebuild changes the coins, the splits, the fill of gap minutes and the features of those minutes, so its numbers aren't directly comparable. Re-establish everything on the rebuild.
* **Next:**
  1. ~~rebuild the dataset~~ (done);
  2. ~~run `model_search`~~ (done);
  3. ~~copy the chosen pair to `model_results/`~~ (done);
  4. finish `strategy` (50 runs) and re-run its results cells. The **test score of the best-val run** is the headline result; read test once.
  5. Run the export cell of `strategy` and copy `strategy_results/live_artifacts/` into `live_pipeline/artifacts/` (§0.12).
  6. Paper / shadow trade before any claim of real profit.
  7. Run the finalization study (§40) on at least 2 coins of different activity; set the live read time from the slowest result.
* **Open ideas, not yet tested:**
  * a **causally cropped test set** (the label mark cut at the window *end*), to measure how much the hindsight cut flatters the results (§0.5);
  * the direction model's candidate filter with other price-position caps (`A50_PP_CAP`: none / 0.5 / 0.7 / 0.8 / the tuned caps);
  * more coins, the biggest lever for everything.
* **Future improvement:** move from Python 3.9.20 to a current Python version (re-run the tests afterwards).

### 0.9 Setup and running
* **Python 3.9.20** (the author's; see 0.8 for upgrading).
* **Packages:** pandas, numpy, numba, xgboost (≥ 2.0, `.ubj` model files), optuna, cmaes (Optuna's CMA-ES sampler), scikit-learn, scipy, pyarrow (parquet caches), matplotlib, requests (pipeline).
* **GPU:** the XGBoost settings use `device: 'cuda'` (an NVIDIA GPU). Without one, set it to `'cpu'` in `model_search` (`PARAMS_FIXED`); training is then much slower.
* **Run All** works for every notebook, and every long notebook is **resumable**: finished studies, models and sim runs are skipped.
  * To start a step fresh, delete its output folder (`processed_data/` is rebuilt anyway).
  * In `model_search`, changing `MOVE_CHOICE` starts a new direction search under its own file names.
* **Rough runtimes** (the author's machine, ~370 coins):
  * cleaning: minutes;
  * dataset build: tens of minutes;
  * `model_search`: many hours (2 × 60 Optuna trials + 200 strategy runs at a few minutes each);
  * `strategy`: hours (N runs × a few minutes).
  * Strategy runs take longer the more trades the found strategy makes.

### 0.10 Maintenance warnings
* **The simulator is copied, not imported.** `model_search.ipynb` and `strategy.ipynb` each contain their own copy of the engine cell and the sim-helper cell, as every research notebook did. **A change to the strategy logic must be made in both** (and in the live bot); otherwise their results silently diverge. After any engine change, run the causality test (§35): a coin simulated on its full data and on data cut at minute C must give identical trades before C.
* **`features.py` (live) contains a verbatim copy of the dataset notebook's feature code.** After any feature change, copy it again; apart from the lines marked `LIVE OPTIMISATION` the code must stay identical (§41).
* **The re-entry rule is the same in the live bot and the research engine (relaxing)** (see 0.11). An earlier live version tightened instead; it has been fixed, and §41 confirms the live strategy trade by trade.

### 0.11 Strategy mechanics (research engine, rebase mode)
The authoritative code is the engine cell (`simulate_coin` / `simulate_trade`).

**Entry candidates:** minutes where all hold:
* the buy score ≥ `confidence_threshold`;
* `price_position_15m` ≤ `max_price_position_15m` and `price_position_1h` ≤ `max_price_position_60m`;
* at least `trading_start_minutes` (720) after the coin's first minute.

Only one position per coin at a time; the next entry must come after the last sale.

**Re-entry rule (after a sale):**
* buy only if the close ≤ last sale close × (1 − `reentry_threshold_pct`);
* after `reentry_decay_start_minutes`, the required drop shrinks by `reentry_decay_rate` per minute (the rule **relaxes** over time). Once it's far below zero, any candidate qualifies.
* **The live `strategy.py` uses the same relaxing rule** (`passes_reentry`). An earlier live version did the opposite (the required drop *tightened* over time); all research results assume the relaxing rule.
* **Tested 2026-11-17: TIGHTENING does worse than relaxing** (`strategy_tighten.ipynb` vs `strategy.ipynb`: same models, same seeds, paired). **Relaxing is the correct rule.**
* `strategy_tighten.ipynb` is kept only as the record of this test.

**Size:**
* the whole capital of that coin,
* capped so the buy moves the price by at most `desired_price_impact_pct` (0.333%): at most `estimated_sol_1pct_impact` × 0.333 SOL. Without a liquidity estimate there's no cap.

**Fills:**
* entry at the close × (1 + `slippage_pct`, 1.5%);
* exit at the close × (1 − 1.5%).

**Exits and phases of a trade:**
* **take-profit:** the peak close ≥ the reference price × (1 + `activation_threshold`) → sell. This is the only way a trade ends before the end of the data.
* **loss floor:** the close ≤ the reference × (1 − `loss_floor_pct`) for `loss_floor_consecutive_minutes` minutes in a row arms the loss-floor race:
  * the position is **not** sold;
  * the next entry candidate that passes the re-entry rule **re-anchors** the reference price (a "rebase"), so the take-profit is then measured from the lower price;
  * a close back above the floor disarms it.
* **stale time:** after `stale_trade_minutes` without the take-profit, the trade enters the stale race: the take-profit or a re-anchor at the next qualifying candidate, again without selling.
* **end of data:** sell at the last close.
* P&L is always measured from the **original** entry price.

**Score:**
* the mean over coins of min(final capital, 10 × $20);
* untraded coins count as $20.

### 0.12 From the notebooks to the live bot
* The last cell of `strategy.ipynb` writes `strategy_results/live_artifacts/`:
  * `move.ubj`, `direction.ubj`: the two models of `model_results/`, unchanged (each file carries its feature list);
  * `ref_move.npy`, `ref_direction.npy`, `ref_combo.npy`: the sorted tune predictions, i.e. the reference distributions of the percentile scores. The combo reference is the product exactly as the sims compute it (same dtype), so live percentiles equal the sims' bit for bit;
  * `best_params.json`: the parameters of the **best val run**, with its val, tune and test scores.
* Copy the files into `live_pipeline/artifacts/` and restart the app: `scoring.py` loads the models and references, `database.py` the parameters (reloaded whenever the file changes).
* The models and the parameters must come from the same `strategy` run: the parameters were tuned on these models' scores.

---

## 1. Research notebooks (historical — see §0 for the current pipeline)

| Step | Notebook | Status |
|---|---|---|
| Clean raw OHLCV (crop dead coins) | `data_death_point_detection` | in use |
| Features + label + 7 coin-level splits | `Dataset_creationV8` | in use (label `label_10pct_30m`, 127 features) |
| Buy model (XGBoost) | `XGBoost_train`, `buy_model_optuna` | in use |
| Strategy optimisation | `optuna_strategy_nopenalty` | **current version** |
| Label search | `label_search_hpo` (v2) | done for 4 labels; see §4 |
| Diagnostic experiments | `optuna_strategy_experiments` | done; see §5 |
| First-passage label experiment | `label_firstpassage_experiment` | done: binary (§6.1–6.5) + direction-only (§6.6) |
| Move + direction with existing exits | `direction_strategy_test` | done; see §7. **`fp15_combo` is the lead candidate** |
| Label pair search (both models, coarse) | `label_pair_search` | done, incl. test confirmation; see §8 |
| Dump filter (model 3) | `dump_filter_experiment` | done; see §9 |
| Do models 1 + 2 + model 3's features replace model 3? | `feature_transfer_test` | done; see §10: **no** |
| Finer dump-label grid (model 3) | `dump_label_refine` | done; see §11: **no robust gain** |
| Hyperparameter search, models 1 + 2 (XGBoost / LightGBM / CatBoost) | `model_hpo_search` | done; see §12: **no gain; current models kept** |
| Dataset V9 (causal liquidity, uniform windows, renamed features, second pump-feature set) | `Dataset_creationV9` | **accepted; current dataset** (§13–15). Notebook cleaned up: no V8-dependent checks |
| V9 check (V9 vs V8) | `v9_check` | done; see §14: rebase same, **stop mode −$2.80** |
| Model-seed control (V8 vs V9 features × model seeds 42, 1, 2, 3) | `seed_control` | done; see §15: half the gap was luck; small residual |
| Baseline on the new V9 split | `baseline_v9` | done; see §16: **the reference for everything that follows** |
| 1% price-impact variants | `impact_variants` | done; see §17: **no variant clearly better** |
| Confirmation of the 5 best impact variants | `impact_confirm` | done; see §17: **none holds up; current method kept** |
| Extra windows + duplicate removal + feature importance | `feature_windows` | done; see §18: **neutral; kept for now** |
| New feature families (42 features, 7 families) | `feature_families` | done; see §19: **worse in the sims (−$0.87); not adopted** |
| Yardstick check (which model metric predicts the sims) | `yardstick_check` | done; see §20: **the old yardstick is anti-correlated; a strategy-free val_tune metric tracks the sims** |
| Strategy-aligned training (A: direction on candidate minutes; B: outcome model) | `aligned_training` | done; see §21: **A50 +$1.17 (5/5 seeds), just misses the Holm rule; outcome models worse** |
| A50 confirmation (new seeds 5–9) + pool size A40 / A60 / A70 | `aligned_confirm` | done; see §22: **A50 CONFIRMED and adopted** |
| Dynamic (coin-relative) buy thresholds + A50 with families | `dynamic_thresholds` | done; see §23: **small gains (≤ +$1.04); candidates dm_prod, i2_prod; families worse** |
| A50 optimised on tune only (no val selection; val once at the end) | `a50_tune_only` | partly run (46 runs); see §24: **val-selected scores carry a ~$4 selection bonus** |
| Tier classifier, ported to V9 (move model only) | `tier_classifier_v9` | run by the user (tier model trained on the new 7-split V9) |
| Strategy with the tier classifier (move model only) | `tier_strategy` | done; see §25–26: **free tiers ≈ 0; with the labels' fixed parameters (fixed_tier) +$4.55 (+10.5%) on 3/3 move models** |
| Tiers v2: A50 on the new split + tier-label diagnostics + tier sims | `tier_v2` | done; see §27–28: **A50 $51.24 is best; tiers +$3.2 for move-only, nothing reliable on top of the combo → tiers dropped** |
| Death diagnostics on A50 (profit per coin per day, loss anatomy, warning signals) | `death_diagnostics` | done; see §29: **"collapses" are where the profit is; no precise death signal; whole-coin vetoes not viable** |
| Rug-candle block + dump-risk veto on A50 (with controls) | `dump_risk` | done; see §30: **the risk model predicts dumps beyond volatility, but vetoing them costs money; rug block no value** |
| Risk-conditioned trade management on A50 (with controls) | `risk_manage` | done; see §31: **no gain; smaller size for risky trades hurts** |
| Three levers on A50: trailing / partial take-profit, seed ensemble, move model on tradable minutes | `levers` | done; see §32: **trailing hurts badly; ensemble and moveA ≈ A50** |
| Different strategies with the two models (signal hold / signal exit / scaled exits / pullback / entry filters / combined) | `strategies` | done; see §33 — **signal_exit / time_exit results INVALID (look-ahead, §35)** |
| Signal-exit follow-up: confirmation with new seeds, time exit, slippage 3%, tune-only | `exit_followup` | done; see §34 — **INVALID (look-ahead, §35)** |
| Earlier ideas re-tested on the new system (each alone, on top of signal_exit) | `new_system` | **invalid — uses the biased signal exit; stop / discard** |
| **GitHub notebook 3** — model search (from buy_model_optuna; metric − α·size): A1 move search → B1 top-10 move models ALONE in sims → A2 direction search on the CHOSEN move model's candidates → B2 top-10 direction models with the chosen move; 10 strategy runs each on tune / val_tune; reports the value of the direction model (pair vs move alone, paired) | `model_search` | built |
| **GitHub notebook 4** — strategy: loads the move + direction model from `model_results/`, N runs × up to X trials (optimise tune, select on val_tune), each run's selected parameters evaluated ONCE on test with the same score; tables ranked by val and by test; parameter plot (95% range, best run, search bounds) | `strategy` | built |
| Signal / time exit with the look-ahead fixed: causality test, re-evaluation, re-tuning, hold-out | `exit_fix` | done; see §35.1–35.2: **the entire gain was the look-ahead; no real gain on the hold-out either; A50 hold-out ≈ $44** |
| Tier classifier / confidence regressor | `tier_classifier_*`, `confidence_regressor_*`, `optuna_strategy_dynconf/regconf` | **parked** |

Data splits (by coin):
* `train` and `val_train` are for the buy model.
* `tune` and `val_tune` are for the strategy.
* `test` is held out.
* `clf_train` and `clf_val` are unused while the classifier is parked.

---

## 2. Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-10 | Tier classifier and confidence regressor parked | Focus on label and buy model first |
| 2026-10 | Trades-per-coin penalty removed from the score | Had no effect (§3.1); would bias the label comparison against rarer labels |
| 2026-10 | Take-profit becomes a tuned parameter (10–35%) | A fixed 15% favours labels near 15%; tuned values land around 0.26–0.29 |
| 2026-10 | Buy-model training drops 90% of negatives, then recomputes `scale_pos_weight` | No AUC loss; faster |
| 2026-10 | Pump-dependent features recomputed per label (25 features) | Otherwise the 10%/30m label gets an unfair head start |
| 2026-10 | Decisions are made on validation only; test is used once at the end | Selecting on test would leave no honest check |
| 2026-10 | Alternative score (capped mean × coin fraction) **not adopted** | No benefit in §5.4 |
| 2026-10 | Next label to test: **first passage**, 3 classes stored (2 = +X before −X, 1 = −X before +X, 0 = neither within T); model trained binary (class 2 against 1+0) for now | The current label rewards volatility, not direction (§5.1) |
| 2026-10 | Tested in a temp folder (`label_fp_experiment/`); the dataset is only changed if it wins | Keep the working pipeline intact |
| 2026-10 | No new exit mechanic yet: the direction models are tested with the existing `stop` and `rebase` exits first | User preference; a bracket exit only if the existing exits don't capture the edge |
| 2026-10 | Lead candidate confidence score: **baseline model × ±15%/120m direction-only model** (`fp15_combo`) | Only score that beats the baseline in both exit modes with large z (§7) |
| 2026-10 | Direction model gets its own pump definition: **first-passage pumps** (+X before −X), marked at the minute they complete; dump features deferred | Match the features to what the direction label measures, without look-ahead |
| 2026-10 | Coarse label search for both models before refining: move {10/30, 15/40, 20/60, 25/90}, direction {±10/60, ±15/120, ±20/180, ±25/240} | The two hand-picked labels were not necessarily best |
| 2026-10 | The test coins are used to confirm the label-pair candidates, re-scoring the saved runs without re-optimising | 91 validation coins are too few to measure $1–3 gains; this spends the test set |
| 2026-10 | **Final candidate: baseline move model (10%/30m) × ±15%/120m direction model with DATASET-STYLE pump features** (`fp15_combo`) | Confirmed on unseen test coins (§8.4) |
| 2026-10 | First-passage pump features for the direction model **dropped** | Behind dataset-style on VAL and TEST, in both modes (§8.4) |
| 2026-10 | **Model 3, a dump filter**, built on top of the confirmed combo. Label: the 15-min median falls ≥ D below the current 15-min median within n hours ('any' = at any point, 'end' = at the end). Grid: D {30, 50%} × n {2, 6, 12, 24h} × {any, end} | 32 of 40 losing coins lost by holding into a collapse (§5.6); dumps count even if the coin recovers |
| 2026-10 | Model 3 gets 13 new OHLCV features (long-window volume decay, zero-volume share, drawdowns, range shrinkage, last major pump (+50% within 24h), lower highs); models 1 and 2 keep their exact inputs | Keep the combo apples to apples |
| 2026-10 | Model 3 is tested as a **veto** (no buy above a tuned dump percentile) and in the **product** combo × (1 − P(dump)); validation only | Test set already used once |
| 2026-10 | Hyperparameter search for all three models (XGBoost, LightGBM, CatBoost) **after** the dump label is chosen | Don't mix the label choice with tuning |
| 2026-10 | Models 1 + 2 keep their current features; the 13 dump features are **not** added to them | No gain on their own, can't replace model 3, nothing added on top (§10) |
| 2026-10 | The 13 dump features are kept for now; feature elimination comes later, decided by sims | A +0.006 within-coin AUC change is within noise, but the features may still matter; only ablation decides |
| 2026-10 | Finer model-3 label grid: end {20, 30, 40%} × {18, 24, 36, 48h} + any {30%/24h, 30%/36h}, screened **on the product** (net result of its top 10/20%), best 3 + previous winner simulated | The coarse winner sat at the edge of the grid; the earlier blocking screen didn't rank the product winner first |
| 2026-10 | **Model 3 (dump filter) not adopted as an entry signal for now** | Its earlier gain did not replicate when retrained on slightly different rows; no label in the finer grid gives a robust gain (§11) |
| 2026-10 | Hyperparameter search for models 1 + 2 with XGBoost, LightGBM and CatBoost; top 3 per model × library kept, in each library's own format (.ubj / .txt / .cbm); 27 combos (9 library pairings × ranks 1-3) simulated, 10 seeds, both modes | Stabilise models 1 + 2 before revisiting model 3 |
| 2026-10 | Search objectives on val_train: move = partial AUC (FPR ≤ 10%), direction = within-coin AUC | The high-confidence end and the within-coin timing are what the combo uses |
| 2026-10 | Search size: 50 trials per model × library, the first 15 random, with Optuna median pruning (from round 200, every 25 rounds, on val AUC; CatBoost on GPU can't be pruned) | 25 trials was a light search for 6–12 settings; pruning keeps the runtime down |
| 2026-10 | **Keep the original XGBoost models 1 + 2** (XGBoost_train settings); no tuned model or other library adopted | No combo of 27 beats the current one robustly; in stop mode every tuned pairing is worse (§12) |
| 2026-10 | Next lever is **information (features), not model tuning** | Three libraries and 300 trials hit the same ceiling (§12) |
| 2026-10 | Dataset V9: hard-coded splits from `coin_assignments.csv` (later versions use the splitter again); windows snapped on a ratio scale; only the current 1% impact method, with the look-ahead fixed | Keep V9 comparable to V8 |
| 2026-10 | Keep both exit modes until feature engineering is done; then choose | Rebase earns more; stop is the more sensitive detector of entry quality, and the best features could change which wins |
| 2026-10 | V9 check with three arms: A = V8; B = V9 models and features (incl. V9's 1% impact as a feature) but V8's 1% impact for sim execution/sizing only; C = V9 everything. 20 paired runs, both modes | B − A isolates the features, C − B the execution change |
| 2026-10 | Pass/fail for V9: **equivalence**, B − A coin-bootstrap 95% interval inside ±$1.50 in both modes | "No significant difference" would pass almost anything at this noise level |
| 2026-10 | **Feature and dataset comparisons from now on use several model seeds per set-up** (e.g. 3 seeds), with the model as the unit of the test | One model's seed alone moves results by $0.4–0.8 (sd), and the original V8 model was a lucky draw in stop mode (§15) |
| 2026-10 | `Dataset_creationV9` uses V8's stratified splitter again (instead of hard-coded splits), adapted to **5 splits** (train 20 / val_train 10 / tune 25 / val_tune 25 / test 20 %, the current shares). Default changed to a **full re-split from scratch** (`PRESERVE_EXISTING = False`, 2026-10-29) | User's choice: with ~360 coins results should generalise. Consequences accepted: earlier numbers aren't comparable with the new split, and the new test set may contain coins used for earlier decisions |
| 2026-10 | 1% price-impact experiment: 35 variants (33 run): EW mean 4h / 3d; rolling median (≥5 candles) 4h / 24h / 3d; Amihud, Kyle, range, Corwin–Schultz, Abdi–Ranaldo, Roll, volume-weighted / trimmed mean / 25th percentile, implied AMM depth (3 windows each; AMM 24h / 3d skipped as duplicates of the medians) | Find the best way to compute this core feature before new feature engineering |
| 2026-10 | In that experiment the sim's execution always uses V9's current estimate; only the model features change. 1 model seed (42) × 20 runs per variant, paired with the baseline's seed-42 runs; finalists to be confirmed with 3 seeds. After each variant, features and models are deleted; the estimate itself, the runs and a summary are kept | Isolate the feature; keep disk use low |
| 2026-10 | Confirm the 5 best impact variants (kyle_3d, median_3d, amihud_1h, vwmean_3d, ew_3d) with 2 NEW model seeds (1, 2), 20 runs, both modes. **Criteria fixed in advance:** (1) the new seeds beat the baseline's seeds 1 + 2 in both modes; (2) over all 3 seeds not worse than the baseline in either mode; (3) winner = largest 3-seed gain, ties within $0.25 go to the simplest (ew_3d first); (4) none qualifies → keep the current method | Small gains are expected at this stage and the downside of picking a near-equal variant is low; fresh model seeds protect against model / optimizer luck (not against the coin sample) |
| 2026-10 | **`estimated_sol_1pct_impact` is locked: the current method** (EW geometric mean, 24h time constant, causal) | No variant held up with new model seeds (§17) |
| 2026-10 | Before new feature families: fill the window gaps of existing families (22 features: returns 1m–3d, volatility 4h/8h/3d, price position 4h/24h, drawdowns 4h/3d/ATH, minutes since ATH, rise from 4h/24h low, long-window volume ratios) and remove exact duplicates (`days_since_first` + any others a rank-correlation scan finds). Built as a **separate add-on CSV**; 5 model seeds; feature importance; 20 runs per seed against the baseline's same 5 seeds | User's choice: windows and duplicates first, new families afterwards |
| 2026-10 | Decision rule for that test, **fixed in advance**: improved if, in rebase mode, the 5-seed mean difference is > 0 and the paired test over the 5 model seeds gives p < 0.05 | Lesson from §17.1; rebase only (decision 2026-10-31) |
| 2026-10-31 | **Rebase mode only from now on** (stop mode only to check finalists). Replaces the earlier "both modes until after feature engineering" | Halves the runs; rebase is the mode you'd trade (higher returns: test $42.40 vs $37.93) and has far less model-seed noise on this split (sd $0.04 vs $0.53), so paired comparisons are sharper. Trade-off: stop mode was the more sensitive detector of entry quality, but a gain invisible in rebase wouldn't earn anything in the strategy you trade |
| 2026-11 | **Keep the 22 new window features for now** although the pre-set rule says "not improved" (exactly neutral) | User's choice; feature elimination later decides what stays |
| 2026-11 | New feature families tested against the current set (V9 − duplicates + windows): order flow (CMF, buy-volume share, OBV flow), VWAP distance, wicks, down-event history (mirror of both pump definitions), past first-passage outcomes (counted only once completed), trend quality (R², variance ratio), range volatility (Parkinson, Garman–Klass). 5 seeds, importance per family, rebase 20 runs. Rule fixed in advance: improved if mean diff > 0 and paired p < 0.05 over the 5 seeds | Add *new kinds* of information; the window expansion only re-described existing information (§18) |
| 2026-11 | **Not now: Optuna search judged by sims** (20 runs per trial). If ever: a cheap proxy objective on tune, several seeds per trial, sims only for a few finalists with criteria fixed in advance | A trial would take ~10 min, the sim score is too noisy to separate small parameter effects, and selecting on val_tune would overfit it (winner's curse, §17.1); the AUC landscape was flat (§12) |
| 2026-11 | **New feature families not adopted**; the current set stays V9 − `days_since_first` + the 22 windows | Sims worse (−$0.87, p = 0.050, 4 of 5 seeds lower) although the direction model's AUC improved (§19) |
| 2026-11 | **Stop using the direction within-coin AUC (or any model AUC) to judge features, tuning or models** | It correlates *negatively* with the sim result (Spearman −0.65, interval −0.87..−0.28, §20) |
| 2026-11 | **Screening yardstick: `tp_net_val_tune_top20`**: the mean net result of a "+24% take-profit or exit after 750 min" trade on the combo's top 20% of eligible val_tune minutes (after the sims' filters, minus 3% cost). Seconds instead of ~10 min per model. Sims (or a confirmation) still decide | Tracks the rebase sims (Spearman +0.74, Pearson +0.84, §20). It uses the same coins as the sims, so it's a fast stand-in for the sim, not an independent check |
| 2026-11 | Strategy-aligned training test (overnight, every arm simulated). Models: direction on candidate minutes (move top 30% / 50%); outcome "+24% within 750 min" on all tradable minutes; the same trade's net result as a regression; the outcome label on candidate minutes only. 8 arms (A30, A50, B_three, B_replace, B_alone, Breg_replace, Bcand_replace, AB) × 5 seeds × 20 runs, rebase. Screen first and **save its predictions before the sims**. Rule fixed in advance: **candidate** if mean sim diff > 0 and **Holm-corrected** paired p < 0.05; adopted only after a confirmation with new seeds | Align training with what the strategy rewards (§20); 8 arms give the yardstick a real test |
| 2026-11 | Outcome models ("+24% within 750 min", as a probability or as the net result) **dropped** | Every arm with them was worse; four of them significantly after Holm (§21) |
| 2026-11 | The val_tune yardstick is **not valid for models trained on its own target** (outcome / net-result models) | Breg_replace scored best on the screen (+1.39%) but was worse in the sims (−$1.04): a model trained on exactly the yardstick's trade games it (§21) |
| 2026-11 | **A50 confirmation, rule fixed in advance:** A50 and the current combo both trained with new seeds 5–9, 20 runs each, rebase; adopt A50 if the mean diff > 0 and the paired p < 0.05 (one pre-specified arm, no correction). A40 / A60 / A70 on the old seeds as a diagnostic only; a different best fraction would need its own confirmation | Independent test of the one promising result (§21) |
| 2026-11 | **ADOPTED: A50.** The combo is now move × direction, with the direction model trained only on candidate minutes: tradable (after 12h, both price-position filters pass), in the move model's top 50% (threshold from the move model's own train predictions), and decided (up first vs down first). Same features (V9 − `days_since_first` + 22 windows) and settings | Confirmed with new seeds 5–9: +$1.10, 4/5 seeds, p = 0.036 (rule fixed in advance); all 10 seeds +$1.13, 9/10, p = 0.0008 (§22) |
| 2026-11 | Pool size stays at **50%** | The gain is a broad plateau from 40% to 70% (+$0.94 to +$1.17); 50% sits in its middle and is the one confirmed (§22) |
| 2026-11 | **Dynamic-threshold screening** on top of A50. Problem: a coin's typical confidence level differs, so one fixed threshold doesn't fit all coins (the old tier classifier gave ~30%). 7 systems, each adjusting the move and direction scores separately ("option 3"): moving mean (absolute / z), past true positives with shrinkage, continuous tier, partial de-meaning, coin percentile, calibration model (trained on the test coins). Each combined as a product (one threshold) and as two gates. All with a floor (0.2–0.7) and an optional rising edge. Plus A50 + families (static). 15 arms × 3 seeds × 20 runs, rebase, up to 2,500 trials. Rule fixed in advance: candidate if better on all 3 seeds and mean > 0; the best 2 go to a 5-new-seed confirmation | Make the buy threshold follow each coin's own level continuously, with few parameters |
| 2026-11 | **Feature families dropped for good** | Also worse with A50: −$0.87, worse on 3/3 seeds, p = 0.028 (§23) |
| 2026-11 | **Confirm dm_prod and i2_prod** (the screening candidates) with new seeds 5–9 against A50's seeds 5–9 (runs exist in `aligned_confirm`). Rule fixed in advance: adopt if the mean diff > 0 and the paired p < 0.05 after a Holm correction over the 2 arms | Screening rule picked them (§23) |
| 2026-11 | **V9 splitter back to 7 splits:** train / val_train / **clf_train / clf_val** / tune / val_tune / test; default V8 shares 14 / 7 / 14 / 7 / 19 / 19 / 20%. The next V9 run re-splits everything, so all reference runs (A50 etc.) must be redone on the new split | The tier classifier needs its own coins, unseen by the move model and by the strategy splits |
| 2026-11 | **Tier classifier first WITHOUT the direction model:** `tier_classifier_v9` labels and bins the confidence of A50's move model only, as when the tiers were in use | User's choice: test the tier idea on the first model before combining it with the direction model |
| 2026-11 | **Tier strategy test:** move model alone (static confidence, seeds 42 / 1 / 2, 10 runs each) vs tiers (a tuned buy confidence per tier 0–4; tier smoothed to the most common tier over the last W minutes, W tuned 3–30, ties → higher tier) with 3 move models × 10 runs. Val-selected as usual; rebase | Test whether the tier idea helps on the current data before combining it with the direction model |
| 2026-11 | **Score test: money × win rate.** `tier_strategy` re-run with SCORE_MODE 'winrate' (mean capped capital × pooled win rate of all trades) for optimisation and val selection, both scores printed; runs in `tier_strategy/sim_winrate/`, compared with the money-only runs in `sim/` | The tuned buy confidence is low (raw ≈ 0.45 = top ~45% of minutes) and the money optimum is flat over thresholds; a win-rate factor prefers fewer, better trades, possibly more robust live. Judge it by its val MONEY, not by its own score |
| 2026-11 | **Money × win rate not adopted** | It raised the confidence (0.52 → 0.75) and the win rate (≈ 60% → 77%) mainly by lowering the take-profit (≈ 0.20 → 0.13), not by trading less (≈ 17 trades/coin either way), and cost about $2 val money; the tiers gained nothing under it |
| 2026-11 | **The old tier design re-created** in `tier_strategy` PART 3 (raw-probability gate, smoothed tiers): `fixed_static` (one confidence, everything else fixed to the live single-tier strategy the labels were made with), `fixed_tier` (a confidence per tier inside the tier's own range: dead 0.75–1.00, 0.50–0.64, 0.64–0.74, 0.74–0.82, 0.82–1.00; rest fixed), `tier_old` (the original: per tier confidence + take-profit, stale, both price-position caps, re-entry, loss floor). Money score | The earlier test only tuned a free confidence per tier; the original bounded each tier's confidence to its label range and tuned the whole strategy per tier, which is likely where its gain came from |
| 2026-11 | **Model 1 gated on its RAW probability again** (`tier_strategy` CONF_SCALE 'raw', bounds 0.20–1.00), as in the original code; runs in `tier_strategy/sim_raw/`, the earlier percentile runs kept in `sim/`. The combo keeps its percentile scale when model 2 returns | The tier labels and ranges are raw probabilities; the percentile was introduced for the move × direction product, not needed for one model |
| 2026-11 | **Tiers only with the labels' own fixed strategy parameters**; per-tier everything (tier_old) dropped | fixed_tier +$4.55 over fixed_static (3/3 models) and the best set-up overall; free tiers ≈ 0; tier_old overfits (§26) |
| 2026-11 | **Tiers v2 in one notebook** (`tier_v2`): (1) A50 on the new 7-split V9, 3 move models × 10 runs, giving the reference + median tuned settings; (2) relabel clf_train / clf_val with those medians, sweep 0.20–1.00, keep every window's full curve, for two variants (move-only gate / A50 combo gate), and rank tier designs offline (2–5 trading tiers, quantile / equal-width cut-offs, case 2/4 dropped / kept) by oracle and realised gain; (3) sims of the best designs as fixed_tier vs fixed_static (same fixed median settings) vs A50 | One relabel per variant evaluates every design; offline numbers only rank, the sims decide |
| 2026-11 | **A50 is the current system on the new split** ($51.24, +$4.43 over move alone, 3/3 seeds) | §27 |
| 2026-11 | **Last tier test (PART 4 of `tier_v2`)**: (A) bounded per-tier confidences + all other settings tuned; (B) labels from the best single A50 run's settings (chosen by tune score, so val stays clean). Rule: tiers are kept only if an arm beats A50 (tuned) on all 3 seeds; otherwise dropped | Two untested combinations that could still let tiers add to A50 (§27) |
| 2026-11 | **Tier classifier dropped**; A50 (tuned) remains the system | Nothing beats A50 on all 3 seeds in the last test (best +$0.33, n.s.) (§28) |
| 2026-11 | Candidate for live: **A50 with fixed settings from one coherent run** | fixed_static_combo_best is level with full tuning (−$0.47, n.s.) (§28) |
| 2026-11 | **No whole-coin death veto / permanent stop** | 50%-in-an-hour crashes occur in 92% of coins and the trading after them earns most of the profit; real deaths cost ≈ $5 per run; a veto needs 82–89% precision while the best signal reaches 7–30% (§29) |
| 2026-11 | **Rug block + dump-risk veto test**: rug_block (after a 5-min fall ≥ x, no entries for M min) vs rug_random (same blocks shifted, control); risk_veto (a dump-risk model trained on A50 candidate minutes; target = price −40% before the take-profit within the stale time) vs risk_shuffled (control); rug_and_risk. 3 seeds × 10 runs | An arm counts only if it beats A50 AND its control; the risk model's AUC is also checked inside volatility deciles (more than volatility?) |
| 2026-11 | **Rug-candle block and dump-risk veto not adopted** | risk_veto −$2.74 vs A50 and −$2.70 vs its shuffled control; rug block −$0.87 vs control (n.s.) (§30) |
| 2026-11 | **Risk-conditioned management test**: every trade still taken; trades entered at high risk (top x% of the dump-risk score, x tuned 5–30%) get their own take-profit / stale / loss floor (hr_exits), a smaller size (hr_size, 0.25–1.0×), or both; shuffled-risk controls. 3 seeds × 10 runs | Use the risk score for HOW to trade, not WHETHER (§30) |
| 2026-11 | **Risk-conditioned management not adopted; the dump-risk line is closed** | hr_exits −$0.28 (n.s.), hr_size −$2.70; nothing beats A50 and its control (§31) |
| 2026-11 | **Three levers test** (`levers`): trail (no sale at the take-profit; trailing stop 5–50% below the high, ≤ 30–2,880 min), partial (a share sold at the take-profit, the rest trails), ens (average of the 3 move × 3 direction models; 30 runs), moveA (move model trained on tradable minutes + direction retrained on its candidates) | Profit concentrated in coins that run far (§29); the optimiser let risky trades run longer (§31); seed noise $0.5–1; alignment worked for the direction model |
| 2026-11 | **Trailing / partial take-profit, seed ensemble and moveA not adopted** | trail −$9.01, partial −$5.16 (3/3 seeds); ens +$0.37 and moveA −$0.30, both n.s. (§32) |
| 2026-11 | **Strategy batch** (`strategies`): signal_hold (hold at the take-profit while the combo ≥ h), signal_exit (sell when direction < x or move < z; take-profit optional), tp_scaled (exits scale with the entry combo), pullback (limit buy x% below within y min), entry_filters (momentum, time of day, coin age, k-minute confirmation; each on/off), combined (~20 params); shuffled controls for the two model-driven exits. Up to 2,500 trials. Rule: candidate if better than A50 on all 3 seeds (and than its control); best 1–2 confirmed with new seeds | The models so far only decide entry; use them during trades and test new entry rules. More parameters allowed when they add flexibility (user, 2026-11) |
| 2026-11 | **signal_exit scored ~$110 on seed 42 (vs A50 ~$51): checked before trusting it.** No bug found (exit price, shares, next-entry logic, causality, the off-switch reproduces A50). Possible same-minute fill advantage: the sim fills at the signal minute's close; an exit that fires on a spike candle can capture a reversal a live order misses. Added an execution check: fills at the next minute's open (≈ live) / close, for signal exits and for ALL entries + exits (A50 too), plus the price move right after each fill | Deciding with the closed candle is legitimate; the fill price is the open question |
| 2026-11 | **Signal-exit follow-up** (`exit_followup`): PART 4 confirmation with NEW move + direction models (seeds 5–7): A50 vs signal_exit vs time_exit; **rule fixed in advance: adopted if mean diff vs A50 > 0 and paired p < 0.05**. Plus a pure time exit (sell at market T min after entry) and time exit + signal on seeds 42/1/2; slippage 3% (re-tuned and not); tune-only selection check | The biggest change in the project needs an independent confirmation and robustness checks before adoption |
| 2026-11 | ~~ADOPTED: signal_exit~~ **WITHDRAWN 2026-11-15:** its results came from a look-ahead in the engine (§35). **A50 remains the system** until the fixed-engine results are in | §35 |
| 2026-11 | **Re-test on the new system** (`new_system`), each idea alone on top of signal_exit (no combinations yet): dynamic thresholds (dm / tier / i1abs, product), entry filters, pullback, tiers (best combo design, free / bounded), families (A50 retrained with the 42 family features on the new split). Rule: candidate if better than signal_exit on all 3 seeds; best 1–2 confirmed with seeds 5–7 | The strategy now trades very differently (loose entries, ~1 h holds), so ideas that failed on A50 may behave differently (user, 2026-11) |
| 2026-11 | **Hold-out check added to `exit_fix`:** EVERY run of A50, signal_exit, shuffled and time_exit evaluated once on clf_train + clf_val (never used to tune or select any strategy). Reported: all-runs mean per seed, best-by-val run per seed, val → hold-out rank correlation. **Rule fixed in advance:** a real gain = all-runs mean beats A50 on all 3 seeds AND the coin-bootstrap 95% CI of the difference excludes 0. The clf coins count as spent after this decision | The fixed signal_exit averages ≈ A50 on val, but a few runs reach ~$55 — check whether that holds on unseen coins without selecting runs by val |
| 2026-11 | **Data pipeline for the presentable version:** (1) gap filling = FLAT candle at the last close (open = high = low = close, volume 0) in the pipeline and the cleaning notebook, matching the live bot (research data so far copied the previous candle's OHLC); (2) the death-point notebook renamed `data_cleaning`: volume mark = the pipeline's rule (48 h window < 1000 SOL, windows start after a 48 h grace, cut at the window end), label mark unchanged (cut at the window start), minimum 12 h instead of 24 h, reads `raw_data/` and writes `processed_data/` (raw never modified); (3) the orphan-CSV metadata fix moved to a one-time notebook on `raw_data/` | Presentable, reproducible data pipeline; live / research parity of gap minutes; fewer future-based exclusions. **All datasets / models / results must be rebuilt from the new processed data** |
| 2026-11 | **Raw data folder rules:** gap filling everywhere = every ZERO-VOLUME minute is a flat candle at the last real close, `backfilled` derived from volume = 0 (fixes lost flags on updates and old copy-the-previous-candle fills); the pipeline writes / updates `raw_data/` (no archive folder; coins marked dead are never updated); one-time notebook `fix_raw_data_once` (orphan metadata, zero-volume minutes flattened, all archived coins marked dead except EXCLUDE) | The API only reaches ~6 months back: updating old coins would leave gaps; the flag was dropped on every update |
| 2026-11 | **Dataset_creationV9 for the presentable version:** reads `processed_data/` (cleaned by `data_cleaning`); 5 splits again (train / val_train / tune / val_tune / test, 15 / 10 / 25 / 25 / 25%; clf splits removed with the tier line); `days_since_first` removed from the feature code (147 features). **The 22 window features are now computed inside Dataset_creationV9** (169 features; identical values to the former add-on; covered by the leakage test) and in the live `features.py` (a verbatim copy of the dataset code + live section); `model_search` reads the splits as they are | Tier classifier dropped; the duplicate feature is gone at the source; one place computes every feature |
| 2026-11 | **Re-entry rule: RELAXING is correct** (the buy-back limit rises after the decay start). Tightening (the live bot's version) tested worse in `strategy_tighten` vs `strategy` (paired, same models and seeds). The live `strategy.py` was fixed to the relaxing formula (§41) | User's test, 2026-11-17 |
| 2026-11 | **Repository layout:** `development_notebooks/` (the five pipeline notebooks), `research_notebooks/` (finalization study), `live_pipeline/`. The live bot takes its models, reference distributions and parameters from the export cell of `strategy` (§0.12) | One path from the data to the live bot, all of it in the presented notebooks |
| 2026-11 | **Live fill rule aligned:** `data_collection.py` applies the research rule (every zero-volume minute flat at the last real close) | It only filled skipped minutes; API zero-volume candles were kept as returned (§41) |
| 2026-10 | Dual gate (a separate threshold per model) **not adopted**; the product with one threshold stays | Worse than the product in both modes; significant in stop mode (p = 0.007), not in rebase (p = 0.17) (§7.4) |

---

## 3. Strategy optimisation (`optuna_strategy`, 20 runs)

### 3.1 Penalty vs no penalty
* The two versions gave **identical results**: the same trial, parameters and scores in all 20 runs.
* The penalty only applies below 2 trades per coin. The selected configs trade 16–21 times per coin, so it never fired.

### 3.2 Run statistics (penalised = no-penalty)

| | Mean | Median | SD | Range |
|---|---|---|---|---|
| Tune | $43.47 | $43.11 | $2.72 | $38.81–48.19 |
| Validation | $41.80 | $41.71 | $1.68 | $39.47–46.02 |
| Test | $37.25 | $37.27 | $4.67 | $29.26–45.61 |

* **Validation does not predict test across runs** (r ≈ −0.13). The best run on validation ranks 11th of 20 on test. The differences between runs are noise; expect about $37 on test, not the winner's validation score.
* **Scores drop at each step from tune to validation to test.** This is optimisation and selection optimism.
* **Test edge is real but comes from a minority of coins.**
  * The capped mean is $36.85 against $20 (p = 0.003). It stays significant after removing the top 3 coins.
  * The median coin ends at $17.88, and 38 of 71 coins finish below $20.
  * The tests on the typical coin (Wilcoxon and sign test) are not significant.

---

## 4. Label search v2 (`label_search_hpo`)

Setup:
* Per label: 60 buy-model trials, the top 3 models, 5 strategy runs per model (15 runs per label), plus a 5-run null (rank-1 model's predictions shuffled within each coin).
* The safeguard passed on all four splits (~4.4M rows) with zero mismatches.
* Labels run: 10% and 20% × 30m and 50m. A grid edit meant 15%, 25%, 40m and 60m were not run.

| Label | Mean val | SD | Null | Edge | Tuned TP | pAUC |
|---|---|---|---|---|---|---|
| 20%/50m | $45.80 | 1.45 | $44.77 | +1.03 | ~0.27 | — |
| 10%/30m | $45.61 | 1.15 | $45.59 | +0.02 | ~0.27 | 0.599 |
| 20%/30m | $44.86 | 2.18 | $44.08 | +0.78 | ~0.27 | 0.642 |
| 10%/50m | $44.69 | 2.48 | $45.48 | −0.79 | ~0.27 | — |

* **All four labels are tied**: the spread between them (~$1.10) is smaller than the spread within a label.
* **No label beats its own null** (|z| ≤ 1.4). The strategy sim cannot tell a real model from a shuffled one, so it cannot rank labels.
* Tuning the take-profit (~0.27 instead of 0.15) raised scores for real and null alike, from $41.80 to about $45.
* Many validation-selected runs kept trial 250, the first validation sample. Tuning on `tune` mostly doesn't transfer to `val_tune`.

---

## 5. Diagnostic experiments (`optuna_strategy_experiments`)

One buy model (10%/30m). 10 paired runs per arm on validation; test not used. Nulls:
* **within:** predictions shuffled in time within each coin.
* **cross:** predictions shuffled across coins.
* **const:** no model.

### 5.1 Entry quality: are high-confidence minutes better? (experiment 1)
**Yes, at predicting upside.** P(max close in the next 60 min ≥ +10%):

| Bucket | Pooled | Within-coin |
|---|---|---|
| Bottom 50% | 12% | 15% |
| Top 1% | 69% | 63% |

* The within-coin rank correlation is +0.29, positive in 89 of 91 coins. Tune and validation are nearly identical.
* **This confirms the live observation**, including for timing within a coin.

**But mostly not direction.** First passage (the price must hit +X before −X), validation:

| | ±10% hit first | ±15% hit first |
|---|---|---|
| All minutes | 48% | 50% |
| Top 1% | 52% | 54% |
| Top 1–25% | 48–51% | 49–51% |

* Median returns at 30, 60 and 240 minutes are negative in every bucket, and most negative in the top 1–25%.
* **The model mainly predicts volatility (big moves both ways).** Only the top ~1% has a small directional edge, below the ~3% round-trip slippage.
* **Cause: the label.** "Max close within 30 min ≥ +10%" is true in most volatile minutes, whatever happens to the price afterwards.

### 5.2 Free gate (experiment 2)

| Arm | Val | Real minus arm (z) |
|---|---|---|
| real | $46.56 | — |
| within | $44.44 | +2.13 (1.9) |
| cross | $43.75 | +2.82 (4.1) |
| const | $46.32 | **+0.24 (0.4)** |

* **The real model equals no model.** A random gate is worse than no gate; the real gate is about the same as no gate.

### 5.3 Strict gate (experiment 3): confidence fixed at the top X% of predictions

| Gate | Real val | Real minus within (z) | Real minus cross (z) | Untraded | Trades per coin |
|---|---|---|---|---|---|
| top 20% | **$47.16** | +6.05 (7.1) | +3.09 (6.6) | 3% | 12.2 |
| top 10% | $38.16 | −1.38 (−1.4) | −5.39 (−7.3) | 7% | 8.2 |
| top 5% | $31.52 | −4.34 (−7.0) | −11.59 (−29.6) | 13% | 5.1 |
| top 2% | $26.52 | −4.45 (−9.6) | −15.32 (−42.1) | 25% | 2.1 |

* A mild gate (top 20%) helps against the nulls. It gives the best score of all experiments, but is only about level with no model.
* **Stricter gates hurt badly.** The highest-confidence minutes are bunched into a few coins and volatile episodes, so fewer coins get traded. The shuffled versions spread the same number of minutes over every coin.

### 5.4 Alternative score (experiment 4)
* No separation from the nulls (z −0.2 and 1.1).
* The tuned confidence barely moves (0.68 against 0.63).
* The capped mean drops from $46.56 to $42.37.
* **Not adopted.**

### 5.5 Stop exits (experiment 5): stale timer and loss floor sell instead of re-basing
* **The real model now beats both nulls**: +$3.95 (z 3.6) and +$3.90 (z 5.0). Entry quality matters once bad entries cost money.
* The strategy earns less overall: $42.06 against $46.56. Stops realise −$6,425 against +$9,332 from take-profits.

### 5.6 Where the money is made and lost (real model, free gate, validation)

| Exit | Share of trades | Summed P&L |
|---|---|---|
| Take-profit | 95.4% | +$4,475 |
| End of data (coin died while held) | 4.6% | −$634 |

* **32 of the 40 losing coins lost by holding a position when the coin's data ended.**
* The main source of losses is dying coins, not bad entries.

---

## 6. First-passage label experiment (`label_firstpassage_experiment`)

Setup:
* Variants: ±10%/60m, ±15%/120m and ±25%/240m (pump features matched to each label), plus the baseline (current label).
* Every model was trained with the XGBoost_train settings on identical rows.
* Binary target: up first against (down first + neither).
* Simulation: 10 paired runs, in hold-until-take-profit ("rebase") and stop-exit modes, real against the within-coin shuffle. Validation only.

Tests:
* All tests passed.
* TEST 2: the baseline recompute matched the dataset exactly on ~4.4M rows (label and 25 features).

### 6.1 Label mix (validation, similar on every split)

| Label | Up first | Down first | Neither |
|---|---|---|---|
| ±10% / 60m | 24.2% | 22.5% | 53.3% |
| ±15% / 120m | 25.1% | 20.4% | 54.5% |
| ±25% / 240m | 23.5% | 14.5% | 62.0% |

* Symmetric % barriers are not symmetric on a log scale (+25% is closer than −25%). Even a random price path hits "up first" more often: about 52.5%, 54% and 56% of decided minutes.
* Observed shares among decided minutes: 52.0%, 55.4% and 62.3%. Only ±25% shows a small upward lean beyond that geometry.
* **Compare directional results with the "all minutes" rate, not with 50%.**

### 6.2 The binary first-passage models learned volatility again
Mean prediction by true class (validation):

| Model | Up first | Down first | Neither |
|---|---|---|---|
| ±10% / 60m | 0.561 | **0.586** | 0.371 |
| ±15% / 120m | 0.555 | **0.584** | 0.386 |
| ±25% / 240m | 0.566 | **0.618** | 0.364 |

* "Neither" is more than half of all minutes, so the easiest way to score well is to separate moving minutes from quiet ones. That is volatility, and the top features are again volatility measures.
* Among minutes that do move, the models rate **down-first slightly higher** than up-first. Directional AUC on their own label: 0.42–0.46, below 0.5.

### 6.3 Direction, same yardstick for every model (validation)
* **Pooled directional AUC is below 0.5 for every model, including the baseline** (0.41–0.47). Across coins, high predictions are more often followed by a fall first. The most volatile coins are also the ones that dump.
* **Within coins:**
  * new labels: 0.50–0.515, essentially no sense of direction;
  * baseline: 0.47–0.485, slightly wrong-way.
* **Top buckets:** only the top 1% of the new-label models lean up, by +4 to +8 points over the "all minutes" rate. The top 5–20% are at or below the overall rate.

### 6.4 Strategy sim (validation)

| Model | Rebase: real | vs null (z) | vs baseline (z) | Stop: real | vs null (z) | vs baseline (z) |
|---|---|---|---|---|---|---|
| baseline | $45.42 | −0.06 (−0.1) | — | **$41.41** | **+3.64 (+10.6)** | — |
| ±10%/60m | $44.74 | +0.31 (+0.4) | −0.68 (−0.9) | $38.99 | +0.54 (+1.0) | −2.43 (−3.7) |
| ±15%/120m | $44.79 | −0.17 (−0.3) | −0.64 (−0.9) | $40.14 | +2.36 (+2.4) | −1.27 (−2.1) |
| ±25%/240m | $45.37 | +0.16 (+0.4) | −0.06 (−0.1) | $39.84 | +1.99 (+1.9) | −1.57 (−1.9) |

* **Rebase mode:** all labels are tied and none beats its null, as expected for hold-until-take-profit.
* **Stop mode:** the **baseline is best** and beats its null strongly. Every new label scores lower than the baseline.
* Even with stops, the strategy is paid for volatility: a take-profit hit (~+26%) is worth more than a stale exit loses. The baseline is the best volatility predictor (AUC 0.77 on its own label).

### 6.5 Conclusion
* **The binary first-passage label does not beat the current label. Do not change the dataset.**
* The binary set-up puts "neither" among the negatives, so the model learns volatility first. It doesn't answer whether direction is predictable at all.
* Within-coin directional AUC is ~0.50–0.515 for every variant, so direction may simply not be learnable from these features. (Answered in §6.6: it is learnable.)

### 6.6 Direction-only test: direction IS learnable
Each label's model was trained only on minutes that hit a barrier (up first = 1, down first = 0), with quiet minutes left out, so volatility cannot help. XGBoost_train settings were used. Evaluation was on two unseen coin sets.

| Label | Train AUC | Within-coin AUC, val_train (38 coins) | Within-coin AUC, val_tune (91 coins) | Coins above 0.5 (val_tune) | Earlier binary model (val_tune) |
|---|---|---|---|---|---|
| ±10% / 60m | 0.642 | 0.594 ± 0.011 | 0.571 ± 0.006 | 85 / 91 | 0.501 |
| ±15% / 120m | 0.675 | 0.609 ± 0.013 | 0.596 ± 0.006 | 87 / 91 | 0.513 |
| ±25% / 240m | 0.701 | 0.609 ± 0.015 | 0.604 ± 0.009 | 76 / 90 | 0.510 |

Up-first share among decided minutes (val_tune), all minutes against the model's top buckets:

| Label | All | Top 1% | Top 5% | Top 10% | Top 20% | Top 10% within each coin |
|---|---|---|---|---|---|---|
| ±10% / 60m | 51.8% | 70.3% | 67.6% | 64.7% | 61.9% | 62.6% |
| ±15% / 120m | 55.1% | 78.5% | 73.3% | 70.1% | 67.4% | 67.2% |
| ±25% / 240m | 61.8% | 88.3% | 83.3% | 80.3% | 76.5% | 75.3% |

* **Direction is learnable.** Within-coin AUC is 0.57–0.61 on both unseen coin sets, against ~0.50 for the earlier binary models.
* **It is consistent:** above 0.5 in 85–87 of 91 coins for ±10% and ±15%.
* **It is not memorised.** The gap between train and validation AUC is small (0.64–0.70 against 0.59–0.63).
* **It is mostly timing, not coin choice:** within-coin AUC is close to pooled AUC.
* **What drives it:** the pump-history features recomputed for each label. `minutes_since_last_pump`, `cooling_off_indicator`, `time_since_cluster` and `buying_pressure_accumulation_rate` lead for ±10% and ±15%. Recent momentum (`sum_positive_returns_60m`, `cumulative_positive_since_last_pump`) leads for ±25%.
* **Why the earlier binary models missed it:** "neither" among the negatives let volatility dominate (§6.2).

**Rough value per trade.** Assumes a symmetric bracket exit (sell at +X or −X), about 3% round-trip slippage, and only trades that hit a barrier; trades that hit neither are ignored. Formula: (2p − 1)·X − 3%.

| Label | Top 10% within coin | Top 5% pooled |
|---|---|---|
| ±10% | p 0.626 → **−0.5%** | p 0.676 → +0.5% |
| ±15% | p 0.672 → **+2.2%** | p 0.733 → +4.0% |
| ±25% | p 0.753 → **+9.7%** | p 0.833 → +13.7% |

These are optimistic upper bounds, for three reasons:
* the model only applies to minutes that end up moving, which you can't know in advance;
* fills at the barriers in fast moves can be worse than modelled;
* neighbouring minutes share outcomes, so the effective sample size is much smaller than the row counts.

They show that ±10% is too small to beat costs, while ±15% and ±25% leave room.

### 6.7 What this means
1. **The edge needs two parts:** will the price move (volatility, which the existing models already predict well), and which way (the new direction models). A 3-class model (up first / down first / neither) learns both at once. An immediate alternative is to combine the existing baseline model with a direction model.
2. **The edge is paid by a bracket exit.** The label assumes selling at +X, selling at −X, or timing out at T. The current hold-until-take-profit strategy earns from volatility and will not show a direction edge (§5.2, §6.4). The sim needs a **bracket mode**: take-profit +X, a real stop at −X, and a maximum hold of T minutes, all matching the label. *(Update: §7 shows that move × direction already improves both existing exit modes, so a bracket mode is optional.)*
3. **Candidates:** ±25%/240m has the largest lift and matches the strategy's own take-profit (~0.27) and loss floor (0.25), but is less consistent across coins (76/90). ±15%/120m is the most consistent (87/91) and still clears costs. ±10%/60m is too small.

---

## 7. Move + direction scores with the existing exits (`direction_strategy_test`)

Setup:
* Nine confidence scores, each converted to its percentile on `tune`, so a confidence threshold of 0.80 means the top 20% of minutes.
* Simulation: the no-penalty engine in `stop` and `rebase` modes, 10 paired runs per arm, real against the same score shuffled within each coin. Validation only.
* All tests passed.

### 7.1 Models
* **3-class models** (balanced classes) on val_train:
  * move AUC 0.85 (moving vs quiet minutes);
  * within-coin direction AUC 0.595 (±15%) and 0.603 (±25%), close to the direction-only models (0.609). They keep the direction signal.
* **The move model and the direction models are strongly anti-correlated.** Spearman between baseline and direction-only is −0.70 (±15%) and −0.78 (±25%). The direction models rate the most volatile minutes as likely to fall first.

### 7.2 Strategy-free check (val_tune, all minutes, 3% cost per trade)
Net result per trade when buying in the score's top 5% and leaving at the bracket (quiet minutes counted at their return after T):

| Score | ±15%/120m bracket | ±25%/240m bracket | Top 5% up / down / neither (±25%) |
|---|---|---|---|
| baseline | −3.01% | −1.17% | 51 / 43 / 6% |
| fp15_dir | −2.45% | −1.70% | 7 / 1 / 92% |
| **fp15_combo** | **−0.83%** | **+2.38%** | 52 / 28 / 20% |
| fp15_3c_up | −0.75% | +2.51% | 49 / 24 / 27% |
| fp15_3c_ev | −1.32% | +0.89% | 31 / 11 / 58% |
| fp25_dir | −2.96% | −2.29% | 7 / 1 / 92% |
| fp25_combo | −0.74% | +2.33% | 50 / 26 / 25% |
| fp25_3c_up | −0.88% | +2.05% | 44 / 20 / 36% |
| fp25_3c_ev | −1.75% | +0.13% | 29 / 11 / 60% |

* **Baseline's top minutes split ~49/49 up/down: pure volatility.** At ±15%, its net is exactly minus the cost.
* **The direction-only models are useless alone.** Applied to all minutes, their top scores fall on quiet minutes (92% neither; move AUC 0.22–0.27).
* **Move + direction (`combo`, `3c_up`) is the first set-up with a positive result after costs.** In the top 5% at the ±25% bracket it nets about +2.0 to +2.5%. In the top 1% it nets +4 to +6%.
* `3c_ev` (P(up) − P(down)) leans towards quiet minutes and is weaker.

### 7.3 Strategy sims (validation)

| Score | Stop: real | vs shuffled (z) | vs baseline (z) | Rebase: real | vs shuffled (z) | vs baseline (z) |
|---|---|---|---|---|---|---|
| baseline | $41.84 | +1.99 (3.5) | — | $46.60 | +1.11 (2.4) | — |
| **fp15_combo** | **$45.68** | **+5.44 (8.6)** | **+3.84 (4.7)** | **$48.31** | **+2.21 (3.4)** | **+1.71 (3.9)** |
| fp15_3c_up | $42.90 | +3.65 (5.5) | +1.06 (1.2) | $47.78 | +2.08 (2.9) | +1.18 (1.9) |
| fp25_combo | $43.29 | +3.64 (4.8) | +1.44 (1.6) | $46.36 | +0.33 (0.8) | −0.24 (−0.7) |
| fp25_3c_up | $43.20 | +4.05 (3.0) | +1.36 (2.0) | $47.16 | +0.96 (1.7) | +0.56 (1.1) |
| fp15_3c_ev | $41.97 | +1.95 (1.7) | +0.12 (0.3) | $47.11 | +1.36 (2.3) | +0.51 (1.2) |
| fp25_3c_ev | $40.68 | +1.43 (2.1) | −1.17 (−1.4) | $47.65 | +1.30 (2.9) | +1.05 (1.7) |
| fp15_dir | $40.46 | +2.05 (1.3) | −1.38 (−1.4) | $46.55 | +0.82 (1.4) | −0.05 (−0.1) |
| fp25_dir | $40.81 | +1.43 (1.9) | −1.04 (−1.1) | $46.55 | +1.08 (1.5) | −0.06 (−0.1) |

* **`fp15_combo` (baseline × ±15%/120m direction model) beats the baseline in both exit modes.**
  * Significance (paired t-test over the 10 seeds, 9 df):
    * stop mode: p = 0.001, still p = 0.018 after a Bonferroni correction for 16 comparisons;
    * rebase mode: p = 0.004, but p = 0.058 after correction, just missing 0.05.
    * *Correction: an earlier version of this log said both survive the correction because it used a normal approximation; with only 10 runs the t-distribution applies, and only the stop-mode result survives.*
  * **What "significant" covers:** the 10 runs differ only in the optimizer seed, on the same 91 validation coins with the same models. So this shows robustness to optimizer randomness, **not** that the gain carries over to new coins. Which coins are in the sample is the larger uncertainty; the test set (or a bootstrap over coins) is needed for that.
  * It beats its shuffled control strongly (z 8.6 and 3.4).
  * $48.31 in rebase mode is the best validation score of any experiment so far.
* **For the first time, the optimizer chose a strict gate on its own:** a confidence threshold of 0.82 / 0.79, i.e. the top ~20% of minutes. It also cut end-of-data exposure (rebase 4.2% against 5.5% of trades).
* `rebase` (hold until take-profit) is still the better exit for every score.
* `fp25_combo` is clearly weaker than `fp15_combo` in rebase mode, so the gain depends somewhat on which direction model is used. `3c_up` is consistently positive but smaller.

### 7.4 Dual gate: a separate tuned threshold per model
Gate: buy only if move percentile ≥ a **and** direction percentile ≥ b, with both tuned along with all other parameters. The shuffle control moves both percentiles together.

| Score | Stop: real | vs baseline (z) | vs product (z) | Rebase: real | vs baseline (z) | vs product (z) |
|---|---|---|---|---|---|---|
| baseline | $41.84 | — | — | $46.60 | — | — |
| **fp15_combo (product)** | **$45.68** | +3.84 (4.7) | — | **$48.31** | +1.71 (3.9) | — |
| dual_fp15 | $43.53 | +1.69 (2.3) | **−2.15 (−3.5)** | $47.20 | +0.60 (1.0) | −1.11 (−1.5) |
| fp25_combo (product) | $43.29 | +1.44 (1.6) | — | $46.36 | −0.24 (−0.7) | — |
| dual_fp25 | $40.69 | −1.15 (−1.5) | −2.59 (−3.1) | $45.90 | −0.70 (−1.0) | −0.46 (−0.7) |

Tuned thresholds (median over runs) for `dual_fp15`:

| Mode | Move threshold | Direction threshold |
|---|---|---|
| Stop | 0.65 | 0.32 (range 0.19–0.41) |
| Rebase | 0.37 | 0.15 (range 0.03–0.39) |

No run sits at a bound.

* **The dual gate is worse than the product in both modes**; clearly so in stop mode (z −3.5 and −3.1).
* The optimizer used the direction model only as a weak filter: it cut the bottom 15–30% of direction scores, and in rebase mode sometimes almost switched it off.
* **Why the product wins:** it is a "soft AND". A minute with a very strong move signal and a moderate direction signal can still pass. A hard cut on each model throws away minutes just under either threshold, and the extra parameter adds tuning noise.
* **Conclusion:** keep the product combo with one threshold.

### 7.5 Boundary check of all tuned parameters
* **No parameter is held back by its search range.** Across every score, mode and arm, no parameter has ≥30% of runs within 2% of a bound. Confidence and take-profit were never at a bound.
* `loss_floor_consecutive_minutes` reached its upper bound (15) in only 1 of 10 runs, for a few scores.
* **Worth noting:** in **stop** mode, `loss_floor_pct` settles at 0.35–0.38 (median), near the top of its 0.15–0.40 range, though not at the bound (max 0.392). The optimizer prefers wide stops. If stop mode becomes the chosen exit, widen this range to ~0.50. In rebase mode it only triggers a re-base, so it doesn't matter there.

### 7.6 Conclusion
**A score built from move × direction improves the strategy with the existing exits.** The direction signal was real; it needed to be combined with the move model, not used instead of it.

Caveats:
* Validation was used both inside the runs and to pick this score among 8, so expect some optimism. Confirm on `test` exactly once.
* Most net results at the ±15% bracket are still negative, because the 3% cost is heavy.

---

## 8. Label pair search, both models (`label_pair_search`, coarse grid)

Setup:
* Move labels 10%/30m, 15%/40m, 20%/60m and 25%/90m, each with the dataset's pump definition.
* Direction labels ±10%/60m, ±15%/120m, ±20%/180m and ±25%/240m, each with **first-passage pumps**.
* 16 pairs screened without the strategy; the best 4, the baseline and the reference pair (10/30 × ±15/120) went to the sims.
* Sims: 10 paired runs, real against shuffled.
* New: a **coin bootstrap** of the gain over the baseline (5,000 resamples of the 91 validation coins).

Tests:
* TEST 2a: no pump feature uses the future (27M values checked).
* TEST 3: the dataset is reproduced exactly (4.4M rows).
* All other tests passed.

### 8.1 Models (val_train)
* **Move models:** AUC 0.77–0.79, with longer and larger labels slightly higher.
* **Direction models** (first-passage pump features), within-coin direction AUC:

| ±10%/60m | ±15%/120m | ±20%/180m | ±25%/240m |
|---|---|---|---|
| 0.584 | 0.605 | 0.620 | 0.620 |

These are about the same as with the dataset-style pump features (§6.6: 0.594 / 0.609 / – / 0.609).

### 8.2 Stage 1 screen (val_tune, net per trade after 3% costs, mean of the top 10% and 20% over three brackets)
* **Every pair beats every move model alone by about 2 points** (−0.3% to −0.9% against −2.5% to −2.7%). The direction model is what matters.
* **Pairs differ little.** The direction label matters more than the move label:
  * ±10%/60m is clearly worst (−0.76% to −0.93%);
  * ±15%/120m to ±25%/240m are close (−0.30% to −0.53%);
  * the move label changes the result by ≤0.2 points.
* Best pair in the screen: the reference pair, 10%/30m × ±15%/120m (−0.30%).

### 8.3 Stage 2 sims (validation)

| Score | Stop: real | vs baseline (p) | coin bootstrap 95% CI, P(≤0) | Rebase: real | vs baseline (p) | coin bootstrap 95% CI, P(≤0) |
|---|---|---|---|---|---|---|
| baseline 10/30 | $41.84 | — | — | $46.60 | — | — |
| **10/30 × ±15/120** (ref) | $44.14 | +2.30 (0.006) | −0.61..+5.01, 0.055 | **$47.22** | +0.62 (0.39) | −1.61..+2.84, 0.29 |
| 10/30 × ±20/180 | $42.93 | +1.09 (0.38) | −2.14..+3.97, 0.23 | $46.89 | +0.29 (0.70) | −2.16..+2.95, 0.42 |
| 25/90 × ±15/120 | $42.48 | +0.64 (0.37) | −2.04..+3.29, 0.31 | $46.95 | +0.35 (0.62) | −1.58..+2.45, 0.36 |
| **15/40 × ±15/120** | **$44.52** | +2.68 (0.010) | −0.31..+5.85, **0.042** | $46.86 | +0.26 (0.62) | −2.30..+2.54, 0.40 |

* **Stop mode:** every pair beats its shuffled control (p ≤ 0.023). The ±15/120 pairs gain about +2.3 to +2.7 over the baseline, borderline on the coin bootstrap (P(≤0) 0.04–0.06).
* **Rebase mode (current strategy):** no pair beats the baseline. Gains are +0.3 to +0.6, and the coin bootstrap intervals are wide around 0.
* **Compared with §7:** the same labels with the dataset-style direction pump features (`fp15_combo`) scored $45.68 (stop) and $48.31 (rebase). That was +3.84 and +1.71 over the same baseline, which is identical here. **The first-passage pump features did not help, and may be worse.** Not yet tested head-to-head.

### 8.4 Test-set confirmation (71 unseen test coins)
How it was done:
* Each saved run's parameters, all selected on val_tune, were re-scored on the test coins, with nothing re-optimised.
* Every re-scored run reproduced its saved validation score exactly, and the test-coin features reproduced `test.csv` exactly.
* Coin bootstrap: 5,000 resamples.

**Stop mode**

| Score | VAL | TEST | TEST vs baseline: diff (p), 95% CI, P(≤0) | VAL+TEST vs baseline: diff, P(≤0) |
|---|---|---|---|---|
| baseline | $41.84 | $34.13 | — | — |
| **fp15_combo (dataset-style pumps)** | $45.68 | **$37.93** | **+3.80 (0.006), +0.79..+6.76, 0.007** | **+3.82, 0.001** |
| 15/40 × ±15/120 (first-passage) | $44.52 | $36.92 | +2.79 (0.009), −0.21..+5.83, 0.034 | +2.73, 0.006 |
| 25/90 × ±15/120 (first-passage) | $42.48 | $39.12 | +4.99 (0.001), +2.09..+8.11, 0.000 | +2.54, 0.006 |
| 10/30 × ±15/120 (first-passage, reference) | $44.14 | $36.00 | +1.87 (0.109), −0.95..+4.85, 0.099 | +2.11, 0.025 |
| 10/30 × ±20/180 (first-passage) | $42.93 | $36.65 | +2.51 (0.048), −0.17..+5.12, 0.033 | +1.71, 0.055 |

**Rebase mode (current strategy)**

| Score | VAL | TEST | TEST vs baseline: diff (p), 95% CI, P(≤0) | VAL+TEST vs baseline: diff, P(≤0) |
|---|---|---|---|---|
| baseline | $46.60 | $40.15 | — | — |
| **fp15_combo (dataset-style pumps)** | $48.31 | **$42.40** | **+2.25 (0.175), −0.05..+4.62, 0.029** | **+1.95, 0.029** |
| 10/30 × ±15/120 (first-passage, reference) | $47.22 | $41.80 | +1.64 (0.081), −0.24..+3.23, 0.041 | +1.07, 0.085 |
| 10/30 × ±20/180 (first-passage) | $46.89 | $41.74 | +1.58 (0.226), −0.66..+3.63, 0.079 | +0.85, 0.159 |
| 15/40 × ±15/120 (first-passage) | $46.86 | $40.37 | +0.22 (0.842), −2.21..+2.39, 0.421 | +0.24, 0.374 |
| 25/90 × ±15/120 (first-passage) | $46.95 | $38.99 | −1.16 (0.395), −4.65..+1.55, 0.763 | −0.31, 0.623 |

**Head-to-head of the direction pump features** (first-passage minus dataset-style, same labels, seeds and coins):

| Mode | VAL | TEST | VAL+TEST | P(first-passage ≤ dataset-style) |
|---|---|---|---|---|
| Stop | −1.54 | −1.93 | −1.71 | 0.94 |
| Rebase | −1.10 | −0.60 | −0.88 | 0.89 |

Not significant at 5%, but **dataset-style is ahead on VAL and TEST in both modes**.

What this shows:
* **`fp15_combo` is confirmed on unseen coins.**
  * In stop mode it gains +$3.80 per coin on TEST (+11% over the baseline; probability of no gain 0.7%).
  * In rebase mode it gains +$2.25 per coin on TEST (+5.6%; probability of no gain about 3%, with the 95% interval just touching 0).
  * It was chosen in advance (last round's winner), so this is a clean confirmation, not a pick among many.
* **Other pairs are inconsistent between VAL and TEST.** For example, 25/90 × ±15/120 is the weakest on VAL in stop mode but the strongest on TEST, and it loses in rebase mode. That's noise, not a better label.
* **Rebase is still the better exit in absolute terms:** $42.40 against $37.93 on TEST.
* **Every test score is below its validation score,** baseline included ($40.15 against $46.60 in rebase), for two reasons: parameters were selected on validation, and the test coins may simply be harder. Expect live results nearer the TEST numbers.

**⚠ The test set is now spent.** Any further choice made after seeing these numbers can only be checked honestly on new data, i.e. coins collected after this dataset, or paper trading.

### 8.5 Conclusion
1. **The label region is settled:** move 10–15% within 30–40m, direction ±15% within 120m. The differences between neighbouring labels are smaller than the noise, so a finer grid is unlikely to pay off.
2. **Keep the dataset-style pump features for the direction model.** Confirmed by the head-to-head in §8.4.
3. **The coin bootstrap changed the picture.** On 91 validation coins alone, the gains were uncertain. With the 71 test coins added (§8.4), `fp15_combo` is confirmed in both modes.

---

## 9. Dump filter, model 3 (`dump_filter_experiment`)

Setup:
* Label: the 15-min median falls ≥ D below the current 15-min median within n hours, at any point ('any') or at the end ('end'). D ∈ {30%, 50%}, n ∈ {2, 6, 12, 24h}: 16 labels.
* Features: the dataset's 127 plus 13 new ones; every minute used.
* Stage 1: strategy-free screen on the combo's top 20% of minutes.
* Stage 2: the best 3 labels in the sims, as a veto and in the product, against the combo and against shuffled-dump controls. Validation only.
* All 6 tests passed. Models 1 and 2 and the combo's earlier runs were reproduced exactly.

### 9.1 Dumps are common, and models predict them well, mostly from volatility
* **Share of minutes followed by a dump:**
  * 30% within 24h ('any'): 36%;
  * 30% within 6h: 13%;
  * 50% within 24h: 15%.
* **Val AUC 0.72–0.91.** Short horizons are easiest (2h: 0.89–0.91; 24h: 0.72–0.83).
* **The top features are volatility** (60m, 120m, 1440m) and recent negative returns. The 13 new features carry only 7–16% of the gain.
* So the dump models largely say "volatile = risky". That clashes with the move model, which seeks out volatility because the strategy profits from it.

### 9.2 Stage 1 screen (val_tune; combo's top 20% = 235k candidate minutes: 67% good, 27% bad, 6% neither)
* **Every label blocks bad entries more often than good ones.** At 10% blocking, the best removes 15.7% of bad entries against 8.5% of good; random would be 10%/10%.
* **But discrimination is weak among the combo's candidates:** bad-vs-good AUC only 0.53–0.57.
* **Warnings come late and are often missed.** The median warning is 0.2–1.6h before the dump. Only 5–42% of dump episodes get any warning at the 10% threshold. False alarms run 0.05–3.3 per coin-day.

### 9.3 Sims (validation, 10 paired runs; vs combo: paired p, Bonferroni over 12 comparisons, coin bootstrap)

| Arm | Stop: VAL | vs combo (p_adj, P(≤0)) | vs own null | Rebase: VAL | vs combo (p_adj, P(≤0)) | vs own null |
|---|---|---|---|---|---|---|
| combo (reference) | $45.68 | — | — | $48.31 | — | — |
| veto end 30%/6h | $44.01 | −1.67 (0.36, 0.83) | −0.80 | $48.04 | −0.28 (1.0, 0.56) | +1.20 |
| veto any 30%/12h | $43.64 | −2.05 (0.47, 0.90) | −1.06 | $46.15 | −2.17 (0.44, 0.94) | −1.47 |
| veto end 30%/24h | $44.62 | −1.07 (1.0, 0.78) | −0.19 | $46.66 | −1.65 (0.45, 0.87) | −0.22 |
| product end 30%/6h | $40.45 | −5.23 (0.001, 0.97) | −0.41 | $47.90 | −0.41 (1.0, 0.60) | +1.71 |
| product any 30%/12h | $39.46 | −6.22 (0.000, 0.99) | −3.22 | $46.88 | −1.43 (0.53, 0.75) | +0.57 |
| **product end 30%/24h** | **$48.30** | +2.62 (1.0, 0.10) | **+3.79 (p 0.012)** | **$50.63** | **+2.31 (0.073, 0.030)** | **+3.30 (p < 0.001)** |

* **The veto doesn't work.** The optimizer pushed its threshold to 0.92–0.96, which means it only blocks the top 4–8% of dump-risk minutes, effectively switching it off. Even then it costs a little, and it never beats its shuffled control.
* **Products with short-horizon dump labels hurt,** badly in stop mode (−5 to −6). The confidence threshold drops to 0.22–0.32 and end-of-data losses triple.
* **One arm is promising: product with end 30%/24h.**
  * Rebase: +$2.31, run-to-run p = 0.006, 0.073 after correction; coin bootstrap P(≤0) = 0.03.
  * Stop: +$2.62, P(≤0) = 0.10.
  * It beats its own shuffled control strongly in both modes.
  * $50.63 is the best validation score so far.
  * But it was picked out of 12 comparisons, and it doesn't pass the Bonferroni correction.
* **No arm reduces end-of-data losses,** the problem this filter was meant to solve. In rebase mode, end-of-data P&L is −$549 for the combo and −$543 to −$678 for the filters.

### 9.4 What this means
1. **An entry filter can't fix the end-of-data losses.** Those positions are usually opened before any dump warning exists. Warnings come 0.2–1.6h ahead and are often missed, and in rebase mode an open position is held to the end regardless. Avoiding those losses needs an **exit** that sells when dump risk is high. The short-horizon dump models (AUC 0.89–0.91) suit that job, but it's the new exit mechanic that was deferred.
2. **The end 30%/24h product probably works as a longer-horizon direction signal** ("will the price be lower in 24h?"), not as death avoidance: its end-of-data losses aren't lower. It complements the ±15%/120m direction model.
3. **The evidence for it is borderline** (best of 12, not significant after correction). It needs an independent check before adoption.

---

## 10. Can models 1 + 2 replace model 3 with its features? (`feature_transfer_test`)

Four set-ups on the same seeds and coins, validation only:
* **A:** the current combo.
* **B:** models 1 + 2 retrained with the 13 new features.
* **C:** A × (1 − P(dump end 30%/24h)).
* **D:** B × (1 − P(dump)).

Tests:
* TEST 1: retrained without the new features, models 1 and 2 reproduced the originals (max difference 1.2e-7, float rounding).
* TEST 3: A and C reproduced their saved runs exactly.

**Models (val_train)**

| Model | AUC (val_train) | Within-coin AUC | Share of learning from the new features |
|---|---|---|---|
| move, current | 0.7706 | 0.7192 | — |
| move, + new features | 0.7707 | 0.7201 | 5% |
| direction, current | 0.6164 | 0.6089 | — |
| direction, + new features | 0.6175 | 0.6147 | 8% |

* **The new features barely change models 1 and 2.**
* The four scores rank minutes almost identically (Spearman 0.94–0.98).

**Sims (validation, paired diff, run p, coin bootstrap P(≤0))**

| Comparison | Stop | Rebase |
|---|---|---|
| B vs A (features help models 1 + 2?) | −0.75 (p 0.51, P 0.64) | −0.17 (p 0.81, P 0.58) |
| C vs A (model 3, reference) | +2.62 (p 0.095, P 0.10) | +2.31 (p 0.006, P 0.030) |
| **B vs C (features replace model 3?)** | **−3.38 (p 0.011, P 0.91)** | **−2.48 (p < 0.001, P 0.95)** |
| D vs C (features add on top of model 3?) | −3.48 (p 0.012, P 0.98) | −0.23 (p 0.76, P 0.60) |
| D vs A | −0.86 | +2.09 (p 0.020, P 0.062) |

**Conclusion:**
* **The features can't replace model 3, and they add nothing to models 1 and 2.**
* Model 3's value comes from **its label** (will the price be ≥30% lower in 24h?), not from its new features. That matches the dump models getting only 7–16% of their learning from those features (§9.1).
* Models 1 and 2 keep their current inputs.
* Open question this raises: does model 3 need the 13 new features at all? A model 3 trained on the dataset's 127 features only would remove them from the pipeline entirely.

---

## 11. Finer dump-label grid (`dump_label_refine`)

Setup:
* 14 labels: end {20, 30, 40%} × {18, 24, 36, 48h}, plus any 30%/24h and 30%/36h.
* All on the same rows: the last 48h of each coin unlabelled, which is **24h more than in §9**.
* Same 127 + 13 features and training settings.
* Screened on the product (net result of its top 10/20%). The best 3, plus the previous winner retrained here, went to the sims.
* All 5 tests passed, and the combo reproduced its saved runs.

### 11.1 Models and screen
* **AUC rises with the drop size and falls with the horizon:** 40%/18h 0.786, 20%/48h 0.602. That's the same volatility-driven pattern as §9.1.
* **The screen gains are small and irregular** (+0.38 to −1.20 points), with no smooth pattern over drop or horizon. The 30% row reads +0.38 / −0.86 / +0.15 / −0.34 across 18/24/36/48h.
* The previous winner (end 30%/24h) ranks only 11th of 14.
* The screen looks dominated by noise.

### 11.2 Sims (validation, 10 paired runs)

| Product with | Stop: VAL | vs combo (p, P(≤0)) | vs own null | Rebase: VAL | vs combo (p, P(≤0)) | vs own null |
|---|---|---|---|---|---|---|
| combo (reference) | $45.68 | — | — | $48.31 | — | — |
| end 30%/18h | $46.61 | +0.93 (0.35, 0.31) | +2.31 (p 0.018) | $49.65 | +1.34 (0.10, 0.15) | +2.10 (p 0.002) |
| end 30%/36h | $47.03 | +1.35 (0.27, 0.17) | +0.93 (p 0.43) | $49.49 | +1.18 (0.20, 0.12) | +2.10 (p 0.022) |
| end 40%/48h | $43.41 | −2.27 (0.03, 0.82) | +0.21 | $47.98 | −0.33 (0.63, 0.57) | −0.38 |
| **end 30%/24h (previous winner, retrained)** | **$43.46** | **−2.22 (0.010, 0.85)** | +1.33 | $49.30 | +0.99 (0.14, 0.30) | +2.82 (p 0.004) |

* **The previous winner does not replicate.** With the same label, features and settings, but trained on the rows that leave the last 48h unlabelled instead of 24h:

  | Mode | §9 | Now |
  |---|---|---|
  | Stop | +2.62 | **−2.22** |
  | Rebase | +2.31 | **+0.99** |

  A small change in the training rows flipped the result. The §9 gain was mostly luck and instability.
* **No label passes the verdict.** 30%/18h and 30%/36h beat the combo in both modes, but only by +0.9 to +1.35. Their coin-bootstrap P(≤0) is 0.12–0.31 and p_adj is 0.8–1.0.
* The products often beat their **own shuffled control** (the dump score carries *some* information), but **not the combo**.
* **Training times of ~0–6 seconds per model** suggest early stopping after very few trees. That would make model 3 unstable. Check `best_iteration` in `dump_label_refine/models/*.json`; a hyperparameter search might stabilise it.

### 11.3 Conclusion
* **Model 3 as an entry signal gives no robust gain.** Differences of about ±$1–2 between set-ups are within the noise of these 91 validation coins.
* **Keep the confirmed combo (models 1 + 2) as the final entry signal.**
* Model 3's remaining potential is as an **exit trigger** (§9.4), and possibly after hyperparameter tuning.

---

## 12. Hyperparameter search, models 1 + 2 (`model_hpo_search`)

Setup:
* 6 Optuna studies (move/direction × XGBoost/LightGBM/CatBoost), 50 trials each (15 random), with median pruning (none for CatBoost on GPU).
* Ranked on val_train: move by pAUC (FPR ≤ 10%), direction by within-coin AUC.
* Top 3 kept per study; 27 combos (9 library pairings × ranks 1–3), 10 paired runs, both modes, against the current combo.
* All tests passed: same rows and features as the originals; every kept model reloaded with identical predictions; the current combo reproduced its saved runs.

### 12.1 The search
| Model | Original | XGBoost best | LightGBM best | CatBoost best |
|---|---|---|---|---|
| move (pAUC) | 0.5990 | 0.5994 | 0.5998 | 0.5990 |
| direction (within-coin AUC) | 0.6089 | 0.6183 | 0.6188 | 0.6159 |

* **Move: no gain from tuning or from switching library.** The original settings came from an earlier search on this exact label, and all three libraries reach the same ~0.599.
* **Direction: about +0.01 on val_train,** partly selection optimism (best of 50 on the same 38 coins; standard error ≈ 0.011).
* **Random trials did score worse** (move down to 0.593; direction down to 0.555). The score surface is flat because early stopping rescues poor settings; it isn't a local optimum.
* Pruning stopped 22/50, 3/50, 13/50 and 16/50 trials for XGBoost/LightGBM (move, direction); CatBoost on GPU 0/50.

### 12.2 Sims (validation, vs the current combo)
**By library pairing (move × direction), ranks 1–3 pooled (30 runs each):**

| Move × direction | Rebase vs current (P(≤0)) | Stop vs current (P(≤0)) |
|---|---|---|
| xgboost × xgboost (re-tuned) | +0.28 (0.41) | −0.85 (0.78) |
| xgboost × lightgbm | +0.59 (0.30) | −0.91 (0.74) |
| xgboost × catboost | −0.39 (0.64) | −0.89 (0.77) |
| lightgbm × xgboost | +0.22 (0.40) | −1.27 (0.88) |
| lightgbm × lightgbm | +0.26 (0.41) | −1.23 (0.79) |
| lightgbm × catboost | −0.86 (0.76) | −2.37 (0.96) |
| catboost × xgboost | −0.12 (0.55) | −0.99 (0.78) |
| catboost × lightgbm | −0.18 (0.56) | −0.46 (0.64) |
| catboost × catboost | −0.86 (0.77) | −2.20 (0.96) |

* **No combo beats the current one robustly.**
  * Best single combo: XGBoost × XGBoost rank 2, rebase +1.59 (p 0.013, but p_adj 0.70; bootstrap P(≤0) 0.045). Its own ranks 1 and 3 are −0.13 and −0.61, so it's a lucky model, not a better set-up.
  * **In stop mode every tuned pairing is worse than the current combo** (−0.46 to −2.37).
  * CatBoost as the direction model is consistently the worst.
* **The +0.01 within-coin AUC gain of the tuned direction models does not carry over** to the strategy.
* **Model-to-model variation within one pairing (±1.5) is as large as the differences between pairings,** another view of the noise floor (§11.3).

### 12.3 Conclusion
* **The current XGBoost models 1 + 2 stay.** Library choice and tuning don't matter for this data; the models are at the ceiling of what the features and labels contain.
* **Further gains must come from information:** better features (Dataset V9 and feature engineering).

---

## 13. Dataset V9 (`Dataset_creationV9`, built from V8)

**Window convention: the only allowed windows, and how they're written in feature names:**

`1m, 2m, 3m, 5m, 10m, 15m, 30m, 1h, 2h, 4h, 8h, 12h, 24h, 36h, 48h, 3d, 7d`

* Names use these exact tokens: e.g. `volatility_1440m` → `volatility_24h`, `volatility_60m` → `volatility_1h`, `return_6h` → its snapped window.
* **Every feature window is snapped to the nearest allowed window,** nearest on a ratio (log) scale, so 6h → 8h, 3h → 4h, 20m → 15m. *(Confirmed 2026-10-28.)*
* **Features longer than 7 days are deleted:** `recent_pump_rate_30d` and `pump_rate_7d_vs_30d`.
* The windows affected are not only the 6h pump rate. 6h is used by `return_6h`, `drawdown_from_6h_peak`, `price_stability_6h`, `launch_momentum_6h`, `pump_velocity_6h` and `liquidity_trend_6h`. 3h is used by `excess_buying_pressure_3h`, `pump_density_3h`, `pump_acceleration_3h` and the pump clustering features. 20-bar windows appear in several others.

**Changes in V9:**
1. `estimated_sol_1pct_impact`: **the current method, with only the look-ahead fixed** (confirmed: no other methods in V9). The whole-coin volume floor and median/MAD clip are computed causally, from data up to each minute. All dependent features are recomputed. The alternative calculation methods are added later, on top of V9.
2. All windows snapped and renamed (above); features longer than 7d deleted.
3. **The second pump-feature set** for the direction model (pump = +15% within 2h), with its windows snapped the same way, plus the first-passage label (±15%/2h, all 3 classes).
4. **Leakage (truncation) test on ALL features,** not just the pump features.
5. **Vectorise** the feature computation only where the result equals V8 within floating-point tolerance (checked).
6. Not now: one shared feature function for offline and live; feature elimination (later).

**Splits:** the splitter re-derives the splits from each coin's length and pump rate (share of label = 1), using tertile cut-offs and a fixed seed. If V9 changes the labelled rows or the label, pump rates shift slightly, some coins can change category, and the shuffle then assigns different coins. So **unchanged splitter code does not guarantee the same coins.** *Decision (2026-10-28):* **V9 uses hard-coded splits from the existing `coin_assignments.csv`**, with a check that every coin keeps its split. Later versions go back to the splitter code as it is.

**As built (2026-10-28):**
* Output in `dataset_v9/`: `per_coin/` (one parquet per coin), `splits/` (split CSVs + `coin_assignments.csv`) and `reports/` (leakage test, V8 comparison, `feature_mapping_v8_v9.csv`).
* **148 features** (V8: 127):
  * 83 of V8's features unchanged, 19 of them only renamed;
  * 42 changed (snapped windows, causal liquidity, bug fixes);
  * 2 deleted;
  * 23 new (`_p15_2h`).
* **Labels:**
  * `label_10pct_30m` (as V8);
  * `label_fpclass_15pct_2h` (2 = up first, 1 = down first, 0 = neither; NaN without a full 2h window);
  * `label_meta_minutes_to_end` (use ≥ 240 to reproduce the experiments' training rows; never a feature).
* **Rows exactly as V8** (row 120 to 30 minutes before each coin's end).
* **Liquidity:** the volume floor and the median/MAD clip are recomputed every 15 minutes from data up to that minute. Internal windows are unchanged (EW τ = 1440); the alternative methods come later.
* **Bugs found and fixed:**
  * V8's `drawdown_from_6h_peak` and `drawdown_from_24h_peak` both used a 120-row (2h) peak;
  * launch features were only available from minute 420; each is now available from its own horizon.
* **Unchanged-value rewrite:** `delta_v_acceleration` uses a 2h window ending 1h ago instead of 3h − 1h (identical values; tested).
* **Splits:** hard-coded from `dataset_splits/coin_assignments.csv`. V8's splitter is in the notebook but switched off (`USE_HARDCODED_SPLITS`).
* Tested on synthetic data: all five tests pass (leakage on all features; 83 unchanged features equal V8 exactly; second pump set equals the V8-era ±15%/120m features where nothing changed; splits identical).

**Real-data run (2026-10-28):** all 5 tests passed.
* 362 coins, built in 39 min.
* Leakage test: 5.3M values.
* 83 unchanged features equal V8 on all 5.5M rows.
* The second pump set matches on 4.4M rows.
* Splits: 38 / 71 / 71 / 91 / 91 coins with V8's exact rows.
* **`estimated_sol_1pct_impact` changed by a median 10%** (correlation 0.986). The liquidity trends changed most: correlation 0.56–0.86.

**Downstream notebooks must change for V9:**
* read from `dataset_v9/splits/`;
* the sim uses **`price_position_1h`** (was `price_position_60m`);
* the direction model uses the `_p15_2h` pump features in place of the 23 original pump-feature columns;
* training rows: `label_meta_minutes_to_end ≥ 240`.

**Check after building V9:** retrain the current combo (models 1 + 2, original settings) on V9 and re-simulate. Features that were only renamed must equal V8 exactly (test). Snapped and causal-liquidity features change, so the sims can't match exactly; they must match V8 **within noise** (paired runs + coin bootstrap) before feature engineering continues.

**Then: the 1% price-impact experiment.**
* Vary only the feature version the models see. **The sim's execution estimate stays the current method** (V9, causal fix) for every variant.
* Variants: mean vs median, the lengths of its two windows, and later different estimators (e.g. Amihud, Kyle's lambda, Roll / Corwin–Schultz).
* For each variant: recompute all dependent features, retrain the current combo, simulate **20 runs** (up from 10), and pick the best.

---

## 14. V9 check: V9 vs V8 (`v9_check`)

Setup:
* **Arm A:** V8 models and features, V8 1% impact for execution.
* **Arm B:** new V9 models and features (incl. V9's 1% impact as a feature), V8 1% impact for execution only.
* **Arm C:** everything V9.
* 20 paired runs per arm and mode, validation only.

Tests:
* The V9 direction model trained on exactly V8's rows and labels.
* Arm A reproduced the saved V8 runs.
* The price-position filters are identical.

**Models (val_train, same rows):**

| Model | V8 | V9 |
|---|---|---|
| move (AUC / pAUC) | 0.7706 / 0.5990 | 0.7707 / 0.5982 |
| direction (AUC / within-coin AUC) | 0.6164 / 0.6089 | 0.6203 / 0.6169 |

The combo scores rank minutes almost identically (Spearman 0.968).

**Sims (validation, 20 paired runs):**

| Mode | A (V8) | B (V9 features) | C (V9) | B − A (p, coin 95% CI) | C − B (execution) | C − A |
|---|---|---|---|---|---|---|
| Rebase | $47.95 | $48.09 | $47.51 | +0.14 (0.73, −1.77..+2.13) | −0.59 (0.25) | −0.44 (0.30) |
| Stop | $45.97 | $43.18 | $42.75 | **−2.80 (0.001, −5.27..−0.59)** | −0.42 (0.44) | **−3.22 (< 0.001)** |

**Reading:**
* **Rebase: no detectable difference.** The means are within $0.14. The ±$1.50 equivalence test still "fails", because the coin-bootstrap interval is about ±$2 wide even when the means agree. *The ±$1.50 margin was too tight for 91 validation coins, so this test could not pass even for identical set-ups (a design flaw in the check).*
* **Stop: V9 is clearly worse** (−$2.80, p = 0.001, bootstrap P(≤0) = 0.994).
* **The execution change (C − B) is small** in both modes (−$0.4 to −$0.6, not significant).
* **Open question: features or model luck?** §12 already showed that every retrained model was worse than the current V8 combo in stop mode. That included XGBoost × XGBoost re-tuned on V8 features, down by $0.5 to $2.4. So the V8 models may simply be a fortunate draw for stop mode, and *any* retrained model, even on identical features, might lose about $1–2 there.
* **Needed before V9 is accepted:** a **model-seed control**. Retrain both models on V8 features AND on V9 features with a few different random seeds, then compare V9 with V8 *averaged over the seeds*. If V8-feature retrains drop as much in stop mode, the V9 gap is model luck, not the features.

---

## 15. Model-seed control (`seed_control`)

Setup:
* Models 1 + 2 retrained on V8 features and on V9 features with XGBoost seeds 1, 2 and 3, plus seed 42 (the `v9_check` models; V8 seed 42 = the original models).
* V8 sizing in every arm; 20 paired runs per model, both modes, validation only.

Tests:
* Retraining V8 with seed 42 reproduced the original model **bit for bit**.
* The reused seed-42 arms reproduced `v9_check` exactly.

**Models (val_train):**
* **Move:** the same in V8 and V9 (AUC 0.770–0.771).
* **Direction:** within-coin AUC **consistently higher in V9**: 0.614–0.619, against 0.609–0.613 for V8.

**Sims (validation, mean of 20 runs per model seed):**

| Mode | V8 seeds 42 / 1 / 2 / 3 | V8 mean | V9 seeds 42 / 1 / 2 / 3 | V9 mean | V9 − V8 | Model-as-unit p | Coin 95% CI |
|---|---|---|---|---|---|---|---|
| Rebase | 47.95 / 47.95 / 48.96 / 47.58 | $48.11 | 48.09 / 47.43 / 47.62 / 47.23 | $47.59 | −0.52 | 0.20 | −1.74..+0.74 |
| Stop | **45.97** / 44.06 / 44.54 / 44.91 | $44.87 | 43.18 / 43.96 / 43.08 / 43.75 | $43.49 | −1.38 | 0.034 | −3.39..+0.47 |

**Reading:**
* **The original V8 model was a lucky draw in stop mode.** At $45.97 it's above every V8 retrain ($44.06–44.91). About **half of the −$2.80 gap in §14 was model luck.**
  * *Consequence for earlier results:* every stop-mode comparison against "the current combo" (§11, §12) was made against this lucky model, so new models looked about $1–1.5 worse than they really were. In rebase mode the original is ordinary (within its retrains).
* **A small residual remains,** in the same direction in both modes: −$0.52 (rebase) and −$1.38 (stop).
  * In stop mode it's significant with the model as the unit (p = 0.034, but only 4 models per side).
  * Across coin samples it isn't conclusive (P(≤0) = 0.92 in stop, 0.79 in rebase).
* **Model-to-model spread from the seed alone:** sd $0.4–0.6 (rebase) and $0.4–0.8 (stop).
* **Likely cause of the residual: the look-ahead V9 removed.** V8's liquidity estimate used each coin's whole-history volume distribution, including its future (whether it later dies, for example). The liquidity features changed most in V9 (correlation with V8 0.56–0.86). If part of V8's edge came from that leak, a small drop in V9 is the honest price of removing it, not a flaw in V9. The snapped windows and the bug fixes could also contribute; only an ablation would separate them.

**Conclusion:**
* **Accept V9.** It's leak-free; it matches V8 in rebase mode within noise; and the stop-mode residual is small, partly luck-corrected, and plausibly the removed leak.
* **From now on compare set-ups with several model seeds.**

---

## 16. Baseline on the new V9 split (`baseline_v9`)

The V9 dataset was rebuilt from scratch with a **full re-split**: train 56, val_train 39, tune 94, val_tune 92, test 92 coins (373 in total). These shares match 15 / 10 / 25 / 25 / 25%, so the old 71 train coins became 56, giving 865k move-model training rows instead of 1.08M.

Setup:
* The current combo (move 10%/30m × direction ±15%/2h), V9 features and V9's 1% impact for sizing.
* Model seeds 42, 1 and 2, each with 20 paired sim seeds, both modes. The move model alone was also run.
* Validation only. Tests: splits disjoint and matching `coin_assignments.csv`; no leakage.

**Models (val_train):**
* Move: AUC 0.771–0.772, pAUC 0.600–0.601.
* Direction: within-coin AUC 0.616–0.617.

So the models are slightly better than on the old split, despite fewer training coins. The combo scores of the three seeds rank minutes almost identically (Spearman 0.988).

**The baseline (validation, mean over 3 model seeds × 20 runs):**

| | Rebase | Stop |
|---|---|---|
| **Combo** | **$48.03** (seeds 48.06 / 48.05 / 47.99; model sd 0.04) | **$45.86** (45.42 / 45.71 / 46.45; model sd 0.53) |
| Move model alone | $46.24 (model sd 0.26) | $42.60 (model sd 0.06) |
| Combo − move alone | about +1.8 | about +3.3 |
| Tuned confidence / take-profit / trades per coin | 0.70 / 0.24 / 16.4 | 0.78 / 0.24 / 19.6 |

**Reading:**
* **The new split gives almost the same baseline as the old one** (old split, seed-averaged: rebase $48.11, stop $44.87). Results carry over across coin samples, as expected with this many coins.
* **The combo again beats the move model alone,** by about $1.8 (rebase) and $3.3 (stop), as on the old split.
* **Model-to-model variation is small on this split** (sd $0.04–0.53).
* **The coin-bootstrap interval of the *level* is wide** (rebase $36.8–60.4): which coins are in the sample moves the absolute number a lot. That's why every comparison from here is **paired** (same model seeds, same sim seeds, same coins), which makes differences far more precise than levels.

**This is the reference for everything that follows on this split.** Saved in `baseline_v9/`: the models, the tune reference distributions, all runs with per-coin results, and `baseline_summary.json`.

---

## 17. 1% price-impact variants (`impact_variants`)

Setup:
* 33 ways to compute `estimated_sol_1pct_impact`, each with its 23 dependent features recomputed and both models retrained (seed 42).
* 20 paired runs per mode against the baseline's seed-42 runs (current method: EW geometric mean, 24h). The sim's sizing is unchanged.

Tests:
* TEST 1: every estimator is causal and the rolling statistics are exact.
* TEST 2: the recompute reproduced V9's 23 dependent features exactly on all 4.3M rows.

**Results (validation, variant − baseline):**

| Variant | Rebase | Stop | Coin 95% CI (stop) |
|---|---|---|---|
| kyle_3d | +0.55 | **+1.83** (p 0.002, p_adj 0.11) | −0.24..+4.11 |
| median_3d | +0.83 | +1.31 (p 0.010, p_adj 0.66) | −1.26..+4.27 |
| amihud_1h | +0.71 | +1.33 (p 0.028) | −1.32..+4.00 |
| vwmean_3d | **+1.17** (p 0.018, p_adj 1.0) | +0.83 | −1.00..+2.63 |
| ew_3d | +0.33 | +0.89 (p 0.070) | −0.73..+2.64 |
| … most others | −1.0 to +0.6 | −1.3 to +1.0 | all include 0 |
| worst: kyle_24h, roll_4h, cs_24h, ar_4h | −0.9 to −1.0 | −0.9 to −1.3 | |

**Reading:**
* **No variant is clearly better.** After correcting for 66 comparisons, nothing is significant (best p_adj 0.11), and every coin-bootstrap interval includes 0.
  * The best result is about what chance alone produces: with 33 single-seed variants and a model-seed spread of ~$0.5 in stop mode (§16), the luckiest one would be expected around +$1.5.
* **A weak hint: long (3-day) windows.** Four of the top five are 3-day windows (kyle_3d, median_3d, vwmean_3d, ew_3d). But trim_3d, p25_3d and amm_7d are not better, so it's not consistent.
* **The spread estimators are worse as a replacement.** Corwin–Schultz, Abdi–Ranaldo and Roll are mostly −$0.5 to −$1.3, and their rank correlation with the current estimate is negative (−0.09 to −0.45): they measure something else (spread). Possible as *extra* features later, not as a substitute.
* **Roll 24h is degenerate:** 96% NaN, because the price-change autocovariance is rarely negative over 24h.
* **Estimate diagnostics:** every impact-type estimator ranks minutes like the current one (Spearman 0.84–0.98).
* **Compute time is the same for all** (156–195 s for all coins; dominated by the shared sample preparation), so speed can't break a tie.
* **Model scores barely move:** move pAUC 0.599–0.601; direction within-coin AUC 0.612–0.621.

### 17.1 Confirmation (`impact_confirm`): the round-1 gains disappear
The 5 best variants were retrained with two **new** model seeds (1, 2), 20 runs each, both modes, and compared with the baseline's seeds 1 and 2. Criteria were fixed in advance.
* All three tests passed, and every rebuilt estimate equaled round 1's exactly.

| Variant | Rebase: s42 → new seeds → all 3 | Stop: s42 → new seeds → all 3 | Crit. 1 | Crit. 2 |
|---|---|---|---|---|
| kyle_3d | +0.55 → −0.06 → +0.14 | +1.83 → **−1.30** → −0.26 | · | · |
| median_3d | +0.83 → −0.93 → −0.34 | +1.31 → −1.16 → −0.33 | · | · |
| amihud_1h | +0.71 → −0.16 → +0.13 | +1.33 → −1.05 → −0.25 | · | · |
| vwmean_3d | +1.17 → +0.24 → +0.55 | +0.83 → −0.40 → +0.01 | · | ✓ |
| ew_3d | +0.33 → −0.13 → +0.02 | +0.89 → −0.84 → −0.26 | · | · |

* **Every round-1 advantage vanished with fresh models.** In stop mode all five turned negative (−$0.4 to −$1.3). Over 3 seeds every variant is within ±$0.55 of the baseline, with p between 0.24 and 0.98.
* This is the **winner's curse** in action: the best of 33 single-seed results were mostly lucky models, as expected (§17). The fixed-in-advance confirmation caught it.
* **Verdict (criteria): no finalist holds up → keep the current method.**

**Conclusion:**
* **The current method is kept and locked** (§17.1).
* If the 3-day hint is worth checking: confirm ew_3d (the smallest change: the same method with a 3-day time constant), median_3d, vwmean_3d and kyle_3d with model seeds 1 and 2 against the baseline's matching seeds.

---

## 18. Extra windows + duplicate removal (`feature_windows`)

Setup:
* 22 new features fill the window gaps of existing families, as a separate add-on CSV.
* Exact duplicates removed.
* Models 1 + 2 with 5 seeds against the baseline's same 5 seeds, rebase mode, 20 runs per seed.

Tests: no future data, rows aligned, and the baseline reproduced `baseline_v9`.

* **Duplicates on the real data:** only `days_since_first` (= `minutes_since_launch` / 1440). `volume_surge_vs_liquidity` and `volume_to_liquidity` are identical *except* when the 30-minute mean volume is zero (then `volume_ratio_current` is defined as 1). That happens often enough on real coins that they aren't exact duplicates: a near-duplicate for the elimination step.
* **Models (val_train):** move pAUC 0.6007 → 0.6005; direction within-coin AUC 0.6149 → 0.6137. No change.
* **Sims (rebase, 5 seeds × 20 runs):** baseline **$48.10**, new **$48.10**, difference **+0.00** (per seed −0.78 to +0.58; p = 0.995; coin 95% CI −0.85..+0.82). **Exactly neutral.** Rule verdict: not improved.
* **Feature importance (5 seeds averaged):**
  * **Direction model:** the new windows are the second most important family (27% of SHAP; shuffling them costs 0.029 within-coin AUC; only the ±15%/2h pump history matters more, at 0.034). `drawup_from_4h_low` is #2 overall, `volume_mean_ratio_24h_3d` #7, `price_position_4h` #8. **`hour_sin` is #1** (11% of SHAP; the time family costs 0.017 when shuffled).
  * **Move model:** `volatility_4h` is #1 overall; the short returns (1m, 2m, 10m) are near the bottom.
  * **Liquidity, volume, candles, return distribution and buying pressure** cost ≈ 0 when shuffled, in both models: elimination candidates.
  * **The new features are used a lot but add no score:** they **replace** information the models already had from correlated features. A model that loses a feature finds the same signal elsewhere.
* **About the 5 models:** they share **identical hyperparameters** (the `XGBoost_train` settings). Only XGBoost's random seed differs, which changes the random row and column sampling while the trees grow. They measure model-to-model chance, not different model settings.

**Decision:** the new features are kept for now (user's choice); feature elimination decides later.

---

## 19. New feature families (`feature_families`)

Setup:
* 42 features in 7 new families (order flow, VWAP distance, wicks, down-event history, past first-passage outcomes, trend quality, range volatility), as an add-on.
* Tested against the current set (V9 − duplicate + 22 windows) with the same 5 seeds, 20 runs each, rebase mode.
* All tests passed, including no future data and the reference reproduced.

**Models (val_train):**
* **Move:** unchanged (pAUC 0.6005 → 0.6004).
* **Direction:** within-coin AUC **0.6137 → 0.6174, higher for all 5 seeds** (+0.0005 to +0.0054). A real, if small, model-level gain.

**Sims (rebase, 5 seeds × 20 runs):** reference **$48.10**, new **$47.23**, difference **−$0.87**.
* Per seed −1.69 to +0.04; **4 of 5 seeds worse**.
* Paired p = 0.050; coin 95% CI −2.02..+0.10.
* **Worse**, at the edge of significance. Rule verdict: not improved.

**Importance (5 seeds averaged):**
* **Direction model:**
  * Used new families: VWAP distance (shuffling costs 0.0055 within-coin AUC; `vwap_distance_4h` #6, `vwap_distance_1h` #10), down-event history (0.0025), range volatility (0.0017) and past first-passage outcomes (`fp_up_share_24h` #4).
  * Barely used: order flow, trend quality and wicks (≤ 0.0004).
* **Move model:** `fp_decided_rate_24h` is the #1 feature (the share of recent brackets that hit a barrier is a volatility-regime measure); dump density 4h / 24h and Parkinson volatility are also used.
* Still the most important for the direction model: ±15%/2h pump history (0.027), time (`hour_sin` #1, 0.018), the windows (0.017).

**The key lesson, now seen three times:**

| Experiment | Direction within-coin AUC | Sims |
|---|---|---|
| Tuning (§12) | +0.01 | no gain |
| V9 (§14–15) | +0.008 | slightly worse |
| These families | +0.004 on all seeds | −$0.87 |

**A better within-coin direction AUC on all decided minutes does not translate into a better strategy, and can make it worse.** The strategy only acts on the combo's top ~30% of minutes, after the move gate, the price-position filters and the exits. The direction model's average ranking quality over *all* decided minutes is the wrong yardstick for that, so improving it is not the goal.

**Next, before more features:** find a model-level yardstick that predicts the sims. Candidates:
* the strategy-free net result of the combo's top 20–30% on `tune`;
* the direction AUC restricted to the minutes the move model selects.

Check it against the ~35 models already simulated (V9 check, seed control, baselines, windows, families), whose val scores are known.

---

## 20. Yardstick check: which model-level metric predicts the sims? (`yardstick_check`)

Setup:
* 15 model pairs already simulated on this split (baseline, windows, families × 5 seeds), each with its rebase sim score (VAL, 20 runs). Sims range $46.34–48.63.
* 22 candidate metrics: model AUCs on val_train, and strategy-free trade results on tune / val_tune for the combo's top 20% / 30% of minutes that pass the sims' filters.
* Nothing trained or simulated.

| Metric | Spearman with the sim (95% interval) | Pearson |
|---|---|---|
| **tp_net_val_tune_top20** (+24% take-profit or exit after 750 min, top 20%) | **+0.74 (+0.37..+0.91)** | **+0.84** |
| br_net_val_tune_top20 (±25% / 4h bracket) | +0.70 (+0.27..+0.91) | +0.85 |
| tp_net_val_tune_top30 | +0.68 (+0.23..+0.88) | +0.80 |
| tp_net_coin_val_tune_top20 (per-coin average) | +0.55 (−0.00..+0.87) | +0.64 |
| move_pauc (val_train) | +0.38 (−0.20..+0.75) | +0.24 |
| dir_within_movetop30 (val_train) | +0.12 | +0.26 |
| strategy-free metrics on **tune** | −0.70 to +0.21 | |
| **dir_within (the yardstick used so far)** | **−0.65 (−0.87..−0.28)** | −0.51 |

**Reading:**
* **The old yardstick points the wrong way.** Models with a higher direction within-coin AUC got *lower* sim scores, and the interval stays below zero. A direction model that ranks *all* decided minutes better gets worse at the top of the combo, where the strategy trades. That explains §12, §14–15 and §19.
* **A strategy-free trade result on val_tune tracks the sims well.** The top-20% net result ranks the feature sets like the sims: families clearly last ($0.0324 vs $0.0378 / $0.0364). Baseline and windows are tied in the sims ($48.10 each), and the metric puts them $0.0014 apart.
* **The same metrics on tune don't track the sims** (some are negative). What matters is how the combo's top minutes do on the evaluation coins themselves, not on other coins.
* **Caveats:**
  * The val_tune metric uses the sim's coins, so it's a **fast stand-in for the sim** (seconds instead of ~10 min per model), not an independent check of generalisation.
  * It has only 15 pairs and 3 feature sets behind it. It should first **predict the next experiment's sim result** before it's trusted to replace sims.

**What this means for the next steps:**
* Judge features and models by the strategy-free top-20% result (screening), then confirm finalists with sims.
* The models are trained on goals that don't match what the strategy rewards (the move label, and the direction label on all decided minutes). The most promising lever now is **aligning training with the strategy**:
  * train the direction model only on minutes the move model would select;
  * or train a model directly on the trade outcome (reaches +24% within 750 minutes, or the trade's net result).

  Screen them with the new yardstick, then confirm with sims.

---

## 21. Strategy-aligned training (`aligned_training`)

Setup:
* Current feature set; 5 seeds; 8 arms × 5 seeds × 20 runs, rebase. Every arm was simulated.
* The screen's predictions were saved before the sims.
* Tests passed: no label leak or coin overlap; the ref reproduced `feature_windows`.

**Models (val_train):**
* The direction models trained on candidate minutes rank candidates **no better** than the current direction model (dirA50 0.594–0.598 vs ref 0.592–0.599 within-coin AUC; dirA30 slightly worse).
* `out24` predicts its own label well (AUC 0.705–0.709).
* `outReg` correlates weakly with the trade result (Spearman 0.17–0.18).

**Sims (rebase, VAL, paired with the ref's same seed; ref = $48.10):**

| Arm | Mean diff | Seeds (42 / 1 / 2 / 3 / 4) | p | p (Holm) | Coin 95% CI |
|---|---|---|---|---|---|
| **A50** (direction on move top 50% candidates) | **+1.17** ($49.27) | +1.76 / +0.13 / +1.10 / +0.81 / +2.02 | 0.026 | **0.079** | −0.69..+3.17 |
| A30 | −0.19 | | 0.58 | 0.60 | |
| B_three | −0.48 | | 0.30 | 0.60 | |
| Breg_replace | −1.04 | | 0.001 | 0.007 | |
| AB | −1.15 | | 0.020 | 0.079 | |
| Bcand_replace | −1.85 | | 0.001 | 0.006 | |
| B_alone | −2.00 | | 0.002 | 0.010 | |
| B_replace | −2.19 | | 0.000 | 0.004 | |

**Reading:**
* **A50 is the first change since the combo itself that improves the strategy on every model seed.** $49.27 is the best average on this split. The tuned confidence is high and stable (~0.80–0.90). After the Holm correction it **just misses** the rule fixed in advance (0.079 > 0.05), so formally there's no candidate.
  * Note that its *model AUC* did not improve; once again only the sims show the difference.
* **A wider candidate pool beats a narrower one** (A50 vs A30); the top 30% probably leaves too few training examples.
* **Outcome models make it worse, every time.** That holds as a factor, as a replacement and alone; four arms are significantly worse after Holm. Predicting "+24% within 750 min" well (AUC 0.71) does not help the strategy.
* **The yardstick failed this test:**
  * Spearman with the sims was +0.29 across arms and +0.53 across arm × seed; the direction was right for 4 of 8 arms.
  * It ranked Breg_replace first (+1.39%), but that arm was worse in the sims. A model trained on exactly the yardstick's trade games it.
  * Among the A arms it was weakly right (A50 above A30, both small).
  * **Lesson:** a screen metric is only valid for models that are not trained on that metric's own target.

**Next:**
* **Confirm A50 with new seeds** (5–9) against the ref with the same new seeds. It's a single arm named in advance, so no multiple-testing correction: adopt if the mean diff > 0 and the paired p < 0.05.
* Optional diagnostics in the same run:
  * **A70**, a wider pool;
  * **A_elig**, all tradable minutes without the move filter, to see whether the gain comes from the move filter or from training only on tradable minutes.

---

## 22. A50 confirmed (`aligned_confirm`)

Setup:
* **Part 1 (decides):** A50 and the current combo, both retrained with **new seeds 5–9**, 20 runs each, rebase. Rule fixed in advance: adopt if the mean diff > 0 and the paired p < 0.05.
* **Part 2 (diagnostic):** A40, A60 and A70 on seeds 42–4, against the reused reference.
* Tests passed: the reference training reproduced `feature_windows`' seed-42 model exactly (max difference 0), and the reference sim reproduced exactly.

**Part 1, new seeds 5–9 (rebase, VAL):**

| Seed | A50 | Current combo | Diff |
|---|---|---|---|
| 5 | $49.01 | $49.26 | −0.25 |
| 6 | $48.67 | $47.07 | +1.60 |
| 7 | $48.67 | $46.95 | +1.72 |
| 8 | $48.83 | $47.51 | +1.32 |
| 9 | $48.87 | $47.77 | +1.10 |
| **Mean** | **$48.81** | **$47.71** | **+1.10** (4/5 seeds, p = 0.036, coin 95% CI −0.09..+2.39) |

* **Verdict by the rule: A50 CONFIRMED → adopted.**
* **All 10 seeds together:** +$1.13, better on 9 of 10, p = 0.0008.
* A50 is also more stable across seeds: $48.67–49.01 on the new seeds, against $46.95–49.26 for the current combo.

**Part 2, gain vs candidate-pool size (seeds 42–4, vs the reference $48.10):**

| Pool | A30 | A40 | A50 | A60 | A70 |
|---|---|---|---|---|---|
| Mean | $47.91 | $49.04 | $49.26 | $49.22 | $49.11 |
| Diff | −0.19 | +0.94 | +1.17 | +1.12 | +1.01 |
| Seeds better | 1/5 | 5/5 | 5/5 | 4/5 | 4/5 |

* **A broad plateau from 40% to 70%; 30% falls off.** The choice of pool size isn't sensitive as long as it's ≥ 40%; 50% (confirmed) sits in the middle.
* **Model AUC again can't see the gain:** the A models rank their candidate minutes about as well as the reference direction model (within-coin AUC 0.589–0.600 vs 0.588–0.597).

**Why it likely works:** the old direction model learned "up first or down first" over *all* decided minutes, most of which the strategy never buys (filtered out, or too low on the move model). Trained only where the strategy can act, it learns what separates good from bad entries *among real candidates*.

**Caveats:**
* All judged on the val_tune coins. The coin interval (−0.09..+2.39) only just includes 0.
* The test set is spent, so the honest out-of-sample check is **new coins or paper trading**.
* Stop mode was not checked; the 2026-10-31 decision allows a stop-mode check for finalists if wanted.

---

## 23. Dynamic (coin-relative) buy thresholds on A50 (`dynamic_thresholds`)

Setup:
* 7 systems × (product / gates) + A50 with families, 3 seeds × 20 runs, rebase, up to 2,500 trials, against A50 (same seeds, A50 = **$49.35**).
* All tests passed: the rolling statistics are exact and causal; the dynamic view reproduced A50 exactly; the calibration coins were disjoint.
* An idea-2 cache bug crashed one trial midway; it was fixed and the run resumed (results unaffected).

| Arm | VAL | Diff | Per seed | p |
|---|---|---|---|---|
| **dm_prod** (partial de-meaning, product) | $50.39 | **+1.04** | +0.07 / +1.69 / +1.35 | 0.17 ← candidate |
| cal_gate | $50.15 | +0.80 | −0.05 / +1.41 / +1.04 | 0.21 |
| tier_gate | $50.14 | +0.79 | −0.33 / +0.92 / +1.79 | 0.33 |
| tier_prod | $50.12 | +0.77 | −0.57 / +1.70 / +1.18 | 0.38 |
| i1abs_prod | $50.10 | +0.75 | −0.77 / +1.65 / +1.38 | 0.43 |
| cal_prod | $49.82 | +0.47 | +1.01 / −0.24 / +0.63 | 0.33 |
| **i2_prod** (past true positives, product) | $49.81 | +0.46 | +0.08 / +0.30 / +0.99 | 0.24 ← candidate |
| i1z_prod, i2_gate, dm_gate | ≈ $49.4 | ≈ 0 | | |
| pct_prod | $48.84 | −0.51 | | |
| **fam** (A50 + families) | $48.48 | **−0.87** | −0.58 / −0.96 / −1.07 | **0.028** |
| i1z_gate, pct_gate, i1abs_gate | $46.7–48.2 | −1.2 to −2.6 | | |

**Reading:**
* **The gains are small, nowhere near the tier classifier's ~30%.** The best arms add about $0.5–1.0 (1–2%), and none is significant on 3 seeds. Either A50 already absorbs much of the coin-level difference (its direction model only learns from candidate minutes), or the classifier's earlier 30% was partly overfitting (there was no validation set then).
* **The systems that do help agree on one shape.** Tier, de-meaning and moving-mean products all tune to:
  * **windows of about 75–120 minutes**, not 5–20;
  * **partial adjustment**: λ ≈ 0.4–0.5 (de-meaning) and b ≈ 0.5 (move) / 0.6–0.8 (direction) for the tier, i.e. **about half relative to the coin's 1–2h mean**;
  * floors of about 0.48 (move) and 0.35 (direction);
  * rising edge mostly off (median 0.31–0.37, below the 0.5 switch).

  Different systems landing on the same shape suggests it's real, if small.
* **Products beat gates again**, except for cal_gate and tier_gate. The moving-mean and percentile gates are clearly worse (−1.2 to −2.6). The same lesson as §7.4.
* **The families also hurt A50** (worse on 3/3 seeds, p = 0.028): dropped for good.
* **Screening rule** (better on all 3 seeds and mean > 0): candidates are **dm_prod** (+1.04) and **i2_prod** (+0.46). Tier, i1abs and cal_gate had one negative seed.

**Next:** confirm dm_prod and i2_prod with new seeds 5–9 against A50's seeds 5–9, with a rule fixed in advance (Holm over the 2 arms).

---

## 24. A50 optimised on tune only (`a50_tune_only`, partial: 46 of 100 runs)

Setup: 2,500 trials on tune only, no val checks or early stopping; the best-on-tune parameters are run once on val. Same sampler seeds as the val-selected A50 runs, compared run for run.

| Seed | Tune-only: tune | Tune-only: val | Val-selected: val |
|---|---|---|---|
| 42 (20 runs) | $53.84 | **$45.69** | $49.79 |
| 1 (20 runs) | ≈ $52.8 | **$44.79** | $48.76 |
| 2 (6 runs) | ≈ $53.3 | **$43.68** | $49.50 |

**Reading:**
* **The usual val scores include a selection bonus of about $4 (~8–9%).** A50 optimised on tune alone earns about **$45** on unseen coins; ~$45 is the more honest live expectation. Comparisons between set-ups share the bonus and stay fair.
* **The optimiser overfits tune:** the best tune trial is almost always late (trials 1,800–2,500), and tune ends ~$8 above val. Part of the val-selected procedure's advantage is that val early stopping picks an earlier, less overfitted trial; the learning curve (summary cell) shows where val peaks.

---

## 25. Tier classifier strategy on the new 7-split V9 (`tier_strategy`)

Setup:
* New V9 split with clf_train / clf_val; tier classifier trained by the user with `tier_classifier_v9` (move model only).
* Move model alone (static confidence) vs tiers: a tuned confidence per tier 0–4, the tier smoothed to the most common tier over W minutes (W 3–30, ties → higher).
* The same 3 move models (seeds 42, 1, 2), 10 runs each, paired; rebase, val-selected.
* Tests passed: no coin overlap; the smoothing is exact; equal tier confidences reproduced the static gate exactly.

| Move model | Alone | Tiers | Diff | Runs better | p (runs) |
|---|---|---|---|---|---|
| seed 42 | $47.34 | $47.54 | +0.21 | 5/10 | 0.75 |
| seed 1 | $46.71 | $48.88 | +2.16 | 9/10 | 0.013 |
| seed 2 | $47.60 | $48.75 | +1.15 | 7/10 | 0.21 |
| **All 3 (model as the unit)** | **$47.22** | **$48.39** | **+1.17 (+2.5%)** | 3/3 models | **0.17** |

**Reading:**
* **The old ~30% is not reproduced.** Tiers add about $1.2 (2.5%), positive for every move model but not significant. That's the same size as the best dynamic thresholds on A50 (§23: +$0.5–1.0). **Two independent ways of adapting to the coin's regime both give ~1–3%.**
* **The tuned confidences per tier show no structure:** tier 0 0.40, tier 1 0.555, tier 2 0.546, tier 3 0.522, tier 4 0.46 (medians; the 25–75% ranges overlap widely). If the tiers carried strong information, the thresholds would differ systematically by tier. The smoothing window settles at ~17 min.
* **Likely reasons the old gain was much larger:**
  1. **Look-ahead in the old features.** V8's liquidity estimate (and the derived features) used whole-coin statistics, and some launch / drawdown features had bugs (fixed in V9, §13). A regime classifier is exactly the kind of model that profits from future information about a coin.
  2. **No validation set then.** The extra tier parameters were judged on the coins they were tuned on, or with less protection against selection; we've now measured ~$4 of selection bonus even with a val set (§24).
  3. **A weaker baseline then:** a single buy model and a fixed take-profit. The current features (pump history, windows) and the tuned exits already absorb part of the regime information.
* **Practical conclusion:** regime adaptation is worth ~$1 here, not ~$14. If it's pursued, the simpler route (a dynamic threshold such as dm_prod, a few parameters and no extra model) is preferable to the tier pipeline (a labelling simulation plus a classifier).

---

## 26. Tiers on the raw scale + the old design (`tier_strategy`, `sim_raw/`)

Setup:
* Model 1 (move) gated on its raw probability; the same 3 move models (42, 1, 2) × 10 runs, rebase, money score.
* Test 4 passed: identical tier settings reproduced the single-tier strategy exactly, through the plain engine too (1,603 trades).

| Set-up | Seed 42 | Seed 1 | Seed 2 | **Mean** |
|---|---|---|---|---|
| Move alone, everything tuned (static) | 46.95 | 46.05 | 47.42 | **46.80** |
| Tiers, free confidence per tier, rest tuned | 46.71 | 46.63 | 47.58 | 46.98 |
| `fixed_static`: one confidence, rest **fixed to the labels' live parameters** | 44.03 | 43.08 | 42.75 | 43.29 |
| **`fixed_tier`**: confidence per tier **inside its label range**, rest fixed as above | **49.05** | **45.42** | **49.03** | **47.83** |
| `tier_old`: the original (7 parameters per tier) | 45.43 | 44.91 | 47.28 | 45.87 |

**Reading:**
* **The tiers DO work, under the parameters the labels were made with:** fixed_tier − fixed_static = **+$4.55 (+10.5%)**, positive for all 3 move models (+5.01 / +2.34 / +6.28; paired p = 0.06 over 3 models).
  * The tuned confidences rise with the tier exactly as designed: tier 1 0.57, tier 2 0.71, tier 3 0.79, tier 4 0.87, dead 0.89.
  * This is the user's point: the labels define the optimal confidence **for one fixed strategy**, so the tiers are informative when the strategy trades with those settings.
* **fixed_tier is the best set-up in the notebook:** +$1.03 above the fully tuned static strategy, with only 6 tuned parameters (5 confidences + window).
* **With freely tuned strategy parameters the tiers add nothing** (46.98 vs 46.80). The optimiser moves to another operating point (other exits and filters) at which the labels' "best confidence" no longer applies.
* **Tuning everything per tier (tier_old, 38 parameters) is worse** (−$0.93 vs tuned static). Most likely overfitting on ~70 tune coins; the per-tier exits barely differ (take-profit 0.18–0.22, loss floor 0.24–0.29).
* **Raw vs percentile scale: no meaningful difference** (raw runs ≈ $0.4–1.0 lower than the earlier percentile runs, within noise).
* The old ~30% likely came from the same mechanism: tiers vs a single-tier strategy with **fixed** parameters. Here that comparison gives +10.5%; the rest of the old figure plausibly came from look-ahead and the missing val set (§25).

**Next:**
* **Confirm fixed_tier** with new move-model seeds (rule fixed in advance), against fixed_static and the tuned static.
* **Align the labels with better fixed parameters:** relabel with the median tuned parameters of the static strategy (instead of the old live ones), then fixed_tier with those same fixed parameters. If the tiers' value comes from consistency between labels and strategy, this should combine both gains.

---

## 27. Tiers v2 (`tier_v2`): A50 on the new split, relabelled tiers, tier sims

Setup: 3 move models (42, 1, 2) × 10 runs, rebase, money score. Tests passed (disjoint splits; the fixed-settings view reproduces the plain engine exactly).

**PART 1, A50 on the new 7-split V9:** **$51.24** vs move alone $46.80 → **+$4.43 (+9.5%)**, all 3 seeds (+3.50 / +5.31 / +4.48, p = 0.014). The direction model's within-coin AUC on candidates is 0.577–0.583. **A50 is confirmed again on new coins, with a larger gain than on the old split.**

**PART 2, relabelled with the median tuned settings, sweep 0.20–1.00, full curves kept:**
* Cases: 1 dead 13%, 2 all-lost 21%, 3 clean 56%, 4 rescued 10–12%.
* Best confidence: move median 0.50 (10–90%: 0.24–0.82), combo median 0.69 (0.26–0.97). The labels now cover the range the strategy trades in.
* **Oracle gain** (true tiers): move up to +$2.12 per window (+71% of the single-confidence profit, 5 tiers); combo up to +$0.89 (+24%). More tiers → more oracle gain.
* **Realised gain** (classifier tiers): combo designs +$0.6–1.4, move designs −$0.24 to +$0.34; macro-F1 0.52–0.74.
  * Caveat: the realised baseline is a confidence chosen on clf_train, the oracle's is clf_val's own best, so realised can exceed oracle. These numbers only rank.

**PART 3, sims (fixed settings = the labelling medians):**

| Set-up | s42 | s1 | s2 | **Mean** |
|---|---|---|---|---|
| Move alone, tuned | 46.95 | 46.05 | 47.42 | 46.80 |
| **A50, tuned** | 50.45 | 51.36 | 51.89 | **51.24** |
| fixed_static_move | 45.42 | 44.32 | 45.00 | 44.91 |
| fixed_tier move N5 quantile keep_all | 48.65 | 48.03 | 47.76 | 48.14 |
| fixed_static_combo | 46.66 | 47.08 | 49.17 | 47.64 |
| fixed_tier combo N4 quantile drop_2_4 | 47.85 | 49.17 | 48.09 | 48.37 |
| fixed_tier combo N4 quantile keep_all | 47.39 | 49.38 | 47.50 | 48.09 |

**Reading:**
* **The tiers work for the move model alone:** +$3.23 over fixed_static_move, all 3 seeds (p = 0.007), confirming §26 with the new labels. The tuned confidences rise with the tier as designed (0.25 / 0.37 / 0.47 / 0.65 / 0.82).
* **On top of the combo the tiers add little:** +$0.74 / +$0.45, not significant, one seed negative. The direction model and the tiers mostly capture the **same** information ("when is buying good right now"), so once the direction model is in, the tiers have little left to add.
* **Fixing the settings to the medians costs a lot:** fixed_static_combo $47.64 vs A50 tuned $51.24 (−$3.6); move −$1.9. The per-parameter median is not a good joint setting, because the parameters interact.
* **A50 with everything tuned stays the best system** ($51.24); the best tier set-up is ~$2.9 below it.
* Keeping or dropping cases 2/4 makes little difference in the sims (combo: drop +0.74, keep +0.45).

**What could still make tiers beat A50:** two untested combinations:
1. **Bounded tier confidences + freely tuned settings** (tiers keep their ranges, the exits are tuned). The free-confidence version lost the tier value (§26), but the bounded one wasn't tried with tuned settings.
2. **Labels made with the best single A50 run's settings** (a coherent joint set) instead of the per-parameter medians.

If neither beats A50, drop the tiers and move on (death checks).

---

## 28. The last tier test (`tier_v2` PART 4)

Setup: same 3 move models × 10 runs, rebase, money score.
* (A) **bounded_tuned**: per-tier confidence ranges kept, all other settings tuned.
* (B) labels remade with the settings of the **best single A50 run by tune score** (seed 2, run 8: take-profit 0.265, stale 255 min, loss floor 0.197, re-entry 0.048, price-position caps 0.66 / 0.81), then fixed_static / fixed_tier / bounded_tuned.

| Set-up | s42 | s1 | s2 | Mean | vs A50 |
|---|---|---|---|---|---|
| **A50, tuned** | 50.45 | 51.36 | 51.89 | **51.24** | |
| bounded_tuned, move N5 (median labels) | 49.94 | 49.00 | 48.44 | 49.13 | −2.11 |
| bounded_tuned, combo N4 (median labels) | 46.58 | 47.52 | 48.78 | 47.63 | −3.61 (p 0.005) |
| fixed_static_combo_best | 49.03 | 50.97 | 52.31 | 50.77 | −0.47 (n.s.) |
| fixed_tier_combo_best N4 | 48.89 | 48.84 | 48.87 | 48.86 | −2.37 |
| bounded_tuned_combo_best N4 | 51.76 | 51.47 | 51.45 | 51.56 | **+0.33** (+1.31 / +0.11 / −0.44; p 0.59) |

**Reading:**
* **No arm beats A50 on all 3 seeds.** The best (bounded_tuned with best-run labels) is +$0.33, within noise, one seed negative. **By the rule fixed in advance: the tiers are dropped; A50 stays.**
* **Bounding the confidences to label ranges with tuned settings hurts** for the median-based labels (−$2.1 / −$3.6): the ranges confine the optimiser where the tuned exits want something else.
* **The best-run settings are a good joint set:** fixed_static_combo_best ($50.77) is level with fully tuned A50 (−$0.47, n.s.), against $47.64 for the per-parameter medians. **The medians, not fixing as such, caused the −$3.6 in §27.** A fixed configuration from one coherent run performs about as well as re-tuning everything. That's useful for live trading (stable settings; less optimiser overfitting, §24).
* Under the best-run settings, tiers with fixed settings are **worse** than one confidence (−$1.90). Once the direction model and good settings are in place, the tiers carry no extra information the strategy can use.
* The best-run labels have many more rescued windows (case 4: 26%), from its short stale time (255 min) and low re-entry threshold.

**Conclusion of the tier line (§25–28):**
* For the move model alone, tiers add about +$3 (10%) with fixed settings.
* On top of the A50 combo, they add nothing reliable. The direction model trained on candidate minutes captures the same "is now a good moment" information.

---

## 29. Death diagnostics on A50 (`death_diagnostics`)

Setup:
* All 30 A50 runs of `tier_v2` re-simulated on tune (71 coins) and val_tune (70); 62,286 trades.
* Test 1 passed: every run reproduced its saved val score exactly.
* "Collapse" = close ≥ 50% below its 60-minute high.

**Profit per coin per day:**

| | PnL / coin (mean) | Median | Pooled $/coin/day | %/day of $20 | Coins in profit | ≥ 10× | Nearly wiped |
|---|---|---|---|---|---|---|---|
| tune | $22.93 | $0.00 | **$2.19** | 10.9% | 47% | 4% | 8% |
| val_tune | $37.63 | $1.51 | **$3.59** | 17.9% | 52% | 9% | 8% |

* **The profit is very concentrated:** the median coin earns about nothing; a minority of coins makes the money.
  * val final capital per coin: 10% $4.9, 25% $13.1, **50% $23.7**, 75% $71, 90% $189.
* Coins average 10.5 days of data.
* Val includes the ~$4 selection bonus (§24). A fair live estimate is roughly **$2–3.5 per coin per day**, mostly from a few strong coins.

**Loss anatomy:**
* **A 50%-in-an-hour crash is normal for these coins, not death:** 92% of tune coins have one.
* **About 90% of all trades come after a coin's first crash, and they are where the profit is:** +$2,374 / +$3,188 net per run.
  * "Stop after the first crash" would cost all of that.
  * Coins with a crash earn far more (+$25 / +$41) than coins without (+$1 / +$2): volatility is the opportunity.
* **The coin's death itself costs almost nothing:** positions open at the end of data lose $4–6 per run (≈ 0%). The death-point crop and the stale exit already handle it.
* **The real loss bucket: positions open while a LATER crash happens:** ~61 trades per run, **−$879 (tune) / −$549 (val) net per run**, 17–23% of all losses, 34 big losses per run.
* **Selling at the crash minute doesn't help** (+$32 tune, −$163 val): by detection the drop has happened, and the rebase strategy often recovers.
* **A whole-coin veto would need 82–89% precision** (68–82% with medians): winners average +$47 / +$77, losers −$10.

**Warning signals** (tune; "a crash starts within H hours", base rate 0.8% / 2.1% / 6.0% for 2 / 6 / 24 h):

| Signal | AUC | Precision on top 1% (2 / 6 / 24 h) |
|---|---|---|
| **rug_recent** (largest 5-min fall in the last 6 h) | 0.75–0.81 | 7% / 12% / 30% (lift 5–9×) |
| liq_drain | 0.53–0.57 | 4% / 12% / 24% |
| after_collapse | 0.53 | ≈ base rate |
| drawdown_ath, hours_since_ath, vol_decay, zero_volume | **0.20–0.37 (inverted)** | ≈ 0% |

* **Crashes come right after new highs, not from slowly dying coins:** a small drawdown and few hours since the high mean MORE crash risk. That's the pump-and-dump cycle, exactly where the strategy buys. A check on these signals would block the best trades.
* **The best signal (a recent rug candle) is 5–9× better than chance but only 7–30% precise:** far from the 82–89% a coin veto needs. A temporary block could still pay, since its cost is a few missed trades rather than a whole coin, but only the sims can say.

**Conclusion:**
* Classic death checks (vetoing or permanently stopping coins) are **not supported**: what looks like death is mostly the volatility the strategy profits from, and the actual deaths are already cheap.
* The only target with money in it is **positions open during a later crash (−$550 to −$880 per run)**, and the only promising signal is a recent rug candle. A small, targeted test is possible:
  * "after a rug candle, block new entries for X minutes" (X and the candle size tuned);
  * a random-block control.

  Expected gain modest; it may simply be skipped.

---

## 30. Rug-candle block + dump-risk veto (`dump_risk`)

Setup:
* On A50 (3 move models × 10 runs, rebase, tuned), each arm with a control that blocks as much at other moments.
* Tests passed: disjoint splits; the dump label on hand-made paths; with nothing blocked every arm reproduces A50 exactly; the rug mask equals brute force.

**The dump-risk model** (trained on A50 candidate minutes; target: price −40% before +22.8%, within 520 min):
* Dump rate 13.7% of candidate minutes.
* **AUC 0.71–0.72; volatility alone 0.61; inside volatility deciles 0.68–0.69.** The model clearly sees more than volatility.
* Offline veto table:

| Veto (riskiest) | Precision | Dumps caught | Good entries lost |
|---|---|---|---|
| 1% | 42% | 3% | 0.7% |
| 5% | 40% | **14.5%** | 3.5% |
| 10% | 31% | 22.6% | 8.0% |

**Sims (val, vs A50 $51.24):**

| Arm | Mean | vs A50 | vs its control |
|---|---|---|---|
| rug_block (fall ≥ 35%, block 48 min) | 50.43 | −0.80 | −0.87 vs rug_random (n.s.) |
| rug_random (control) | 51.31 | +0.07 | |
| **risk_veto** (vetoes ~6% of minutes) | 48.49 | **−2.74** (3/3 seeds, p 0.008) | **−2.70 vs shuffled (p 0.044)** |
| risk_shuffled (control) | 51.19 | −0.04 | |
| rug_and_risk | 47.64 | −3.59 | |

**Reading:**
* **The risk model works as a predictor but hurts as a veto.** Vetoing its riskiest ~6% of minutes loses $2.7, and **the same veto with shuffled scores costs nothing**. So it's not "trading less" that hurts; it's that the **minutes it flags are worth more than average**.
  * A dump of −40% before +23% often happens in exactly the moments with the biggest upside. The strategy's loss floor, re-entries and rebase recover enough of the dumps that avoiding those moments costs more than it saves.
  * An offline "dumps caught" figure is not money: a trade that dumps first can still end in profit.
* **The rug-candle block has no value** (−$0.87 vs its control, n.s.); the tuned block settles on rare, very large falls (≥ 35%) for under an hour.
* **Conclusion for the death / dump line (§29–30):** keeping the strategy out of risky moments doesn't pay; risky moments are where the profit is. **Not adopted.**

**A remaining idea (not run):** use the risk score to change **how** a trade is managed rather than **whether** it is opened, e.g. tighter loss floor / take-profit, or a smaller size, when the risk at entry is high (2 buckets, ~4 parameters). The engine already supports per-bucket exits (the per-tier mechanism).

---

## 31. Risk-conditioned trade management (`risk_manage`)

Setup:
* Every trade still taken; trades entered at high risk (the top x% of the dump-risk score) get their own exits (hr_exits), a smaller size (hr_size) or both, against shuffled-risk controls.
* A50, 3 move models × 10 runs.
* Tests passed: nothing flagged, or flagged with normal settings, reproduces A50 exactly.

| Arm | Mean | vs A50 ($51.24) | vs its control |
|---|---|---|---|
| hr_exits | 50.96 | −0.28 (n.s.) | +0.30 vs shuffled (n.s.) |
| hr_exits_shuffled | 50.66 | −0.57 | |
| hr_size (0.69× for the riskiest 12%) | 48.54 | **−2.70** (3/3 seeds) | −1.14 vs shuffled |
| hr_size_shuffled | 49.67 | −1.56 | |
| hr_both | 48.27 | −2.97 | |

**Reading:**
* **No arm helps.** Separate exits for risky trades are neutral; a smaller size for them **costs** money, more than shrinking random trades does. Risky trades are the ones with the most upside, so shrinking them shrinks the profit.
* **Telling detail:** given its own exits, the optimiser gives high-risk trades a **higher** take-profit (0.241 vs 0.223) and a **longer** stale time (694 vs 618 min), and a slightly tighter loss floor. It wants to let risky trades run **more**, not cut them. That points to the "let winners run" idea (trailing / partial take-profit).
* **The dump-risk line is closed (§29–31):** a genuine predictor (AUC 0.72, 0.69 within volatility deciles), but not avoiding, vetoing, shrinking nor separately managing risky trades improves the strategy. **Not adopted.**

---

## 32. Three levers on A50 (`levers`)

Setup:
* 3 move models × 10 runs (ens: one model, 30 runs), rebase, tuned, against A50 ($51.24).
* Tests passed: moveA trained only on tradable minutes; trailing with zero time reproduced A50 exactly.

| Arm | s42 | s1 | s2 | Mean | vs A50 |
|---|---|---|---|---|---|
| trail (no sale at TP; trailing stop) | 42.43 | 43.18 | 41.07 | 42.23 | **−9.01** (3/3, p 0.010) |
| partial (share sold at TP, rest trails) | 45.03 | 46.43 | 46.76 | 46.07 | **−5.16** (3/3, p 0.001) |
| moveA (move model on tradable minutes) | 50.91 | 51.03 | 50.88 | 50.94 | −0.30 (n.s.) |
| ens (average of 3 move × 3 direction models) | | | | 51.60 | +0.37 (Welch p 0.50) |

**Reading:**
* **Letting winners run loses badly.** Even with the optimiser choosing a wide trail (≈ 40%) and the shortest possible trailing time (68–92 min), trailing costs $5–9.
  * After the +25% take-profit the price usually reverses: the take-profit sells near local tops.
  * The strategy captures long runs by **selling and re-entering** (rebase), which a trailing position blocks while it is open.
  * The fixed take-profit is not the bottleneck it looked like.
* **The seed ensemble adds nothing:** the 3 models' combos rank minutes almost identically (Spearman 0.997), so averaging them changes almost nothing (+$0.37, run spread unchanged).
* **The move model on tradable minutes adds nothing** (−$0.30). Unlike the direction model, the move model doesn't gain from seeing only tradable minutes.

**Where this leaves the system (§24–32):**
* A50 (move × direction on candidate minutes, tuned) is the only robust improvement since the combo itself: confirmed three times, +$4.4 over the move model alone on the new split.
* Tiers, dynamic thresholds, death / dump checks (veto, block, size, exits), trailing exits, ensembling and further alignment all failed to beat it reliably.
* The val level is ~$51 (≈ $47 without the selection bonus, ≈ $2–3.5 per coin per day).
* **The system looks to be at a plateau** for this data and these model types.

---

## 33. Different strategies with the two models (`strategies`)

Setup: 8 arms on A50, 3 move models × 10 runs, rebase, up to 2,500 trials. Tests passed (the rules on hand-made data; every rule switched off reproduces A50 exactly).

| Arm | Mean | vs A50 ($51.24) |
|---|---|---|
| **signal_exit** | **110.82** | **+59.59** (+60.06 / +60.14 / +58.55) ← candidate by the rule |
| signal_exit_shuffled (control) | 104.47 | +53.23 |
| tp_scaled | 51.15 | −0.08 |
| signal_hold_shuffled | 50.92 | −0.32 |
| pullback | 50.30 | −0.94 |
| entry_filters | 48.31 | −2.93 |
| signal_hold | 47.61 | −3.62 |
| combined | 45.53 | −5.71 |

**Reading:**
* **The huge gain comes mostly from the exit MECHANISM, not from the models.** The shuffled control (signal exits on scores shuffled within each coin, i.e. nearly random timing) earns +$53; the real model scores add **+$6.35 on top** (3/3 seeds).
* **Why:** in rebase mode, A50 only ever closes a position at the take-profit (measured from the re-anchored price) or at the end of the data. **The loss floor and the stale time never sell; they only re-anchor or start a race.** So a losing position stays open, sometimes for a very long time, and its capital is locked.
  * The signal exit is the **first rule that sells at market before the take-profit.** With the tuned thresholds (direction < 0.26, move < 0.14 after ≥ 46 min), it acts like a **time stop**: if the take-profit isn't hit within roughly an hour, sell and redeploy.
  * The tuned strategy then trades ~3× as often (31–47 trades per coin), with a higher take-profit (0.32) and a low entry threshold (0.18–0.25).
* **No bug found:** P&L is measured from the original entry price and the shares are unchanged by rebases; next entries come only after the exit; scores are causal; the off-switch reproduces A50.
  * **Caveat:** ~3× more trades means 3× more exposure to the fill / slippage assumptions (1.5% per side modelled). The execution check (fills at the next minute's open / close) is running.
* **Everything else is ≤ A50:** holding on the signal (signal_hold) is worse than its own shuffled control; entry filters, pullback and the combined arm lose $1–6; scaled exits are neutral.

### 33.1 Execution check: the gain is NOT a fill artefact
Every saved run re-evaluated on val under other fill assumptions (A50 included):

| Fills | A50 | signal_exit | shuffled | signal_exit − A50 |
|---|---|---|---|---|
| as tuned (signal minute's close) | 51.24 | 110.82 | 104.47 | +59.59 |
| signal exits at the next open | — | 107.82 | 101.87 | |
| signal exits at the next close | — | 108.68 | 102.17 | |
| **ALL entries + exits at the next open** | 49.80 | 107.79 | 101.51 | **+58.00** (+58.0 / +56.2 / +59.7) |
| ALL at the next close (pessimistic) | 45.32 | 105.10 | 97.58 | **+59.77** |

* **The lead survives every fill assumption,** including a full minute of delay on every entry and exit.
* The price move in the minute after a signal exit is ≈ 0% (next open) / +0.18% (next close): **no reversal is being captured.**
* **Diagnostics (val, per run, 70 coins):**
  * A50 makes 1,027 trades, 95% at the take-profit (avg +$3.17). Positions still open at the end of the data lose **−$475** (avg −$10.2, median hold 1,596 min): the locked losing positions.
  * signal_exit makes 2,807 trades: 75% signal exits (net **+$829**, i.e. they cut roughly at break-even; shuffled: **−$1,483**), 24% take-profits (avg **+$16.66**, from compounding of recycled capital and the higher 0.32 target). End-of-data losses nearly vanish (−$21).
* **Interpretation:**
  * Being able to sell a trade that hasn't worked out within about an hour frees capital that compounds in new trades.
  * The models choose better moments to cut (+$829 vs −$1,483 for random timing), worth about +$6 in the score.

**Next:**
1. A clean test of the mechanism: a pure **time exit** (sell at market after T minutes if no take-profit; no models) vs the model signal exit.
2. Robustness: higher slippage (3%), and the tune-only selection check (§24) for signal_exit.
3. **Confirm with new seeds** (new move + direction models on the new split).

---

## 34. Signal-exit follow-up (`exit_followup`)

Tests passed:
* splits disjoint;
* retraining seed 42 reproduced the existing move model exactly;
* the exit rules on hand-made data;
* the exits switched off reproduced A50, and a saved signal_exit run reproduced ($117.60).

**PART 4, confirmation with NEW move + direction models (seeds 5, 6, 7):**

| | s5 | s6 | s7 | Mean | vs A50 |
|---|---|---|---|---|---|
| A50 | 51.57 | 49.54 | 51.47 | 50.86 | |
| **signal_exit** | 110.62 | 112.25 | 112.58 | **111.82** | **+60.96** (p < 0.001) → **CONFIRMED** |
| time_exit | 103.54 | 106.06 | 105.79 | 105.13 | +54.27 (p 0.001) → confirmed |

**PART 1, mechanism (seeds 42 / 1 / 2):**

| | Mean | vs A50 |
|---|---|---|
| A50 | 51.24 | |
| signal_exit | 110.82 | +59.59 |
| signal_exit_shuffled | 104.47 | +53.23 |
| **time_exit** (sell after ~70 min if no take-profit) | 105.47 | +54.24 |
| **te_signal** (time exit + signal, whichever first) | **112.16** | +60.92 |

* **A plain clock captures most of it:** time_exit sells at market ~70 min (57–78) after entry if the take-profit (0.33) hasn't come, with a very loose entry (0.16), and earns +$54.
* **The models add +$5.35 over the clock** (3/3 seeds; signal_exit − time_exit).
* Combining both (te_signal: time exit ~450 min as a backstop + the signal) is the best on these seeds (+$0.9 over signal_exit), but it has not been confirmed on new seeds.

**PART 2, slippage 3% per side:**
* Both re-tuned: A50 **$41.09**, signal_exit **$79.10**, **+$38.01** (3/3 seeds).
* Not re-tuned: A50 $37.15, signal_exit $75.99.
* The lead shrinks with costs, as expected with ~3× the trades, but stays very large.

**PART 3, tune-only (no val selection):**
* A50: honest val ≈ $47 (selection bonus ≈ $4, as §24).
* **signal_exit: honest val $112.12**, i.e. **no selection bonus** (−$1.3). Its tune score ($124.6) is above its val score, but its val result doesn't depend on picking the run by val.
* The summary cell ran in its old version, so A50's per-seed tune-only values are missing for seeds 42 / 1 (the fixed cell reads them from `a50_tune_only`); the conclusion doesn't depend on it.

**Raw equivalents of the exit levels:** direction < percentile 0.23–0.27 = raw ≈ 0.47–0.48; move < percentile 0.10–0.19 = raw ≈ 0.12–0.19.

**Conclusion:**
* **signal_exit is adopted.** It was confirmed with independent models at more than double A50 (+$61), survives next-minute fills (§33.1) and 3% slippage (+$38), and shows no selection bonus.
* **Most of the value is structural:** being able to sell a trade that hasn't worked within about an hour.
* **time_exit is the simple fallback** (one parameter, no models in the exit, ~$5 less).

---

## 35. LOOK-AHEAD in the signal / time exit (found 2026-11-15) — §33–34 are invalid

**Found by the live-pipeline chat.**
* The engine applies the signal exit (and the time exit) **after the fact**: `simulate_coin` first plays each trade out to its take-profit or the end of the data, then moves the exit back to the signal minute.
* The entry-candidate pointer `c_ptr` was **not reset**. Rebases and loss-floor / stale races in that imaginary continuation still consumed candidates.
* After an early exit, the next buy therefore skipped to the first candidate after the continuation's last rebase, **just before a take-profit rise the simulator had already seen**. A live bot can't know this.

**Consequences:**
* All signal_exit / time_exit / te_signal results are invalid: the +$60 in §33, the next-minute-fill check, the confirmation with new seeds, the 3% slippage test and the tune-only check in §34, and every arm of `new_system`.
* **The shuffled control's +$53 is now explained:** most of the "exit mechanism" gain was this bias.
* The execution check and the new seeds could not catch it: it's a simulator flaw, not a fill or model effect.
* (I had wrongly argued in §33 that skipping candidates could only lower the score; the skip is selective, onto a candidate just before a known rise.)
* **A50 is not affected:** without truncating exits, the pointer moves only with real events. `exit_fix` checks this (test 4 and the causality test).

**Fix (`exit_fix`):**
* After any exit that moves a trade's end earlier, the candidate pointer is rewound to the first candidate after the real exit.
* A **causality test** for the simulator: a coin simulated on its full data and on data cut at minute C must give identical trades before C. The old engine fails it for the signal exit; the fixed engine must pass for A50, signal_exit and time_exit.
* Then the saved runs are re-evaluated and the arms re-tuned with the fixed engine.

**Lesson:** every new engine rule must pass the causality test before its results count. A rule decided "after the fact" can leak the future through any state it doesn't fully undo.

### 35.1 Results with the fixed engine (`exit_fix`)

**Causality test** (12 val coins × 2 cuts):

| Strategy | Old engine | Fixed engine |
|---|---|---|
| A50 | 0 / 24 mismatches | 0 / 24 |
| signal_exit | **7 / 24** | 0 / 24 |
| time_exit | **10 / 24** | 0 / 24 |

* A50 is clean in both engines: its results stand.
* Test 4: A50 reproduces with the fixed engine; the old signal_exit run reproduces only in legacy mode ($117.60 → $8.94 with the fix).

**Re-evaluation** (same tuned parameters, fixed engine):
* signal_exit $110.82 → **$16.94**;
* shuffled $104.47 → $8.69;
* time_exit $105.47 → $8.60;
* te_signal $112.16 → $14.62.

Those parameter sets traded 80–128 times per coin, entirely on the bias.

**Re-tuned with the fixed engine** (3 seeds × 10 runs):

| | s42 | s1 | s2 | Mean | vs A50 |
|---|---|---|---|---|---|
| A50 | 50.45 | 51.36 | 51.89 | 51.24 | |
| signal_exit | 50.62 | 49.32 | 52.62 | 50.85 | −0.38 (n.s.) |
| time_exit | 50.75 | 51.35 | 51.61 | 51.24 | ±0.00 |
| signal_exit_shuffled | 38.61 | 39.92 | 40.51 | 39.68 | −11.56 |

* **The whole +$54–60 was the look-ahead.** Honestly simulated, a market exit before the take-profit adds nothing.
  * The optimiser sets the time exit to ~773 min (≈ never before the stale time) and the signal exit to very low levels (direction < 0.125, move < 0.18), i.e. it nearly switches them off. Selling early at random costs $11.6.
  * The models time exits better than chance (+$11.2 vs shuffled), but not better than not selling.
* **Conclusion:** the market-exit line is closed. **A50 remains the system** (val ≈ $51, ≈ $47 without the selection bonus).

### 35.2 Hold-out check (clf_train + clf_val: 78 coins never used to tune or select any strategy)
Every run evaluated once, fixed engine:

| Arm | All runs (s42 / s1 / s2) | Mean | Best-by-val run (s42 / s1 / s2) | val → hold-out ρ |
|---|---|---|---|---|
| **A50** | 43.72 / 44.45 / 44.38 | **44.18** | 43.21 / 47.92 / 43.21 | 0.18 |
| signal_exit | 41.13 / 41.92 / 43.42 | 42.16 | 45.04 / 48.78 / 43.40 | 0.71 |
| time_exit | 46.43 / 43.49 / 44.83 | 44.92 | 45.68 / 44.17 / 42.47 | 0.09 |
| signal_exit_shuffled | 33.15 / 31.15 / 30.23 | 31.51 | | −0.11 |

* **By the rule fixed in advance, no arm is a real gain:**
  * signal_exit −$2.02 (CI −6.8..+2.5);
  * time_exit +$0.74 (CI −2.4..+4.1; one seed negative);
  * shuffled −$12.7 (a real loss).
* signal_exit's best-by-val runs are +$0.96 over A50's best-by-val (n.s.). Its high ρ (0.71) means val separates its two kinds of solutions (exit-heavy = worse, A50-like = fine). **The ~$55 val runs are A50-like and hold A50's level.**
* **A50's honest level: about $44 per coin from $20 on unseen coins** (val $51 includes the selection bonus plus coin differences). Picking A50 runs by val barely helps (ρ 0.18); plan live expectations with ≈ $44.
* The clf coins are now spent for strategy decisions.

---

## 36. First presentable model search (`model_search`, previous data)

Run by the user on the data before the rebuild.
* A1: move-model search; B1: top 10 move models alone, 10 strategy runs each.
* A2: direction search on the chosen move model's candidates; B2: top 10 direction models paired with it, 10 runs each.

| Stage | Result |
|---|---|
| B1 | best move model **move_trial044** (0.14 MB, pAUC 0.598): **$48.44 ± 0.54** alone; the 10 move models span $46.7–48.4 (close, mostly within noise) |
| B2 | best direction model **dir_for_move_trial044_trial041** (0.10 MB, AUC 0.594): **$51.76 ± 0.61** with move_trial044 |
| **value of the direction model** | **+$3.33** (+6.9%) for the chosen pair, better in 8/10 runs (paired p 0.003); across all 10 direction models **median +$2.13**, 9/10 better than the move model alone |

* The chosen pair matches the old A50 on val ($51.24) with **far smaller models**: 0.24 MB together.
* The direction model's contribution is real and consistent. The median (+$2.1, about 4%) is the fairer estimate, since the chosen pair is the best of 10.
* These are val numbers on the old data. The rebuild re-runs this, and the test split gives the honest number.

---

## 37. Overall conclusions so far

1. **No bug found.** Predictions are aligned correctly, and the feature recompute matches the original exactly.
2. **The buy model is a volatility detector.** It ranks upside well, but direction only weakly.
3. **The current strategy (hold until take-profit) profits from volatility, not direction.** A model that predicts volatility therefore adds nothing over trading every dip.
4. **The strategy sim is the wrong yardstick for choosing labels.** Use direct entry-quality measures (§5.1) and/or the stop-exit sim (§5.5), where entry quality shows up.
5. **Losses are concentrated in dying coins** (§5.6).
6. **A binary first-passage label also learns volatility** and does not improve on the current label (§6.1–6.5).
7. **But direction IS learnable** once quiet minutes are removed: within-coin AUC 0.57–0.61, consistent across coins and both unseen coin sets (§6.6).
8. **Move × direction works with the existing exits**, as a product with one threshold; a separate threshold per model is worse (§7.4). Baseline × ±15%/120m direction model beats the baseline by +$3.84 (stop mode, z 4.7) and +$1.71 (rebase mode, z 3.9). Best validation score so far: $48.31 (§7).
9. **Label search (§8):** direction ±15%/120m is best and the move label barely matters. First-passage pump features did not help.
10. **Confirmed on unseen test coins (§8.4):** `fp15_combo` (move 10%/30m × direction ±15%/120m, dataset-style pumps) beats the current model by **+$3.80/coin in stop mode** and **+$2.25/coin in rebase mode** (+11% / +5.6%). This is the final candidate. The test set is now spent.
11. **Dump filter (§9, §11):** as a veto it doesn't help. In the product, the end 30%/24h gain from §9 **did not replicate** when retrained on slightly different rows, and no label in a finer grid gives a robust gain. **No entry filter reduces end-of-data losses; that needs a dump-triggered exit.**
12. **Model 3's features don't transfer (§10):** given to models 1 and 2 they change nothing and can't replace model 3. Its value comes from its 24h label.
13. **Tuning and other libraries don't help (§12):** models 1 + 2 are at the ceiling of the current features and labels; the original XGBoost models stay. **Next lever: features.**
14. **The direction model trained only on candidate minutes (A50) is confirmed (§22):** +$1.10 on new seeds, +$1.13 over all 10 seeds (9/10 better). Aligning training with where the strategy acts beats more features or tuning. **Current combo = move × direction-on-candidates.**
15. **Lessons for every comparison:** the best of many single-model results is mostly luck, so confirm winners with fresh model seeds and criteria fixed in advance (§17.1). Model AUCs (and strategy-free yardsticks) don't reliably predict the strategy; the sims decide (§20–22).
16. ~~Selling at market before the take-profit doubles the score (§33–34).~~ **Withdrawn: a look-ahead in the engine (§35).** With the fixed engine, a signal or time exit adds nothing (−$0.38 / ±$0) (§35.1). **Current system = A50.** Every engine change must now pass the causality test.

---

## 38. Next steps / open questions

Done:
- [x] Directional label (binary): no improvement (§6).
- [x] Is direction learnable? Yes, within-coin AUC 0.57–0.61 (§6.6).
- [x] Move × direction with the existing exits: `fp15_combo` wins (§7).
- [x] Dual gate: worse than the product; not adopted (§7.4).
- [x] Boundary check: passed (§7.5).
- [x] Coarse label search for both models (§8).
- [x] Head-to-head of the direction pump features: dataset-style wins (§8.4).
- [x] **Test-set confirmation: `fp15_combo` confirmed** (§8.4).

Open:
- [ ] **Make the final candidate permanent:**
  - `Dataset_creationV8`: add the ±15%/120m first-passage label (all 3 classes) and a second set of the 25 pump features with pump = +15% above the 120-minute low;
  - a training notebook for the direction model (minutes that hit a barrier only);
  - the strategy notebook: confidence = percentile of P(move) × P(up first), with the tune reference distribution saved alongside the models.
- [ ] (Optional) Can the direction model use the ORIGINAL 10%/30m pump features? That would remove the second feature set. Judge on validation only, since the test set is spent.
- [ ] **Forward check on new data:** paper-trade, or use coins collected after this dataset, since the test set is used up.
- [x] **Dump filter (§9):** veto rejected. Product with end 30%/24h is promising but borderline.
- [ ] **Diagnostic: "no new buys in a coin's last 48h".** Re-score the saved runs of the combo and the end 30%/24h product with this rule (no re-optimisation). If model 3's gain survives, it is genuine entry quality; if it disappears, it came from avoiding late-life entries. This is a diagnostic only, not the tuning rule: live you can't know a coin's last 48h, and the dead stretch is a real risk. (Every coin's data ends in its dead stretch; none ends while still alive.)
- [x] **Feature-transfer test:** the features don't replace model 3 and don't help models 1 + 2 (§10).
- [ ] (Optional, simplification) **Model 3 without the 13 new features** (dataset features only), to see whether they can be dropped from the pipeline.
- [x] **Finer model-3 label grid:** no robust gain; model 3 not adopted as an entry signal (§11).
- [ ] ~~Independent check of the best model-3 label~~ (dropped: nothing robust to check): a single pre-specified look at `test` (second use of the test set), or wait for new data.
- [ ] **Dump-triggered exit** (when ready for a new exit mechanic): sell an open position when the short-horizon dump probability is high. This is the only route found so far to cut end-of-data losses.
- [x] **Hyperparameter search, models 1 + 2:** no gain; current models kept (§12).
- [x] **Run `Dataset_creationV9`:** all five tests passed.
- [x] **V9 check:** rebase the same; stop mode −$2.80 (§14).
- [x] **Model-seed control:** half the V9 gap was model luck; a small residual (−$0.5 rebase, −$1.4 stop) remains, likely the removed liquidity look-ahead. **V9 accepted** (§15).
- [x] ~~Ablation of the residual~~ (decided not to do it).
- [x] **`Dataset_creationV9` cleaned up** into a normal dataset notebook: build, unit tests, leakage test, and V8's stratified splitter (5 splits, preserve mode by default).
- [x] **Rebuilt V9 with a full re-split and re-established the baseline** (§16): combo $48.03 (rebase) / $45.86 (stop).
- [x] **1% price-impact variants:** no variant clearly better; weak hint for 3-day windows (§17).
- [x] **Impact confirmation:** no finalist holds up; **current method locked** (§17.1). `impact_variants/` and `impact_confirm/` can be deleted (results documented here).
- [x] **Extra windows + duplicates:** neutral ($48.10 vs $48.10); kept for now (§18).
- [x] **New feature families:** worse (−$0.87); not adopted. The direction AUC improved anyway, a sign the model yardstick is misaligned (§19).
- [x] **Yardstick check:** direction AUC anti-correlated with the sims; `tp_net_val_tune_top20` tracks them (§20).
- [x] **Strategy-aligned training:** A50 +$1.17 on all 5 seeds (Holm p 0.079, just misses); outcome models dropped; yardstick invalid for models trained on its target (§21).
- [x] **A50 confirmed and adopted** (§22).
- [ ] **Make A50 permanent:** a training notebook / pipeline step. Train the move model, then the direction model on candidate minutes (tradable, move top 50% by that move model's train predictions, decided). Save the move threshold with the models.
- [ ] **New reference for every later experiment:** A50 with seeds 42–4 (+ 5–9); its runs exist in `aligned_training/` and `aligned_confirm/`.
- [ ] (Optional) A50 in **stop mode** as a finalist check.
- [x] **Dynamic thresholds screening:** small gains; candidates dm_prod and i2_prod; families dropped (§23).
- [ ] **Confirm dm_prod and i2_prod** with seeds 5–9 (A50's seeds 5–9 reused from `aligned_confirm`).
- [x] **Tune-only check (partial):** ~$4 selection bonus in val-selected scores; tune overfits with 2,500 trials (§24). Optional: run its summary cell for the learning curve.
- [ ] **Re-run Dataset_creationV9 with 7 splits** (consider building the 22 window features into V9 and dropping `days_since_first` there), then re-establish A50 on the new split.
- [x] Tier classifier trained (`tier_classifier_v9`).
- [x] **Tier strategy:** +$1.17 (+2.5%), not significant; old 30% not reproduced (§25).
- [x] Win-rate score tested: −$2 money; not adopted.
- [x] **Raw-scale tier runs:** fixed_tier +$4.55 over fixed_static, best overall (§26).
- [x] **tier_v2:** A50 $51.24 best; tiers +$3.2 for move-only, ≈ +$0.6 on the combo (§27).
- [x] **Last tier test:** tiers dropped; A50 stays (§28).
- [x] **Death diagnostics:** ~$2–3.5 per coin per day, very concentrated; crashes are normal and profitable; no precise death signal; whole-coin checks not viable (§29).
- [x] **Rug block + dump-risk veto:** both not adopted; the risk model predicts dumps beyond volatility but vetoes cost money (§30).
- [x] **Risk-conditioned management:** no gain; risk line closed (§31).
- [x] **Three levers:** none beats A50 (§32).
- [x] **strategies:** signal_exit +$59.6, mostly the market-exit mechanism (shuffled +$53.2) (§33).
- [x] **Execution check:** signal_exit's lead survives next-minute fills (+$58.0 at next open, +$59.8 at next close) (§33.1).
- [x] **Signal exit confirmed and adopted** (§34).
- [ ] ~~Run `new_system`~~ — invalid (biased signal exit); re-do only if the fixed signal exit still beats A50.
- [x] **exit_fix:** the market exit adds nothing once the look-ahead is fixed (§35.1). Hold-out check (all runs on clf coins) running.
- [x] Hold-out check: no real gain; A50 ≈ $44 on unseen coins (§35.2).
- [ ] Remaining ideas: a direction label matched to the trade, on candidates; confirm dm_prod on the new split; robust parameter selection; conviction sizing; more coins.
- [ ] **Prepare A50 for paper trading:** fixed settings from one coherent run (§28), the move + direction models of one seed, a forward check on new coins.
- [ ] Death checks (block / block + force sell / permanent stop / random control) + per-day profit and loss anatomy, on the best tier set-up.
- [ ] **Re-establish A50 on the new 7-split V9** (move + direction on candidates, several seeds): the confirmed main improvement, and the reference for everything after.
- [ ] Then, if wanted: confirm one regime method on top of A50 (dm_prod preferred for simplicity; or tiers with A50).
- [ ] Revisit with A50 as the base: feature elimination; model 3.
- [ ] **Forward check on new coins / paper trading** (the test set is spent).
- [ ] Model 3 (dump) again, with dump-relevant features (down-event history, past first-passage outcomes, range volatility) and its best labels from §11, as a product, 5 seeds, rule fixed in advance.
- [ ] Then feature elimination.
- [ ] (Later) model 3 again, once the feature set is settled.
- [ ] (Before going live) fix the liquidity look-ahead in `Dataset_creationV8` (§39).
- [ ] (Optional, later) a bracket exit mode (§7.2), and the 3-class model as an alternative to the two-model combo.

## 39. Known issues and quirks

### ⚠ Look-ahead in the dataset's liquidity estimate (found 2026-10-26)
The new leakage test (`label_pair_search`, TEST 2) recomputes features on data cut off at a given minute and compares them with the full-data values.

`compute_liquidity_robust` in `Dataset_creationV8` computes two **whole-coin** statistics over the coin's entire history, including its future:
* the volume floor (`np.quantile(pos_vol, 0.05)`);
* the median/MAD clip of the per-bar log-impact.

So every liquidity value depends slightly on data that wasn't available yet. On synthetic data:
* ~90% of values change when the future is removed;
* median change 0.1%, 95th percentile 0.5%, max 2.5%.

**On the real data (TEST 2b):** 96% of values change; median 3.3%, 95th percentile 17.5%, max 35%.

**Decision (2026-10-26): accepted for now.** The 1% price-impact estimate moves slowly, and the look-ahead affects every model equally, so comparisons are unaffected. Revisit before going live.

**What it affects:** `estimated_sol_1pct_impact`, every liquidity-based feature, the five liquidity-dependent pump features, and the sim's position sizing. It affects every model and experiment equally, including the baseline. The leak is small in size, but it is information from the future.

**Fix (needs a dataset rebuild):** compute the floor and the clip causally, from data up to each minute (expanding or rolling windows) or from fixed constants. Then retrain the baseline to measure the effect.

### How the current 1% impact estimate behaves (documented 2026-10-30)
* **The "24h window" is a time constant, not a hard window.** It's an exponentially weighted geometric mean of all valid candles since launch. Weight: 37% at 24h old, 14% at 48h, 5% at 72h.
* **"5 candles" is only a warm-up rule.** No estimate (NaN) until 5 usable candles since launch.
* **With no valid candles the estimate freezes at its last value**, however long the gap. The weights decay equally, so the average doesn't move. Only `liquidity_sample_age_minutes` (up) and `liquidity_confidence` (down by about 63% per 24h) show the staleness.
* **Consequence for the sim:** on coins that go quiet or die, positions are sized on the last known liquidity, which is probably optimistic exactly where it matters.

### Rebase mode never sells at a loss before the take-profit (found 2026-11-14)
* In rebase mode, the loss floor and the stale time do not sell: the loss floor re-anchors the reference price at a later candidate, and the stale time starts a race between the take-profit and a re-entry.
* A position therefore ends only at the take-profit (possibly below the original entry, after re-anchoring) or at the end of the data.
* The signal exit (§33) is the first rule that sells at market before that, which is why it changes the results so much.

### Quirks (not affecting conclusions)

* All sim notebooks predict with `iteration_range=(0, best_iteration)`, which leaves out the final best tree. Kept for comparability.
* The cross-run table in `optuna_strategy` numbers runs from 0; the main output numbers them from 1.
* The `tier_classifier_optuna_hpo_xgb_ordinal` markdown describes an older multi-class setup and old paths.
* Experiment 1 ignores slippage and the price-position filters, so its edges are rough.
* pandas `groupby.apply` FutureWarning in experiment 1: harmless.

---

## 40. Candle finalization study (`finalization_study.ipynb`, live pipeline)

**Status:** built and tested on a simulator (§40.5). The first real run (COGE, 2026-11-07) was discarded because the computer's clock was 2.3 s off (§40.7). **Results:** a first reading of about one day on COGE (§40.8); more data and a second coin are needed.

### 40.1 Question
* The dataset was collected long after the fact, so its candles are **final**. The live bot reads the minute that has just ended a few seconds after its close.
* If GeckoTerminal has not finished that candle yet (late-indexed trades), the live features differ from the ones the models were trained on: a mismatch between training and live data.
* The author's manual tests suggest candles are final ~10 s after the close and that the size of the remaining changes shrinks with every second.
* **Goal:** the earliest second after the close at which 95 / 99 / 99.9% of candles are final, per field and for the whole candle, and the size of the remaining changes at each second. The result sets the bot's read time (`grace_period_seconds`).

### 40.2 Design
* **Sweep.** Reads at 1.0, 1.5, … 40.0 s after a minute ends (79 offsets, step 0.5 s), **3 consecutive offsets per minute**: minute 1 reads 1.0 / 1.5 / 2.0 s, minute 2 reads 2.5 / 3.0 / 3.5 s, and so on. One sweep = all 79 offsets once = 27 minutes (the last minute has only 40.0 s), then it starts again at 1.0 s.
* **The read.** Every call returns the last 1000 candles; the read is the candle of the minute that just ended. Each candle is read 3 times (at 3 consecutive offsets); the curve over 1–40 s is built from many candles across sweeps.
* **Reference ("true" values).** The burst made **≥ 5 minutes after the candle closed** (`REFERENCE_AFTER_S = 300`; for candle 12:00, which closes at 12:01:00, the burst at 12:06) reads the same candle again; its last call's values are the true values. It costs no extra calls: every burst both reads the newest candle and serves as the reference for the candle of 5 minutes earlier. `ref_consistent` records whether all 3 calls of the reference burst agreed, a check on the 5-minute assumption for every candle.
* **Rate limit.** At most 3 calls in any rolling 60 s + 1 s margin. Moving 1.5 s later each minute keeps every 4-call span at 61.5 s, so the step must stay ≥ 0.34 s (0.5 s used). The limiter is seeded from the latest calls of **all** coins (`recent_calls.json`), and only **one coin at a time** may collect (the limit is per IP). A 429 pauses collection for 180 s.
* **Clock.** Reads are timed with the computer's clock, which must match the server's:
  * every call brackets the server time (the `Date` header, whole seconds) between its send and receive moments; intersecting the brackets gives the offset to about **±0.3 s** (the sweep sends at whole and half seconds, which limits the precision);
  * the run **refuses to start** when the clock is surely more than `MAX_CLOCK_OFFSET_S` = 1 s off within its first 10 minutes; later it warns hourly and keeps going;
  * every row stores the offset and the corrected age; the statistics leave out rows read with a clock off by more than 1 s.
* **Robustness.** Each row is written and synced to disk when its reference arrives. Reads still waiting for it are kept in `.pending.json` and completed after a restart (each call still returns the last ~16 h). A half-written last line after a hard stop is cut off on restart. The same coin continues its CSV; another coin gets its own.

### 40.3 Files (`finalization_data/`)

| File | Content |
|---|---|
| `finalization_<coin>.csv` | one row per read (below) |
| `finalization_<coin>.pending.json` | reads waiting for their reference; the position in the sweep |
| `recent_calls.json` | send times of the latest calls (all coins) + the coin collecting |
| `rate_limits.csv` | one row per 429: `time_utc, t_send, coin, call_in_burst, calls_in_burst, offset_s, since_prev_call_s, calls_prev_60s, calls_prev_10s, other_calls_in_burst_ok` |

Columns of `finalization_<coin>.csv`:

| Column | Meaning |
|---|---|
| `minute_utc`, `minute_ts` | the candle (minute start, UTC) |
| `offset_s` | planned read time, seconds after the candle closed |
| `age_s`, `age_true_s` | actual age at the read (midpoint of send and receive); corrected for the clock offset |
| `clock_offset_s`, `clock_offset_pm_s` | server − local clock, and its uncertainty |
| `latency_s`, `t_send` | request latency; send time (local, unix) |
| `present` | was the candle listed in the response yet |
| `read_<f>`, `true_<f>` | the value read, the true value (f = open, high, low, close, volume) |
| `ref_age_s`, `ref_consistent` | age of the reference read; all 3 reference calls agreed |
| `traded` | the true volume > 0 |
| `pct_<f>` | (read − true) / true × 100 (signed) |
| `final_<f>`, `final_candle` | the read equalled the true value; for all five fields |

### 40.4 Statistics (the notebook's last cell; standalone)
* Reads every `finalization_*.csv`; by default only **traded minutes** (untraded minutes rarely change).
* Per offset: reads, share listed, share final per field and for the whole candle with a **Wilson 95% interval**, share within 0.01 / 0.1 / 1% of the true values, mean and p99 |% change|.
* A **non-decreasing fit** (isotonic regression weighted by reads) pools neighbouring offsets. **Headline:** the earliest second at which the fitted share reaches 95 / 99 / 99.9%, per coin, plus a table across coins.
* Also: the clock check (rows left out) and a rate-limit summary by call position. Mostly calls 2–3 of a burst → the close spacing trips the limit; mostly call 1 after a quiet minute → the per-minute total or other traffic on the IP.

### 40.5 Validation of the method (simulator; not results)
* A fake API with **known** finalization times: prices final after a median of 4 s, volume after 6 s, all within 200 s; the just-ended minute listed only after 1.2 s.
* **4 virtual days** through the notebook's own code: 95% final estimated at **20.0 s** (true 18.5 s), 99% at **27.0 s** (true 31.5 s), 99.9% not reached by 40 s (true: not reached either). The rate limit held throughout (every 4-call span 61.5 s); the reads at 1.0 s were correctly recorded as "not listed yet".
* **Real-time runs against a fake API:**
  * interrupt and restart: waiting reads completed after the restart, the sweep continued at the right offset, no duplicates;
  * a hard stop with a half-written line: repaired on restart;
  * a clock 2.3 s off: refused at the first burst, nothing stored;
  * a 429 on call 2 of a burst: logged correctly (0.5 s after call 1, 1 call in the preceding 60 s);
  * a clock drifting 1.5 s halfway through a day: warned, and exactly the affected rows left out.
* With ~160 reads per offset (4 days), the 99% figure was still off by a few seconds: 1% is under 2 reads per offset.

### 40.6 How much data, and the procedure
* Each offset is read once per sweep: **~53 reads per offset per day** (~40–45 on traded minutes; fewer with 429 pauses). Only 3 useful reads per minute exist on one IP; more needs more IPs or fewer offsets.

  | Running time | What it gives |
  |---|---|
  | 1 day | a first look; the 95% second within a few seconds |
  | 3–4 days | a solid 95% second and a usable 99% second (±1–2 s) |
  | ~1 week | a solid 99% second |
  | weeks | 99.9%: not realistic this way |

* **Stop a coin** when the 99% second has stayed within ~1 s on two consecutive daily checks, its confidence band is narrow (about ±1–2 points), the clock stayed small and 429s were rare. Stop at whole days, so every offset was read across all hours.
* **More reads where it matters:** after the first day, set `OFFSET_END_S` a few seconds past where the curve reaches ~100% (e.g. 1–20 s → ~110 reads per offset per day; 1–15 s → ~150). Keep the step ≥ 0.5 s.
* **Coins:** at least 2, of clearly different activity. If they agree within 1–2 s, treat the result as general; otherwise add a third and use the slowest.
* **Setup:** keep the trading bots and anything else using GeckoTerminal off the IP; sync the clock first (Windows: `w32tm /resync` in an admin prompt, check on time.is). The statistics cell can run from a second notebook while collecting.

### 40.7 First run and lessons (COGE, 2026-11-07)
* **Clock 2.3 s off** (the PC ahead of the server). A read labelled "second 1.0" happened ~1.3 s *before* the minute ended; every age was shifted. The run was discarded, and the clock check (refuse / warn / per-row offset) was added.
* **Two 429s in 13 minutes**, the first on call 1 of a burst after the usual 61.5 s spacing (most other bursts with the same spacing went through). This suggests other traffic on the same IP (a shared VPN exit or the bots), not the schedule; the per-call 429 log was added to tell these apart. Fallbacks: `CALLS_PER_MINUTE = 2` (a sweep then takes 40 minutes) or `BACKOFF_AFTER_429_S = 65`.

### 40.8 Results (to be filled in)

| Coin | Days | Reads / offset | 95% final at | 99% final at | 99.9% final at | `ref_consistent` | Notes |
|---|---|---|---|---|---|---|---|
| COGE (first reading) | 1.2 (28.7 h) | 39–44 | not reached by 40 s | not reached | not reached | 99.65% | only `open` reaches 95% (at 37.5 s); 24 × 429, all on call 3 of a burst |

**First reading (COGE, traded minutes only):**
* Candles become final far later than the ~10 s the manual tests suggested. At 1–3 s almost no candle is final (0–2%); at 38–40 s only about half are (49–60%). Within 1% of the true values: 12% at 1 s, 77–84% at 38–40 s.
* The 5-minute reference is not always final either: 0.35% of the reference bursts disagreed among their own 3 calls.
* All 24 rate-limit events were the third call of a burst (0.5 s after the second): the close spacing trips the limit (§40.4).
* **Consequences:** extend `OFFSET_END_S` well past 40 s on the next run; the current 5 s grace period reads candles that are rarely final yet.

### 40.9 Use in the live pipeline (open)
* Set the bot's read time (`grace_period_seconds`) from the slowest coin's 99% second. In the current `bot_loop.py`, the grace period applies **in paper mode only** and shadow / live read at second 0; the study's value should apply in every mode.
* The Pi's clock must be right for the read time to mean anything: NTP (`chrony`), and start the app only after the first sync. A clock guard in the app (the same `Date`-header estimate from the bots' own calls; coarser, ~±0.5 s, because the bots always call at the same second) is proposed, not built.

---

## 41. Live pipeline vs the notebooks: alignment check (2026-11)

Checked before publishing the repository, with the code taken straight from the notebooks.

| What | How | Result |
|---|---|---|
| Features (`features.py` vs `Dataset_creationV9`) | function-by-function diff; then the notebook's feature cells against `compute_live_features` on synthetic coins (4 and 8.7 days, with zero-volume stretches): single minutes, multi-minute batches and 40 consecutive minutes (the incremental liquidity cache) | **bit-identical**, all 169 features. The code differs only in the `LIVE OPTIMISATION` lines (row range), in where the 22 window features are merged, and in the `days_since_first` drop; the last two cancel out |
| Strategy (`strategy.py` + `execution.py` vs the engine of `strategy.ipynb`; rebase mode, signal exit off) | 600 random coins × random parameter sets: the engine's `simulate_coin` against a minute-by-minute replay of the live state machine with paper fills | **identical** in every case: entry and exit minutes, rebases, loss-floor arms and recoveries, final capital. Every branch was exercised (5,647 rebases, 1,074 loss-floor arms, 389 sales with an open re-entry scan) |
| Engine copies (`model_search` vs `strategy`) | text diff of the engine and sim-helper cells | identical (one progress print differs) |
| Fill rule (`data_collection.py` vs `data_cleaning`) | after the fix below: against `backfill_missing_candles` on synthetic API data with skipped minutes and non-flat zero-volume candles; full history, and minute-by-minute appends with late corrections | identical |
| Scores (`scoring.py` vs `strategy.ipynb`) | the export cell (§0.12) on test models, then `LiveScorer` against the notebook's `predict` / `to_percentile` | identical raw predictions and percentiles |

**Fixed in the live code:**
* **Fill rule.** `forward_fill_missing_minutes` only filled minutes the API skipped (with the previous row's close); zero-volume candles returned by the API were kept as they were. It now applies the research rule: every zero-volume minute is a flat candle at the last real close. New minutes are appended with the last stored minute as the anchor, and a late revision of the last minute is put on the same rule before it is compared.
* **Live inputs.** Models, reference distributions and parameters come from the export cell of `strategy.ipynb` (§0.12). `scoring.py` takes each model's feature list from the model file. The combo reference must be the product with the sims' rounding (float32 predictions): computing it in float64 gives slightly different percentiles for some minutes.
* **Re-entry rule:** already the relaxing rule; the note that the live bot tightened was out of date (§0.10–0.11).

**Not checked:** the signal exit (off) against the fixed engine; shadow fills (they come from Jupiter quotes by design).

---

## 42. Presentable rebuild: dataset, model search, strategy (rebuilt data)

**Dataset** (`Dataset_creationV9`): 402 coins, 169 features, splits 15 / 10 / 25 / 25 / 25% (train 59, val_train 39, tune 102, val_tune 102, test 100 coins). Leakage test passed (4.1 M feature values identical with and without future data). Stratification is thin at this coin count: no split has an old high-pump coin.

**Model search** (`model_search`, val_tune, 10 strategy runs per model):

| Stage | Result |
|---|---|
| B1 | best move model **move_trial057** (0.36 MB, pAUC 0.616): **$48.32 ± 0.48** alone; the 10 move models span $45.8–48.3 |
| B2 | best direction model **dir_for_move_trial057_trial002** (0.07 MB, AUC 0.595): **$50.62 ± 0.73** with move_trial057 |
| **value of the direction model** | **+$2.30** (+4.8%), better in 9/10 runs (paired p 0.003); across all 10 direction models median +$1.47, 10/10 better than the move model alone |

**Strategy** (`strategy`, up to 6,000 trials per run; the saved results cells cover the **first 30 of 50 runs**):

| | val_tune | test |
|---|---|---|
| mean over runs | $50.03 ± 0.34 | $47.74 ± 0.48 |
| **best val run (run 4): the honest result** | $54.03 | **$43.76** |
| best test run (run 27; information only) | $50.46 | $52.91 |

* From $20 per coin, the best val run ends at $43.76 on unseen coins (2.2×, per-coin capital capped at 10×). That is in line with the earlier hold-out estimate (≈ $44, §35.2).
* The best val run is among the three worst on test: picking the best of many runs on val selects for val luck (val − test: −$10.3 for that run, −$2.3 on average; the selection bonus of §24).
* Its parameters: confidence 0.869; price position 15m ≤ 0.449, 1h ≤ 0.712; take-profit +28.8%; stale 477 min; loss floor −32.5% for 6 min; re-entry 15.4% below the sale, relaxing 0.87%/min after 156 min.
* **To do:** re-run the results cells after the last run; the numbers above will change.

