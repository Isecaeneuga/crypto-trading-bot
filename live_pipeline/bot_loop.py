"""
Bot Loop - Main orchestrator for each trading bot process.
Launched as a subprocess per bot by bot_manager.py.

Flow per closed minute:
    fetch OHLCV -> features (features.py, the Dataset_creationV9 code) -> move + direction
    models -> percentile scores (scoring.py) -> strategy state machine
    (strategy.py: A50, rebase mode) -> execute -> record

Strategy (see strategy.py for the exact order of checks):
  * buy when combo_pct >= confidence_threshold, both price-position filters
    pass, the coin is >= 720 minutes old and the re-entry rule allows it;
  * sell at the take-profit; the loss floor and the stale time never sell,
    they start races that may re-anchor (rebase) the position;
  * the signal exit is OFF unless strategy_config.signal_exit_enabled is set.

Minute index = row index of the forward-filled OHLCV = minutes since the coin's
first minute, exactly as in the research data. The bot therefore always holds
the coin's FULL history (fetched on the first start, kept in ohlcv_data).

The state machine (strategy.StrategyState) is persisted as JSON after every
minute, so a restart resumes the same position and the same race / re-entry
state. Minutes missed while the bot was down are replayed through the state
machine on restart (no buys on past minutes; a due sell is executed at once).
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

from database import (
    get_bot, get_global_settings, get_latest_sol_price,
    update_bot_runtime, update_bot_trading_state,
    insert_ohlcv_data, insert_predictions,
    insert_open_trade,
    update_trade_on_exit, get_open_trade_for_bot,
    deduct_trading_fee, get_bot_cumulative_fees_sol,
    insert_capital_snapshot, log_message,
    get_latest_timestamp, get_latest_ohlcv, get_all_ohlcv,
    get_strategy_config, sync_strategy_config_from_file,
    insert_wallet_capital_snapshot, get_wallet_status, get_all_bots_from_db,
    DEFAULT_MODEL_DIR
)
from features import compute_live_features, display_columns, check_minute_grid, START_IDX
from scoring import LiveScorer
from strategy import (
    SignalExitStrategy, StrategyState, MinuteInput, is_legacy_config,
    ACTION_BUY, ACTION_SELL, REASON_TAKE_PROFIT, REASON_SIGNAL_EXIT,
    PHASE_NORMAL, DEFAULT_MAX_PRICE_IMPACT_PCT, SIGNAL_EXIT_PARAMS
)
from execution import execute_buy, execute_sell
from data_collection import (
    fetch_historical_ohlcv,
    fetch_ohlcv_from_timestamp,
    fetch_ohlcv_since,
    forward_fill_missing_minutes
)

OHLCV_COLS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
DISPLAY_COLS = ['returns', 'completed_pump', 'vol_120', 'estimated_sol_1pct_impact',
                'liquidity_sample_age_minutes', 'liquidity_estimate_std_pct']
CHART_BACKFILL_MINUTES = 240      # scores computed at start-up for the chart

# ========================================================================
# HELPERS
# ========================================================================

def format_price_si(value):
    """Render a price using SI prefixes (m, µ, n, p, f) for sub-1 values
    instead of scientific notation or long runs of zeros.

    Mirrors the dashboard's formatPriceSI so log lines and the UI read the
    same way. Memecoin prices in SOL are typically far below 1, so e.g.
    0.0000000012 prints as '1.20n' rather than '0.00000000' or '1.2e-09'.
    """
    if value is None or value != value or value in (float('inf'), float('-inf')) or value == 0:
        return '0'
    abs_v = abs(value)
    if abs_v >= 1:
        if abs_v >= 100:
            return f'{value:.0f}'
        if abs_v >= 10:
            return f'{value:.1f}'
        return f'{value:.2f}'
    for factor, sym in ((1e-3, 'm'), (1e-6, 'µ'), (1e-9, 'n'), (1e-12, 'p'), (1e-15, 'f')):
        if abs_v >= factor:
            scaled = value / factor
            sa = abs(scaled)
            if sa >= 100:
                return f'{scaled:.0f}{sym}'
            if sa >= 10:
                return f'{scaled:.1f}{sym}'
            return f'{scaled:.2f}{sym}'
    # Below femto — fallback (extremely unlikely for any real price)
    return f'{value:.1e}'


def check_force_sell_files(bot_id):
    """Check for file-based force sell signals."""
    bot_dir = f'data/bot_{bot_id}'
    immediate = os.path.exists(f'{bot_dir}/force_sell_immediate')
    on_signal = os.path.exists(f'{bot_dir}/force_sell_on_signal')
    return immediate, on_signal

def clear_force_sell_files(bot_id):
    """Remove force sell signal files after acting on them."""
    bot_dir = f'data/bot_{bot_id}'
    for filename in ['force_sell_immediate', 'force_sell_on_signal']:
        filepath = f'{bot_dir}/{filename}'
        if os.path.exists(filepath):
            os.remove(filepath)


def write_wallet_snapshot():
    """Write an immediate wallet capital snapshot reading from capital_history."""
    try:
        wallet = get_wallet_status()
        if not wallet.get('initialized'):
            return
        all_bots = get_all_bots_from_db()
        total_initial = 0
        total_open = 0
        total_profit = 0
        from database import get_db_connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            for b in all_bots:
                if b['status'] != 'inactive':
                    cursor.execute('''
                        SELECT closed_capital_sol, open_position_sol
                        FROM capital_history WHERE bot_id = ?
                        ORDER BY timestamp DESC LIMIT 1
                    ''', (b['id'],))
                    row = cursor.fetchone()
                    if row:
                        closed = row['closed_capital_sol'] or 0
                        initial = b.get('initial_capital_sol', 0) or 0
                        total_initial += min(closed, initial)
                        total_profit += max(0, closed - initial)
                        total_open += row['open_position_sol'] or 0
        from datetime import datetime
        insert_wallet_capital_snapshot(
            datetime.now().strftime('%Y-%m-%d %H:%M:00'),
            wallet['unallocated_wallet_sol'],
            total_initial, total_open, total_profit,
            wallet.get('total_fees_sol', 0)
        )
    except Exception:
        pass  # Non-critical



def _ohlcv_record(all_data, idx):
    """The ohlcv_data row of minute idx (OHLCV + display columns + action)."""
    row = all_data.iloc[idx]
    action = row.get('bot_action')
    rec = {'timestamp': str(row['timestamp']), 'open': row['open'], 'high': row['high'],
           'low': row['low'], 'close': row['close'], 'volume': row['volume'],
           'bot_action': action if isinstance(action, str) else None}
    for c in DISPLAY_COLS:
        rec[c] = row.get(c)
    if rec.get('completed_pump') is not None and pd.isna(rec['completed_pump']):
        rec['completed_pump'] = None
    return rec


def _fill_display_columns(all_data, first, last, feats=None):
    """Display columns for rows first..last. Cheap columns from a recent slice;
    the liquidity estimate from the feature frame (the same causal estimate the
    sizing uses), falling back to NaN for rows without features."""
    lo = max(0, first - 130)
    part = all_data.iloc[lo:last + 1]
    close = part['close'].astype(float)
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(120, min_periods=120).std()
    rmin = close.rolling(31, min_periods=31).min()
    pump = ((close - rmin) / rmin >= 0.10).astype(int)
    for idx in range(first, last + 1):
        k = idx - lo
        all_data.at[idx, 'returns'] = ret.iloc[k]
        all_data.at[idx, 'vol_120'] = vol.iloc[k]
        all_data.at[idx, 'completed_pump'] = int(pump.iloc[k]) if idx >= 30 else 0
    if feats is not None:
        f = feats.set_index('minute_idx')
        for c in ('estimated_sol_1pct_impact', 'liquidity_sample_age_minutes', 'liquidity_estimate_std_pct'):
            for idx in range(first, last + 1):
                if idx in f.index:
                    all_data.at[idx, c] = f.at[idx, c]


def _ts_utc(ts):
    t = pd.Timestamp(ts)
    return (t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')).isoformat()


def _append_minutes(all_data, new_df, timezone, bot_id):
    """Append fetched candles to the full history on a gap-free minute grid,
    with the research fill rule (data_collection.forward_fill_missing_minutes;
    the last stored minute anchors the last real close)."""
    if new_df is None or len(new_df) == 0:
        return all_data, 0
    tail = all_data[OHLCV_COLS].iloc[[-1]]
    merged = pd.concat([tail, new_df[OHLCV_COLS]], ignore_index=True)
    merged = forward_fill_missing_minutes(merged, timezone, bot_id, log=False,
                                          anchor_first_row=True).iloc[1:]
    merged = merged[merged['timestamp'] > all_data['timestamp'].iloc[-1]]
    if len(merged) == 0:
        return all_data, 0
    merged = merged.assign(bot_action=None)
    for c in DISPLAY_COLS:
        merged[c] = np.nan
    out = pd.concat([all_data, merged], ignore_index=True)
    return out, len(merged)


def _correct_last_minute_if_changed(all_data, refetched_row, rel_tol=1e-4):
    """Re-check the most recent stored minute against a fresh fetch and overwrite
    it if GeckoTerminal revised it after our read. The features are recomputed
    from the raw OHLCV every minute, so nothing else needs repairing (the
    state machine already acted on the provisional value; that cannot be undone).
    The fetched candle is first put on the research fill rule: with zero volume
    it is a flat candle at the last real close (= the previous stored close,
    which is already on the rule). Returns (all_data, changed)."""
    last_idx = len(all_data) - 1
    stored = all_data.iloc[last_idx]
    cols = ['open', 'high', 'low', 'close', 'volume']
    new = {c: float(refetched_row[c]) for c in cols}
    if new['volume'] <= 0:
        flat = float(all_data['close'].iloc[last_idx - 1]) if last_idx > 0 else new['close']
        new.update(open=flat, high=flat, low=flat, close=flat, volume=0.0)
    changed = any(abs(new[c] - float(stored[c]))
                  > rel_tol * max(abs(float(stored[c])), 1e-12) for c in cols)
    if changed:
        for c in cols:
            all_data.at[last_idx, c] = new[c]
    return all_data, changed


def _strategy_summary(engine):
    c = engine.cfg
    tp = (f"take-profit +{c['activation_threshold'] * 100:.1f}% (peak close vs the segment reference)"
          if engine.tp_enabled else "take-profit OFF")
    se = (f"signal exit ON (direction < {c['se_dir']:.3f} or move < {c['se_move']:.3f} after "
          f"{c['se_min_hold']} min)" if engine.signal_on else "signal exit OFF")
    return (f"Strategy A50, rebase mode: buy if combo >= {c['confidence_threshold']:.4f}, "
            f"pp15m <= {c['max_price_position_15m']:.3f}, pp1h <= {c['max_price_position_60m']:.3f}, "
            f"coin age >= {c['trading_start_minutes']} min; {tp}; loss floor "
            f"-{c['loss_floor_pct'] * 100:.1f}% x {c['loss_floor_consecutive_minutes']} min and stale "
            f"{c['stale_trade_minutes']} min start races (never sell); re-entry "
            f"{c['reentry_threshold_pct'] * 100:.2f}% below the sale, relaxing "
            f"{c['reentry_decay_rate'] * 100:.3f}%/min after {c['reentry_decay_start_minutes']} min; {se}")


# ========================================================================
# THE BOT
# ========================================================================

class Bot:
    def __init__(self, bot_id):
        self.bot_id = bot_id
        g = get_global_settings()
        b = get_bot(bot_id)
        if not b or b['status'] == 'inactive':
            log_message(f"Bot {bot_id} not configured", bot_id, 'ERROR')
            sys.exit(1)
        if not g:
            log_message("Global settings not found", bot_id, 'ERROR')
            sys.exit(1)
        self.tz = g['timezone']
        self.finalization_lag = g['finalization_lag_minutes']
        self.historical_days = g['historical_days'] or 30
        self.mode = g.get('trading_mode', 'paper')
        self.model_dir = g.get('model_dir') or DEFAULT_MODEL_DIR
        self.pool = b['pool_address']
        self.coin = b['coin_symbol']
        # Paper-mode static fee (global, chosen at wallet init); falls back to
        # the per-bot slippage. Shadow/live derive real costs from quotes.
        slip = g.get('static_trading_fee_pct')
        self.slippage = slip if slip is not None else b['slippage_pct']
        self.grace = int(g.get('grace_period_seconds') or 5) if self.mode == 'paper' else 0
        self.initial_capital = b['initial_capital_sol']
        self.max_impact = b.get('max_price_impact_pct') or DEFAULT_MAX_PRICE_IMPACT_PCT

        self.swap_ctx = None
        if self.mode in ('shadow', 'live'):
            from data_collection import resolve_token_mint_and_decimals
            mint, decimals = resolve_token_mint_and_decimals(self.pool, bot_id)
            if mint is None or decimals is None:
                log_message(f"Could not resolve token mint/decimals — required for {self.mode} mode. "
                            f"Exiting.", bot_id, 'ERROR')
                sys.exit(1)
            self.swap_ctx = {'token_mint': mint, 'token_decimals': decimals,
                             'slippage_bps': int(round((b.get('slippage_pct') or 0.015) * 10000)),
                             'wallet_pubkey': g.get('wallet_pubkey'), 'bot_id': bot_id}
            log_message(f"{self.mode.capitalize()} swap context ready: {mint} ({decimals} decimals)",
                        bot_id, 'INFO')

        # ---- strategy + models ----
        cfg = sync_strategy_config_from_file() or get_strategy_config()
        if not cfg or is_legacy_config(cfg):
            log_message("No valid strategy config (best_params.json missing or the old 5-tier "
                        "config is still stored). Exiting.", bot_id, 'ERROR')
            sys.exit(1)
        try:
            self.engine = SignalExitStrategy(cfg)
        except Exception as e:
            log_message(f"Invalid strategy config: {e}. Exiting.", bot_id, 'ERROR')
            sys.exit(1)
        log_message(_strategy_summary(self.engine), bot_id, 'INFO')
        if not self.engine.signal_on and any(k in cfg for k in SIGNAL_EXIT_PARAMS):
            log_message("Note: these parameters come from a run tuned WITH the signal exit, which is "
                        "switched off here; upload parameters tuned without it for the plain strategy.",
                        bot_id, 'WARNING')
        try:
            self.scorer = LiveScorer(self.model_dir)
        except Exception as e:
            log_message(f"Could not load the models from {self.model_dir}: {e}. Exiting.", bot_id, 'ERROR')
            sys.exit(1)
        log_message(f"Models loaded from {self.model_dir}: move ({len(self.scorer.move_features)} "
                    f"features), direction ({len(self.scorer.dir_features)} features)", bot_id, 'INFO')

        sol = get_latest_sol_price()
        self.sol_usd = sol['price_usd'] if sol else 130.0
        os.makedirs(f'data/bot_{bot_id}', exist_ok=True)

        # runtime state (refreshed from the DB each loop)
        self.capital = b['current_capital_sol'] or 0.0
        self.shares = b['shares_held'] or 0.0
        self.trades_count = b['trades_count'] or 0
        self.all_data = None
        self.state = None
        self.latest = {}          # latest minute's scores (for the log line)

    # ------------------------------------------------------------------ data
    def load_history(self):
        """Fresh start: fetch the coin's full history. Resume: load it from the
        DB and fill the gap. Returns the index of the first minute that still
        needs the state machine (None on a fresh start)."""
        bid = self.bot_id
        last_ts = get_latest_timestamp(bid)
        if last_ts is None:
            log_message("Fresh start: fetching the coin's full history...", bid, 'INFO')
            num_batches = int(self.historical_days * 1.5) + 1
            df = fetch_historical_ohlcv(self.pool, num_batches, self.tz, bid)
            if df is None or len(df) == 0:
                log_message("Could not fetch historical data", bid, 'ERROR')
                sys.exit(1)
            if not df.attrs.get('reached_start'):
                log_message(f"The history does not reach the coin's first minute within "
                            f"{self.historical_days} days ({num_batches} batches): the features "
                            f"(minutes since launch, launch / ATH features, the 720-min trading start) "
                            f"would not match the research data. Not starting; raise historical_days "
                            f"if the coin is older.", bid, 'ERROR')
                sys.exit(1)
            df = forward_fill_missing_minutes(df, self.tz, bid)
            df['bot_action'] = None
            disp = display_columns(df)
            for c in DISPLAY_COLS:
                df[c] = disp[c].to_numpy()
            self.all_data = df.reset_index(drop=True)
            log_message(f"Retrieved the full history: {len(df)} minutes "
                        f"({len(df) / 1440:.1f} days since the first candle)", bid, 'INFO')
            insert_ohlcv_data(bid, [_ohlcv_record(self.all_data, i) for i in range(len(self.all_data))])
            return None

        log_message("Resuming: loading the stored history...", bid, 'INFO')
        rows = get_all_ohlcv(bid)
        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(self.tz)
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = df[c].astype(float)
        df = df.sort_values('timestamp').drop_duplicates('timestamp', keep='last').reset_index(drop=True)
        if not check_minute_grid(df):
            n0 = len(df)
            actions = df.set_index('timestamp')['bot_action']
            df = forward_fill_missing_minutes(df[OHLCV_COLS], self.tz, bid)
            df['bot_action'] = df['timestamp'].map(actions)
            log_message(f"Stored history had gaps: forward-filled {len(df) - n0} minutes", bid, 'WARNING')
        for c in DISPLAY_COLS:
            df[c] = np.nan
        self.all_data = df
        n_stored = len(df)
        log_message(f"Loaded {n_stored} stored minutes", bid, 'INFO')

        since = int(df['timestamp'].iloc[-1].timestamp())
        gap = fetch_ohlcv_since(self.pool, since, self.tz, bid)
        if gap is not None and len(gap):
            gap = gap[gap['timestamp'] <= self._cutoff()]
            recheck = gap[gap['timestamp'] == df['timestamp'].iloc[-1]]
            if len(recheck) == 1:
                self.all_data, _ = _correct_last_minute_if_changed(self.all_data, recheck.iloc[0])
            self.all_data, n_new = _append_minutes(self.all_data, gap[gap['timestamp'] > df['timestamp'].iloc[-1]],
                                                   self.tz, bid)
        else:
            n_new = 0
        if not check_minute_grid(self.all_data):
            raise RuntimeError("history is not a gap-free minute grid after the gap fill")
        log_message(f"Gap filled: {n_new} new minutes", bid, 'INFO')
        return n_stored          # minutes from here on are new

    def _cutoff(self):
        now = datetime.now(tz=pd.Timestamp.now(tz=self.tz).tz)
        return (pd.Timestamp(now) - pd.Timedelta(minutes=self.finalization_lag)).replace(second=0, microsecond=0)

    # ------------------------------------------------------------------ state
    def load_state(self, bot_row):
        """The persisted state machine, or a best-effort hand-over from the old
        5-tier version (no saved state machine)."""
        bid = self.bot_id
        coin_start = _ts_utc(self.all_data['timestamp'].iloc[0])
        if bot_row.get('strategy_state'):
            st = StrategyState.from_json(bot_row['strategy_state'])
            if st.coin_start and st.coin_start != coin_start:
                shift = int((pd.Timestamp(coin_start) - pd.Timestamp(st.coin_start)).total_seconds() // 60)
                log_message(f"Stored history starts {shift} min after the saved state's minute 0; "
                            f"shifting the state's minute indices", bid, 'WARNING')
                for f in ('last_idx', 'last_sell_idx', 'block_ref_idx', 'orig_entry_idx',
                          'seg_entry_idx', 'trade_ref_idx'):
                    v = getattr(st, f)
                    if v is not None and v >= 0:
                        setattr(st, f, v - shift)
            st.coin_start = coin_start
            return st

        st = StrategyState(coin_start=coin_start)
        if bot_row.get('last_sell_close') and bot_row.get('last_sell_idx') is not None:
            st.last_sell_close = float(bot_row['last_sell_close'])
            st.last_sell_idx = int(bot_row['last_sell_idx'])
        if bot_row['position'] == 'open':
            orig_idx = self._idx_of(bot_row.get('original_entry_timestamp') or bot_row.get('entry_timestamp'))
            orig_price = bot_row.get('original_entry_price') or bot_row.get('entry_price_sol')
            if orig_idx is None or not orig_price:
                log_message("Open position from the old version cannot be located in the history; "
                            "force-sell it manually.", bid, 'ERROR')
                sys.exit(1)
            self.engine.open_position(st, orig_idx, orig_price)
            seg_idx = self._idx_of(bot_row.get('entry_timestamp'))
            if seg_idx is not None and bot_row.get('entry_price_sol'):
                st.seg_entry_idx, st.seg_entry_price = seg_idx, float(bot_row['entry_price_sol'])
            log_message("Adopted an open position from the old 5-tier version: new state machine "
                        "from a NORMAL phase (its old race state is not carried over).", bid, 'WARNING')
        return st

    def _idx_of(self, ts):
        if not ts:
            return None
        t = pd.Timestamp(ts)
        t = t.tz_localize(self.tz) if t.tzinfo is None else t
        hit = np.flatnonzero((pd.to_datetime(self.all_data['timestamp'], utc=True) == t.tz_convert('UTC')).to_numpy())
        return int(hit[0]) if len(hit) else None

    def _ts(self, idx):
        return str(self.all_data['timestamp'].iloc[idx])

    def persist(self):
        st, eng = self.state, self.engine
        is_open = st.position == 'open'
        update_bot_trading_state(
            self.bot_id,
            position=st.position,
            shares_held=self.shares if is_open else 0,
            entry_price_sol=st.seg_entry_price if is_open else None,
            entry_timestamp=self._ts(st.seg_entry_idx) if is_open else None,
            original_entry_price=st.orig_entry_price if is_open else None,
            original_entry_timestamp=self._ts(st.orig_entry_idx) if is_open else None,
            current_capital_sol=self.capital,
            trades_count=self.trades_count,
            n_reentries=st.n_reentries if is_open else 0,
            last_sell_close=st.last_sell_close,
            last_sell_idx=st.last_sell_idx,
            latest_activation_pct=eng.act if eng.tp_enabled else None,
            trade_phase=st.phase if is_open else PHASE_NORMAL,
            below_floor_count=st.below_floor if is_open else 0,
            max_price_seen=(st.max_price if is_open and st.max_price > 0 else None),
            strategy_state=st.to_json(),
        )

    # ------------------------------------------------------------------ minutes
    def process_minutes(self, first, last, allow_trading, catch_up=False, force_on_signal=False):
        """Scores + state machine for minutes first..last (in order). Buys only
        on `last` (the current minute); a sell decided on an earlier (backlog)
        minute is executed at once, at the current minute. Returns True if a
        sell-on-signal fired (the caller pauses the bot)."""
        bid = self.bot_id
        n_req = last - first + 1
        feats = compute_live_features(self.all_data, last_n=n_req)
        _fill_display_columns(self.all_data, first, last, feats)
        if feats is None:
            self.state.last_idx = max(self.state.last_idx, last)
            return False
        sc = self.scorer.score(feats)
        pred_rows = []
        minutes = []
        for k, idx in enumerate(feats['minute_idx'].to_numpy()):
            idx = int(idx)
            close = float(self.all_data['close'].iloc[idx])
            m = MinuteInput(idx=idx, close=close, combo_pct=float(sc['combo_pct'][k]),
                            dir_pct=float(sc['dir_pct'][k]), move_pct=float(sc['move_pct'][k]),
                            pp_15m=float(feats['price_position_15m'].iloc[k]),
                            pp_1h=float(feats['price_position_1h'].iloc[k]))
            minutes.append((m, float(feats['estimated_sol_1pct_impact'].iloc[k])))
            pred_rows.append({'timestamp': self._ts(idx), 'combo_pct': m.combo_pct, 'price': close,
                              'dir_pct': m.dir_pct, 'move_pct': m.move_pct,
                              'raw_move': float(sc['raw_move'][k]), 'raw_dir': float(sc['raw_dir'][k]),
                              'price_position_15m': m.pp_15m, 'price_position_1h': m.pp_1h})
        insert_predictions(bid, pred_rows)
        self.latest = {'combo': minutes[-1][0].combo_pct, 'dir': minutes[-1][0].dir_pct,
                       'move': minutes[-1][0].move_pct}

        paused_by_signal = False
        for m, liq in minutes:
            if m.idx <= self.state.last_idx:
                continue
            is_current = (m.idx == last) and allow_trading and not catch_up
            res = self.engine.step(self.state, m)
            for ev in res.events:
                log_message(f"{ev} @ {self._ts(m.idx)[11:16]}", bid,
                            'WARNING' if 'loss floor armed' in ev else 'INFO')
            if res.action == ACTION_BUY:
                if is_current:
                    self.execute_buy(m, liq)
                elif not catch_up:
                    log_message(f"Entry signal on a backlog minute ({self._ts(m.idx)[11:16]}) — not "
                                f"bought (only the current minute is traded)", bid, 'INFO')
            elif res.action == ACTION_SELL:
                if not allow_trading:
                    log_message(f"Sell signal ({res.reason}) while not running — not executed", bid, 'WARNING')
                    continue
                reason = self._reason_label(res.reason)
                if force_on_signal:
                    reason = f"Force sell on signal ({reason})"
                late = m.idx != last
                self.execute_sell(last, reason + (f" [signal at {self._ts(m.idx)[11:16]}, filled late]"
                                                  if late else ""))
                if force_on_signal:
                    paused_by_signal = True
                break                   # minutes after the sale were consumed by close_position
        self.state.last_idx = max(self.state.last_idx, last)
        return paused_by_signal

    def _reason_label(self, reason):
        if reason == REASON_TAKE_PROFIT:
            return f"Take-profit (peak +{self.engine.act * 100:.1f}% over the reference)"
        if reason == REASON_SIGNAL_EXIT:
            return "Signal exit"
        return reason

    # ------------------------------------------------------------------ orders
    def execute_buy(self, m, liq):
        bid, eng = self.bot_id, self.engine
        amount, cap = eng.buy_amount(self.capital, liq, self.max_impact)
        if amount <= 0:
            log_message(f"Skipping buy at {format_price_si(m.close)}: buy amount {amount:.6f} SOL", bid, 'WARNING')
            return
        result = execute_buy(amount, m.close, self.slippage, self.mode, self.swap_ctx)
        if result.get('used_fallback'):
            log_message("Shadow buy used static fallback (no live quote)", bid, 'WARNING')
        fee_capital = result.get('capital_fee_sol', 0) or 0     # real SOL burned (shadow/live)
        self.capital = max(0.0, self.capital - amount - fee_capital)
        self.shares = result['shares_received']
        eng.open_position(self.state, m.idx, result['fill_price_sol'])
        self.all_data.at[m.idx, 'bot_action'] = 'BUY'
        ts = self._ts(m.idx)
        insert_open_trade(bid, ts, result['fill_price_sol'], self.sol_usd, amount, result['fee_sol'])
        deduct_trading_fee(result['fee_sol'])
        self.persist()
        cap_txt = f"{cap:.4f} SOL" if cap is not None else "n/a"
        log_message(f"BUY at {format_price_si(result['fill_price_sol'])} SOL, spent {amount:.4f} SOL "
                    f"(liquidity cap {cap_txt}), combo {m.combo_pct:.3f}, direction {m.dir_pct:.3f}, "
                    f"move {m.move_pct:.3f}, pp15m {m.pp_15m:.2f}, pp1h {m.pp_1h:.2f}", bid, 'INFO')

    def execute_sell(self, idx, reason):
        """Sell the whole position at minute idx (the current minute)."""
        bid, eng, st = self.bot_id, self.engine, self.state
        close = float(self.all_data['close'].iloc[idx])
        orig = st.orig_entry_price
        orig_idx = st.orig_entry_idx
        n_re = st.n_reentries
        exit_fill = eng.exit_fill_price(close, self.slippage)       # paper: close x (1 - slippage)
        result = execute_sell(self.shares, exit_fill, orig, self.initial_capital,
                              self.slippage, self.mode, self.swap_ctx)
        if result.get('used_fallback'):
            log_message("Shadow sell used static fallback (no live quote)", bid, 'WARNING')
        self.capital += result['proceeds_sol'] - (result.get('capital_fee_sol', 0) or 0)
        self.all_data.at[idx, 'bot_action'] = 'SELL'
        trade = get_open_trade_for_bot(bid)
        if trade:
            update_trade_on_exit(trade['id'], self._ts(idx), result['fill_price_sol'], self.sol_usd,
                                 result['pnl_sol'], result['pnl_pct'], idx - orig_idx, reason,
                                 result['fee_sol'])
        deduct_trading_fee(result['fee_sol'])
        eng.close_position(st, idx, close)
        self.shares = 0.0
        self.trades_count += 1
        self.persist()
        insert_ohlcv_data(bid, [_ohlcv_record(self.all_data, idx)])   # SELL marker for the chart
        log_message(f"SELL {self.coin} at {format_price_si(result['fill_price_sol'])} SOL, "
                    f"P&L {result['pnl_pct'] * 100:+.2f}% ({result['pnl_sol']:+.4f} SOL), held "
                    f"{idx - orig_idx} min, rebases {n_re}, fee {result['fee_sol']:.6f} SOL, "
                    f"reason: {reason}", bid, 'INFO')

    # ------------------------------------------------------------------ snapshots
    def snapshots(self, first, last):
        fees = get_bot_cumulative_fees_sol(self.bot_id)
        for idx in range(first, last + 1):
            price = float(self.all_data['close'].iloc[idx])
            open_val = self.shares * price if self.state.position == 'open' else 0.0
            insert_capital_snapshot(self.bot_id, self._ts(idx), self.capital, open_val, fees)
        write_wallet_snapshot()


# ========================================================================
# MAIN
# ========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bot-id', type=int, required=True)
    args = parser.parse_args()
    BOT_ID = args.bot_id

    bot = Bot(BOT_ID)
    update_bot_runtime(BOT_ID, initializing=1)
    first_new = bot.load_history()
    bot_row = get_bot(BOT_ID)
    bot.state = bot.load_state(bot_row)
    n = len(bot.all_data)

    if first_new is None:
        # Fresh start: nothing to replay. Score the recent minutes for the chart.
        bot.state.last_idx = n - 1
        lo = max(START_IDX, n - CHART_BACKFILL_MINUTES)
        if lo <= n - 1:
            bot.process_minutes(lo, n - 1, allow_trading=False, catch_up=True)
    else:
        # Resume: replay every minute the state machine has not seen yet
        # (no buys on past minutes; a due sell is executed at once).
        start = max(bot.state.last_idx + 1, first_new if bot_row.get('strategy_state') is None else 0)
        start = max(start, n - 20000)
        if start <= n - 1:
            log_message(f"Replaying {n - start} minute(s) through the state machine", BOT_ID, 'INFO')
            running = (get_bot(BOT_ID) or {}).get('status') == 'running'
            bot.process_minutes(start, n - 1, allow_trading=running, catch_up=True)
        if first_new < n:
            insert_ohlcv_data(BOT_ID, [_ohlcv_record(bot.all_data, i) for i in range(first_new, n)])
            bot.snapshots(max(first_new, n - 5), n - 1)
    bot.persist()

    start_timestamp_unix = int(bot.all_data['timestamp'].max().timestamp())
    update_bot_runtime(BOT_ID, initializing=0)
    log_message(f"Starting live trading for {bot.coin} (mode: {bot.mode}, "
                f"{n} minutes of history, position {bot.state.position})", BOT_ID, 'INFO')

    try:
        _prev_debug_mode = None
        while True:
            # 1. Read state from DB
            b = get_bot(BOT_ID)
            if not b or b['status'] not in ('running', 'paused'):
                log_message(f"Bot {BOT_ID} status: {b['status'] if b else 'missing'}. Exiting.", BOT_ID, 'WARNING')
                break
            debug_mode = bool(b.get('debug_mode'))
            if _prev_debug_mode is not None and debug_mode != _prev_debug_mode:
                log_message(f"Advanced debugging {'enabled' if debug_mode else 'disabled'}", BOT_ID, 'INFO')
            _prev_debug_mode = debug_mode
            is_paused = (b['status'] == 'paused')
            sol = get_latest_sol_price()
            if sol:
                bot.sol_usd = sol['price_usd']

            # 2. Force sell immediate
            force_immediate, force_on_signal = check_force_sell_files(BOT_ID)
            if force_immediate and bot.state.position == 'open':
                log_message("FORCE SELL IMMEDIATE triggered", BOT_ID, 'WARNING')
                cur = len(bot.all_data) - 1
                bot.state.last_idx = max(bot.state.last_idx, cur)
                bot.execute_sell(cur, "Force sell immediate")
                update_bot_runtime(BOT_ID, status='paused')
                clear_force_sell_files(BOT_ID)
                bot.snapshots(cur, cur)
                continue

            # 2.5 Paused: no data collection, keep the capital charts moving
            if is_paused:
                last = get_latest_ohlcv(BOT_ID, limit=1)
                if last:
                    price = last[0]['close']
                    open_val = bot.shares * price if bot.state.position == 'open' else 0.0
                    insert_capital_snapshot(BOT_ID, datetime.now().strftime('%Y-%m-%d %H:%M:00'),
                                            bot.capital, open_val, get_bot_cumulative_fees_sol(BOT_ID))
                write_wallet_snapshot()
                target = time.monotonic() + 30
                while time.monotonic() < target:
                    time.sleep(min(1.0, max(0.0, target - time.monotonic())))
                    s = get_bot(BOT_ID)
                    if not s or s['status'] != 'paused':
                        break
                continue

            # 3. Wait for the next candle (+ grace period), in <= 1 s chunks
            now = datetime.now(tz=pd.Timestamp.now(tz=bot.tz).tz)
            wait = (bot.grace - now.second) if now.second < bot.grace else (60 - now.second + bot.grace)
            target = time.monotonic() + wait
            interrupted = False
            while time.monotonic() < target:
                time.sleep(min(1.0, max(0.0, target - time.monotonic())))
                s = get_bot(BOT_ID)
                if s is not None:
                    _wd = bool(s.get('debug_mode'))
                    if _prev_debug_mode is not None and _wd != _prev_debug_mode:
                        log_message(f"Advanced debugging {'enabled' if _wd else 'disabled'}", BOT_ID, 'INFO')
                        _prev_debug_mode = _wd
                if not s or s['status'] != 'running':
                    interrupted = True
                    break
            if interrupted:
                continue

            # 4. Fetch (from ~2 min before the cursor, so the last stored minute
            #    comes back too and can be checked for a late revision)
            prev_last_ts = bot.all_data['timestamp'].iloc[-1]
            t0 = time.monotonic()
            new_df = fetch_ohlcv_from_timestamp(bot.pool, start_timestamp_unix - 120, bot.tz, BOT_ID,
                                                debug=debug_mode)
            api_secs = time.monotonic() - t0
            if new_df is None or len(new_df) == 0:
                log_message(f"Fetch returned no data{f' (api={api_secs:.2f}s)' if debug_mode else ''} — "
                            f"skipping this minute; will catch up on the next successful fetch",
                            BOT_ID, 'WARNING')
                continue
            new_df = new_df[new_df['timestamp'] <= bot._cutoff()]

            # 4a. Late revision of the previous minute
            recheck = new_df[new_df['timestamp'] == prev_last_ts]
            if len(recheck) == 1:
                bot.all_data, corrected = _correct_last_minute_if_changed(bot.all_data, recheck.iloc[0])
                if corrected:
                    insert_ohlcv_data(BOT_ID, [_ohlcv_record(bot.all_data, len(bot.all_data) - 1)])
                    log_message(f"Corrected previous minute @ {str(prev_last_ts)[11:16]} (late revision)",
                                BOT_ID, 'INFO')

            # 4b. Genuinely new minutes, on a gap-free grid
            first = len(bot.all_data)
            bot.all_data, n_new = _append_minutes(bot.all_data, new_df[new_df['timestamp'] > prev_last_ts],
                                                  bot.tz, BOT_ID)
            if n_new == 0:
                log_message(f"No new finalized candle past cutoff{f' (api={api_secs:.2f}s)' if debug_mode else ''}"
                            f" — nothing to append this minute", BOT_ID, 'INFO')
                continue
            last = len(bot.all_data) - 1

            # Re-check status right before acting (a pause / stop during the
            # wait or fetch must not produce one more trade).
            fresh = get_bot(BOT_ID)
            allow_trading = bool(fresh and fresh['status'] == 'running')
            if not allow_trading:
                log_message(f"Status changed to '{fresh['status'] if fresh else 'missing'}' during the "
                            f"candle wait/fetch — recording data but skipping trade actions this minute.",
                            BOT_ID, 'INFO')

            # 5. Scores + state machine
            t1 = time.monotonic()
            paused = bot.process_minutes(first, last, allow_trading, force_on_signal=force_on_signal)
            proc_secs = time.monotonic() - t1
            bot.persist()
            if paused:
                update_bot_runtime(BOT_ID, status='paused')
                clear_force_sell_files(BOT_ID)
                log_message(f"Bot {BOT_ID} paused after force sell on signal", BOT_ID, 'INFO')

            # 6. Save OHLCV + snapshots
            insert_ohlcv_data(BOT_ID, [_ohlcv_record(bot.all_data, i) for i in range(first, last + 1)])
            start_timestamp_unix = int(bot.all_data['timestamp'].iloc[-1].timestamp())
            bot.snapshots(first, last)

            # 7. Log line
            st = bot.state
            price = float(bot.all_data['close'].iloc[-1])
            value = (bot.capital + (bot.shares * price if st.position == 'open' else 0)) * bot.sol_usd
            pos = (f"open ({st.phase}, held {last - st.orig_entry_idx} min)" if st.position == 'open'
                   else 'flat')
            lt = bot.latest
            dbg = f" | api={api_secs:.2f}s compute={proc_secs:.2f}s" if debug_mode else ""
            if debug_mode:
                liq = bot.all_data['estimated_sol_1pct_impact'].iloc[-1]
                if liq is not None and not pd.isna(liq) and liq > 0:
                    log_message(f"Liquidity: est={liq:.4f} SOL/1% impact | cap={liq * bot.max_impact * 100:.4f} "
                                f"SOL @ {bot.max_impact * 100:.3f}%", BOT_ID, 'INFO')
            log_message(f"Candles: {len(bot.all_data)} | Capital: {bot.capital:.4f} SOL (${value:.2f}) | "
                        f"Trades: {bot.trades_count} | Position: {pos} | combo {lt.get('combo', float('nan')):.3f} "
                        f"dir {lt.get('dir', float('nan')):.3f} move {lt.get('move', float('nan')):.3f}{dbg}",
                        BOT_ID, 'INFO')

    except KeyboardInterrupt:
        log_message(f"Bot {BOT_ID} interrupted", BOT_ID, 'WARNING')
    except Exception as e:
        log_message(f"Bot {BOT_ID} error: {e}", BOT_ID, 'ERROR')
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}", BOT_ID, 'ERROR')
    finally:
        update_bot_runtime(BOT_ID, status='stopped', pid=None, stopped_at=datetime.now())
        if bot.state is not None:
            if bot.state.position == 'open':
                log_message(f"Bot {BOT_ID} stopped with an OPEN position ({bot.shares} shares). The "
                            f"saved state machine resumes it on the next start; force-sell to close it.",
                            BOT_ID, 'WARNING')
            bot.persist()
        log_message(f"Bot {BOT_ID} stopped", BOT_ID, 'INFO')


if __name__ == '__main__':
    main()
