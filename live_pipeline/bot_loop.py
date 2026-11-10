"""
Bot Loop - Main orchestrator for each trading bot process.
Launched as a subprocess per bot by bot_manager.py.

Flow: fetch data -> compute features -> score -> decide -> execute -> record

Data: the coin's full available 1-minute history from its first candle, filled as the
data notebooks do (every zero-volume minute flat at the last real close). The V9
features (features.py) are computed from that history; the move and direction models
(scoring.py) give combo = P(move) x P(direction), the buy score.

Trade state machine (strategy.py, hard-coded parameters):
  - position == 'closed':  look for an entry candidate (combo >= combo_cutoff + price
                           positions + re-entry threshold)
  - position == 'open':    drive the per-minute state machine
        * three phases:  NORMAL  ->  STALE_RACE / LOSS_FLOOR_RACE
        * priority within a minute: activation > loss-floor > stale
        * in a race, a qualifying candidate re-bases the trade (no new buy)
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
    insert_ohlcv_data, insert_prediction,
    insert_open_trade,
    update_trade_on_exit, get_open_trade_for_bot,
    deduct_trading_fee, get_bot_cumulative_fees_sol,
    insert_capital_snapshot, log_message,
    get_latest_timestamp, get_latest_ohlcv, get_all_ohlcv,
    insert_wallet_capital_snapshot, get_wallet_status, get_all_bots_from_db
)
from features import LiveFeatureEngine, START_IDX
from scoring import LiveScorer, verify_models
from strategy import (
    TUNED_PARAMS, FIXED_PARAMS, PRICE_POSITION_60M_FEATURE, use_strategy,
    passes_entry_filters, passes_reentry_threshold,
    step_open_position, compute_exit_price,
    PHASE_NORMAL, PHASE_STALE_RACE, PHASE_LOSS_FLOOR_RACE
)
from execution import execute_buy, execute_sell
from data_collection import (
    fetch_historical_ohlcv,
    fetch_ohlcv_since,
    fetch_ohlcv_from_timestamp,
    backfill_missing_candles
)

# ========================================================================
# CONSTANTS
# ========================================================================

OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']

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


def get_entry_idx(all_data, entry_timestamp):
    """Look up DataFrame index from stored entry timestamp."""
    if not entry_timestamp:
        return None
    mask = all_data['timestamp'].astype(str) == entry_timestamp
    matches = all_data.index[mask]
    return int(matches[0]) if len(matches) > 0 else None

def _ohlcv_record(row):
    """Build the OHLCV dict for one DataFrame row, as stored in ohlcv_data.

    Single source of truth for the column set written by every insert path
    (historical load, gap backfill, live minute, force sell).
    """
    return {
        'timestamp': str(row['timestamp']),
        'open': row['open'], 'high': row['high'],
        'low': row['low'], 'close': row['close'],
        'volume': row['volume'], 'bot_action': row['bot_action'],
        'backfilled': row.get('backfilled'),
        'estimated_sol_1pct_impact': row.get('estimated_sol_1pct_impact'),
    }


def _last_real_close(all_data, upto):
    """Close of the last real (volume > 0) candle among rows [0, upto), or None."""
    vol = all_data['volume'].to_numpy(dtype=np.float64)[:upto]
    real = np.flatnonzero(vol > 0)
    return float(all_data['close'].iat[int(real[-1])]) if len(real) else None


def _store_liquidity(all_data, engine, start_idx):
    """Copy the engine's liquidity estimate into all_data for rows >= start_idx (for the
    ohlcv_data table and the buy-size cap)."""
    if 'estimated_sol_1pct_impact' not in all_data.columns:
        all_data['estimated_sol_1pct_impact'] = np.nan
    est = engine.liq.arrays()[0]
    all_data.loc[start_idx:, 'estimated_sol_1pct_impact'] = est[start_idx:len(all_data)]
    return all_data


def _correct_last_minute_if_changed(all_data, refetched_row, engine, rel_tol=1e-4):
    """Re-check the most recent stored minute against a fresh fetch and repair it
    if GeckoTerminal revised it after our read.

    The just-closed candle can be read while still provisional; by the next poll
    (~60s later) it is finalized. The refetched candle first gets the same fill rule as
    every other minute (zero volume -> flat at the last real close). If it differs from
    the stored row, the row is overwritten and the feature engine re-processes it, so a
    provisional value cannot leak into later features.

    Returns (all_data, changed: bool).
    """
    last_idx = len(all_data) - 1
    stored = all_data.iloc[last_idx]

    fixed = {c: float(refetched_row[c]) for c in OHLCV_COLS}
    if fixed['volume'] <= 0:
        prc = _last_real_close(all_data, last_idx)
        if prc is None:                      # no real candle yet: the fill uses the row's own close
            prc = fixed['close']
        fixed.update(open=prc, high=prc, low=prc, close=prc, volume=0.0)

    changed = any(
        abs(fixed[c] - float(stored[c])) > rel_tol * max(abs(float(stored[c])), 1e-12)
        for c in OHLCV_COLS
    )
    if not changed:
        return all_data, False

    for c in OHLCV_COLS:
        all_data.at[last_idx, c] = fixed[c]
    all_data.at[last_idx, 'backfilled'] = int(fixed['volume'] <= 0)
    engine.revise(all_data, last_idx)
    all_data = _store_liquidity(all_data, engine, last_idx)
    return all_data, True


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


# ========================================================================
# MAIN
# ========================================================================

def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--bot-id', type=int, required=True)
    args = parser.parse_args()
    BOT_ID = args.bot_id

    # Load config from database
    global_config = get_global_settings()
    bot_config = get_bot(BOT_ID)

    if not bot_config or bot_config['status'] == 'inactive':
        log_message(f"Bot {BOT_ID} not configured", BOT_ID, 'ERROR')
        sys.exit(1)
    if not global_config:
        log_message(f"Global settings not found", BOT_ID, 'ERROR')
        sys.exit(1)

    # Extract settings
    TIMEZONE = global_config['timezone']
    FINALIZATION_LAG_MINUTES = global_config['finalization_lag_minutes']
    MODEL_DIR = global_config.get('model_dir') or 'artifacts'

    # The bot's strategy (parameters + combo cutoff) from the model folder, the one chosen
    # when the bot was set up (the first strategy file if none was stored); it must belong
    # to the models in that folder. Loaded before the settings below use FIXED_PARAMS.
    try:
        strategy_doc = use_strategy(MODEL_DIR, bot_config.get('strategy_name'))
        verify_models(MODEL_DIR, strategy_doc)
    except Exception as e:
        log_message(f"Could not load the strategy: {e}", BOT_ID, 'ERROR')
        sys.exit(1)
    _sel = strategy_doc.get('selection', {})
    log_message(f"Strategy '{strategy_doc['name']}' (run {_sel.get('run', '?')} of {_sel.get('n_runs', '?')}, "
                f"chosen by {_sel.get('selected_by', '?')} score, test ${_sel.get('test_score', float('nan')):.2f})",
                BOT_ID, 'INFO')
    TRADING_MODE = global_config.get('trading_mode', 'paper')

    POOL_ADDRESS = bot_config['pool_address']
    COIN = bot_config['coin_symbol']
    # Paper-mode static trading fee is a global setting chosen at wallet init;
    # fall back to the per-bot slippage if it isn't set. (Shadow/live will later
    # derive real fees from the transaction instead.)
    SLIPPAGE_PCT = global_config.get('static_trading_fee_pct')
    if SLIPPAGE_PCT is None:
        SLIPPAGE_PCT = bot_config['slippage_pct']
    if SLIPPAGE_PCT is None:
        SLIPPAGE_PCT = FIXED_PARAMS['slippage_pct']
    # Per-minute grace period (seconds) chosen at wallet init. Paper only —
    # shadow/live quote at decision time and don't use a grace window.
    GRACE_PERIOD_SECONDS = int(global_config.get('grace_period_seconds') or 5)
    if TRADING_MODE != 'paper':
        GRACE_PERIOD_SECONDS = 0

    # Shadow/live swap context: resolve the base token mint + decimals once so
    # execute_buy/execute_sell can quote real Jupiter swaps at decision time.
    # (Paper never quotes, so SWAP_CTX stays None.)
    SWAP_CTX = None
    if TRADING_MODE in ('shadow', 'live'):
        from data_collection import resolve_token_mint_and_decimals
        _mint, _decimals = resolve_token_mint_and_decimals(POOL_ADDRESS, BOT_ID)
        if _mint is None or _decimals is None:
            log_message(
                "Could not resolve token mint/decimals — required for "
                f"{TRADING_MODE} mode. Exiting.", BOT_ID, 'ERROR'
            )
            sys.exit(1)
        SWAP_CTX = {
            'token_mint': _mint,
            'token_decimals': _decimals,
            'slippage_bps': int(round((bot_config.get('slippage_pct') or 0.015) * 10000)),
            'wallet_pubkey': global_config.get('wallet_pubkey'),
            'bot_id': BOT_ID,
        }
        log_message(
            f"{TRADING_MODE.capitalize()} swap context ready: {_mint} "
            f"({_decimals} decimals)", BOT_ID, 'INFO'
        )

    INITIAL_CAPITAL_SOL = bot_config['initial_capital_sol']
    # Per-bot liquidity cap on buy size (the notebook's desired_price_impact_pct):
    # buy at most estimated_sol_1pct_impact * (pct * 100) SOL.
    DESIRED_PRICE_IMPACT_PCT = (bot_config.get('max_price_impact_pct')
                                or FIXED_PARAMS['desired_price_impact_pct'])

    os.makedirs(f'data/bot_{BOT_ID}', exist_ok=True)

    # Load the move + direction models (combo = P_move x P_dir)
    log_message(f"Loading the move and direction models from {MODEL_DIR}/ ...", BOT_ID, 'INFO')
    try:
        scorer = LiveScorer(MODEL_DIR)
    except Exception as e:
        log_message(f"Could not load the models: {e}", BOT_ID, 'ERROR')
        sys.exit(1)
    log_message(
        f"Models loaded (move {len(scorer.move_features)} features, direction "
        f"{len(scorer.dir_features)} features); buy when combo >= "
        f"{TUNED_PARAMS['combo_cutoff']:.6f}", BOT_ID, 'INFO')

    # Get SOL price
    sol_price_data = get_latest_sol_price()
    SOL_PRICE_USD = sol_price_data['price_usd'] if sol_price_data else 130.0

    # Mark the bot as initialising while it collects historical/gap data. The
    # dashboard shows this state (blue card + INITIALISING badge) until live
    # trading begins. The manager health-check only touches `status`, so it
    # won't clear this flag.
    update_bot_runtime(BOT_ID, initializing=1)

    # Check if this is a fresh start or a resume
    last_stored_timestamp = get_latest_timestamp(BOT_ID)

    def _finalized_cutoff():
        cutoff = datetime.now(tz=pd.Timestamp.now(tz=TIMEZONE).tz) - pd.Timedelta(minutes=FINALIZATION_LAG_MINUTES)
        return cutoff.replace(second=0, microsecond=0)

    if last_stored_timestamp is None:
        # FRESH START — fetch the full available history (from the coin's first candle)
        log_message("Fresh start: fetching the full history...", BOT_ID, 'INFO')
        all_data = fetch_historical_ohlcv(POOL_ADDRESS, TIMEZONE, BOT_ID)

        if all_data is None or len(all_data) == 0:
            log_message("Could not fetch the historical data", BOT_ID, 'ERROR')
            sys.exit(1)

        all_data = all_data[all_data['timestamp'] <= _finalized_cutoff()]
        all_data = backfill_missing_candles(all_data)
        all_data['bot_action'] = None
        log_message(f"Retrieved {len(all_data)} candles from {all_data['timestamp'].iloc[0]}",
                    BOT_ID, 'INFO')

        log_message("Computing the liquidity history...", BOT_ID, 'INFO')
        engine = LiveFeatureEngine(all_data[['timestamp'] + OHLCV_COLS])
        all_data = _store_liquidity(all_data, engine, 0)

        # Save to database
        log_message("Saving historical data to database...", BOT_ID, 'INFO')
        insert_ohlcv_data(BOT_ID, [_ohlcv_record(all_data.iloc[idx]) for idx in range(len(all_data))])

    else:
        # RESUME — load the stored history and fetch the gap
        log_message("Resuming: loading existing OHLCV from database...", BOT_ID, 'INFO')

        existing_ohlcv = get_all_ohlcv(BOT_ID)
        if not existing_ohlcv:
            log_message("No existing OHLCV found despite timestamp existing", BOT_ID, 'ERROR')
            sys.exit(1)

        all_data = pd.DataFrame(existing_ohlcv)
        all_data['timestamp'] = pd.to_datetime(all_data['timestamp'], utc=True).dt.tz_convert(TIMEZONE)
        for col in OHLCV_COLS:
            all_data[col] = all_data[col].astype(float)
        all_data = all_data[['timestamp'] + OHLCV_COLS + ['bot_action', 'backfilled']]
        all_data = all_data.sort_values('timestamp').reset_index(drop=True)
        log_message(f"Loaded {len(all_data)} existing candles", BOT_ID, 'INFO')

        # Fetch every candle since the last stored one (paging back as far as needed)
        last_ts = all_data['timestamp'].max()
        log_message(f"Fetching gap candles since {last_ts}...", BOT_ID, 'INFO')
        gap_df = fetch_ohlcv_since(POOL_ADDRESS, last_ts, TIMEZONE, BOT_ID)
        if gap_df is None:
            log_message("Could not fetch the gap since the last stored candle", BOT_ID, 'ERROR')
            sys.exit(1)
        gap_df = gap_df[gap_df['timestamp'] <= _finalized_cutoff()]
        gap_df = backfill_missing_candles(
            gap_df, prev_last_ts=last_ts,
            prev_real_close=_last_real_close(all_data, len(all_data)))

        new_start_idx = len(all_data)
        if gap_df is not None and len(gap_df) > 0:
            gap_df = gap_df.copy()
            gap_df['bot_action'] = None
            all_data = pd.concat([all_data, gap_df], ignore_index=True)
            log_message(f"Fetched {len(gap_df)} gap candles", BOT_ID, 'INFO')
        else:
            log_message("No gap candles to fetch", BOT_ID, 'INFO')

        log_message("Rebuilding the liquidity history...", BOT_ID, 'INFO')
        engine = LiveFeatureEngine(all_data[['timestamp'] + OHLCV_COLS])
        all_data = _store_liquidity(all_data, engine, 0)

        if len(all_data) > new_start_idx:
            gap_records = [_ohlcv_record(all_data.iloc[idx]) for idx in range(new_start_idx, len(all_data))]
            insert_ohlcv_data(BOT_ID, gap_records)
            log_message(f"Saved {len(gap_records)} gap candles to database", BOT_ID, 'INFO')

    start_timestamp_unix = int(all_data['timestamp'].max().timestamp())
    update_bot_runtime(BOT_ID, initializing=0)   # data collected; entering live loop
    log_message(f"Starting live trading for {COIN} (mode: {TRADING_MODE})", BOT_ID, 'INFO')

    # Init local trade-state vars
    capital_sol = INITIAL_CAPITAL_SOL
    position = 'closed'
    shares_held = 0
    entry_price_sol = None
    entry_timestamp = None
    trades_count = 0

    # ================================================================
    # MAIN LOOP
    # ================================================================
    try:
        _prev_debug_mode = None  # baseline for the debug-toggle transition log
        while True:
            # 1. Read state from DB
            bot_state = get_bot(BOT_ID)
            if not bot_state or bot_state['status'] not in ('running', 'paused'):
                log_message(f"Bot {BOT_ID} status: {bot_state['status'] if bot_state else 'missing'}. Exiting.", BOT_ID, 'WARNING')
                break

            # Per-bot advanced-debug toggle, re-read every iteration so the
            # dashboard button takes effect within a minute without a restart.
            # When off, the extra timing/diff values below are neither computed
            # nor logged.
            debug_mode = bool(bot_state.get('debug_mode'))

            # Log only when the toggle actually changes. The "disabled" notice
            # is intentionally NOT gated by debug_mode — it must print even
            # though debug is now off. First iteration just sets the baseline.
            if _prev_debug_mode is not None and debug_mode != _prev_debug_mode:
                log_message(
                    f"Advanced debugging {'enabled' if debug_mode else 'disabled'}",
                    BOT_ID, 'INFO')
            _prev_debug_mode = debug_mode

            is_paused = (bot_state['status'] == 'paused')
            position = bot_state['position']
            capital_sol = bot_state['current_capital_sol'] or 0
            shares_held = bot_state['shares_held'] or 0
            entry_price_sol = bot_state['entry_price_sol']
            trades_count = bot_state['trades_count'] or 0
            entry_timestamp = bot_state.get('entry_timestamp')
            entry_idx = get_entry_idx(all_data, entry_timestamp)

            # Trade state
            original_entry_price = bot_state.get('original_entry_price')
            original_entry_timestamp = bot_state.get('original_entry_timestamp')
            trade_phase = bot_state.get('trade_phase') or PHASE_NORMAL
            below_floor_count = bot_state.get('below_floor_count') or 0
            max_price_seen = bot_state.get('max_price_seen')
            if max_price_seen is None:
                max_price_seen = entry_price_sol if entry_price_sol else 0.0
            n_reentries = bot_state.get('n_reentries', 0) or 0
            last_sell_close = bot_state.get('last_sell_close')
            last_sell_idx = bot_state.get('last_sell_idx')

            # Update SOL price
            sol_price_data = get_latest_sol_price()
            if sol_price_data:
                SOL_PRICE_USD = sol_price_data['price_usd']

            # 2. Check force sell
            force_immediate, force_on_signal = check_force_sell_files(BOT_ID)

            if force_immediate and position == 'open':
                log_message("FORCE SELL IMMEDIATE triggered", BOT_ID, 'WARNING')
                current_idx = len(all_data) - 1
                current_close = all_data.iloc[current_idx]['close']

                # P&L always measured against original entry
                use_entry = original_entry_price or entry_price_sol
                exit_price = compute_exit_price(current_close, SLIPPAGE_PCT)

                result = execute_sell(
                    shares_held, exit_price, use_entry,
                    INITIAL_CAPITAL_SOL, SLIPPAGE_PCT, TRADING_MODE, SWAP_CTX
                )
                if result.get('used_fallback'):
                    log_message("Shadow sell used static fallback (no live quote)",
                                BOT_ID, 'WARNING')

                all_data.at[current_idx, 'bot_action'] = 'SELL'

                # Persist the SELL marker to the OHLCV table immediately.
                # The bulk OHLCV save at the bottom of the loop is skipped
                # by the `continue` below, so without this the chart never
                # sees the force-sell and no red triangle gets drawn.
                # insert_ohlcv_data uses INSERT OR REPLACE on
                # (bot_id, timestamp), so this cleanly overwrites the
                # existing row for `current_idx`.
                insert_ohlcv_data(BOT_ID, [_ohlcv_record(all_data.iloc[current_idx])])

                open_trade = get_open_trade_for_bot(BOT_ID)
                minutes_held = (current_idx - entry_idx) if entry_idx is not None else 0

                if open_trade:
                    update_trade_on_exit(
                        open_trade['id'],
                        str(all_data.iloc[current_idx]['timestamp']),
                        result['fill_price_sol'],
                        SOL_PRICE_USD,
                        result['pnl_sol'], result['pnl_pct'],
                        minutes_held, "Force sell immediate",
                        result['fee_sol']
                    )

                deduct_trading_fee(result['fee_sol'])
                log_message(f"FORCE SOLD at {format_price_si(result['fill_price_sol'])} SOL, P&L: {result['pnl_pct']*100:+.2f}%", BOT_ID, 'WARNING')

                # capital_sol may be > 0 if liquidity-aware buy sizing left unused funds
                new_capital_sol = ((capital_sol or 0) + result['proceeds_sol']
                                   - (result.get('capital_fee_sol', 0) or 0))

                update_bot_trading_state(BOT_ID,
                    position='closed', shares_held=0, entry_price_sol=None,
                    entry_timestamp=None, current_capital_sol=new_capital_sol,
                    trades_count=trades_count + 1,
                    original_entry_price=None, original_entry_timestamp=None,
                    trade_phase=PHASE_NORMAL, below_floor_count=0, max_price_seen=None,
                    n_reentries=0,
                    last_sell_close=current_close,
                    last_sell_idx=current_idx,
                    latest_activation_pct=None
                )
                update_bot_runtime(BOT_ID, status='paused')
                clear_force_sell_files(BOT_ID)

                # Write capital snapshot so wallet snapshot reads correct data
                force_fees = get_bot_cumulative_fees_sol(BOT_ID)
                force_ts = str(all_data.iloc[current_idx]['timestamp'])
                insert_capital_snapshot(BOT_ID, force_ts, new_capital_sol, 0, force_fees)

                write_wallet_snapshot()
                continue

            # 2.5. If paused, skip data collection — only update capital snapshots
            if is_paused:
                # Write a fresh capital snapshot every loop iteration so the
                # per-bot Capital Composition chart keeps advancing while
                # paused — matches the wallet snapshot's behavior.
                #
                # Subtlety: the previous version reused last_ohlcv[0]['timestamp']
                # here, which is frozen while paused (no new candles are being
                # fetched). insert_capital_snapshot uses INSERT OR REPLACE on
                # (bot_id, timestamp), so reusing the same timestamp just
                # overwrote one row and the chart's right edge stayed pinned
                # to the moment the bot was paused. Using the current
                # wall-clock minute gives a new row every iteration.
                last_ohlcv = get_latest_ohlcv(BOT_ID, limit=1)
                if last_ohlcv:
                    last_price = last_ohlcv[0]['close']
                    cumulative_fees_sol = get_bot_cumulative_fees_sol(BOT_ID)
                    timestamp_now = datetime.now().strftime('%Y-%m-%d %H:%M:00')

                    if position == 'open' and shares_held > 0:
                        open_position_sol = shares_held * last_price
                        closed_capital = capital_sol or 0
                    else:
                        open_position_sol = 0
                        closed_capital = capital_sol

                    insert_capital_snapshot(BOT_ID, timestamp_now, closed_capital,
                                           open_position_sol, cumulative_fees_sol)

                # Keep the portfolio chart populated even when all bots are paused.
                write_wallet_snapshot()

                # Sleep while paused — but in <=1s chunks so a resume/stop
                # is picked up within ~1s instead of up to 30s later. The
                # capital snapshot above is still written once per 30s cycle;
                # only the status polling is added. Any status change breaks
                # out and the top of the loop handles it (running -> resume,
                # stopped/missing -> clean exit).
                pause_target = time.monotonic() + 30
                while True:
                    remaining = pause_target - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(1.0, remaining))
                    pause_state = get_bot(BOT_ID)
                    if not pause_state or pause_state['status'] != 'paused':
                        break
                continue

            # 3. Wait for next candle
            now = datetime.now(tz=pd.Timestamp.now(tz=TIMEZONE).tz)
            seconds_into_minute = now.second
            grace_period = GRACE_PERIOD_SECONDS
            if seconds_into_minute < grace_period:
                wait_seconds = grace_period - seconds_into_minute
            else:
                wait_seconds = 60 - seconds_into_minute + grace_period

            # Sleep in <=1s chunks instead of one long block so a pause/stop
            # issued from the dashboard is noticed within ~1s rather than only
            # at the next iteration boundary. Total wait is unchanged (measured
            # with the monotonic clock). If the status leaves 'running'
            # mid-wait, abort and let the top of the loop re-evaluate (paused ->
            # paused branch, stopped/missing -> clean exit).
            wait_target = time.monotonic() + wait_seconds
            wait_interrupted = False
            while True:
                remaining = wait_target - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(1.0, remaining))
                wait_state = get_bot(BOT_ID)
                # Pick up a debug toggle within ~1s (not only at the top of the
                # loop) so the enabled/disabled line appears promptly.
                if wait_state is not None:
                    _wd = bool(wait_state.get('debug_mode'))
                    if _prev_debug_mode is not None and _wd != _prev_debug_mode:
                        log_message(
                            f"Advanced debugging {'enabled' if _wd else 'disabled'}",
                            BOT_ID, 'INFO')
                        _prev_debug_mode = _wd
                if not wait_state or wait_state['status'] != 'running':
                    wait_interrupted = True
                    break
            if wait_interrupted:
                continue

            # 4. Fetch new data. Fetch from ~2 min before the cursor so the most
            #    recently stored minute returns too — we re-check it for a late
            #    revision (step 4a) before accepting genuinely new minutes.
            prev_last_ts = all_data['timestamp'].max()
            # --- timing probe: measure the GeckoTerminal fetch (api=) and the
            #     full request->log span (total=). A late 'Candles:' line with a
            #     high api= is API latency; high (total - api) is local
            #     processing. Uses the monotonic clock so it can't go negative.
            _t_fetch_start = time.monotonic() if debug_mode else None
            new_df = fetch_ohlcv_from_timestamp(
                POOL_ADDRESS, start_timestamp_unix - 120, TIMEZONE, BOT_ID,
                debug=debug_mode)
            _api_secs = (time.monotonic() - _t_fetch_start) if debug_mode else None
            if new_df is None or len(new_df) == 0:
                # Fetch returned nothing. Without this line the minute is
                # skipped silently (no 'Candles:' line) and the backlog is
                # caught up on a later fetch — the invisible failure mode. A
                # A high api here ~= a timeout/slow API; a low api ~= the API
                # simply had no data. api= is shown only under debug.
                _api_str = f" (api={_api_secs:.2f}s)" if debug_mode else ""
                log_message(
                    f"Fetch returned no data{_api_str} — skipping this minute; "
                    f"will catch up on the next successful fetch",
                    BOT_ID, 'WARNING')
                continue

            cutoff_time = datetime.now(tz=pd.Timestamp.now(tz=TIMEZONE).tz) - pd.Timedelta(minutes=FINALIZATION_LAG_MINUTES)
            cutoff_time = cutoff_time.replace(second=0, microsecond=0)
            new_df = new_df[new_df['timestamp'] <= cutoff_time]

            # 4a. Re-check the previous minute for a late revision. The +5s read
            #     can be provisional; by now (~60s later) the candle is settled.
            #     If it changed, correct it and repair the affected features so a
            #     stale value cannot leak into later rolling windows. Most minutes
            #     are unchanged, so this is a cheap no-op the vast majority of
            #     the time.
            recheck = new_df[new_df['timestamp'] == prev_last_ts]
            if len(recheck) == 1:
                refetched_row = recheck.iloc[0]
                # Pre-revision OHLCV, captured only when debugging so we can
                # report how far the late revision moved each field.
                _pre = (all_data.iloc[-1][['open', 'high', 'low', 'close', 'volume']].copy()
                        if debug_mode else None)
                all_data, corrected = _correct_last_minute_if_changed(
                    all_data, refetched_row, engine)
                if corrected:
                    insert_ohlcv_data(BOT_ID, [_ohlcv_record(all_data.iloc[-1])])
                    _ts_hm = (prev_last_ts.strftime('%H:%M')
                              if hasattr(prev_last_ts, 'strftime') else str(prev_last_ts))
                    _msg = f"Corrected previous minute @ {_ts_hm}"
                    if debug_mode and _pre is not None:
                        # All five fields (short labels), incl. unchanged (+0.00%).
                        _labels = {'open': 'o', 'high': 'h', 'low': 'l',
                                   'close': 'c', 'volume': 'v'}
                        _parts = []
                        for _c in ['open', 'high', 'low', 'close', 'volume']:
                            _old = float(_pre[_c]); _new = float(refetched_row[_c])
                            if abs(_old) > 1e-12:
                                _parts.append(f"{_labels[_c]} {(_new - _old) / _old * 100:+.2f}%")
                            elif _new == _old:            # 0 -> 0
                                _parts.append(f"{_labels[_c]} +0.00%")
                            else:                          # 0 -> nonzero (real change)
                                _parts.append(f"{_labels[_c]} 0->{_new:g}")
                        _msg += " | " + " ".join(_parts)
                    log_message(_msg, BOT_ID, 'INFO')

            # 4b. Keep only genuinely new minutes for the normal append path.
            new_df = new_df[new_df['timestamp'] > prev_last_ts]

            if len(new_df) == 0:
                # Fetch succeeded, but nothing newer than the last stored minute
                # cleared the finalization-lag cutoff, so there's no candle to
                # append and no 'Candles:' line this iteration. Benign timing
                # (distinct from the fetch failure above), logged so the skip
                # is still visible. Any late-revision correction above was still
                # applied. api= is shown only under debug.
                _api_str = f" (api={_api_secs:.2f}s)" if debug_mode else ""
                log_message(
                    f"No new finalized candle past cutoff{_api_str} — "
                    f"nothing to append this minute",
                    BOT_ID, 'INFO')
                continue

            # Fill the new minutes exactly as the data notebooks (continuous grid,
            # zero-volume minutes flat at the last real close) and append them.
            new_df = backfill_missing_candles(
                new_df, prev_last_ts=prev_last_ts,
                prev_real_close=_last_real_close(all_data, len(all_data)))
            new_df = new_df.copy()
            new_df['bot_action'] = None

            new_start_idx = len(all_data)
            num_new_rows = len(new_df)

            all_data = pd.concat([all_data, new_df], ignore_index=True)
            engine.sync(all_data)
            all_data = _store_liquidity(all_data, engine, new_start_idx)

            # Re-resolve entry_idx after concat
            if entry_timestamp:
                entry_idx = get_entry_idx(all_data, entry_timestamp)

            # Re-check status right before acting on anything. The status read
            # at the top of the loop is now stale by up to a full candle wait
            # (~60s) plus the fetch time (which can include rate-limit backoff).
            # A pause / stop-all / deallocate issued during that window must NOT
            # produce one more trade. We still record OHLCV + predictions below
            # so charts stay consistent; we only suppress trade actions.
            fresh_state = get_bot(BOT_ID)
            allow_trading = bool(fresh_state and fresh_state['status'] == 'running')
            if not allow_trading:
                log_message(
                    f"Status changed to "
                    f"'{fresh_state['status'] if fresh_state else 'missing'}' during "
                    f"the candle wait/fetch — recording data but skipping trade "
                    f"actions this minute.",
                    BOT_ID, 'INFO'
                )

            # 5. Process each new candle
            #
            # Only the most recent finalized minute is "live". When a batch of
            # backlogged candles arrives (slow historical fetch, API stall, or
            # rate-limit backoff), the earlier candles are ingested for feature
            # continuity and recorded, but trade decisions are made ONLY on the
            # current minute. Acting on stale candles would simulate buys/sells
            # several minutes in the past. In healthy operation num_new_rows == 1
            # so last_new_idx == new_start_idx and this gate is a no-op.
            last_new_idx = new_start_idx + num_new_rows - 1
            combo = 0.0

            # V9 features for the new minutes (rows before START_IDX have none, as in the
            # dataset) and the two models' scores, in one batch.
            score_rows = [r for r in range(new_start_idx, new_start_idx + num_new_rows) if r >= START_IDX]
            feature_df = (engine.features(all_data[['timestamp'] + OHLCV_COLS], score_rows)
                          if score_rows else pd.DataFrame())
            scores = scorer.score(feature_df) if len(feature_df) else None

            for new_idx in range(new_start_idx, new_start_idx + num_new_rows):
                if scores is None or new_idx not in feature_df.index:
                    continue
                k = feature_df.index.get_loc(new_idx)
                features = feature_df.loc[new_idx].to_dict()
                combo = float(scores['combo'][k])
                raw_move = float(scores['raw_move'][k])
                raw_dir = float(scores['raw_dir'][k])

                insert_prediction(BOT_ID, str(all_data.iloc[new_idx]['timestamp']),
                                  combo, raw_move, raw_dir, all_data.iloc[new_idx]['close'],
                                  price_position_15m=features.get('price_position_15m'),
                                  price_position_60m=features.get(PRICE_POSITION_60M_FEATURE))

                current_close = all_data.iloc[new_idx]['close']

                # ---- Trade-action gate ----------------------------------
                # Skip all buy/sell/state-machine logic when either:
                #   (Bug 1) this is not the current minute — a stale backlog
                #           candle that must not be traded on; or
                #   (Bug 2) the bot is no longer 'running' (paused/stopped/
                #           deallocated during the candle wait or fetch).
                # The prediction above is still recorded; only trading is
                # suppressed.
                if new_idx != last_new_idx or not allow_trading:
                    continue

                # ============================================================
                # CLOSED — look for an entry candidate
                # ============================================================
                if position == 'closed':
                    if (passes_entry_filters(combo, features)
                            and passes_reentry_threshold(
                                current_close, new_idx, last_sell_close, last_sell_idx)):

                        # --- Liquidity-aware buy sizing ---
                        liq_sol_per_1pct = all_data.iloc[new_idx].get('estimated_sol_1pct_impact')
                        try:
                            liq_sol_per_1pct = float(liq_sol_per_1pct)
                        except (TypeError, ValueError):
                            liq_sol_per_1pct = float('nan')

                        if not np.isnan(liq_sol_per_1pct) and liq_sol_per_1pct > 0:
                            max_sol_for_impact = liq_sol_per_1pct * (DESIRED_PRICE_IMPACT_PCT * 100.0)
                            buy_amount_sol = min(capital_sol, max_sol_for_impact)
                        else:
                            buy_amount_sol = capital_sol

                        if buy_amount_sol <= 0:
                            log_message(
                                f"Skipping buy at {format_price_si(current_close)}: "
                                f"capped buy amount is {buy_amount_sol:.6f} SOL",
                                BOT_ID, 'WARNING'
                            )
                            continue

                        buy_result = execute_buy(
                            buy_amount_sol, current_close,
                            SLIPPAGE_PCT, TRADING_MODE, SWAP_CTX
                        )
                        if buy_result.get('used_fallback'):
                            log_message("Shadow buy used static fallback (no live quote)",
                                        BOT_ID, 'WARNING')

                        # Network fee (shadow/live) is real SOL burned — deduct
                        # it from capital on top of the amount spent on the swap.
                        # Paper leaves capital_fee_sol unset (fee is implicit in
                        # the fill price), so this is a no-op there.
                        buy_fee = buy_result.get('capital_fee_sol', 0) or 0
                        unused_capital = capital_sol - buy_amount_sol - buy_fee
                        if unused_capital < 0:
                            unused_capital = 0.0
                        position = 'open'
                        shares_held = buy_result['shares_received']
                        entry_price_sol = buy_result['fill_price_sol']
                        original_entry_price = entry_price_sol
                        original_entry_timestamp = str(all_data.iloc[new_idx]['timestamp'])
                        max_price_seen = current_close
                        below_floor_count = 0
                        trade_phase = PHASE_NORMAL
                        n_reentries = 0
                        entry_idx = new_idx
                        entry_timestamp = original_entry_timestamp
                        all_data.at[new_idx, 'bot_action'] = 'BUY'

                        update_bot_trading_state(BOT_ID,
                            position='open', shares_held=shares_held,
                            entry_price_sol=entry_price_sol,
                            entry_timestamp=entry_timestamp,
                            current_capital_sol=unused_capital,
                            original_entry_price=original_entry_price,
                            original_entry_timestamp=original_entry_timestamp,
                            trade_phase=PHASE_NORMAL, below_floor_count=0,
                            max_price_seen=max_price_seen,
                            n_reentries=0
                        )

                        insert_open_trade(
                            BOT_ID, entry_timestamp, entry_price_sol,
                            SOL_PRICE_USD, buy_amount_sol, buy_result['fee_sol']
                        )
                        deduct_trading_fee(buy_result['fee_sol'])
                        capital_sol = unused_capital

                        log_message(
                            f"BUY at {format_price_si(entry_price_sol)} SOL, "
                            f"Spent: {buy_amount_sol:.4f} SOL "
                            f"(capital: {capital_sol + buy_amount_sol:.4f}, "
                            f"liq cap: {('%.4f' % (liq_sol_per_1pct * DESIRED_PRICE_IMPACT_PCT * 100.0)) if not np.isnan(liq_sol_per_1pct) and liq_sol_per_1pct > 0 else 'n/a'}), "
                            f"Combo: {combo:.4f} (move {raw_move:.4f} x dir {raw_dir:.4f})",
                            BOT_ID, 'INFO'
                        )

                # ============================================================
                # OPEN — drive the state machine
                # ============================================================
                elif position == 'open' and entry_idx is not None:
                    minutes_held = new_idx - entry_idx

                    # ------------------------------------------------
                    # Path A: in a race phase, check re-entry candidate
                    # ------------------------------------------------
                    reentry_happened = False
                    if trade_phase in (PHASE_STALE_RACE, PHASE_LOSS_FLOOR_RACE):
                        if (passes_entry_filters(combo, features)
                                and passes_reentry_threshold(
                                    current_close, new_idx, last_sell_close, last_sell_idx)):
                            # Re-enter: refresh segment, keep holding shares
                            entry_price_sol = current_close  # no slippage — already holding
                            entry_idx = new_idx
                            entry_timestamp = str(all_data.iloc[new_idx]['timestamp'])
                            max_price_seen = current_close
                            below_floor_count = 0
                            trade_phase = PHASE_NORMAL
                            n_reentries += 1
                            reentry_happened = True

                            update_bot_trading_state(BOT_ID,
                                entry_price_sol=entry_price_sol,
                                entry_timestamp=entry_timestamp,
                                trade_phase=PHASE_NORMAL, below_floor_count=0,
                                max_price_seen=max_price_seen,
                                n_reentries=n_reentries
                            )

                            log_message(
                                f"RE-ENTRY #{n_reentries} at {format_price_si(current_close)} SOL, "
                                f"Combo: {combo:.4f}",
                                BOT_ID, 'INFO'
                            )

                    # ------------------------------------------------
                    # Path B: state-machine step (sell / phase transition / hold)
                    # ------------------------------------------------
                    if not reentry_happened:
                        step = step_open_position(
                            current_close, entry_price_sol, minutes_held,
                            max_price_seen, below_floor_count, trade_phase
                        )
                        max_price_seen = step['max_price_seen']
                        below_floor_count = step['below_floor_count']

                        # Persist running state every minute (telemetry)
                        update_bot_trading_state(BOT_ID,
                            max_price_seen=max_price_seen,
                            below_floor_count=below_floor_count,
                            latest_activation_pct=step.get('effective_activation_pct')
                        )

                        action = step['action']

                        # ---- SELL (activation) ----
                        if action == 'sell':
                            reason = step['reason']
                            if force_on_signal:
                                reason = "Force sell on signal"

                            # P&L measured against the original (first-leg) entry
                            use_entry = original_entry_price or entry_price_sol
                            exit_price = compute_exit_price(current_close, SLIPPAGE_PCT)

                            sell_result = execute_sell(
                                shares_held, exit_price, use_entry,
                                INITIAL_CAPITAL_SOL, SLIPPAGE_PCT, TRADING_MODE, SWAP_CTX
                            )
                            if sell_result.get('used_fallback'):
                                log_message("Shadow sell used static fallback (no live quote)",
                                            BOT_ID, 'WARNING')

                            capital_sol = (capital_sol + sell_result['proceeds_sol']
                                           - (sell_result.get('capital_fee_sol', 0) or 0))
                            all_data.at[new_idx, 'bot_action'] = 'SELL'

                            open_trade = get_open_trade_for_bot(BOT_ID)
                            total_minutes = new_idx - get_entry_idx(all_data, original_entry_timestamp) \
                                if original_entry_timestamp else minutes_held
                            if open_trade:
                                update_trade_on_exit(
                                    open_trade['id'],
                                    str(all_data.iloc[new_idx]['timestamp']),
                                    sell_result['fill_price_sol'],
                                    SOL_PRICE_USD,
                                    sell_result['pnl_sol'], sell_result['pnl_pct'],
                                    total_minutes, reason, sell_result['fee_sol']
                                )

                            deduct_trading_fee(sell_result['fee_sol'])
                            log_message(
                                f"SELL {COIN} at {format_price_si(sell_result['fill_price_sol'])} SOL, "
                                f"P&L: {sell_result['pnl_pct']*100:+.2f}% ({sell_result['pnl_sol']:+.4f} SOL), "
                                f"Fee: {sell_result['fee_sol']:.6f} SOL, "
                                f"Re-entries: {n_reentries}, Reason: {reason}",
                                BOT_ID, 'INFO'
                            )

                            position = 'closed'
                            shares_held = 0
                            trades_count += 1
                            entry_timestamp = None
                            should_stop = force_on_signal

                            sell_state = dict(
                                position='closed', shares_held=0, entry_price_sol=None,
                                entry_timestamp=None, current_capital_sol=capital_sol,
                                trades_count=trades_count,
                                original_entry_price=None, original_entry_timestamp=None,
                                trade_phase=PHASE_NORMAL, below_floor_count=0,
                                max_price_seen=None,
                                n_reentries=0,
                                last_sell_close=current_close,
                                last_sell_idx=new_idx,
                                latest_activation_pct=None
                            )

                            if should_stop:
                                update_bot_trading_state(BOT_ID, **sell_state)
                                update_bot_runtime(BOT_ID, status='paused')
                                clear_force_sell_files(BOT_ID)
                                is_paused = True
                                log_message(f"Bot {BOT_ID} paused after force sell on signal", BOT_ID, 'INFO')
                            else:
                                update_bot_trading_state(BOT_ID, **sell_state)

                            # Update local state for re-entry filter
                            last_sell_close = current_close
                            last_sell_idx = new_idx
                            # reset segment-level state
                            max_price_seen = None
                            below_floor_count = 0
                            trade_phase = PHASE_NORMAL

                        # ---- Enter STALE RACE ----
                        elif action == 'enter_stale_race':
                            trade_phase = PHASE_STALE_RACE
                            update_bot_trading_state(BOT_ID,
                                trade_phase=PHASE_STALE_RACE,
                                below_floor_count=0
                            )
                            log_message(
                                f"STALE_RACE entered after {minutes_held} min: "
                                f"scanning for re-entry, activation still armed.",
                                BOT_ID, 'INFO'
                            )

                        # ---- Enter LOSS-FLOOR RACE ----
                        elif action == 'enter_loss_floor_race':
                            trade_phase = PHASE_LOSS_FLOOR_RACE
                            update_bot_trading_state(BOT_ID,
                                trade_phase=PHASE_LOSS_FLOOR_RACE,
                                below_floor_count=below_floor_count
                            )
                            log_message(
                                f"LOSS_FLOOR_RACE entered: {step['reason']}",
                                BOT_ID, 'WARNING'
                            )

                        # ---- Recover from LF race back to NORMAL ----
                        elif action == 'recover_loss_floor':
                            trade_phase = PHASE_NORMAL
                            below_floor_count = 0
                            update_bot_trading_state(BOT_ID,
                                trade_phase=PHASE_NORMAL,
                                below_floor_count=0
                            )
                            log_message(
                                f"LOSS_FLOOR recovered: resuming NORMAL phase.",
                                BOT_ID, 'INFO'
                            )

                        # action == 'hold' is a no-op beyond the state persist above

            # 6. Save OHLCV and capital snapshots
            new_records = []
            for idx in range(new_start_idx, new_start_idx + num_new_rows):
                row = all_data.iloc[idx]
                new_records.append(_ohlcv_record(row))
            insert_ohlcv_data(BOT_ID, new_records)

            start_timestamp_unix = int(all_data['timestamp'].max().timestamp())

            cumulative_fees_sol = get_bot_cumulative_fees_sol(BOT_ID)
            for idx in range(new_start_idx, new_start_idx + num_new_rows):
                timestamp_str = str(all_data.iloc[idx]['timestamp'])
                current_token_price = all_data.iloc[idx]['close']

                if position == 'open' and shares_held > 0:
                    open_position_sol = shares_held * current_token_price
                    closed_capital = capital_sol or 0
                else:
                    open_position_sol = 0
                    closed_capital = capital_sol

                insert_capital_snapshot(BOT_ID, timestamp_str, closed_capital,
                                       open_position_sol, cumulative_fees_sol)

            # 7. Update bot state (top-level summary; segment-level state was
            # persisted minute-by-minute above)
            update_bot_trading_state(BOT_ID,
                current_capital_sol=capital_sol, position=position,
                shares_held=shares_held,
                entry_price_sol=entry_price_sol if position == 'open' else None,
                trades_count=trades_count
            )

            # 8. Wallet capital snapshot — write every iteration so the portfolio
            # chart (driven by wallet_capital_history) has the same per-minute
            # cadence as the per-bot capital_history table.
            write_wallet_snapshot()

            current_value = capital_sol * SOL_PRICE_USD if position == 'closed' \
                else (shares_held * all_data.iloc[-1]['close'] + (capital_sol or 0)) * SOL_PRICE_USD
            if debug_mode:
                _liq_est = all_data.iloc[-1].get('estimated_sol_1pct_impact')
                if (_liq_est is not None
                        and not (isinstance(_liq_est, float) and np.isnan(_liq_est))
                        and _liq_est > 0):
                    _liq_cap = _liq_est * DESIRED_PRICE_IMPACT_PCT * 100.0
                    _, _liq_age, _liq_std, _liq_conf = engine.liquidity_state_at(len(all_data) - 1)
                    log_message(
                        f"Liquidity: est={_liq_est:.4f} SOL/1% impact | "
                        f"cap={_liq_cap:.4f} SOL @ {DESIRED_PRICE_IMPACT_PCT * 100:.2f}% | "
                        f"age={_liq_age:.0f} min, dispersion={_liq_std:.1f}%, confidence={_liq_conf:.2f}",
                        BOT_ID, 'INFO')
                else:
                    log_message(
                        "Liquidity: est=n/a (fewer than 5 usable candles so far)",
                        BOT_ID, 'INFO')
            _dbg_timing = (
                f" | api={_api_secs:.2f}s total={time.monotonic() - _t_fetch_start:.2f}s"
                if debug_mode else ""
            )
            log_message(
                f"Candles: {len(all_data)} | Capital: {capital_sol:.4f} SOL "
                f"(${current_value:.2f}) | Trades: {trades_count} | "
                f"Combo: {combo:.4f}{_dbg_timing}",
                BOT_ID, 'INFO'
            )

    except KeyboardInterrupt:
        log_message(f"Bot {BOT_ID} interrupted", BOT_ID, 'WARNING')
    except Exception as e:
        log_message(f"Bot {BOT_ID} error: {e}", BOT_ID, 'ERROR')
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}", BOT_ID, 'ERROR')
    finally:
        # Final state update
        update_bot_runtime(BOT_ID, status='stopped', pid=None, stopped_at=datetime.now())

        if position == 'open':
            log_message(
                f'CRITICAL: Bot {BOT_ID} stopped with OPEN position. '
                f'Shares: {shares_held}, Entry: {entry_price_sol} SOL. '
                f'Manual intervention required.',
                BOT_ID, 'ERROR'
            )

        update_bot_trading_state(BOT_ID,
            current_capital_sol=capital_sol, position=position,
            shares_held=shares_held,
            entry_price_sol=entry_price_sol if position == 'open' else None,
            trades_count=trades_count
        )
        log_message(f"Bot {BOT_ID} stopped", BOT_ID, 'INFO')


if __name__ == '__main__':
    main()
