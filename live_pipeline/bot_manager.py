"""
Bot Manager - Manages bot process lifecycle.

Handles starting, stopping, pausing, resuming bots and checking process health.
Only writes to bot_config and bot_runtime tables, never touches bot_trading_state.
"""

import subprocess
import os
import signal
import time
from datetime import datetime
import json
from database import (
    get_bot, get_all_bots_from_db, get_all_trades,
    get_latest_ohlcv, log_message, get_strategy_config,
    update_bot_runtime,
    initialize_bot_in_db, deallocate_from_bot,
    get_wallet_status, get_db_connection,
    insert_wallet_capital_snapshot
)


# ----------------------------------------------------------------------------
# Per-bot process resources (RAM / CPU)
#
# get_bot_status() is polled roughly once a second, but process resource
# figures only need to refresh once a minute. We cache per bot_id and reuse a
# persistent psutil.Process handle so CPU% is measured across the ~60s gap
# (cpu_percent(None) returns the delta since the previous call on that handle).
# RAM is the resident set size in MB.
# ----------------------------------------------------------------------------
_bot_resource_cache = {}   # bot_id -> {'minute', 'pid', 'ram_mb', 'cpu_pct'}
_bot_proc_handles = {}     # bot_id -> (pid, psutil.Process)


def _get_bot_resources(bot_id, pid):
    """
    Return (ram_mb, cpu_pct) for a bot's process, refreshed at most once per
    minute. Returns (None, None) when the bot has no live process.
    """
    if not pid:
        _bot_resource_cache.pop(bot_id, None)
        _bot_proc_handles.pop(bot_id, None)
        return None, None

    now_minute = datetime.now().strftime('%Y-%m-%d %H:%M')
    cached = _bot_resource_cache.get(bot_id)
    if cached and cached['minute'] == now_minute and cached['pid'] == pid:
        return cached['ram_mb'], cached['cpu_pct']

    try:
        import psutil
        handle_pid, proc = _bot_proc_handles.get(bot_id, (None, None))
        if proc is None or handle_pid != pid or not proc.is_running():
            # New/replaced process: prime cpu_percent (first read is always 0.0)
            proc = psutil.Process(pid)
            proc.cpu_percent(None)
            _bot_proc_handles[bot_id] = (pid, proc)
            cpu_pct = 0.0
        else:
            # Delta since the previous minute's call -> average CPU over the gap
            cpu_pct = round(proc.cpu_percent(None), 1)
        ram_mb = round(proc.memory_info().rss / (1024 * 1024), 1)
        _bot_resource_cache[bot_id] = {
            'minute': now_minute, 'pid': pid,
            'ram_mb': ram_mb, 'cpu_pct': cpu_pct,
        }
        return ram_mb, cpu_pct
    except Exception:
        _bot_resource_cache.pop(bot_id, None)
        _bot_proc_handles.pop(bot_id, None)
        return None, None


class BotManager:

    def __init__(self):
        pass

    def get_bot_status(self, bot_id):
        """Get current status of a bot, checking if process is actually alive."""
        bot = get_bot(bot_id)
        if not bot:
            return {'error': 'Bot not found'}

        # Fetch latest prediction data for buy readiness display
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Latest scores: prediction = combo percentile (buy gate),
            # dir_pct / move_pct = the direction / move percentiles,
            # price_position_60m = the feature price_position_1h.
            cursor.execute('''
                SELECT prediction, price, dir_pct, move_pct, price_position_15m, price_position_60m
                FROM predictions
                WHERE bot_id = ?
                ORDER BY timestamp DESC LIMIT 1
            ''', (bot_id,))
            row = cursor.fetchone()
            bot['latest_prediction'] = row['prediction'] if row else None
            bot['latest_close'] = row['price'] if row else None
            bot['latest_dir_pct'] = row['dir_pct'] if row else None
            bot['latest_move_pct'] = row['move_pct'] if row else None
            bot['latest_pp_15m'] = row['price_position_15m'] if row else None
            bot['latest_pp_60m'] = row['price_position_60m'] if row else None

            # Fetch latest capital snapshot for Capital Overview chart
            cursor.execute('''
                SELECT closed_capital_sol, open_position_sol, cumulative_fees_sol
                FROM capital_history
                WHERE bot_id = ?
                ORDER BY timestamp DESC LIMIT 1
            ''', (bot_id,))
            cap_row = cursor.fetchone()
            if cap_row:
                bot['latest_closed_sol'] = cap_row['closed_capital_sol']
                bot['latest_open_sol'] = cap_row['open_position_sol']
            else:
                bot['latest_closed_sol'] = bot.get('current_capital_sol', 0) or 0
                bot['latest_open_sol'] = 0

            # Latest liquidity estimate: SOL needed to move the price by 1%.
            # The Bot Info card expresses the max-price-impact cap in SOL as
            #   cap_sol = estimated_sol_1pct_impact * (max_price_impact_pct * 100)
            # mirroring the sizing cap the bot loop applies at entry. Use the
            # most recent candle that actually carries an estimate.
            cursor.execute('''
                SELECT estimated_sol_1pct_impact
                FROM ohlcv_data
                WHERE bot_id = ? AND estimated_sol_1pct_impact IS NOT NULL
                ORDER BY timestamp DESC LIMIT 1
            ''', (bot_id,))
            liq_row = cursor.fetchone()
            bot['estimated_sol_1pct_impact'] = (
                liq_row['estimated_sol_1pct_impact'] if liq_row else None
            )

            # Completed-trade count + win rate for the small status card.
            # "Completed" = trades with an exit; matches the Bot Info stats.
            cursor.execute('''
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN pnl_sol > 0 THEN 1 ELSE 0 END) AS wins
                FROM trades
                WHERE bot_id = ? AND exit_timestamp IS NOT NULL
            ''', (bot_id,))
            tr_row = cursor.fetchone()
            completed = (tr_row['total'] or 0) if tr_row else 0
            wins = (tr_row['wins'] or 0) if tr_row else 0
            bot['completed_trades'] = completed
            bot['winning_trades'] = wins
            bot['win_rate'] = (wins / completed * 100) if completed > 0 else 0

            # Strategy state for the dashboard (from the persisted state machine)
            # and the effective re-entry threshold (the reference's rule:
            # relaxing after reentry_decay_start_minutes, gone below -500%).
            cursor.execute('''
                SELECT COUNT(*) as cnt FROM ohlcv_data WHERE bot_id = ?
            ''', (bot_id,))
            current_idx = cursor.fetchone()['cnt'] - 1
            bot['current_minute_idx'] = current_idx
            bot['reentry_price'] = None
            bot['reentry_pct'] = None
            bot['signal_exit_from_idx'] = None
            bot['scan_open'] = False
            bot['reentry_block_active'] = False
            cfg = get_strategy_config()
            try:
                from strategy import SignalExitStrategy, StrategyState
                engine = SignalExitStrategy(cfg) if cfg else None
                st = StrategyState.from_json(bot.get('strategy_state'))
            except Exception:
                engine, st = None, None
            if engine is not None and st is not None:
                bot['signal_exit_enabled'] = engine.signal_on
                bot['signal_exit_from_idx'] = engine.signal_exit_armed_at(st)
                bot['scan_open'] = bool(st.scan_open)
                bot['reentry_block_active'] = bool(st.block_active)
                bot['take_profit_price'] = engine.take_profit_price(st)
                bot['loss_floor_price'] = engine.loss_floor_price(st)
                bot['orig_entry_idx'] = st.orig_entry_idx
                if bot.get('last_sell_close') and bot['last_sell_close'] > 0 \
                        and bot.get('last_sell_idx') is not None:
                    eff = engine.reentry_effective_pct(current_idx, bot['last_sell_idx'])
                    if eff is not None:
                        bot['reentry_pct'] = eff
                        bot['reentry_price'] = bot['last_sell_close'] * (1.0 - eff)
        bot.pop('strategy_state', None)

        # Surface the file-based force-sell flags so the dashboard can show
        # whether the operator has armed a force-sell. These are written by
        # the /api/force_sell_* endpoints and cleared by bot_loop once acted
        # on, so their presence == "armed and pending".
        bot_dir = f'data/bot_{bot_id}'
        bot['force_sell_immediate_pending'] = os.path.exists(f'{bot_dir}/force_sell_immediate')
        bot['force_sell_on_signal_pending'] = os.path.exists(f'{bot_dir}/force_sell_on_signal')

        # Check if process is actually running
        if bot['pid']:
            try:
                os.kill(bot['pid'], 0)
                if bot['status'] not in ('running', 'paused'):
                    update_bot_runtime(bot_id, status='running')
                    bot['status'] = 'running'
            except OSError:
                # Process doesn't exist but DB thinks it's running
                if bot['status'] == 'running':
                    if bot['position'] == 'open':
                        log_message(
                            f'CRITICAL: Bot {bot_id} crashed with open position. '
                            f'Shares held: {bot["shares_held"]}, Entry price: {bot["entry_price_sol"]}, '
                            f'Original entry: {bot.get("original_entry_price")}, '
                            f'Re-entries: {bot.get("n_reentries", 0)}, '
                            f'Phase: {bot.get("trade_phase", "normal")}. '
                            f'Manual intervention may be needed.',
                            bot_id, 'ERROR'
                        )
                    update_bot_runtime(bot_id, status='stopped', pid=None)
                    bot['status'] = 'stopped'
                    bot['pid'] = None

        # ---- Assigned-to-coin duration ----
        # Wall-clock age since the coin was assigned to this slot. Persists
        # across pause / resume / restart / crash and only resets on
        # deallocation (coin_assigned_at is cleared there). Independent of
        # process state, so it keeps counting while a bot is paused.
        bot['assigned_seconds'] = None
        bot['assigned_duration'] = None
        if bot.get('coin_assigned_at'):
            try:
                assigned = datetime.fromisoformat(bot['coin_assigned_at'])
                delta = datetime.now() - assigned
                bot['assigned_seconds'] = delta.total_seconds()
                a_hours = delta.seconds // 3600
                a_mins = (delta.seconds % 3600) // 60
                bot['assigned_duration'] = f"{delta.days}d {a_hours}h {a_mins}m"
            except (ValueError, TypeError):
                pass

        # ---- Per-bot process resources (RAM / CPU), refreshed once a minute ----
        bot['ram_mb'], bot['cpu_pct'] = _get_bot_resources(bot_id, bot.get('pid'))

        return bot

    def check_all_bot_processes(self):
        """Batch check if all bot processes are alive."""
        import psutil

        bots = get_all_bots_from_db()
        running_bots = [b for b in bots if b['status'] == 'running' and b['pid']]

        if not running_bots:
            return

        try:
            system_pids = {p.pid for p in psutil.process_iter(['pid'])}
        except:
            return

        for bot in running_bots:
            if bot['pid'] not in system_pids:
                if bot['position'] == 'open':
                    log_message(
                        f'CRITICAL: Bot {bot["id"]} crashed with open position. '
                        f'Shares held: {bot["shares_held"]}, Entry price: {bot["entry_price_sol"]}, '
                        f'Original entry: {bot.get("original_entry_price")}, '
                        f'Re-entries: {bot.get("n_reentries", 0)}, '
                        f'Phase: {bot.get("trade_phase", "normal")}. '
                        f'Manual intervention may be needed.',
                        bot['id'], 'ERROR'
                    )
                update_bot_runtime(bot['id'], status='stopped', pid=None)

    def get_all_bots_status(self):
        """Get status of all 10 bots."""
        self.check_all_bot_processes()
        bots = []
        for bot_id in range(1, 11):
            bot = self.get_bot_status(bot_id)
            bots.append(bot)
        return bots

    def initialize_bot(self, bot_id, coin_symbol, pool_address,
                       allocation_type, allocation_value,
                       image_thumb_base64=None, image_small_base64=None,
                       confidence_threshold=None, max_price_impact_pct=0.00333):
        """
        Initialize a bot with trading parameters and wallet allocation.

        Note: confidence_threshold is stored in bot_config but unused; the
        strategy's buy gate comes from strategy_config (best_params.json).
        """
        try:
            wallet = get_wallet_status()
            if not wallet['initialized']:
                return {'success': False, 'error': 'Wallet not initialized. Please initialize wallet first.'}

            # Calculate SOL to allocate
            if allocation_type == 'percentage':
                if allocation_value <= 0 or allocation_value > 100:
                    return {'success': False, 'error': 'Percentage must be between 0 and 100'}
                sol_amount = (allocation_value / 100.0) * wallet['wallet_sol']
            elif allocation_type == 'sol_amount':
                if allocation_value <= 0:
                    return {'success': False, 'error': 'SOL amount must be positive'}
                sol_amount = allocation_value
            else:
                return {'success': False, 'error': 'Invalid allocation type'}

            if sol_amount > wallet['unallocated_wallet_sol']:
                return {'success': False, 'error': f'Insufficient funds. Available: {wallet["unallocated_wallet_sol"]:.4f} SOL, Requested: {sol_amount:.4f} SOL'}

            # Funds already validated above against the same wallet snapshot;
            # record the allocation in the log (previously done inside the
            # now-removed allocate_from_wallet helper). The actual capital
            # assignment happens in initialize_bot_in_db below.
            log_message(f'Allocated {sol_amount:.4f} SOL to Bot {bot_id}', bot_id, 'INFO')

            # Initialize bot - pass base64 strings directly, no blob conversion
            initialize_bot_in_db(
                bot_id, pool_address, coin_symbol, sol_amount,
                image_thumb_base64, image_small_base64,
                confidence_threshold, max_price_impact_pct
            )

            os.makedirs(f'data/bot_{bot_id}', exist_ok=True)
            log_message(f'Bot {bot_id} initialized: {coin_symbol} with {sol_amount:.4f} SOL', bot_id, 'INFO')

            # Record updated wallet snapshot reading from capital_history
            updated_wallet = get_wallet_status()
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

            insert_wallet_capital_snapshot(
                datetime.now().strftime('%Y-%m-%d %H:%M:00'),
                updated_wallet['unallocated_wallet_sol'],
                total_initial, total_open, total_profit,
                updated_wallet.get('total_fees_sol', 0)
            )

            return {'success': True, 'bot_id': bot_id}

        except Exception as e:
            log_message(f'Error initializing bot {bot_id}: {str(e)}', bot_id, 'ERROR')
            return {'success': False, 'error': str(e)}

    def start_bot(self, bot_id):
        """Start a bot trading process."""
        try:
            bot = get_bot(bot_id)
            if not bot:
                return {'success': False, 'error': 'Bot not found'}
            if bot['status'] == 'inactive':
                return {'success': False, 'error': 'Bot not initialized. Configure it first.'}
            if bot['status'] == 'running':
                return {'success': False, 'error': 'Bot already running'}

            # Kill any existing process first
            self._kill_bot_process(bot_id)

            # Launch bot_loop.py
            process = subprocess.Popen(
                ['python3', 'bot_loop.py', '--bot-id', str(bot_id)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            update_bot_runtime(
                bot_id,
                status='running',
                pid=process.pid,
                started_at=datetime.now(),
                initializing=1  # set with status so there's no running-but-not-initialising window
            )

            # Clear any leftover force sell files
            self._clear_force_sell_files(bot_id)

            log_message(f'Bot {bot_id} started (PID: {process.pid})', bot_id, 'INFO')
            return {'success': True, 'pid': process.pid}

        except Exception as e:
            log_message(f'Error starting bot {bot_id}: {str(e)}', bot_id, 'ERROR')
            return {'success': False, 'error': str(e)}

    def _kill_bot_process(self, bot_id):
        """Kill any existing process for this bot."""
        bot = get_bot(bot_id)

        if bot and bot['pid']:
            try:
                os.kill(bot['pid'], signal.SIGTERM)
                time.sleep(1)
                try:
                    os.kill(bot['pid'], 0)
                    os.kill(bot['pid'], signal.SIGKILL)
                    time.sleep(0.5)
                except OSError:
                    pass
            except OSError:
                pass

        # Check for rogue processes
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and 'bot_loop.py' in ' '.join(cmdline) and f'--bot-id {bot_id}' in ' '.join(cmdline):
                        proc.kill()
                        log_message(f'Killed rogue process PID {proc.pid} for bot {bot_id}', bot_id, 'WARNING')
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except ImportError:
            pass

    def _clear_force_sell_files(self, bot_id):
        """Remove any force sell signal files."""
        bot_dir = f'data/bot_{bot_id}'
        for filename in ['force_sell_immediate', 'force_sell_on_signal']:
            filepath = f'{bot_dir}/{filename}'
            if os.path.exists(filepath):
                os.remove(filepath)

    def pause_bot(self, bot_id):
        """Pause a running bot. Process continues collecting data but won't trade."""
        try:
            bot = get_bot(bot_id)
            if not bot:
                return {'success': False, 'error': 'Bot not found'}
            if bot['status'] != 'running':
                return {'success': False, 'error': 'Bot not running'}

            if bot['position'] == 'open':
                log_message(f'Warning: Pausing bot {bot_id} with open position.', bot_id, 'WARNING')

            # Don't kill process, just set status to paused
            update_bot_runtime(bot_id, status='paused')
            self._clear_force_sell_files(bot_id)

            log_message(f'Bot {bot_id} paused', bot_id, 'INFO')
            return {'success': True}

        except Exception as e:
            log_message(f'Error pausing bot {bot_id}: {str(e)}', bot_id, 'ERROR')
            return {'success': False, 'error': str(e)}

    def resume_bot(self, bot_id):
        """Resume a paused bot by starting a new process."""
        return self.start_bot(bot_id)

    def deallocate_bot(self, bot_id):
        """Deallocate a bot and return its capital to the wallet."""
        try:
            # Kill any running process first
            self._kill_bot_process(bot_id)
            self._clear_force_sell_files(bot_id)

            deallocate_from_bot(bot_id)
            
            # Record updated wallet snapshot reading from capital_history
            updated_wallet = get_wallet_status()
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

            insert_wallet_capital_snapshot(
                datetime.now().strftime('%Y-%m-%d %H:%M:00'),
                updated_wallet['unallocated_wallet_sol'],
                total_initial, total_open, total_profit,
                updated_wallet.get('total_fees_sol', 0)
            )
            
            return {'success': True, 'message': f'Bot {bot_id} deallocated successfully'}

        except Exception as e:
            log_message(f'Error deallocating bot {bot_id}: {str(e)}', bot_id, 'ERROR')
            return {'success': False, 'error': str(e)}

    def stop_all_bots(self):
        """Emergency stop all running bots."""
        results = []
        for bot_id in range(1, 11):
            bot = get_bot(bot_id)
            if bot and bot['status'] == 'running':
                result = self.pause_bot(bot_id)
                results.append({'bot_id': bot_id, 'result': result})

        log_message('All bots stopped', None, 'WARNING')
        return results

    def get_dashboard_data(self):
        """Get all data needed for dashboard."""
        bots = self.get_all_bots_status()
        all_trades = get_all_trades(limit=50)

        active_count = sum(1 for bot in bots if bot['status'] == 'running')
        total_trades = sum(bot['trades_count'] or 0 for bot in bots)
        wallet_status = get_wallet_status()

        return {
            'bots': bots,
            'trades': all_trades,
            'active_count': active_count,
            'total_trades': total_trades,
            'wallet': wallet_status
        }

    def get_bot_chart_data(self, bot_id):
        """Get OHLCV data with predictions for charting."""
        ohlcv = get_latest_ohlcv(bot_id, limit=200)

        if len(ohlcv) == 0:
            return []

        oldest_timestamp = ohlcv[0]['timestamp']
        newest_timestamp = ohlcv[-1]['timestamp']

        # Fetch predictions within the OHLCV time range
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, CAST(prediction AS REAL) as prediction, dir_pct, move_pct,
                       price_position_15m, price_position_60m
                FROM predictions
                WHERE bot_id = ?
                AND timestamp >= ?
                AND timestamp <= ?
                ORDER BY timestamp ASC
            ''', (bot_id, oldest_timestamp, newest_timestamp))
            predictions = [dict(row) for row in cursor.fetchall()]

        pred_by_ts = {pred['timestamp']: pred for pred in predictions}

        chart_data = []
        for row in ohlcv:
            chart_data.append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'action': row['bot_action'],
                # combo percentile (buy gate) and the two exit-side scores
                'confidence': pred_by_ts.get(row['timestamp'], {}).get('prediction'),
                'dir_pct': pred_by_ts.get(row['timestamp'], {}).get('dir_pct'),
                'move_pct': pred_by_ts.get(row['timestamp'], {}).get('move_pct'),
                'pp_15m': pred_by_ts.get(row['timestamp'], {}).get('price_position_15m'),
                'pp_1h': pred_by_ts.get(row['timestamp'], {}).get('price_position_60m'),
            })

        return chart_data


# Singleton instance
bot_manager = BotManager()