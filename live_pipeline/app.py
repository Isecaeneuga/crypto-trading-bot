"""
Flask Dashboard Server

Serves the frontend, provides API endpoints for bot management,
and runs the SOL price collector in a background thread.

All monetary values returned by the API are in SOL.
The frontend handles USD conversion using the current SOL price.
"""

from flask import Flask, render_template, jsonify, request, Response, stream_with_context
import psutil
import os
import re
from datetime import timedelta
import threading
import time
import requests
import json
from database import (
    init_database, get_recent_logs, get_global_settings,
    get_bot_trades, get_all_trades, log_message,
    get_latest_sol_price, insert_sol_price, get_latest_ohlcv,
    get_bot, get_wallet_status, initialize_wallet,
    get_bot_stats_precomputed,
    get_last_modified_timestamps, get_change_markers,
    get_capital_history_page, get_capital_history_after,
    insert_wallet_capital_snapshot, get_wallet_capital_page, get_wallet_capital_after,
    update_bot_config
)

app = Flask(__name__)

# Initialize database before importing bot_manager
init_database()

from bot_manager import bot_manager
from datetime import datetime
from strategy import TUNED_PARAMS, FIXED_PARAMS, STRATEGY, use_strategy, strategy_files, read_strategy
from scoring import verify_models
from features import START_IDX, PUMP_SETS, LIQ_STATS_REFRESH_MIN

# Track last wallet snapshot minute to avoid duplicates
_last_wallet_snapshot_minute = None

# ========================================================================
# SOL PRICE BACKGROUND COLLECTOR
# ========================================================================

BASE_URL = "https://api.geckoterminal.com/api/v2"
NETWORK = "solana"
SOL_PRICE_UPDATE_INTERVAL = 900  # 15 minutes

def collect_sol_price():
    """Background thread that fetches SOL/USD price from Binance every 15 minutes."""
    print("Starting SOL price collector background thread...")

    while True:
        try:
            response = requests.get(
                "https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT",
                timeout=10
            )
            response.raise_for_status()
            sol_price = float(response.json()["price"])

            global_config = get_global_settings()
            timezone = global_config['timezone'] if global_config else 'UTC'

            import pandas as pd
            timestamp = pd.Timestamp.now(tz=timezone)

            insert_sol_price(sol_price, str(timestamp))
            print(f"[SOL Collector] Updated: ${sol_price:.2f} at {timestamp}")
            log_message(f"SOL price updated: ${sol_price:.2f}", None, 'INFO')

        except Exception as e:
            print(f"[SOL Collector] Error: {e}")
            log_message(f"SOL price collector error: {e}", None, 'ERROR')

        time.sleep(SOL_PRICE_UPDATE_INTERVAL)

sol_price_thread = threading.Thread(target=collect_sol_price, daemon=True)
sol_price_thread.start()

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def get_pi_uptime():
    """Get system uptime."""
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
        uptime_td = timedelta(seconds=uptime_seconds)
        days = uptime_td.days
        hours, remainder = divmod(uptime_td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{days}d {hours}h {minutes}m {seconds}s"
    except:
        return "N/A"

def get_system_stats():
    """Get CPU and RAM usage."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        return {
            'cpu_percent': round(cpu_percent, 1),
            'ram_percent': round(memory.percent, 1),
            'ram_used_gb': round(memory.used / (1024**3), 2),
            'ram_total_gb': round(memory.total / (1024**3), 2)
        }
    except:
        return {'cpu_percent': 0, 'ram_percent': 0, 'ram_used_gb': 0, 'ram_total_gb': 0}

# ========================================================================
# ROUTES
# ========================================================================

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/stream')
def api_stream():
    """
    Live update stream (Server-Sent Events). Checks the database's change
    markers once a second and sends a 'markers' event whenever any of them
    changed; the dashboard then refetches only the parts that changed. A
    comment line every 15 s keeps the connection open through proxies.
    """
    def generate():
        last = None
        idle = 0
        yield 'retry: 3000\n\n'
        while True:
            try:
                markers = get_change_markers()
            except Exception as e:
                markers = {'error': str(e)}
            if markers != last:
                last = markers
                idle = 0
                yield f"event: markers\ndata: {json.dumps(markers, default=str)}\n\n"
            else:
                idle += 1
                if idle >= 15:
                    idle = 0
                    yield ': keep-alive\n\n'
            time.sleep(1)

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/system')
def api_system():
    return jsonify({'uptime': get_pi_uptime(), **get_system_stats()})

@app.route('/api/sol-price')
def api_sol_price():
    """Get current SOL price."""
    try:
        sol_price = get_latest_sol_price()
        if sol_price:
            return jsonify(sol_price)
        else:
            return jsonify({'price_usd': 135.0, 'timestamp': None})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet/status')
def api_wallet_status():
    """Get wallet status. All values in SOL, frontend converts to USD."""
    try:
        wallet_data = get_wallet_status()
        return jsonify(wallet_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet/initialize', methods=['POST'])
def api_wallet_initialize():
    """Initialize wallet with a SOL amount and trading mode.

    Body: { sol_amount, trading_mode, static_trading_fee_pct?, grace_period_seconds? }
    Only 'paper' is currently supported end-to-end; 'shadow' and 'live' have
    no backend yet and are rejected here (the UI offers them but blocks init).
    """
    try:
        data = request.json
        sol_amount = data.get('sol_amount')
        trading_mode = (data.get('trading_mode') or 'paper').lower()

        if trading_mode not in ('paper', 'shadow', 'live'):
            return jsonify({'success': False, 'error': 'Invalid trading mode'}), 400

        # Live still has no backend.
        if trading_mode == 'live':
            return jsonify({
                'success': False,
                'error': "Live mode is not available yet"
            }), 400

        if not sol_amount or sol_amount <= 0:
            return jsonify({'success': False, 'error': 'Valid SOL amount required'}), 400

        # Paper-only extras. Fee arrives as a decimal (e.g. 0.015 = 1.5%).
        static_fee = data.get('static_trading_fee_pct')
        grace = data.get('grace_period_seconds')
        finalization_lag = data.get('finalization_lag_minutes')
        wallet_pubkey = None

        if trading_mode == 'paper':
            if static_fee is not None:
                static_fee = float(static_fee)
                if static_fee < 0 or static_fee > 1:
                    return jsonify({'success': False, 'error': 'Trading fee must be between 0 and 1 (decimal)'}), 400
            if grace is not None:
                grace = int(grace)
                if grace < 0 or grace > 59:
                    return jsonify({'success': False, 'error': 'Grace period must be 0–59 seconds'}), 400
            if finalization_lag is not None:
                finalization_lag = int(finalization_lag)
                if finalization_lag < 0 or finalization_lag > 120:
                    return jsonify({'success': False, 'error': 'Finalization lag must be 0–120 minutes'}), 400
        elif trading_mode == 'shadow':
            # Shadow needs a real wallet public key to quote/monitor against.
            wallet_pubkey = (data.get('wallet_pubkey') or '').strip()
            if not re.fullmatch(r'[1-9A-HJ-NP-Za-km-z]{32,44}', wallet_pubkey):
                return jsonify({'success': False, 'error': 'A valid Solana public key is required for shadow mode'}), 400
            # Shadow ignores static fee / grace / finalization lag.
            static_fee = grace = finalization_lag = None

        result = initialize_wallet(sol_amount, trading_mode, static_fee, grace,
                                   finalization_lag, wallet_pubkey)
        return jsonify({'success': True, 'wallet': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/token-price/<int:bot_id>')
def api_token_price(bot_id):
    """Get current token price for a bot."""
    try:
        bot = bot_manager.get_bot_status(bot_id)
        if not bot or bot['status'] == 'inactive':
            return jsonify({'error': 'Bot not found or inactive'}), 404

        ohlcv = get_latest_ohlcv(bot_id, limit=1)
        if ohlcv and len(ohlcv) > 0:
            return jsonify({'price_sol': ohlcv[0]['close']})
        else:
            return jsonify({'error': 'No price data available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def api_status():
    """
    Get status of all bots. All monetary values in SOL.
    Frontend uses current SOL price for USD display.
    """
    try:
        dashboard_data = bot_manager.get_dashboard_data()

        for bot in dashboard_data['bots']:
            if bot['status'] != 'inactive':
                # Live value = idle (closed) capital + current mark-to-market
                # value of any open position. This is the bot's true current
                # worth (open + closed + realized profit), used by the small
                # status cards. current_capital_sol from the DB is only the
                # closed (idle) leg, so on its own it makes an open trade look
                # flat. We expose live_value_sol and open_value_sol as new
                # fields and leave current_capital_sol untouched (it still
                # equals the closed leg, which is what deallocation returns).
                closed_leg = bot.get('latest_closed_sol')
                if closed_leg is None:
                    closed_leg = bot.get('current_capital_sol', 0) or 0

                open_leg = 0
                if bot['position'] == 'open' and bot.get('shares_held', 0) > 0:
                    ohlcv = get_latest_ohlcv(bot['id'], limit=1)
                    if ohlcv and len(ohlcv) > 0:
                        open_leg = bot['shares_held'] * ohlcv[0]['close']
                    else:
                        open_leg = bot.get('latest_open_sol', 0) or 0

                bot['open_value_sol'] = open_leg
                bot['live_value_sol'] = closed_leg + open_leg

                # Pass through base64 images
                bot['coin_image_thumb_base64'] = bot.get('coin_image_thumb_base64')
                bot['coin_image_small_base64'] = bot.get('coin_image_small_base64')

        # Add wallet data
        wallet_data = get_wallet_status()
        dashboard_data['wallet'] = wallet_data

        # Record wallet-level capital snapshot (once per minute)
        global _last_wallet_snapshot_minute
        now = datetime.now()
        current_minute = now.strftime('%Y-%m-%d %H:%M')
        if wallet_data.get('initialized') and current_minute != _last_wallet_snapshot_minute:
            _last_wallet_snapshot_minute = current_minute
            unallocated = wallet_data.get('unallocated_wallet_sol', 0)

            # total_initial and total_profit still derive from capital_history
            # (closed_capital_sol is stable while position is closed, and the
            # freed capital flows into unallocated when a bot is deallocated,
            # so these series stay visually correct).
            #
            # total_open is now computed two ways:
            #   1. Only include bots whose live trading state says position is
            #      actually 'open' with shares_held > 0. This prevents phantom
            #      open positions when capital_history has a stale pre-close
            #      row (e.g. bot crashed between SELL and the snapshot loop).
            #   2. Use bot['current_capital_sol'] — already updated above to
            #      shares_held * latest OHLCV close — instead of the possibly
            #      stale capital_history.open_position_sol. If no fresh OHLCV
            #      is available, current_capital_sol stays at 0 and we
            #      contribute 0, which is the truthful answer.
            total_initial = 0
            total_open = 0
            total_profit = 0
            from database import get_db_connection
            with get_db_connection() as conn:
                cursor = conn.cursor()
                for bot in dashboard_data['bots']:
                    if bot['status'] != 'inactive':
                        cursor.execute('''
                            SELECT closed_capital_sol
                            FROM capital_history WHERE bot_id = ?
                            ORDER BY timestamp DESC LIMIT 1
                        ''', (bot['id'],))
                        row = cursor.fetchone()
                        if row:
                            closed = row['closed_capital_sol'] or 0
                            initial = bot.get('initial_capital_sol', 0) or 0
                            total_initial += min(closed, initial)
                            total_profit += max(0, closed - initial)

                        # Fix 1: position cross-check.
                        # Fix 2: live mark-to-market via open_value_sol (the
                        # fresh shares_held * latest close computed above).
                        if (bot.get('position') == 'open'
                                and bot.get('shares_held', 0) > 0):
                            total_open += bot.get('open_value_sol', 0) or 0

            try:
                insert_wallet_capital_snapshot(
                    now.strftime('%Y-%m-%d %H:%M:00'),
                    unallocated, total_initial, total_open, total_profit,
                    wallet_data.get('total_fees_sol', 0)
                )
            except Exception:
                pass

        # Add last modified timestamps
        dashboard_data['last_modified'] = get_last_modified_timestamps()

        return jsonify(dashboard_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot/<int:bot_id>')
def api_bot_details(bot_id):
    """Get bot status and trades (lightweight, no chart/capital data)."""
    try:
        bot = bot_manager.get_bot_status(bot_id)
        trades = get_bot_trades(bot_id, limit=10)
        return jsonify({
            'bot': bot,
            'trades': trades
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot/<int:bot_id>/chart')
def api_bot_chart(bot_id):
    """Get OHLCV + predictions for the candlestick chart."""
    try:
        bot = bot_manager.get_bot_status(bot_id)
        chart_data = bot_manager.get_bot_chart_data(bot_id)
        return jsonify({
            'bot': bot,
            'chart_data': chart_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot/<int:bot_id>/capital')
def api_bot_capital(bot_id):
    """
    Get capital history with pagination or incremental updates.
    
    Query params:
      ?after=<timestamp>  — incremental: rows after timestamp
      ?offset=N           — paginated: 1000 rows starting from offset (newest first)
      (no params)         — first page (most recent 1000)
    """
    try:
        after = request.args.get('after')
        if after:
            return jsonify(get_capital_history_after(bot_id, after))

        offset = request.args.get('offset', 0, type=int)
        return jsonify(get_capital_history_page(bot_id, limit=1000, offset=offset))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet/capital')
def api_wallet_capital():
    """
    Get portfolio capital history with pagination or incremental updates.
    
    Query params:
      ?after=<timestamp>  — incremental: rows after timestamp
      ?offset=N           — paginated: 1000 rows starting from offset (newest first)
      (no params)         — first page (most recent 1000)
    """
    try:
        after = request.args.get('after')
        if after:
            return jsonify(get_wallet_capital_after(after))

        offset = request.args.get('offset', 0, type=int)
        return jsonify(get_wallet_capital_page(limit=1000, offset=offset))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot/<int:bot_id>/stats')
def api_bot_stats(bot_id):
    """Get pre-computed statistics for a bot. All values in SOL."""
    try:
        bot = get_bot(bot_id)
        if not bot:
            return jsonify({'error': 'Bot not found'}), 404
        if bot['status'] == 'inactive':
            return jsonify({'error': 'Bot not initialized'}), 404

        stats = get_bot_stats_precomputed(bot_id)
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot/initialize', methods=['POST'])
def initialize_bot():
    """Initialize a bot with trading parameters. Images passed as base64 strings."""
    try:
        data = request.json
        bot_id = data.get('botId')
        coin_symbol = data.get('coin_symbol')
        pool = data.get('pool')
        allocation_type = data.get('allocation_type')
        allocation_value = data.get('allocation_value')
        image_thumb_base64 = data.get('image_thumb')
        image_small_base64 = data.get('image_small')
        max_price_impact_pct = data.get('max_price_impact_pct', 0.00333)
        strategy_name = data.get('strategy_name')

        if not all([bot_id, coin_symbol, pool, allocation_type, allocation_value]):
            return jsonify({'success': False, 'error': 'Missing required parameters'}), 400

        # the strategy must be a usable strategy file in the model folder (default: the first one)
        listing = list_strategies()
        usable = {s['name'] for s in listing['strategies'] if s['valid']}
        if not strategy_name:
            strategy_name = listing['default']
        if not strategy_name:
            return jsonify({'success': False, 'error': f"No usable strategy in {listing['model_dir']}/: "
                                                       f"copy strategy.ipynb's live_artifacts/ files there"}), 400
        if strategy_name not in usable:
            bad = next((s for s in listing['strategies'] if s['name'] == strategy_name), None)
            return jsonify({'success': False, 'error': f"Strategy '{strategy_name}' can't be used: "
                                                       f"{bad['error'] if bad else 'not in the model folder'}"}), 400

        result = bot_manager.initialize_bot(
            bot_id, coin_symbol, pool,
            allocation_type, allocation_value,
            image_thumb_base64, image_small_base64,
            max_price_impact_pct, strategy_name
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/pause', methods=['POST'])
def pause_bot():
    try:
        data = request.json
        bot_id = data.get('botId')
        if not bot_id:
            return jsonify({'success': False, 'error': 'Bot ID required'}), 400
        result = bot_manager.pause_bot(bot_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/resume', methods=['POST'])
def resume_bot():
    try:
        data = request.json
        bot_id = data.get('botId')
        if not bot_id:
            return jsonify({'success': False, 'error': 'Bot ID required'}), 400
        result = bot_manager.resume_bot(bot_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/debug', methods=['POST'])
def set_bot_debug():
    """Toggle per-bot advanced-debug logging. Persists debug_mode on bot_config;
    the bot re-reads it each loop iteration, so it takes effect within a minute
    without a restart."""
    try:
        data = request.json
        bot_id = data.get('botId')
        enabled = data.get('enabled')
        if not bot_id:
            return jsonify({'success': False, 'error': 'Bot ID required'}), 400
        flag = 1 if enabled else 0
        update_bot_config(bot_id, debug_mode=flag)
        return jsonify({'success': True, 'debug_mode': flag})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/force-sell-immediate', methods=['POST'])
def force_sell_immediate():
    """Create force sell immediate signal file."""
    try:
        data = request.json
        bot_id = data.get('botId')
        if not bot_id:
            return jsonify({'success': False, 'error': 'Bot ID required'}), 400

        bot_dir = f'data/bot_{bot_id}'
        os.makedirs(bot_dir, exist_ok=True)
        with open(f'{bot_dir}/force_sell_immediate', 'w') as f:
            f.write('')

        log_message(f'Bot {bot_id} force sell immediate triggered', bot_id, 'WARNING')
        return jsonify({'success': True, 'message': 'Force sell signal sent to bot'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/force-sell-on-signal', methods=['POST'])
def force_sell_on_signal():
    """Toggle the force-sell-on-signal flag.

    If the flag is not set, arm it (the bot sells on its next sell signal and
    stops). If it is already set, disarm it (cancel the pending sell-on-signal)
    so pressing the same button again turns it off.
    """
    try:
        data = request.json
        bot_id = data.get('botId')
        if not bot_id:
            return jsonify({'success': False, 'error': 'Bot ID required'}), 400

        bot_dir = f'data/bot_{bot_id}'
        os.makedirs(bot_dir, exist_ok=True)
        flag_path = f'{bot_dir}/force_sell_on_signal'

        if os.path.exists(flag_path):
            os.remove(flag_path)
            log_message(f'Bot {bot_id} force sell on signal disarmed', bot_id, 'WARNING')
            return jsonify({'success': True, 'armed': False, 'message': 'Sell-on-signal disarmed'})

        with open(flag_path, 'w') as f:
            f.write('')
        log_message(f'Bot {bot_id} force sell on signal triggered', bot_id, 'WARNING')
        return jsonify({'success': True, 'armed': True, 'message': 'Bot will sell on next signal and stop'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/deallocate', methods=['POST'])
def deallocate_bot():
    try:
        data = request.json
        bot_id = data.get('botId')
        if not bot_id:
            return jsonify({'success': False, 'error': 'Bot ID required'}), 400
        result = bot_manager.deallocate_bot(bot_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bots/pause-all', methods=['POST'])
def pause_all_bots():
    """Pause every running bot (paused and stopped bots stay as they are)."""
    try:
        results = bot_manager.pause_all_bots()
        failed = [r for r in results if not r['result'].get('success')]
        return jsonify({'success': not failed, 'count': len(results) - len(failed), 'results': results,
                        'error': '; '.join(f"Bot {r['bot_id']}: {r['result'].get('error')}" for r in failed) or None})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bots/resume-all', methods=['POST'])
def resume_all_bots():
    """Start every paused or stopped bot (running bots stay as they are)."""
    try:
        results = bot_manager.resume_all_bots()
        failed = [r for r in results if not r['result'].get('success')]
        return jsonify({'success': not failed, 'count': len(results) - len(failed), 'results': results,
                        'error': '; '.join(f"Bot {r['bot_id']}: {r['result'].get('error')}" for r in failed) or None})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/stop-all', methods=['POST'])
def stop_all_bots():
    try:
        results = bot_manager.stop_all_bots()
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/token/fetch-info', methods=['POST'])
def fetch_token_info():
    """Fetch token info from GeckoTerminal. Returns base64 images directly."""
    try:
        data = request.json
        token_address = data.get('token_address')
        if not token_address:
            return jsonify({'success': False, 'error': 'Token address required'}), 400

        url = f"https://api.geckoterminal.com/api/v2/networks/solana/tokens/{token_address}/info"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        attributes = response.json()["data"]["attributes"]

        import base64
        from concurrent.futures import ThreadPoolExecutor

        # The two images live at separate URLs, so they can't arrive in a single
        # request — but download them concurrently instead of one-then-the-other,
        # which removes ~one image round-trip from the total wait.
        def _fetch_image_b64(image_url):
            return base64.b64encode(requests.get(image_url, timeout=10).content).decode('utf-8')

        with ThreadPoolExecutor(max_workers=2) as _pool:
            _thumb_future = _pool.submit(_fetch_image_b64, attributes["image"]["thumb"])
            _small_future = _pool.submit(_fetch_image_b64, attributes["image"]["small"])
            thumb_base64 = _thumb_future.result()
            small_base64 = _small_future.result()

        return jsonify({
            'success': True,
            'name': attributes["name"],
            'symbol': attributes["symbol"],
            'image_thumb': thumb_base64,
            'image_small': small_base64
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ========================================================================
# CONFIGURATION ENDPOINTS
# ========================================================================

def model_dir_setting():
    return (get_global_settings() or {}).get('model_dir') or 'artifacts'


def list_strategies():
    """The strategy files in the model folder, in file-name order, each checked against the
    models there. The default is the first usable one."""
    model_dir = model_dir_setting()
    out = []
    for name, path in strategy_files(model_dir).items():
        entry = {'name': name}
        try:
            doc = read_strategy(path)
            verify_models(model_dir, doc)
            sel = doc.get('selection', {})
            entry.update(valid=True, combo_cutoff=doc['combo_cutoff'], created_utc=doc.get('created_utc'),
                         selected_by=sel.get('selected_by'), run=sel.get('run'), n_runs=sel.get('n_runs'),
                         test_score=sel.get('test_score'), val_score=sel.get('val_score'))
        except Exception as e:
            entry.update(valid=False, error=str(e))
        out.append(entry)
    default = next((s['name'] for s in out if s['valid']), None)
    return {'model_dir': model_dir, 'strategies': out, 'default': default}


def ensure_strategy():
    """Load the default strategy (the first usable strategy file in the model folder) into
    strategy.py, for the parts of the dashboard that show one strategy. Loaded once; retried
    while it fails, so copying the artifacts in later works without a restart. Returns None
    or the error."""
    if STRATEGY:
        return None
    try:
        model_dir = model_dir_setting()
        default = list_strategies()['default']
        if default is None:
            raise ValueError(f"no usable strategy_<name>.json in {model_dir}/")
        doc = use_strategy(model_dir, default)
        verify_models(model_dir, doc)
        return None
    except Exception as e:
        STRATEGY.clear()
        TUNED_PARAMS.clear()
        return str(e)


def strategy_summary():
    sel = STRATEGY.get('selection', {})
    return {'name': STRATEGY.get('name'), 'created_utc': STRATEGY.get('created_utc'),
            'selected_by': sel.get('selected_by'), 'run': sel.get('run'), 'n_runs': sel.get('n_runs'),
            'val_score': sel.get('val_score'), 'test_score': sel.get('test_score')}


_startup_strategy_error = ensure_strategy()
print(f"Strategy: {_startup_strategy_error}" if _startup_strategy_error else
      f"Strategy '{STRATEGY.get('name')}' loaded (combo cutoff {TUNED_PARAMS.get('combo_cutoff')})")


@app.route('/api/strategies')
def api_strategies():
    """The strategies in the model folder for the bot setup dropdown (default: the first usable one)."""
    try:
        return jsonify(list_strategies())
    except Exception as e:
        return jsonify({'error': str(e), 'strategies': [], 'default': None}), 500


@app.route('/api/hardcoded-params')
def api_hardcoded_params():
    """The feature/model settings and the loaded strategy, for display."""
    err = ensure_strategy()
    return jsonify({
        'training_params': {
            'feature_set': 'Dataset_creationV9',
            'first_feature_row': START_IDX,
            'pump_sets': PUMP_SETS,
            'liquidity_refresh_minutes': LIQ_STATS_REFRESH_MIN,
            'move_label': '+10% within 30 min',
            'direction_label': '+15% before -15% within 2 h',
        },
        'strategy': None if err else strategy_summary(),
        'strategy_params': None if err else dict(TUNED_PARAMS, **FIXED_PARAMS),
        'strategy_error': err,
        'note': 'Feature settings are fixed in features.py; the strategy comes from the strategy file '
                'in the model folder.'
    })

@app.route('/api/config/strategy', methods=['GET', 'POST'])
def api_strategy_config():
    """The loaded strategy's parameters (from the strategy file; read-only)."""
    if request.method == 'POST':
        return jsonify({'success': False,
                        'error': 'Strategy parameters come from the strategy file in the model folder '
                                 '(strategy.ipynb export); they cannot be edited here'}), 405
    err = ensure_strategy()
    if err:
        return jsonify({'error': err}), 503
    return jsonify(dict(TUNED_PARAMS, **FIXED_PARAMS, strategy_name=STRATEGY.get('name')))

# ========================================================================
# EXISTING DATA ENDPOINTS
# ========================================================================

@app.route('/api/trades')
def api_trades():
    """Get all trades. All values in SOL."""
    try:
        limit = request.args.get('limit', 50, type=int)
        trades = get_all_trades(limit=limit)
        for trade in trades:
            if trade['entry_timestamp']:
                trade['entry_timestamp'] = str(trade['entry_timestamp'])
            if trade['exit_timestamp']:
                trade['exit_timestamp'] = str(trade['exit_timestamp'])
        return jsonify(trades)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades/<int:bot_id>')
def api_bot_trades(bot_id):
    """Get trades for a specific bot. All values in SOL."""
    try:
        limit = request.args.get('limit', 10, type=int)
        trades = get_bot_trades(bot_id, limit=limit)
        for trade in trades:
            if trade['entry_timestamp']:
                trade['entry_timestamp'] = str(trade['entry_timestamp'])
            if trade['exit_timestamp']:
                trade['exit_timestamp'] = str(trade['exit_timestamp'])
        return jsonify(trades)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs')
def api_logs():
    try:
        limit = request.args.get('limit', 100, type=int)
        bot_id = request.args.get('bot_id', None, type=int)
        logs = get_recent_logs(limit=limit, bot_id=bot_id)

        log_text = []
        for log in reversed(logs):
            bot_prefix = f"[Bot {log['bot_id']}] " if log['bot_id'] else "[SYSTEM] "
            log_text.append(f"{log['timestamp']} {log['log_level']}: {bot_prefix}{log['message']}")

        return jsonify({'logs': '\n'.join(log_text)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========================================================================
# RUN
# ========================================================================

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    for i in range(1, 11):
        os.makedirs(f'data/bot_{i}', exist_ok=True)

    print("=" * 70)
    print("TRADING BOT DASHBOARD SERVER")
    print("=" * 70)
    print(f"Database: {os.path.abspath('data/trading_bots.db')}")
    print(f"Starting server on http://0.0.0.0:8050")
    print(f"SOL price collector: Running (updates every 15 minutes)")
    print("=" * 70)

    app.run(host='0.0.0.0', port=8050, debug=True, use_reloader=False, threaded=True)