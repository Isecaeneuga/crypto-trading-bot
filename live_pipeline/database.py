import sqlite3
import os
import json
import hashlib
import time
import math
import functools
from datetime import datetime, timedelta
from contextlib import contextmanager

# Path to the SQLite database file, stored in the data/ subdirectory
DATABASE_PATH = "data/trading_bots.db"

# The research artifacts: the two models, the tune reference distributions and
# the tuned parameters, written by the export cell of strategy.ipynb.
DEFAULT_MODEL_DIR = "artifacts"
DEFAULT_PARAMS_PATH = "artifacts/best_params.json"


def _ensure_column(cursor, table, column, decl):
    """Add a column to an existing table if it is missing (schema migration
    for databases created by an older version)."""
    cursor.execute(f"PRAGMA table_info({table})")
    if column not in {row[1] for row in cursor.fetchall()}:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _sql_float(value):
    """Convert a value to float for SQLite, mapping None and NaN to NULL."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def retry_on_locked(max_retries=3, delay=0.5):
    """
    Decorator that retries a DB operation when SQLite reports the database
    is locked. Mirrors the hand-rolled loops this replaced: up to
    `max_retries` attempts with a fixed `delay` (seconds) between them,
    re-raising any non-'locked' OperationalError immediately and re-raising
    the 'locked' error if the final attempt still fails.

    This is a secondary guard on top of the connection-level
    busy_timeout=60000 set in get_db_connection().
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if 'locked' in str(e) and attempt < max_retries - 1:
                        time.sleep(delay)
                        continue
                    raise
        return wrapper
    return decorator

@contextmanager
def get_db_connection():
    """
    Context manager that provides a database connection with automatic
    commit/rollback handling. Every function that touches the database
    opens and closes its own connection through this.
    
    Settings:
    - timeout=60.0: Wait up to 60 seconds if another process holds the lock
    - row_factory=sqlite3.Row: Returns rows as dict-like objects instead of tuples
    - WAL journal mode: Allows concurrent readers while one writer is active
    - busy_timeout=60000: SQLite-level retry timeout (milliseconds) if DB is locked
    
    On success: commits the transaction
    On exception: rolls back the transaction, then re-raises
    Always: closes the connection
    """
    conn = sqlite3.connect(DATABASE_PATH, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA busy_timeout=60000')
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def init_database():
    """
    Creates all database tables if they don't exist, and seeds initial data.
    Called once at application startup (before any bot processes are launched).
    
    Tables created:
    - global_settings: Single-row table for system-wide config (wallet, timezone, etc.)
    - sol_price_history: Time series of SOL/USD prices for historical reference
    - bot_config: Static bot configuration (coin_symbol, pool, thresholds, images)
    - bot_runtime: Process lifecycle state (status, PID, force-sell flags)
    - bot_trading_state: Trading position state (capital, shares, entry price)
    - ohlcv_data: Per-bot OHLCV candle data with pre-computed features
    - trades: Trade history with entry/exit prices and P&L (SOL-denominated)
    - predictions: ML model predictions per bot per minute
    - capital_history: Time series of capital composition snapshots
    - system_logs: Application-wide log messages
    """
    os.makedirs("data", exist_ok=True)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # ============================================
        # GLOBAL_SETTINGS TABLE
        # ============================================

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS global_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                timezone TEXT DEFAULT 'Europe/Amsterdam',
                model_dir TEXT DEFAULT 'artifacts',
                params_path TEXT DEFAULT 'artifacts/best_params.json',
                finalization_lag_minutes INTEGER DEFAULT 1,
                historical_days INTEGER DEFAULT 30,
                wallet_sol REAL DEFAULT 0,
                wallet_initialized INTEGER DEFAULT 0,
                total_fees_sol REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                trading_mode TEXT DEFAULT 'paper',
                static_trading_fee_pct REAL DEFAULT 0.015,
                grace_period_seconds INTEGER DEFAULT 5,
                wallet_pubkey TEXT
            )
        ''')
        
        # Seed the single settings row if the table was just created
        cursor.execute("SELECT COUNT(*) as count FROM global_settings")
        if cursor.fetchone()['count'] == 0:
            cursor.execute('''
                INSERT INTO global_settings (id, historical_days) VALUES (1, 30)
            ''')

        # ============================================
        # SOL_PRICE_HISTORY TABLE
        # ============================================

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sol_price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                price_usd REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_sol_price_timestamp 
        ON sol_price_history(timestamp DESC)
        ''')
        
        # ============================================
        # BOT_CONFIG TABLE
        # ============================================

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bot_config (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                pool_address TEXT,
                coin_symbol TEXT,
                confidence_threshold REAL DEFAULT 0.907,
                slippage_pct REAL DEFAULT 0.015,
                max_price_impact_pct REAL DEFAULT 0.00333,
                debug_mode INTEGER DEFAULT 0,
                coin_image_thumb_base64 TEXT,
                coin_image_small_base64 TEXT,
                coin_assigned_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # ============================================
        # BOT_RUNTIME TABLE
        # ============================================

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bot_runtime (
                bot_id INTEGER PRIMARY KEY REFERENCES bot_config(id),
                status TEXT NOT NULL DEFAULT 'inactive',
                pid INTEGER,
                started_at TIMESTAMP,
                stopped_at TIMESTAMP,
                initializing INTEGER DEFAULT 0
            )
        ''')

        # ============================================
        # BOT_TRADING_STATE TABLE
        # ============================================

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bot_trading_state (
                bot_id INTEGER PRIMARY KEY REFERENCES bot_config(id),
                initial_capital_sol REAL,
                current_capital_sol REAL,
                shares_held REAL DEFAULT 0,
                entry_price_sol REAL,
                entry_timestamp TEXT,
                position TEXT DEFAULT 'closed',
                trades_count INTEGER DEFAULT 0,

                -- Trade state fields (display copies of strategy_state) --
                original_entry_price REAL,
                original_entry_timestamp TEXT,
                n_reentries INTEGER DEFAULT 0,
                last_sell_close REAL,
                last_sell_idx INTEGER,
                latest_activation_pct REAL,

                -- State-machine fields --
                trade_phase TEXT DEFAULT 'normal',
                below_floor_count INTEGER DEFAULT 0,
                max_price_seen REAL,
                -- The full strategy state machine (strategy.StrategyState as JSON);
                -- the source of truth, the columns above are copies for display.
                strategy_state TEXT,

                CHECK (
                    (position = 'closed' AND shares_held = 0 AND entry_price_sol IS NULL)
                    OR
                    (position = 'open' AND shares_held > 0 AND entry_price_sol IS NOT NULL)
                )
            )
        ''')

        # ============================================
        # STRATEGY_CONFIG TABLE
        # ============================================

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_config (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                config_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # The strategy config is seeded from best_params.json further below
        # (sync_strategy_config_from_file), after all tables exist.

        # ============================================
        # OHLCV_DATA TABLE
        # ============================================

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                bot_action TEXT,
                returns REAL,
                completed_pump INTEGER,
                vol_120 REAL,
                estimated_sol_1pct_impact REAL,
                liquidity_sample_age_minutes REAL,
                liquidity_estimate_std_pct REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(bot_id, timestamp),
                FOREIGN KEY (bot_id) REFERENCES bot_config(id)
            )
        ''')
        
        # Composite index for the most common query pattern:
        # "get latest N candles for bot X" (ORDER BY timestamp DESC)
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ohlcv_bot_timestamp 
            ON ohlcv_data(bot_id, timestamp DESC)
        ''')
        
        # ============================================
        # TRADES TABLE
        # ============================================

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id INTEGER NOT NULL,
                entry_timestamp TIMESTAMP NOT NULL,
                exit_timestamp TIMESTAMP,
                entry_price_sol REAL NOT NULL,
                exit_price_sol REAL,
                sol_price_at_entry REAL NOT NULL,
                sol_price_at_exit REAL,
                capital_sol REAL NOT NULL,
                pnl_sol REAL,
                pnl_pct REAL,
                minutes_held INTEGER,
                reason TEXT,
                entry_fee_sol REAL DEFAULT 0,
                exit_fee_sol REAL,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tx_signature_entry TEXT,
                intended_entry_price_sol REAL,
                intended_exit_price_sol REAL,
                tx_signature_exit TEXT,
                FOREIGN KEY (bot_id) REFERENCES bot_config(id)
            )
        ''')
        
        # Index for "get recent trades for bot X" queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_trades_bot_timestamp 
            ON trades(bot_id, exit_timestamp DESC)
        ''')
        
        # ============================================
        # PREDICTIONS TABLE
        # ============================================

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                prediction REAL NOT NULL,          -- combo percentile (buy gate)
                price REAL NOT NULL,
                dir_pct REAL,                      -- direction percentile
                move_pct REAL,                     -- move percentile
                raw_move REAL,
                raw_dir REAL,
                price_position_15m REAL,
                price_position_60m REAL,           -- the feature price_position_1h
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (bot_id) REFERENCES bot_config(id)
            )
        ''')

        # ============================================
        # CAPITAL_HISTORY TABLE
        # ============================================

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS capital_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                closed_capital_sol REAL NOT NULL,
                open_position_sol REAL NOT NULL,
                cumulative_fees_sol REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(bot_id, timestamp),
                FOREIGN KEY (bot_id) REFERENCES bot_config(id)
            )
        ''')
        
        # Index for prediction lookups (get latest predictions for bot X)
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_bot_timestamp 
            ON predictions(bot_id, timestamp DESC)
        ''')

        # Index for capital history lookups (get timeline for bot X)
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_capital_history_bot_timestamp 
            ON capital_history(bot_id, timestamp DESC)
        ''')

        # ============================================
        # WALLET_CAPITAL_HISTORY TABLE
        # ============================================

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS wallet_capital_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL UNIQUE,
                unallocated_sol REAL NOT NULL,
                total_initial_sol REAL NOT NULL DEFAULT 0,
                total_open_sol REAL NOT NULL DEFAULT 0,
                total_profit_sol REAL NOT NULL DEFAULT 0,
                total_fees_sol REAL NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_wallet_capital_history_timestamp 
            ON wallet_capital_history(timestamp DESC)
        ''')
        
        # ============================================
        # SYSTEM_LOGS TABLE
        # ============================================

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id INTEGER,
                log_level TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (bot_id) REFERENCES bot_config(id)
            )
        ''')
        
        # Index for "get recent logs" queries (sorted by newest first)
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_logs_timestamp 
            ON system_logs(timestamp DESC)
        ''')
        
        # ============================================
        # SEED 10 BOT SLOTS
        # ============================================

        cursor.execute("SELECT COUNT(*) as count FROM bot_config")
        count = cursor.fetchone()['count']

        if count == 0:
            for i in range(1, 11):
                cursor.execute('''
                    INSERT INTO bot_config (id, name) VALUES (?, ?)
                ''', (i, f'Bot {i}'))
                cursor.execute('''
                    INSERT INTO bot_runtime (bot_id, status) VALUES (?, 'inactive')
                ''', (i,))
                cursor.execute('''
                    INSERT INTO bot_trading_state (bot_id, position, shares_held) VALUES (?, 'closed', 0)
                ''', (i,))

        # ============================================
        # MIGRATIONS (databases created by the 5-tier version)
        # ============================================
        # Old tier columns (tier_model_path, entry_regime_tier, last_sell_tier,
        # predictions.regime_tier) are left in place but no longer used.
        _ensure_column(cursor, 'global_settings', 'model_dir', f"TEXT DEFAULT '{DEFAULT_MODEL_DIR}'")
        _ensure_column(cursor, 'global_settings', 'params_path', f"TEXT DEFAULT '{DEFAULT_PARAMS_PATH}'")
        # A stored path that no longer exists (e.g. from an older layout) falls
        # back to the default when the default is in place.
        cursor.execute("SELECT model_dir, params_path FROM global_settings WHERE id = 1")
        _row = cursor.fetchone()
        if _row:
            if _row['model_dir'] != DEFAULT_MODEL_DIR and not os.path.isdir(_row['model_dir'] or '') \
                    and os.path.isdir(DEFAULT_MODEL_DIR):
                cursor.execute("UPDATE global_settings SET model_dir = ? WHERE id = 1", (DEFAULT_MODEL_DIR,))
            if _row['params_path'] != DEFAULT_PARAMS_PATH and not os.path.exists(_row['params_path'] or '') \
                    and os.path.exists(DEFAULT_PARAMS_PATH):
                cursor.execute("UPDATE global_settings SET params_path = ? WHERE id = 1", (DEFAULT_PARAMS_PATH,))
        _ensure_column(cursor, 'bot_trading_state', 'strategy_state', 'TEXT')
        for col in ('dir_pct', 'move_pct', 'raw_move', 'raw_dir'):
            _ensure_column(cursor, 'predictions', col, 'REAL')

    sync_strategy_config_from_file()


def _file_sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def sync_strategy_config_from_file(force=False):
    """
    Make strategy_config the parameter set of best_params.json.

    Replaced when: nothing is stored yet, the stored config is the old 5-tier
    format, or best_params.json changed since it was loaded (its hash is stored
    as params_sha). Edits made through /api/config/strategy survive restarts
    until the file changes. The stored signal_exit_enabled switch is kept.
    """
    from strategy import load_best_params, is_legacy_config
    settings = get_global_settings() or {}
    path = settings.get('params_path') or DEFAULT_PARAMS_PATH
    current = get_strategy_config()
    if not os.path.exists(path):
        if current is None or is_legacy_config(current):
            log_message(f"Strategy parameters not found at {path}: bots cannot start until "
                        f"best_params.json is in place", None, 'ERROR')
        return current
    sha = _file_sha(path)
    if (not force and current and not is_legacy_config(current)
            and current.get('params_sha') == sha):
        return current
    config = load_best_params(path)
    config['params_sha'] = sha
    if current and not is_legacy_config(current) and 'signal_exit_enabled' in current:
        config['signal_exit_enabled'] = bool(current['signal_exit_enabled'])
    with get_db_connection() as conn:
        conn.execute('''
            INSERT INTO strategy_config (id, config_json, updated_at) VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET config_json = excluded.config_json,
                                          updated_at = excluded.updated_at
        ''', (json.dumps(config), datetime.now()))
    log_message(f"Strategy config loaded from {path} (run {config.get('params_run')}, "
                f"signal exit {'ON' if config.get('signal_exit_enabled') else 'off'})", None, 'INFO')
    return config


# ============================================
# GLOBAL SETTINGS OPERATIONS
# ============================================

def get_global_settings():
    """
    Fetch the single global settings row.
    Returns a dict of all settings, or None if the table is empty
    (which shouldn't happen after init_database runs).
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM global_settings WHERE id = 1")
        row = cursor.fetchone()
        return dict(row) if row else None


# ============================================
# STRATEGY CONFIG OPERATIONS
# ============================================

def get_strategy_config():
    """
    Fetch the strategy config as a Python dict.
    Returns the parsed JSON config, or None if not seeded.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT config_json FROM strategy_config WHERE id = 1")
        row = cursor.fetchone()
        return json.loads(row['config_json']) if row else None

def update_strategy_config(config_dict):
    """
    Replace the strategy config with a new dict.
    Stores as JSON in the single-row strategy_config table.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE strategy_config SET config_json = ?, updated_at = ? WHERE id = 1
        ''', (json.dumps(config_dict), datetime.now()))
    return get_strategy_config()


# ============================================
# BOT OPERATIONS
# ============================================

# Shared SELECT + JOINs for the full bot view (config + runtime + trading state).
# get_bot appends a WHERE clause; get_all_bots_from_db appends ORDER BY.
_BOT_VIEW_SELECT = '''
    SELECT c.*, r.status, r.pid, r.started_at, r.stopped_at, r.initializing,
           t.initial_capital_sol, t.current_capital_sol, t.shares_held,
           t.entry_price_sol, t.entry_timestamp,
           t.position, t.trades_count,
           t.original_entry_price, t.original_entry_timestamp,
           t.n_reentries,
           t.last_sell_close, t.last_sell_idx,
           t.latest_activation_pct,
           t.trade_phase, t.below_floor_count, t.max_price_seen,
           t.strategy_state
    FROM bot_config c
    JOIN bot_runtime r ON c.id = r.bot_id
    JOIN bot_trading_state t ON c.id = t.bot_id
'''

def get_bot(bot_id):
    """Get complete bot view by joining all three bot tables."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(_BOT_VIEW_SELECT + ' WHERE c.id = ?', (bot_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_all_bots_from_db():
    """Get all 10 bots with full details."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(_BOT_VIEW_SELECT + ' ORDER BY c.id')
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

# Columns that belong to each table, used for routing
BOT_CONFIG_COLUMNS = {
    'name', 'pool_address', 'confidence_threshold', 'slippage_pct',
    'max_price_impact_pct', 'coin_symbol', 'coin_image_thumb_base64',
    'coin_image_small_base64', 'debug_mode', 'coin_assigned_at'
}

BOT_RUNTIME_COLUMNS = {
    'status', 'pid', 'started_at', 'stopped_at', 'initializing'
}

BOT_TRADING_STATE_COLUMNS = {
    'initial_capital_sol', 'current_capital_sol', 'shares_held',
    'entry_price_sol', 'entry_timestamp',
    'position', 'trades_count',
    'original_entry_price', 'original_entry_timestamp',
    'n_reentries', 'last_sell_close', 'last_sell_idx',
    'latest_activation_pct',
    # State-machine fields
    'trade_phase', 'below_floor_count', 'max_price_seen', 'strategy_state',
}

def _update_bot_table(table, where_col, bot_id, allowed_columns,
                      add_updated_at, kwargs):
    """
    Shared implementation for the update_bot_* helpers.

    Builds a dynamic UPDATE from kwargs, keeping only keys present in
    allowed_columns. Optionally appends an updated_at = now() assignment.
    No-op (returns without touching the DB) if nothing valid was passed.
    """
    fields = []
    values = []
    for key, value in kwargs.items():
        if key in allowed_columns:
            fields.append(f"{key} = ?")
            values.append(value)
    if not fields:
        return
    if add_updated_at:
        fields.append("updated_at = ?")
        values.append(datetime.now())
    values.append(bot_id)
    with get_db_connection() as conn:
        conn.execute(
            f"UPDATE {table} SET {', '.join(fields)} WHERE {where_col} = ?",
            values
        )

def update_bot_config(bot_id, **kwargs):
    """Update static bot configuration."""
    _update_bot_table('bot_config', 'id', bot_id,
                      BOT_CONFIG_COLUMNS, add_updated_at=True, kwargs=kwargs)

def update_bot_runtime(bot_id, **kwargs):
    """Update bot process lifecycle state."""
    _update_bot_table('bot_runtime', 'bot_id', bot_id,
                      BOT_RUNTIME_COLUMNS, add_updated_at=False, kwargs=kwargs)

def update_bot_trading_state(bot_id, **kwargs):
    """Update bot trading position state. Should only be called by the bot process."""
    _update_bot_table('bot_trading_state', 'bot_id', bot_id,
                      BOT_TRADING_STATE_COLUMNS, add_updated_at=False, kwargs=kwargs)

def initialize_bot_in_db(bot_id, pool_address, coin_symbol, initial_capital_sol,
                        image_thumb_base64=None, image_small_base64=None,
                        confidence_threshold=0.907, max_price_impact_pct=0.00333):
    """Initialize a bot across all three tables."""

    # Clear old data (run once, not part of the retried section below)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM capital_history WHERE bot_id = ?', (bot_id,))
        cursor.execute('DELETE FROM predictions WHERE bot_id = ?', (bot_id,))
        cursor.execute('DELETE FROM ohlcv_data WHERE bot_id = ?', (bot_id,))
        cursor.execute('DELETE FROM trades WHERE bot_id = ?', (bot_id,))

    return _write_initial_bot_state(
        bot_id, pool_address, coin_symbol, initial_capital_sol,
        image_thumb_base64, image_small_base64,
        confidence_threshold, max_price_impact_pct
    )


@retry_on_locked()
def _write_initial_bot_state(bot_id, pool_address, coin_symbol, initial_capital_sol,
                             image_thumb_base64, image_small_base64,
                             confidence_threshold, max_price_impact_pct):
    """Write the fresh config/runtime/trading-state rows for a bot. Retried on lock."""
    update_bot_config(
        bot_id,
        pool_address=pool_address,
        coin_symbol=coin_symbol,
        coin_image_thumb_base64=image_thumb_base64,
        coin_image_small_base64=image_small_base64,
        confidence_threshold=confidence_threshold,
        max_price_impact_pct=max_price_impact_pct,
        # Start the assigned-to-coin clock. This is the only place it is set;
        # it is untouched by start/pause/resume and cleared only on deallocate.
        coin_assigned_at=datetime.now()
    )
    update_bot_runtime(bot_id, status='stopped')
    update_bot_trading_state(
        bot_id,
        initial_capital_sol=initial_capital_sol,
        current_capital_sol=initial_capital_sol,
        shares_held=0,
        entry_price_sol=None,
        position='closed',
        trades_count=0,
        entry_timestamp=None,
        original_entry_price=None,
        original_entry_timestamp=None,
        n_reentries=0,
        last_sell_close=None,
        last_sell_idx=None,
        latest_activation_pct=None,
        # State-machine reset
        trade_phase='normal',
        below_floor_count=0,
        max_price_seen=None,
        strategy_state=None
    )

    # Record initial capital snapshots (2 rows for visible chart area)
    now = datetime.now()
    ts_prev = (now - timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:00')
    ts_now = now.strftime('%Y-%m-%d %H:%M:00')
    insert_capital_snapshot(bot_id, ts_prev, initial_capital_sol, 0, 0)
    insert_capital_snapshot(bot_id, ts_now, initial_capital_sol, 0, 0)

    return get_bot(bot_id)



# ============================================
# OHLCV OPERATIONS
# ============================================

def insert_ohlcv_data(bot_id, ohlcv_list):
    """
    Bulk insert OHLCV candle data for a bot, including pre-computed
    feature columns. Uses INSERT OR REPLACE so duplicate timestamps
    (same bot_id + timestamp) get overwritten.

    Uses _sql_float() to convert NaN values to SQL NULL, since SQLite
    doesn't understand Python's float('nan').
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        for row in ohlcv_list:
            cursor.execute('''
                INSERT OR REPLACE INTO ohlcv_data 
                (bot_id, timestamp, open, high, low, close, volume, bot_action,
                 returns, completed_pump, vol_120, 
                 estimated_sol_1pct_impact, liquidity_sample_age_minutes, liquidity_estimate_std_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                bot_id,
                str(row['timestamp']),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']),
                row.get('bot_action'),
                _sql_float(row.get('returns')),
                int(row['completed_pump']) if row.get('completed_pump') is not None else None,
                _sql_float(row.get('vol_120')),
                _sql_float(row.get('estimated_sol_1pct_impact')),
                _sql_float(row.get('liquidity_sample_age_minutes')),
                _sql_float(row.get('liquidity_estimate_std_pct'))
            ))

def get_latest_ohlcv(bot_id, limit=200):
    """
    Get the most recent OHLCV candles for a bot, ordered oldest-first.
    
    Fetches the newest `limit` rows (ORDER BY DESC), then reverses them
    so the caller gets chronological order. Used for chart rendering
    and feature computation.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM ohlcv_data 
            WHERE bot_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (bot_id, limit))
        rows = cursor.fetchall()
        return [dict(row) for row in reversed(rows)]

def get_all_ohlcv(bot_id):
    """
    Every stored candle of a bot (the coin's full history, which the features
    need), oldest first. Callers sort again by parsed time: ordering by the
    timestamp TEXT can misplace rows around a DST change.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, open, high, low, close, volume, bot_action
            FROM ohlcv_data WHERE bot_id = ? ORDER BY timestamp ASC
        ''', (bot_id,))
        return [dict(row) for row in cursor.fetchall()]


def get_latest_timestamp(bot_id):
    """
    Get the most recent candle timestamp for a bot.
    Used to determine where to start fetching new data from the API.
    Returns None if no data exists yet.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT MAX(timestamp) as latest FROM ohlcv_data WHERE bot_id = ?
        ''', (bot_id,))
        result = cursor.fetchone()
        return result['latest'] if result else None


# ============================================
# TRADE OPERATIONS
# ============================================

def insert_open_trade(bot_id, entry_timestamp, entry_price_sol,
                      sol_price_at_entry, capital_sol, entry_fee_sol):
    """Record a new BUY trade with exit fields set to NULL."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades
            (bot_id, entry_timestamp, entry_price_sol, sol_price_at_entry,
             capital_sol, entry_fee_sol, last_modified)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (bot_id, entry_timestamp, entry_price_sol, sol_price_at_entry,
              capital_sol, entry_fee_sol, datetime.now()))
        return cursor.lastrowid

def update_trade_on_exit(trade_id, exit_timestamp, exit_price_sol,
                        sol_price_at_exit, pnl_sol, pnl_pct,
                        minutes_held, reason, exit_fee_sol):
    """Close an open trade by filling in exit data."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE trades
            SET exit_timestamp = ?,
                exit_price_sol = ?,
                sol_price_at_exit = ?,
                pnl_sol = ?,
                pnl_pct = ?,
                minutes_held = ?,
                reason = ?,
                exit_fee_sol = ?,
                last_modified = ?
            WHERE id = ?
        ''', (exit_timestamp, exit_price_sol, sol_price_at_exit,
              pnl_sol, pnl_pct, minutes_held, reason,
              exit_fee_sol, datetime.now(), trade_id))
        return cursor.rowcount > 0

def get_open_trade_for_bot(bot_id):
    """
    Find the currently open trade for a bot (one where exit_timestamp is NULL).
    Returns the most recent one if somehow multiple exist (shouldn't happen
    in normal operation).
    
    Used by the bot process to find the trade record to update on SELL,
    and to get sol_price_at_entry for P&L calculation.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM trades 
            WHERE bot_id = ? AND exit_timestamp IS NULL
            ORDER BY entry_timestamp DESC
            LIMIT 1
        ''', (bot_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_bot_trades(bot_id, limit=10):
    """
    Get the most recent trades for a specific bot, newest first.
    Used for the per-bot trade history display in the Manage tab.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT *
            FROM trades
            WHERE bot_id = ? 
            ORDER BY entry_timestamp DESC 
            LIMIT ?
        ''', (bot_id, limit))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_all_trades(limit=50):
    """
    Get the most recent trades across ALL bots, newest first.
    Joins with bot_config to include coin symbol.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT t.*, c.coin_symbol, c.name as bot_name
            FROM trades t
            JOIN bot_config c ON t.bot_id = c.id
            ORDER BY t.entry_timestamp DESC 
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


# ============================================
# PREDICTION OPERATIONS
# ============================================

def insert_predictions(bot_id, rows):
    """
    Store the scores of one or more minutes (one transaction). Each row:
    timestamp, combo_pct (stored as `prediction`), price, dir_pct, move_pct,
    raw_move, raw_dir, price_position_15m, price_position_1h (stored in the
    price_position_60m column).
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        for r in rows:
            cursor.execute('''
                INSERT INTO predictions (bot_id, timestamp, prediction, price, dir_pct, move_pct,
                                         raw_move, raw_dir, price_position_15m, price_position_60m)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (bot_id, str(r['timestamp']), float(r['combo_pct']), float(r['price']),
                  _sql_float(r.get('dir_pct')), _sql_float(r.get('move_pct')),
                  _sql_float(r.get('raw_move')), _sql_float(r.get('raw_dir')),
                  _sql_float(r.get('price_position_15m')), _sql_float(r.get('price_position_1h'))))


def insert_capital_snapshot(bot_id, timestamp, closed_capital_sol, open_position_sol, cumulative_fees_sol):
    """Record a capital composition snapshot for one minute."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO capital_history
            (bot_id, timestamp, closed_capital_sol, open_position_sol, cumulative_fees_sol)
            VALUES (?, ?, ?, ?, ?)
        ''', (bot_id, timestamp, closed_capital_sol, open_position_sol, cumulative_fees_sol))

def get_bot_cumulative_fees_sol(bot_id):
    """
    Calculate total fees paid by a bot by summing entry and exit fees
    across all its trades. Returns 0.0 if no trades exist.
    
    Used during capital snapshot recording to fill the
    cumulative_fees_sol field.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT SUM(COALESCE(entry_fee_sol, 0) + COALESCE(exit_fee_sol, 0)) as total_fees
            FROM trades
            WHERE bot_id = ?
        ''', (bot_id,))
        result = cursor.fetchone()
        return result['total_fees'] if result['total_fees'] else 0.0

def get_capital_history_page(bot_id, limit=1000, offset=0):
    """
    Get capital history page, newest first, returned in chronological order.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM capital_history 
            WHERE bot_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        ''', (bot_id, limit, offset))
        rows = cursor.fetchall()
        return [dict(row) for row in reversed(rows)]


def get_capital_history_after(bot_id, after_timestamp):
    """Get capital history rows after a given timestamp for incremental updates."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM capital_history 
            WHERE bot_id = ? AND timestamp > ?
            ORDER BY timestamp ASC
        ''', (bot_id, after_timestamp))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


# ============================================
# WALLET CAPITAL HISTORY OPERATIONS
# ============================================

def insert_wallet_capital_snapshot(timestamp, unallocated_sol, total_initial_sol=0,
                                   total_open_sol=0, total_profit_sol=0, total_fees_sol=0):
    """Record a wallet-level capital composition snapshot for one minute."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO wallet_capital_history
            (timestamp, unallocated_sol, total_initial_sol, total_open_sol, total_profit_sol, total_fees_sol)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (str(timestamp), _sql_float(unallocated_sol), _sql_float(total_initial_sol),
              _sql_float(total_open_sol), _sql_float(total_profit_sol), _sql_float(total_fees_sol)))


def get_wallet_capital_page(limit=1000, offset=0):
    """Get wallet capital history page, newest first, returned in chronological order."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, unallocated_sol, total_initial_sol,
                   total_open_sol, total_profit_sol, total_fees_sol
            FROM wallet_capital_history
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        rows = cursor.fetchall()
        return [dict(row) for row in reversed(rows)]


def get_wallet_capital_after(after_timestamp):
    """Get wallet capital rows after a timestamp for incremental updates."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, unallocated_sol, total_initial_sol,
                   total_open_sol, total_profit_sol, total_fees_sol
            FROM wallet_capital_history
            WHERE timestamp > ?
            ORDER BY timestamp ASC
        ''', (after_timestamp,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


    
def get_last_modified_timestamps():
    """Get the most recent modification timestamps for change detection."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute('SELECT MAX(updated_at) as config_modified FROM bot_config')
        config_modified = cursor.fetchone()['config_modified']

        cursor.execute('SELECT MAX(last_modified) as trades_modified FROM trades')
        trades_modified = cursor.fetchone()['trades_modified']

        cursor.execute('SELECT MAX(created_at) as capital_modified FROM capital_history')
        capital_modified = cursor.fetchone()['capital_modified']

        cursor.execute('SELECT bot_id, MAX(timestamp) as ts FROM ohlcv_data GROUP BY bot_id')
        ohlcv_modified = {row['bot_id']: row['ts'] for row in cursor.fetchall()}

        return {
            'bots': config_modified,
            'trades': trades_modified,
            'capital': capital_modified,
            'ohlcv': ohlcv_modified
        }


# ============================================
# LOGGING OPERATIONS
# ============================================

def log_message(message, bot_id=None, level='INFO'):
    """
    Insert a log message into the system_logs table.
    bot_id=None means it's a system-level message.
    
    Called from both the Flask app and bot processes.
    NOTE: This is one of the tables where multiple processes write,
    but since logs are append-only (INSERT only, no UPDATE), lock
    contention is minimal.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO system_logs (bot_id, log_level, message)
            VALUES (?, ?, ?)
        ''', (bot_id, level, message))

def get_recent_logs(limit=100, bot_id=None):
    """
    Get recent log messages, newest first.
    
    If bot_id is provided: returns logs for that specific bot
    PLUS system-level logs (bot_id IS NULL). This is used for
    the per-bot terminal view.
    
    If bot_id is None: returns all logs (used for the main
    Pipeline Terminal).
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        if bot_id:
            cursor.execute('''
                SELECT * FROM system_logs 
                WHERE bot_id = ? OR bot_id IS NULL
                ORDER BY id DESC 
                LIMIT ?
            ''', (bot_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM system_logs 
                ORDER BY id DESC 
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    
# ============================================
# SOL PRICE OPERATIONS
# ============================================

def insert_sol_price(price_usd, timestamp):
    """
    Append a new SOL/USD price point. Called by the background
    price collector thread in app.py every 15 minutes.
    Returns the new row's ID.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sol_price_history (price_usd, timestamp)
            VALUES (?, ?)
        ''', (price_usd, timestamp))
        
        return cursor.lastrowid

def get_latest_sol_price():
    """
    Get the most recent SOL/USD price.
    Returns a dict with 'price_usd' and 'timestamp', or None
    if no prices have been recorded yet (e.g., first startup
    before the collector has run).
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT price_usd, timestamp 
            FROM sol_price_history 
            ORDER BY timestamp DESC 
            LIMIT 1
        ''')
        row = cursor.fetchone()
        return dict(row) if row else None



# ============================================
# ANALYTICS OPERATIONS
# ============================================

def get_bot_stats_precomputed(bot_id):
    """
    Compute detailed statistics for a bot, used by the Bot Info card
    in the Manage tab. Includes counts, win-rate, holding times,
    fees, latest capital snapshot, time-between-trades analysis,
    and time-since-last-trade.
    
    Uses SQL window functions (LAG) to calculate gaps between
    consecutive trades without fetching all trades to Python.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Closed-trade-only aggregates (counts, win-rate, holding times).
        # Open trades have no exit yet so they're excluded from these.
        cursor.execute('''
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl_sol > 0 THEN 1 ELSE 0 END) as winning_trades,
                AVG(minutes_held) as avg_holding_minutes,
                SUM(minutes_held) as total_minutes_held
            FROM trades
            WHERE bot_id = ? AND exit_timestamp IS NOT NULL
        ''', (bot_id,))

        stats = dict(cursor.fetchone())
        stats['avg_holding_minutes'] = stats.get('avg_holding_minutes') or 0.0
        stats['total_trades'] = stats.get('total_trades') or 0
        stats['winning_trades'] = stats.get('winning_trades') or 0

        # Fees include the entry fee of an open trade as well — that fee
        # has already been paid out of the wallet at buy time. Summing
        # entry_fee + exit_fee per row captures both legs; an open trade
        # has a NULL exit_fee, so COALESCE keeps just its entry fee.
        cursor.execute('''
            SELECT SUM(COALESCE(entry_fee_sol, 0) + COALESCE(exit_fee_sol, 0)) as total_fees_sol
            FROM trades
            WHERE bot_id = ?
        ''', (bot_id,))
        fee_row = cursor.fetchone()
        stats['total_fees_sol'] = (fee_row['total_fees_sol'] or 0.0) if fee_row else 0.0

        # Latest capital snapshot — includes both the bot's closed
        # (cash) capital and the open-position value at the time of the
        # snapshot. The Bot Info card uses this so "Current Capital" and
        # "P&L" reflect total wealth (cash + open position), not just
        # cash, while a position is open.
        cursor.execute('''
            SELECT closed_capital_sol, open_position_sol
            FROM capital_history
            WHERE bot_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (bot_id,))
        snap = cursor.fetchone()
        if snap:
            stats['latest_closed_capital_sol'] = snap['closed_capital_sol'] or 0.0
            stats['latest_open_position_sol'] = snap['open_position_sol'] or 0.0
        else:
            stats['latest_closed_capital_sol'] = None
            stats['latest_open_position_sol'] = None

        total_trades = stats['total_trades'] or 0
        winning_trades = stats['winning_trades'] or 0

        stats['win_rate'] = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate average time gap between consecutive trades
        # Uses LAG window function to get previous trade's exit time
        # for each trade's entry time, then computes the gap in minutes
        if total_trades > 1:
            cursor.execute('''
                SELECT 
                    entry_timestamp,
                    LAG(exit_timestamp) OVER (ORDER BY entry_timestamp) as prev_exit
                FROM trades
                WHERE bot_id = ? AND exit_timestamp IS NOT NULL
                ORDER BY entry_timestamp
            ''', (bot_id,))
            
            gaps = cursor.fetchall()
            total_gap_minutes = 0
            gap_count = 0
            
            for row in gaps:
                if row['prev_exit']:
                    entry = datetime.fromisoformat(row['entry_timestamp'])
                    prev_exit = datetime.fromisoformat(row['prev_exit'])
                    
                    # Strip timezone info for subtraction (both should be
                    # in the same timezone, but fromisoformat preserves it)
                    if entry.tzinfo is not None:
                        entry = entry.replace(tzinfo=None)
                    if prev_exit.tzinfo is not None:
                        prev_exit = prev_exit.replace(tzinfo=None)
                    
                    gap_minutes = (entry - prev_exit).total_seconds() / 60
                    if gap_minutes >= 0:
                        total_gap_minutes += gap_minutes
                        gap_count += 1
            
            stats['avg_gap_minutes'] = total_gap_minutes / gap_count if gap_count > 0 else None
        else:
            stats['avg_gap_minutes'] = None
        
        # Calculate how long since the bot last opened a position
        # Used to show "Time Since Last Open" in the Bot Info card
        cursor.execute('''
            SELECT MAX(entry_timestamp) as last_entry
            FROM trades
            WHERE bot_id = ?
        ''', (bot_id,))
        
        last_entry_row = cursor.fetchone()
        if last_entry_row and last_entry_row['last_entry']:
            last_entry = datetime.fromisoformat(last_entry_row['last_entry'])
            if last_entry.tzinfo is not None:
                last_entry = last_entry.replace(tzinfo=None)
            minutes_since = (datetime.now() - last_entry).total_seconds() / 60
            stats['minutes_since_last_open'] = minutes_since
        else:
            stats['minutes_since_last_open'] = None

        # How long since the bot last SOLD (closed a position). Backs the
        # "Time Since Last Open" metric when the bot is currently flat: it
        # reports how long the position has been closed. None if no completed
        # sell exists yet.
        cursor.execute('''
            SELECT MAX(exit_timestamp) as last_exit
            FROM trades
            WHERE bot_id = ? AND exit_timestamp IS NOT NULL
        ''', (bot_id,))

        last_exit_row = cursor.fetchone()
        if last_exit_row and last_exit_row['last_exit']:
            last_exit = datetime.fromisoformat(last_exit_row['last_exit'])
            if last_exit.tzinfo is not None:
                last_exit = last_exit.replace(tzinfo=None)
            stats['minutes_since_last_sell'] = (datetime.now() - last_exit).total_seconds() / 60
        else:
            stats['minutes_since_last_sell'] = None

        return stats

# ============================================
# WALLET OPERATIONS
# ============================================

@retry_on_locked()
def initialize_wallet(sol_amount, trading_mode='paper',
                      static_trading_fee_pct=None, grace_period_seconds=None,
                      finalization_lag_minutes=None, wallet_pubkey=None):
    """
    Initialize the wallet with a SOL amount and a trading mode.

    trading_mode is one of 'paper', 'shadow', 'live'. For paper mode the
    caller may also set the static trading fee (decimal, e.g. 0.015), the
    per-minute grace period (seconds), and the finalization lag (minutes) —
    how far behind real time the most recent usable candle is; when omitted
    the existing/default values are kept. Shadow and live modes ignore these
    (they derive real fees from the (un)executed transaction).

    The mode is a one-time choice — there is no supported path to change it
    after initialization.
    """
    updates = {
        'wallet_sol': sol_amount,
        'wallet_initialized': 1,
        'trading_mode': trading_mode,
        'updated_at': datetime.now(),
    }
    if trading_mode == 'paper':
        if static_trading_fee_pct is not None:
            updates['static_trading_fee_pct'] = static_trading_fee_pct
        if grace_period_seconds is not None:
            updates['grace_period_seconds'] = grace_period_seconds
        if finalization_lag_minutes is not None:
            updates['finalization_lag_minutes'] = finalization_lag_minutes
    elif trading_mode in ('shadow', 'live'):
        # Real wallet the bot quotes/monitors against (public key only for now).
        if wallet_pubkey is not None:
            updates['wallet_pubkey'] = wallet_pubkey

    cols = list(updates.keys())  # column names are fixed literals, not user input
    set_clause = ', '.join(f'{c} = ?' for c in cols)
    params = [updates[c] for c in cols]

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE global_settings SET {set_clause} WHERE id = 1",
            params
        )

    log_message(
        f'Wallet initialized with {sol_amount} SOL (mode: {trading_mode})',
        None, 'INFO'
    )

    # Record initial wallet capital snapshots (2 rows for visible chart area)
    now = datetime.now()
    ts_prev = (now - timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:00')
    ts_now = now.strftime('%Y-%m-%d %H:%M:00')
    insert_wallet_capital_snapshot(ts_prev, sol_amount, 0, 0, 0, 0)
    insert_wallet_capital_snapshot(ts_now, sol_amount, 0, 0, 0, 0)

    return get_global_settings()

def get_wallet_status():
    """Get current wallet status with a LIVE allocation breakdown.

    'allocated_sol' (Allocated to Bots) and 'total_value_sol' (Total Wallet)
    reflect the CURRENT value of each bot's capital — its closed (idle) capital
    plus the marked-to-market value of any open position — so both move as bots
    trade and hold open positions, rather than being frozen at the amounts
    committed when the bots were first allocated.

    'unallocated_wallet_sol' is the free wallet pool (deposit minus what was
    handed to bots at allocation time); it only changes when bots are
    (de)allocated, not as they trade. 'wallet_sol' stays the deposit base used
    for percentage-based allocation sizing.
    """
    settings = get_global_settings()
    
    if not settings or settings['wallet_initialized'] != 1:
        return {
            'initialized': False,
            'wallet_sol': 0,
            'unallocated_wallet_sol': 0,
            'total_fees_sol': 0,
            'allocated_sol': 0,
            'total_value_sol': 0,
            # Current defaults so the init form can prefill paper-mode fields.
            'trading_mode': (settings or {}).get('trading_mode', 'paper'),
            'static_trading_fee_pct': (settings or {}).get('static_trading_fee_pct', 0.015),
            'grace_period_seconds': (settings or {}).get('grace_period_seconds', 5),
            'finalization_lag_minutes': (settings or {}).get('finalization_lag_minutes', 1)
        }
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # For each active bot: the amount committed at allocation (initial) and
        # the latest capital snapshot's closed + open value. open_position_sol is
        # marked to market by the bot loop every minute, so closed + open is the
        # bot's live worth. Falls back to current_capital_sol (then initial) when
        # no snapshot exists yet (e.g. a freshly initialized bot).
        cursor.execute('''
            SELECT
                t.initial_capital_sol AS initial_sol,
                t.current_capital_sol AS current_sol,
                (SELECT ch.closed_capital_sol FROM capital_history ch
                   WHERE ch.bot_id = t.bot_id
                   ORDER BY ch.timestamp DESC LIMIT 1) AS latest_closed,
                (SELECT ch.open_position_sol FROM capital_history ch
                   WHERE ch.bot_id = t.bot_id
                   ORDER BY ch.timestamp DESC LIMIT 1) AS latest_open
            FROM bot_trading_state t
            JOIN bot_runtime r ON t.bot_id = r.bot_id
            WHERE r.status != 'inactive'
        ''')
        
        committed_initial = 0.0   # handed to bots at allocation (for unallocated)
        allocated_sol = 0.0       # live value held by bots (closed + open)
        for row in cursor.fetchall():
            committed_initial += row['initial_sol'] or 0
            
            if row['latest_closed'] is not None:
                closed = row['latest_closed'] or 0
            elif row['current_sol'] is not None:
                closed = row['current_sol']
            else:
                closed = row['initial_sol'] or 0
            open_val = row['latest_open'] or 0
            allocated_sol += closed + open_val
        
        # Read persistent wallet-level fees counter (incremented by deduct_trading_fee
        # on every buy/sell). Survives bot deallocations because it is never
        # recomputed from the trades table.
        total_fees_sol = settings['total_fees_sol'] if settings['total_fees_sol'] else 0
        
        # Free wallet pool = deposit - sum(initial_capital) committed to bots.
        # Bot P&L lives inside allocated_sol, not here.
        unallocated_wallet_sol = settings['wallet_sol'] - committed_initial
    
    # Live total portfolio value = free pool + current value held by bots.
    total_value_sol = unallocated_wallet_sol + allocated_sol
    
    return {
        'initialized': True,
        'wallet_sol': settings['wallet_sol'],          # deposit base (allocation %)
        'unallocated_wallet_sol': unallocated_wallet_sol,
        'total_fees_sol': total_fees_sol,
        'allocated_sol': allocated_sol,                # live: closed + open
        'total_value_sol': total_value_sol,            # live total portfolio value
        'trading_mode': settings.get('trading_mode', 'paper'),
        'wallet_pubkey': settings.get('wallet_pubkey')
    }

def deallocate_from_bot(bot_id):
    """Return bot's capital to unallocated wallet and reset bot."""
    bot = get_bot(bot_id)

    if not bot:
        raise Exception("Bot not found")
    if bot['status'] == 'running':
        raise Exception("Cannot deallocate running bot. Pause it first.")
    if bot['position'] == 'open':
        raise Exception("Cannot deallocate bot with open position.")
    if bot['status'] == 'inactive':
        raise Exception("Bot is already inactive")

    capital_to_return = bot['current_capital_sol'] or 0
    initial_allocated = bot['initial_capital_sol'] or 0
    profit_or_loss = capital_to_return - initial_allocated

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Update wallet_sol to reflect profit/loss from this bot
        if profit_or_loss != 0:
            cursor.execute('''
                UPDATE global_settings
                SET wallet_sol = wallet_sol + ?
            ''', (profit_or_loss,))

        # Reset config
        cursor.execute('''
            UPDATE bot_config
            SET pool_address = NULL, coin_symbol = NULL,
                coin_image_thumb_base64 = NULL, coin_image_small_base64 = NULL,
                coin_assigned_at = NULL,
                updated_at = ?
            WHERE id = ?
        ''', (datetime.now(), bot_id))

        # Reset runtime
        cursor.execute('''
            UPDATE bot_runtime
            SET status = 'inactive', pid = NULL, started_at = NULL, stopped_at = NULL
            WHERE bot_id = ?
        ''', (bot_id,))

        # Reset trading state
        cursor.execute('''
            UPDATE bot_trading_state
            SET initial_capital_sol = NULL, current_capital_sol = NULL,
                shares_held = 0, entry_price_sol = NULL, position = 'closed',
                trades_count = 0, entry_timestamp = NULL,
                original_entry_price = NULL, original_entry_timestamp = NULL,
                n_reentries = 0,
                last_sell_close = NULL, last_sell_idx = NULL,
                latest_activation_pct = NULL,
                trade_phase = 'normal', below_floor_count = 0, max_price_seen = NULL,
                strategy_state = NULL
            WHERE bot_id = ?
        ''', (bot_id,))

        # Clear historical data
        cursor.execute('DELETE FROM capital_history WHERE bot_id = ?', (bot_id,))
        cursor.execute('DELETE FROM predictions WHERE bot_id = ?', (bot_id,))
        cursor.execute('DELETE FROM ohlcv_data WHERE bot_id = ?', (bot_id,))
        cursor.execute('DELETE FROM trades WHERE bot_id = ?', (bot_id,))

    log_message(f'Bot {bot_id} deallocated. Returned {capital_to_return:.4f} SOL to wallet.', bot_id, 'INFO')
    return True

@retry_on_locked()
def deduct_trading_fee(fee_sol):
    """Deduct trading fee from wallet by adding to total fees"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE global_settings 
            SET total_fees_sol = total_fees_sol + ?,
                updated_at = ?
            WHERE id = 1
        ''', (fee_sol, datetime.now()))

    return True