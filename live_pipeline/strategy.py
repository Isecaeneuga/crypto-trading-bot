"""
Trading strategy: A50 (rebase exit mode), with an optional signal exit that is
OFF by default.

Replaces the 5-tier strategy (tier classifier, per-tier parameters and tier
lookups are gone). One position per coin, evaluated once per closed minute,
strictly in minute order. Parameters come from artifacts/best_params.json
(written by the export cell of development_notebooks/strategy.ipynb): the run
that was best on val, used as a whole.

Authoritative reference: the simulation engine of strategy.ipynb
(`simulate_coin` / `simulate_trade`). The reference scans a whole coin at once;
this module re-implements the same logic as a forward state machine that sees
one minute at a time. Minute indices are minutes since the coin's FIRST OHLCV
minute (= the row index of the forward-filled OHLCV, as in the research data).

Per minute i the checks run in this order:

  CLOSED
    1. not a candidate                                       -> nothing
       candidate = combo_pct >= confidence_threshold
                   and price_position_15m <= max_price_position_15m
                   and price_position_1h  <= max_price_position_60m
    2. candidate still blocked by the previous trade's open re-entry scan
       (see "Re-entry scans" below)                          -> consumed, nothing
    3. i < trading_start_minutes (720)                       -> nothing
    4. after a sale: re-entry rule fails                     -> nothing
    5. BUY at the close x (1 + slippage); size capped by the price-impact cap

  OPEN
    1. re-entry scan bookkeeping. In a race phase the first candidate that
       passes the re-entry rule of the PREVIOUS sale re-anchors (rebases) the
       position: reference price = that close (no slippage), peak reset,
       phase NORMAL, shares unchanged. A rebase minute has no take-profit
       check (the reference's race stops one minute before the qualifier and
       the new segment starts one minute after it).
    2. take-profit: peak close since the segment start
       >= segment reference x (1 + activation_threshold)    -> SELL
       (disabled when se_tp_off > 0.5)
    3. signal exit (ONLY if signal_exit_enabled; OFF by default):
       minute >= ORIGINAL entry + 1 + se_min_hold and
       (dir_pct < se_dir or move_pct < se_move)              -> SELL at market
    4. phase transitions (these never sell):
         NORMAL:    close <= reference x (1 - loss_floor_pct) for
                    loss_floor_consecutive_minutes in a row -> LOSS_FLOOR_RACE;
                    otherwise minutes since the segment start >
                    stale_trade_minutes                     -> STALE_RACE.
                    Each race start opens a re-entry scan.
         LF RACE:   first close above the floor -> NORMAL (the scan stays
                    open); otherwise stale -> STALE_RACE (terminal).
         STALE:     nothing; only the take-profit, the signal exit or a rebase
                    ends it.

Re-entry scans (a reference quirk, reproduced on purpose)
---------------------------------------------------------
When a race starts, the reference scans its candidate list forward for the
first candidate that passes the re-entry rule of the PREVIOUS sale. Candidates
that fail are permanently consumed, also after the trade has closed: if the
trade is sold before that qualifier arrives, the next trade can only start at
or after it. The forward equivalent is `scan_open` during the trade and
`block_active` after it: while blocked, candidates that fail the previous
sale's rule are skipped; the first that passes lifts the block and is then
evaluated normally (with the new sale's rule).

Signal exit: off by default
---------------------------
With the signal exit off, a position ends only at the take-profit (rebase mode
never sells at the loss floor or the stale time) and this module matches the
reference exactly, minute by minute (checked trade by trade on random coins
and parameter sets; FINDINGS.md §41). The signal exit is kept behind the
`signal_exit_enabled` switch for research only. Its research results came from
a look-ahead in the engine (the trade was simulated past the signal-exit minute
and that imaginary continuation moved the candidate pointer; FINDINGS.md §35).
With the engine fixed it adds nothing, so it is off. This module is causal:
the trade ends at the signal-exit minute and only what happened up to that
minute affects later entries. It has not been checked against the fixed engine
with the signal exit on.
"""

import json
import math
from dataclasses import dataclass, field, asdict, fields

# ============================================================================
# CONSTANTS
# ============================================================================

PHASE_NORMAL = 'normal'
PHASE_STALE_RACE = 'stale_race'
PHASE_LOSS_FLOOR_RACE = 'loss_floor_race'

ACTION_NONE = 'none'
ACTION_BUY = 'buy'
ACTION_SELL = 'sell'

REASON_TAKE_PROFIT = 'take_profit'
REASON_SIGNAL_EXIT = 'signal_exit'
REASON_END_OF_DATA = 'end_of_data'

# Fixed in every research sim (not tuned): see base_config in strategy.ipynb.
TRADING_START_MINUTES = 720          # no entries before 12 h after the first minute
DEFAULT_SLIPPAGE_PCT = 0.015         # per side
DEFAULT_MAX_PRICE_IMPACT_PCT = 0.00333

SIGNAL_EXIT_ENABLED_DEFAULT = False   # see "Signal exit: off by default" above

# The tuned A50 parameters (best_params.json -> "params").
REQUIRED_PARAMS = (
    'confidence_threshold', 'max_price_position_15m', 'max_price_position_60m',
    'activation_threshold', 'stale_trade_minutes', 'loss_floor_pct',
    'loss_floor_consecutive_minutes', 'reentry_threshold_pct',
    'reentry_decay_rate', 'reentry_decay_start_minutes',
)
# Signal-exit parameters: only needed when signal_exit_enabled is true.
SIGNAL_EXIT_PARAMS = ('se_dir', 'se_move', 'se_min_hold', 'se_tp_off')
ALL_PARAMS = REQUIRED_PARAMS + SIGNAL_EXIT_PARAMS
INT_PARAMS = ('stale_trade_minutes', 'loss_floor_consecutive_minutes',
              'reentry_decay_start_minutes', 'se_min_hold')
# Fixed settings that may be overridden explicitly in the config.
FIXED_SETTINGS = {'trading_start_minutes': TRADING_START_MINUTES,
                  'exit_mode': 'rebase',
                  'signal_exit_enabled': SIGNAL_EXIT_ENABLED_DEFAULT}

# Keys of the old 5-tier config. Their presence means a stale stored config.
LEGACY_TIER_MARKERS = ('regime_tier_smoothing_minutes', 'confidence_threshold_t1',
                       'activation_threshold_t1')


# ============================================================================
# PARAMETERS
# ============================================================================

def load_best_params(path):
    """Read best_params.json (the format of strategy.ipynb's export cell:
    {"run", "val_score", "tune_score", "test_score", "params": {...}}) and
    return a validated strategy config."""
    with open(path) as f:
        raw = json.load(f)
    params = raw.get('params', raw)
    config = build_strategy_config(params)
    config['params_source'] = str(path)
    for k in ('run', 'seed', 'val_score', 'tune_score', 'test_score'):
        if k in raw:
            config[f'params_{k}'] = raw[k]
    return config


def build_strategy_config(params):
    """Validate the tuned parameters and add the fixed sim settings."""
    legacy = [k for k in params if k.endswith(('_t1', '_t2', '_t3', '_t4', '_t5'))]
    if legacy:
        raise ValueError(f"tier parameters are no longer supported: {legacy[:5]}")
    enabled = bool(params.get('signal_exit_enabled', SIGNAL_EXIT_ENABLED_DEFAULT))
    needed = REQUIRED_PARAMS + (SIGNAL_EXIT_PARAMS if enabled else ())
    missing = [k for k in needed if k not in params]
    if missing:
        raise ValueError(f"strategy parameters missing: {missing}")
    config = {}
    for k in ALL_PARAMS:
        if k in params:
            config[k] = int(params[k]) if k in INT_PARAMS else float(params[k])
    for k, default in FIXED_SETTINGS.items():
        config[k] = params.get(k, default)
    config['trading_start_minutes'] = int(config['trading_start_minutes'])
    config['signal_exit_enabled'] = bool(config['signal_exit_enabled'])
    if config['exit_mode'] != 'rebase':
        raise ValueError("only exit_mode 'rebase' is implemented (the adopted strategy)")
    for k in ('params_source', 'params_sha', 'params_run', 'params_seed', 'params_val_score',
              'params_tune_score', 'params_test_score'):
        if k in params:
            config[k] = params[k]
    return config


def is_legacy_config(config):
    return bool(config) and any(k in config for k in LEGACY_TIER_MARKERS)


# ============================================================================
# STATE
# ============================================================================

@dataclass
class StrategyState:
    """Everything the state machine needs between minutes. JSON-serialisable;
    persisted in bot_trading_state.strategy_state after every minute."""
    position: str = 'closed'
    last_idx: int = -1                  # last minute processed (guards double steps)
    coin_start: str = None              # UTC timestamp of minute 0 (index sanity check on resume)

    # ---- coin level (survives trades) ----
    last_sell_close: float = None       # close of the last sale minute
    last_sell_idx: int = None
    block_active: bool = False          # previous trade ended with an open scan
    block_ref_close: float = None       # ... whose rule is the sale BEFORE that trade
    block_ref_idx: int = None

    # ---- trade level ----
    orig_entry_idx: int = None
    orig_entry_price: float = None      # fill price of the buy (P&L reference)
    seg_entry_idx: int = None           # current segment (changes on a rebase)
    seg_entry_price: float = None       # take-profit / loss-floor reference
    max_price: float = -1.0             # peak close of the segment (-1 = none yet)
    below_floor: int = 0                # consecutive minutes at/below the floor
    phase: str = PHASE_NORMAL
    scan_open: bool = False             # a re-entry scan is looking for its qualifier
    trade_ref_close: float = None       # the sale before this trade (scan rule)
    trade_ref_idx: int = None
    n_reentries: int = 0                # rebases in this trade
    n_lf_arms: int = 0
    n_lf_recoveries: int = 0
    last_reentry_price: float = None

    def to_json(self):
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text):
        if not text:
            return cls()
        data = json.loads(text)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def copy(self):
        return StrategyState(**asdict(self))


@dataclass
class MinuteInput:
    idx: int                 # minutes since the coin's first OHLCV minute
    close: float
    combo_pct: float = float('nan')
    dir_pct: float = float('nan')
    move_pct: float = float('nan')
    pp_15m: float = float('nan')
    pp_1h: float = float('nan')


@dataclass
class StepResult:
    action: str = ACTION_NONE
    reason: str = ''
    events: list = field(default_factory=list)   # human-readable state changes


def _nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


# ============================================================================
# THE STATE MACHINE
# ============================================================================

class SignalExitStrategy:
    """A50 + signal exit. Stateless itself; all state lives in StrategyState."""

    def __init__(self, config):
        self.cfg = dict(config)
        c = self.cfg
        self.conf = float(c['confidence_threshold'])
        self.max_pp15 = float(c['max_price_position_15m'])
        self.max_pp60 = float(c['max_price_position_60m'])
        self.signal_on = bool(c.get('signal_exit_enabled', SIGNAL_EXIT_ENABLED_DEFAULT))
        self.act = float(c['activation_threshold'])
        # se_tp_off only exists inside the signal exit (as in the reference,
        # which reads it only when the signal exit is on).
        self.tp_enabled = not (self.signal_on and float(c.get('se_tp_off', 0.0)) > 0.5)
        if not self.tp_enabled:
            self.act = 100.0
        self.stale = float(c['stale_trade_minutes'])
        self.lf_pct = float(c['loss_floor_pct'])
        self.lf_consec = int(c['loss_floor_consecutive_minutes'])
        self.re_thresh = float(c['reentry_threshold_pct'])
        self.re_rate = float(c['reentry_decay_rate'])
        self.re_start = float(c['reentry_decay_start_minutes'])
        self.se_dir = float(c.get('se_dir', 0.0))
        self.se_move = float(c.get('se_move', 0.0))
        self.se_min = int(c.get('se_min_hold', 0))
        self.trading_start = int(c.get('trading_start_minutes', TRADING_START_MINUTES))

    # ---------------------------------------------------------------- rules
    def is_candidate(self, m):
        if _nan(m.combo_pct) or _nan(m.pp_15m) or _nan(m.pp_1h):
            return False
        return (m.combo_pct >= self.conf and m.pp_15m <= self.max_pp15
                and m.pp_1h <= self.max_pp60)

    def passes_reentry(self, idx, close, ref_close, ref_idx):
        """The reference's re-entry rule (relaxing after the decay start).

        Before reentry_decay_start_minutes: close <= ref x (1 - threshold).
        After it the required drop shrinks by reentry_decay_rate per minute,
        goes negative (re-entry above the sale price allowed) and below -5
        (i.e. -500%) the rule no longer applies.
        """
        if ref_close is None or ref_close <= 0:
            return True
        minutes_since_sell = float(idx - ref_idx)
        if minutes_since_sell > self.re_start:
            relaxation = (minutes_since_sell - self.re_start) * self.re_rate
            effective = self.re_thresh - relaxation
            if effective < -5.0:
                return True
            return close <= ref_close * (1.0 - effective)
        return close <= ref_close * (1.0 - self.re_thresh)

    def reentry_effective_pct(self, idx, ref_idx):
        """Required drop below the last sale at minute idx (None = no rule)."""
        if ref_idx is None:
            return None
        m = float(idx - ref_idx)
        if m > self.re_start:
            eff = self.re_thresh - (m - self.re_start) * self.re_rate
            return None if eff < -5.0 else eff
        return self.re_thresh

    def take_profit_price(self, state):
        if state.seg_entry_price is None or not self.tp_enabled:
            return None
        return state.seg_entry_price * (1.0 + self.act)

    def loss_floor_price(self, state):
        if state.seg_entry_price is None:
            return None
        return state.seg_entry_price * (1.0 - self.lf_pct)

    def signal_exit_armed_at(self, state):
        """First minute at which the signal exit may fire (None if flat)."""
        if state.orig_entry_idx is None or not self.signal_on:
            return None
        return state.orig_entry_idx + 1 + self.se_min

    def signal_fires(self, m):
        return (not _nan(m.dir_pct) and m.dir_pct < self.se_dir) or \
               (not _nan(m.move_pct) and m.move_pct < self.se_move)

    # ---------------------------------------------------------------- step
    def step(self, state, m, is_last=False):
        """Advance the state machine by one minute.

        Mutates `state` for every transition that does not trade (peak, loss
        floor counter, races, rebases, scan bookkeeping). For a trade it
        returns ACTION_BUY / ACTION_SELL WITHOUT changing the position: the
        caller executes the order and then calls open_position() /
        close_position(). If the order is not executed the state stays as it
        is and the next minute is evaluated normally.

        is_last: the final minute of a replay (end of data). The reference
        never signal-exits on that minute (its exit is the end of data, at the
        same price); passing is_last=True reproduces its exit reason.
        """
        res = StepResult()
        if m.idx <= state.last_idx:
            return res                                   # already processed
        state.last_idx = m.idx
        cand = self.is_candidate(m)

        if state.position != 'open':
            if not cand:
                return res
            if state.block_active:
                if self.passes_reentry(m.idx, m.close, state.block_ref_close, state.block_ref_idx):
                    state.block_active = False           # the scan's qualifier: block lifts
                    res.events.append('re-entry scan qualifier reached (block lifted)')
                else:
                    res.events.append("candidate consumed by the previous trade's re-entry scan")
                    return res
            if m.idx < self.trading_start:
                return res
            if state.last_sell_idx is not None and m.idx <= state.last_sell_idx:
                return res
            if not self.passes_reentry(m.idx, m.close, state.last_sell_close, state.last_sell_idx):
                return res
            res.action = ACTION_BUY
            res.reason = 'entry'
            return res

        # ------------------------------------------------------------ OPEN
        # 1) re-entry scan / rebase
        rebased = False
        if state.scan_open and cand:
            if self.passes_reentry(m.idx, m.close, state.trade_ref_close, state.trade_ref_idx):
                state.scan_open = False
                if state.phase in (PHASE_STALE_RACE, PHASE_LOSS_FLOOR_RACE):
                    old_phase = state.phase
                    state.seg_entry_idx = m.idx
                    state.seg_entry_price = m.close
                    state.max_price = -1.0
                    state.below_floor = 0
                    state.phase = PHASE_NORMAL
                    state.n_reentries += 1
                    state.last_reentry_price = m.close
                    rebased = True
                    res.events.append(f'rebase #{state.n_reentries} at the close (from {old_phase})')

        # 2) take-profit (not on a rebase minute)
        if not rebased:
            if m.close > state.max_price:
                state.max_price = m.close
            if state.max_price >= state.seg_entry_price * (1.0 + self.act):
                res.action = ACTION_SELL
                res.reason = REASON_TAKE_PROFIT
                return res

        # 3) signal exit (sell at market) — only if enabled
        if (not is_last and self.signal_on
                and m.idx >= state.orig_entry_idx + 1 + self.se_min
                and self.signal_fires(m)):
            res.action = ACTION_SELL
            res.reason = REASON_SIGNAL_EXIT
            return res

        if rebased:
            return res

        # 4) loss floor / stale: they only change the phase (rebase mode)
        floor_price = state.seg_entry_price * (1.0 - self.lf_pct)
        minutes_held = m.idx - state.seg_entry_idx
        if state.phase == PHASE_NORMAL:
            armed = False
            if m.close <= floor_price:
                state.below_floor += 1
                if state.below_floor >= self.lf_consec:
                    state.phase = PHASE_LOSS_FLOOR_RACE
                    state.scan_open = True
                    state.n_lf_arms += 1
                    armed = True
                    res.events.append(f'loss floor armed ({self.lf_consec} min at/below '
                                      f'-{self.lf_pct * 100:.2f}%): loss-floor race')
            else:
                state.below_floor = 0
            if not armed and minutes_held > self.stale:
                state.phase = PHASE_STALE_RACE
                state.scan_open = True
                res.events.append(f'stale after {minutes_held} min: stale race')
        elif state.phase == PHASE_LOSS_FLOOR_RACE:
            if m.close > floor_price:
                state.phase = PHASE_NORMAL
                state.below_floor = 0
                state.n_lf_recoveries += 1
                res.events.append('loss floor recovered: normal phase')
            elif minutes_held > self.stale:
                state.phase = PHASE_STALE_RACE
                res.events.append(f'stale after {minutes_held} min (during the loss-floor race): stale race')
        # PHASE_STALE_RACE: nothing to do here
        return res

    # ---------------------------------------------------------------- trades
    def open_position(self, state, idx, entry_price):
        """Record an executed buy at minute idx (entry_price = fill price)."""
        state.position = 'open'
        state.orig_entry_idx = int(idx)
        state.orig_entry_price = float(entry_price)
        state.seg_entry_idx = int(idx)
        state.seg_entry_price = float(entry_price)
        state.max_price = -1.0
        state.below_floor = 0
        state.phase = PHASE_NORMAL
        state.scan_open = False
        state.trade_ref_close = state.last_sell_close
        state.trade_ref_idx = state.last_sell_idx
        state.n_reentries = 0
        state.n_lf_arms = 0
        state.n_lf_recoveries = 0
        state.last_reentry_price = None
        state.last_idx = max(state.last_idx, int(idx))

    def close_position(self, state, idx, close):
        """Record an executed sale at minute idx (close = that minute's close)."""
        state.block_active = bool(state.scan_open)
        state.block_ref_close = state.trade_ref_close
        state.block_ref_idx = state.trade_ref_idx
        state.position = 'closed'
        state.last_sell_close = float(close)
        state.last_sell_idx = int(idx)
        state.orig_entry_idx = state.orig_entry_price = None
        state.seg_entry_idx = state.seg_entry_price = None
        state.max_price = -1.0
        state.below_floor = 0
        state.phase = PHASE_NORMAL
        state.scan_open = False
        state.last_idx = max(state.last_idx, int(idx))

    # ---------------------------------------------------------------- prices
    @staticmethod
    def entry_fill_price(close, slippage_pct):
        """Paper buy fill: the close plus slippage (as the reference)."""
        return close * (1.0 + slippage_pct)

    @staticmethod
    def exit_fill_price(exit_close, slippage_pct):
        """Paper sell fill: the close minus slippage (as the reference)."""
        return exit_close * (1.0 - slippage_pct)

    @staticmethod
    def buy_amount(capital, liquidity_sol_1pct, max_price_impact_pct, price_scale=1.0):
        """Size of a buy: the capital, capped so the buy stays within the
        price-impact cap. Returns (amount, cap or None). No (or non-positive)
        liquidity estimate -> no cap, exactly as the reference."""
        liq = liquidity_sol_1pct
        if liq is not None and not (isinstance(liq, float) and math.isnan(liq)) and liq > 0:
            cap = liq * (max_price_impact_pct * 100) * price_scale
            return min(capital, cap), cap
        return capital, None
