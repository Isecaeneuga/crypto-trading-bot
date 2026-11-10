"""
Trading Strategy — the A50 strategy of strategy.ipynb (rebase exits).

Buy score: combo = P(move) x P(direction) from the two XGBoost models (scoring.py).
The research gate "combo, as a percentile of its tune distribution, >= confidence_threshold"
is exactly "raw combo >= combo_cutoff", so the live bot only needs combo_cutoff.

The parameters come from strategy files, strategy_<name>.json, written by the export
cell of strategy.ipynb into its live_artifacts/ folder and copied into the model folder
(global setting model_dir, default artifacts/) next to move.ubj and direction.ubj. Each
bot is set up with one of them (bot_config.strategy_name); without a name, the first
strategy file in the folder (by file name) is used. use_strategy() loads one into
TUNED_PARAMS; the decision functions below use that dict by default. The file also carries fingerprints of the models it was tuned on, which
scoring.verify_models() checks against move.ubj and direction.ubj.

Open trades pass through three phases, exactly as the notebook's simulator:
    NORMAL  ->  STALE_RACE        (terminal: only activation or a re-entry ends it)
            ->  LOSS_FLOOR_RACE   (revertible: recovery returns to NORMAL, or stale
                                   fires and moves the trade to STALE_RACE)
Within each phase the priority is:  activation  >  loss floor  >  stale.
During a race, a qualifying entry candidate re-bases the trade (new reference
price = that minute's close, no new buy) and takes priority over activation in
that minute; bot_loop.py checks it before calling step_open_position().

Flow (managed by bot_loop.py):
  1. passes_entry_filters()      — does this minute qualify as an entry candidate?
  2. passes_reentry_threshold()  — has the price dropped enough since the last sell?
  3. step_open_position()        — single-minute state-machine step
"""

import json
from pathlib import Path


# ============================================================================
# PARAMETERS — loaded from a strategy file
# ============================================================================

STRATEGY_FORMAT = 'bot-strategy/1'
STRATEGY_PARAM_KEYS = ('confidence_threshold', 'max_price_position_15m', 'max_price_position_60m',
                       'activation_threshold', 'stale_trade_minutes', 'reentry_threshold_pct',
                       'loss_floor_pct', 'reentry_decay_rate', 'reentry_decay_start_minutes',
                       'loss_floor_consecutive_minutes')

# Filled in place by use_strategy(); the functions' default `params` is this dict.
TUNED_PARAMS = {}
# The loaded strategy file (name, selection, models, simulation settings).
STRATEGY = {}

# Defaults for the settings the bot takes from the wallet / bot configuration; the
# strategy file's simulation settings replace them when it is loaded.
FIXED_PARAMS = {
    'slippage_pct': 0.015,
    'desired_price_impact_pct': 0.00333,
    'target_fill_pct': 1,
    'exit_mode': 'rebase',
}


class StrategyError(ValueError):
    pass


def strategy_files(model_dir):
    """The strategy files in a model folder, by name, in file-name order (the first is the default)."""
    return {p.stem[len('strategy_'):]: p for p in sorted(Path(model_dir).glob('strategy_*.json'))}


def read_strategy(path):
    """Read and validate one strategy file."""
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise StrategyError(f'cannot read {path}: {e}') from e
    if doc.get('format') != STRATEGY_FORMAT:
        raise StrategyError(f"{path}: unknown format {doc.get('format')!r} (expected {STRATEGY_FORMAT!r})")
    missing = [k for k in STRATEGY_PARAM_KEYS if k not in doc.get('params', {})]
    if missing:
        raise StrategyError(f'{path}: missing parameters {missing}')
    if not isinstance(doc.get('combo_cutoff'), (int, float)):
        raise StrategyError(f'{path}: missing combo_cutoff')
    for m in ('move', 'direction'):
        if not doc.get('models', {}).get(m, {}).get('sha256'):
            raise StrategyError(f'{path}: missing the {m} model fingerprint')
    return doc


def strategy_path(model_dir, name=None):
    """The file of strategy `name` in model_dir; name=None: the first strategy file there."""
    files = strategy_files(model_dir)
    if not files:
        raise StrategyError(f'no strategy_<name>.json in {model_dir}/: copy the files of strategy.ipynb\'s '
                            f'live_artifacts/ folder there')
    if name:
        if name not in files:
            raise StrategyError(f"no strategy_{name}.json in {model_dir}/ (found: {', '.join(files)})")
        return files[name]
    return next(iter(files.values()))


_PARAMS_CACHE = {}


def strategy_params(model_dir, name=None):
    """(name, parameters incl. combo_cutoff) of a strategy file, without loading it into
    TUNED_PARAMS; for the web app, which shows bots with different strategies. Cached until
    the file changes."""
    path = strategy_path(model_dir, name)
    key = (str(path), path.stat().st_mtime)
    if key not in _PARAMS_CACHE:
        doc = read_strategy(path)
        _PARAMS_CACHE[key] = (doc['name'], dict(doc['params'], combo_cutoff=float(doc['combo_cutoff'])))
    return _PARAMS_CACHE[key]


def use_strategy(model_dir, name=None):
    """
    Load a strategy file from model_dir into TUNED_PARAMS / STRATEGY / FIXED_PARAMS and
    return it. name=None: the first strategy file in the folder.
    """
    path = strategy_path(model_dir, name)
    doc = read_strategy(path)
    TUNED_PARAMS.clear()
    TUNED_PARAMS.update(doc['params'])
    TUNED_PARAMS['combo_cutoff'] = float(doc['combo_cutoff'])
    sim = doc.get('simulation', {})
    for k in FIXED_PARAMS:
        if k in sim:
            FIXED_PARAMS[k] = sim[k]
    STRATEGY.clear()
    STRATEGY.update(doc)
    return doc


# ============================================================================
# DECISIONS
# ============================================================================

PHASE_NORMAL          = 'normal'
PHASE_STALE_RACE      = 'stale_race'
PHASE_LOSS_FLOOR_RACE = 'loss_floor_race'

# The model feature behind the strategy's "60m" price position.
PRICE_POSITION_60M_FEATURE = 'price_position_1h'


# ============================================================================
# ENTRY FILTERS
# ============================================================================

def passes_entry_filters(combo, features, params=TUNED_PARAMS):
    """
    Entry candidate (the notebook's find_entry_candidates):
        combo >= combo_cutoff
        price_position_15m <= max_price_position_15m
        price_position_1h  <= max_price_position_60m
    A missing price position counts as 1.0, as in the notebook's merge.
    """
    if combo is None or not (combo >= params['combo_cutoff']):
        return False
    pp15 = features.get('price_position_15m', 1.0)
    pp60 = features.get(PRICE_POSITION_60M_FEATURE, 1.0)
    pp15 = 1.0 if pp15 is None or pp15 != pp15 else pp15
    pp60 = 1.0 if pp60 is None or pp60 != pp60 else pp60
    if pp15 > params['max_price_position_15m']:
        return False
    if pp60 > params['max_price_position_60m']:
        return False
    return True


def effective_reentry_threshold(minutes_since_sell, params=TUNED_PARAMS):
    """
    The required drop below the last sell close after `minutes_since_sell` minutes.
    After reentry_decay_start_minutes it RELAXES by reentry_decay_rate per minute (it can go
    negative: re-entry is then allowed above the last sell close). Returns None once it is
    below -5, where the notebook allows any re-entry.
    """
    re_thresh = params['reentry_threshold_pct']
    decay_start = float(params['reentry_decay_start_minutes'])
    if minutes_since_sell > decay_start:
        relaxation = (minutes_since_sell - decay_start) * params['reentry_decay_rate']
        effective = re_thresh - relaxation
        if effective < -5.0:
            return None
        return effective
    return re_thresh


def passes_reentry_threshold(current_close, current_idx, last_sell_close, last_sell_idx,
                             params=TUNED_PARAMS):
    """
    Has the price dropped enough since the last sell (the notebook's passes_reentry_threshold)?
        current_close <= last_sell_close * (1 - effective threshold)
    No previous sell -> True.
    """
    if last_sell_close is None or last_sell_close <= 0 or last_sell_idx is None:
        return True
    effective = effective_reentry_threshold(float(current_idx - last_sell_idx), params)
    if effective is None:
        return True
    return current_close <= last_sell_close * (1.0 - effective)


# ============================================================================
# EXIT / STATE MACHINE — single-minute step
# ============================================================================

def step_open_position(current_close, entry_price, minutes_held,
                       max_price_seen, below_floor_count, trade_phase,
                       params=TUNED_PARAMS):
    """
    Single-minute state-machine step for an open position (the notebook's simulate_trade,
    one minute at a time). Priority within a minute: activation > loss floor > stale.

    Inputs:
        current_close: this minute's close
        entry_price: the CURRENT segment's reference price (the buy fill, or the close of the
                     last re-base); activation and loss floor are measured from it
        minutes_held: minutes since the current segment started
        max_price_seen: running peak close since the segment started (kept through a loss-
                        floor recovery, reset on a re-base)
        below_floor_count: consecutive minutes at/below the loss floor (NORMAL phase only)
        trade_phase: PHASE_NORMAL | PHASE_STALE_RACE | PHASE_LOSS_FLOOR_RACE

    Returns dict with keys:
        action: 'hold' | 'sell' | 'enter_stale_race' | 'enter_loss_floor_race'
                | 'recover_loss_floor'
        reason, max_price_seen, below_floor_count,
        effective_activation_pct (telemetry: the activation threshold)
    """
    act_thresh = params['activation_threshold']
    stale_minutes = params['stale_trade_minutes']
    floor_pct = params['loss_floor_pct']
    n_consec_required = int(params['loss_floor_consecutive_minutes'])

    result = {
        'action': 'hold',
        'reason': '',
        'max_price_seen': max_price_seen,
        'below_floor_count': below_floor_count,
        'effective_activation_pct': act_thresh,
    }

    # ---- Update peak ----
    if max_price_seen is None or current_close > max_price_seen:
        max_price_seen = current_close
        result['max_price_seen'] = max_price_seen

    # ---- 1) ACTIVATION (highest priority, all phases) ----
    if max_price_seen >= entry_price * (1.0 + act_thresh):
        result['action'] = 'sell'
        result['reason'] = f'Activation reached (peak +{act_thresh * 100:.2f}%)'
        return result

    # ---- 2) Phase-specific transitions ----
    floor_price = entry_price * (1.0 - floor_pct)

    if trade_phase == PHASE_NORMAL:
        # 2a) Loss-floor arm-watch
        if current_close <= floor_price:
            below_floor_count += 1
            result['below_floor_count'] = below_floor_count
            if below_floor_count >= n_consec_required:
                result['action'] = 'enter_loss_floor_race'
                result['reason'] = (
                    f'Loss floor armed: {n_consec_required} consecutive minutes '
                    f'at/below -{floor_pct * 100:.2f}%'
                )
                return result
        else:
            below_floor_count = 0
            result['below_floor_count'] = 0

        # 2b) Stale (lowest priority)
        if minutes_held > stale_minutes:
            result['action'] = 'enter_stale_race'
            result['reason'] = f'Trade stale after {minutes_held} minutes'
            return result

    elif trade_phase == PHASE_LOSS_FLOOR_RACE:
        # Recovery-watch: the first minute above the floor disarms back to NORMAL
        if current_close > floor_price:
            result['action'] = 'recover_loss_floor'
            result['reason'] = (
                f'Loss floor recovered (close > entry * (1 - {floor_pct * 100:.2f}%))'
            )
            result['below_floor_count'] = 0
            return result
        # Stale still ticks during the loss-floor race -> terminal stale race
        if minutes_held > stale_minutes:
            result['action'] = 'enter_stale_race'
            result['reason'] = f'Trade stale after {minutes_held} minutes (during LF race)'
            return result

    # PHASE_STALE_RACE: only activation (above) or a re-entry (bot_loop) ends it.
    return result


# ============================================================================
# EXIT PRICE
# ============================================================================

def compute_exit_price(current_close, slippage_pct):
    """
    The notebook's exit fill with target_fill_pct = 1: the exit minute's close less
    slippage,  exit_price = current_close * (1 - slippage_pct).
    """
    return current_close * (1.0 - slippage_pct)
