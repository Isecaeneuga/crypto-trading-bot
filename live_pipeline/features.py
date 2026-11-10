"""
Feature computation — the V9 features of Dataset_creationV9.ipynb, for the live bot.

The feature code below is the notebook's own (compute_liquidity_robust_causal,
compute_pump_set, window_features and the per-minute feature loop of
compute_coin_features_v9), with only these changes for live use:
  * features are computed for the requested rows (normally the newest minute)
    instead of START_IDX .. n - 30, and no labels are computed;
  * the hour features use UTC hours, exactly as the dataset (its loader parses
    the timestamps as UTC and never converts them);
  * the liquidity estimate can be supplied by CausalLiquidity, an incremental
    version of compute_liquidity_robust_causal with identical output, so a new
    minute does not recompute the whole coin.

Every feature at row i depends only on rows <= i (the notebook's leakage test),
so computing on the coin's full history up to now gives the dataset's values.
That requires the full history from the coin's FIRST candle: minutes since
launch, the launch features, the all-time high and the liquidity floor are all
anchored to row 0.

Public API (used by bot_loop.py and scoring.py):
  LiveFeatureEngine(ohlcv_df)        incremental liquidity state for one coin
    .sync(ohlcv_df)                  process rows appended since the last call
    .revise(ohlcv_df, first_changed) re-process from a row whose OHLCV changed
    .features(ohlcv_df, rows)        feature DataFrame (index = row positions)
    .liquidity_at(idx)               estimated_sol_1pct_impact at a row
  compute_live_features(df, rows, liquidity)   the feature function itself
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import warnings

warnings.simplefilter('ignore', pd.errors.PerformanceWarning)

# ── V9 settings (Dataset_creationV9 config cell) ─────────────────────────────
START_IDX = 120               # first row per coin with features (as the dataset)
PUMP_SETS = [{'suffix': '', 'rise': 0.10, 'window': 30},          # move model
             {'suffix': '_p15_2h', 'rise': 0.15, 'window': 120}]  # direction model
LIQ_STATS_REFRESH_MIN = 15    # liquidity floor + median/MAD clip recomputed every 15 min (causal)


# ============================================================================
# LIQUIDITY — estimated_sol_1pct_impact without look-ahead (Dataset_creationV9, verbatim)
# ============================================================================
def compute_liquidity_robust_causal(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    n: int,
    *,
    tau: float = 1440.0,        # recency half-life-ish (minutes); EW decay = exp(-1/tau)
    min_samples: int = 5,       # usable trades needed before the FIRST estimate (else NaN: coin too new)
    min_body_ret: float = 1e-4,  # ignore moves below 1 bp (no usable signal, just blows up lambda)
    max_body_ret: float = 0.50,  # winsorize per-bar move at 50% so one pump candle can't dominate
    vol_floor_q: float = 0.05,   # volume floor = this quantile of positive volume (guards micro-volume blow-up)
    mad_clip: float = 5.0,       # clip per-bar log-impact to global median +/- this * MAD
    clip_sol_1pct: tuple = (1e-6, 1e12),  # final sane clamp on the sol-for-1%% estimate
    refresh: int = 15,          # V9: minutes between causal recomputations of the floor and the clip
):
    """
    V9: IDENTICAL to Dataset_creationV8's compute_liquidity_robust except for
    the look-ahead fix. V8 computed the volume floor (a quantile of ALL the
    coin's positive volume) and the median/MAD clip (over ALL its bars) from
    the whole coin, including the future. Here both are recomputed every
    `refresh` minutes from data up to and including that minute, and used for
    the following bars until the next refresh, so bar i only depends on bars <= i.

    Robust estimator of 'SOL volume required to move price 1%' for every bar.

    Drop-in replacement for the marubozu-gated tracker. Returns arrays on the
    SAME scale as the original `liq_est` (0.01 / lambda == v / (pct*100)), so
    downstream features that divide by `liq_est` keep working unchanged.

    Design (regime-robust):
      * Uses EVERY bar with volume, not just perfect marubozus. The marubozu
        gate becomes a continuous 'directional efficiency' weight |c-o|/(h-l):
        clean directional candles still count most, dojis count ~0, but nothing
        is thrown away.
      * Per-bar Amihud impact  lambda_i = |body_ret_i| / V_i  (fractional move
        per unit SOL). sol_for_1pct = 0.01 / lambda.
      * Aggregation is an exponentially-weighted GEOMETRIC mean (EW mean of
        log lambda) -> recency-weighted, heavy-tail-safe, O(n) streaming.
      * Per-bar log-impact is winsorized to a global median +/- mad_clip*MAD
        band, and body return / volume are clamped, so a single freak candle
        cannot poison the estimate.
      * NO GUESSING. A new coin returns NaN until it has accumulated
        `min_samples` usable trades. Once warmed up, the EW estimate is
        carried forward unchanged through any data gap (forward-fill) -- it is
        never replaced by a volume prior or a default constant.
      * Staleness is reported, not hidden: `liq_conf_arr` decays during a gap
        and `liq_age_arr` counts minutes since the last usable trade, so a
        stale forward-filled value is flagged without inventing a number.

    Returns
    -------
    liq_est       : sol-for-1%% impact (same scale/meaning as before); NaN only
                    while the coin is too new to have `min_samples` usable trades
    liq_age_arr   : minutes since the last *usable* (volume+move) bar
    liq_std_arr   : EW dispersion of log-impact, in %% (uncertainty proxy)
    liq_conf_arr  : freshness in [0,1]; decays during gaps, NaN before warm-up
    """
    eps = 1e-12

    # ------------------------------------------------------------------
    # 1. Per-bar impact samples (vectorized) -- uses ALL bars with volume
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        body_ret = np.where(open_arr > eps, (close_arr - open_arr) / open_arr, 0.0)
    abs_body = np.abs(body_ret)

    rng = high_arr - low_arr
    # directional efficiency: how much of the bar's range was net direction.
    # marubozu ~ 1.0, doji ~ 0.0. This GENERALIZES the old binary gate.
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(rng > eps, np.abs(close_arr - open_arr) / rng, 0.0)
    eff = np.clip(eff, 0.0, 1.0)

    # ---- V9: CAUSAL volume floor + robust clip, refreshed every `refresh` bars ----
    # V8 (look-ahead):  vfloor = quantile(all positive volume); med/MAD over all bars
    move = np.clip(abs_body, min_body_ret, max_body_ret)
    valid = np.zeros(n, dtype=bool)
    log_lam = np.full(n, np.nan)
    pool = np.full(n, np.nan)          # raw log-impact of the valid bars so far (for med/MAD)
    n_pool = 0
    for r in range(0, n, refresh):
        # volume floor from the coin's positive volume up to and including bar r
        pv = volume_arr[:r + 1]
        pv = pv[pv > 0]
        vfloor = np.quantile(pv, vol_floor_q) if pv.size else 0.0
        if pv.size and vfloor >= np.max(pv):
            vfloor = 0.0                # degenerate guard, as in V8
        end = min(n, r + refresh)
        blk = slice(r, end)
        v_blk = (volume_arr[blk] > max(vfloor, eps)) & (abs_body[blk] >= min_body_ret) & (rng[blk] > eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam_blk = np.where(v_blk, move[blk] / volume_arr[blk], np.nan)
            ll_blk = np.log(lam_blk, where=v_blk, out=np.full(end - r, np.nan))
        v_blk &= np.isfinite(ll_blk)
        # the clip for this block uses the valid bars up to and including bar r
        if v_blk[0]:
            pool[n_pool] = ll_blk[0]
            n_pool += 1
        raw = ll_blk.copy()
        if n_pool:
            p = pool[:n_pool]
            med = np.median(p)
            mad = np.median(np.abs(p - med)) * 1.4826 + eps
            ll_blk = np.clip(ll_blk, med - mad_clip * mad, med + mad_clip * mad)
        valid[blk] = v_blk
        log_lam[blk] = np.where(v_blk, ll_blk, np.nan)
        # the rest of the block's valid bars join the pool for later blocks
        add = raw[1:][v_blk[1:]]
        pool[n_pool:n_pool + len(add)] = add
        n_pool += len(add)

    # ------------------------------------------------------------------
    # 3. EW geometric aggregation + warm-up gate + forward-fill (O(n))
    # ------------------------------------------------------------------
    log001 = np.log(0.01)
    decay = np.exp(-1.0 / tau)
    S = 0.0      # EW sum of weight*log_lam
    W = 0.0      # EW sum of weights == effective sample count
    Sq = 0.0     # EW sum of weight*log_lam^2  (for dispersion)

    liq_est = np.full(n, np.nan)
    liq_age_arr = np.full(n, np.nan)
    liq_std_arr = np.full(n, np.nan)
    liq_conf_arr = np.full(n, np.nan)
    last_valid = -1
    n_usable = 0          # raw count of usable trades seen so far
    warmed = False
    last_ll = np.nan      # last EW log-impact computed on real data (for forward-fill)
    last_std = np.nan

    for i in range(n):
        S *= decay
        W *= decay
        Sq *= decay

        fresh = bool(valid[i] and np.isfinite(log_lam[i]))
        if fresh:
            w = eff[i] + 1e-3          # efficiency weight (recency handled by decay)
            x = log_lam[i]
            S += w * x
            W += w
            Sq += w * x * x
            last_valid = i
            n_usable += 1

        if not warmed:
            # Coin too new: not enough usable trades yet -> leave NaN, don't guess.
            if n_usable >= min_samples:
                warmed = True
            else:
                continue

        if fresh:
            # Recompute the EW geometric estimate from real data...
            last_ll = S / W
            if W > eps:
                var = max(Sq / W - last_ll * last_ll, 0.0)
                last_std = (np.exp(np.sqrt(var)) - 1.0) * 100.0
        # ...otherwise carry the last real estimate forward unchanged.

        sol_1pct = np.exp(-last_ll + log001)   # 0.01 / lambda
        liq_est[i] = float(np.clip(sol_1pct, *clip_sol_1pct))
        liq_std_arr[i] = last_std
        liq_age_arr[i] = (i - last_valid) if last_valid >= 0 else np.nan
        # freshness: full when a trade just arrived, decays through a gap
        liq_conf_arr[i] = W / (W + min_samples)

    return liq_est, liq_age_arr, liq_std_arr, liq_conf_arr


class CausalLiquidity:
    """
    Incremental compute_liquidity_robust_causal: feeding rows one at a time (or in any
    chunks) gives the same four arrays as the batch function on the same data.

    It keeps the batch function's state between calls: the volume floor and the
    median/MAD clip of the current 15-minute block, the pool of valid log-impacts and the
    exponentially weighted sums. A snapshot of that state is kept at the start of each
    recent block so a revised minute can be re-processed (revise) without starting over.
    """

    def __init__(self, *, tau=1440.0, min_samples=5, min_body_ret=1e-4, max_body_ret=0.50,
                 vol_floor_q=0.05, mad_clip=5.0, clip_sol_1pct=(1e-6, 1e12),
                 refresh=LIQ_STATS_REFRESH_MIN, keep_blocks=96):
        self.tau, self.min_samples = tau, min_samples
        self.min_body_ret, self.max_body_ret = min_body_ret, max_body_ret
        self.vol_floor_q, self.mad_clip, self.clip_sol_1pct = vol_floor_q, mad_clip, clip_sol_1pct
        self.refresh, self.keep_blocks = int(refresh), int(keep_blocks)
        self.eps = 1e-12
        self.decay = np.exp(-1.0 / tau)
        self.log001 = np.log(0.01)
        self.n = 0
        self._out = np.full((4, 1024), np.nan)      # liq_est, liq_age, liq_std, liq_conf
        self._pool = np.full(1024, np.nan)           # raw log-impact of the valid bars so far
        self._snapshots = {}                         # block start -> state before that row
        self._reset_state()

    # ---- state ------------------------------------------------------------
    def _reset_state(self):
        self.n = 0
        self.n_pool = 0
        self.S = 0.0
        self.W = 0.0
        self.Sq = 0.0
        self.last_valid = -1
        self.n_usable = 0
        self.warmed = False
        self.last_ll = np.nan
        self.last_std = np.nan
        self.vfloor = 0.0
        self.clip_on = False
        self.med = np.nan
        self.mad = np.nan
        self._snapshots = {}

    _STATE = ('n_pool', 'S', 'W', 'Sq', 'last_valid', 'n_usable', 'warmed', 'last_ll', 'last_std')

    def _snapshot(self, j):
        self._snapshots[j] = {k: getattr(self, k) for k in self._STATE}
        if len(self._snapshots) > self.keep_blocks:
            for key in sorted(self._snapshots)[:-self.keep_blocks]:
                del self._snapshots[key]

    def rewind_to(self, idx):
        """Forget rows >= idx (rounded down to their block start). Returns False when that
        block's snapshot is no longer kept (the caller then rebuilds from scratch)."""
        idx = max(0, min(int(idx), self.n))
        if idx >= self.n:
            return True
        b = (idx // self.refresh) * self.refresh
        snap = self._snapshots.get(b)
        if snap is None:
            return False
        for k, v in snap.items():
            setattr(self, k, v)
        self._out[:, b:self.n] = np.nan
        self.n = b
        for key in [k for k in self._snapshots if k > b]:
            del self._snapshots[key]
        del self._snapshots[b]          # re-taken when row b is processed again
        return True

    def _grow(self, n_rows):
        if n_rows > self._out.shape[1]:
            cap = max(n_rows, 2 * self._out.shape[1])
            out = np.full((4, cap), np.nan)
            out[:, :self.n] = self._out[:, :self.n]
            self._out = out

    def _pool_add(self, x):
        if self.n_pool >= len(self._pool):
            pool = np.full(2 * len(self._pool), np.nan)
            pool[:self.n_pool] = self._pool[:self.n_pool]
            self._pool = pool
        self._pool[self.n_pool] = x
        self.n_pool += 1

    # ---- processing --------------------------------------------------------
    def update(self, open_arr, high_arr, low_arr, close_arr, volume_arr):
        """Process every row of the given (full-history) arrays not processed yet."""
        open_arr = np.asarray(open_arr, dtype=np.float64)
        high_arr = np.asarray(high_arr, dtype=np.float64)
        low_arr = np.asarray(low_arr, dtype=np.float64)
        close_arr = np.asarray(close_arr, dtype=np.float64)
        volume_arr = np.asarray(volume_arr, dtype=np.float64)
        n = len(close_arr)
        if n <= self.n:
            return
        self._grow(n)
        eps, refresh = self.eps, self.refresh
        start = self.n

        # per-bar quantities, exactly as the batch function (element-wise, so any slice
        # gives the same values as the full array)
        sl = slice(start, n)
        o, h, l, c, v = open_arr[sl], high_arr[sl], low_arr[sl], close_arr[sl], volume_arr[sl]
        with np.errstate(divide="ignore", invalid="ignore"):
            body_ret = np.where(o > eps, (c - o) / o, 0.0)
        abs_body = np.abs(body_ret)
        rng = h - l
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = np.where(rng > eps, np.abs(c - o) / rng, 0.0)
        eff = np.clip(eff, 0.0, 1.0)
        move = np.clip(abs_body, self.min_body_ret, self.max_body_ret)

        # walk the rows in pieces that never cross a block boundary
        j = start
        while j < n:
            block_start = (j // refresh) * refresh
            end = min(n, block_start + refresh)
            k0, k1 = j - start, end - start
            if j == block_start:
                self._snapshot(j)
                # volume floor from the coin's positive volume up to and including bar j
                pv = volume_arr[:j + 1]
                pv = pv[pv > 0]
                vfloor = np.quantile(pv, self.vol_floor_q) if pv.size else 0.0
                if pv.size and vfloor >= np.max(pv):
                    vfloor = 0.0
                self.vfloor = vfloor
            v_blk = (v[k0:k1] > max(self.vfloor, eps)) & (abs_body[k0:k1] >= self.min_body_ret) & (rng[k0:k1] > eps)
            with np.errstate(divide="ignore", invalid="ignore"):
                lam_blk = np.where(v_blk, move[k0:k1] / v[k0:k1], np.nan)
                ll_blk = np.log(lam_blk, where=v_blk, out=np.full(k1 - k0, np.nan))
            v_blk &= np.isfinite(ll_blk)
            raw = ll_blk.copy()
            first_add = 0
            if j == block_start:
                # the clip for this block uses the valid bars up to and including bar j
                if v_blk[0]:
                    self._pool_add(ll_blk[0])
                first_add = 1
                self.clip_on = bool(self.n_pool)
                if self.clip_on:
                    p = self._pool[:self.n_pool]
                    self.med = np.median(p)
                    self.mad = np.median(np.abs(p - self.med)) * 1.4826 + eps
            if self.clip_on:
                ll_blk = np.clip(ll_blk, self.med - self.mad_clip * self.mad, self.med + self.mad_clip * self.mad)
            log_lam = np.where(v_blk, ll_blk, np.nan)
            # the rest of the block's valid bars join the pool for later blocks
            for x in raw[first_add:][v_blk[first_add:]]:
                self._pool_add(x)

            # EW geometric aggregation + warm-up gate + forward-fill
            for kk in range(k1 - k0):
                i = j + kk
                self.S *= self.decay
                self.W *= self.decay
                self.Sq *= self.decay
                fresh = bool(v_blk[kk] and np.isfinite(log_lam[kk]))
                if fresh:
                    w = eff[k0 + kk] + 1e-3
                    x = log_lam[kk]
                    self.S += w * x
                    self.W += w
                    self.Sq += w * x * x
                    self.last_valid = i
                    self.n_usable += 1
                if not self.warmed:
                    if self.n_usable >= self.min_samples:
                        self.warmed = True
                    else:
                        continue
                if fresh:
                    self.last_ll = self.S / self.W
                    if self.W > eps:
                        var = max(self.Sq / self.W - self.last_ll * self.last_ll, 0.0)
                        self.last_std = (np.exp(np.sqrt(var)) - 1.0) * 100.0
                sol_1pct = np.exp(-self.last_ll + self.log001)
                self._out[0, i] = float(np.clip(sol_1pct, *self.clip_sol_1pct))
                self._out[1, i] = (i - self.last_valid) if self.last_valid >= 0 else np.nan
                self._out[2, i] = self.last_std
                self._out[3, i] = self.W / (self.W + self.min_samples)
            j = end
        self.n = n

    def arrays(self):
        """(liq_est, liq_age, liq_std, liq_conf) for the processed rows — the batch order."""
        o = self._out[:, :self.n]
        return o[0], o[1], o[2], o[3]


# ============================================================================
# PUMP FEATURES — once per pump definition in PUMP_SETS (Dataset_creationV9, verbatim)
# ============================================================================
PUMP_FEATURES = [
    'cooling_off_indicator', 'excess_buying_pressure_4h', 'cumulative_positive_since_last_pump',
    'buying_pressure_accumulation_rate', 'recent_pump_rate_7d', 'pump_density_1h', 'pump_density_2h',
    'pump_density_4h', 'pump_acceleration_4h', 'pump_clustering_coef', 'time_since_cluster',
    'pump_velocity_8h', 'time_weighted_pump_density', 'minutes_since_last_pump', 'last_pump_magnitude',
    'pump_failure_rate_7d', 'time_between_pumps_mean', 'time_between_pumps_std',
    'liquidity_change_since_pump', 'pump_velocity_per_liquidity', 'liquidity_weighted_pump_density',
    'pump_time_liquidity_staleness', 'cooling_off_liquidity_adjusted',
]
PUMP_WIN = 240        # was 180 (3h -> 4h)
VELOCITY_WIN = 480    # was 360 (6h -> 8h)


def compute_pump_set(close_s, close_arr, rows, ctx, rise, window):
    """ctx: per-row arrays at the rows (vol_120, vol_30, vol_mean_30 read at
    [i-1]; cur_liq, liq_age, msf) and full-length arrays (liq_est, sum_pos_240,
    cumsum_pos). Returns {feature: array over rows}."""
    n = len(close_arr)
    prev = rows - 1
    rolling_min = close_s.rolling(window=window + 1, min_periods=window + 1).min()
    gains = (close_s - rolling_min) / rolling_min
    completed = (gains >= rise).astype(int).values.copy()
    completed[:window] = 0
    pump_indices = np.flatnonzero(completed == 1).astype(np.int64)
    n_p = len(pump_indices)

    sust_defined = pump_indices < n - 10
    sustained = np.zeros(n_p, dtype=bool)
    if sust_defined.any():
        pd_idx = pump_indices[sust_defined]
        win = np.lib.stride_tricks.sliding_window_view(close_arr, 10)[pd_idx + 1]
        pc = close_arr[pd_idx][:, None]
        sustained[sust_defined] = ((win - pc) / pc >= 0.03).sum(axis=1) >= 5
    mag = np.full(n_p, np.nan)
    mag_defined = pump_indices >= window
    if mag_defined.any():
        pm = pump_indices[mag_defined]
        mins = np.lib.stride_tricks.sliding_window_view(close_arr, window + 1)[pm - window].min(axis=1)
        pcl = close_arr[pm]
        with np.errstate(divide='ignore', invalid='ignore'):
            mag[mag_defined] = np.where(mins > 1e-10, (pcl - mins) / mins, 0.0)

    ps = pd.Series(completed.astype(np.float64))
    sum_60 = ps.rolling(60, min_periods=1).sum().values
    sum_120 = ps.rolling(120, min_periods=1).sum().values
    sum_240 = ps.rolling(PUMP_WIN, min_periods=1).sum().values
    sum_480 = ps.rolling(VELOCITY_WIN, min_periods=1).sum().values
    rate_7d = ps.rolling(7 * 1440, min_periods=1).mean().values

    hi = np.searchsorted(pump_indices, rows, side='left')
    has_prev = hi > 0
    last_p = np.where(has_prev, pump_indices[np.maximum(hi - 1, 0)] if n_p else 0, -1)
    msl = np.where(has_prev, (rows - last_p).astype(np.float64), np.nan)
    if n_p:
        _m = mag[np.maximum(hi - 1, 0)]
        lpm = np.where(has_prev, np.where(np.isnan(_m), 0.0, _m), np.nan)
    else:
        lpm = np.full(len(rows), np.nan)

    vol_120, vol_30, vol_mean_30 = ctx['vol_120'], ctx['vol_30'], ctx['vol_mean_30']
    cur_liq, liq_age, msf = ctx['cur_liq'], ctx['liq_age'], ctx['msf']
    with np.errstate(divide='ignore', invalid='ignore'):
        _bl = np.where(vol_30 > 1e-10, vol_mean_30 / (vol_30 * 100), cur_liq)
    baseline_liq = np.where(np.isnan(cur_liq), np.nan, _bl)
    liq_ok = (~np.isnan(cur_liq) & ~np.isnan(baseline_liq) & (cur_liq > 1e-10) & (baseline_liq > 1e-10))

    f = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_vol = np.where(vol_120 > 1e-10, 1 / vol_120, 1.0)
    f['cooling_off_indicator'] = np.where(np.isnan(msl), 0.0, msl * inv_vol)
    f['excess_buying_pressure_4h'] = ctx['sum_pos_240'][prev] - (sum_240[prev] * rise)
    lp_safe = np.where(has_prev, last_p, 0)
    cps = ctx['cumsum_pos'][prev] - ctx['cumsum_pos'][lp_safe]
    f['cumulative_positive_since_last_pump'] = np.where(has_prev, cps, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        bpr = np.where(msl > 0, cps / msl, 0.0)
    f['buying_pressure_accumulation_rate'] = np.where(has_prev, bpr, np.nan)
    f['recent_pump_rate_7d'] = rate_7d[prev]
    f['pump_density_1h'] = sum_60[prev] / 60
    f['pump_density_2h'] = sum_120[prev] / 120
    f['pump_density_4h'] = sum_240[prev] / PUMP_WIN
    f['pump_acceleration_4h'] = f['pump_density_1h'] - f['pump_density_4h']
    f['pump_velocity_8h'] = np.where(msf >= VELOCITY_WIN, sum_480[prev] / (VELOCITY_WIN / 60), np.nan)

    lo4 = np.searchsorted(pump_indices, np.maximum(0, rows - PUMP_WIN), side='left')
    clus = np.zeros(len(rows))
    tsc = np.full(len(rows), np.nan)
    twpd = np.zeros(len(rows))
    for k in np.flatnonzero(hi - lo4 > 0):
        i = int(rows[k])
        pumps_w = pump_indices[lo4[k]:hi[k]]
        if len(pumps_w) >= 2:
            pi = np.diff(pumps_w)
            clus[k] = pi.std() / pi.mean() if pi.mean() > 1e-10 else 0.0
            lci = None
            for j in range(len(pumps_w) - 1, 0, -1):
                if pumps_w[j] - pumps_w[j - 1] <= 30:
                    lci = pumps_w[j]
                    break
            tsc[k] = i - lci if lci is not None else np.nan
        weights = np.exp(-0.01 * (i - pumps_w))
        twpd[k] = weights.sum() / PUMP_WIN
    f['pump_clustering_coef'] = clus
    f['time_since_cluster'] = tsc
    f['time_weighted_pump_density'] = twpd
    f['minutes_since_last_pump'] = msl
    f['last_pump_magnitude'] = lpm

    lo7 = np.searchsorted(pump_indices, np.maximum(0, rows - 7 * 1440), side='left')
    hi10 = np.searchsorted(pump_indices, rows - 10, side='left')
    c_valid = np.concatenate([[0], np.cumsum(sust_defined)])
    c_fail = np.concatenate([[0], np.cumsum(sust_defined & ~sustained)])
    top = np.maximum(hi10, lo7)
    valid_p = c_valid[top] - c_valid[lo7]
    failures = c_fail[top] - c_fail[lo7]
    with np.errstate(divide='ignore', invalid='ignore'):
        fr = np.where(valid_p > 0, failures / np.maximum(valid_p, 1), np.nan)
    f['pump_failure_rate_7d'] = np.where(hi - lo7 > 0, fr, np.nan)

    tbm = np.full(len(rows), np.nan)
    tbs = np.full(len(rows), np.nan)
    for k in np.flatnonzero(hi >= 2):
        last_3p = pump_indices[max(0, hi[k] - 3):hi[k]]
        intv = np.diff(last_3p)
        tbm[k] = float(intv.mean())
        tbs[k] = float(intv.std()) if len(intv) > 1 else 0.0
    f['time_between_pumps_mean'] = tbm
    f['time_between_pumps_std'] = tbs

    liq_est = ctx['liq_est']
    liq_at_pump = np.where(has_prev, liq_est[lp_safe], np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        lcp = np.where(~np.isnan(liq_at_pump) & (liq_at_pump > 1e-10), np.clip(cur_liq / liq_at_pump, 0, 10), np.nan)
        f['liquidity_change_since_pump'] = np.where(has_prev & ~np.isnan(cur_liq), lcp, np.nan)
        pv = f['pump_velocity_8h']
        f['pump_velocity_per_liquidity'] = np.where(~np.isnan(pv) & ~np.isnan(cur_liq) & (cur_liq > 1e-10),
                                                    pv / cur_liq, np.nan)
        f['liquidity_weighted_pump_density'] = np.where(liq_ok, f['time_weighted_pump_density'] * (baseline_liq / cur_liq),
                                                        np.nan)
        f['pump_time_liquidity_staleness'] = np.where(~np.isnan(msl) & ~np.isnan(liq_age), msl * np.log1p(liq_age), np.nan)
        f['cooling_off_liquidity_adjusted'] = np.where(liq_ok, f['cooling_off_indicator'] * (cur_liq / baseline_liq), np.nan)
    assert set(f) == set(PUMP_FEATURES)
    return f


# ============================================================================
# THE 22 WINDOW FEATURES (Dataset_creationV9)
# ============================================================================
def window_features(df, rows):
    """The 22 window features (Dataset_creationV9), for the given rows. Causal: rolling
    statistics use windows ending at the previous minute; returns compare the current close
    with the close w minutes earlier (the first close early on)."""
    n = len(df)
    rows = np.asarray(rows, dtype=np.int64)
    prev = rows - 1
    c = df['close'].to_numpy(np.float64)
    hi_s, lo_s, c_s = df['high'], df['low'], df['close']
    v_s = df['volume'].astype(np.float64)
    ret_s = c_s.pct_change(fill_method=None)
    W_ = {'1m': 1, '2m': 2, '10m': 10, '2h': 120, '4h': 240, '8h': 480, '12h': 720, '24h': 1440, '48h': 2880, '3d': 4320}
    f = {}
    for w in ('1m', '2m', '10m', '2h', '4h', '12h', '48h', '3d'):
        past = np.roll(c, W_[w])
        past[:W_[w]] = c[0]
        f[f'return_{w}'] = np.where(past > 1e-10, (c - past) / past, 0.0)[rows]
    for w in ('4h', '8h', '3d'):
        f[f'volatility_{w}'] = ret_s.rolling(W_[w], min_periods=W_[w]).std().to_numpy()[prev]
    cur = c[rows]
    with np.errstate(divide='ignore', invalid='ignore'):
        for w in ('4h', '24h'):
            hh = hi_s.rolling(W_[w], min_periods=1).max().to_numpy()[prev]
            ll = lo_s.rolling(W_[w], min_periods=1).min().to_numpy()[prev]
            rng = hh - ll
            rel = np.where(cur > 1e-10, rng / cur, 0.0)
            f[f'price_position_{w}'] = np.where(rel >= 0.01, np.clip((cur - ll) / rng, 0, 1), 0.0)
        for w in ('4h', '3d'):
            pk = c_s.rolling(W_[w], min_periods=1).max().to_numpy()[prev]
            f[f'drawdown_from_{w}_peak'] = np.where(pk > 1e-10, (cur - pk) / pk, 0.0)
        ath = np.maximum.accumulate(c)[prev]
        f['drawdown_from_ath'] = np.where(ath > 1e-10, (cur - ath) / ath, 0.0)
        idx = np.arange(n)
        ath_idx = np.maximum.accumulate(np.where(c >= np.maximum.accumulate(c), idx, 0))
        f['minutes_since_ath'] = (rows - ath_idx[prev]).astype(np.float64)
        for w in ('4h', '24h'):
            lo = c_s.rolling(W_[w], min_periods=1).min().to_numpy()[prev]
            f[f'drawup_from_{w}_low'] = np.where(lo > 1e-10, (cur - lo) / lo, 0.0)
        vm = {k: v_s.rolling(k, min_periods=1).mean().to_numpy()[prev] for k in (60, 240, 1440, 4320)}
        f['volume_mean_ratio_1h_24h'] = np.where(vm[1440] > 1e-10, vm[60] / vm[1440], np.nan)
        f['volume_mean_ratio_4h_3d'] = np.where(vm[4320] > 1e-10, vm[240] / vm[4320], np.nan)
        f['volume_mean_ratio_24h_3d'] = np.where(vm[4320] > 1e-10, vm[1440] / vm[4320], np.nan)
    return f, rows


WINDOW_FEATURES = ['return_1m', 'return_2m', 'return_10m', 'return_2h', 'return_4h', 'return_12h', 'return_48h',
                   'return_3d', 'volatility_4h', 'volatility_8h', 'volatility_3d', 'price_position_4h',
                   'price_position_24h', 'drawdown_from_4h_peak', 'drawdown_from_3d_peak', 'drawdown_from_ath',
                   'minutes_since_ath', 'drawup_from_4h_low', 'drawup_from_24h_low', 'volume_mean_ratio_1h_24h',
                   'volume_mean_ratio_4h_3d', 'volume_mean_ratio_24h_3d']


# ============================================================================
# ALL FEATURES FOR GIVEN ROWS (Dataset_creationV9's compute_coin_features_v9 without
# the labels and the 30-minute end cut)
# ============================================================================
def compute_live_features(df, rows=None, liquidity=None):
    """All V9 model features for the given rows of one coin's OHLCV.

    df: the coin's FULL 1-minute OHLCV from its first candle (continuous grid, zero-volume
        minutes flat at the last real close), columns timestamp/open/high/low/close/volume.
    rows: positions in df to compute (default: the last row). Rows before START_IDX are
        skipped, as in the dataset.
    liquidity: optional (liq_est, liq_age, liq_std, liq_conf) arrays covering every row of
        df, e.g. from CausalLiquidity; computed here when not given.
    Returns a DataFrame indexed by row position (empty if no row qualifies), or None if a
    required column is missing. No labels.
    """
    df = df.copy().reset_index(drop=True)
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return None

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # PRE-COMPUTE COLUMNS
    # ------------------------------------------------------------------
    df['returns'] = df['close'].pct_change(fill_method=None)

    # V9: pump detection moved to compute_pump_set (vectorised, both pump sets)

    # ==================================================================
    # EXTRACT NUMPY ARRAYS (avoids df.iloc overhead in loop)
    # ==================================================================
    n = len(df)
    close_arr = df['close'].values.astype(np.float64)
    open_arr = df['open'].values.astype(np.float64)
    high_arr = df['high'].values.astype(np.float64)
    low_arr = df['low'].values.astype(np.float64)
    volume_arr = df['volume'].values.astype(np.float64)
    returns_arr = df['returns'].values.astype(np.float64)
    timestamps = df['timestamp'].values
    # The dataset parses timestamps as UTC (load_ohlcv: utc=True, no conversion), so the hour
    # features use UTC hours whatever timezone the live data is kept in.
    ts_hours = pd.to_datetime(df['timestamp'], utc=True).dt.hour.values

    # ==================================================================
    # PRE-COMPUTE ROLLING STATISTICS AS ARRAYS
    # ==================================================================
    _ret_s = pd.Series(returns_arr)
    _vol_s = pd.Series(volume_arr)
    _close_s = pd.Series(close_arr)

    # Rolling volatilities (std of returns)
    _vol_10 = _ret_s.rolling(10, min_periods=10).std().values
    _vol_15 = _ret_s.rolling(15, min_periods=15).std().values
    _vol_30 = _ret_s.rolling(30, min_periods=30).std().values
    _vol_60 = _ret_s.rolling(60, min_periods=60).std().values
    _vol_120 = _ret_s.rolling(120, min_periods=120).std().values
    _vol_1440 = _ret_s.rolling(1440, min_periods=1440).std().values

    # vol_120 as DataFrame column (needed for volatility_trend_strength all-positions approach)
    df['vol_120'] = _vol_120

    # Rolling volume stats
    _vol_mean_30 = _vol_s.rolling(30, min_periods=30).mean().values
    _vol_mean_60 = _vol_s.rolling(60, min_periods=60).mean().values
    _vol_std_30 = _vol_s.rolling(30, min_periods=30).std().values
    _vol_sum_5 = _vol_s.rolling(5, min_periods=1).sum().values
    _vol_sum_30 = _vol_s.rolling(30, min_periods=1).sum().values
    _vol_sum_60 = _vol_s.rolling(60, min_periods=1).sum().values

    # Rolling close stats (MAs)
    _ma5 = _close_s.rolling(5, min_periods=5).mean().values
    _ma10 = _close_s.rolling(10, min_periods=10).mean().values
    _ma30 = _close_s.rolling(30, min_periods=30).mean().values
    _ma60 = _close_s.rolling(60, min_periods=60).mean().values

    # Bollinger Band components (V9: 20-bar -> 15m)
    _bb_mean = _close_s.rolling(15, min_periods=15).mean().values
    _bb_std = _close_s.rolling(15, min_periods=15).std().values

    # Previous BB (15-bar window ending 5 bars ago)
    _bb_mean_prev = _close_s.shift(5).rolling(15, min_periods=15).mean().values
    _bb_std_prev = _close_s.shift(5).rolling(15, min_periods=15).std().values

    # Rolling max/min for price positions and drawdowns
    _high_s = pd.Series(high_arr)
    _low_s = pd.Series(low_arr)
    _roll_high_15 = _high_s.rolling(15, min_periods=1).max().values
    _roll_low_15 = _low_s.rolling(15, min_periods=1).min().values
    _roll_high_30 = _high_s.rolling(30, min_periods=1).max().values
    _roll_low_30 = _low_s.rolling(30, min_periods=1).min().values
    _roll_high_60 = _high_s.rolling(60, min_periods=1).max().values
    _roll_low_60 = _low_s.rolling(60, min_periods=1).min().values
    _roll_high_120 = _high_s.rolling(120, min_periods=1).max().values
    _roll_low_120 = _low_s.rolling(120, min_periods=1).min().values

    # Peak close for drawdowns
    _peak_60 = _close_s.rolling(60, min_periods=1).max().values
    # V9 bug fix: V8 used a 120-row window for BOTH (so the '6h' and '24h' peaks were 2h peaks).
    # Now the real windows, with 6h snapped to 8h.
    _peak_480 = _close_s.rolling(480, min_periods=1).max().values
    _peak_1440 = _close_s.rolling(1440, min_periods=1).max().values

    # ==================================================================
    # PRE-COMPUTE RETURN ARRAYS (vectorized)
    # ==================================================================
    def _safe_return_arr(cur, past):
        """Vectorized safe_return: (cur - past) / past where past > 1e-10, else 0"""
        result = np.where(past > 1e-10, (cur - past) / past, 0.0)
        return result

    def _clamped_past(arr, shift):
        """Original used max(0, i-N) which clamps to index 0 for early bars."""
        past = np.roll(arr, shift)
        past[:shift] = arr[0]
        return past

    _return_3m = _safe_return_arr(close_arr, _clamped_past(close_arr, 3))
    _return_5m = _safe_return_arr(close_arr, _clamped_past(close_arr, 5))
    _return_15m = _safe_return_arr(close_arr, _clamped_past(close_arr, 15))
    _return_30m = _safe_return_arr(close_arr, _clamped_past(close_arr, 30))
    _return_60m = _safe_return_arr(close_arr, _clamped_past(close_arr, 60))
    _return_8h = _safe_return_arr(close_arr, _clamped_past(close_arr, 480))   # V9: 6h -> 8h
    _return_24h = _safe_return_arr(close_arr, _clamped_past(close_arr, 1440))

    # ==================================================================
    # PRE-COMPUTE LAUNCH FEATURES (constant per coin)
    # ==================================================================
    coin_first_ts = df['timestamp'].min()
    _minutes_since_launch = (df['timestamp'] - coin_first_ts).dt.total_seconds().values / 60.0

    # Launch features (constant values computed once). V9: 6h -> 8h; each is
    # available from its own horizon on (gates in the loop).
    _launch_vol_explosion = np.nan
    _launch_mom_1h = np.nan
    _launch_mom_8h = np.nan
    _launch_mom_24h = np.nan
    _price_stab_8h = np.nan
    _price_stab_12h = np.nan
    _price_stab_24h = np.nan

    launch_c = close_arr[0]
    if n > 540:
        fh_vol = volume_arr[0:60].mean()
        n8h_vol = volume_arr[60:540].mean()
        _launch_vol_explosion = fh_vol / n8h_vol if n8h_vol > 1e-10 else 1.0
    if n > 60:
        _launch_mom_1h = (close_arr[60] - launch_c) / launch_c if launch_c > 1e-10 else 0.0
    if n > 480:
        _launch_mom_8h = (close_arr[480] - launch_c) / launch_c if launch_c > 1e-10 else 0.0
    if n > 1440:
        _launch_mom_24h = (close_arr[min(1440, n - 1)] - close_arr[0]) / close_arr[0] if close_arr[0] > 1e-10 else 0.0

    if n >= 480:
        _price_stab_8h = np.nanstd(returns_arr[0:480], ddof=1)
    if n >= 720:
        _price_stab_12h = np.nanstd(returns_arr[0:720], ddof=1)
    if n >= 1440:
        _price_stab_24h = np.nanstd(returns_arr[0:1440], ddof=1)

    # ==================================================================
    # PRE-COMPUTE DELTA V ARRAYS
    # ==================================================================
    _gains_s = _ret_s.clip(lower=0)
    _losses_s = (-_ret_s.clip(upper=0))
    _sum_pos_ret_30 = _gains_s.rolling(30, min_periods=1).sum().values
    _sum_pos_ret_60 = _gains_s.rolling(60, min_periods=1).sum().values
    _sum_neg_ret_30 = _losses_s.rolling(30, min_periods=1).sum().values
    _sum_neg_ret_60 = _losses_s.rolling(60, min_periods=1).sum().values
    _sum_pos_ret_120 = _gains_s.rolling(120, min_periods=1).sum().values
    _sum_pos_ret_240 = _gains_s.rolling(240, min_periods=1).sum().values   # pump sets: excess buying pressure 4h
    _cumsum_pos_ret = np.cumsum(np.nan_to_num(_gains_s.values))

    # Volume-weighted positive returns
    _vol_weighted_pos = np.where(returns_arr > 0, returns_arr * volume_arr, 0.0)
    _vol_weighted_pos_sum_60 = pd.Series(_vol_weighted_pos).rolling(60, min_periods=1).sum().values

    # ==================================================================
    # PRE-COMPUTE LIQUIDITY (robust, regime-aware 1% price-impact estimator)
    # ==================================================================
    if liquidity is None:
        liquidity = compute_liquidity_robust_causal(
            open_arr, high_arr, low_arr, close_arr, volume_arr, n, refresh=LIQ_STATS_REFRESH_MIN
        )
    liq_est, liq_age_arr, liq_std_arr, liq_conf_arr = (np.asarray(a, dtype=np.float64)[:n] for a in liquidity)
    if len(liq_est) != n:
        raise ValueError(f"liquidity arrays cover {len(liq_est)} rows, the OHLCV has {n}")

    # hist_vol_dict for volatility_percentile — values at [idx] (not [idx-1])
    hist_vol_dict = {}
    for idx in range(120, n):
        if idx % 60 == 0 or idx == 120:
            v = _vol_120[idx] if idx < n else np.nan
            if not np.isnan(v):
                hist_vol_dict[idx] = v

    # ------------------------------------------------------------------
    # ROWS (live: the requested rows; the dataset starts every coin at START_IDX)
    # ------------------------------------------------------------------
    if rows is None:
        rows = [n - 1]
    rows = np.asarray([int(r) for r in np.atleast_1d(rows) if START_IDX <= int(r) < n], dtype=np.int64)
    if len(rows) == 0:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # FEATURE LOOP (optimized: reads from pre-computed arrays)
    # ------------------------------------------------------------------
    samples = []

    for i in (int(r) for r in rows):
        current_close = close_arr[i]
        current_volume = volume_arr[i]
        current_ts = pd.Timestamp(timestamps[i])
        hour = ts_hours[i]

        sample = {'timestamp': current_ts}

        features = {}
        minutes_since_first = _minutes_since_launch[i]

        # ==============================================================
        # LIFECYCLE & TEMPORAL
        # ==============================================================
        features['minutes_since_launch'] = minutes_since_first
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        # ==============================================================
        # MULTI-SCALE VOLATILITY (from pre-computed rolling arrays)
        # ==============================================================
        # rolling at index i-1 corresponds to hist_window.tail(N)
        features['volatility_10m'] = _vol_10[i - 1]
        vol_15 = _vol_15[i - 1]
        features['volatility_15m'] = vol_15
        vol_30 = _vol_30[i - 1]
        features['volatility_30m'] = vol_30
        vol_60 = _vol_60[i - 1]
        features['volatility_1h'] = vol_60
        vol_120 = _vol_120[i - 1]
        features['volatility_2h'] = vol_120
        vol_1440 = _vol_1440[i]  # original read at [i]
        features['volatility_24h'] = vol_1440

        # ==============================================================
        # VOLATILITY DYNAMICS
        # ==============================================================
        features['volatility_change_30m'] = np.clip(
            (vol_30 - vol_60) / vol_60 if vol_60 > 1e-10 else 0.0, -5, 5)

        # volatility_clustering: std of rolling 10-bar vol over last 30 bars
        recent_vols_slice = _vol_10[max(0, i - 30):i]  # rolling vol_10 values
        # The original: hist_window.tail(30)['returns'].rolling(10,5).std() then .std()
        # That computes rolling 10-bar std within the 30 bar window, then std of those.
        # _vol_10 at index j = std of returns[j-9:j+1].
        # hist_window.tail(30) = indices [i-30, i-1].
        # rolling(10,5).std() applied to returns in that window produces ~21 values.
        # Then .std() of those ~21 values.
        # We can approximate using _vol_10[i-21:i] (the rolling 10-bar vols ending within the 30-bar window)
        # But to match exactly, we need the rolling within the tail(30) slice, which differs from the full-series rolling.
        # Keep this computation as-is for exact match:
        ret_slice_30 = returns_arr[i - 30:i]
        rv = pd.Series(ret_slice_30).rolling(window=10, min_periods=5).std()
        rv_std = rv.std()
        features['volatility_clustering'] = rv_std if not np.isnan(rv_std) and rv_std > 0 else 0.0

        # vol_stability
        if i >= 60:
            vol_samples = []
            for j in range(i - 60, i, 10):
                vs = np.nanstd(returns_arr[max(0, j - 10):j], ddof=1) if j >= 10 else np.nan
                if not np.isnan(vs) and vs > 0:
                    vol_samples.append(vs)
            if len(vol_samples) >= 3:
                vm = np.mean(vol_samples)
                features['vol_stability'] = 1 / (1 + np.std(vol_samples) / vm) if vm > 1e-10 else 0.5
            else:
                features['vol_stability'] = 0.5
        else:
            features['vol_stability'] = 0.5

        # volatility_trend_strength — all positions, pre-computed vol_120, thresh>5
        lb_start_vt = max(120, i - 120)
        vol_slice = df.iloc[lb_start_vt:i]
        valid_vols = vol_slice[vol_slice['vol_120'].notna()]
        if len(valid_vols) > 5:
            vt_arr = valid_vols.index.to_numpy()
            vv_arr = valid_vols['vol_120'].to_numpy()
            if np.std(vt_arr) > 1e-10 and np.std(vv_arr) > 1e-10:
                try:
                    c = np.corrcoef(vt_arr, vv_arr)[0, 1]
                    features['volatility_trend_strength'] = 0.0 if np.isnan(c) else np.clip(c, -1, 1)
                except:
                    features['volatility_trend_strength'] = 0.0
            else:
                features['volatility_trend_strength'] = 0.0
        else:
            features['volatility_trend_strength'] = 0.0

        # ==============================================================
        # VOLATILITY CONTEXT
        # ==============================================================
        hist_volatilities = [v for idx_v, v in hist_vol_dict.items()
                             if idx_v < i and idx_v >= max(120, i - 7 * 24 * 60)]   # V9: 30d -> 7d
        cur_vol_val = np.nanstd(returns_arr[max(0, i - 120):i], ddof=1)  # inline std
        if len(hist_volatilities) > 10 and not np.isnan(cur_vol_val) and cur_vol_val > 0:
            try:
                sv = sorted(hist_volatilities)
                features['volatility_percentile'] = (np.searchsorted(sv, cur_vol_val) / len(sv)) * 100
            except:
                features['volatility_percentile'] = np.nan
        else:
            features['volatility_percentile'] = np.nan

        # intraday_vol_pattern
        if minutes_since_first >= 2880:
            hourly_vols = []
            hist_hours = ts_hours[:i]
            hist_rets = returns_arr[:i]
            for hh in range(24):
                hmask = hist_hours == hh
                hdata = hist_rets[hmask]
                if len(hdata) >= 10:
                    hv = np.nanstd(hdata, ddof=1)
                    if not np.isnan(hv):
                        hourly_vols.append((hh, hv))
            if len(hourly_vols) >= 12:
                hdict = dict(hourly_vols)
                if hour in hdict:
                    other = np.mean([v for h, v in hourly_vols if h != hour])
                    features['intraday_vol_pattern'] = hdict[hour] / other if other > 1e-10 else 1.0
                else:
                    features['intraday_vol_pattern'] = np.nan
            else:
                features['intraday_vol_pattern'] = np.nan
        else:
            features['intraday_vol_pattern'] = np.nan

        # ==============================================================
        # VOLUME FEATURES (from pre-computed arrays)
        # ==============================================================
        vol_mean_30 = _vol_mean_30[i - 1]
        vol_mean_60 = _vol_mean_60[i - 1]
        vol_std_30 = _vol_std_30[i - 1]
        volumes_30m = volume_arr[i - 30:i]

        features['volume_ratio_current'] = (
            current_volume / vol_mean_30 if vol_mean_30 > 1e-10 else 1.0)

        volume_surge_now = current_volume / vol_mean_30 if vol_mean_30 > 1e-10 else 1.0

        features['volume_spike_zscore'] = (
            (current_volume - vol_mean_30) / vol_std_30 if vol_std_30 > 1e-10 else 0.0)

        features['volume_cv_30m'] = vol_std_30 / vol_mean_30 if vol_mean_30 > 1e-10 else 0.0

        if len(volumes_30m) >= 16:
            vl5 = volumes_30m[5:]
            vl0 = volumes_30m[:-5]
            if np.std(vl5) > 1e-10 and np.std(vl0) > 1e-10:
                try:
                    c = np.corrcoef(vl5, vl0)[0, 1]
                    features['volume_autocorr_lag5'] = 0.0 if np.isnan(c) else np.clip(c, -1, 1)
                except:
                    features['volume_autocorr_lag5'] = 0.0
            else:
                features['volume_autocorr_lag5'] = 0.0
        else:
            features['volume_autocorr_lag5'] = 0.0

        features['volume_skewness_30m'] = (
            skew(volumes_30m) if len(volumes_30m) >= 10 and np.std(volumes_30m) > 1e-10 else 0.0)

        v60_sum = _vol_sum_60[i - 1]
        v5_sum = _vol_sum_5[i - 1]
        features['volume_concentration_5m'] = v5_sum / v60_sum if v60_sum > 1e-10 else 0.0

        # ==============================================================
        # VOLUME-VOLATILITY INTERACTION
        # ==============================================================
        returns_30m = returns_arr[i - 30:i]

        if len(returns_30m) >= 10:
            abs_rets = np.abs(returns_30m)
            hv_thresh = np.percentile(abs_rets, 50)
            hv_mask = abs_rets >= hv_thresh
            volume_during_high_vol = volumes_30m[hv_mask].mean() if hv_mask.sum() > 0 else vol_mean_30
        else:
            volume_during_high_vol = vol_mean_30

        features['volume_vol_product_normalized'] = (
            features['volume_ratio_current'] * vol_30)

        abs_rets_30 = np.abs(returns_30m)
        if (len(returns_30m) > 5 and
                np.std(volumes_30m) > 1e-10 and np.std(abs_rets_30) > 1e-10):
            try:
                c = np.corrcoef(volumes_30m, abs_rets_30)[0, 1]
                features['volume_price_correlation'] = 0.0 if np.isnan(c) else np.clip(c, -1, 1)
            except:
                features['volume_price_correlation'] = 0.0
        else:
            features['volume_price_correlation'] = 0.0

        # ==============================================================
        # MULTI-SCALE MOMENTUM (from pre-computed arrays)
        # ==============================================================
        features['return_3m'] = _return_3m[i]
        features['return_5m'] = _return_5m[i]
        features['return_15m'] = _return_15m[i]
        features['return_30m'] = _return_30m[i]
        features['return_1h'] = _return_60m[i]
        features['return_8h'] = _return_8h[i]
        features['return_24h'] = _return_24h[i]

        # ==============================================================
        # MOMENTUM PATTERNS
        # ==============================================================
        mom_5 = np.sign(features['return_5m'])
        mom_15 = np.sign(features['return_15m'])
        mom_30 = np.sign(features['return_30m'])
        features['momentum_alignment'] = mom_5 + mom_15 + mom_30
        features['risk_adj_momentum_30m'] = (
            features['return_30m'] / vol_30 if vol_30 > 1e-10 else 0.0)

        # ==============================================================
        # DRAWDOWN & PRICE POSITION (from pre-computed rolling max/min)
        # ==============================================================
        peak_1h = _peak_60[i - 1]
        peak_8h = _peak_480[i - 1] if i >= 480 else peak_1h
        peak_24h = _peak_1440[i - 1] if i >= 1440 else peak_8h

        features['drawdown_from_1h_peak'] = (current_close - peak_1h) / peak_1h if peak_1h > 1e-10 else 0.0
        features['drawdown_from_8h_peak'] = (
            (current_close - peak_8h) / peak_8h if peak_8h > 1e-10 else 0.0) if i >= 480 else features['drawdown_from_1h_peak']
        features['drawdown_from_24h_peak'] = (
            (current_close - peak_24h) / peak_24h if peak_24h > 1e-10 else 0.0) if i >= 1440 else features['drawdown_from_8h_peak']

        # Price positions from pre-computed rolling high/low
        for pp_name, rh, rl in [('price_position_15m', _roll_high_15, _roll_low_15),
                                 ('price_position_30m', _roll_high_30, _roll_low_30),
                                 ('price_position_1h', _roll_high_60, _roll_low_60)]:
            hi = rh[i - 1]
            lo = rl[i - 1]
            rng = hi - lo
            # range must be >= 1% of price to be meaningful (scale-invariant); else default 0
            rel_rng = rng / current_close if current_close > 1e-10 else 0.0
            features[pp_name] = np.clip((current_close - lo) / rng, 0, 1) if rel_rng >= 0.01 else 0.0

        recent_high_30m = _roll_high_30[i - 1]
        recent_low_30m = _roll_low_30[i - 1]
        features['distance_to_resistance'] = (recent_high_30m - current_close) / current_close if current_close > 1e-10 else 0.0
        features['distance_to_support'] = (current_close - recent_low_30m) / recent_low_30m if recent_low_30m > 1e-10 else 0.0

        # ==============================================================
        # MOVING AVERAGES (from pre-computed arrays)
        # ==============================================================
        ma5 = _ma5[i - 1]
        ma10 = _ma10[i - 1]
        ma30 = _ma30[i - 1]
        ma60 = _ma60[i - 1]

        features['price_to_ma_5m'] = np.clip(current_close / ma5 if ma5 > 1e-10 else 1.0, 0, 10)
        features['price_to_ma_10m'] = np.clip(current_close / ma10 if ma10 > 1e-10 else 1.0, 0, 10)
        features['price_to_ma_30m'] = np.clip(current_close / ma30 if ma30 > 1e-10 else 1.0, 0, 10)
        features['price_to_ma_1h'] = np.clip(current_close / ma60 if ma60 > 1e-10 else 1.0, 0, 10)
        features['ma_crossover_30m_1h'] = 1 if close_arr[max(0, i-30):i].mean() > close_arr[max(0, i-60):i].mean() else 0

        # ==============================================================
        # TECHNICAL INDICATORS
        # ==============================================================
        # BB from pre-computed arrays
        bb_mean_val = _bb_mean[i - 1]
        bb_std_val = _bb_std[i - 1]
        bb_upper_val = bb_mean_val + 2 * bb_std_val
        bb_lower_val = bb_mean_val - 2 * bb_std_val
        bb_range = bb_upper_val - bb_lower_val

        # bb_width is the band width relative to mean price; NaN if < 1% (flat/degenerate window)
        bb_w = bb_range / bb_mean_val if bb_mean_val > 1e-10 else np.nan
        features['bb_width_15m'] = bb_w if (np.isfinite(bb_w) and bb_w >= 0.01) else np.nan

        if i >= 20:
            bb_std_p = _bb_std_prev[i - 1]
            bb_mean_p = _bb_mean_prev[i - 1]
            bb_w_prev = (bb_std_p * 4) / bb_mean_p if (not np.isnan(bb_mean_p) and bb_mean_p > 1e-10) else np.nan
            # apply the same 1% floor to the previous band; NaN if either side undefined
            if not (np.isfinite(bb_w_prev) and bb_w_prev >= 0.01):
                bb_w_prev = np.nan
            features['bb_width_change_15m'] = features['bb_width_15m'] - bb_w_prev
        else:
            features['bb_width_change_15m'] = np.nan

        # ==============================================================
        # CANDLESTICK MICROSTRUCTURE
        # ==============================================================
        high_i = high_arr[i]
        low_i = low_arr[i]
        open_i = open_arr[i]
        close_i = close_arr[i]
        candle_range = high_i - low_i

        features['hl_ratio'] = candle_range / close_i if close_i > 1e-10 else 0.0

        hl_ranges_30 = high_arr[i - 30:i] - low_arr[i - 30:i]
        hl_closes_30 = close_arr[i - 30:i]
        valid_mask = hl_closes_30 > 1e-10
        features['hl_mean_30m'] = (
            (hl_ranges_30[valid_mask] / hl_closes_30[valid_mask]).mean() if valid_mask.sum() > 0 else 0.0)

        rc_bodies = np.abs(close_arr[i - 10:i] - open_arr[i - 10:i])
        rc_ranges = high_arr[i - 10:i] - low_arr[i - 10:i]
        vr = rc_ranges > 1e-10
        features['avg_candle_body_10m'] = (rc_bodies[vr] / rc_ranges[vr]).mean() if vr.sum() > 0 else 0.0

        rc15_open = open_arr[i - 15:i]
        rc15_close = close_arr[i - 15:i]
        if len(rc15_open) >= 2:
            gaps = np.abs(rc15_open[1:] - rc15_close[:-1])
            avg_c = rc15_close.mean()
            features['gap_frequency_15m'] = (
                (gaps > avg_c * 0.001).sum() / len(gaps) if avg_c > 1e-10 else 0.0)
        else:
            features['gap_frequency_15m'] = 0.0

        hist_range = _roll_high_120[i - 1] - _roll_low_120[i - 1]
        features['range_compression_30m_2h'] = np.clip(
            (recent_high_30m - recent_low_30m) / hist_range if hist_range > 1e-10 else 1.0, 0, 1)

        # ==============================================================
        # CONSECUTIVE PATTERNS
        # ==============================================================
        recent_rets = returns_arr[i - 10:i]
        features['consecutive_up'] = (recent_rets > 0).sum()
        features['consecutive_down'] = (recent_rets < 0).sum()

        highs_10 = high_arr[i - 10:i]
        features['consecutive_lower_highs'] = sum(
            1 for j in range(1, len(highs_10)) if highs_10[j] < highs_10[j - 1])

        # ==============================================================
        # RISK & RETURN DISTRIBUTION
        # ==============================================================
        neg_rets = returns_30m[returns_30m < 0]
        pos_rets = returns_30m[returns_30m > 0]

        features['downside_deviation_30m'] = neg_rets.std() if len(neg_rets) > 0 else 0.0
        features['upside_capture_30m'] = pos_rets.mean() if len(pos_rets) > 0 else 0.0
        features['upside_deviation_30m'] = pos_rets.std() if len(pos_rets) > 0 else 0.0

        avg_gain = pos_rets.mean() if len(pos_rets) > 0 else 0.0
        avg_loss = abs(neg_rets.mean()) if len(neg_rets) > 0 else 0.0
        features['gain_loss_ratio_30m'] = (
            avg_gain / avg_loss if avg_loss > 1e-10 else (10.0 if avg_gain > 1e-10 else 1.0))

        features['loss_tail_risk_30m'] = np.percentile(neg_rets, 5) if len(neg_rets) >= 3 else 0.0
        features['gain_tail_potential_30m'] = np.percentile(pos_rets, 95) if len(pos_rets) >= 3 else 0.0

        if len(returns_30m) >= 10:
            features['return_skewness_30m'] = skew(returns_30m)
            features['return_kurtosis_30m'] = kurtosis(returns_30m)
        else:
            features['return_skewness_30m'] = 0.0
            features['return_kurtosis_30m'] = 0.0

        features['max_single_loss_30m'] = returns_30m.min() if len(returns_30m) > 0 else 0.0
        features['max_single_gain_30m'] = returns_30m.max() if len(returns_30m) > 0 else 0.0

        v30_sum = _vol_sum_30[i - 1]
        features['price_efficiency_30m'] = abs(features['return_30m'] * current_close) / v30_sum if v30_sum > 1e-10 else 0.0

        # ==============================================================
        # COMPOSITE INDICATORS
        # ==============================================================
        features['volume_adj_return_5m'] = (
            features['return_5m'] / np.log1p(current_volume) if current_volume > 0 else 0.0)
        features['vol_norm_momentum'] = (
            features['return_30m'] / vol_120 if vol_120 > 1e-10 else 0.0)

        vol_spike = 1 if features['volume_spike_zscore'] > 2 else 0
        vol_surge = 1 if volume_surge_now > 2 else 0
        mom_align = 1 if features['momentum_alignment'] >= 2 else 0
        pump_likelihood_score = vol_spike + vol_surge + mom_align

        # (cooling_off_indicator: pump-dependent -> compute_pump_set)

        # ==============================================================
        # DELTA V FEATURES (cumulative directional returns)
        # ==============================================================
        # Core delta V: cumulative positive/negative return sums
        features['sum_positive_returns_30m'] = _sum_pos_ret_30[i - 1]
        features['sum_positive_returns_1h'] = _sum_pos_ret_60[i - 1]
        features['sum_negative_returns_30m'] = _sum_neg_ret_30[i - 1]
        features['sum_negative_returns_1h'] = _sum_neg_ret_60[i - 1]

        # (excess buying pressure, cumulative positive since last pump, accumulation rate:
        #  pump-dependent -> compute_pump_set)

        # Delta V acceleration: is buying pressure ramping up?
        # sum of positive returns in last 1h vs average hourly rate in the 2h before that
        sum_pos_1h = _sum_pos_ret_60[i - 1]
        # V9: the 2h before the last hour as a 2h window ending 1h ago (same value as V8's
        # 3h-minus-1h, without the 3h window)
        sum_pos_prev_2h = _sum_pos_ret_120[i - 61]
        features['delta_v_acceleration'] = sum_pos_1h - (sum_pos_prev_2h / 2)

        # Volume-weighted positive return sum
        features['volume_weighted_positive_return_sum_1h'] = _vol_weighted_pos_sum_60[i - 1]

        # ==============================================================
        # LAUNCH PHASE FEATURES (pre-computed constants)
        # ==============================================================
        features['launch_volume_explosion'] = _launch_vol_explosion if minutes_since_first >= 540 else np.nan
        features['launch_momentum_1h'] = _launch_mom_1h if minutes_since_first >= 60 else np.nan
        features['launch_momentum_8h'] = _launch_mom_8h if minutes_since_first >= 480 else np.nan
        features['price_stability_8h'] = _price_stab_8h if minutes_since_first >= 480 else np.nan
        features['price_stability_12h'] = _price_stab_12h if minutes_since_first >= 720 else np.nan
        features['price_stability_24h'] = _price_stab_24h if minutes_since_first >= 1440 else np.nan
        features['launch_momentum_24h'] = _launch_mom_24h if minutes_since_first >= 1440 else np.nan

        # (PUMP HISTORY: pump-dependent -> compute_pump_set, once per pump set)

        # ==============================================================
        # LIQUIDITY FEATURES
        # ==============================================================
        cur_liq = liq_est[i]
        liq_age = liq_age_arr[i]
        liq_std_pct = liq_std_arr[i]
        liq_conf = liq_conf_arr[i]

        features['estimated_sol_1pct_impact'] = cur_liq
        features['liquidity_sample_age_minutes'] = liq_age
        features['liquidity_estimate_std_pct'] = liq_std_pct
        features['liquidity_confidence'] = liq_conf

        if not np.isnan(cur_liq):
            baseline_liq = vol_mean_30 / (vol_30 * 100) if vol_30 > 1e-10 else cur_liq
        else:
            baseline_liq = np.nan

        def liq_trend_raw(lookback, min_valid):
            if i >= lookback:
                vals = liq_est[i - lookback:i]
                valid = vals[~np.isnan(vals)]
                if len(valid) >= min_valid:
                    return np.polyfit(np.arange(len(valid)), valid, 1)[0]
            return np.nan

        raw_trend_30m = liq_trend_raw(30, 10)
        raw_trend_2h = liq_trend_raw(120, 30)
        raw_trend_8h = liq_trend_raw(480, 80)   # V9: 6h -> 8h (min valid scaled 60 -> 80)
        raw_trend_24h = liq_trend_raw(1440, 240)
        raw_trend_7d = liq_trend_raw(10080, 1000)

        if not np.isnan(cur_liq) and cur_liq > 1e-10:
            features['liquidity_trend_30m'] = raw_trend_30m / cur_liq if not np.isnan(raw_trend_30m) else np.nan
            features['liquidity_trend_2h'] = raw_trend_2h / cur_liq if not np.isnan(raw_trend_2h) else np.nan
            features['liquidity_trend_8h'] = raw_trend_8h / cur_liq if not np.isnan(raw_trend_8h) else np.nan
            features['liquidity_trend_24h'] = raw_trend_24h / cur_liq if not np.isnan(raw_trend_24h) else np.nan
            features['liquidity_trend_7d'] = raw_trend_7d / cur_liq if not np.isnan(raw_trend_7d) else np.nan
        else:
            features['liquidity_trend_30m'] = np.nan
            features['liquidity_trend_2h'] = np.nan
            features['liquidity_trend_8h'] = np.nan
            features['liquidity_trend_24h'] = np.nan
            features['liquidity_trend_7d'] = np.nan

        # (liquidity_change_since_pump, pump_velocity_per_liquidity, liquidity_weighted_pump_density,
        #  pump_time_liquidity_staleness: pump-dependent -> compute_pump_set)

        features['volatility_per_liquidity'] = (
            vol_120 / cur_liq if not np.isnan(cur_liq) and cur_liq > 1e-10 and vol_120 > 0 else np.nan)

        features['liquidity_stability_vs_volatility'] = (
            liq_std_pct / (vol_30 * 100)
            if not np.isnan(liq_std_pct) and vol_30 > 1e-10 else np.nan)

        if (not np.isnan(cur_liq) and not np.isnan(baseline_liq) and
                cur_liq > 1e-10 and baseline_liq > 1e-10):
            features['liquidity_adjusted_volatility'] = vol_120 * (baseline_liq / cur_liq)
        else:
            features['liquidity_adjusted_volatility'] = np.nan

        if not np.isnan(cur_liq) and not np.isnan(liq_age):
            features['liquidity_age_adjusted'] = cur_liq * np.exp(-liq_age / 240)
        else:
            features['liquidity_age_adjusted'] = np.nan

        features['liquidity_high_uncertainty'] = int(
            (not np.isnan(liq_std_pct) and liq_std_pct > 50) or
            (not np.isnan(liq_age) and liq_age > 300))

        # (cooling_off_liquidity_adjusted: pump-dependent -> compute_pump_set)

        if not np.isnan(cur_liq) and cur_liq > 0:
            features['pump_likelihood_liquidity_capacity'] = pump_likelihood_score * np.log1p(cur_liq)
        else:
            features['pump_likelihood_liquidity_capacity'] = np.nan

        # ==============================================================
        # VOLUME-LIQUIDITY INTERACTION
        # ==============================================================
        if not np.isnan(cur_liq) and cur_liq > 1e-10:
            features['volume_to_liquidity'] = current_volume / cur_liq
            features['volume_mean_30m_to_liquidity'] = vol_mean_30 / cur_liq
            features['volume_mean_1h_to_liquidity'] = vol_mean_60 / cur_liq
            features['volume_surge_vs_liquidity'] = features['volume_ratio_current'] * (vol_mean_30 / cur_liq)
            features['volume_during_high_vol_to_liquidity'] = volume_during_high_vol / cur_liq
        else:
            features['volume_to_liquidity'] = np.nan
            features['volume_mean_30m_to_liquidity'] = np.nan
            features['volume_mean_1h_to_liquidity'] = np.nan
            features['volume_surge_vs_liquidity'] = np.nan
            features['volume_during_high_vol_to_liquidity'] = np.nan

        sample.update(features)
        samples.append(sample)

    out = pd.DataFrame(samples, index=rows)
    prev = rows - 1
    ctx = {'vol_120': _vol_120[prev], 'vol_30': _vol_30[prev], 'vol_mean_30': _vol_mean_30[prev],
           'cur_liq': liq_est[rows], 'liq_age': liq_age_arr[rows], 'msf': _minutes_since_launch[rows],
           'liq_est': liq_est, 'sum_pos_240': _sum_pos_ret_240, 'cumsum_pos': _cumsum_pos_ret}
    for ps in PUMP_SETS:
        f = compute_pump_set(df['close'], close_arr, rows, ctx, ps['rise'], ps['window'])
        for k in PUMP_FEATURES:
            out[k + ps['suffix']] = f[k]
    wf, wrows = window_features(df, rows)                             # the 22 window features
    for k in WINDOW_FEATURES:
        out[k] = wf[k]
    return out


# ============================================================================
# LIVE FEATURE ENGINE — one per coin (bot)
# ============================================================================
_OHLCV = ['open', 'high', 'low', 'close', 'volume']


class LiveFeatureEngine:
    """
    Holds the incremental liquidity state for one coin and computes the V9 features of
    any row from the coin's full OHLCV frame (kept by the caller, rows = positions from the
    coin's first candle).
    """

    def __init__(self, ohlcv_df):
        self.liq = CausalLiquidity()
        self.sync(ohlcv_df)

    def _arrays(self, ohlcv_df):
        return [ohlcv_df[c].to_numpy(dtype=np.float64) for c in _OHLCV]

    def sync(self, ohlcv_df):
        """Process the rows appended to ohlcv_df since the last call."""
        self.liq.update(*self._arrays(ohlcv_df))

    def revise(self, ohlcv_df, first_changed):
        """Rows >= first_changed have new OHLCV values: re-process from there."""
        if not self.liq.rewind_to(first_changed):
            self.liq = CausalLiquidity()
        self.liq.update(*self._arrays(ohlcv_df))

    def liquidity_at(self, idx):
        idx = int(idx)
        return float(self.liq._out[0, idx]) if 0 <= idx < self.liq.n else float('nan')

    def liquidity_state_at(self, idx):
        """(estimate, minutes since last usable bar, dispersion %, confidence) at a row."""
        idx = int(idx)
        if not 0 <= idx < self.liq.n:
            return (float('nan'),) * 4
        return tuple(float(x) for x in self.liq._out[:, idx])

    def features(self, ohlcv_df, rows=None):
        """V9 features for the given rows (default: the last row) as a DataFrame indexed by
        row position. Rows before START_IDX are left out."""
        if self.liq.n != len(ohlcv_df):
            self.sync(ohlcv_df)
        return compute_live_features(ohlcv_df, rows=rows, liquidity=self.liq.arrays())
