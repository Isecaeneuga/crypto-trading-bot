"""
Data Collection - Handles all GeckoTerminal API interaction.

Fetches the full available OHLCV history and live candles, handles rate limiting
and retries, and fills missing / zero-volume minutes exactly as the data notebooks
(meme_coin_pipeline, data_cleaning). No database access, no trading logic.
"""

import time
import requests
import pandas as pd
from database import log_message

BASE_URL = "https://api.geckoterminal.com/api/v2"
NETWORK = "solana"


def _ohlcv_list_to_df(ohlcv_list, timezone):
    """Raw GeckoTerminal ohlcv_list -> sorted, de-duplicated DataFrame (the pipeline's _raw_to_df)."""
    df = pd.DataFrame(ohlcv_list, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s', utc=True).dt.tz_convert(timezone)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df.drop_duplicates(subset=["timestamp"]).sort_values('timestamp').reset_index(drop=True)


def _fetch_batch_before(pool_address, before_timestamp, bot_id, label, limit=1000, max_retries=5):
    """
    One backward page: up to `limit` 1-minute candles before `before_timestamp` (None = now).
    Retries rate limits and errors; returns the ohlcv_list ([] = no older data) or None if
    every attempt failed.
    """
    params = {
        "aggregate": "1",
        "limit": int(limit),
        "currency": "token",
        "token": "base",
        "include_empty_intervals": "true"
    }
    if before_timestamp is not None:
        params["before_timestamp"] = int(before_timestamp)

    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"{BASE_URL}/networks/{NETWORK}/pools/{pool_address}/ohlcv/minute",
                params=params, timeout=15
            )
            if response.status_code == 429:
                wait_time = 30 * (attempt + 1)
                log_message(f"Rate limited on {label}, waiting {wait_time}s "
                            f"(attempt {attempt + 1}/{max_retries})", bot_id, 'WARNING')
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            return response.json()["data"]["attributes"]["ohlcv_list"]
        except Exception as e:
            wait_time = 10 * (attempt + 1)
            log_message(f"Error fetching {label} (attempt {attempt + 1}/{max_retries}): "
                        f"{type(e).__name__}: {e} — retrying in {wait_time}s", bot_id, 'WARNING')
            time.sleep(wait_time)
    return None


def fetch_historical_ohlcv(pool_address, timezone, bot_id, batch_delay=15):
    """
    Fetch ALL available 1-minute OHLCV history, paging backwards from now until the API
    has nothing older (the coin's first candle). The V9 features are anchored to the coin's
    first candle, so a partial history is treated as a failure: returns the sorted DataFrame,
    or None if a page could not be fetched.
    """
    all_data = []
    before_ts = None
    low_count_streak = 0
    batch_num = 0

    log_message("Fetching the full available history...", bot_id, 'INFO')
    while True:
        batch_num += 1
        ohlcv_list = _fetch_batch_before(pool_address, before_ts, bot_id, f"history batch {batch_num}")
        if ohlcv_list is None:
            log_message(f"History batch {batch_num} failed — the full history could not be fetched",
                        bot_id, 'ERROR')
            return None
        if not ohlcv_list:
            log_message(f"Batch {batch_num}: no older data — reached the first candle", bot_id, 'INFO')
            break

        oldest = min(row[0] for row in ohlcv_list)
        if before_ts is not None and oldest >= before_ts:
            log_message(f"Batch {batch_num}: no older candles returned — stopping", bot_id, 'INFO')
            break
        all_data.extend(ohlcv_list)
        before_ts = oldest
        log_message(f"Batch {batch_num}: {len(ohlcv_list)} candles (total {len(all_data)})", bot_id, 'INFO')

        # The collection pipeline's stop rule: three nearly empty pages in a row = the start.
        if len(ohlcv_list) <= 5:
            low_count_streak += 1
            if low_count_streak >= 3:
                log_message("Stopping: 3 consecutive batches with <= 5 candles", bot_id, 'INFO')
                break
        else:
            low_count_streak = 0
        time.sleep(batch_delay)

    if not all_data:
        return None
    return _ohlcv_list_to_df(all_data, timezone)


def fetch_ohlcv_since(pool_address, since_ts, timezone, bot_id, batch_delay=15):
    """
    Fetch every candle newer than `since_ts` (a tz-aware timestamp), paging backwards from now
    until the pages reach back to it. Used to close the gap after a restart, however long the
    bot was down. Returns a DataFrame (possibly empty) or None if a page failed.
    """
    since_unix = int(pd.Timestamp(since_ts).timestamp())
    all_data = []
    before_ts = None
    batch_num = 0
    while True:
        batch_num += 1
        ohlcv_list = _fetch_batch_before(pool_address, before_ts, bot_id, f"gap batch {batch_num}")
        if ohlcv_list is None:
            return None
        if not ohlcv_list:
            break
        oldest = min(row[0] for row in ohlcv_list)
        if before_ts is not None and oldest >= before_ts:
            break
        all_data.extend(ohlcv_list)
        before_ts = oldest
        if oldest <= since_unix:
            break
        time.sleep(batch_delay)

    if not all_data:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = _ohlcv_list_to_df(all_data, timezone)
    return df[df["timestamp"] > pd.Timestamp(since_ts)].reset_index(drop=True)


def fetch_ohlcv_from_timestamp(pool_address, after_timestamp, timezone, bot_id, debug=False):
    """
    Fetch recent OHLCV data after a given unix timestamp.
    Used for live data updates. Returns DataFrame or None.
    """
    params = {
        "aggregate": "1",
        "limit": 1000,
        "currency": "token",
        "token": "base",
        "include_empty_intervals": "true",
        "after_timestamp": after_timestamp
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"{BASE_URL}/networks/{NETWORK}/pools/{pool_address}/ohlcv/minute",
                params=params, timeout=10
            )

            if response.status_code == 429:
                wait_time = 30 * (attempt + 1)
                log_message(f"Rate limited during live fetch, waiting {wait_time}s", bot_id, 'WARNING')
                time.sleep(wait_time)
                continue

            # Under per-bot debug, surface any non-OK status (404/5xx/etc.) that
            # would otherwise be swallowed by the except below. 429 is already
            # reported above and stays on by default.
            if debug and response.status_code != 200:
                log_message(f"Live fetch HTTP {response.status_code} from GeckoTerminal", bot_id, 'WARNING')

            response.raise_for_status()
            ohlcv_list = response.json()["data"]["attributes"]["ohlcv_list"]

            if not ohlcv_list:
                return None

            df = pd.DataFrame(ohlcv_list, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s', utc=True).dt.tz_convert(timezone)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            return df

        # Note: a 429 is already caught by the status_code check above, so any
        # HTTPError here is a non-429 error — handled by the generic block below.
        except Exception as e:
            # Under per-bot debug, surface the specific error (timeout,
            # connection error, non-429 HTTP error) instead of failing silently.
            if debug:
                log_message(f"Live fetch error: {type(e).__name__}: {e}", bot_id, 'WARNING')
            return None

    log_message(f"Live fetch failed after {max_retries} retries", bot_id, 'WARNING')
    return None


def resolve_token_mint_and_decimals(pool_address, bot_id):
    """
    Resolve the pool's BASE token mint and its decimals from GeckoTerminal.

    The bot prices the base token (OHLCV is fetched with token=base), so the
    base token is the one shadow/live must quote on Jupiter. Returns
    (mint, decimals) or (None, None) on failure.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            pool_resp = requests.get(
                f"{BASE_URL}/networks/{NETWORK}/pools/{pool_address}", timeout=10
            )
            if pool_resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            pool_resp.raise_for_status()

            base_id = (pool_resp.json()['data']['relationships']
                       ['base_token']['data']['id'])
            # GeckoTerminal ids look like "solana_<mint>"
            mint = base_id.split('_', 1)[1] if '_' in base_id else base_id

            tok_resp = requests.get(
                f"{BASE_URL}/networks/{NETWORK}/tokens/{mint}", timeout=10
            )
            if tok_resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            tok_resp.raise_for_status()

            decimals = int(tok_resp.json()['data']['attributes']['decimals'])
            log_message(
                f"Resolved base token mint {mint} ({decimals} decimals)",
                bot_id, 'INFO'
            )
            return mint, decimals

        except Exception as e:
            log_message(
                f"Error resolving token mint/decimals (attempt {attempt + 1}/"
                f"{max_retries}): {type(e).__name__}: {e}", bot_id, 'WARNING'
            )
            if attempt == max_retries - 1:
                return None, None
            time.sleep(2)

    return None, None


def backfill_missing_candles(df, prev_last_ts=None, prev_real_close=None):
    """
    The notebooks' gap fill (meme_coin_pipeline / data_cleaning backfill_missing_candles):
    put the series on a continuous 1-minute grid and make EVERY zero-volume minute a FLAT
    candle at the last real close: open = high = low = close, volume 0. That covers minutes
    the API skipped and the API's own empty intervals. Adds 'backfilled' (1 = zero-volume
    minute, 0 = real candle), derived from the volume.

    Without the optional arguments this is the notebook function on a whole series. To
    extend an existing (already filled) series with newer candles, pass its last timestamp
    as prev_last_ts and its last real close as prev_real_close: the grid then starts one
    minute after prev_last_ts and zero-volume minutes before the first real candle in `df`
    take prev_real_close — the same result as filling the whole joined series at once.
    """
    if df is None or len(df) == 0:
        return df

    df = df.drop(columns=["backfilled"], errors="ignore")
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if prev_last_ts is not None:
        df = df[df["timestamp"] > prev_last_ts].reset_index(drop=True)
        if len(df) == 0:
            return df.assign(backfilled=pd.Series(dtype=int))
        start = pd.Timestamp(prev_last_ts) + pd.Timedelta(minutes=1)
    else:
        start = df["timestamp"].min()

    full_range = pd.date_range(start=start, end=df["timestamp"].max(), freq="1min")
    df_filled = df.set_index("timestamp").reindex(full_range)
    df_filled["volume"] = df_filled["volume"].fillna(0)
    zero = (df_filled["volume"] <= 0).to_numpy()
    last_close = df_filled["close"].where(~zero).ffill()          # last REAL close
    if prev_real_close is not None:
        last_close = last_close.fillna(float(prev_real_close))
    last_close = last_close.fillna(df_filled["close"].bfill())    # zero-volume minutes before any real candle
    for col in ("open", "high", "low", "close"):
        df_filled.loc[zero, col] = last_close[zero]
    df_filled["backfilled"] = zero.astype(int)
    df_filled = df_filled.reset_index().rename(columns={"index": "timestamp"})
    return df_filled
