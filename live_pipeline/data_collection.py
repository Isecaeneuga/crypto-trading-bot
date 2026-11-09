"""
Data Collection - Handles all GeckoTerminal API interaction.

Fetches historical and live OHLCV data, handles rate limiting and retries,
and forward-fills missing minutes. No database access, no trading logic.
"""

import time
import requests
import pandas as pd
from database import log_message

BASE_URL = "https://api.geckoterminal.com/api/v2"
NETWORK = "solana"


def fetch_historical_ohlcv(pool_address, num_batches, timezone, bot_id):
    """
    Fetch historical 1-minute OHLCV data in batches, paginating backwards.
    Returns a sorted, de-duplicated DataFrame or None if no data retrieved.

    df.attrs['reached_start'] is True when the API ran out of older data, i.e.
    the frame starts at the pool's first minute. The features (minutes since
    launch, launch / ATH features, the 12 h trading start) need exactly that.
    """
    all_data = []
    current_before_timestamp = None

    log_message(f"Fetching historical data ({num_batches} batches)...", bot_id, 'INFO')

    done = False
    for i in range(num_batches):
        params = {
            "aggregate": "1",
            "limit": 1000,
            "currency": "token",
            "token": "base",
            "include_empty_intervals": "true"
        }
        if current_before_timestamp:
            params["before_timestamp"] = current_before_timestamp

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{BASE_URL}/networks/{NETWORK}/pools/{pool_address}/ohlcv/minute",
                    params=params, timeout=10
                )

                if response.status_code == 429:
                    wait_time = 30 * (attempt + 1)
                    log_message(f"Rate limited on batch {i+1}, waiting {wait_time}s (attempt {attempt+1}/{max_retries})", bot_id, 'WARNING')
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                ohlcv_list = response.json()["data"]["attributes"]["ohlcv_list"]

                if not ohlcv_list:
                    log_message(f"Batch {i+1}: No more data available", bot_id, 'INFO')
                    done = True
                    break

                all_data.extend(ohlcv_list)
                current_before_timestamp = ohlcv_list[-1][0]
                log_message(f"Batch {i+1}: Retrieved {len(ohlcv_list)} candles", bot_id, 'INFO')

                # A short batch (fewer than the page limit) means the API
                # returned everything that exists earlier than our cursor.
                # No need to make another request that would come back
                # empty — flag done and skip the inter-batch sleep so the
                # outer loop exits immediately on its next iteration.
                if len(ohlcv_list) < 1000:
                    log_message(
                        f"Batch {i+1}: Short batch ({len(ohlcv_list)} < 1000) — "
                        f"reached start of available data, no further fetches needed",
                        bot_id, 'INFO'
                    )
                    done = True
                    break

                # Sleep between batches to respect rate limits — but not after
                # the final allowed batch, since no further fetch will follow it.
                if i < num_batches - 1:
                    time.sleep(15)
                break

            # Note: a 429 is already caught by the status_code check above (which
            # sleeps and continues before raise_for_status runs), so HTTPError
            # here is only ever a non-429 error. Handled by the generic Exception
            # block below — no separate HTTPError branch needed.
            except Exception as e:
                log_message(f"Error fetching batch {i+1}: {e}", bot_id, 'ERROR')
                break
        else:
            log_message(f"Batch {i+1}: Failed after {max_retries} retries", bot_id, 'ERROR')
            break
        
        if done:
            break

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s', utc=True).dt.tz_convert(timezone)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    # Consecutive pages can overlap at their boundary candle.
    df = (df.drop_duplicates('timestamp', keep='last')
            .sort_values('timestamp').reset_index(drop=True))
    df.attrs['reached_start'] = bool(done)

    return df


def fetch_ohlcv_since(pool_address, since_unix, timezone, bot_id, max_batches=60):
    """
    Fetch every 1-minute candle from `since_unix` (inclusive) up to now, paging
    backwards with before_timestamp (one call returns at most 1000 candles, so a
    single call cannot fill a gap longer than ~16 h). Used to fill the gap after
    a restart. Returns a sorted, de-duplicated DataFrame or None.
    """
    pages = []
    before = None
    for i in range(max_batches):
        params = {"aggregate": "1", "limit": 1000, "currency": "token", "token": "base",
                  "include_empty_intervals": "true"}
        if before:
            params["before_timestamp"] = before
        ohlcv_list = None
        for attempt in range(3):
            try:
                resp = requests.get(f"{BASE_URL}/networks/{NETWORK}/pools/{pool_address}/ohlcv/minute",
                                    params=params, timeout=10)
                if resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    log_message(f"Rate limited while filling the gap, waiting {wait}s", bot_id, 'WARNING')
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                ohlcv_list = resp.json()["data"]["attributes"]["ohlcv_list"]
                break
            except Exception as e:
                log_message(f"Gap fetch error (page {i + 1}): {type(e).__name__}: {e}", bot_id, 'WARNING')
                time.sleep(2)
        if not ohlcv_list:
            break
        pages.extend(ohlcv_list)
        oldest = min(c[0] for c in ohlcv_list)
        if oldest <= since_unix or len(ohlcv_list) < 1000:
            break
        before = oldest
        time.sleep(3)
    else:
        log_message(f"Gap fetch stopped after {max_batches} pages; the gap may be incomplete",
                    bot_id, 'WARNING')
    if not pages:
        return None
    df = pd.DataFrame(pages, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df[df["timestamp"] >= since_unix]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s', utc=True).dt.tz_convert(timezone)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return (df.drop_duplicates('timestamp', keep='last')
              .sort_values('timestamp').reset_index(drop=True))


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


def forward_fill_missing_minutes(df, timezone, bot_id, log=True, anchor_first_row=False):
    """
    Put the candles on a gap-free 1-minute grid with the research fill rule, the
    same as meme_coin_pipeline.ipynb and data_cleaning.ipynb (FINDINGS.md §0.5):
    EVERY zero-volume minute, whether the API skipped it or returned it as an
    empty interval, is a flat candle at the last REAL close (open = high = low =
    close), volume 0. The features rely on one row per minute (row index ==
    minutes since the first minute) and on exactly these flat candles.

    anchor_first_row: the first row is the last minute already stored, which is
    already on this rule, so its close is the last real close for the
    zero-volume minutes that follow it. Used when new minutes are appended to
    the history (bot_loop._append_minutes).
    """
    if len(df) == 0:
        return df
    df = df.drop_duplicates('timestamp', keep='last').sort_values('timestamp').reset_index(drop=True)

    if log:
        log_message("Forward filling missing minutes...", bot_id, 'INFO')

    complete_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(),
                                   freq='1min', tz=timezone)
    merged_df = pd.DataFrame({'timestamp': complete_range}).merge(df, on='timestamp', how='left')

    merged_df['volume'] = merged_df['volume'].fillna(0.0)
    zero = (merged_df['volume'] <= 0).to_numpy()
    real_close = merged_df['close'].where(~zero)              # closes of real candles only
    if anchor_first_row:
        real_close.iloc[0] = merged_df['close'].iloc[0]
    last_close = real_close.ffill()                           # the last REAL close
    last_close = last_close.fillna(merged_df['close'].bfill())  # zero-volume minutes before any real candle
    for col in ('open', 'high', 'low', 'close'):
        merged_df.loc[zero, col] = last_close[zero]

    filled_count = len(merged_df) - len(df)
    if log or filled_count:
        log_message(f"Filled {filled_count} missing minutes", bot_id, 'INFO')

    return merged_df.reset_index(drop=True)
