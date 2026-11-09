"""
DEX interaction (Jupiter) for shadow and (future) live trading.

Shadow mode uses this to QUOTE a real swap at decision time — the same
routing/pricing a live trade would get — without ever signing or submitting.
It returns the expected output amount (which already reflects AMM + platform
fees and price impact), so the recorded fill is realistic. Only live mode will
ever need the private key; nothing here signs anything.

This module talks only to Jupiter. GeckoTerminal (OHLCV, token metadata) stays
in data_collection.py; fill economics stay in execution.py.
"""

import time
import requests

# Wrapped SOL / native SOL mint used by Jupiter for SOL swaps.
SOL_MINT = "So11111111111111111111111111111111111111112"

# Free public Jupiter endpoints. Swap to the keyed host ("https://api.jup.ag/
# swap/v1/...") with an x-api-key header if you hit rate limits.
JUPITER_QUOTE_URL = "https://lite-api.jup.ag/swap/v1/quote"
JUPITER_SWAP_URL = "https://lite-api.jup.ag/swap/v1/swap"

# Solana base fee: 5000 lamports per signature. A Jupiter swap the user signs
# is one signature, so this is the fixed floor on top of the priority fee.
BASE_SIGNATURE_FEE_SOL = 0.000005

# Fallback network-fee estimate (SOL) used only when the exact /swap-based
# estimate is unavailable (no public key, or /swap failed).
ESTIMATED_NETWORK_FEE_SOL = 0.00005

# Default quote slippage tolerance (basis points). For shadow this only affects
# the (unused) otherAmountThreshold, not the recorded outAmount.
DEFAULT_SLIPPAGE_BPS = 150


def _log(msg, bot_id, level):
    """Best-effort logging that doesn't hard-depend on the DB layer."""
    try:
        from database import log_message
        log_message(msg, bot_id, level)
    except Exception:
        pass


def get_swap_quote(input_mint, output_mint, amount_smallest,
                   slippage_bps=DEFAULT_SLIPPAGE_BPS, bot_id=None):
    """
    Get an ExactIn swap quote from Jupiter.

    Args:
        input_mint / output_mint: token mint addresses (SOL_MINT for SOL)
        amount_smallest: input amount in the input token's smallest units
                         (lamports for SOL)
        slippage_bps: tolerance in basis points (quote-only; not submitted)
        bot_id: for logging

    Returns dict (all integer amounts in smallest units) or None on failure:
        - out_amount:       expected output (already net of AMM + platform fees)
        - in_amount:        echoed input amount
        - price_impact_pct: price impact as a fraction (e.g. 0.0123 = 1.23%)
        - platform_fee:     explicit platform fee in output smallest units (0 if none)
    """
    if amount_smallest <= 0:
        return None

    params = {
        'inputMint': input_mint,
        'outputMint': output_mint,
        'amount': int(amount_smallest),
        'slippageBps': int(slippage_bps),
        'swapMode': 'ExactIn',
        # Route only through liquid intermediates so quotes are stable/realistic.
        'restrictIntermediateTokens': 'true',
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(JUPITER_QUOTE_URL, params=params, timeout=10)

            if resp.status_code == 429:
                wait = 2 * (attempt + 1)
                _log(f"Jupiter rate limited, waiting {wait}s "
                     f"(attempt {attempt + 1}/{max_retries})", bot_id, 'WARNING')
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            out_amount = int(data['outAmount'])
            in_amount = int(data.get('inAmount', amount_smallest))
            price_impact = float(data.get('priceImpactPct') or 0.0)

            platform_fee = 0
            pf = data.get('platformFee')
            if pf and pf.get('amount'):
                platform_fee = int(pf['amount'])

            return {
                'out_amount': out_amount,
                'in_amount': in_amount,
                'price_impact_pct': price_impact,
                'platform_fee': platform_fee,
                'raw': data,   # unmodified quote, required by /swap
            }

        except Exception as e:
            _log(f"Jupiter quote error (attempt {attempt + 1}/{max_retries}): "
                 f"{type(e).__name__}: {e}", bot_id, 'WARNING')
            if attempt == max_retries - 1:
                return None
            time.sleep(1)

    return None


def estimate_network_fee(raw_quote, user_pubkey, bot_id=None):
    """
    Build the real (unsigned, unsent) swap transaction via Jupiter /swap and
    read the exact network fee it would cost: the Solana base signature fee
    plus the priority (prioritization) fee Jupiter sets for this route.

    Returns the fee in SOL. Falls back to ESTIMATED_NETWORK_FEE_SOL when there's
    no public key or the /swap build fails. Nothing is signed or submitted.

    Note: this does not include one-time ATA rent (~0.002 SOL, refundable) when
    a token account must be created — that's inside the transaction but not
    surfaced as a fee here.
    """
    if not user_pubkey or not raw_quote:
        return ESTIMATED_NETWORK_FEE_SOL

    payload = {
        'userPublicKey': user_pubkey,
        'quoteResponse': raw_quote,
        'dynamicComputeUnitLimit': True,
        'dynamicSlippage': True,
        # Let Jupiter pick a realistic priority fee for the route (capped at
        # 0.005 SOL). We only read it back; we never send the transaction.
        'prioritizationFeeLamports': 'auto',
    }

    for attempt in range(2):
        try:
            resp = requests.post(JUPITER_SWAP_URL, json=payload, timeout=10)
            if resp.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            prio_lamports = int(data.get('prioritizationFeeLamports') or 0)
            return BASE_SIGNATURE_FEE_SOL + prio_lamports / 1e9
        except Exception as e:
            _log(f"Jupiter /swap fee estimate error (attempt {attempt + 1}/2): "
                 f"{type(e).__name__}: {e}", bot_id, 'WARNING')

    return ESTIMATED_NETWORK_FEE_SOL
