"""
Trade Execution Layer

Handles the actual execution of buy and sell orders. Contains two paths:
- Paper trading: simulates fills using the strategy's exit-price economics
- Live trading: submits real transactions to a Solana DEX (not yet implemented)

Both paths return the same standardized result dicts so the caller
(bot_loop.py) doesn't need to know which mode is running.

Shadow mode sits between the two: it QUOTES a real swap on Jupiter at
decision time (via dex.py) so the recorded fill/shares/impact are what a live
trade would get, but it never signs or submits. The quoted output amount
already reflects AMM + platform fees and price impact; on top of that we add a
small estimated network fee. If a quote can't be obtained, shadow falls back to
the paper computation for that fill and flags it (used_fallback) so the sim
keeps running.

This file has NO database access and NO strategy logic.
It takes execution parameters in and returns results out.

Paper-mode economics:
- execute_buy: entry price is close * (1 + slippage).
- execute_sell: fill_price_sol is the pre-computed exit price from
  A50Strategy.exit_fill_price() in strategy.py (close x (1 - slippage)).
  No additional slippage is applied here.
"""


import dex


def execute_buy(capital_sol, close_price_sol, slippage_pct, trading_mode,
                swap_ctx=None):
    """
    Execute a buy order (swap SOL -> token).

    In paper mode: the fill price is the candle close inflated by slippage,
    matching the entry economics:
        entry_price = close * (1 + slippage)
        shares = capital / entry_price
        fee = shares * close * slippage

    Args:
        capital_sol: Total SOL available to spend on this buy
        close_price_sol: Candle close price in SOL (raw market price)
        slippage_pct: Fee/slippage percentage as decimal (e.g. 0.005 = 0.5%)
        trading_mode: 'paper' or 'live'

    Returns:
        dict with keys:
        - shares_received: Number of tokens obtained
        - fill_price_sol: Actual entry price per token (close + slippage)
        - fee_sol: Total fee paid in SOL
        - tx_signature: Transaction signature (None in paper mode)
    """
    if trading_mode == 'paper':
        entry_price = close_price_sol * (1.0 + slippage_pct)
        shares_received = capital_sol / entry_price
        fee_sol = shares_received * close_price_sol * slippage_pct

        return {
            'shares_received': shares_received,
            'fill_price_sol': entry_price,
            'fee_sol': fee_sol,
            'tx_signature': None
        }

    elif trading_mode == 'shadow':
        # Quote a real SOL -> token swap at decision time; don't submit.
        # capital_sol is the (paper) SOL spent; out_amount is the tokens we'd
        # actually receive for that size, so shares/fill reflect real impact.
        bot_id = (swap_ctx or {}).get('bot_id')
        amount_lamports = int(round(capital_sol * 1e9))
        quote = dex.get_swap_quote(
            dex.SOL_MINT, swap_ctx['token_mint'], amount_lamports,
            swap_ctx.get('slippage_bps', dex.DEFAULT_SLIPPAGE_BPS), bot_id
        )

        if not quote or quote['out_amount'] <= 0:
            # Fall back to the paper computation so the sim keeps running.
            dex._log("Shadow buy: quote unavailable, using static fallback",
                     bot_id, 'WARNING')
            entry_price = close_price_sol * (1.0 + slippage_pct)
            shares_received = capital_sol / entry_price
            return {
                'shares_received': shares_received,
                'fill_price_sol': entry_price,
                'fee_sol': shares_received * close_price_sol * slippage_pct,
                'tx_signature': None,
                'price_impact_pct': None,
                'used_fallback': True,
            }

        shares_received = quote['out_amount'] / (10 ** swap_ctx['token_decimals'])
        fill_price = capital_sol / shares_received if shares_received > 0 else 0.0
        # Exact network fee from the real (unsent) swap transaction. This SOL is
        # actually burned, so it's a separate cost deducted from capital
        # (capital_fee_sol) on top of the swap output, which already reflects
        # AMM + impact + platform fees.
        network_fee = dex.estimate_network_fee(
            quote.get('raw'), swap_ctx.get('wallet_pubkey'), bot_id
        )
        return {
            'shares_received': shares_received,
            'fill_price_sol': fill_price,
            'fee_sol': network_fee,
            'capital_fee_sol': network_fee,
            'tx_signature': None,
            'price_impact_pct': quote['price_impact_pct'],
            'used_fallback': False,
        }

    elif trading_mode == 'live':
        # Future implementation:
        # 1. Get quote from DEX (Jupiter/Raydium) for swapping capital_sol -> token
        # 2. Check that quoted price impact is within MAX_PRICE_IMPACT_PCT
        # 3. Submit swap transaction
        # 4. Wait for transaction confirmation
        # 5. Query confirmed transaction for actual fill details
        # 6. Return actual shares received, actual fill price, actual fee, tx signature
        raise NotImplementedError("Live trading not yet implemented")

    else:
        raise ValueError(f"Unknown trading mode: {trading_mode}")


def execute_sell(shares_held, fill_price_sol, entry_price_sol, initial_capital_sol,
                 slippage_pct, trading_mode, swap_ctx=None):
    """
    Execute a sell order (swap token -> SOL).

    In paper mode: fill_price_sol is the pre-computed exit price from
    A50Strategy.exit_fill_price() in strategy.py (the close minus
    slippage). No additional slippage is applied.

    Fee is derived from the gross price (before slippage):
        actual_price = exit_price / (1 - slippage)
        fee = shares * actual_price * slippage
        proceeds = shares * exit_price  (= shares * actual_price - fee)

    Args:
        shares_held: Number of tokens to sell
        fill_price_sol: Pre-computed exit price (close x (1 - slippage))
        entry_price_sol: Original entry price, used for the P&L percentage and
                         the per-trade SOL P&L (shares * entry_price = cost basis)
        initial_capital_sol: SOL originally allocated to this bot. Retained for
                             signature compatibility; no longer used for P&L.
        slippage_pct: Fee/slippage percentage as decimal (e.g. 0.005 = 0.5%)
                      Used only to derive the fee from the gross price
        trading_mode: 'paper' or 'live'

    Returns:
        dict with keys:
        - proceeds_sol: Net SOL received (fill_price already includes slippage)
        - fill_price_sol: The exit price per token (as passed in)
        - fee_sol: Total fee paid in SOL
        - pnl_sol: Per-trade profit/loss in SOL (proceeds minus this trade's
                   buy cost = shares * (fill_price - entry_price))
        - pnl_pct: Percentage return based on entry vs exit price
        - tx_signature: Transaction signature (None in paper mode)
    """
    if trading_mode == 'paper':
        # fill_price_sol already includes slippage (strategy.exit_fill_price)
        proceeds_sol = shares_held * fill_price_sol

        # Derive gross price (before slippage) to compute fee
        # exit_price = actual_price * (1 - slippage) => actual_price = exit_price / (1 - slippage)
        actual_price = fill_price_sol / (1.0 - slippage_pct)
        fee_sol = shares_held * actual_price * slippage_pct

        # P&L for THIS trade, measured against what the position actually cost
        # to buy (shares * entry_price), NOT the bot's original allocation.
        # Using initial capital made the SOL P&L sign disagree with the
        # price-based percentage whenever the deployed amount differed from the
        # starting capital (e.g. after gains/losses or partial deployment).
        # proceeds_sol = shares * fill_price, so this equals
        # shares * (fill_price - entry_price) and is sign-consistent with pnl_pct.
        cost_basis_sol = shares_held * entry_price_sol
        pnl_sol = proceeds_sol - cost_basis_sol
        pnl_pct = (fill_price_sol - entry_price_sol) / entry_price_sol if entry_price_sol > 1e-10 else 0.0

        return {
            'proceeds_sol': proceeds_sol,
            'fill_price_sol': fill_price_sol,
            'fee_sol': fee_sol,
            'pnl_sol': pnl_sol,
            'pnl_pct': pnl_pct,
            'tx_signature': None
        }

    elif trading_mode == 'shadow':
        # Quote a real token -> SOL swap for the full position at decision time;
        # don't submit. fill_price_sol passed in (candle-based) is ignored — the
        # quote's output is the realistic proceeds. entry_price_sol is still the
        # cost basis reference for P&L.
        bot_id = (swap_ctx or {}).get('bot_id')
        decimals = swap_ctx['token_decimals']
        amount_smallest = int(round(shares_held * (10 ** decimals)))
        quote = dex.get_swap_quote(
            swap_ctx['token_mint'], dex.SOL_MINT, amount_smallest,
            swap_ctx.get('slippage_bps', dex.DEFAULT_SLIPPAGE_BPS), bot_id
        )

        if not quote or quote['out_amount'] <= 0:
            # Fall back to the pre-computed (candle-based) exit price.
            dex._log("Shadow sell: quote unavailable, using static fallback",
                     bot_id, 'WARNING')
            proceeds_sol = shares_held * fill_price_sol
            actual_price = fill_price_sol / (1.0 - slippage_pct)
            cost_basis_sol = shares_held * entry_price_sol
            return {
                'proceeds_sol': proceeds_sol,
                'fill_price_sol': fill_price_sol,
                'fee_sol': shares_held * actual_price * slippage_pct,
                'pnl_sol': proceeds_sol - cost_basis_sol,
                'pnl_pct': (fill_price_sol - entry_price_sol) / entry_price_sol
                           if entry_price_sol > 1e-10 else 0.0,
                'tx_signature': None,
                'price_impact_pct': None,
                'used_fallback': True,
            }

        proceeds_sol = quote['out_amount'] / 1e9
        realized_fill = proceeds_sol / shares_held if shares_held > 0 else 0.0
        cost_basis_sol = shares_held * entry_price_sol
        pnl_sol = proceeds_sol - cost_basis_sol
        pnl_pct = ((realized_fill - entry_price_sol) / entry_price_sol
                   if entry_price_sol > 1e-10 else 0.0)
        network_fee = dex.estimate_network_fee(
            quote.get('raw'), swap_ctx.get('wallet_pubkey'), bot_id
        )
        return {
            'proceeds_sol': proceeds_sol,
            'fill_price_sol': realized_fill,
            'fee_sol': network_fee,
            'capital_fee_sol': network_fee,
            'pnl_sol': pnl_sol,
            'pnl_pct': pnl_pct,
            'tx_signature': None,
            'price_impact_pct': quote['price_impact_pct'],
            'used_fallback': False,
        }

    elif trading_mode == 'live':
        # Future implementation:
        # 1. Get quote from DEX for swapping all held tokens -> SOL
        # 2. Submit swap transaction
        # 3. Wait for transaction confirmation
        # 4. Query confirmed transaction for actual SOL received and fee
        # 5. Calculate P&L from actual proceeds vs initial capital
        # 6. Return actual proceeds, actual fill price, actual fee, P&L, tx signature
        raise NotImplementedError("Live trading not yet implemented")

    else:
        raise ValueError(f"Unknown trading mode: {trading_mode}")
