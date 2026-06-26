from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import product
from typing import Any


def find_best_exit(
    df: pd.DataFrame,
    entry_conditions: list[list[dict[str, Any]]],
    buy_or_sell: str,
    target_deltas: list[float] | None = None,
    stoploss_deltas: list[float] | None = None,
    target_deltas_normalized: list[float] | None = None,
    stoploss_deltas_normalized: list[float] | None = None,
    exit_conditions_list: list[list[list[dict[str, Any]]]] | None = None,
    leverage: float = 1,
    initial_capital: float = 1000,
    risk_free_rate: float = 0,
    metric: str = "sharpe",
    higher_is_better: bool = True,
    trading_cost_factor: float = 0.0002,
    stoploss_wait_candleclose: bool = False,
    stoploss_consider_slipage: bool = True,
    verbose: bool = False,
) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    """Grid search over target/stoploss deltas and exit conditions to find the combination that optimizes the chosen metric. Returns (best_params, results_df)."""
    if not any(x is not None for x in [target_deltas, stoploss_deltas,
                                        target_deltas_normalized, stoploss_deltas_normalized,
                                        exit_conditions_list]):
        raise ValueError(
            "At least one exit parameter grid must be provided. Pass one or more of: "
            "target_deltas, stoploss_deltas, target_deltas_normalized, "
            "stoploss_deltas_normalized, exit_conditions_list"
        )

    for name, grid in [("target_deltas", target_deltas), ("stoploss_deltas", stoploss_deltas),
                        ("target_deltas_normalized", target_deltas_normalized),
                        ("stoploss_deltas_normalized", stoploss_deltas_normalized)]:
        if grid is not None:
            for i, val in enumerate(grid):
                if val is not None and not isinstance(val, (int, float)):
                    raise ValueError(f"{name}[{i}] must be numeric or None, got {type(val).__name__}")

    if initial_capital <= 0:
        raise ValueError(f"initial_capital must be > 0, got {initial_capital}")
    if leverage <= 0:
        raise ValueError(f"leverage must be > 0, got {leverage}")

    known_metrics = {"sharpe", "sharpe_ratio", "sharpe ratio", "final_capital", "capital", "volatility", "max_drawdown", "drawdown", "max drawdown"}
    if metric.lower() not in known_metrics:
        raise ValueError(
            f"Unknown metric '{metric}'. Known metrics: {', '.join(sorted(known_metrics))}"
        )

    from mtrader.exit_strategy import precalculate_exit_time_amount_profit
    from mtrader.trading import take_trade_on_condition_numpy

    target_grid = target_deltas if target_deltas is not None else [None]
    stoploss_grid = stoploss_deltas if stoploss_deltas is not None else [None]
    target_norm_grid = target_deltas_normalized if target_deltas_normalized is not None else [None]
    stoploss_norm_grid = stoploss_deltas_normalized if stoploss_deltas_normalized is not None else [None]
    exit_cond_grid = exit_conditions_list if exit_conditions_list is not None else [None]

    param_grid = list(product(target_grid, stoploss_grid, target_norm_grid, stoploss_norm_grid, exit_cond_grid))

    results = []
    best_score = -np.inf if higher_is_better else np.inf
    best_params = None

    exit_cols = {"exit_signal", "next_exit_index", "next_exit_value", "next_exit_profit",
                 "next_exit_index_cond", "next_exit_datetime", "next_exit_tradingcost",
                 "next_exit_capital_multiplier_in_percent", "target_price", "target_index",
                 "next_exit_value_target", "stoploss_index", "stoploss_price",
                 "next_exit_value_stoploss", "take_trade", "capital_at_exit"}
    stash = {}

    for tg, sl, tgn, sln, ec in param_grid:
        for col in exit_cols:
            if col in df.columns:
                stash[col] = df[col]
                del df[col]

        conds = ec if ec is not None else entry_conditions

        precalculate_exit_time_amount_profit(
            df, conds, buy_or_sell=buy_or_sell,
            target_delta=tg, stoploss_delta=sl,
            target_delta_normalized=tgn, stoploss_delta_normalized=sln,
            trading_cost_factor=trading_cost_factor,
            leverage=leverage,
            stoploss_wait_candleclose=stoploss_wait_candleclose,
            stoploss_consider_slipage=stoploss_consider_slipage,
        )

        _, final_capital, metrics = take_trade_on_condition_numpy(
            df, entry_conditions, leverage=leverage,
            initial_capital=initial_capital, risk_free_rate=risk_free_rate,
        )

        score = metrics.get(metric_map(metric), final_capital)
        if higher_is_better:
            is_better = score > best_score
        else:
            is_better = score < best_score

        if is_better:
            best_score = score
            best_params = {
                "target_delta": tg,
                "stoploss_delta": sl,
                "target_delta_normalized": tgn,
                "stoploss_delta_normalized": sln,
                "exit_conditions": ec,
            }

        entry_count = int(df["take_trade"].sum()) if "take_trade" in df.columns else 0

        results.append({
            "target_delta": tg,
            "stoploss_delta": sl,
            "target_delta_normalized": tgn,
            "stoploss_delta_normalized": sln,
            "exit_conditions_used": ec is not None,
            "trades": entry_count,
            "final_capital": final_capital,
            "sharpe": metrics.get("Sharpe Ratio", np.nan),
            "volatility": metrics.get("Volatility", np.nan),
            "max_drawdown": metrics.get("Max Drawdown", np.nan),
        })

        if verbose:
            _log_param(tg, sl, tgn, sln, ec, entry_count, final_capital, metrics)

        for col, val in stash.items():
            df[col] = val
        stash.clear()

    return best_params, pd.DataFrame(results)


def metric_map(name: str) -> str:
    """Map a short metric name (e.g. 'sharpe', 'drawdown') to the canonical dict key used in metrics."""
    m = {
        "sharpe": "Sharpe Ratio",
        "sharpe_ratio": "Sharpe Ratio",
        "final_capital": "final_capital",
        "capital": "final_capital",
        "volatility": "Volatility",
        "max_drawdown": "Max Drawdown",
        "drawdown": "Max Drawdown",
    }
    return m.get(name.lower(), name)


def _log_param(tg: Any, sl: Any, tgn: Any, sln: Any, ec: Any, trades: int, fc: float, metrics: dict[str, Any]) -> None:
    ec_label = "custom" if ec is not None else "entry"
    import builtins
    builtins.print(
        f"  t={tg} sl={sl} t_n={tgn} sl_n={sln} exit={ec_label} "
        f"| {trades:3d} trades | cap={fc:10.2f} "
        f"sharpe={metrics.get('Sharpe Ratio', np.nan):.3f} "
        f"dd={metrics.get('Max Drawdown', np.nan):.1f}%"
    )
