from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd


def condition(
    first: str,
    second: str = "zero",
    lower: float = -np.inf,
    upper: float = np.inf,
    shift_first: int = 0,
    shift_second: int = 0,
    normalize: bool = False,
) -> dict[str, Any]:
    """Create a single condition dict: compares (first shifted by shift_first) - (second shifted by shift_second) against [lower, upper]."""
    return {
        "first_column_name": first,
        "second_column_name": second,
        "shift_down_first": shift_first,
        "shift_down_second": shift_second,
        "lower_range_of_difference": lower,
        "upper_range_of_difference": upper,
        "perform_normalization_of_diff": normalize,
    }


def cross_above(first: str, second: str, include_equal: bool = True) -> list[dict[str, Any]]:
    """Generate two conditions that detect when `first` crosses above `second`: prior diff <= 0 and current diff >= 0."""
    prior_upper = 0 if include_equal else -np.finfo(float).eps
    current_lower = 0 if include_equal else np.finfo(float).eps
    return [
        condition(first, second, upper=prior_upper, shift_first=1, shift_second=1),
        condition(first, second, lower=current_lower),
    ]


def cross_below(first: str, second: str, include_equal: bool = True) -> list[dict[str, Any]]:
    """Generate two conditions that detect when `first` crosses below `second`: prior diff >= 0 and current diff <= 0."""
    prior_lower = 0 if include_equal else np.finfo(float).eps
    current_upper = 0 if include_equal else -np.finfo(float).eps
    return [
        condition(first, second, lower=prior_lower, shift_first=1, shift_second=1),
        condition(first, second, upper=current_upper),
    ]


def validate_ohlcv(df: pd.DataFrame, require_volume: bool = False) -> bool:
    """Validate that a DataFrame has proper OHLCV columns: datetime (sorted, unique), open/high/low/close numeric, no NaNs, and valid ranges."""
    required = {"datetime", "open", "high", "low", "close"}
    if require_volume:
        required.add("volume")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        raise ValueError("'datetime' column must be a pandas datetime dtype")

    numeric_cols = ["open", "high", "low", "close"] + (["volume"] if require_volume else [])
    non_numeric = [c for c in numeric_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Columns must be numeric: {non_numeric}")

    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("OHLC columns cannot contain NaN values")

    bad_high = df["high"] < df[["open", "close", "low"]].max(axis=1)
    bad_low = df["low"] > df[["open", "close", "high"]].min(axis=1)
    if bool((bad_high | bad_low).any()):
        raise ValueError("'high'/'low' values are inconsistent with open/close prices")

    if df["datetime"].duplicated().any():
        raise ValueError("'datetime' column contains duplicate timestamps")

    if not df["datetime"].is_monotonic_increasing:
        raise ValueError("'datetime' column must be sorted ascending")

    return True


@dataclass
class BacktestResult:
    df: pd.DataFrame
    trades: pd.DataFrame
    final_capital: float
    metrics: dict[str, Any]
    report: dict[str, Any]
    equity: pd.DataFrame

    def to_html(self: BacktestResult, output_path: str | None = None, title: str = "mtrader Backtest Report", strategy_name: str | None = None, parameters: dict[str, Any] | None = None) -> str:
        """Generate an HTML report from this BacktestResult. Delegates to html_backtest_report."""
        from mtrader.report import html_backtest_report

        return html_backtest_report(
            self,
            output_path=output_path,
            title=title,
            strategy_name=strategy_name,
            parameters=parameters,
        )


def run_backtest(
    df: pd.DataFrame,
    entry_conditions: list[list[dict[str, Any]]],
    buy_or_sell: str = "buy",
    exit_conditions: list[list[dict[str, Any]]] | None = None,
    indicators: list[str] | None = None,
    rolling_minutes: list[int] | None = None,
    target_delta: float | None = None,
    stoploss_delta: float | None = None,
    target_delta_normalized: float | None = None,
    stoploss_delta_normalized: float | None = None,
    target_delta_column: str | None = None,
    stoploss_delta_column: str | None = None,
    initial_capital: float = 1000,
    leverage: float = 1,
    risk_free_rate: float = 0,
    trading_cost_factor: float = 0.0002,
    stoploss_wait_candleclose: bool = False,
    stoploss_consider_slipage: bool = True,
    copy: bool = True,
    capital_per_trade_pct: float = 1.0,
    sizing_fn: Callable | None = None,
    min_hold_bars: int = 0,
    max_hold_bars: int | None = None,
    max_trades_per_day: int | None = None,
    cooldown_bars: int = 0,
    max_daily_loss_pct: float | None = None,
) -> BacktestResult:
    """Run a complete backtest: validate, add indicators, pre-calculate exits, execute trades, and return a BacktestResult with report/equity/trade_log."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df must be a non-empty DataFrame")
    required_ohlcv = {"datetime", "open", "high", "low", "close"}
    missing = sorted(required_ohlcv - set(df.columns))
    if missing:
        raise ValueError(f"DataFrame missing required OHLCV columns: {missing}")
    if not isinstance(entry_conditions, list) or len(entry_conditions) == 0:
        raise ValueError("entry_conditions must be a non-empty list")
    if buy_or_sell not in ("buy", "sell"):
        raise ValueError(f"buy_or_sell must be 'buy' or 'sell', got '{buy_or_sell}'")
    if rolling_minutes is not None:
        if not isinstance(rolling_minutes, list):
            raise ValueError("rolling_minutes must be a list if provided")
        for i, rm in enumerate(rolling_minutes):
            if not isinstance(rm, int) or rm <= 0:
                raise ValueError(f"rolling_minutes[{i}] must be a positive integer, got {rm}")
    if indicators is not None and not isinstance(indicators, list):
        raise ValueError("indicators must be a list if provided")

    from mtrader.exit_strategy import precalculate_exit_time_amount_profit
    from mtrader.indicator_engine import add_indicators
    from mtrader.report import backtest_report, equity_curve
    from mtrader.trading import take_trade_on_condition_numpy

    data = df.copy() if copy else df
    validate_ohlcv(data, require_volume=bool(indicators and any(i in indicators for i in ["vwap", "obv", "mfi"])))

    if indicators:
        data = add_indicators(data, add=list(indicators), rolling_minutes=rolling_minutes or [])

    if "zero" not in data.columns:
        data["zero"] = 0.0

    exit_conditions = exit_conditions if exit_conditions is not None else entry_conditions
    data = precalculate_exit_time_amount_profit(
        data,
        exit_conditions,
        buy_or_sell=buy_or_sell,
        trading_cost_factor=trading_cost_factor,
        leverage=leverage,
        target_delta=target_delta,
        stoploss_delta=stoploss_delta,
        target_delta_normalized=target_delta_normalized,
        stoploss_delta_normalized=stoploss_delta_normalized,
        target_delta_column=target_delta_column,
        stoploss_delta_column=stoploss_delta_column,
        stoploss_wait_candleclose=stoploss_wait_candleclose,
        stoploss_consider_slipage=stoploss_consider_slipage,
    )
    # Resolve capital_per_trade_pct per trade if sizing_fn is provided
    if sizing_fn is not None and not callable(sizing_fn):
        raise ValueError("sizing_fn must be a callable(entry_idx, capital_before, df) -> float")

    # First pass: compute trades with position sizing
    trades, final_capital, metrics = take_trade_on_condition_numpy(
        data,
        entry_conditions,
        leverage=leverage,
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
        capital_per_trade_pct=capital_per_trade_pct,
        sizing_fn=sizing_fn,
    )

    # Apply holding period filters
    hold_filter = bool(min_hold_bars > 0 or max_hold_bars is not None)
    # Apply risk control filters
    trade_filter = bool(max_trades_per_day is not None or cooldown_bars > 0 or max_daily_loss_pct is not None)

    if hold_filter or trade_filter:
        log = trade_log(data, side=buy_or_sell, initial_capital=initial_capital)
        allowed = pd.Series(True, index=log.index)

        # Holding period filter
        if hold_filter and not log.empty:
            hold_bars = log["exit_index"].to_numpy() - log["entry_index"].to_numpy()
            if min_hold_bars > 0:
                allowed &= hold_bars >= min_hold_bars
            if max_hold_bars is not None:
                allowed &= hold_bars <= max_hold_bars

        # Risk control filter
        if trade_filter and not log.empty:
            from mtrader.advanced import apply_risk_controls
            controlled = apply_risk_controls(
                log,
                max_trades_per_day=max_trades_per_day,
                cooldown_bars=cooldown_bars,
                max_daily_loss_pct=max_daily_loss_pct,
            )
            allowed &= controlled["allowed"]

        # Apply filters
        if not log.empty and (~allowed).any():
            data["take_trade"] = False
            allowed_indices = log.index[allowed]
            if len(allowed_indices) > 0:
                orig_indices = log.loc[allowed_indices, "entry_index"].values
                data.loc[orig_indices, "take_trade"] = True

            # Recompute capital
            take_trade_idx = np.where(data["take_trade"].to_numpy())[0]
            caps = np.zeros(len(data), dtype=float)
            cap_before = initial_capital
            cap_mult = data["next_exit_capital_multiplier_in_percent"].to_numpy()
            for ti in take_trade_idx:
                pct = sizing_fn(ti, cap_before, data) if sizing_fn else capital_per_trade_pct
                deploy = cap_before * pct
                cash = cap_before - deploy
                cap_after = deploy * (cap_mult[ti] / 100.0) + cash
                caps[ti] = cap_after
                cap_before = cap_after
            data["capital_at_exit"] = caps
            final_capital = float(cap_before)

    report = backtest_report(data, initial_capital=initial_capital, risk_free_rate=risk_free_rate)
    equity = equity_curve(data, initial_capital=initial_capital)
    log = trade_log(data, side=buy_or_sell, initial_capital=initial_capital)

    # Enhance metrics with report data
    for k in ["sortino_ratio", "calmar_ratio", "win_rate_pct", "profit_factor",
              "total_trades", "total_return_pct"]:
        if k in report and (k not in metrics or metrics.get(k) is None or (
            isinstance(metrics.get(k), float) and np.isnan(metrics.get(k)))):
            metrics[k] = report[k]

    return BacktestResult(data, log, float(final_capital), metrics, report, equity)


def run_oms_backtest(
    df: pd.DataFrame,
    strategy: Any,
    *,
    lot_size: float = 1.0,
    initial_capital: float | None = None,
    history_size: int = 512,
    close_open_at_end: bool = False,
) -> BacktestResult:
    """
    Run a mtrader Strategy with OMS-like single-position execution semantics.

    This is intentionally different from the vectorized run_backtest path:
    it evaluates mtrader live signals bar-by-bar, allows only one open trade,
    fills market entries at the signal bar open, and exits via SL/TP before
    evaluating the next signal.
    """
    validate_ohlcv(df, require_volume=False)
    data = df.copy().sort_values("datetime").reset_index(drop=True)
    capital_start = float(
        initial_capital if initial_capital is not None
        else getattr(strategy, "initial_capital", 1000.0)
    )
    side = (getattr(strategy, "side", "buy") or "buy").lower()
    qty = float(lot_size)
    live_engine = None
    open_trade = None
    trades = []
    equity_rows = []
    equity = capital_start

    for i, row in data.iterrows():
        bar = row.to_dict()

        if open_trade is not None:
            closed = _oms_exit_for_bar(open_trade, i, bar)
            if closed is not None:
                equity += closed["profit"]
                trades.append(closed)
                open_trade = None

        if live_engine is None:
            if i == 0:
                equity_rows.append({
                    "datetime": bar["datetime"],
                    "equity": equity,
                    "drawdown_pct": 0.0,
                    "trade": False,
                })
                continue
            from mtrader.live import live_strategy_from_history

            history = data.iloc[max(0, i - history_size):i]
            live_engine = live_strategy_from_history(
                history,
                indicators=list(getattr(strategy, "indicators", []) or []),
                periods=list(getattr(strategy, "rolling_minutes", []) or []),
                entry_conditions=getattr(strategy, "entry_conditions", []) or [],
                exit_conditions=getattr(strategy, "exit_conditions", None),
                side=side,
                history_size=history_size,
                warmup_batch=True,
            )

        signal = live_engine.update(bar)

        if open_trade is not None and signal.get("exit_signal"):
            closed = _close_oms_trade(open_trade, i, bar, float(bar["close"]), "mtrader_exit_signal")
            equity += closed["profit"]
            trades.append(closed)
            open_trade = None
            equity_rows.append({
                "datetime": bar["datetime"],
                "equity": equity,
                "drawdown_pct": 0.0,
                "trade": True,
            })
            continue

        if open_trade is None and signal.get("entry_signal"):
            entry_price = float(bar["open"])
            open_trade = _open_oms_trade(strategy, side, qty, i, bar, entry_price)

        equity_rows.append({
            "datetime": bar["datetime"],
            "equity": equity,
            "drawdown_pct": 0.0,
            "trade": False,
        })

    if close_open_at_end and open_trade is not None and len(data) > 0:
        row = data.iloc[-1].to_dict()
        closed = _close_oms_trade(open_trade, len(data) - 1, row, float(row["close"]), "end")
        equity += closed["profit"]
        trades.append(closed)

    trades_df = pd.DataFrame(trades)
    report = _oms_report_from_trades(trades_df, capital_start, equity)
    equity_df = pd.DataFrame(equity_rows)
    if not equity_df.empty:
        equity_df["equity"] = equity_df["equity"].ffill().fillna(capital_start)
        peak = equity_df["equity"].cummax()
        equity_df["drawdown_pct"] = (peak - equity_df["equity"]) / peak * 100.0
    metrics = {
        "total_trades": report["total_trades"],
        "final_capital": report["final_capital"],
        "profit_factor": report["profit_factor"],
        "win_rate_pct": report["win_rate_pct"],
    }
    return BacktestResult(data, trades_df, float(equity), metrics, report, equity_df)


def _open_oms_trade(strategy: Any, side: str, qty: float, index: int, bar: dict, entry_price: float) -> dict:
    target = _target_price(strategy, side, entry_price)
    stop = _stop_price(strategy, side, entry_price)
    return {
        "entry_index": int(index),
        "entry_time": bar["datetime"],
        "side": side,
        "qty": qty,
        "entry_price": entry_price,
        "target_price": target,
        "stoploss_price": stop,
    }


def _target_price(strategy: Any, side: str, entry_price: float) -> float | None:
    absolute = getattr(strategy, "target_delta", None)
    percent = getattr(strategy, "target_delta_normalized", None)
    if absolute is None and percent is None:
        return None
    points = float(absolute) if absolute is not None else entry_price * float(percent) / 100.0
    return entry_price + points if side == "buy" else entry_price - points


def _stop_price(strategy: Any, side: str, entry_price: float) -> float | None:
    absolute = getattr(strategy, "stoploss_delta", None)
    percent = getattr(strategy, "stoploss_delta_normalized", None)
    if absolute is None and percent is None:
        return None
    points = float(absolute) if absolute is not None else entry_price * float(percent) / 100.0
    return entry_price - points if side == "buy" else entry_price + points


def _oms_exit_for_bar(trade: dict, exit_index: int, bar: dict) -> dict | None:
    side = trade["side"]
    open_price = float(bar["open"])
    high = float(bar["high"])
    low = float(bar["low"])
    stop = trade.get("stoploss_price")
    target = trade.get("target_price")

    if side == "buy":
        stop_hit = stop is not None and low <= stop
        target_hit = target is not None and high >= target
        if stop_hit:
            exit_price = open_price if open_price <= stop else stop
            return _close_oms_trade(trade, exit_index, bar, exit_price, "sl_hit")
        if target_hit:
            exit_price = open_price if open_price >= target else target
            return _close_oms_trade(trade, exit_index, bar, exit_price, "tp_hit")
    else:
        stop_hit = stop is not None and high >= stop
        target_hit = target is not None and low <= target
        if stop_hit:
            exit_price = open_price if open_price >= stop else stop
            return _close_oms_trade(trade, exit_index, bar, exit_price, "sl_hit")
        if target_hit:
            exit_price = open_price if open_price <= target else target
            return _close_oms_trade(trade, exit_index, bar, exit_price, "tp_hit")
    return None


def _close_oms_trade(trade: dict, exit_index: int | None, bar: dict, exit_price: float, reason: str) -> dict:
    entry_price = float(trade["entry_price"])
    qty = float(trade["qty"])
    is_long = trade["side"] == "buy"
    profit = round(((exit_price - entry_price) if is_long else (entry_price - exit_price)) * qty, 4)
    idx = int(exit_index) if exit_index is not None else None
    return {
        "entry_index": trade["entry_index"],
        "entry_time": trade["entry_time"],
        "exit_index": idx,
        "exit_time": bar["datetime"],
        "side": trade["side"],
        "qty": qty,
        "entry_price": entry_price,
        "exit_price": float(exit_price),
        "profit": profit,
        "return_pct": (profit / (entry_price * qty)) * 100.0 if entry_price and qty else 0.0,
        "exit_reason": reason,
        "target_price": trade.get("target_price"),
        "stoploss_price": trade.get("stoploss_price"),
    }


def _oms_report_from_trades(trades: pd.DataFrame, initial_capital: float, final_capital: float) -> dict[str, Any]:
    if trades.empty:
        return {
            "total_trades": 0,
            "final_capital": round(final_capital, 4),
            "total_return_pct": round((final_capital - initial_capital) / initial_capital * 100, 2),
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "max_drawdown_pct": 0.0,
        }
    profits = trades["profit"].to_numpy(dtype=np.float64)
    wins = profits[profits > 0]
    losses = profits[profits < 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(losses.sum()) if len(losses) else 0.0
    profit_factor = None if gross_loss == 0 else gross_profit / abs(gross_loss)
    return {
        "total_trades": int(len(trades)),
        "final_capital": round(final_capital, 4),
        "total_return_pct": round((final_capital - initial_capital) / initial_capital * 100, 2),
        "win_rate_pct": round((len(wins) / len(trades)) * 100.0, 2),
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
        "net_pnl": round(float(profits.sum()), 4),
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
        "max_drawdown_pct": 0.0,
    }


def trade_log(df: pd.DataFrame, side: str | None = None, initial_capital: float = 1000) -> pd.DataFrame:
    """Extract a trade-by-trade log from backtest results: entry/exit times, prices, profit, return %, capital progression, and exit reason."""
    required = {"take_trade", "next_exit_datetime", "next_exit_value", "capital_at_exit", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"DataFrame is missing trade-result columns: {missing}")

    trades = df.loc[df["take_trade"]].copy()
    if trades.empty:
        return pd.DataFrame(columns=[
            "entry_index", "entry_time", "exit_index", "exit_time", "side",
            "entry_price", "exit_price", "profit", "return_pct", "capital_at_exit",
        ])

    out = pd.DataFrame(index=trades.index)
    out["entry_index"] = trades.index.to_numpy()
    out["entry_time"] = trades["datetime"].to_numpy() if "datetime" in trades.columns else trades.index.to_numpy()
    out["exit_index"] = trades["next_exit_index"].to_numpy() if "next_exit_index" in trades.columns else np.nan
    out["exit_time"] = trades["next_exit_datetime"].to_numpy()
    out["side"] = side or ""
    out["entry_price"] = trades["close"].to_numpy(dtype=np.float64)
    out["exit_price"] = trades["next_exit_value"].to_numpy(dtype=np.float64)
    if side == "sell":
        out["profit"] = out["entry_price"] - out["exit_price"]
        out["return_pct"] = (out["profit"] / out["entry_price"]) * 100.0
    else:
        out["profit"] = out["exit_price"] - out["entry_price"]
        out["return_pct"] = (out["profit"] / out["entry_price"]) * 100.0
    out["capital_at_exit"] = trades["capital_at_exit"].to_numpy(dtype=np.float64)
    out["capital_before"] = np.concatenate([[initial_capital], out["capital_at_exit"].to_numpy()[:-1]])
    out["capital_return_pct"] = ((out["capital_at_exit"] - out["capital_before"]) / out["capital_before"]) * 100.0

    # Exit reason tagging
    exit_reason = []
    has_target = "target_index" in df.columns
    has_stop = "stoploss_index" in df.columns
    for _, row in out.iterrows():
        ei = int(row["entry_index"])
        xi = int(row["exit_index"])
        if xi <= ei:
            exit_reason.append("end")
        elif has_target and ei in df.index and xi == int(df.loc[ei, "target_index"]):
            exit_reason.append("target")
        elif has_stop and ei in df.index and xi == int(df.loc[ei, "stoploss_index"]):
            exit_reason.append("stoploss")
        else:
            exit_reason.append("condition")
    out["exit_reason"] = exit_reason

    # Holding bars
    out["hold_bars"] = (out["exit_index"] - out["entry_index"]).astype(int)

    return out.reset_index(drop=True)


def walk_forward_splits(df: pd.DataFrame, train_days: int, test_days: int, step_days: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate train/test index pairs for walk-forward analysis. Splits by unique trading days. Returns list of (train_idx, test_idx) tuples."""
    if train_days <= 0 or test_days <= 0:
        raise ValueError("train_days and test_days must be positive")
    step_days = test_days if step_days is None else step_days
    if step_days <= 0:
        raise ValueError("step_days must be positive")
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must have a 'datetime' column")

    dates = pd.Index(pd.to_datetime(df["datetime"]).dt.normalize().unique()).sort_values()
    splits = []
    start = 0
    while start + train_days + test_days <= len(dates):
        train_dates = set(dates[start:start + train_days])
        test_dates = set(dates[start + train_days:start + train_days + test_days])
        normalized = pd.to_datetime(df["datetime"]).dt.normalize()
        train_idx = df.index[normalized.isin(train_dates)].to_numpy()
        test_idx = df.index[normalized.isin(test_dates)].to_numpy()
        splits.append((train_idx, test_idx))
        start += step_days
    return splits


def parameter_grid(**params: Any) -> list[dict[str, Any]]:
    """Build a Cartesian product of parameter lists, returning a list of dicts. Scalar values are wrapped in a single-element list."""
    keys = list(params)
    values = [v if isinstance(v, (list, tuple, np.ndarray, pd.Index)) else [v] for v in params.values()]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def run_scenarios(
    df: pd.DataFrame,
    base_params: dict[str, Any],
    param_grid: dict[str, list[Any]],
    metric: str = "final_capital",
    verbose: bool = False,
) -> pd.DataFrame:
    """Run backtests across multiple parameter combinations and return a comparison DataFrame.

    base_params: shared parameters passed to every run_backtest call.
    param_grid: dict mapping parameter names to lists of values to try.
    metric: which metric to sort by (from report or metrics dict).

    Returns a DataFrame with one row per scenario, sorted by metric descending.
    """
    from copy import deepcopy

    keys = list(param_grid)
    all_combos = list(product(*(param_grid[k] for k in keys)))
    rows = []

    for combo in all_combos:
        params = dict(zip(keys, combo))
        run_kwargs = {**base_params, **params}
        try:
            result = run_backtest(df, **run_kwargs)
            score = result.report.get(metric, result.final_capital)
            row = {
                **params,
                "final_capital": result.final_capital,
                "total_trades": result.report.get("total_trades", 0),
                "sharpe": result.metrics.get("Sharpe Ratio", float("nan")),
                "sortino": result.report.get("sortino_ratio", float("nan")),
                "calmar": result.report.get("calmar_ratio", float("nan")),
                "win_rate": result.report.get("win_rate_pct", float("nan")),
                "profit_factor": result.report.get("profit_factor", float("nan")),
                metric: score,
            }
            rows.append(row)
            if verbose:
                print(f"  {params}: {metric}={score:.4f}, fc={result.final_capital:.2f}, trades={row['total_trades']}")
        except Exception as e:
            rows.append({**params, "final_capital": 0, "total_trades": 0, metric: float("-inf"), "error": str(e)})

    result_df = pd.DataFrame(rows)
    sort_col = metric if metric in result_df.columns else "final_capital"
    result_df = result_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return result_df


def find_best_entry_conditions(
    df: pd.DataFrame,
    entry_condition_template: list[list[dict[str, Any]]],
    condition_ranges: dict[int, tuple[list[float], list[float]]],
    buy_or_sell: str = "buy",
    exit_conditions: list[list[dict[str, Any]]] | None = None,
    indicators: list[str] | None = None,
    rolling_minutes: list[int] | None = None,
    metric: str = "sharpe",
    higher_is_better: bool = True,
    verbose: bool = False,
    **run_kwargs: Any,
) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    """Grid-search over entry condition range combinations to find the best-performing set.

    condition_ranges: dict mapping condition index (within the flat list) to (lower_values, upper_values).
    Example: {0: ([0, 10, 20], [np.inf, np.inf, np.inf])} tries 3 ranges for condition 0.

    Returns (best_params, results_df) where best_params contains the best (lower, upper) pairs.
    """
    from mtrader.indicator_engine import add_indicators
    from mtrader.exit_strategy import precalculate_exit_time_amount_profit
    from mtrader.trading import take_trade_on_condition_numpy
    from mtrader.report import backtest_report
    from copy import deepcopy

    data = df.copy()

    if indicators:
        data = add_indicators(data, add=list(indicators), rolling_minutes=rolling_minutes or [])
    if "zero" not in data.columns:
        data["zero"] = 0.0

    exit_conds = exit_conditions if exit_conditions is not None else entry_condition_template
    data = precalculate_exit_time_amount_profit(data, exit_conds, buy_or_sell=buy_or_sell, **{k: v for k, v in run_kwargs.items() if k in ("trading_cost_factor", "leverage", "target_delta", "stoploss_delta", "target_delta_normalized", "stoploss_delta_normalized", "target_delta_column", "stoploss_delta_column")})

    # Flatten conditions to assign indices
    flat_conds = []
    for group in entry_condition_template:
        for cond in group:
            flat_conds.append(cond)

    # Build all combos: zip lower/upper pairs for each condition, then product across conditions
    keys = sorted(condition_ranges.keys())
    all_combos = product(*(zip(condition_ranges[k][0], condition_ranges[k][1]) for k in keys))
    results_rows = []
    best = None
    best_score = -np.inf if higher_is_better else np.inf

    for combo in all_combos:
        params = {}
        conds = deepcopy(entry_condition_template)
        for ki, (lower_val, upper_val) in zip(keys, combo):
            params[f"cond_{ki}_lower"] = lower_val
            params[f"cond_{ki}_upper"] = upper_val
            # Find and update the condition
            idx = 0
            for g in conds:
                for c in g:
                    if idx == ki:
                        c["lower_range_of_difference"] = lower_val
                        c["upper_range_of_difference"] = upper_val
                    idx += 1

        data2 = data.copy()
        trades, fc, met = take_trade_on_condition_numpy(
            data2, conds,
            initial_capital=run_kwargs.get("initial_capital", 1000),
            leverage=run_kwargs.get("leverage", 1),
            risk_free_rate=run_kwargs.get("risk_free_rate", 0),
            capital_per_trade_pct=run_kwargs.get("capital_per_trade_pct", 1.0),
        )
        report = backtest_report(data2, initial_capital=run_kwargs.get("initial_capital", 1000))
        score = report.get(metric, met.get(metric, fc))
        if score is None or (isinstance(score, float) and np.isnan(score)):
            score = -np.inf if higher_is_better else np.inf

        row = {**params, metric: score, "final_capital": fc, "total_trades": report.get("total_trades", 0)}
        results_rows.append(row)

        if (higher_is_better and score > best_score) or (not higher_is_better and score < best_score):
            best_score = score
            best = params.copy()
            best["score"] = score

        if verbose:
            print(f"  {params}: {metric}={score:.4f}, fc={fc:.2f}")

    return best, pd.DataFrame(results_rows)
