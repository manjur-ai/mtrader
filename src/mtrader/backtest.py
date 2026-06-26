from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Any

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
    trades, final_capital, metrics = take_trade_on_condition_numpy(
        data,
        entry_conditions,
        leverage=leverage,
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
    )
    report = backtest_report(data, initial_capital=initial_capital, risk_free_rate=risk_free_rate)
    equity = equity_curve(data, initial_capital=initial_capital)
    log = trade_log(data, side=buy_or_sell, initial_capital=initial_capital)
    return BacktestResult(data, log, float(final_capital), metrics, report, equity)


def trade_log(df: pd.DataFrame, side: str | None = None, initial_capital: float = 1000) -> pd.DataFrame:
    """Extract a trade-by-trade log from backtest results: entry/exit times, prices, profit, return %, and capital progression."""
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
