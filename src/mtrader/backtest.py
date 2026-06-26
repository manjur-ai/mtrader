from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd


def condition(
    first,
    second="zero",
    lower=-np.inf,
    upper=np.inf,
    shift_first=0,
    shift_second=0,
    normalize=False,
):
    return {
        "first_column_name": first,
        "second_column_name": second,
        "shift_down_first": shift_first,
        "shift_down_second": shift_second,
        "lower_range_of_difference": lower,
        "upper_range_of_difference": upper,
        "perform_normalization_of_diff": normalize,
    }


def cross_above(first, second, include_equal=True):
    prior_upper = 0 if include_equal else -np.finfo(float).eps
    current_lower = 0 if include_equal else np.finfo(float).eps
    return [
        condition(first, second, upper=prior_upper, shift_first=1, shift_second=1),
        condition(first, second, lower=current_lower),
    ]


def cross_below(first, second, include_equal=True):
    prior_lower = 0 if include_equal else np.finfo(float).eps
    current_upper = 0 if include_equal else -np.finfo(float).eps
    return [
        condition(first, second, lower=prior_lower, shift_first=1, shift_second=1),
        condition(first, second, upper=current_upper),
    ]


def validate_ohlcv(df, require_volume=False):
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
    metrics: dict
    report: dict
    equity: pd.DataFrame

    def to_html(self, output_path=None, title="mtrader Backtest Report", strategy_name=None, parameters=None):
        from mtrader.report import html_backtest_report

        return html_backtest_report(
            self,
            output_path=output_path,
            title=title,
            strategy_name=strategy_name,
            parameters=parameters,
        )


def run_backtest(
    df,
    entry_conditions,
    buy_or_sell="buy",
    exit_conditions=None,
    indicators=None,
    rolling_minutes=None,
    target_delta=None,
    stoploss_delta=None,
    target_delta_normalized=None,
    stoploss_delta_normalized=None,
    target_delta_column=None,
    stoploss_delta_column=None,
    initial_capital=1000,
    leverage=1,
    risk_free_rate=0,
    trading_cost_factor=0.0002,
    stoploss_wait_candleclose=False,
    stoploss_consider_slipage=True,
    copy=True,
):
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


def trade_log(df, side=None, initial_capital=1000):
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


def walk_forward_splits(df, train_days, test_days, step_days=None):
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


def parameter_grid(**params):
    keys = list(params)
    values = [v if isinstance(v, (list, tuple, np.ndarray, pd.Index)) else [v] for v in params.values()]
    return [dict(zip(keys, combo)) for combo in product(*values)]
