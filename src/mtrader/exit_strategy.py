from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Any

try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False

from mtrader.monotonic_stack import monotonic_stack_for_value1_lessthan_value2, monotonic_stack_for_value1_gt_value2


def _replace_missing_exit_indices(indices: NDArray[np.int64] | NDArray[np.int32], last_index: int) -> NDArray[np.int64]:
    indices = np.asarray(indices, dtype=np.int64)
    return np.where(indices < 0, last_index, indices)


def precalculate_exit_time_amount_profit(
    df: pd.DataFrame,
    conditions: list[list[dict[str, Any]]],
    buy_or_sell: str,
    trading_cost_factor: float = 0.0002,
    leverage: float = 1,
    target_delta: float | None = None,
    stoploss_delta: float | None = None,
    target_delta_normalized: float | None = None,
    stoploss_delta_normalized: float | None = None,
    target_delta_column: str | None = None,
    stoploss_delta_column: str | None = None,
    stoploss_wait_candleclose: bool = False,
    stoploss_consider_slipage: bool = True,
    stoploss_consider_slippage: bool | None = None,
    cupy: bool = False
) -> pd.DataFrame:
    """Pre-calculate exit index, exit value, trading cost, profit, and capital multiplier for every bar based on conditions, target, and stoploss. Adds columns: next_exit_index, next_exit_value, next_exit_profit, next_exit_capital_multiplier_in_percent, etc."""
    if stoploss_consider_slippage is not None:
        stoploss_consider_slipage = stoploss_consider_slippage
    array_lib = cp if (cupy and has_cupy) else np

    close_prices = df['close'].to_numpy()
    high_prices = df['high'].to_numpy()
    low_prices = df['low'].to_numpy()

    shifted_data = {}
    for condition_group in conditions:
        for condition in condition_group:
            key_first = (condition["first_column_name"], condition["shift_down_first"])
            key_second = (condition["second_column_name"], condition["shift_down_second"])

            if key_first not in shifted_data:
                shifted_data[key_first] = array_lib.asarray(
                    df[condition["first_column_name"]].shift(condition["shift_down_first"]).to_numpy()
                )
            if key_second not in shifted_data:
                shifted_data[key_second] = array_lib.asarray(
                    df[condition["second_column_name"]].shift(condition["shift_down_second"]).to_numpy()
                )

    exit_signals = array_lib.zeros(len(df), dtype=bool)

    for condition_group in conditions:
        if len(condition_group) > 0:
            condition_met_group = array_lib.ones(len(df), dtype=bool)
        else:
            condition_met_group = array_lib.zeros(len(df), dtype=bool)

        for unit_condition in condition_group:
            shifted_first = shifted_data[(unit_condition["first_column_name"], unit_condition["shift_down_first"])]
            shifted_second = shifted_data[(unit_condition["second_column_name"], unit_condition["shift_down_second"])]
            difference = shifted_first - shifted_second

            if unit_condition["perform_normalization_of_diff"]:
                difference *= 10000 / df["close"].to_numpy()

            unit_condition_met = (difference >= unit_condition["lower_range_of_difference"]) & \
                                 (difference <= unit_condition["upper_range_of_difference"])

            condition_met_group &= unit_condition_met

        exit_signals |= condition_met_group

    df["exit_signal"] = exit_signals

    exit_indices = array_lib.where(exit_signals)[0]

    if len(exit_indices) > 0:
        next_exit_indices = array_lib.full(len(df), array_lib.nan, dtype=float)
        next_exit_indices[exit_indices] = exit_indices
        next_exit_indices = pd.Series(next_exit_indices).bfill().to_numpy()
        next_exit_indices = array_lib.nan_to_num(next_exit_indices, nan=len(df) - 1).astype(int)
    else:
        next_exit_indices = array_lib.full(len(df), len(df) - 1, dtype=int)

    next_exit_value = close_prices[next_exit_indices]
    next_exit_indices_cond = next_exit_indices.copy()
    next_exit_value_cond = next_exit_value.copy()

    df["next_exit_index_cond"] = next_exit_indices_cond

    if target_delta is not None or target_delta_normalized is not None or target_delta_column is not None:
        if target_delta_column is not None:
            target_delta = df[target_delta_column].to_numpy(dtype=np.float64)
        elif target_delta_normalized is not None:
            target_delta = target_delta_normalized * (close_prices / 100.0)

        if buy_or_sell == "buy":
            target_prices = close_prices + target_delta
            target_index = monotonic_stack_for_value1_gt_value2(high_prices, target_prices)
            next_exit_value_target = target_prices
        elif buy_or_sell == "sell":
            target_prices = close_prices - target_delta
            target_index = monotonic_stack_for_value1_lessthan_value2(low_prices, target_prices)
            next_exit_value_target = target_prices
        else:
            raise ValueError("Invalid value for 'buy_or_sell'. It must be either 'buy' or 'sell'.")

        target_index = _replace_missing_exit_indices(target_index, len(df) - 1)
        df["target_price"] = target_prices
        df["target_index"] = target_index
        df["next_exit_value_target"] = next_exit_value_target

        next_exit_indices = np.minimum(next_exit_indices, target_index)
        next_exit_value = np.where(next_exit_indices == target_index, next_exit_value_target, next_exit_value)

    if stoploss_delta is not None or stoploss_delta_normalized is not None or stoploss_delta_column is not None:
        if stoploss_delta_column is not None:
            stoploss_delta = df[stoploss_delta_column].to_numpy(dtype=np.float64)
        elif stoploss_delta_normalized is not None:
            stoploss_delta = stoploss_delta_normalized * (close_prices / 100.0)

        if buy_or_sell == "buy":
            if stoploss_wait_candleclose:
                values1 = close_prices
                stoploss_prices = close_prices - stoploss_delta
            else:
                values1 = low_prices
                stoploss_prices = close_prices - stoploss_delta

            stoploss_index = monotonic_stack_for_value1_lessthan_value2(values1, stoploss_prices)
            stoploss_index = _replace_missing_exit_indices(stoploss_index, len(df) - 1)

            if stoploss_consider_slipage:
                next_exit_value_stoploss = np.minimum(close_prices[stoploss_index], stoploss_prices)
            else:
                next_exit_value_stoploss = stoploss_prices

        elif buy_or_sell == "sell":
            if stoploss_wait_candleclose:
                values1 = close_prices
                stoploss_prices = close_prices + stoploss_delta
            else:
                values1 = high_prices
                stoploss_prices = close_prices + stoploss_delta

            stoploss_index = monotonic_stack_for_value1_gt_value2(values1, stoploss_prices)
            stoploss_index = _replace_missing_exit_indices(stoploss_index, len(df) - 1)

            if stoploss_consider_slipage:
                next_exit_value_stoploss = np.maximum(close_prices[stoploss_index], stoploss_prices)
            else:
                next_exit_value_stoploss = stoploss_prices
        else:
            print(f"buy_or_sell :{buy_or_sell}")
            raise ValueError("Invalid value for 'buy_or_sell'. It must be either 'buy' or 'sell'.")

        df["stoploss_index"] = stoploss_index
        df["stoploss_price"] = stoploss_prices
        df["next_exit_value_stoploss"] = next_exit_value_stoploss

        next_exit_indices = np.minimum(next_exit_indices, stoploss_index)
        next_exit_value = np.where(next_exit_indices == stoploss_index, next_exit_value_stoploss, next_exit_value)

    next_exit_datetime = array_lib.asarray(df["datetime"].to_numpy())[next_exit_indices]
    df["next_exit_datetime"] = next_exit_datetime

    if buy_or_sell == "buy":
        trading_cost = next_exit_value * trading_cost_factor
        next_exit_profit = next_exit_value - close_prices - trading_cost
        next_exit_capital_multiplier_in_percent = 100.0 + (leverage * next_exit_profit * 100) / close_prices
    elif buy_or_sell == "sell":
        trading_cost = close_prices * trading_cost_factor
        next_exit_profit = close_prices - next_exit_value - trading_cost
        next_exit_capital_multiplier_in_percent = 100.0 + (leverage * next_exit_profit * 100) / close_prices
    else:
        raise ValueError("Invalid value for 'buy_or_sell'. It must be either 'buy' or 'sell'.")

    df["next_exit_index"] = next_exit_indices
    df["next_exit_value"] = next_exit_value
    df["next_exit_tradingcost"] = trading_cost
    df["next_exit_profit"] = next_exit_profit
    df["next_exit_capital_multiplier_in_percent"] = next_exit_capital_multiplier_in_percent

    return df
