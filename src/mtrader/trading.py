import numpy as np
import pandas as pd
from itertools import product

try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False

def take_trade_on_condition(df, conditions, cupy=False, leverage=1, initial_capital=1000, risk_free_rate=0):
    if cupy and has_cupy:
        array_lib = cp
    else:
        array_lib = np

    condition_met_all = array_lib.ones(len(df), dtype=bool)
    for condition_group in conditions:
        condition_met_group = array_lib.ones(len(df), dtype=bool)
        for condition in condition_group:
            shifted_first = array_lib.asarray(
                df[condition["first_column_name"]].shift(condition["shift_down_first"]).fillna(0).to_numpy()
            )
            shifted_second = array_lib.asarray(
                df[condition["second_column_name"]].shift(condition["shift_down_second"]).fillna(0).to_numpy()
            )
            difference = shifted_first - shifted_second
            condition_met = (difference >= condition["lower_range_of_difference"]) & \
                            (difference <= condition["upper_range_of_difference"])
            condition_met_group &= condition_met
        condition_met_all &= condition_met_group

    next_exit_index = array_lib.asarray(df["next_exit_index"].to_numpy())
    valid_indices = array_lib.where(condition_met_all)[0]
    filtered_next_exit_index = next_exit_index[valid_indices]
    unique_exit_indices, first_occurrence = array_lib.unique(filtered_next_exit_index, return_index=True)
    valid_trade_indices = valid_indices[first_occurrence]

    take_trade = array_lib.zeros(len(df), dtype=bool)
    take_trade[valid_trade_indices] = True

    next_exit_capital_multiplier = array_lib.asarray(df["next_exit_capital_multiplier_in_percent"].to_numpy())

    capital_at_exit = array_lib.zeros(len(df), dtype=float)
    if len(valid_trade_indices) > 0:
        capital_at_exit[valid_trade_indices[0]] = initial_capital
        capital_multiplier = array_lib.ones(len(df), dtype=float)
        capital_multiplier[valid_trade_indices] = next_exit_capital_multiplier[valid_trade_indices] / 100
        cumulative_capital = array_lib.cumprod(capital_multiplier[valid_trade_indices]) * initial_capital
        capital_at_exit[valid_trade_indices] = cumulative_capital

    df["take_trade"] = take_trade.get() if cupy and has_cupy else take_trade
    df["capital_at_exit"] = capital_at_exit.get() if cupy and has_cupy else capital_at_exit

    df_filtered = df[df["take_trade"]] if array_lib.any(take_trade) else df.iloc[[]]

    final_capital = capital_at_exit[take_trade][-1] if array_lib.any(take_trade) else initial_capital

    capital_at_exit = capital_at_exit[capital_at_exit > 0]
    if len(capital_at_exit) > 1:
        log_capital_at_exit = array_lib.log(array_lib.clip(capital_at_exit, 1e-10, None))
        log_returns = log_capital_at_exit[1:] - log_capital_at_exit[:-1]
        volatility = array_lib.std(log_returns)
        mean_log_return = array_lib.mean(log_returns)
        sharpe_ratio = (mean_log_return - risk_free_rate) / volatility if volatility > 0 else float('nan')
        peak_capital = array_lib.maximum.accumulate(capital_at_exit)
        drawdowns = (peak_capital - capital_at_exit) / peak_capital
        max_drawdown = array_lib.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
    else:
        volatility = 0.0
        sharpe_ratio = float('nan')
        max_drawdown = 0.0

    metrics = {
        "Volatility": volatility.get() if cupy and has_cupy else volatility,
        "Sharpe Ratio": sharpe_ratio.get() if cupy and has_cupy else sharpe_ratio,
        "Max Drawdown": max_drawdown.get() if cupy and has_cupy else max_drawdown,
    }

    return df_filtered, final_capital, metrics


def take_trade_on_condition_numpy(df, conditions, leverage=1, initial_capital=1000, risk_free_rate=0):
    condition_met_all = np.ones(len(df), dtype=bool)
    for condition_group in conditions:
        condition_met_group = np.ones(len(df), dtype=bool)
        for condition in condition_group:
            shifted_first = df[condition["first_column_name"]].shift(condition["shift_down_first"]).fillna(0).to_numpy()
            shifted_second = df[condition["second_column_name"]].shift(condition["shift_down_second"]).fillna(0).to_numpy()
            difference = shifted_first - shifted_second
            condition_met = (difference >= condition["lower_range_of_difference"]) & \
                            (difference <= condition["upper_range_of_difference"])
            condition_met_group &= condition_met
        condition_met_all &= condition_met_group

    next_exit_index = df["next_exit_index"].to_numpy()
    valid_indices = np.where(condition_met_all)[0]
    filtered_next_exit_index = next_exit_index[valid_indices]
    unique_exit_indices, first_occurrence = np.unique(filtered_next_exit_index, return_index=True)
    valid_trade_indices = valid_indices[first_occurrence]

    take_trade = np.zeros(len(df), dtype=bool)
    take_trade[valid_trade_indices] = True

    next_exit_capital_multiplier = df["next_exit_capital_multiplier_in_percent"].to_numpy()
    capital_at_exit = np.zeros(len(df), dtype=float)
    if len(valid_trade_indices) > 0:
        capital_at_exit[valid_trade_indices[0]] = initial_capital
        capital_multiplier = np.ones(len(df), dtype=float)
        capital_multiplier[valid_trade_indices] = next_exit_capital_multiplier[valid_trade_indices] / 100
        cumulative_capital = np.cumprod(capital_multiplier[valid_trade_indices]) * initial_capital
        capital_at_exit[valid_trade_indices] = cumulative_capital

    df["take_trade"] = take_trade
    df["capital_at_exit"] = capital_at_exit

    df_filtered = df[df["take_trade"]] if np.any(take_trade) else df.iloc[[]]
    final_capital = capital_at_exit[take_trade][-1] if np.any(take_trade) else initial_capital

    positive_capital = capital_at_exit[capital_at_exit > 0]
    if len(positive_capital) > 1:
        log_capital_at_exit = np.log(np.clip(positive_capital, 1e-10, None))
        log_returns = log_capital_at_exit[1:] - log_capital_at_exit[:-1]
        volatility = np.std(log_returns)
        mean_log_return = np.mean(log_returns)
        sharpe_ratio = (mean_log_return - risk_free_rate) / volatility if volatility > 0 else float('nan')
        cumulative_max = np.maximum.accumulate(log_capital_at_exit)
        drawdowns = cumulative_max - log_capital_at_exit
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
    else:
        volatility = 0.0
        sharpe_ratio = float('nan')
        max_drawdown = 0.0

    metrics = {
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
    }

    return df_filtered, final_capital, metrics


def take_trade_on_condition2(
    df_cupy, conditions, leverage=1, initial_capital=1000, risk_free_rate=0, calculation_start_index=0
):
    n_rows = df_cupy["close"].shape[0]

    for condition_group in conditions:
        for condition in condition_group:
            if condition["shift_down_first"] > calculation_start_index or condition["shift_down_second"] > calculation_start_index:
                raise ValueError(
                    f"Shift values ({condition['shift_down_first']}, {condition['shift_down_second']}) exceed calculation_start_index ({calculation_start_index})."
                )

    condition_met_all = cp.ones(n_rows - calculation_start_index, dtype=cp.bool_)

    for condition_group in conditions:
        condition_met_group = cp.ones(n_rows - calculation_start_index, dtype=cp.bool_)
        for condition in condition_group:
            col1_data = df_cupy[condition["first_column_name"]]
            col2_data = df_cupy[condition["second_column_name"]]

            shifted_first = col1_data[
                calculation_start_index - condition["shift_down_first"]: n_rows - condition["shift_down_first"]
            ]
            shifted_second = col2_data[
                calculation_start_index - condition["shift_down_second"]: n_rows - condition["shift_down_second"]
            ]

            difference = shifted_first - shifted_second
            condition_met = (difference >= condition["lower_range_of_difference"]) & \
                            (difference <= condition["upper_range_of_difference"])
            condition_met_group &= condition_met

        condition_met_all &= condition_met_group

    next_exit_index = df_cupy["next_exit_index"][calculation_start_index:]
    valid_indices = cp.where(condition_met_all)[0]
    filtered_next_exit_index = next_exit_index[valid_indices]
    unique_exit_indices, first_occurrence = cp.unique(filtered_next_exit_index, return_index=True)
    valid_trade_indices = valid_indices[first_occurrence]

    if len(valid_trade_indices) == 0:
        return None, None, None

    take_trade = cp.zeros(n_rows, dtype=cp.bool_)
    take_trade[valid_trade_indices + calculation_start_index] = True

    next_exit_capital_multiplier = df_cupy["next_exit_capital_multiplier_in_percent"]
    capital_at_exit = cp.zeros(n_rows, dtype=cp.float32)

    if len(valid_trade_indices) > 0:
        capital_at_exit[valid_trade_indices[0] + calculation_start_index] = initial_capital
        capital_multiplier = cp.ones(n_rows, dtype=cp.float32)
        capital_multiplier[valid_trade_indices + calculation_start_index] = (
            next_exit_capital_multiplier[valid_trade_indices + calculation_start_index] / 100
        )
        cumulative_capital = cp.cumprod(capital_multiplier[valid_trade_indices + calculation_start_index]) * initial_capital
        capital_at_exit[valid_trade_indices + calculation_start_index] = cumulative_capital

    df_cupy["take_trade"] = take_trade
    df_cupy["capital_at_exit"] = capital_at_exit

    df_filtered = {key: value[take_trade] for key, value in df_cupy.items()} if cp.any(take_trade) else {}
    final_capital = capital_at_exit[take_trade][-1] if cp.any(take_trade) else initial_capital

    log_capital_at_exit = cp.log(cp.clip(capital_at_exit[capital_at_exit > 0], 1e-10, None))
    log_returns = log_capital_at_exit[1:] - log_capital_at_exit[:-1]
    volatility = cp.std(log_returns)
    mean_log_return = cp.mean(log_returns)
    sharpe_ratio = (mean_log_return - risk_free_rate) / volatility if volatility > 0 else float("nan")

    cumulative_max = cp.zeros_like(log_capital_at_exit)
    cumulative_max[0] = log_capital_at_exit[0]
    for i in range(1, len(log_capital_at_exit)):
        cumulative_max[i] = cp.maximum(cumulative_max[i - 1], log_capital_at_exit[i])

    drawdowns = cumulative_max - log_capital_at_exit
    max_drawdown = cp.max(drawdowns) if len(drawdowns) > 0 else 0

    metrics = {
        "Volatility": float(volatility),
        "Sharpe Ratio": float(sharpe_ratio),
        "Max Drawdown": float(max_drawdown),
    }

    return df_filtered, float(final_capital), metrics


def take_trade_on_condition3(
    df_cupy, conditions, leverage=1, initial_capital=1000, risk_free_rate=0, calculation_start_index=0
):
    n_rows = df_cupy["close"].shape[0]

    for condition_group in conditions:
        for condition in condition_group:
            if condition["shift_down_first"] > calculation_start_index or condition["shift_down_second"] > calculation_start_index:
                raise ValueError(
                    f"Shift values ({condition['shift_down_first']}, {condition['shift_down_second']}) exceed calculation_start_index ({calculation_start_index})."
                )

    condition_met_all = cp.ones(n_rows - calculation_start_index, dtype=cp.bool_)

    for condition_group in conditions:
        condition_met_group = cp.ones(n_rows - calculation_start_index, dtype=cp.bool_)
        for condition in condition_group:
            col1_data = df_cupy[condition["first_column_name"]]
            col2_data = df_cupy[condition["second_column_name"]]

            shifted_first = col1_data[calculation_start_index - condition["shift_down_first"]: n_rows - condition["shift_down_first"]]
            shifted_second = col2_data[calculation_start_index - condition["shift_down_second"]: n_rows - condition["shift_down_second"]]

            difference = shifted_first - shifted_second
            condition_met = (difference >= condition["lower_range_of_difference"]) & (difference <= condition["upper_range_of_difference"])
            condition_met_group &= condition_met

        condition_met_all &= condition_met_group

    valid_indices = cp.where(condition_met_all)[0]

    if len(valid_indices) == 0:
        return None, None, None

    capital_at_exit = cp.zeros(n_rows, dtype=cp.float32)
    capital_multiplier = cp.ones(n_rows, dtype=cp.float32)
    capital_multiplier[valid_indices + calculation_start_index] = df_cupy["next_exit_capital_multiplier_in_percent"][valid_indices + calculation_start_index] / 100
    cumulative_capital = cp.cumprod(capital_multiplier[valid_indices + calculation_start_index]) * initial_capital
    capital_at_exit[valid_indices + calculation_start_index] = cumulative_capital

    df_cupy["take_trade"] = cp.zeros(n_rows, dtype=cp.bool_)
    df_cupy["take_trade"][valid_indices + calculation_start_index] = True
    df_cupy["capital_at_exit"] = capital_at_exit

    df_filtered = {key: value[cp.where(df_cupy["take_trade"])[0]] for key, value in df_cupy.items()}
    final_capital = capital_at_exit[cp.where(df_cupy["take_trade"])[0][-1]] if cp.any(df_cupy["take_trade"]) else initial_capital

    log_capital_at_exit = cp.log(cp.clip(capital_at_exit[capital_at_exit > 0], 1e-10, None))
    log_returns = log_capital_at_exit[1:] - log_capital_at_exit[:-1]
    volatility = cp.std(log_returns)
    mean_log_return = cp.mean(log_returns)
    sharpe_ratio = (mean_log_return - risk_free_rate) / volatility if volatility > 0 else float("nan")

    cumulative_max = cp.zeros_like(log_capital_at_exit)
    cumulative_max[0] = log_capital_at_exit[0]
    for i in range(1, len(log_capital_at_exit)):
        cumulative_max[i] = cp.maximum(cumulative_max[i - 1], log_capital_at_exit[i])

    drawdowns = cumulative_max - log_capital_at_exit
    max_drawdown = cp.max(drawdowns) if len(drawdowns) > 0 else 0

    metrics = {
        "Volatility": float(volatility),
        "Sharpe Ratio": float(sharpe_ratio),
        "Max Drawdown": float(max_drawdown),
    }

    return df_filtered, float(final_capital), metrics


def calculate_difference_for_columns(df_cupy, column1, column2, shift1, shift2, calculation_start_index, n_rows, perform_normalization=False):
    col1_data = df_cupy[column1]
    col2_data = df_cupy[column2]

    shifted_first = col1_data[calculation_start_index - shift1: n_rows - shift1]
    shifted_second = col2_data[calculation_start_index - shift2: n_rows - shift2]

    difference = shifted_first - shifted_second

    if perform_normalization:
        close_data = df_cupy["close"]
        close_data_slice = close_data[calculation_start_index: n_rows]
        difference *= 10000 / close_data_slice

    min_difference = cp.min(difference)
    max_difference = cp.max(difference)

    return min_difference, max_difference


def take_trade_on_condition2_for_all_ranges(df_cupy, conditions, lower_ranges, upper_ranges, column1, column2, shift1, shift2, perform_normalization, leverage=1, initial_capital=1000, risk_free_rate=0, calculation_start_index=0):
    n_rows = df_cupy["close"].shape[0]

    min_difference, max_difference = calculate_difference_for_columns(
        df_cupy, column1, column2, shift1, shift2, calculation_start_index, n_rows, perform_normalization
    )

    valid_range_combinations = []
    for lower_range in lower_ranges:
        if lower_range < min_difference:
            continue
        for upper_range in upper_ranges:
            if upper_range > max_difference:
                continue
            valid_range_combinations.append((lower_range, upper_range))

    if not valid_range_combinations:
        return None

    all_results = {}

    for lower_range, upper_range in valid_range_combinations:
        condition_met_all = cp.ones(n_rows - calculation_start_index, dtype=cp.bool_)

        for condition_group in conditions:
            condition_met_group = cp.ones(n_rows - calculation_start_index, dtype=cp.bool_)

            for condition in condition_group:
                col1_data = df_cupy[condition["first_column_name"]]
                col2_data = df_cupy[condition["second_column_name"]]

                shifted_first = col1_data[calculation_start_index - condition["shift_down_first"]: n_rows - condition["shift_down_first"]]
                shifted_second = col2_data[calculation_start_index - condition["shift_down_second"]: n_rows - condition["shift_down_second"]]

                difference = shifted_first - shifted_second

                if condition["perform_normalization_of_diff"]:
                    close_data = df_cupy["close"][calculation_start_index: n_rows]
                    difference *= 10000 / close_data

                condition_met = (difference >= lower_range) & (difference <= upper_range)
                condition_met_group &= condition_met

            condition_met_all &= condition_met_group

            valid_indices = cp.where(condition_met_all)[0]

            if len(valid_indices) == 0:
                all_results[(lower_range, upper_range)] = (None, None, None)
                continue

            take_trade = cp.zeros(n_rows, dtype=cp.bool_)
            take_trade[valid_indices + calculation_start_index] = True

            next_exit_capital_multiplier = df_cupy["next_exit_capital_multiplier_in_percent"]
            capital_at_exit = cp.zeros(n_rows, dtype=cp.float32)

            capital_multiplier = cp.ones(n_rows, dtype=cp.float32)
            capital_multiplier[valid_indices + calculation_start_index] = next_exit_capital_multiplier[valid_indices + calculation_start_index] / 100
            cumulative_capital = cp.cumprod(capital_multiplier[valid_indices + calculation_start_index]) * initial_capital
            capital_at_exit[valid_indices + calculation_start_index] = cumulative_capital

            df_cupy["take_trade"] = take_trade
            df_cupy["capital_at_exit"] = capital_at_exit

            df_filtered = {key: value[take_trade] for key, value in df_cupy.items()} if cp.any(take_trade) else {}
            final_capital = capital_at_exit[take_trade][-1] if cp.any(take_trade) else initial_capital

            log_capital_at_exit = cp.log(cp.clip(capital_at_exit[capital_at_exit > 0], 1e-10, None))
            log_returns = log_capital_at_exit[1:] - log_capital_at_exit[:-1]
            volatility = cp.std(log_returns)
            mean_log_return = cp.mean(log_returns)
            sharpe_ratio = (mean_log_return - risk_free_rate) / volatility if volatility > 0 else float("nan")

            cumulative_max = cp.maximum.accumulate(log_capital_at_exit)
            drawdowns = cumulative_max - log_capital_at_exit
            max_drawdown = cp.max(drawdowns) if len(drawdowns) > 0 else 0

            metrics = {
                "Volatility": float(volatility),
                "Sharpe Ratio": float(sharpe_ratio),
                "Max Drawdown": float(max_drawdown),
            }

            all_results[(lower_range, upper_range)] = (df_filtered, float(final_capital), metrics)

        return all_results


def take_trade_on_condition_vectorized(
    df_cupy, conditions, lower_ranges, upper_ranges, leverage=1, initial_capital=1000, risk_free_rate=0, calculation_start_index=0
):
    n_rows = df_cupy["close"].shape[0]
    range_combinations = cp.array(list(product(lower_ranges, upper_ranges)))
    n_combinations = range_combinations.shape[0]

    precomputed_shifts = {}
    for condition_group in conditions:
        for condition in condition_group:
            col1_data = df_cupy[condition["first_column_name"]]
            col2_data = df_cupy[condition["second_column_name"]]
            shift_down_first = condition["shift_down_first"]
            shift_down_second = condition["shift_down_second"]

            if (condition["first_column_name"], shift_down_first) not in precomputed_shifts:
                precomputed_shifts[(condition["first_column_name"], shift_down_first)] = col1_data[
                    calculation_start_index - shift_down_first: n_rows - shift_down_first
                ]
            if (condition["second_column_name"], shift_down_second) not in precomputed_shifts:
                precomputed_shifts[(condition["second_column_name"], shift_down_second)] = col2_data[
                    calculation_start_index - shift_down_second: n_rows - shift_down_second
                ]

    condition_met_all = cp.ones((n_rows - calculation_start_index, n_combinations), dtype=cp.bool_)

    for condition_group in conditions:
        condition_met_group = cp.ones((n_rows - calculation_start_index, n_combinations), dtype=cp.bool_)

        for condition in condition_group:
            shifted_first = precomputed_shifts[(condition["first_column_name"], condition["shift_down_first"])]
            shifted_second = precomputed_shifts[(condition["second_column_name"], condition["shift_down_second"])]

            difference = (shifted_first - shifted_second)
            if condition["perform_normalization_of_diff"]:
                close_data = df_cupy["close"][calculation_start_index: n_rows]
                difference *= 10000 / close_data

            if condition["find_best_cond"]:
                lower_ranges_comb, upper_ranges_comb = range_combinations[:, 0], range_combinations[:, 1]
                lower_ranges_comb = lower_ranges_comb.reshape(1, -1)
                upper_ranges_comb = upper_ranges_comb.reshape(1, -1)
                difference_2d = difference.reshape(-1, 1)
                condition_met = (difference_2d >= lower_ranges_comb) & (difference_2d <= upper_ranges_comb)
            else:
                condition_met_singlerange = (difference >= condition["lower_range_of_difference"]) & (difference <= condition["upper_range_of_difference"])
                condition_met = cp.tile(condition_met_singlerange[:, cp.newaxis], (1, n_combinations))

            condition_met_group &= condition_met

        condition_met_all &= condition_met_group

    valid_indices = cp.where(condition_met_all)
    combination_indices = valid_indices[1]
    filtered_exit_indices = valid_indices[0]

    stacked_indices = cp.stack((filtered_exit_indices, combination_indices))
    sorted_order = cp.lexsort(stacked_indices)
    sorted_combinations = combination_indices[sorted_order]
    sorted_exit_indices = filtered_exit_indices[sorted_order]

    unique_mask = cp.ones_like(sorted_exit_indices, dtype=cp.bool_)
    unique_mask[1:] = (sorted_combinations[1:] != sorted_combinations[:-1]) | (
        sorted_exit_indices[1:] != sorted_exit_indices[:-1]
    )

    unique_combinations = sorted_combinations[unique_mask]
    unique_exit_indices = sorted_exit_indices[unique_mask]

    valid_trade_indices = cp.zeros((n_rows, n_combinations), dtype=cp.bool_)
    valid_trade_indices[unique_exit_indices, unique_combinations] = True

    take_trade = valid_trade_indices

    next_exit_capital_multiplier = df_cupy["next_exit_capital_multiplier_in_percent"]
    capital_multiplier = cp.ones((n_rows, n_combinations), dtype=cp.float32)
    next_exit_capital_multiplier_expanded = cp.tile(next_exit_capital_multiplier[:, cp.newaxis], (1, n_combinations))
    capital_multiplier[take_trade] = next_exit_capital_multiplier_expanded[take_trade] / 100

    cumulative_capital = cp.cumprod(capital_multiplier, axis=0) * initial_capital
    final_capital = cumulative_capital[-1:]

    log_cumulative_capital = cp.log(cp.clip(cumulative_capital, 1e-10, None))
    log_returns = log_cumulative_capital[1:] - log_cumulative_capital[:-1]

    volatility = cp.std(log_returns, axis=0)
    mean_log_return = cp.mean(log_returns, axis=0)
    sharpe_ratio = (mean_log_return - risk_free_rate) / volatility

    metrics = {
        "Volatility": volatility.tolist(),
        "Sharpe Ratio": sharpe_ratio.tolist(),
        "Final Capital": final_capital.tolist(),
    }

    return metrics


def take_trade_on_condition_vectorized2(
    df_cupy, conditions, lower_ranges, upper_ranges, leverage=1, initial_capital=1000, risk_free_rate=0, calculation_start_index=0
):
    n_rows = df_cupy["close"].shape[0]
    range_combinations = cp.array(list(product(lower_ranges, upper_ranges)), dtype=cp.float32)
    n_combinations = range_combinations.shape[0]

    precomputed_shifts = {}
    for condition_group in conditions:
        for condition in condition_group:
            col1_data = df_cupy[condition["first_column_name"]]
            col2_data = df_cupy[condition["second_column_name"]]
            shift_down_first = condition["shift_down_first"]
            shift_down_second = condition["shift_down_second"]

            if (condition["first_column_name"], shift_down_first) not in precomputed_shifts:
                precomputed_shifts[(condition["first_column_name"], shift_down_first)] = col1_data[
                    calculation_start_index - shift_down_first: n_rows - shift_down_first
                ]
            if (condition["second_column_name"], shift_down_second) not in precomputed_shifts:
                precomputed_shifts[(condition["second_column_name"], shift_down_second)] = col2_data[
                    calculation_start_index - shift_down_second: n_rows - shift_down_second
                ]

    condition_met_all = cp.ones((n_rows - calculation_start_index, n_combinations), dtype=cp.bool_)

    for condition_group in conditions:
        condition_met_group = cp.ones((n_rows - calculation_start_index, n_combinations), dtype=cp.bool_)

        for condition in condition_group:
            shifted_first = precomputed_shifts[(condition["first_column_name"], condition["shift_down_first"])]
            shifted_second = precomputed_shifts[(condition["second_column_name"], condition["shift_down_second"])]

            difference = shifted_first - shifted_second
            if condition["perform_normalization_of_diff"]:
                close_data = df_cupy["close"][calculation_start_index: n_rows]
                difference *= 10000 / close_data

            if condition["find_best_cond"]:
                lower_ranges_comb, upper_ranges_comb = range_combinations[:, 0], range_combinations[:, 1]
                difference_2d = difference[:, cp.newaxis]
                condition_met = (difference_2d >= lower_ranges_comb) & (difference_2d <= upper_ranges_comb)
            else:
                condition_met_singlerange = (difference >= condition["lower_range_of_difference"]) & (
                    difference <= condition["upper_range_of_difference"]
                )
                condition_met = cp.broadcast_to(condition_met_singlerange[:, cp.newaxis], (n_rows - calculation_start_index, n_combinations))

            condition_met_group &= condition_met

        condition_met_all &= condition_met_group

    valid_indices = cp.where(condition_met_all)
    unique_indices = cp.stack((valid_indices[1], valid_indices[0]), axis=1)
    unique_combinations, unique_exit_indices = cp.unique(unique_indices, axis=0).T

    take_trade = cp.zeros((n_rows, n_combinations), dtype=cp.bool_)
    take_trade[unique_exit_indices, unique_combinations] = True

    next_exit_capital_multiplier = df_cupy["next_exit_capital_multiplier_in_percent"]
    next_exit_capital_multiplier_expanded = cp.expand_dims(next_exit_capital_multiplier, axis=1)

    capital_multiplier = cp.ones((n_rows, n_combinations), dtype=cp.float32)
    capital_multiplier[take_trade] = next_exit_capital_multiplier_expanded[take_trade] / 100

    cumulative_capital = cp.cumprod(capital_multiplier, axis=0, dtype=cp.float32) * initial_capital
    final_capital = cumulative_capital[-1, :]

    log_cumulative_capital = cp.log(cp.clip(cumulative_capital, 1e-10, None))
    log_returns = log_cumulative_capital[1:] - log_cumulative_capital[:-1]

    volatility = cp.std(log_returns, axis=0, dtype=cp.float32)
    mean_log_return = cp.mean(log_returns, axis=0, dtype=cp.float32)
    sharpe_ratio = (mean_log_return - risk_free_rate) / volatility

    metrics = {
        "Volatility": volatility.tolist(),
        "Sharpe Ratio": sharpe_ratio.tolist(),
        "Final Capital": final_capital.tolist(),
    }

    return metrics


def update_cond(data, new_first_column_name, new_second_column_name,
                shift1=None, shift2=None,
                lower=None, upper=None,
                norm=None):
    updates = {
        "first_column_name": new_first_column_name,
        "second_column_name": new_second_column_name,
    }
    if shift1 is not None:
        updates["shift_down_first"] = shift1
    if shift2 is not None:
        updates["shift_down_second"] = shift2
    if lower is not None:
        updates["lower_range_of_difference"] = lower
    if upper is not None:
        updates["upper_range_of_difference"] = upper
    if norm is not None:
        updates["perform_normalization_of_diff"] = norm

    if isinstance(data, list):
        return [update_cond(item, new_first_column_name, new_second_column_name,
                            shift1, shift2, lower, upper, norm) for item in data]
    elif isinstance(data, dict):
        if data.get("first_column_name") == "ind_fast":
            data.update(updates)
        return {key: update_cond(value, new_first_column_name, new_second_column_name,
                                 shift1, shift2, lower, upper, norm) for key, value in data.items()}
    else:
        return data
