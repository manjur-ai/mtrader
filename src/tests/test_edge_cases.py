import numpy as np
import pandas as pd
import builtins


def _tiny_df():
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-02 09:15", periods=30, freq="min"),
        "open": np.linspace(15000, 15100, 30),
        "high": np.linspace(15020, 15120, 30),
        "low": np.linspace(14980, 15080, 30),
        "close": np.linspace(15000, 15100, 30),
        "volume": np.full(30, 1000),
    })


def test_clean_data_empty_df():
    from mtrader import clean_data
    df = pd.DataFrame()
    try:
        clean_data(df, stopprint=True)
        assert False, "Should have raised"
    except (ValueError, KeyError, AttributeError):
        pass
    builtins.print("  edge empty_df: OK")


def test_clean_data_few_columns():
    from mtrader import clean_data
    df = pd.DataFrame({"a": [1, 2]})
    try:
        clean_data(df, stopprint=True)
        assert False, "Should have raised"
    except (ValueError, KeyError, AttributeError):
        pass
    builtins.print("  edge few_cols: OK")


def test_find_best_exit_empty_grid():
    from mtrader import find_best_exit
    df = _tiny_df()
    try:
        find_best_exit(df, [[{"first_column_name": "close", "second_column_name": "close",
                              "shift_down_first": 0, "shift_down_second": 0,
                              "lower_range_of_difference": 0, "upper_range_of_difference": 0,
                              "perform_normalization_of_diff": False}]],
                       buy_or_sell="buy")
        assert False, "Should have raised"
    except ValueError:
        pass
    builtins.print("  edge empty_grid: OK")


def test_trading_no_conditions():
    from mtrader import take_trade_on_condition_numpy
    df = _tiny_df()
    df["next_exit_index"] = np.arange(30)
    df["next_exit_capital_multiplier_in_percent"] = 100.0
    try:
        take_trade_on_condition_numpy(df, [], leverage=1, initial_capital=1000)
        assert False, "Should have raised"
    except (ValueError, IndexError, TypeError):
        pass
    builtins.print("  edge no_conditions: OK")


def test_validate_ohlcv():
    from mtrader import validate_ohlcv
    df = _tiny_df()
    assert validate_ohlcv(df) is True
    bad = df.drop(columns=["high"])
    try:
        validate_ohlcv(bad)
        assert False, "Should have raised"
    except ValueError:
        pass
    builtins.print("  edge validate_ohlcv: OK")


def test_backtest_report_no_trade_columns():
    from mtrader import backtest_report
    df = _tiny_df()
    try:
        backtest_report(df)
        assert False, "Should have raised"
    except ValueError:
        pass
    builtins.print("  edge report_no_cols: OK")


def test_run_backtest_empty_df():
    from mtrader import run_backtest
    df = pd.DataFrame()
    try:
        run_backtest(df, [[{"first_column_name": "close", "second_column_name": "close",
                            "shift_down_first": 0, "shift_down_second": 0,
                            "lower_range_of_difference": 0, "upper_range_of_difference": 0,
                            "perform_normalization_of_diff": False}]],
                     indicators=["sma1"], rolling_minutes=[5])
        assert False, "Should have raised"
    except (ValueError, KeyError):
        pass
    builtins.print("  edge run_empty: OK")


def test_run_backtest_bad_side():
    from mtrader import run_backtest
    df = _tiny_df()
    try:
        run_backtest(df, [[{"first_column_name": "close", "second_column_name": "close",
                            "shift_down_first": 0, "shift_down_second": 0,
                            "lower_range_of_difference": 0, "upper_range_of_difference": 0,
                            "perform_normalization_of_diff": False}]],
                     buy_or_sell="invalid")
        assert False, "Should have raised"
    except ValueError:
        pass
    builtins.print("  edge bad_side: OK")


def test_monotonic_stack_edge():
    from mtrader import monotonic_stack_for_value1_gt_value2, monotonic_stack_for_value1_lessthan_value2
    empty = np.array([], dtype=np.float64)
    r1 = monotonic_stack_for_value1_gt_value2(empty, empty)
    assert len(r1) == 0
    r2 = monotonic_stack_for_value1_lessthan_value2(empty, empty)
    assert len(r2) == 0
    single = np.array([100.0], dtype=np.float64)
    r3 = monotonic_stack_for_value1_gt_value2(single, single)
    assert r3[0] == -1
    builtins.print("  edge monotonic_stack: OK")


def test_indicators_nan_handling():
    from mtrader import rsi, atr, cci, willr, mfi
    nan_arr = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    r = rsi(nan_arr, 3)
    assert np.all(np.isnan(r))
    h = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    l_arr = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    c = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    a = atr(h, l_arr, c, 3)
    assert np.all(np.isnan(a))
    builtins.print("  edge NaN_indicators: OK")


if __name__ == "__main__":
    test_clean_data_empty_df()
    test_clean_data_few_columns()
    test_find_best_exit_empty_grid()
    test_trading_no_conditions()
    test_validate_ohlcv()
    test_backtest_report_no_trade_columns()
    test_run_backtest_empty_df()
    test_run_backtest_bad_side()
    test_monotonic_stack_edge()
    test_indicators_nan_handling()
    builtins.print("\nAll edge case tests passed!")
