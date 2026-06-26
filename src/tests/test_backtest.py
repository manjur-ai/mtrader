import numpy as np
import pandas as pd
import builtins


def _gen_ohlc(n_candles=200):
    np.random.seed(42)
    base = 15000.0
    closes = []
    for i in range(n_candles):
        step = np.random.normal(0, 30)
        base += step
        if base < 100:
            base = 100
        closes.append(round(base, 2))
    close = np.array(closes)
    high = close + np.random.uniform(5, 40, n_candles)
    low = close - np.random.uniform(5, 40, n_candles)
    open_p = close + np.random.uniform(-20, 20, n_candles)
    volume = np.random.randint(100, 5000, n_candles)
    dti = pd.date_range("2024-01-02 09:15", periods=n_candles, freq="min")
    df = pd.DataFrame({
        "datetime": dti,
        "open": open_p.round(2),
        "high": high.round(2),
        "low": low.round(2),
        "close": close.round(2),
        "volume": volume,
    })
    return df


def test_clean_data_roundtrip():
    from mtrader import clean_data

    raw = _gen_ohlc(100)
    csv_lines = ["datetime,open,high,low,close,volume"]
    for _, r in raw.iterrows():
        csv_lines.append(
            f"{r['datetime'].strftime('%Y-%m-%d %H:%M:%S')},"
            f"{r['open']},{r['high']},{r['low']},{r['close']},{r['volume']}"
        )
    import io
    csv_buf = io.StringIO("\n".join(csv_lines))
    df_in = pd.read_csv(csv_buf)

    cleaned = clean_data(df_in, start_time="09:15", end_time="15:30",
                         start_date="2024-01-02", end_date="2024-01-02",
                         stopprint=True)

    assert "datetime" in cleaned.columns
    assert "open" in cleaned.columns
    assert "high" in cleaned.columns
    assert "low" in cleaned.columns
    assert "close" in cleaned.columns
    assert "volume" in cleaned.columns
    assert len(cleaned) <= len(raw)
    assert cleaned["close"].isna().sum() == 0
    builtins.print(f"  clean_data: {len(raw)} -> {len(cleaned)} rows, OK")


def test_add_indicators_basic():
    from mtrader import add_indicators

    df = _gen_ohlc(200)
    result = add_indicators(df, add=["sma1", "close"], rolling_minutes=[5])

    col = "can1_sma1_p5"
    assert col in result.columns, f"Missing column: {col}"
    assert result[col].notna().sum() > 0

    first_5_close = df["close"].iloc[:5].mean()
    assert abs(result[col].iloc[4] - first_5_close) < 0.01, \
        f"SMA mismatch at index 4: {result[col].iloc[4]} vs {first_5_close}"
    builtins.print(f"  add_indicators SMA: {col} computed OK")


def test_add_indicators_ema():
    from mtrader import add_indicators

    df = _gen_ohlc(200)
    result = add_indicators(df, add=["ema1", "close"], rolling_minutes=[10])

    col = "can1_ema1_p10"
    assert col in result.columns, f"Missing column: {col}"
    assert result[col].notna().sum() > 0
    builtins.print(f"  add_indicators EMA: {col} computed OK")


def test_add_indicators_distance():
    from mtrader import add_indicators

    df = _gen_ohlc(200)
    result = add_indicators(df, add=["smadis1", "close"], rolling_minutes=[14])

    sma_col = "can1_sma1_p14"
    dis_col = "can1_smadis1_p14"
    assert sma_col in result.columns, f"Missing dependency: {sma_col}"
    builtins.print(f"  add_indicators distance dependency: {sma_col} auto-resolved OK")


def test_add_indicators_vwap():
    from mtrader import add_indicators

    df = _gen_ohlc(200)
    result = add_indicators(df, add=["vwap"], rolling_minutes=[])

    col = "can1_vwap"
    assert col in result.columns
    assert result[col].notna().sum() > 0
    builtins.print(f"  add_indicators VWAP: {col} computed OK")


def test_precalculate_exit():
    from mtrader import add_indicators, precalculate_exit_time_amount_profit

    df = _gen_ohlc(200)
    df = add_indicators(df, add=["sma1", "close", "high", "low", "volume"], rolling_minutes=[5])

    conditions = [[
        {
            "first_column_name": "can1_sma1_p5",
            "second_column_name": "close",
            "shift_down_first": 0,
            "shift_down_second": 0,
            "lower_range_of_difference": -np.inf,
            "upper_range_of_difference": -50,
            "perform_normalization_of_diff": False,
        }
    ]]

    result = precalculate_exit_time_amount_profit(
        df, conditions, buy_or_sell="buy",
        target_delta=200, stoploss_delta=100,
    )

    assert "exit_signal" in result.columns
    assert "next_exit_index" in result.columns
    assert "next_exit_value" in result.columns
    assert "next_exit_profit" in result.columns
    assert result["next_exit_index"].notna().all()
    builtins.print(f"  precalculate_exit: {result['exit_signal'].sum()} exit signals generated, "
                   f"last index={result['next_exit_index'].iloc[-1]}")


def test_precalculate_exit_normalized():
    from mtrader import add_indicators, precalculate_exit_time_amount_profit

    df = _gen_ohlc(200)
    df = add_indicators(df, add=["sma1", "close", "high", "low", "volume"], rolling_minutes=[5])

    conditions = [[
        {
            "first_column_name": "close",
            "second_column_name": "can1_sma1_p5",
            "shift_down_first": 0,
            "shift_down_second": 0,
            "lower_range_of_difference": 50,
            "upper_range_of_difference": np.inf,
            "perform_normalization_of_diff": False,
        }
    ]]

    result = precalculate_exit_time_amount_profit(
        df, conditions, buy_or_sell="buy",
        target_delta_normalized=1.0,
        stoploss_delta_normalized=0.5,
    )

    assert "target_price" in result.columns
    assert "stoploss_price" in result.columns
    builtins.print(f"  precalculate_exit normalized: OK")


def _prepare_backtest_df():
    from mtrader import add_indicators
    df = _gen_ohlc(200)
    df = add_indicators(df, add=["sma1", "close", "high", "low", "volume"], rolling_minutes=[5])
    return df


def test_take_trade_on_condition_numpy():
    from mtrader import (
        precalculate_exit_time_amount_profit,
        take_trade_on_condition_numpy,
    )

    df = _prepare_backtest_df()

    conditions = [[
        {
            "first_column_name": "can1_sma1_p5",
            "second_column_name": "close",
            "shift_down_first": 0,
            "shift_down_second": 0,
            "lower_range_of_difference": -np.inf,
            "upper_range_of_difference": -30,
            "perform_normalization_of_diff": False,
        }
    ]]

    df = precalculate_exit_time_amount_profit(
        df, conditions, buy_or_sell="buy",
        target_delta=150, stoploss_delta=75,
    )

    df_filtered, final_capital, metrics = take_trade_on_condition_numpy(
        df, conditions, leverage=1, initial_capital=1000, risk_free_rate=0.05
    )

    assert "take_trade" in df.columns
    assert "capital_at_exit" in df.columns
    assert isinstance(final_capital, (int, float, np.floating))
    assert isinstance(metrics, dict)
    assert "Sharpe Ratio" in metrics
    assert "Max Drawdown" in metrics
    assert "Volatility" in metrics
    builtins.print(f"  take_trade_on_condition_numpy: {df['take_trade'].sum()} trades, "
                   f"final capital={final_capital:.2f}, sharpe={metrics['Sharpe Ratio']:.3f}")


def test_take_trade_on_condition():
    from mtrader import (
        precalculate_exit_time_amount_profit,
        take_trade_on_condition,
    )

    df = _prepare_backtest_df()

    conditions = [[
        {
            "first_column_name": "close",
            "second_column_name": "can1_sma1_p5",
            "shift_down_first": 0,
            "shift_down_second": 0,
            "lower_range_of_difference": 20,
            "upper_range_of_difference": np.inf,
            "perform_normalization_of_diff": False,
        }
    ]]

    df = precalculate_exit_time_amount_profit(
        df, conditions, buy_or_sell="sell",
        target_delta=100, stoploss_delta=50,
    )

    df_filtered, final_capital, metrics = take_trade_on_condition(
        df, conditions, cupy=False, leverage=2, initial_capital=5000, risk_free_rate=0.03
    )

    assert "take_trade" in df.columns
    assert df["take_trade"].sum() >= 0
    builtins.print(f"  take_trade_on_condition: {df['take_trade'].sum()} trades, "
                   f"final capital={final_capital:.2f}")


def test_empty_trade():
    from mtrader import (
        precalculate_exit_time_amount_profit,
        take_trade_on_condition_numpy,
    )

    df = _prepare_backtest_df()

    impossible_cond = [[
        {
            "first_column_name": "close",
            "second_column_name": "close",
            "shift_down_first": 0,
            "shift_down_second": 0,
            "lower_range_of_difference": 999999,
            "upper_range_of_difference": 999999,
            "perform_normalization_of_diff": False,
        }
    ]]

    df = precalculate_exit_time_amount_profit(
        df, impossible_cond, buy_or_sell="buy",
        target_delta=999999, stoploss_delta=999999,
    )

    df_filtered, final_capital, metrics = take_trade_on_condition_numpy(
        df, impossible_cond, leverage=1, initial_capital=1000
    )

    assert df["take_trade"].sum() == 0
    assert final_capital == 1000
    builtins.print(f"  empty_trade: no trades taken, capital preserved ({final_capital})")


def test_update_cond():
    from mtrader import update_cond

    conditions = [[
        {
            "first_column_name": "ind_fast",
            "second_column_name": "close",
            "shift_down_first": 0,
            "shift_down_second": 0,
            "lower_range_of_difference": -100,
            "upper_range_of_difference": 100,
            "perform_normalization_of_diff": False,
        }
    ]]

    updated = update_cond(conditions, "ind_slow", "ind_fast", shift1=1, lower=-50, upper=50)

    assert updated[0][0]["first_column_name"] == "ind_slow"
    assert updated[0][0]["second_column_name"] == "ind_fast"
    assert updated[0][0]["shift_down_first"] == 1
    assert updated[0][0]["lower_range_of_difference"] == -50
    assert updated[0][0]["upper_range_of_difference"] == 50
    builtins.print(f"  update_cond: recursively updated OK")


def test_monotonic_stack():
    from mtrader import monotonic_stack_for_value1_gt_value2, monotonic_stack_for_value1_lessthan_value2

    v1 = np.array([10, 20, 30, 40, 50], dtype=np.float64)
    v2 = np.array([25, 25, 25, 25, 25], dtype=np.float64)

    gt = monotonic_stack_for_value1_gt_value2(v1, v2)
    expected_gt = np.array([2, 2, 3, 4, -1], dtype=np.int32)
    assert np.array_equal(gt, expected_gt), f"gt mismatch: {gt} vs {expected_gt}"

    v1_lt = np.array([50, 40, 30, 20, 10], dtype=np.float64)
    lt = monotonic_stack_for_value1_lessthan_value2(v1_lt, v2)
    expected_lt = np.array([3, 3, 3, 4, -1], dtype=np.int32)
    assert np.array_equal(lt, expected_lt), f"lt mismatch: {lt} vs {expected_lt}"
    builtins.print(f"  monotonic_stack: all results match expected")


def test_timenum():
    from mtrader import timenum

    assert timenum("09:15") == 555
    assert timenum("15:30") == 930
    assert timenum("09:15:00") == 555
    builtins.print("  timenum: all conversions correct")


def test_detect_data_types():
    from mtrader import detect_data_types_with_formats

    df = pd.DataFrame({
        "dt": ["2024-01-02 09:15:00", "2024-01-02 09:16:00"],
        "price": [15000.5, 15100.0],
        "qty": [100, 200],
    })

    result = detect_data_types_with_formats(df)
    assert result["dt"]["type"] in ("datetime", "str")
    assert result["price"]["type"] == "float"
    assert result["qty"]["type"] == "int"
    builtins.print(f"  detect_data_types: {result}")


def test_indicators_numerical():
    from mtrader import ema, evol, wma, wvol, ssma, ssvol

    x = np.sin(np.linspace(0, 4 * np.pi, 100)) * 100 + 15000
    for name, fn in [("ema", ema), ("evol", evol), ("wma", wma),
                      ("wvol", wvol), ("ssma", ssma), ("ssvol", ssvol)]:
        out = fn(x, 14)
        assert len(out) == len(x)
        assert np.all(np.isfinite(out)), f"{name} produced non-finite values"
    builtins.print("  indicators: all 6 functions produce finite output")


def test_find_best_exit():
    from mtrader import find_best_exit

    df = _prepare_backtest_df()

    entry_conditions = [[
        {
            "first_column_name": "can1_sma1_p5",
            "second_column_name": "close",
            "shift_down_first": 0,
            "shift_down_second": 0,
            "lower_range_of_difference": -np.inf,
            "upper_range_of_difference": -30,
            "perform_normalization_of_diff": False,
        }
    ]]

    best_params, results_df = find_best_exit(
        df,
        entry_conditions=entry_conditions,
        buy_or_sell="buy",
        target_deltas=[50, 150],
        stoploss_deltas=[25, 75],
        leverage=1,
        initial_capital=1000,
        risk_free_rate=0.05,
        metric="sharpe",
        verbose=False,
    )

    assert best_params is not None
    assert "target_delta" in best_params
    assert "stoploss_delta" in best_params
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 4
    assert "sharpe" in results_df.columns

    best_params_cap, results_df_cap = find_best_exit(
        df,
        entry_conditions=entry_conditions,
        buy_or_sell="buy",
        target_deltas=[50, 150],
        stoploss_deltas=[25, 75],
        metric="final_capital",
    )

    assert best_params_cap is not None
    builtins.print(f"  find_best_exit (sharpe): best target={best_params['target_delta']}, "
                   f"stoploss={best_params['stoploss_delta']}")
    builtins.print(f"  find_best_exit (capital): best target={best_params_cap['target_delta']}, "
                   f"stoploss={best_params_cap['stoploss_delta']}")


def test_find_best_exit_normalized():
    from mtrader import find_best_exit

    df = _prepare_backtest_df()

    entry_conditions = [[
        {
            "first_column_name": "close",
            "second_column_name": "can1_sma1_p5",
            "shift_down_first": 0,
            "shift_down_second": 0,
            "lower_range_of_difference": 20,
            "upper_range_of_difference": np.inf,
            "perform_normalization_of_diff": False,
        }
    ]]

    best, results = find_best_exit(
        df,
        entry_conditions=entry_conditions,
        buy_or_sell="sell",
        target_deltas_normalized=[0.5, 1.0, 2.0],
        stoploss_deltas_normalized=[0.25, 0.5],
        metric="sharpe",
    )

    assert best is not None
    assert best["target_delta_normalized"] is not None
    assert len(results) == 6
    builtins.print(f"  find_best_exit normalized: best t_norm={best['target_delta_normalized']}, "
                   f"sl_norm={best['stoploss_delta_normalized']}")


def test_rsi():
    from mtrader import rsi
    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                      110, 108, 111, 113, 112, 114, 115, 113, 116, 118], dtype=np.float64)
    vals = rsi(close, 14)
    assert len(vals) == 20
    assert np.isnan(vals[13])
    assert not np.isnan(vals[14])
    assert 0 <= vals[14] <= 100
    builtins.print(f"  RSI: last value={vals[-1]:.2f}")


def test_atr():
    from mtrader import atr
    h = np.array([105, 107, 106, 108, 110], dtype=np.float64)
    l = np.array([95, 97, 96, 98, 100], dtype=np.float64)
    c = np.array([100, 102, 101, 103, 105], dtype=np.float64)
    vals = atr(h, l, c, 3)
    assert len(vals) == 5
    assert np.isnan(vals[0:2]).all()
    assert not np.isnan(vals[4])
    assert vals[4] > 0
    builtins.print(f"  ATR: value={vals[4]:.4f}")


def test_stoch():
    from mtrader import stoch_k, stoch_d
    h = np.array([110, 112, 111, 115, 114, 116, 118], dtype=np.float64)
    l = np.array([90, 92, 91, 95, 94, 96, 98], dtype=np.float64)
    c = np.array([100, 102, 101, 105, 104, 106, 108], dtype=np.float64)
    k = stoch_k(h, l, c, 5)
    d = stoch_d(k, 3)
    assert len(k) == 7
    assert np.isnan(k[0:4]).all()
    assert not np.isnan(k[4])
    assert 0 <= k[4] <= 100
    assert np.isnan(d[0:6]).all() or not np.isnan(d[6])
    builtins.print(f"  Stoch: %K={k[-1]:.2f}, %D={d[-1]:.2f}")


def test_bollinger_b():
    from mtrader import bollinger_b
    c = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                  110, 108, 111, 113, 112], dtype=np.float64)
    vals = bollinger_b(c, 10)
    assert len(vals) == 15
    assert np.isnan(vals[0:9]).all()
    assert not np.isnan(vals[14])
    assert 0 <= vals[14] <= 1
    builtins.print(f"  Bollinger %B: value={vals[-1]:.4f}")


def test_obv():
    from mtrader import obv
    c = np.array([100, 102, 101, 103, 105], dtype=np.float64)
    v = np.array([1000, 1500, 1200, 1800, 2000], dtype=np.float64)
    vals = obv(c, v)
    assert len(vals) == 5
    assert vals[0] == 1000
    assert vals[1] == 2500
    builtins.print(f"  OBV: {vals}")


def test_add_indicators_popular():
    from mtrader import add_indicators

    df = _gen_ohlc(300)
    result = add_indicators(
        df,
        add=["rsi", "atr", "stochk", "stochd", "bbp", "obv", "close", "high", "low", "volume"],
        rolling_minutes=[14],
    )

    for col in ["can1_rsi_p14", "can1_atr_p14", "can1_stochk_p14",
                 "can1_stochd_p14", "can1_bbp_p14", "can1_obv"]:
        assert col in result.columns, f"Missing column: {col}"
        assert result[col].notna().sum() > 0, f"All NaN in {col}"
    builtins.print(f"  add_indicators popular: RSI/ATR/Stoch/%B/OBV all computed")


def test_find_best_exit_raises_on_empty():
    from mtrader import find_best_exit
    import pytest

    df = _prepare_backtest_df()
    with pytest.raises(ValueError, match="At least one"):
        find_best_exit(df, entry_conditions=[[]], buy_or_sell="buy")


if __name__ == "__main__":
    test_clean_data_roundtrip()
    test_add_indicators_basic()
    test_add_indicators_ema()
    test_add_indicators_distance()
    test_add_indicators_vwap()
    test_precalculate_exit()
    test_precalculate_exit_normalized()
    test_take_trade_on_condition_numpy()
    test_take_trade_on_condition()
    test_empty_trade()
    test_update_cond()
    test_monotonic_stack()
    test_timenum()
    test_detect_data_types()
    test_indicators_numerical()
    builtins.print("\nAll tests passed!")
