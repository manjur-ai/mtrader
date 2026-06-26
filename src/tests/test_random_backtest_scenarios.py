import numpy as np
import pandas as pd


def _gen_intraday(days=5, bars_per_day=90, seed=2026):
    rng = np.random.default_rng(seed)
    rows = []
    price = 15000.0
    for day in range(days):
        start = pd.Timestamp("2024-01-02 09:15") + pd.Timedelta(days=day)
        overnight_gap = rng.normal(0, 85)
        price = max(100.0, price + overnight_gap)
        for minute in range(bars_per_day):
            ts = start + pd.Timedelta(minutes=minute)
            drift = 5.0 * np.sin((minute / bars_per_day) * np.pi * 2.0)
            price = max(100.0, price + rng.normal(drift, 28))
            open_p = price + rng.normal(0, 10)
            close = price + rng.normal(0, 10)
            high = max(open_p, close) + rng.uniform(2, 35)
            low = min(open_p, close) - rng.uniform(2, 35)
            volume = int(rng.integers(500, 8000))
            rows.append((ts, open_p, high, low, close, volume))
    return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"]).round(4)


def _cond(first, second="zero", lower=-np.inf, upper=np.inf, s1=0, s2=0):
    return {
        "first_column_name": first,
        "second_column_name": second,
        "shift_down_first": s1,
        "shift_down_second": s2,
        "lower_range_of_difference": lower,
        "upper_range_of_difference": upper,
        "perform_normalization_of_diff": False,
    }


def _run_backtest(name, df, entry, side="buy", exit_cond=None, target=0.7, stop=0.35):
    from mtrader import precalculate_exit_time_amount_profit, take_trade_on_condition_numpy, backtest_report

    exit_cond = exit_cond or entry
    df = precalculate_exit_time_amount_profit(
        df,
        exit_cond,
        buy_or_sell=side,
        target_delta_normalized=target,
        stoploss_delta_normalized=stop,
    )
    trades, final_capital, metrics = take_trade_on_condition_numpy(df, entry, initial_capital=1000)
    report = backtest_report(df, initial_capital=1000)

    assert "take_trade" in df.columns, name
    assert "next_exit_index" in df.columns, name
    assert isinstance(final_capital, (int, float, np.floating)), name
    assert isinstance(metrics, dict), name
    assert isinstance(report, dict), name
    assert trades is not None, name
    return df, trades, final_capital, metrics


def test_20_random_popular_backtests_are_executable():
    from mtrader import add_indicators

    results = []

    df = add_indicators(_gen_intraday(), add=["supertrend", "zero"], rolling_minutes=[10])
    results.append(_run_backtest(
        "supertrend trend follow",
        df,
        [[_cond("can1_supertrend_dir_p10", lower=1)]],
        exit_cond=[[_cond("zero", "can1_supertrend_dir_p10", lower=1)]],
    ))

    df = add_indicators(_gen_intraday(seed=2), add=["ichimoku", "zero"], rolling_minutes=[])
    results.append(_run_backtest(
        "ichimoku cloud breakout",
        df,
        [[_cond("close", "can1_ichi_span_a", lower=0), _cond("close", "can1_ichi_span_b", lower=0)]],
        exit_cond=[[_cond("can1_ichi_kijun", "close", lower=0)]],
    ))

    df = add_indicators(_gen_intraday(seed=3), add=["pivot", "zero"], rolling_minutes=[])
    results.append(_run_backtest(
        "daily pivot s1 reversal",
        df,
        [[_cond("close", "can1_pivot_s1", lower=0), _cond("can1_pivot", "close", lower=0)]],
        exit_cond=[[_cond("close", "can1_pivot", lower=0)]],
    ))

    df = add_indicators(_gen_intraday(seed=4), add=["prev_day", "zero"], rolling_minutes=[])
    results.append(_run_backtest(
        "previous day high breakout",
        df,
        [[_cond("close", "can1_prev_day_high", lower=0)]],
        exit_cond=[[_cond("can1_prev_day_low", "close", lower=0)]],
    ))

    df = add_indicators(_gen_intraday(seed=5), add=["gap", "vwap", "zero"], rolling_minutes=[])
    results.append(_run_backtest(
        "gap up fade short",
        df,
        [[_cond("can1_gap_pct", lower=0.2), _cond("open", "close", lower=0)]],
        side="sell",
        exit_cond=[[_cond("can1_vwap", "close", lower=0)]],
    ))

    df = add_indicators(_gen_intraday(seed=6), add=["inside_bar", "high", "zero"], rolling_minutes=[])
    results.append(_run_backtest(
        "inside bar breakout",
        df,
        [[_cond("can1_inside_bar", lower=1, s1=1), _cond("close", "high", lower=0, s2=1)]],
    ))

    df = add_indicators(_gen_intraday(seed=7), add=["engulfing", "zero"], rolling_minutes=[])
    results.append(_run_backtest(
        "bullish engulfing reversal",
        df,
        [[_cond("can1_bullish_engulfing", lower=1)]],
    ))

    df = add_indicators(_gen_intraday(seed=8), add=["engulfing", "zero"], rolling_minutes=[])
    results.append(_run_backtest(
        "bearish engulfing short",
        df,
        [[_cond("can1_bearish_engulfing", lower=1)]],
        side="sell",
    ))

    df = add_indicators(_gen_intraday(seed=9), add=["obv", "sma1", "zero"], rolling_minutes=[20])
    df["obv_sma20"] = df["can1_obv"].rolling(20, min_periods=1).mean()
    results.append(_run_backtest(
        "obv confirmation",
        df,
        [[_cond("can1_obv", "obv_sma20", lower=0), _cond("close", "can1_sma1_p20", lower=0)]],
    ))

    df = add_indicators(_gen_intraday(seed=10), add=["mfi", "zero"], rolling_minutes=[14])
    results.append(_run_backtest(
        "mfi oversold bounce",
        df,
        [[_cond("can1_mfi_p14", upper=35)]],
        exit_cond=[[_cond("can1_mfi_p14", lower=55)]],
    ))

    df = add_indicators(_gen_intraday(seed=11), add=["adx", "ema1", "zero"], rolling_minutes=[14, 20, 50])
    results.append(_run_backtest(
        "adx ema trend filter",
        df,
        [[_cond("can1_ema1_p20", "can1_ema1_p50", lower=0), _cond("can1_adx_p14", lower=20)]],
    ))

    df = add_indicators(_gen_intraday(seed=12), add=["cci", "zero"], rolling_minutes=[20])
    results.append(_run_backtest(
        "cci mean reversion",
        df,
        [[_cond("can1_cci_p20", upper=-80)]],
        exit_cond=[[_cond("can1_cci_p20", lower=0)]],
    ))

    df = add_indicators(_gen_intraday(seed=13), add=["willr", "zero"], rolling_minutes=[14])
    results.append(_run_backtest(
        "williams r oversold",
        df,
        [[_cond("can1_willr_p14", upper=-80)]],
        exit_cond=[[_cond("can1_willr_p14", lower=-50)]],
    ))

    df = add_indicators(_gen_intraday(seed=14), add=["macd", "zero"], rolling_minutes=[])
    results.append(_run_backtest(
        "macd bullish crossover",
        df,
        [[_cond("can1_macd", "can1_macdsignal", upper=0, s1=1, s2=1),
          _cond("can1_macd", "can1_macdsignal", lower=0)]],
        exit_cond=[[_cond("can1_macdsignal", "can1_macd", lower=0)]],
    ))

    df = add_indicators(_gen_intraday(seed=15), add=["psar", "zero"], rolling_minutes=[])
    results.append(_run_backtest(
        "psar trend follow",
        df,
        [[_cond("close", "can1_psar", lower=0)]],
        exit_cond=[[_cond("can1_psar", "close", lower=0)]],
    ))

    df = add_indicators(_gen_intraday(seed=16), add=["ha", "zero"], rolling_minutes=[])
    results.append(_run_backtest(
        "heikin ashi two candle trend",
        df,
        [[_cond("can1_ha_close", "can1_ha_open", lower=0, s1=1, s2=1),
          _cond("can1_ha_close", "can1_ha_open", lower=0)]],
        exit_cond=[[_cond("can1_ha_open", "can1_ha_close", lower=0)]],
    ))

    df = add_indicators(_gen_intraday(seed=17), add=["ema1", "atr", "zero"], rolling_minutes=[20, 14])
    df["keltner_lower"] = df["can1_ema1_p20"] - 1.5 * df["can1_atr_p14"]
    results.append(_run_backtest(
        "keltner pullback",
        df,
        [[_cond("keltner_lower", "close", lower=0)]],
        exit_cond=[[_cond("close", "can1_ema1_p20", lower=0)]],
    ))

    df = add_indicators(_gen_intraday(seed=18), add=["max", "min", "zero"], rolling_minutes=[20, 10])
    results.append(_run_backtest(
        "donchian breakout",
        df,
        [[_cond("close", "can1_max_p20", lower=0, s2=1)]],
        exit_cond=[[_cond("can1_min_p10", "close", lower=0)]],
    ))

    df = add_indicators(_gen_intraday(seed=19), add=["vwap", "zero"], rolling_minutes=[])
    results.append(_run_backtest(
        "vwap mean reversion",
        df,
        [[_cond("can1_vwap", "close", lower=15)]],
        exit_cond=[[_cond("close", "can1_vwap", lower=0)]],
    ))

    df = add_indicators(_gen_intraday(seed=20), add=["bbp", "rsi", "zero"], rolling_minutes=[20, 14])
    results.append(_run_backtest(
        "bollinger rsi squeeze bounce",
        df,
        [[_cond("can1_bbp_p20", upper=0.25), _cond("can1_rsi_p14", upper=45)]],
        exit_cond=[[_cond("can1_bbp_p20", lower=0.5)]],
    ))

    assert len(results) == 20
