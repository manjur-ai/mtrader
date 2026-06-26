import numpy as np
import pandas as pd


def _live_df(n=80):
    rng = np.random.default_rng(909)
    price = 100.0
    rows = []
    start = pd.Timestamp("2024-01-02 09:15")
    for i in range(n):
        price += rng.normal(0, 0.5)
        open_p = price + rng.normal(0, 0.1)
        close = price + rng.normal(0, 0.1)
        high = max(open_p, close) + rng.uniform(0.1, 0.8)
        low = min(open_p, close) - rng.uniform(0.1, 0.8)
        rows.append((start + pd.Timedelta(minutes=i), open_p, high, low, close, 1000 + i))
    return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])


def _live_df_days(days=3, bars=35):
    frames = []
    base = pd.Timestamp("2024-01-02 09:15")
    for day in range(days):
        frame = _live_df(bars)
        frame["datetime"] = frame["datetime"] + pd.Timedelta(days=day)
        frame[["open", "high", "low", "close"]] += day * 1.5
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_live_update_matches_batch_for_o1_indicators():
    from mtrader import LiveIndicatorEngine, atr, ema

    df = _live_df(60)
    history = df.iloc[:-1]
    new_bar = df.iloc[-1]
    engine = LiveIndicatorEngine.from_history(
        history,
        indicators=["sma", "ema", "atr", "vwap", "zero"],
        periods=[5, 14],
    )
    out = engine.update(new_bar)
    full = df

    assert np.isclose(out["can1_sma1_p5"], full["close"].tail(5).mean())
    assert np.isclose(out["can1_ema1_p14"], ema(full["close"].to_numpy(), 14)[-1])
    assert np.isclose(out["can1_atr_p14"], atr(full["high"].to_numpy(), full["low"].to_numpy(), full["close"].to_numpy(), 14)[-1])

    typical = (full["high"] + full["low"] + full["close"]) / 3.0
    expected_vwap = (typical * full["volume"]).sum() / full["volume"].sum()
    assert np.isclose(out["can1_vwap"], expected_vwap)
    assert np.isclose(out["live_sma_p5"], out["can1_sma1_p5"])
    assert len(engine.to_frame()) == 60


def test_live_engine_generates_realtime_buy_sell_signals():
    from mtrader import LiveIndicatorEngine, condition, cross_above, cross_below

    df = _live_df(40)
    buy = [[condition("close", "can1_sma1_p5", lower=0)]]
    sell = [cross_below("close", "can1_ema1_p5")]
    engine = LiveIndicatorEngine.from_history(
        df.iloc[:-1],
        indicators=["sma", "ema", "rsi", "atr", "vwap", "zero"],
        periods=[5, 14],
        buy_conditions=buy,
        sell_conditions=sell,
    )
    out = engine.update(df.iloc[-1])

    assert "can1_sma1_p5" in out
    assert "can1_ema1_p5" in out
    assert "can1_rsi_p14" in out
    assert "can1_atr_p14" in out
    assert "can1_vwap" in out
    assert isinstance(out["buy_signal"], bool)
    assert isinstance(out["sell_signal"], bool)


def test_live_signal_from_history_helper():
    from mtrader import condition, live_signal_from_history

    df = _live_df(30)
    out = live_signal_from_history(
        df.iloc[:-1],
        indicators=["sma", "ema", "atr", "vwap", "zero"],
        periods=[5],
        new_bar=df.iloc[-1],
        buy_conditions=[[condition("close", "can1_sma1_p5", lower=0)]],
    )
    assert "buy_signal" in out
    assert "can1_sma1_p5" in out


def test_strategy_converts_backtest_definition_to_live_signals():
    from mtrader import Strategy, condition, convert_conditions_to_live, live_column_name

    df = _live_df(50)
    strategy = Strategy(
        name="same definition",
        indicators=["sma1", "ema1", "rsi", "atr", "vwap", "zero"],
        rolling_minutes=[5, 14],
        entry_conditions=[[
            condition("close", "can1_sma1_p5", lower=0),
            condition("can1_rsi_p14", "zero", lower=40),
        ]],
        exit_conditions=[[condition("can1_ema1_p5", "close", lower=0)]],
        target_delta_normalized=0.4,
        stoploss_delta_normalized=0.2,
    )

    result = strategy.run(df.iloc[:-1])
    assert result.final_capital > 0

    live = strategy.to_live(df.iloc[:-1])
    out = live.update(df.iloc[-1])

    assert live_column_name("can1_sma1_p5") == "can1_sma1_p5"
    converted = convert_conditions_to_live(strategy.entry_conditions)
    assert converted == strategy.entry_conditions
    assert "can1_sma1_p5" in out
    assert "can1_ema1_p5" in out
    assert "can1_rsi_p14" in out
    assert "can1_atr_p14" in out
    assert "can1_vwap" in out
    assert isinstance(out["entry_signal"], bool)
    assert isinstance(out["exit_signal"], bool)
    assert out["action"] in {"BUY", "EXIT_BUY", "HOLD"}


def test_sell_strategy_maps_entry_to_sell_action():
    from mtrader import Strategy, condition

    df = _live_df(40)
    strategy = Strategy(
        name="short definition",
        indicators=["sma1", "zero"],
        rolling_minutes=[5],
        side="sell",
        entry_conditions=[[condition("can1_sma1_p5", "close", lower=0)]],
        exit_conditions=[[condition("close", "can1_sma1_p5", lower=0)]],
    )
    live = strategy.to_live(df.iloc[:-1])
    out = live.update(df.iloc[-1])
    assert isinstance(out["entry_signal"], bool)
    assert isinstance(out["exit_signal"], bool)
    assert out["action"] in {"SELL", "EXIT_SELL", "HOLD"}


def test_live_strategy_stream_loops_over_candle_feed():
    from mtrader import Strategy, condition, stream_live_signals

    df = _live_df(45)
    strategy = Strategy(
        name="streaming definition",
        indicators=["sma1", "ema1", "zero"],
        rolling_minutes=[5],
        entry_conditions=[[condition("close", "can1_sma1_p5", lower=0)]],
        exit_conditions=[[condition("can1_ema1_p5", "close", lower=0)]],
    )
    live = strategy.to_live(df.iloc[:30])
    feed = (row for _, row in df.iloc[30:].iterrows())
    signals = list(live.stream(feed))

    assert len(signals) == 15
    assert all("action" in signal for signal in signals)
    assert all("can1_sma1_p5" in signal for signal in signals)

    live2 = strategy.to_live(df.iloc[:30])
    seen = []
    feed2 = (row for _, row in df.iloc[30:35].iterrows())
    returned = list(stream_live_signals(live2, feed2, on_signal=lambda signal: seen.append(signal["action"])))

    assert len(returned) == 5
    assert len(seen) == 5


def test_live_fallback_supports_non_o1_batch_indicators():
    from mtrader import LiveIndicatorEngine, add_indicators

    df = _live_df_days(days=3, bars=40)
    history = df.iloc[:-1]
    new_bar = df.iloc[-1]
    indicators = [
        "wma1", "ssma1", "stochk", "stochd", "bbp", "willr", "cci", "adx", "mfi",
        "macd", "obv", "psar", "ha", "supertrend", "ichimoku", "pivot", "prev_day",
        "gap", "inside_bar", "engulfing", "max", "min",
    ]

    engine = LiveIndicatorEngine.from_history(history, indicators=indicators, periods=[10, 14], history_size=200)
    out = engine.update(new_bar)
    batch = add_indicators(df.copy(), add=indicators, rolling_minutes=[10, 14]).iloc[-1]

    expected_columns = [
        "can1_wma1_p10", "can1_ssma1_p10", "can1_stochk_p14", "can1_stochd_p14",
        "can1_bbp_p14", "can1_willr_p14", "can1_cci_p14", "can1_adx_p14",
        "can1_mfi_p14", "can1_macd", "can1_macdsignal", "can1_obv", "can1_psar",
        "can1_ha_close", "can1_supertrend_p10", "can1_ichi_kijun", "can1_pivot",
        "can1_prev_day_high", "can1_gap_pct", "can1_inside_bar",
        "can1_bullish_engulfing", "can1_max_p10", "can1_min_p10",
    ]
    for col in expected_columns:
        assert col in out, col
        if pd.notna(batch[col]):
            assert np.isclose(out[col], batch[col], equal_nan=True), col


def test_live_warmup_uses_batch_indicator_calculation_before_streaming():
    from mtrader import LiveIndicatorEngine, add_indicators

    df = _live_df_days(days=2, bars=40)
    indicators = ["sma", "ema", "rsi", "atr", "vwap", "macd", "supertrend", "pivot", "zero"]
    engine = LiveIndicatorEngine.from_history(df, indicators=indicators, periods=[10, 14], history_size=100)
    latest = engine.latest()
    batch = add_indicators(
        df.copy(),
        add=["sma1", "ema1", "rsi", "atr", "vwap", "macd", "supertrend", "pivot", "zero"],
        rolling_minutes=[10, 14],
    ).iloc[-1]

    for col in ["can1_sma1_p10", "can1_ema1_p10", "can1_rsi_p14", "can1_atr_p14", "can1_vwap", "can1_macd", "can1_supertrend_p10", "can1_pivot"]:
        assert col in latest
        if pd.notna(batch[col]):
            assert np.isclose(latest[col], batch[col], equal_nan=True), col

    new_bar = _live_df_days(days=2, bars=41).iloc[-1]
    out = engine.update(new_bar)
    assert "can1_sma1_p10" in out
    assert "can1_macd" in out


def test_strategy_to_live_accepts_advanced_indicator_conditions():
    from mtrader import Strategy, condition

    df = _live_df(70)
    strategy = Strategy(
        name="macd live fallback",
        indicators=["macd", "zero"],
        rolling_minutes=[],
        entry_conditions=[[condition("can1_macd", "can1_macdsignal", lower=0)]],
        exit_conditions=[[condition("can1_macdsignal", "can1_macd", lower=0)]],
    )
    live = strategy.to_live(df.iloc[:-1])
    out = live.update(df.iloc[-1])
    assert "can1_macd" in out
    assert "can1_macdsignal" in out
    assert isinstance(out["entry_signal"], bool)
