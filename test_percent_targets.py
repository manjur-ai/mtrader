import pandas as pd

import mtrader


def test_target_delta_normalized_is_percent():
    df = pd.DataFrame({
        "datetime": pd.to_datetime([
            "2025-01-01 09:15",
            "2025-01-01 09:16",
        ]),
        "open": [100.0, 100.0],
        "high": [100.0, 102.0],
        "low": [100.0, 99.0],
        "close": [100.0, 101.0],
        "volume": [1.0, 1.0],
    })
    strategy = mtrader.Strategy(
        name="percent target",
        indicators=["zero"],
        rolling_minutes=[],
        entry_conditions=[[mtrader.condition("close", "zero", lower=1)]],
        exit_conditions=[[mtrader.condition("zero", "zero", lower=1, upper=2)]],
        side="buy",
        target_delta_normalized=1.5,
        stoploss_delta_normalized=1.0,
        trading_cost_factor=0.0,
    )

    result = strategy.run(df, trading_cost_factor=0.0)

    assert result.df.loc[0, "target_price"] == 101.5
    assert result.df.loc[0, "stoploss_price"] == 99.0


def test_run_oms_target_delta_normalized_is_percent():
    df = pd.DataFrame({
        "datetime": pd.to_datetime([
            "2025-01-01 09:15",
            "2025-01-01 09:16",
            "2025-01-01 09:17",
        ]),
        "open": [100.0, 100.0, 100.0],
        "high": [100.0, 100.0, 102.0],
        "low": [100.0, 100.0, 100.0],
        "close": [100.0, 100.0, 101.0],
        "volume": [1.0, 1.0, 1.0],
    })
    strategy = mtrader.Strategy(
        name="oms percent target",
        indicators=["zero"],
        rolling_minutes=[],
        entry_conditions=[[mtrader.condition("close", "zero", lower=1)]],
        exit_conditions=[[mtrader.condition("zero", "zero", lower=1, upper=2)]],
        side="buy",
        target_delta_normalized=1.5,
        stoploss_delta_normalized=1.0,
        trading_cost_factor=0.0,
    )

    result = strategy.run_oms(df, lot_size=1.0, initial_capital=1000.0)

    assert len(result.trades) == 1
    assert result.trades.loc[0, "entry_price"] == 100.0
    assert result.trades.loc[0, "exit_price"] == 101.5
    assert result.trades.loc[0, "profit"] == 1.5
