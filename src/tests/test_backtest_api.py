import numpy as np
import pandas as pd
import pytest


def _sample_df(days=4, bars=40):
    rng = np.random.default_rng(123)
    rows = []
    price = 100.0
    for day in range(days):
        start = pd.Timestamp("2024-01-02 09:15") + pd.Timedelta(days=day)
        for bar in range(bars):
            price += rng.normal(0, 0.8)
            open_p = price + rng.normal(0, 0.2)
            close = price + rng.normal(0, 0.2)
            high = max(open_p, close) + rng.uniform(0.1, 1.0)
            low = min(open_p, close) - rng.uniform(0.1, 1.0)
            rows.append((start + pd.Timedelta(minutes=bar), open_p, high, low, close, 1000 + bar))
    return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])


def test_run_backtest_convenience_api():
    from mtrader import condition, run_backtest

    result = run_backtest(
        _sample_df(),
        entry_conditions=[[condition("close", "can1_sma1_p5", lower=0)]],
        exit_conditions=[[condition("can1_sma1_p5", "close", lower=0)]],
        indicators=["sma1", "zero"],
        rolling_minutes=[5],
        buy_or_sell="buy",
        target_delta_normalized=0.5,
        stoploss_delta_normalized=0.25,
        initial_capital=5000,
    )

    assert result.final_capital > 0
    assert "total_trades" in result.report
    assert "equity" in result.equity.columns
    assert set(["entry_time", "exit_time", "entry_price", "exit_price"]).issubset(result.trades.columns)
    assert "take_trade" not in _sample_df().columns


def test_condition_cross_helpers_and_parameter_grid():
    from mtrader import condition, cross_above, cross_below, parameter_grid

    assert condition("a", "b", lower=1)["first_column_name"] == "a"
    assert len(cross_above("fast", "slow")) == 2
    assert len(cross_below("fast", "slow")) == 2
    grid = parameter_grid(target=[1, 2], stop=[0.5, 1.0], side="buy")
    assert len(grid) == 4
    assert grid[0]["side"] == "buy"


def test_validate_ohlcv_catches_bad_data():
    from mtrader import validate_ohlcv

    df = _sample_df()
    assert validate_ohlcv(df, require_volume=True)

    bad = df.copy()
    bad.loc[0, "high"] = bad.loc[0, "low"] - 1
    with pytest.raises(ValueError, match="high"):
        validate_ohlcv(bad)

    unsorted = df.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="sorted"):
        validate_ohlcv(unsorted)


def test_walk_forward_splits():
    from mtrader import walk_forward_splits

    df = _sample_df(days=6, bars=10)
    splits = walk_forward_splits(df, train_days=3, test_days=1)
    assert len(splits) == 3
    train_idx, test_idx = splits[0]
    assert len(train_idx) == 30
    assert len(test_idx) == 10
    assert max(train_idx) < min(test_idx)


def test_backtest_report_does_not_mutate_capital_column():
    from mtrader import backtest_report

    df = _sample_df(days=1, bars=5)
    df["take_trade"] = [False, True, False, False, False]
    df["capital_at_exit"] = [0.0, 1010.0, 0.0, 0.0, 0.0]
    before = df["capital_at_exit"].copy()
    report = backtest_report(df)
    assert report["total_trades"] == 1
    assert df["capital_at_exit"].equals(before)


def test_html_backtest_report_generates_file(tmp_path):
    from mtrader import condition, html_backtest_report, run_backtest

    params = {
        "strategy": "SMA trend",
        "side": "buy",
        "target_delta_normalized": 0.5,
        "stoploss_delta_normalized": 0.25,
        "rolling_minutes": [5],
    }
    result = run_backtest(
        _sample_df(),
        entry_conditions=[[condition("close", "can1_sma1_p5", lower=0)]],
        exit_conditions=[[condition("can1_sma1_p5", "close", lower=0)]],
        indicators=["sma1", "zero"],
        rolling_minutes=[5],
        target_delta_normalized=0.5,
        stoploss_delta_normalized=0.25,
    )
    output = tmp_path / "report.html"
    html = html_backtest_report(
        result,
        output_path=output,
        title="Backtest Tearsheet",
        strategy_name="SMA trend",
        parameters=params,
    )

    assert output.exists()
    assert "<!doctype html>" in html
    assert "Backtest Tearsheet" in html
    assert "SMA trend" in html
    assert "Final Capital" in html
    assert "Equity Curve" in html
    assert "Drawdown" in html
    assert "Trade Log" in html
    assert "<svg" in html
    assert "target_delta_normalized" in html
    assert output.read_text(encoding="utf-8") == html


def test_backtest_result_to_html():
    from mtrader import condition, run_backtest

    result = run_backtest(
        _sample_df(),
        entry_conditions=[[condition("close", "can1_sma1_p5", lower=0)]],
        indicators=["sma1", "zero"],
        rolling_minutes=[5],
        target_delta_normalized=0.5,
    )
    html = result.to_html(strategy_name="Method API", parameters={"capital": 1000})
    assert "Method API" in html
    assert "capital" in html
