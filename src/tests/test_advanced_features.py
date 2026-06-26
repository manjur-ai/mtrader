import numpy as np
import pandas as pd


def _df(days=5, bars=30, seed=77):
    rng = np.random.default_rng(seed)
    rows = []
    price = 100.0
    for day in range(days):
        start = pd.Timestamp("2024-01-02 09:15") + pd.Timedelta(days=day)
        for i in range(bars):
            price += rng.normal(0, 0.7)
            open_p = price + rng.normal(0, 0.1)
            close = price + rng.normal(0, 0.1)
            high = max(open_p, close) + 0.5
            low = min(open_p, close) - 0.5
            rows.append((start + pd.Timedelta(minutes=i), open_p, high, low, close, 1000 + i))
    return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])


def _strategy(period=5, target=0.4):
    from mtrader import Strategy, condition

    return Strategy(
        name="SMA trend",
        indicators=["sma1", "zero"],
        rolling_minutes=[period],
        entry_conditions=[[condition("close", f"can1_sma1_p{period}", lower=0)]],
        exit_conditions=[[condition(f"can1_sma1_p{period}", "close", lower=0)]],
        target_delta_normalized=target,
        stoploss_delta_normalized=0.25,
    )


def test_strategy_cost_sizing_and_risk_helpers():
    from mtrader import (
        apply_risk_controls,
        atr_risk_size,
        fixed_capital_size,
        fixed_quantity_size,
        india_intraday_cost_model,
        percent_equity_size,
    )

    result = _strategy().run(_df(), initial_capital=2000)
    assert result.final_capital > 0

    costs = india_intraday_cost_model()
    assert costs.cost_rate() > 0
    assert costs.estimate(np.array([10000.0]))[0] > 0

    prices = np.array([100.0, 200.0])
    assert np.allclose(fixed_quantity_size(prices, 3), [3, 3])
    assert np.allclose(fixed_capital_size(prices, 1000), [10, 5])
    assert np.allclose(percent_equity_size(prices, np.array([1000.0, 2000.0]), pct=0.5), [5, 5])
    assert np.allclose(atr_risk_size(prices, np.array([2.0, 4.0]), np.array([1000.0, 1000.0])), [5, 2.5])

    controlled = apply_risk_controls(result.trades, max_trades_per_day=1, cooldown_bars=3)
    assert "allowed" in controlled.columns
    if not controlled.empty:
        assert controlled["allowed"].sum() <= controlled["entry_time"].dt.date.nunique()


def test_multitimeframe_portfolio_and_optimization_helpers():
    from mtrader import (
        add_higher_timeframe_indicators,
        grid_from_ranges,
        random_parameter_search,
        resample_ohlcv,
        run_portfolio,
        walk_forward_optimize,
    )

    data = _df(days=6, bars=20)
    higher = resample_ohlcv(data, "5min")
    assert len(higher) < len(data)
    enriched = add_higher_timeframe_indicators(data, "5min", add=["sma1"], rolling_minutes=[3])
    assert "can5_sma1_p3" in enriched.columns

    portfolio = run_portfolio({"AAA": data, "BBB": _df(seed=78)}, _strategy(), initial_capital=5000)
    assert portfolio["final_capital"] > 0
    assert "equity" in portfolio["equity"].columns

    grid = grid_from_ranges(period=[3, 5], target=[0.3])
    assert len(grid) == 2
    best, rows = random_parameter_search(data, _strategy, {"period": [3, 5], "target": [0.3, 0.4]}, n_iter=3, seed=1)
    assert best["result"].final_capital > 0
    assert len(rows) == 3

    wf = walk_forward_optimize(data, _strategy, grid, train_days=3, test_days=1)
    assert {"split", "train_score", "test_final_capital", "params"}.issubset(wf.columns)


def test_html_report_has_advanced_sections():
    result = _strategy().run(_df(days=4), initial_capital=1000)
    html = result.to_html(strategy_name="Advanced Report", parameters={"period": 5})
    assert "Monthly Returns" in html
    assert "Trade Distribution" in html
    assert "Drawdown Periods" in html
    assert "Best And Worst Trades" in html


def test_strategy_save_and_load_text_file(tmp_path):
    from mtrader import Strategy, load_strategy, save_strategy

    strategy = _strategy(period=7, target=0.6)
    path = tmp_path / "strategy.txt"
    returned = save_strategy(strategy, path)

    assert returned == path
    text = path.read_text(encoding="utf-8")
    assert '"schema": "mtrader.strategy"' in text
    assert '"name": "SMA trend"' in text

    loaded = load_strategy(path)
    assert isinstance(loaded, Strategy)
    assert loaded.to_dict() == strategy.to_dict()

    result = loaded.run(_df(), initial_capital=1000)
    assert result.final_capital > 0

    path2 = tmp_path / "method_strategy.txt"
    strategy.save(path2)
    loaded2 = Strategy.load(path2)
    assert loaded2.to_dict() == strategy.to_dict()
