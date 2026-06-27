"""10 Trading Strategy Test Suite for mtrader library"""
import builtins
import numpy as np
import pandas as pd
import sys, os, traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from mtrader import *

np.random.seed(42)


def _gen_intraday(days=5, bars_per_day=90, seed=2026):
    rng = np.random.default_rng(seed)
    rows = []
    price = 15000.0
    for day in range(days):
        start = pd.Timestamp("2024-01-02 09:15") + pd.Timedelta(days=day)
        price = max(100.0, price + rng.normal(0, 85))
        for minute in range(bars_per_day):
            ts = start + pd.Timedelta(minutes=minute)
            drift = 5.0 * np.sin((minute / bars_per_day) * np.pi * 2.0)
            price = max(100.0, price + rng.normal(drift, 28))
            hp = price + rng.normal(0, 10)
            lp = price + rng.normal(0, 10)
            high = max(price, hp, lp) + rng.uniform(2, 35)
            low = min(price, hp, lp) - rng.uniform(2, 35)
            open_p = hp
            close = lp
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


results = []


def wrap(name, fn):
    builtins.print(f"\n{'='*60}")
    builtins.print(f"  [{len(results)+1}] {name}")
    builtins.print(f"{'='*60}")
    try:
        fn()
        results.append((name, "PASS", ""))
        builtins.print(f"  >>> PASS")
    except Exception as e:
        tb = traceback.format_exc()
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        builtins.print(f"  >>> FAIL: {type(e).__name__}: {e}")
        builtins.print(tb[:500])


# ── STRATEGY 1: Basic SMA Crossover (Long)
def s1_sma_crossover():
    df = _gen_intraday(seed=1)
    result = run_backtest(
        df,
        entry_conditions=[[condition("close", "can1_sma1_p20", lower=0)]],
        exit_conditions=[[condition("can1_sma1_p50", "close", lower=0)]],
        indicators=["sma1", "zero"], rolling_minutes=[20, 50],
        target_delta_normalized=1.0, stoploss_delta_normalized=0.5,
        initial_capital=10000,
    )
    assert result.final_capital > 0
    assert len(result.trades) > 0
    assert result.report["total_trades"] > 0
    assert "Sharpe Ratio" in result.metrics
    builtins.print(f"     Trades: {result.report['total_trades']}, Final: {result.final_capital:.2f}, Sharpe: {result.metrics['Sharpe Ratio']:.3f}")


# ── STRATEGY 2: RSI Short Reversal (Short)
def s2_rsi_short():
    df = _gen_intraday(seed=2)
    result = run_backtest(
        df,
        entry_conditions=[[condition("can1_rsi_p14", lower=70)]],
        exit_conditions=[[condition("can1_rsi_p14", upper=40)]],
        indicators=["rsi", "zero"], rolling_minutes=[14],
        buy_or_sell="sell",
        target_delta_normalized=0.5, stoploss_delta_normalized=0.25,
        initial_capital=10000,
    )
    assert result.final_capital > 0
    assert len(result.trades) > 0
    builtins.print(f"     Trades: {result.report['total_trades']}, Final: {result.final_capital:.2f}")


# ── STRATEGY 3: MACD + Bollinger Multi-Condition
def s3_macd_bollinger():
    df = _gen_intraday(seed=3)
    entry = [[
        condition("can1_macd", "can1_macdsignal", lower=0),
        condition("can1_bbp_p20", lower=0.2, upper=0.8),
    ]]
    result = run_backtest(
        df, entry,
        exit_conditions=[[condition("can1_macdsignal", "can1_macd", lower=0)]],
        indicators=["macd", "bbp", "zero"], rolling_minutes=[20],
        target_delta_normalized=0.75, stoploss_delta_normalized=0.35,
        initial_capital=10000,
    )
    assert result.final_capital > 0
    assert len(result.trades) > 0
    assert "can1_macd" in result.df.columns
    assert "can1_bbp_p20" in result.df.columns
    builtins.print(f"     Trades: {result.report['total_trades']}, Final: {result.final_capital:.2f}")


# ── STRATEGY 4: Ichimoku Cloud Breakout
def s4_ichimoku():
    df = _gen_intraday(seed=4, days=8)
    entry = [[
        condition("close", "can1_ichi_span_a", lower=0),
        condition("close", "can1_ichi_span_b", lower=0),
    ]]
    result = run_backtest(
        df, entry,
        exit_conditions=[[condition("can1_ichi_kijun", "close", lower=0)]],
        indicators=["ichimoku", "zero"], rolling_minutes=[],
        target_delta_normalized=1.5, stoploss_delta_normalized=0.75,
        initial_capital=10000,
    )
    assert result.final_capital > 0
    assert "can1_ichi_tenkan" in result.df.columns
    assert "can1_ichi_span_a" in result.df.columns
    builtins.print(f"     Trades: {result.report['total_trades']}, Final: {result.final_capital:.2f}")


# ── STRATEGY 5: Opening Range + ATR Dynamic Exits
def s5_or_atr():
    df = add_indicators(
        _gen_intraday(seed=5),
        add=["or_high", "atr", "sma1", "close", "high", "low", "zero"],
        rolling_minutes=[15, 14, 20],
    )
    df["target_2atr"] = df["can1_atr_p14"] * 2.0
    df["stoploss_1atr"] = df["can1_atr_p14"] * 1.0
    result = run_backtest(
        df, entry_conditions=[[condition("close", "can1_or_high_p15", lower=0)]],
        indicators=[], rolling_minutes=[],
        target_delta_column="target_2atr",
        stoploss_delta_column="stoploss_1atr",
        initial_capital=10000,
    )
    assert result.final_capital > 0
    assert len(result.trades) > 0
    assert "can1_or_high_p15" in result.df.columns
    builtins.print(f"     Trades: {result.report['total_trades']}, Final: {result.final_capital:.2f}")


# ── STRATEGY 6: Multi-Timeframe (15min + 1min)
def s6_multi_tf():
    df = _gen_intraday(seed=6, days=3)
    df = add_higher_timeframe_indicators(
        df, rule="15T", add=["sma1"], rolling_minutes=[5, 15], prefix="can15",
    )
    df = add_indicators(df, add=["sma1", "zero"], rolling_minutes=[5])
    entry = [[
        condition("close", "can1_sma1_p5", lower=0),
        _cond("close", "can15_sma1_p15", lower=0),
    ]]
    result = run_backtest(
        df, entry,
        indicators=[], rolling_minutes=[],
        target_delta_normalized=0.5, stoploss_delta_normalized=0.25,
        initial_capital=10000,
    )
    assert result.final_capital > 0
    assert "can15_sma1_p15" in result.df.columns or "can15_sma1_p15" in df.columns
    builtins.print(f"     Trades: {result.report['total_trades']}, Final: {result.final_capital:.2f}")


# ── STRATEGY 7: Supertrend + PSAR Dual Confirmation
def s7_supertrend_psar():
    df = _gen_intraday(seed=7)
    entry = [[
        condition("can1_supertrend_dir_p10", lower=1),
        condition("close", "can1_psar", lower=0),
    ]]
    result = run_backtest(
        df, entry,
        exit_conditions=[[condition("can1_psar", "close", lower=0)]],
        indicators=["supertrend", "psar", "zero"], rolling_minutes=[10],
        target_delta_normalized=1.0, stoploss_delta_normalized=0.5,
        initial_capital=10000,
    )
    assert result.final_capital > 0
    assert "can1_supertrend_dir_p10" in result.df.columns
    assert "can1_psar" in result.df.columns
    builtins.print(f"     Trades: {result.report['total_trades']}, Final: {result.final_capital:.2f}")


# ── STRATEGY 8: Custom Indicator + CCI Mean Reversion
def s8_custom_cci():
    df = _gen_intraday(seed=8)
    df = add_indicators(df, add=["cci", "sma1", "zero"], rolling_minutes=[20, 50])
    df["custom_mom"] = (df["close"] - df["can1_sma1_p50"]) / df["can1_sma1_p50"] * 100
    entry = [[
        condition("can1_cci_p20", upper=-80),
        _cond("custom_mom", upper=0),
    ]]
    result = run_backtest(
        df, entry,
        exit_conditions=[[condition("custom_mom", lower=0)]],
        indicators=[], rolling_minutes=[],
        target_delta_normalized=1.0, stoploss_delta_normalized=0.5,
        initial_capital=10000,
    )
    assert result.final_capital > 0
    assert "custom_mom" in result.df.columns
    builtins.print(f"     Trades: {result.report['total_trades']}, Final: {result.final_capital:.2f}")


# ── STRATEGY 9: Walk-Forward + Exit Optimization + Strategy Serialization
def s9_optimization():
    df = _gen_intraday(seed=9, days=10)
    df = add_indicators(df, add=["ema1", "rsi", "zero"], rolling_minutes=[20, 50, 14])

    splits = walk_forward_splits(df, train_days=4, test_days=1)
    assert len(splits) >= 2

    strat = Strategy(
        name="TestStrat", indicators=["ema1", "rsi"],
        rolling_minutes=[20, 50],
        entry_conditions=[[condition("can1_ema1_p20", "can1_ema1_p50", lower=0)]],
        exit_conditions=[[condition("can1_ema1_p50", "can1_ema1_p20", lower=0)]],
    )
    # Strategy.run with data that already has indicators - OK because run_backtest handles it
    res = strat.run(df)
    assert res.final_capital > 0

    d = strat.to_dict()
    loaded = Strategy.from_dict(d)
    assert loaded.name == strat.name

    entry = [[_cond("can1_ema1_p20", "can1_ema1_p50", lower=0)]]
    best, opt_df = find_best_exit(
        df, entry, buy_or_sell="buy",
        target_deltas_normalized=[0.25, 0.5, 1.0],
        stoploss_deltas_normalized=[0.125, 0.25, 0.5],
        metric="sharpe",
    )
    assert best is not None
    assert len(opt_df) == 9

    def factory(**kw):
        return Strategy(name="Rnd", indicators=["ema1"], rolling_minutes=[20],
                        entry_conditions=[[condition("can1_ema1_p20", "close", lower=0)]],
                        **kw)
    best_r, rdf = random_parameter_search(
        df, factory,
        param_space={"target_delta_normalized": [0.25, 0.5],
                     "stoploss_delta_normalized": [0.125, 0.25]},
        n_iter=4, metric="sharpe",
    )
    assert best_r is not None
    builtins.print(f"     Walk-forward splits: {len(splits)}, Best: target={best['target_delta_normalized']}, stop={best['stoploss_delta_normalized']}")


# ── STRATEGY 10: Portfolio + Risk Controls + Position Sizing
def s10_portfolio():
    symbols = ["AAPL", "TSLA", "GOOG"]
    data = {}
    for i, sym in enumerate(symbols):
        df = _gen_intraday(seed=10 + i, days=3)
        # Pre-add indicators manually
        data[sym] = add_indicators(df, add=["ema1", "rsi", "zero"], rolling_minutes=[9, 21, 14])

    strat = Strategy(
        name="Portfolio", indicators=["ema1", "rsi"],
        rolling_minutes=[9, 21],
        entry_conditions=[[
            condition("can1_ema1_p9", "can1_ema1_p21", lower=0),
            condition("can1_rsi_p14", lower=30, upper=80),
        ]],
        exit_conditions=[[condition("can1_ema1_p9", "can1_ema1_p21", upper=0)]],
        side="buy",
        target_delta_normalized=0.5, stoploss_delta_normalized=0.25,
        initial_capital=30000,
    )

    pr = run_portfolio(data, strat, initial_capital=30000)
    assert "results" in pr
    assert "equity" in pr
    assert "trades" in pr
    assert pr["final_capital"] > 0
    assert len(pr["results"]) == 3

    # Risk controls
    if not pr["trades"].empty:
        ctrl = apply_risk_controls(pr["trades"], max_trades_per_day=3, cooldown_bars=5)
        assert "allowed" in ctrl.columns
        builtins.print(f"     Risk-filtered trades: {ctrl['allowed'].sum()} / {len(ctrl)}")

    # Position sizing functions
    prices = np.array([100.0, 200.0, 150.0])
    eq = np.array([100000, 100000, 100000])

    qty = fixed_quantity_size(prices, 10)
    assert np.all(qty == 10)

    cap = fixed_capital_size(prices, 1000)
    assert np.isclose(cap[0], 10.0)

    pct = percent_equity_size(prices, eq, pct=0.5)
    assert np.isclose(pct[0], 500.0)

    risk = atr_risk_size(prices, np.array([5.0, 10.0, 7.5]), eq, risk_pct=0.01)
    assert np.isclose(risk[0], 200.0)

    # Cost model
    cm = india_intraday_cost_model()
    assert cm.cost_rate() > 0
    crypto_cost_model()  # just ensure it doesn't crash

    builtins.print(f"     Final capital: {pr['final_capital']:.2f}, Winners: {len(pr['results'])} symbols")


# ── Run all ──
if __name__ == "__main__":
    wrap("SMA Crossover (Long)", s1_sma_crossover)
    wrap("RSI Short Reversal", s2_rsi_short)
    wrap("MACD + Bollinger Multi-Condition", s3_macd_bollinger)
    wrap("Ichimoku Cloud Breakout", s4_ichimoku)
    wrap("Opening Range + ATR Dynamic Exit", s5_or_atr)
    wrap("Multi-Timeframe (15min/1min)", s6_multi_tf)
    wrap("Supertrend + PSAR Confirmation", s7_supertrend_psar)
    wrap("Custom Indicator + CCI", s8_custom_cci)
    wrap("Walk-Forward + Optimization", s9_optimization)
    wrap("Portfolio + Risk Controls", s10_portfolio)

    builtins.print(f"\n{'='*60}")
    builtins.print(f"  SUMMARY")
    builtins.print(f"{'='*60}")
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    builtins.print(f"  Total: {len(results)} | PASS: {passed} | FAIL: {failed}")
    for name, status, msg in results:
        builtins.print(f"  [{status}] {name}")
        if msg:
            builtins.print(f"         {msg}")
