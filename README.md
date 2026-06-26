# mtrader

Vectorized backtesting and forward/live-testing framework for intraday trading strategies.

`mtrader` helps you move through the full workflow:

1. Clean OHLCV data.
2. Add technical indicators.
3. Define entry and exit rules.
4. Run historical backtests.
5. Generate a standalone HTML report.
6. Convert the same tested strategy into a live signal generator.

The library is built around pandas/NumPy, with optional Numba and CuPy paths for faster workloads.

---

## Installation

```bash
pip install mtrader
```

For local development from this repository:

```bash
pip install -e .
```

Optional acceleration:

```bash
pip install numba
pip install cupy-cuda12x
```

---

## Minimal Data Format

Most APIs expect a pandas DataFrame with:

```text
datetime, open, high, low, close, volume
```

`datetime` must be a pandas datetime column sorted ascending.

Example:

```python
import pandas as pd

df = pd.read_csv("one_minute_data.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
```

---

## Quick Backtest

```python
from mtrader import condition, run_backtest

result = run_backtest(
    df,
    indicators=["sma1", "rsi", "zero"],
    rolling_minutes=[5, 14],
    entry_conditions=[[
        condition("close", "can1_sma1_p5", lower=0),
        condition("can1_rsi_p14", "zero", lower=50),
    ]],
    exit_conditions=[[condition("can1_sma1_p5", "close", lower=0)]],
    buy_or_sell="buy",
    target_delta_normalized=0.7,
    stoploss_delta_normalized=0.35,
    initial_capital=100000,
)

print(result.report)
print(result.trades.head())
```

`run_backtest()` returns a `BacktestResult`:

```python
result.df              # full DataFrame with indicators and backtest columns
result.trades          # trade log
result.final_capital
result.metrics
result.report
result.equity
```

---

## HTML Report

Generate a standalone HTML report with metrics, parameters, equity curve, drawdown, monthly returns, trade distribution, and trade log.

```python
result.to_html(
    "backtest_report.html",
    strategy_name="SMA + RSI Trend",
    parameters={
        "side": "buy",
        "target_delta_normalized": 0.7,
        "stoploss_delta_normalized": 0.35,
        "rolling_minutes": [5, 14],
    },
)
```

You can also call:

```python
from mtrader import html_backtest_report

html_backtest_report(
    result,
    output_path="backtest_report.html",
    title="Backtest Tearsheet",
    strategy_name="SMA + RSI Trend",
)
```

---

## Strategy Object

Use `Strategy` when you want one reusable definition for backtesting and live signals.

```python
from mtrader import Strategy, condition

strategy = Strategy(
    name="SMA RSI live-ready",
    indicators=["sma1", "ema1", "rsi", "atr", "vwap", "zero"],
    rolling_minutes=[14, 20],
    entry_conditions=[[
        condition("close", "can1_sma1_p20", lower=0),
        condition("can1_rsi_p14", "zero", lower=50),
    ]],
    exit_conditions=[[condition("can1_ema1_p20", "close", lower=0)]],
    side="buy",
    target_delta_normalized=0.7,
    stoploss_delta_normalized=0.35,
    initial_capital=100000,
)

backtest = strategy.run(df)
print(backtest.report)
```

### Save And Load A Strategy

Strategies can be saved to a simple JSON text file and loaded later.

```python
from mtrader import Strategy, load_strategy

strategy.save("sma_rsi_strategy.txt")

loaded = Strategy.load("sma_rsi_strategy.txt")
backtest = loaded.run(df)
```

The file is plain text, so you can inspect or version it:

```json
{
  "schema": "mtrader.strategy",
  "version": 1,
  "name": "SMA RSI live-ready",
  "indicators": ["sma1", "ema1", "rsi", "atr", "vwap", "zero"],
  "rolling_minutes": [14, 20],
  "side": "buy"
}
```

---

## Convert Backtest To Live Signal Generator

The same `Strategy` can become a live signal engine.

Indicator names stay the same in backtest and live mode:

| Indicator | Column |
|---|---|
| SMA close | `can1_sma1_p20` |
| EMA close | `can1_ema1_p20` |
| RSI | `can1_rsi_p14` |
| ATR | `can1_atr_p14` |
| VWAP | `can1_vwap` |

```python
warmup_df = df.tail(6 * 30 * 24 * 60)  # example: last 6 months of 1-minute bars
live = strategy.to_live(warmup_df, history_size=5000)
```

During this warm-up phase, mtrader uses the batch backtest indicator engine on
the historical data first. After warm-up, each call updates only the new candle
and returns the latest trading signal.

Then use it in a loop with broker candles, websocket candles, or any generator that yields one OHLCV bar at a time:

```python
while market_is_open():
    new_1m_candle = broker.get_latest_1m_candle()
    signal = live.update(new_1m_candle)

    print(signal["datetime"], signal["action"])

    if signal["action"] == "BUY":
        place_buy_order()
    elif signal["action"] == "EXIT_BUY":
        exit_long_position()
```

For long strategies, `action` is:

```text
BUY, EXIT_BUY, HOLD
```

For short strategies, `action` is:

```text
SELL, EXIT_SELL, HOLD
```

Each live update is O(1) per configured indicator/period where possible:

- SMA: rolling sum/deque
- EMA: recursive update
- RSI: recursive gain/loss state
- ATR: rolling true-range state
- VWAP: session cumulative price-volume/volume

---

## Live Engine Without Strategy

If you do not want to use `Strategy`, use `LiveIndicatorEngine` directly.

```python
from mtrader import LiveIndicatorEngine, condition, cross_below

buy_conditions = [[
    condition("close", "can1_sma1_p20", lower=0),
    condition("can1_rsi_p14", "zero", lower=50),
]]

sell_conditions = [cross_below("close", "can1_ema1_p20")]

engine = LiveIndicatorEngine.from_history(
    warmup_df,
    indicators=["sma", "ema", "rsi", "atr", "vwap", "zero"],
    periods=[14, 20],
    buy_conditions=buy_conditions,
    sell_conditions=sell_conditions,
    history_size=5000,
)

while market_is_open():
    new_1m_candle = broker.get_latest_1m_candle()
    latest = engine.update(new_1m_candle)

    if latest["buy_signal"]:
        print("buy")
    elif latest["sell_signal"]:
        print("sell")
```

A candle can be a dict or pandas Series:

```python
new_candle = {
    "datetime": "2024-01-02 10:31:00",
    "open": 100.0,
    "high": 101.0,
    "low": 99.5,
    "close": 100.7,
    "volume": 2500,
}

latest = engine.update(new_candle)
```

---

## Condition Helpers

Conditions are OR groups containing AND rules.

```python
from mtrader import condition, cross_above, cross_below

entry_conditions = [[
    condition("close", "can1_sma1_p20", lower=0),
    *cross_above("can1_ema1_p5", "can1_ema1_p20"),
]]

exit_conditions = [
    cross_below("can1_ema1_p5", "can1_ema1_p20")
]
```

The equivalent raw condition format is:

```python
{
    "first_column_name": "close",
    "second_column_name": "can1_sma1_p20",
    "shift_down_first": 0,
    "shift_down_second": 0,
    "lower_range_of_difference": 0,
    "upper_range_of_difference": float("inf"),
    "perform_normalization_of_diff": False,
}
```

---

## Supported Indicators

### Batch Backtesting Indicators

Use `add_indicators(df, add=[...], rolling_minutes=[...])` or pass them to `run_backtest()` / `Strategy`.

Common indicator codes:

| Code | Output example |
|---|---|
| `sma1` | `can1_sma1_p20` |
| `ema1` | `can1_ema1_p20` |
| `wma1` | `can1_wma1_p20` |
| `ssma1` | `can1_ssma1_p20` |
| `rsi` | `can1_rsi_p14` |
| `atr` | `can1_atr_p14` |
| `stochk` | `can1_stochk_p14` |
| `stochd` | `can1_stochd_p14` |
| `bbp` | `can1_bbp_p20` |
| `willr` | `can1_willr_p14` |
| `cci` | `can1_cci_p20` |
| `adx` | `can1_adx_p14` |
| `mfi` | `can1_mfi_p14` |
| `vwap` | `can1_vwap` |
| `obv` | `can1_obv` |
| `macd` | `can1_macd`, `can1_macdsignal`, `can1_macdhist` |
| `psar` | `can1_psar` |
| `ha` | Heikin Ashi columns |
| `supertrend` | `can1_supertrend_p10`, `can1_supertrend_dir_p10` |
| `ichimoku` | Ichimoku columns |
| `pivot` | daily pivot columns |
| `prev_day` | previous day high/low/close |
| `gap` | `can1_gap_pct` |
| `inside_bar` | `can1_inside_bar` |
| `engulfing` | bullish/bearish engulfing columns |

Example:

```python
from mtrader import add_indicators

df = add_indicators(
    df,
    add=["sma1", "ema1", "rsi", "atr", "vwap", "macd"],
    rolling_minutes=[14, 20],
)
```

### Live Indicators

The live engine accepts the same indicator names used by `Strategy` / `add_indicators()`.
When created with `from_history()` or `Strategy.to_live()`, it first runs a
warm-up phase with the batch backtest indicator calculator. This is the right
place to pass the last 3-6 months of one-minute candles, or whatever history
your longest indicator needs.

Core indicators are updated incrementally in O(1) per new candle:

```text
sma, ema, rsi, atr, vwap, zero
```

Use these in `LiveIndicatorEngine`:

```python
indicators=["sma", "ema", "rsi", "atr", "vwap", "zero"]
periods=[14, 20]
```

Use these in `Strategy`:

```python
indicators=["sma1", "ema1", "rsi", "atr", "vwap", "zero"]
rolling_minutes=[14, 20]
```

All other batch indicators are also available in live mode through a rolling
buffer fallback that recalculates only the stored live buffer and copies the
newest row's indicator values into the signal. This makes indicators such as
`wma1`, `ssma1`, `stochk`, `stochd`, `bbp`, `willr`, `cci`, `adx`, `mfi`,
`macd`, `obv`, `psar`, `ha`, `supertrend`, `ichimoku`, `pivot`, `prev_day`,
`gap`, `inside_bar`, and `engulfing` usable in forward testing too.

For heavy indicators, set enough `history_size` when creating the live engine:

```python
engine = LiveIndicatorEngine.from_history(
    df,
    indicators=["macd", "supertrend", "ichimoku", "pivot", "zero"],
    periods=[10, 14, 52],
    buy_conditions=buy_conditions,
    sell_conditions=sell_conditions,
    history_size=1000,
)
```

---

## Data Cleaning

```python
from mtrader import clean_data

df = clean_data(
    raw_df,
    start_time="09:15",
    end_time="15:30",
    start_date="2024-01-01",
    end_date="2024-12-31",
    fill_gap=True,
)
```

`clean_data()` can:

- detect datetime and numeric columns
- normalize OHLCV columns
- round timestamps to the minute
- remove duplicates
- filter dates and session time
- fill missing intraday rows
- handle stock split-like jumps

---

## Low-Level Backtest Pipeline

Use this if you want full control.

```python
import numpy as np
from mtrader import add_indicators
from mtrader import precalculate_exit_time_amount_profit
from mtrader import take_trade_on_condition_numpy

df = add_indicators(
    df,
    add=["sma1", "close", "high", "low", "zero"],
    rolling_minutes=[20],
)

entry_conditions = [[
    {
        "first_column_name": "close",
        "second_column_name": "can1_sma1_p20",
        "shift_down_first": 0,
        "shift_down_second": 0,
        "lower_range_of_difference": 0,
        "upper_range_of_difference": np.inf,
        "perform_normalization_of_diff": False,
    }
]]

df = precalculate_exit_time_amount_profit(
    df,
    entry_conditions,
    buy_or_sell="buy",
    target_delta_normalized=0.7,
    stoploss_delta_normalized=0.35,
)

trades, final_capital, metrics = take_trade_on_condition_numpy(
    df,
    entry_conditions,
    initial_capital=100000,
)
```

---

## Exit Optimization

```python
from mtrader import find_best_exit

best_params, results_df = find_best_exit(
    df,
    entry_conditions=entry_conditions,
    buy_or_sell="buy",
    target_deltas_normalized=[0.5, 0.7, 1.0],
    stoploss_deltas_normalized=[0.25, 0.35, 0.5],
    metric="sharpe",
)

print(best_params)
print(results_df.sort_values("sharpe", ascending=False).head())
```

---

## Portfolio Backtesting

```python
from mtrader import run_portfolio

portfolio = run_portfolio(
    {
        "RELIANCE": reliance_df,
        "TCS": tcs_df,
        "INFY": infy_df,
    },
    strategy,
    initial_capital=300000,
)

print(portfolio["final_capital"])
print(portfolio["trades"].head())
```

---

## Multi-Timeframe Indicators

Example: use 15-minute EMA on 1-minute data.

```python
from mtrader import add_higher_timeframe_indicators

df = add_higher_timeframe_indicators(
    df,
    "15min",
    add=["ema1"],
    rolling_minutes=[20],
)

print(df["can15_ema1_p20"])
```

---

## Risk, Costs, And Sizing

```python
from mtrader import (
    india_intraday_cost_model,
    fixed_capital_size,
    percent_equity_size,
    atr_risk_size,
    apply_risk_controls,
)

costs = india_intraday_cost_model()
estimated_cost = costs.estimate(turnover=100000)

qty = fixed_capital_size(entry_price=100, capital_per_trade=10000)

controlled_trades = apply_risk_controls(
    result.trades,
    max_trades_per_day=3,
    cooldown_bars=5,
    max_daily_loss_pct=2.0,
)
```

---

## Walk-Forward Optimization

```python
from mtrader import Strategy, condition, grid_from_ranges, walk_forward_optimize

def make_strategy(period=20, target=0.7):
    return Strategy(
        name="walk-forward sma",
        indicators=["sma1", "zero"],
        rolling_minutes=[period],
        entry_conditions=[[condition("close", f"can1_sma1_p{period}", lower=0)]],
        exit_conditions=[[condition(f"can1_sma1_p{period}", "close", lower=0)]],
        target_delta_normalized=target,
        stoploss_delta_normalized=0.35,
    )

grid = grid_from_ranges(period=[10, 20, 50], target=[0.5, 0.7, 1.0])

wf = walk_forward_optimize(
    df,
    strategy_factory=make_strategy,
    param_grid=grid,
    train_days=60,
    test_days=20,
)

print(wf)
```

---

## Random Parameter Search

```python
from mtrader import random_parameter_search

best, runs = random_parameter_search(
    df,
    strategy_factory=make_strategy,
    param_space={
        "period": [10, 20, 50],
        "target": [0.5, 0.7, 1.0],
    },
    n_iter=20,
    seed=42,
)

print(best["params"])
print(runs.head())
```

---

## Testing

Run the package test suite:

```bash
python -m pytest -q
```

Basic syntax check:

```bash
python -m py_compile src/mtrader/*.py
```

---

## Notes For Live Trading

`mtrader` generates signals. It does not place broker orders by itself.

Typical production flow:

```text
broker/websocket candle -> live.stream(...) -> signal/action -> your order adapter
```

Recommended checks before connecting real money:

- verify timezone and session boundaries
- verify candle close timing
- paper trade first
- log every signal and order response
- add broker-side risk limits
- compare live indicator values against batch `add_indicators()` during dry runs

---

## Dependencies

Required:

- `numpy`
- `pandas`
- `inspecty`

Optional:

- `numba` for monotonic-stack acceleration
- `cupy-cuda12x` for GPU paths

---

## License

MIT
