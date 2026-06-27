# mtrader

Vectorized backtesting & forward/live-testing framework for intraday trading strategies.  
**38+ indicators, 7 exit types, position sizing, risk controls, walk-forward optimization, auto strategy discovery.**

```python
import mtrader as mt

result = mt.run_backtest(df, entry_conditions, buy_or_sell="buy",
                          indicators=["sma1", "rsi"], rolling_minutes=[20, 14])
result.to_html("report.html")
```

---

## Quick start

```python
import pandas as pd
import numpy as np
import mtrader as mt

# 1. Load & clean data
df = pd.read_csv("ticks.csv")
df = mt.clean_data(df, start_time="09:15", end_time="15:30")

# 2. Define strategy
entry = mt.cross_above("can1_sma1_p20", "close")
exit_cond = mt.cross_below("can1_rsi_p14", mt.condition("close", upper=70))

# 3. Run backtest
result = mt.run_backtest(
    df, entry, buy_or_sell="buy", exit_conditions=exit_cond,
    indicators=["sma1", "rsi"], rolling_minutes=[20, 14],
    target_delta_normalized=0.5, stoploss_delta_normalized=0.25,
)
print(result.metrics)
result.to_html("backtest.html")
```

---

## Pipeline

```
raw CSV/DataFrame
    │
    ▼
clean_data()              — auto-detect types, normalize OHLCV, fill gaps, handle splits
    │
    ▼
add_indicators()          — 38+ indicators across multiple rolling windows
    │
    ▼
add_trailing_stop_column()   — trailing stop price (optional)
add_time_filter_column()     — restrict to trading hours (optional)
add_regime_filter_column()   — ADX-based trend/ranging filter (optional)
    │
    ▼
precalculate_exit_time_amount_profit()
    │                       — exit signals: conditional, target, stoploss, trailing
    │
    ▼
take_trade_on_condition*()
    │                       — capital simulation (with sizing_fn, risk controls, hold filters)
    │
    ▼
backtest_report()          — Sharpe, Sortino, Calmar, win rate, profit factor, drawdown
equity_curve()             — per-bar equity + drawdown
html_backtest_report()     — standalone HTML with SVG charts
trade_log()                — per-trade log with exit reason, hold bars
```

---

## One-call backtest

```python
from mtrader import run_backtest

result = run_backtest(
    df,
    entry_conditions=mt.cross_above("can1_sma1_p20", "close"),
    buy_or_sell="buy",
    exit_conditions=mt.cross_below("can1_ema1_p9", "can1_ema1_p21"),
    indicators=["sma1", "ema1", "rsi"],
    rolling_minutes=[20, 9, 14],
    target_delta_normalized=0.5,
    stoploss_delta_normalized=0.25,
    initial_capital=100000,
)

result.final_capital     # 112345.67
result.metrics           # {'Sharpe Ratio': 1.23, 'sortino_ratio': 1.45, 'win_rate_pct': 55.0, ...}
result.report            # backtest_report dict (extended)
result.equity            # equity_curve DataFrame
result.trades            # trade_log DataFrame (with exit_reason, hold_bars)
result.to_html("report.html")
```

---

## Position Sizing

Control how much capital is deployed per trade — fixed fraction or dynamic callable.

### Fixed fraction

```python
result = run_backtest(df, entry, indicators=[], rolling_minutes=[],
                      initial_capital=100000, capital_per_trade_pct=0.25)
# Only 25% of capital at risk per trade. Remaining 75% stays as cash.
```

### Dynamic sizing callable

Use `sizing_fn` for ATR-based, Kelly, or equity-curve sizing:

```python
def atr_sizing(entry_idx, capital_before, df):
    atr_pct = df.loc[entry_idx, "can1_atr_p14"] / df.loc[entry_idx, "close"]
    return min(0.5, 0.02 / max(atr_pct, 0.001))  # risk 2% per ATR unit

result = run_backtest(df, entry, indicators=[], rolling_minutes=[],
                      initial_capital=100000, sizing_fn=atr_sizing)
```

### Position sizing utilities

```python
from mtrader import (
    fixed_quantity_size, fixed_capital_size,
    percent_equity_size, atr_risk_size,
)

qty = atr_risk_size(price, atr_values, equity=100000, risk_pct=0.01, atr_multiple=2)
# Returns quantity of shares to take: equity * risk_pct / (atr * atr_multiple)
```

---

## Risk Controls

Apply max-trades-per-day, cooldown, and max-daily-loss directly in `run_backtest`:

```python
result = run_backtest(
    df, entry, indicators=[], rolling_minutes=[],
    capital_per_trade_pct=0.5,
    max_trades_per_day=3,       # at most 3 entries per day
    cooldown_bars=5,            # wait 5 bars between trades
    max_daily_loss_pct=2.0,     # stop trading for the day if -2%
)
```

Post-hoc filtering also available:

```python
from mtrader import apply_risk_controls
controlled = apply_risk_controls(trades, max_trades_per_day=3, cooldown_bars=5)
filtered = trades[controlled["allowed"]]
```

---

## Holding Period Filters

Skip trades that exit too quickly or hold too long:

```python
result = run_backtest(df, entry, indicators=[], rolling_minutes=[],
                      min_hold_bars=5,     # skip trades < 5 bars
                      max_hold_bars=50)    # skip trades > 50 bars
```

---

## Exit Strategies

### Target & stoploss

```python
# Fixed price delta
result = run_backtest(df, entry, target_delta=200, stoploss_delta=100)

# Normalized (% of price / 10000)
result = run_backtest(df, entry, target_delta_normalized=1.0, stoploss_delta_normalized=0.5)

# Dynamic column-based (e.g., ATR multiples)
df["target_2atr"] = df["can1_atr_p14"] * 2.0
df["stoploss_1atr"] = df["can1_atr_p14"]
result = run_backtest(df, entry, target_delta_column="target_2atr",
                      stoploss_delta_column="stoploss_1atr")
```

### Trailing stop

```python
from mtrader import add_trailing_stop_column

df = add_trailing_stop_column(df, trail_pct=0.5, lookback=20)
df["trail_delta"] = df["close"] - df["trailing_stop_price"]
result = run_backtest(df, entry, stoploss_delta_column="trail_delta")
```

### Condition-based exit

```python
# Exit when RSI crosses above 70
exit_cond = mt.cross_above("can1_rsi_p14", mt.condition("close", upper=70))
result = run_backtest(df, entry, exit_conditions=exit_cond)
```

---

## Filters

### Time filter (trading hours)

```python
from mtrader import add_time_filter_column

df = add_time_filter_column(df, start_time="09:45", end_time="14:30")
entry = [[
    condition("close", "can1_sma1_p20", lower=0),
    condition("time_filter", lower=1),     # only trade 09:45–14:30
]]
result = run_backtest(df, entry)
```

### Regime filter (ADX trend detection)

```python
from mtrader import add_regime_filter_column

df = add_regime_filter_column(df, adx_period=14, adx_threshold=25.0)
entry = [[
    condition("close", "can1_ema1_p20", lower=0),
    condition("regime_filter", lower=1),   # only trade trending markets
]]
result = run_backtest(df, entry)
# Invert for ranging markets: condition("regime_filter", upper=1)
```

---

## Indicators

### Moving averages

| Code | Description |
|------|-------------|
| `sma` | Simple Moving Average |
| `ema` | Exponential Moving Average |
| `wma` | Weighted Moving Average |
| `ssma` | Smoothed Simple Moving Average |

### Popular technical indicators

| Code | Description |
|------|-------------|
| `rsi` | Relative Strength Index |
| `atr` | Average True Range |
| `macd` | MACD line, signal, histogram |
| `stochk` / `stochd` | Stochastic %K / %D |
| `bbp` | Bollinger Band %B |
| `cci` | Commodity Channel Index |
| `willr` | Williams %R |
| `adx` | Average Directional Index |
| `mfi` | Money Flow Index (volume-weighted) |
| `obv` | On-Balance Volume |
| `psar` | Parabolic SAR |
| `supertrend` | SuperTrend (direction 1/-1) |
| `ichimoku` | Ichimoku Cloud (5 lines) |
| `ha` | Heikin Ashi candles |
| `vwap` / `ewap` / `iwap` | Volume / Equal / Incremental WAP |

### Candlestick patterns

| Code | Description |
|------|-------------|
| `inside_bar` | 1 when bar inside previous bar's range |
| `engulfing` | 1 bullish, -1 bearish engulfing |

### Session-aware

| Code | Description |
|------|-------------|
| `or_high` / `or_low` | Opening range expanding high/low |
| `prev_day` | Previous day high/low/close |
| `pivot` | Classic pivot points (P, S1, S2, R1, R2) |
| `gap` | Gap up/down detection |

### Feature codes (base signals)

| Code | Signal |
|------|--------|
| `0`, `1` | `close` |
| `2` | `av2` (H+L)/2 |
| `3` | `av3` (H+L+C)/3 |
| `4` | `av4` (O+H+L+C)/4 |
| `5` | `open` |
| `6` | `high` |
| `7` | `low` |
| `8..34` | `dif`, `ret`, `lret` for 1/3/5/7/10/15/20/30/60 bars |

### Normalized indicators

Prefix for ratio/statistical forms: `SMN_`, `EMN_`, `WMN_`, `SSMN_`, `SVN_`, `EVN_`, `WVN_`, `SSVN_`, `Z_`, `BRN_`, `TMN_`.

Suffix controls denominator: `""`/`F` = signal itself, `P`/`0` = close, `B` = base signal, numeric code = that feature.

```python
df = add_indicators(df, add=["Z_sma1", "SMN_ema1", "EMN_rsi"], rolling_minutes=[14])
# can1_Z_sma1_p14  — Z-score of SMA(14)
# can1_SMN_ema1_p14 — EMA(14) / SMA(14)
```

### Distance indicators

```python
df = add_indicators(df, add=["smadis1", "emadis1"], rolling_minutes=[14])
# can1_smadis1_p14 — close - SMA(14)
# can1_emadis1_p14 — close - EMA(14)
```

### Column naming

```
can1_{indicator}_p{period}
can1_{normalization}_{indicator}_p{period}
```

Examples:
- `can1_sma1_p20` — SMA(20) of close
- `can1_rsi_p14` — RSI(14)
- `can1_macd` — MACD line (no period)
- `can1_supertrend_dir_p10` — SuperTrend direction
- `can1_Z_sma1_p14` — Z-score of SMA(14)
- `can1_ichi_span_a` — Ichimoku cloud span A

---

## Conditions

Conditions compare two columns: `(first - second) ∈ [lower, upper]`.

```python
{
    "first_column_name": "can1_sma1_p20",
    "second_column_name": "close",
    "shift_down_first": 0,
    "shift_down_second": 0,
    "lower_range_of_difference": 0,       # first - second >= lower
    "upper_range_of_difference": np.inf,  # first - second <= upper
    "perform_normalization_of_diff": False,
}
```

### Helper functions

```python
from mtrader import condition, cross_above, cross_below

cond = condition("close", "can1_sma1_p20", lower=0)            # close > sma
crossover = cross_above("can1_macd", "can1_macdsignal")         # MACD crossover
crossunder = cross_below("can1_stochk_p14", "can1_stochd_p14")  # Stoch crossunder
```

### AND / OR logic

**AND within a group, OR across groups:**

```python
entry = [
    # Group 1: both must be true
    [
        condition("close", "can1_sma1_p20", lower=0),
        condition("can1_rsi_p14", lower=30, upper=70),
    ],
    # Group 2: OR with Group 1
    [
        condition("can1_macd", "can1_macdsignal", lower=0),
    ],
]
```

### update_cond helper

Modify conditions programmatically:

```python
from mtrader import update_cond

base_entry = [[condition("ema9", "ema21", lower=0)]]

# Update ALL conditions (default behavior)
updated = update_cond(base_entry, "ema20", "ema50", lower=-5)

# Or update only matching a specific first_column_name
updated = update_cond(base_entry, "ema20", "ema50",
                      match_first="ema9")  # only updates conditions with first="ema9"
```

---

## Strategy object (serializable)

```python
from mtrader import Strategy

strat = Strategy(
    name="SMA Crossover",
    indicators=["sma1", "ema1"],
    rolling_minutes=[20, 50],
    entry_conditions=cross_above("can1_sma1_p20", "can1_sma1_p50"),
    exit_conditions=cross_below("can1_sma1_p20", "can1_sma1_p50"),
    side="buy",
    target_delta_normalized=0.5,
    stoploss_delta_normalized=0.25,
    capital_per_trade_pct=0.5,
    max_trades_per_day=3,
    cooldown_bars=3,
)

result = strat.run(df)
strat.save("strat.json")
loaded = Strategy.load("strat.json")

live_engine = strat.to_live(df)  # convert to live engine with warmup
```

---

## Exit Optimization

Find the best target/stoploss combination via grid search:

```python
from mtrader import find_best_exit

best, results = find_best_exit(
    df, entry_conditions, buy_or_sell="buy",
    target_deltas=[50, 100, 150, 200],
    stoploss_deltas=[25, 50, 75, 100],
    metric="sharpe",
    verbose=True,
)
print(best)  # {'target_delta': 150, 'stoploss_delta': 75}
```

### Entry condition optimization

Grid-search for the best entry threshold values:

```python
from mtrader import find_best_entry_conditions

entry = [[
    {"first_column_name": "can1_rsi_p14", "second_column_name": "zero",
     "lower_range_of_difference": -np.inf, "upper_range_of_difference": 30, ...}
]]
best, results = find_best_entry_conditions(
    df, entry,
    condition_ranges={0: ([-np.inf, -np.inf], [20, 30])},  # try RSI < 20, RSI < 30
    metric="sharpe",
)
print(best)  # {'cond_0_lower': -inf, 'cond_0_upper': 20}
```

---

## Walk-Forward Optimization

```python
from mtrader import walk_forward_optimize, Strategy

results = walk_forward_optimize(
    df,
    strategy_factory=lambda params: Strategy(**params),
    param_grid={"rolling_minutes": [[10, 30], [20, 50]]},
    train_days=60,
    test_days=20,
    metric="sharpe",
)
```

### Random parameter search

```python
from mtrader import random_parameter_search

def factory(**kw):
    return Strategy(name="Opt", indicators=["ema1"], rolling_minutes=[20],
                    entry_conditions=[[condition("can1_ema1_p20", "close", lower=0)]],
                    **kw)

best, results_df = random_parameter_search(
    df, factory,
    param_space={"target_delta_normalized": [0.25, 0.5, 0.75, 1.0],
                 "stoploss_delta_normalized": [0.125, 0.25, 0.5]},
    n_iter=20, metric="sharpe",
)
```

---

## Scenario Sweeper

Run the same strategy across multiple parameter combos and compare results:

```python
from mtrader import run_scenarios

base = {
    "entry_conditions": entry,
    "indicators": [], "rolling_minutes": [],
    "initial_capital": 100000,
}
grid = {
    "target_delta_normalized": [0.25, 0.5, 1.0, 2.0],
    "stoploss_delta_normalized": [0.125, 0.25, 0.5],
}

scenarios = run_scenarios(df, base, grid, metric="sharpe", verbose=True)
# Returns sorted DataFrame: target, stoploss, final_capital, sharpe, sortino,
#                           calmar, win_rate, profit_factor, total_trades
best_params = scenarios.iloc[0]
```

---

## Auto Strategy Discovery

Automatically generate, evaluate, and rank trading strategies from 7 indicator families:

```python
from mtrader import discover_strategies

results, best_candidates = discover_strategies(
    df,
    train_days=5,               # training window per fold
    test_days=1,                # test window per fold
    strategy_types=["crossover", "threshold", "macd", "psar", "supertrend",
                    "vwap", "price_action"],
    exit_targets=[0.5, 1.0],    # try these target levels
    exit_stops=[0.25, 0.5],     # try these stoploss levels
    metric="sharpe",             # ranking metric
    top_n=10,                   # walk-forward validate top 10
    verbose=True,
)

# results: DataFrame with train/test scores per candidate
# best_candidates: list of StrategyCandidate objects for the top strategies
```

### Quick-rank pre-defined strategies

Faster than `discover_strategies` — pre-computes all indicators once:

```python
from mtrader import quick_rank_strategies, StrategyCandidate

strategies = [
    StrategyCandidate(name="SMA Crossover", indicators=["sma1"],
                      rolling_minutes=[20, 50],
                      entry_conditions=[[_cond("can1_sma1_p20", "can1_sma1_p50", lower=0)]]),
    StrategyCandidate(name="RSI Oversold", indicators=["rsi"],
                      rolling_minutes=[14],
                      entry_conditions=[[_cond("can1_rsi_p14", upper=30)]]),
]
ranked = quick_rank_strategies(df, strategies, metric="sharpe")
```

---

## Trade log (enhanced)

The trade log now includes `exit_reason` and `hold_bars` columns:

```python
result = run_backtest(df, entry, exit_conditions=exit_cond,
                      target_delta_normalized=1.0, stoploss_delta_normalized=0.5)
print(result.trades.columns)
# ['entry_index', 'entry_time', 'exit_index', 'exit_time', 'side',
#  'entry_price', 'exit_price', 'profit', 'return_pct', 'capital_at_exit',
#  'capital_before', 'capital_return_pct', 'exit_reason', 'hold_bars']

print(result.trades["exit_reason"].value_counts())
# target      14
# stoploss     7
# condition    3
# end          1
```

Exit reasons: `"target"` (hit profit target), `"stoploss"` (hit stop), `"condition"` (exit condition triggered), `"end"` (ran to end of data).

---

## Performance reports

```python
from mtrader import backtest_report, equity_curve, html_backtest_report

report = backtest_report(df, initial_capital=1000)
# total_trades, win_rate_pct, profit_factor, avg_win/loss_pct,
# sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown_pct,
# max_consecutive_wins/losses, ...

# Enhanced BacktestResult.metrics includes:
# Sharpe Ratio, Volatility, Max Drawdown, sortino_ratio,
# calmar_ratio, win_rate_pct, profit_factor, total_trades

eq = equity_curve(df)
# columns: datetime, equity, drawdown_pct, trade

html_backtest_report(result, output_path="report.html")
# standalone HTML with SVG charts, no external dependencies
```

---

## Live trading

```python
from mtrader import live_strategy_from_history, stream_live_signals

engine = live_strategy_from_history(
    df, indicators=["sma1", "ema1", "rsi"],
    periods=[20, 50, 14],
    entry_conditions=cross_above("can1_sma1_p20", "can1_sma1_p50"),
    side="buy",
)

for signal in stream_live_signals(engine, candle_feed):
    if signal["action"] == "BUY":
        place_order(signal)
```

Incremental indicators (SMA, EMA, RSI, ATR, VWAP) update in O(1) per bar.

---

## Data cleaning

```python
from mtrader import clean_data

cleaned = clean_data(
    raw_df,
    start_time="09:15", end_time="15:30",
    start_date="2024-01-01", end_date="2024-12-31",
    fill_gap=True, adjustsplit=True, multiplier=100,
)
```

Auto-detects: datetime columns, OHLCV columns, stock splits/reverse mergers, gap-fills to 1-min frequency.

---

## Multi-timeframe indicators

```python
from mtrader import add_higher_timeframe_indicators

# Compute 15-min EMA(5) and SMA(15) merged onto 1-min bars
df = add_higher_timeframe_indicators(
    df, rule="15min", add=["sma1", "ema1"],
    rolling_minutes=[5, 15], prefix="can15",
)
# Columns: can15_sma1_p5, can15_ema1_p15

# Entry: 1-min EMA > 15-min SMA AND 15-min EMA > 60-min SMA
entry = [[
    condition("can1_ema1_p20", "can15_sma1_p15", lower=0),
    condition("can15_ema1_p15", "can60_ema1_p5", lower=0),
]]
```

---

## Multi-symbol portfolio

```python
from mtrader import run_portfolio

data = {"AAPL": df_aapl, "TSLA": df_tsla, "GOOG": df_goog}
result = run_portfolio(data, strategy, initial_capital=300000)
# result.equity — combined portfolio equity curve
# result.trades — all trades with symbol column
# result.results — per-symbol BacktestResult objects
```

---

## Complete Strategy Examples

### 1. Short selling (RSI overbought fade)

```python
result = run_backtest(
    df, entry_conditions=[[condition("can1_rsi_p14", lower=70)]],
    buy_or_sell="sell",
    exit_conditions=[[condition("can1_rsi_p14", upper=40)]],
    target_delta_normalized=0.5, stoploss_delta_normalized=0.25,
)
```

### 2. MACD + Bollinger Band squeeze

```python
entry = [[
    condition("can1_macd", "can1_macdsignal", lower=0),
    condition("can1_bbp_p20", lower=0.2, upper=0.8),
]]
result = run_backtest(df, entry,
    exit_conditions=[[condition("can1_macdsignal", "can1_macd", lower=0)]],
    capital_per_trade_pct=0.5, max_trades_per_day=3)
```

### 3. Multi-TF trend with trailing stop

```python
df = add_higher_timeframe_indicators(df, "15min", add=["ema1"], ...)
df = add_trailing_stop_column(df, trail_pct=0.5, lookback=20)
df["trail_delta"] = df["close"] - df["trailing_stop_price"]
df = add_time_filter_column(df, start_time="09:45", end_time="15:00")
df = add_regime_filter_column(df, adx_threshold=20)

entry = [[
    condition("can1_ema1_p20", "can15_ema1_p15", lower=0),
    condition("time_filter", lower=1),
    condition("regime_filter", lower=1),
]]
result = run_backtest(df, entry, stoploss_delta_column="trail_delta",
                      capital_per_trade_pct=0.3, max_trades_per_day=2)
```

### 4. Ichimoku cloud breakout

```python
entry = [[
    condition("close", "can1_ichi_span_a", lower=0),
    condition("close", "can1_ichi_span_b", lower=0),
]]
result = run_backtest(df, entry,
    exit_conditions=[[condition("can1_ichi_kijun", "close", lower=0)]])
```

### 5. Opening range breakout with ATR exits

```python
df = add_indicators(df, add=["or_high", "atr"], rolling_minutes=[15, 14])
df["target_2atr"] = df["can1_atr_p14"] * 2.0
df["stoploss_1atr"] = df["can1_atr_p14"]

entry = [[condition("close", "can1_or_high_p15", lower=0)]]
result = run_backtest(df, entry, target_delta_column="target_2atr",
                      stoploss_delta_column="stoploss_1atr")
```

### 6. SuperTrend + PSAR dual confirmation

```python
entry = [[
    condition("can1_supertrend_dir_p10", lower=1),
    condition("close", "can1_psar", lower=0),
]]
result = run_backtest(df, entry,
    exit_conditions=[[condition("can1_psar", "close", lower=0)]])
```

### 7. Long-short combined

```python
r_long  = run_backtest(df, long_entry,  buy_or_sell="buy")
r_short = run_backtest(df, short_entry, buy_or_sell="sell")
combined_pnl = r_long.final_capital + r_short.final_capital - 2 * initial_capital
```

---

## Modules

| Module | Key exports |
|--------|-------------|
| `data_cleaner` | `clean_data`, `detect_data_types_with_formats`, `fill_missing_rows` |
| `indicators` | All standalone indicator functions (ema, rsi, macd, psar, ichimoku, ...) |
| `indicator_engine` | `add_indicators`, `add_indicators_on_group`, `FEATURE_CODE` |
| `exit_strategy` | `precalculate_exit_time_amount_profit` |
| `trading` | `take_trade_on_condition*`, `update_cond` |
| `optimize_exit` | `find_best_exit` |
| `backtest` | `run_backtest`, `BacktestResult`, `condition`, `cross_above/below`, `walk_forward_splits`, `parameter_grid`, `trade_log`, `run_scenarios`, `find_best_entry_conditions` |
| `advanced` | `Strategy`, `CostModel`, sizing functions, `run_portfolio`, `walk_forward_optimize`, `random_parameter_search`, `resample_ohlcv`, `add_higher_timeframe_indicators`, `add_trailing_stop_column`, `add_time_filter_column`, `add_regime_filter_column`, `apply_risk_controls` |
| `strategy_discovery` | `discover_strategies`, `quick_rank_strategies`, `StrategyCandidate` |
| `live` | `LiveIndicatorEngine`, `LiveStrategyEngine`, `stream_live_signals`, `live_strategy_from_history` |
| `report` | `backtest_report`, `equity_curve`, `html_backtest_report` |
| `monotonic_stack` | `monotonic_stack_for_value1_gt/lt_value2` |
| `utils` | `timenum` |

---

## Testing

```bash
python -m pytest src/tests/ -v
```

**163+ tests** cover: data cleaning, all indicators, exit precalculation, trade simulation (NumPy + CuPy), exit optimization, performance reports, 20 strategy scenarios, edge cases, position sizing, risk controls, trailing stop, time/regime filters, entry optimization, scenario sweeper, exit reason tagging, and 6 complex strategy pipelines.

---

## Dependencies

- **Required:** `numpy>=1.21`, `pandas>=1.3`, `inspecty>=0.1`
- **Optional:** `numba>=0.58` (monotonic stack), `cupy-cuda12x` (GPU trading)

---

## License

MIT
