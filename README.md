# mtrader

Vectorized backtesting and forward/live-testing framework for intraday trading strategies.

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
clean_data()         — auto-detect types, normalize OHLCV, fill gaps, handle splits
    │
    ▼
add_indicators()     — 38+ indicators: SMA/EMA/WMA/SSMA, RSI, MACD, ATR, Stochastic,
    │                   Bollinger %B, CCI, Williams %R, ADX, MFI, PSAR, Supertrend,
    │                   Ichimoku, VWAP, Heikin Ashi, engulfing patterns, and more
    │
    ▼
precalculate_exit_time_amount_profit()
    │                 — exit signals: conditional, target, stoploss (absolute / % / column)
    │
    ▼
take_trade_on_condition*()
    │                 — capital simulation, Sharpe, drawdown
    │
    ▼
backtest_report()    — comprehensive stats: win rate, profit factor, Sharpe, Sortino, Calmar
equity_curve()       — per-bar equity + drawdown
html_backtest_report() — standalone HTML with SVG charts
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
result.metrics           # {'Sharpe Ratio': 1.23, ...}
result.report            # backtest_report dict
result.equity            # equity_curve DataFrame
result.trades            # trade_log DataFrame
result.to_html("report.html")
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
| `supertrend` | SuperTrend (returns direction 1/-1) |
| `ichimoku` | Ichimoku Cloud (tenkan/kijun/senkou A/senkou B/chikou) |
| `ha` | Heikin Ashi candles (open/high/low/close) |
| `vwap` / `ewap` / `iwap` | Volume / Equal / Incremental weighted avg price |

### Candlestick patterns
| Code | Description |
|------|-------------|
| `inside_bar` | 1 when bar inside previous bar's range |
| `engulfing` | 1 for bullish engulfing, -1 for bearish engulfing |

### Session-aware
| Code | Description |
|------|-------------|
| `or_high` / `or_low` | Opening range expanding high/low (first N min) |
| `prev_day` | Previous day high/low/close |
| `pivot` | Classic pivot point levels (P, S1, S2, R1, R2) |
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

### Normalizations
Prefix an indicator name for ratio-based forms: `SMN_`, `EMN_`, `WMN_`, `SSMN_`, `SVN_`, `EVN_`, `WVN_`, `SSVN_`, `Z_`, `BRN_`, `TMN_`.

Suffix controls denominator: `""`/`F` = signal itself, `P`/`0` = close, `B` = base signal, numeric code = that feature.

### Column naming
Indicators computed via `add_indicators()` produce columns named `can1_{indicator}_{period}`:
- `can1_sma1_p20` — SMA(20) on close
- `can1_rsi_p14` — RSI(14)
- `can1_macd` — MACD line (no period suffix, fixed 12/26/9)
- `can1_obv` — On-Balance Volume

---

## Conditions

Conditions use a declarative dict format:

```python
{
    "first_column_name": "can1_sma1_p20",
    "second_column_name": "close",
    "shift_down_first": 0,
    "shift_down_second": 0,
    "lower_range_of_difference": 0,      # first - second >= lower
    "upper_range_of_difference": np.inf, # first - second <= upper
    "perform_normalization_of_diff": False,
}
```

Helpers build these for you:

```python
from mtrader import condition, cross_above, cross_below

cond = condition("close", "can1_sma1_p20", lower=0)           # close > sma
crossover = cross_above("can1_macd", "can1_macdsignal")       # MACD crossover
crossunder = cross_below("can1_stochk_p14", "can1_stochd_p14") # Stochastic crossunder
```

**AND within a group, OR across groups:**

```python
[
    {"first": "ema9", "second": "ema21", ...},  # Group 1: both must be true
    {"first": "close", "second": "vwap", ...},
]
# OR
[
    [{"first": "rsi", "second": "30", ...}],     # Group 1 OR Group 2
    [{"first": "stochk", "second": "20", ...}],
]
```

---

## Exit strategy

```python
from mtrader import precalculate_exit_time_amount_profit

df = precalculate_exit_time_amount_profit(
    df, exit_conditions, buy_or_sell="buy",
    target_delta=200,                # absolute price target
    stoploss_delta=100,              # absolute stoploss
    target_delta_normalized=0.5,     # 0.5% target
    stoploss_delta_normalized=0.25,  # 0.25% stoploss
    target_delta_column="can1_atr_p14",      # per-bar column values
    stoploss_delta_column="can1_atr_p14_halfloss",
)
```

---

## Trade simulation

```python
trades, final_capital, metrics = mt.take_trade_on_condition_numpy(
    df, entry_conditions, leverage=1, initial_capital=100000)
# metrics: Sharpe Ratio, Volatility, Max Drawdown
```

CuPy-accelerated variants: `take_trade_on_condition`, `take_trade_on_condition2`, `take_trade_on_condition3`.

Grid search variants: `take_trade_on_condition_vectorized`, `take_trade_on_condition_vectorized2`.

---

## Performance reports

```python
from mtrader import backtest_report, equity_curve, html_backtest_report

report = backtest_report(df, initial_capital=1000)
# total_trades, win_rate_pct, profit_factor, avg_win/loss_pct,
# sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown_pct,
# max_consecutive_wins/losses, ...

eq = equity_curve(df)
# columns: datetime, equity, drawdown_pct, trade

html_backtest_report(result, output_path="report.html")
# standalone HTML with SVG charts, no external dependencies
```

---

## Exit optimization

```python
from mtrader import find_best_exit

best, results = find_best_exit(
    df, entry_conditions, buy_or_sell="buy",
    target_deltas=[50, 100, 150, 200],
    stoploss_deltas=[25, 50, 75, 100],
    metric="sharpe",      # or "final_capital", "max_drawdown"
    verbose=True,
)
print(best)  # {'target_delta': 150, 'stoploss_delta': 75}
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
)

result = strat.run(df)
strat.save("strat.json")
loaded = Strategy.load("strat.json")

live_engine = strat.to_live(df)  # convert to live engine with warmup
```

---

## Walk-forward optimization

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

---

## Position sizing & risk controls

```python
from mtrader import (
    fixed_quantity_size, fixed_capital_size,
    percent_equity_size, atr_risk_size,
    apply_risk_controls, india_intraday_cost_model,
)

qty = atr_risk_size(price, atr_values, equity=100000, risk_pct=0.01)
trades = apply_risk_controls(trades, max_trades_per_day=3, cooldown_bars=5)
```

---

## Multi-symbol portfolio

```python
from mtrader import run_portfolio

data = {"AAPL": df_aapl, "TSLA": df_tsla, "GOOG": df_goog}
result = run_portfolio(data, strategy, initial_capital=300000)
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

Incremental indicators (EMA, RSI, ATR, VWAP) update in O(1) per bar.

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

## Modules

| Module | Key exports |
|--------|-------------|
| `data_cleaner` | `clean_data`, `detect_data_types_with_formats`, `fill_missing_rows` |
| `indicators` | All standalone indicator functions (ema, rsi, macd, psar, ichimoku, ...) |
| `indicator_engine` | `add_indicators`, `add_indicators_on_group`, `FEATURE_CODE` |
| `exit_strategy` | `precalculate_exit_time_amount_profit` |
| `trading` | `take_trade_on_condition*`, `update_cond` |
| `optimize_exit` | `find_best_exit` |
| `backtest` | `run_backtest`, `BacktestResult`, `condition`, `cross_above/below`, `walk_forward_splits`, `parameter_grid`, `trade_log` |
| `advanced` | `Strategy`, `CostModel`, sizing functions, `run_portfolio`, `walk_forward_optimize`, `random_parameter_search`, `resample_ohlcv` |
| `live` | `LiveIndicatorEngine`, `LiveStrategyEngine`, `stream_live_signals`, `live_strategy_from_history` |
| `report` | `backtest_report`, `equity_curve`, `html_backtest_report` |
| `monotonic_stack` | `monotonic_stack_for_value1_gt/lt_value2` |
| `utils` | `timenum` |

---

## Testing

```bash
python -m pytest src/tests/ -v
```

46 tests cover: data cleaning, all indicators, exit precalculation, trade simulation (NumPy + CuPy), exit optimization, performance reports, and 20 strategy scenarios.

---

## Dependencies

- **Required:** `numpy`, `pandas`, `inspecty`
- **Optional:** `numba` (monotonic stack), `cupy` (GPU trading)

---

## License

MIT
