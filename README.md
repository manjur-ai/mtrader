# mtrader

**Vectorized backtesting & forward-testing framework for intraday strategies.**

Clean raw tick/minute data, compute 40+ technical indicators, define entry/exit conditions in a declarative dict format, and execute vectorized simulations with Sharpe, drawdown, and volatility metrics — all in NumPy/CuPy with optional Numba acceleration.

---

## Installation

```bash
pip install numpy pandas inspecty numba
# optional — GPU acceleration
pip install cupy
```

## Quick start

```python
import pandas as pd
from mtrader import clean_data, add_indicators
from mtrader import precalculate_exit_time_amount_profit
from mtrader import take_trade_on_condition_numpy

# 1. Load & clean OHLC data
df = pd.read_csv("data.csv")
df = clean_data(df, start_time="09:15", end_time="15:30",
                start_date="2024-01-01", end_date="2024-12-31")

# 2. Add indicators
df = add_indicators(df, add=["sma1", "ema1", "vwap", "close", "high", "low"],
                    rolling_minutes=[5, 15, 30])

# 3. Define exit strategy
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

df = precalculate_exit_time_amount_profit(
    df, conditions, buy_or_sell="buy",
    target_delta=200, stoploss_delta=100,
)

# 4. Run backtest
trades, final_capital, metrics = take_trade_on_condition_numpy(
    df, conditions, leverage=1, initial_capital=100000
)
print(metrics)
```

---

## Pipeline

```
raw CSV/DataFrame
    │
    ▼
clean_data()          — auto-detect column types, normalize OHLC,
    │                    fill gaps, handle stock splits
    ▼
add_indicators()      — SMA, EMA, WMA, SSMA + volatilities + VWAP,
    │                    distance-to-MA, z-score, normalizations
    ▼
precalculate_exit_time_amount_profit()
    │                    — vectorized exit signal engine with
    │                      target & stoploss (absolute or %)
    ▼
take_trade_on_condition*()
                         — capital simulation, Sharpe, drawdown
```

---

## Modules

| Module | Exports | Description |
|--------|---------|-------------|
| `data_cleaner` | `clean_data`, `detect_data_types_with_formats`, `fill_missing_rows` | Column-type detection, OHLC normalization, gap filling, split adjustment |
| `indicators` | `ema`, `evol`, `wma`, `wvol`, `ssma`, `ssvol` | Individual MA & volatility functions on raw numpy arrays |
| `indicator_engine` | `add_indicators`, `add_indicators_on_group` | Multi-period indicator computation with automatic dependency resolution |
| `monotonic_stack` | `monotonic_stack_for_value1_gt_value2`, `monotonic_stack_for_value1_lessthan_value2` | Numba-accelerated next-index lookup for exit timing |
| `exit_strategy` | `precalculate_exit_time_amount_profit` | Vectorized exit signal generation with target, stoploss, and condition-based exits |
| `trading` | `take_trade_on_condition`, `take_trade_on_condition_numpy`, `take_trade_on_condition2`, `take_trade_on_condition3`, `take_trade_on_condition_vectorized`, `take_trade_on_condition_vectorized2`, `update_cond` | Trade execution engines — NumPy, CuPy, multi-range grid search, GPU-optimized variants |
| `optimize_exit` | `find_best_exit` | Grid search over exit params (target, stoploss) to maximize Sharpe / capital |
| `utils` | `printo`, `timenum` | Time-string conversion |

---

## Indicators

### Moving averages
| Code | Name |
|------|------|
| `sma` | Simple Moving Average |
| `ema` | Exponential Moving Average |
| `wma` | Weighted Moving Average |
| `ssma` | Smoothed Simple Moving Average |

### Volatility
| Code | Name |
|------|------|
| `svol` | Simple volatility (std) |
| `evol` | Exponentially-weighted volatility |
| `wvol` | Weighted volatility |
| `ssvol` | Smoothed volatility |

### Price / volume
- `vwap`, `ewap`, `iwap` — volume, equal-weighted, and incremental-weighted average price
- `max`, `min` — rolling high/low
- `max_inday`, `min_inday` — intraday rolling high/low

### Feature codes
Base signals are referenced by numeric codes:

| Code | Signal |
|------|--------|
| `0`, `1` | `close` |
| `2` | `av2` (H+L)/2 |
| `3` | `av3` (H+L+C)/3 |
| `4` | `av4` (O+H+L+C)/4 |
| `5` | `open` |
| `6` | `high` |
| `7` | `low` |
| `8..34` | `dif`, `ret`, `lret` for 1/3/5/10/15/20/30/60 bars |

**Column naming:** `can1_{indicator}{code}_p{period}`, e.g. `can1_sma1_p5`.

### Popular technical indicators

Directly computable via `add_indicators()`:

| Indicator | Code | Formula |
|-----------|------|---------|
| RSI | `rsi` | 100 - 100/(1 + avg_gain/avg_loss) |
| ATR | `atr` | SMA of True Range (max(H-L, H-prevC, prevC-L)) |
| Stochastic %K | `stochk` | (close - min_L_N) / (max_H_N - min_L_N) × 100 |
| Stochastic %D | `stochd` | 3-period SMA of %K |
| Bollinger %B | `bbp` | (close - lower) / (upper - lower), kde"=2 |
| OBV | `obv` | Cumulative signed volume |

**Example:**
```python
df = add_indicators(df, add=["rsi", "atr", "stochk", "stochd", "bbp", "obv"],
                    rolling_minutes=[14])
# Columns created: can1_rsi_p14, can1_atr_p14, can1_stochk_p14, ...
```

These functions are also available directly:
```python
from mtrader import rsi, atr, stoch_k, stoch_d, bollinger_b, obv

rsi_values = rsi(close_array, period=14)
atr_values = atr(high, low, close, period=14)
k_values = stoch_k(high, low, close, period=14)
d_values = stoch_d(k_values, period=3)
bbp_values = bollinger_b(close, period=20, k=2.0)
obv_values = obv(close, volume)
```

### Compose your own

Many popular indicators can be composed from existing primitives without new code:

| Desired | How |
|---------|-----|
| **MACD line** | `ema12 - ema26` (use `add=["ema1"]` with periods 12 & 26, then diff) |
| **MACD histogram** | MACD line minus its EMA(9) |
| **Bollinger Upper/Lower** | `sma1 + k*svol1` / `sma1 - k*svol1` |
| **Keltner Channels** | `ema1 + k*atr` / `ema1 - k*atr` |
| **CCI** | (close - sma1) / (0.015 × svol1) |

### Normalizations
Prefix an indicator name for ratio-based forms:

| Prefix | Normalization |
|--------|--------------|
| `SMN` | Divide by SMA |
| `EMN` | Divide by EMA |
| `WMN` | Divide by WMA |
| `SSMN` | Divide by SSMA |
| `SVN` | Divide by volatility (std) |
| `EVN` | Divide by EW volatility |
| `WVN` | Divide by weighted volatility |
| `SSVN` | Divide by smoothed volatility |
| `Z` | Z-score |
| `BRN`, `TMN` | Base-ratio / time normalization |

Suffix controls the denominator source: `""`/`F` uses the signal itself; `P`/`0` uses close price; `B` uses the base signal; a numeric code uses that feature.

---

## Condition format

Entry and exit conditions use a declarative dict structure:

```python
[
    [  # OR group — any matching group triggers
        {  # AND within a group — all must match
            "first_column_name": "can1_sma1_p5",
            "second_column_name": "close",
            "shift_down_first": 0,
            "shift_down_second": 0,
            "lower_range_of_difference": -np.inf,
            "upper_range_of_difference": -50,
            "perform_normalization_of_diff": False,
        }
    ]
]
```

The helper `update_cond()` recursively rewrites fields across nested condition lists.

---

## Exit strategy

`precalculate_exit_time_amount_profit()` supports:

- **Condition-based exits** — exit when column difference falls in a range
- **Target (take-profit)** — absolute delta or normalized (% of price)
- **Stoploss** — absolute or normalized; configurable slippage and candle-close wait
- **Combined** — earliest of condition, target, or stoploss wins
- **Side** — `buy` or `sell` mode

---

## Trade simulation

| Function | Backend | Use case |
|----------|---------|---------|
| `take_trade_on_condition_numpy` | NumPy | Standard backtesting |
| `take_trade_on_condition` | NumPy/CuPy auto | Same, with GPU fallback |
| `take_trade_on_condition2` | CuPy views | Memory-efficient GPU |
| `take_trade_on_condition3` | CuPy | Simplified GPU variant |
| `take_trade_on_condition_vectorized` | CuPy 2D | Grid search over range combinations |
| `take_trade_on_condition_vectorized2` | CuPy 2D | Optimized grid search |

All return `(filtered_trades_df, final_capital, metrics_dict)` where metrics include:

- `Volatility`
- `Sharpe Ratio`
- `Max Drawdown`

---

## Exit optimization

Given a fixed entry condition, `find_best_exit()` grid-searches over exit parameters to maximize Sharpe, final capital, or any other metric.

```python
from mtrader import find_best_exit

best_params, results_df = find_best_exit(
    df,
    entry_conditions=entry_conditions,
    buy_or_sell="buy",
    target_deltas=[50, 100, 150, 200, 300],
    stoploss_deltas=[25, 50, 75, 100],
    metric="sharpe",           # or "final_capital", "max_drawdown"
    risk_free_rate=0.05,
)

print(best_params)
# {'target_delta': 150, 'stoploss_delta': 75, ...}

# Full grid in results_df:
#   target_delta | stoploss_delta | trades | final_capital | sharpe | volatility | max_drawdown
```

Works with absolute deltas, normalized (% of price) deltas, and custom exit conditions. Set `verbose=True` to see each combination's results as they run.

---

## Data cleaning

`clean_data()` handles the messy real-world pipeline:

- Auto-detects datetime, numeric, and string columns
- Identifies date/time columns and combines them
- Maps columns to `open`, `high`, `low`, `close`, `volume`
- Rounds timestamps to the minute and deduplicates
- Forward-fills zero-price candles
- **Split detection** — automatically detects and adjusts for stock splits and reverse mergers (≥30% gap, validates integer ratio 2:1 to 5:1)
- Filters by date and time range
- Drops days with insufficient records
- Optionally fills intraday gaps

---

## Testing

```bash
python -m pytest mtrader/tests/test_backtest.py -v
```

15 end-to-end tests cover: data cleaning, indicator computation, exit precalculation, trade simulation (NumPy + CuPy pipelines), empty-trade edge cases, monotonic stack correctness, and utility functions.

---

## Dependencies

- **Required:** `numpy`, `pandas`, `inspecty`
- **Optional:** `numba` (monotonic stack acceleration), `cupy` (GPU trading)

---

## License

MIT
