from __future__ import annotations
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

import numpy as np
import pandas as pd


class RollingWindow:
    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        self.period = int(period)
        self.values: deque = deque()
        self.total: float = 0.0

    def update(self, value: float) -> float:
        """Push a new value into the rolling window and return the current mean."""
        value = float(value)
        self.values.append(value)
        self.total += value
        if len(self.values) > self.period:
            self.total -= self.values.popleft()
        return self.total / len(self.values)

    def strict_mean(self) -> float:
        """Return the mean only if the window is full (len >= period), otherwise NaN."""
        if len(self.values) < self.period:
            return np.nan
        return self.total / self.period


class LiveEMA:
    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        self.period = int(period)
        self.value: float | None = None
        self.count: int = 0

    def update(self, value: float) -> float:
        """Push a new value and return the live EMA."""
        value = float(value)
        if self.value is None:
            self.value = value
        elif self.count < self.period:
            self.value = (self.value * self.count + value * 2.0) / (self.count + 2.0)
        else:
            self.value = (self.value * (self.period - 1.0) + value * 2.0) / (self.period + 1.0)
        self.count += 1
        return self.value


class LiveRSI:
    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        self.period = int(period)
        self.prev_close: float | None = None
        self.avg_gain: float | None = None
        self.avg_loss: float | None = None
        self.gains: deque = deque()
        self.losses: deque = deque()

    def update(self, close: float) -> float:
        """Push a new close price and return the live RSI value."""
        close = float(close)
        if self.prev_close is None:
            self.prev_close = close
            return np.nan

        diff = close - self.prev_close
        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)
        self.prev_close = close

        if self.avg_gain is None:
            self.gains.append(gain)
            self.losses.append(loss)
            if len(self.gains) < self.period:
                return np.nan
            self.avg_gain = sum(self.gains) / self.period
            self.avg_loss = sum(self.losses) / self.period
        else:
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period

        if self.avg_loss == 0.0:
            return 100.0 if self.avg_gain > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + self.avg_gain / self.avg_loss)


class LiveATR:
    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        self.period = int(period)
        self.prev_close: float | None = None
        self.window = RollingWindow(period)

    def update(self, high: float, low: float, close: float) -> float:
        """Push a new bar (high, low, close) and return the live ATR value."""
        high = float(high)
        low = float(low)
        close = float(close)
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        self.prev_close = close
        self.window.update(tr)
        return self.window.strict_mean()


class LiveVWAP:
    def __init__(self) -> None:
        self.session: Any = None
        self.price_volume: float = 0.0
        self.volume: float = 0.0

    def update(self, timestamp: Any, high: float, low: float, close: float, volume: float) -> float:
        """Push a new bar and return the live VWAP, resetting at each session change."""
        session = pd.Timestamp(timestamp).date()
        if self.session != session:
            self.session = session
            self.price_volume = 0.0
            self.volume = 0.0
        typical = (float(high) + float(low) + float(close)) / 3.0
        volume = float(volume)
        self.price_volume += typical * volume
        self.volume += volume
        if self.volume == 0:
            return np.nan
        return self.price_volume / self.volume


@dataclass
class LiveIndicatorEngine:
    indicators: list
    periods: list
    buy_conditions: list | None = None
    sell_conditions: list | None = None
    history_size: int = 512
    rows: deque = field(init=False)
    sma_state: dict = field(init=False, default_factory=dict)
    ema_state: dict = field(init=False, default_factory=dict)
    rsi_state: dict = field(init=False, default_factory=dict)
    atr_state: dict = field(init=False, default_factory=dict)
    vwap_state: LiveVWAP = field(init=False, default_factory=LiveVWAP)
    fast_indicators: set = field(init=False, default_factory=set)
    fallback_indicators: list = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Initialize per-period state objects (SMA, EMA, RSI, ATR, VWAP) and split indicators into fast/fallback groups."""
        self.indicators = list(self.indicators or [])
        self.periods = [int(p) for p in (self.periods or [])]
        self.rows = deque(maxlen=self.history_size)
        self.fast_indicators, self.fallback_indicators = _split_live_indicators(self.indicators)
        for period in self.periods:
            self.sma_state[period] = RollingWindow(period)
            self.ema_state[period] = LiveEMA(period)
            self.rsi_state[period] = LiveRSI(period)
            self.atr_state[period] = LiveATR(period)

    @classmethod
    def from_history(cls, df: pd.DataFrame, indicators: list, periods: list, buy_conditions: list | None = None, sell_conditions: list | None = None, history_size: int = 512, warmup_batch: bool = True) -> LiveIndicatorEngine:
        """Create a LiveIndicatorEngine from historical data, optionally batch-warming up indicators."""
        engine = cls(
            indicators=indicators,
            periods=periods,
            buy_conditions=buy_conditions,
            sell_conditions=sell_conditions,
            history_size=history_size,
        )
        if warmup_batch:
            engine.warmup(df)
        else:
            for _, row in df.iterrows():
                engine.update(row, evaluate_signals=False)
        return engine

    def warmup(self, df: pd.DataFrame | None) -> LiveIndicatorEngine:
        """Batch-warm the engine on historical data, pre-computing fallback indicators and seeding the deque. Returns self."""
        from mtrader.indicator_engine import add_indicators

        if df is None or len(df) == 0:
            return self

        raw = df.copy()
        add = _batch_indicators_for_live(self.indicators)
        calc = add_indicators(raw.copy(), add=add, rolling_minutes=self.periods) if add else raw

        old_fallback = self.fallback_indicators
        self.fallback_indicators = []
        self.rows.clear()
        for _, row in raw.iterrows():
            self.update(row, evaluate_signals=False)
        self.fallback_indicators = old_fallback

        warmed_rows = calc.tail(self.history_size).to_dict("records")
        self.rows.clear()
        for row in warmed_rows:
            self.rows.append(row)
        return self

    def update(self, bar: Any, evaluate_signals: bool = True) -> dict[str, Any]:
        """Process a new bar: update fast indicators, run fallback calc if needed, evaluate buy/sell signals. Returns the enriched bar dict."""
        row = _coerce_bar(bar)
        out = dict(row)
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        for period in self.periods:
            if "sma" in self.fast_indicators:
                value = self.sma_state[period].update(close)
                out[f"can1_sma1_p{period}"] = value
                out[f"live_sma_p{period}"] = value
            if "ema" in self.fast_indicators:
                value = self.ema_state[period].update(close)
                out[f"can1_ema1_p{period}"] = value
                out[f"live_ema_p{period}"] = value
            if "rsi" in self.fast_indicators:
                value = self.rsi_state[period].update(close)
                out[f"can1_rsi_p{period}"] = value
                out[f"live_rsi_p{period}"] = value
            if "atr" in self.fast_indicators:
                value = self.atr_state[period].update(high, low, close)
                out[f"can1_atr_p{period}"] = value
                out[f"live_atr_p{period}"] = value

        if "vwap" in self.fast_indicators:
            if "volume" not in row:
                raise ValueError("volume is required for live vwap")
            value = self.vwap_state.update(row["datetime"], high, low, close, row["volume"])
            out["can1_vwap"] = value
            out["live_vwap"] = value

        if "zero" in self.indicators or _conditions_need_zero(self.buy_conditions) or _conditions_need_zero(self.sell_conditions):
            out["zero"] = 0.0

        self.rows.append(out)
        if self.fallback_indicators:
            self._update_fallback_indicators()
            out = self.rows[-1]
        if evaluate_signals:
            out["buy_signal"] = self.evaluate(self.buy_conditions)
            out["sell_signal"] = self.evaluate(self.sell_conditions)
        return out

    def evaluate(self, conditions: list[list[dict[str, Any]]] | None) -> bool:
        """Evaluate a set of OR/AND conditions against the current deque state. Returns True if any group is fully satisfied."""
        if not conditions:
            return False
        for group in conditions:
            if not group:
                continue
            if all(self._condition_met(cond) for cond in group):
                return True
        return False

    def _condition_met(self, cond: dict[str, Any]) -> bool:
        first = self._value(cond["first_column_name"], cond.get("shift_down_first", 0))
        second = self._value(cond["second_column_name"], cond.get("shift_down_second", 0))
        if pd.isna(first) or pd.isna(second):
            return False
        difference = first - second
        if cond.get("perform_normalization_of_diff", False):
            close = self._value("close", 0)
            if close == 0 or pd.isna(close):
                return False
            difference *= 10000.0 / close
        return cond["lower_range_of_difference"] <= difference <= cond["upper_range_of_difference"]

    def _value(self, name: str, shift: int) -> float:
        shift = int(shift)
        if shift < 0:
            raise ValueError("live conditions cannot use negative shifts")
        if shift >= len(self.rows):
            return np.nan
        row = self.rows[-1 - shift]
        return row.get(name, np.nan)

    def latest(self) -> dict[str, Any] | None:
        """Return the most recently processed bar (enriched with indicators and signals)."""
        return self.rows[-1] if self.rows else None

    def to_frame(self) -> pd.DataFrame:
        """Convert the internal deque history to a DataFrame."""
        return pd.DataFrame(list(self.rows))

    def stream(self, candle_feed: Any, on_signal: Callable[[dict[str, Any]], bool | None] | None = None, stop_on_callback_false: bool = False) -> Generator[dict[str, Any], None, None]:
        """Stream candles through the engine, yielding enriched bars. Optionally call on_signal on each result."""
        return stream_live_signals(self, candle_feed, on_signal=on_signal, stop_on_callback_false=stop_on_callback_false)

    def _update_fallback_indicators(self) -> None:
        from mtrader.indicator_engine import add_indicators

        data = pd.DataFrame(list(self.rows))
        drop_cols = [c for c in data.columns if c.startswith("live_") or c in {"buy_signal", "sell_signal", "entry_signal", "exit_signal", "action"}]
        if drop_cols:
            data = data.drop(columns=drop_cols)
        calc = add_indicators(data, add=self.fallback_indicators, rolling_minutes=self.periods)
        latest = calc.iloc[-1].to_dict()
        self.rows[-1].update({k: v for k, v in latest.items() if k not in {"datetime", "open", "high", "low", "close", "volume"}})


class LiveStrategyEngine:
    def __init__(self, engine: LiveIndicatorEngine, side: str = "buy") -> None:
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        self.engine = engine
        self.side = side

    def update(self, bar: Any) -> dict[str, Any]:
        """Process a new bar and map buy/sell signals to action (BUY/SELL/EXIT_BUY/EXIT_SELL/HOLD) based on strategy side."""
        out = self.engine.update(bar)
        if self.side == "buy":
            out["entry_signal"] = bool(out.get("buy_signal", False))
            out["exit_signal"] = bool(out.get("sell_signal", False))
            out["action"] = "BUY" if out["entry_signal"] else ("EXIT_BUY" if out["exit_signal"] else "HOLD")
        else:
            out["entry_signal"] = bool(out.get("sell_signal", False))
            out["exit_signal"] = bool(out.get("buy_signal", False))
            out["action"] = "SELL" if out["entry_signal"] else ("EXIT_SELL" if out["exit_signal"] else "HOLD")
        return out

    def latest(self) -> dict[str, Any] | None:
        """Return the latest processed bar from the underlying engine."""
        return self.engine.latest()

    def to_frame(self) -> pd.DataFrame:
        """Convert engine history to a DataFrame."""
        return self.engine.to_frame()

    def stream(self, candle_feed: Any, on_signal: Callable[[dict[str, Any]], bool | None] | None = None, stop_on_callback_false: bool = False) -> Generator[dict[str, Any], None, None]:
        """Stream candles, mapping to actions. See stream_live_signals."""
        return stream_live_signals(self, candle_feed, on_signal=on_signal, stop_on_callback_false=stop_on_callback_false)


def live_indicators_from_backtest(indicators: list[str] | None) -> list[str]:
    """Map backtest indicator names to live-compatible indicator names (e.g. ema1 -> ema)."""
    live = set()
    for item in indicators or []:
        if item == "zero":
            live.add("zero")
        elif item == "vwap":
            live.add("vwap")
        elif item == "rsi":
            live.add("rsi")
        elif item == "atr":
            live.add("atr")
        elif item.startswith("sma"):
            live.add("sma")
        elif item.startswith("ema"):
            live.add("ema")
        else:
            live.add(item)
    return sorted(live)


def convert_conditions_to_live(conditions: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """Deep-copy backtest conditions so they can be used in live evaluation without mutating originals."""
    return deepcopy(conditions)


def live_column_name(column: str) -> str:
    """Identity function for column name mapping (placeholder for future live column renaming)."""
    return column


def live_strategy_from_history(
    df: pd.DataFrame,
    indicators: list[str],
    periods: list[int],
    entry_conditions: list[list[dict[str, Any]]],
    exit_conditions: list[list[dict[str, Any]]] | None = None,
    side: str = "buy",
    history_size: int = 512,
    warmup_batch: bool = True,
) -> LiveStrategyEngine:
    """Build a fully-configured LiveStrategyEngine from historical data, mapping backtest indicators/conditions to live equivalents."""
    live_indicators = live_indicators_from_backtest(indicators)
    entry_live = convert_conditions_to_live(entry_conditions)
    exit_live = convert_conditions_to_live(exit_conditions or [])
    if side == "buy":
        buy_conditions = entry_live
        sell_conditions = exit_live
    else:
        buy_conditions = exit_live
        sell_conditions = entry_live
    engine = LiveIndicatorEngine.from_history(
        df,
        indicators=live_indicators,
        periods=periods,
        buy_conditions=buy_conditions,
        sell_conditions=sell_conditions,
        history_size=history_size,
        warmup_batch=warmup_batch,
    )
    return LiveStrategyEngine(engine, side=side)


def live_signal_from_history(
    df: pd.DataFrame,
    indicators: list[str],
    periods: list[int],
    new_bar: Any,
    buy_conditions: list[list[dict[str, Any]]] | None = None,
    sell_conditions: list[list[dict[str, Any]]] | None = None,
    history_size: int = 512,
    warmup_batch: bool = True,
) -> dict[str, Any]:
    """Warm up a LiveIndicatorEngine on historical data, then evaluate a single new bar for signals."""
    engine = LiveIndicatorEngine.from_history(
        df,
        indicators=indicators,
        periods=periods,
        buy_conditions=buy_conditions,
        sell_conditions=sell_conditions,
        history_size=history_size,
        warmup_batch=warmup_batch,
    )
    return engine.update(new_bar)


def stream_live_signals(live_engine: LiveIndicatorEngine | LiveStrategyEngine, candle_feed: Any, on_signal: Callable[[dict[str, Any]], bool | None] | None = None, stop_on_callback_false: bool = False) -> Generator[dict[str, Any], None, None]:
    """Generator that feeds candles from an iterable through a live engine, yielding enriched bars. Optionally invokes on_signal callback."""
    for candle in candle_feed:
        signal = live_engine.update(candle)
        if on_signal is not None:
            result = on_signal(signal)
            if stop_on_callback_false and result is False:
                break
        yield signal


def _coerce_bar(bar: Any) -> dict[str, Any]:
    if isinstance(bar, pd.Series):
        row = bar.to_dict()
    elif isinstance(bar, dict):
        row = dict(bar)
    else:
        raise TypeError("bar must be a dict or pandas Series")
    required = {"datetime", "open", "high", "low", "close"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"bar is missing required fields: {missing}")
    row["datetime"] = pd.Timestamp(row["datetime"])
    return row


def _conditions_need_zero(conditions: list[list[dict[str, Any]]] | None) -> bool:
    if not conditions:
        return False
    for group in conditions:
        for cond in group:
            if cond.get("first_column_name") == "zero" or cond.get("second_column_name") == "zero":
                return True
    return False


def _split_live_indicators(indicators: list[str] | None) -> tuple[set[str], list[str]]:
    fast = set()
    fallback = []
    for item in indicators or []:
        if item == "zero":
            fast.add("zero")
        elif item == "vwap":
            fast.add("vwap")
        elif item == "rsi":
            fast.add("rsi")
        elif item == "atr":
            fast.add("atr")
        elif item == "sma" or item == "sma1" or item == "sma_close":
            fast.add("sma")
        elif item == "ema" or item == "ema1" or item == "ema_close":
            fast.add("ema")
        else:
            fallback.append(item)
    return fast, fallback


def _batch_indicators_for_live(indicators: list[str] | None) -> list[str]:
    add = []
    for item in indicators or []:
        if item == "sma":
            add.append("sma1")
        elif item == "ema":
            add.append("ema1")
        else:
            add.append(item)
    return sorted(set(add))
