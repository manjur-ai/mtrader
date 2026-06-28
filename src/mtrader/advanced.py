from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from mtrader.backtest import condition


@dataclass
class CostModel:
    brokerage_per_order: float = 0.0
    brokerage_pct: float = 0.0
    slippage_pct: float = 0.0
    spread_pct: float = 0.0
    exchange_fee_pct: float = 0.0
    tax_pct: float = 0.0
    min_cost: float = 0.0

    def cost_rate(self) -> float:
        """Sum of all percentage-based cost components (brokerage, slippage, spread, exchange fee, tax)."""
        return self.brokerage_pct + self.slippage_pct + self.spread_pct + self.exchange_fee_pct + self.tax_pct

    def estimate(self, turnover: float) -> float:
        """Estimate total trading cost for a given turnover: variable costs + brokerage_per_order, floored at min_cost."""
        variable = turnover * self.cost_rate()
        return np.maximum(variable + self.brokerage_per_order, self.min_cost)


def india_intraday_cost_model() -> CostModel:
    """Return a CostModel with typical Indian intraday trading costs (brokerage, STT, exchange fees, etc.)."""
    return CostModel(
        brokerage_per_order=20.0,
        brokerage_pct=0.0003,
        slippage_pct=0.0002,
        spread_pct=0.0001,
        exchange_fee_pct=0.0000345,
        tax_pct=0.00025,
    )


def crypto_cost_model(fee_pct: float = 0.001, slippage_pct: float = 0.0005, spread_pct: float = 0.0002) -> CostModel:
    """Return a CostModel with typical crypto exchange fee structure (maker/taker fee, slippage, spread)."""


def fixed_quantity_size(entry_price: np.ndarray, quantity: float) -> np.ndarray:
    """Position sizing: return a constant quantity regardless of entry price."""
    entry_price = np.asarray(entry_price, dtype=np.float64)
    return np.full(entry_price.shape, float(quantity), dtype=np.float64)


def fixed_capital_size(entry_price: np.ndarray, capital_per_trade: float) -> np.ndarray:
    """Position sizing: quantity = capital_per_trade / entry_price (fixed capital per trade)."""
    entry_price = np.asarray(entry_price, dtype=np.float64)
    return np.divide(capital_per_trade, entry_price, out=np.zeros_like(entry_price), where=entry_price != 0)


def percent_equity_size(entry_price: np.ndarray, equity: np.ndarray, pct: float = 1.0) -> np.ndarray:
    """Position sizing: quantity = equity * pct / entry_price (fraction of current equity)."""
    entry_price = np.asarray(entry_price, dtype=np.float64)
    equity = np.asarray(equity, dtype=np.float64)
    return np.divide(equity * pct, entry_price, out=np.zeros_like(entry_price), where=entry_price != 0)


def atr_risk_size(entry_price: np.ndarray, atr_values: np.ndarray, equity: np.ndarray, risk_pct: float = 0.01, atr_multiple: float = 1.0) -> np.ndarray:
    """Position sizing: quantity = equity * risk_pct / (atr * atr_multiple). Risk-based sizing using ATR."""
    entry_price = np.asarray(entry_price, dtype=np.float64)
    atr_values = np.asarray(atr_values, dtype=np.float64)
    equity = np.asarray(equity, dtype=np.float64)
    risk_amount = equity * risk_pct
    risk_per_unit = atr_values * atr_multiple
    return np.divide(risk_amount, risk_per_unit, out=np.zeros_like(entry_price), where=risk_per_unit > 0)


def add_trailing_stop_column(df: pd.DataFrame, trail_pct: float = 0.5, lookback: int | None = None,
                             high_col: str = "high", low_col: str = "low") -> pd.DataFrame:
    """Add a trailing stop price column (`trailing_stop_price`) for trailing stop exits.

    For each bar j, the trailing stop price is:
        max(high[max(0, j-lookback):j+1]) * (1 - trail_pct/100)

    If lookback is None, uses expanding maximum from start of data.
    For use with stoploss_delta_column in run_backtest:
        df['trailing_sl_delta'] = df['close'] - df['trailing_stop_price']
        run_backtest(..., stoploss_delta_column='trailing_sl_delta')
    """
    if lookback is not None:
        trail_high = df[high_col].rolling(window=lookback, min_periods=1).max()
    else:
        trail_high = df[high_col].expanding(min_periods=1).max()
    df["trailing_stop_price"] = trail_high * (1.0 - trail_pct / 100.0)
    return df


def add_time_filter_column(df: pd.DataFrame, start_time: str | None = None, end_time: str | None = None,
                           time_col: str = "datetime") -> pd.DataFrame:
    """Add a `time_filter` boolean column for restricting trades to specific trading hours.

    True for bars within [start_time, end_time). Pass to entry/exit conditions:
        entry_conditions = [
            [condition("close", "can1_sma1_p20", lower=0),
             condition("time_filter", lower=1)],  # only trade during filtered hours
        ]
    """
    from mtrader.utils import timenum
    if start_time is None:
        start_num = -np.inf
    else:
        start_num = timenum(start_time)
    if end_time is None:
        end_num = np.inf
    else:
        end_num = timenum(end_time)
    df["time_filter"] = df[time_col].apply(
        lambda x: start_num <= (x.hour * 60 + x.minute) < end_num
    ).astype(float)
    return df


def add_regime_filter_column(df: pd.DataFrame, adx_period: int = 14, adx_threshold: float = 25.0,
                             adx_col: str = "can1_adx_p14") -> pd.DataFrame:
    """Add a `regime_filter` boolean column: True when ADX >= threshold (trending market).

    For ranging markets, invert with condition("regime_filter", upper=1) (i.e., regime_filter == 0).
    Requires 'adx' indicator to be pre-computed via add_indicators.
    """
    if adx_col not in df.columns:
        from mtrader.indicator_engine import add_indicators
        df = add_indicators(df, add=["adx"], rolling_minutes=[adx_period])
    df["regime_filter"] = (df[adx_col] >= adx_threshold).astype(float)
    return df


def apply_risk_controls(
    trades: pd.DataFrame,
    max_trades_per_day: int | None = None,
    cooldown_bars: int = 0,
    max_daily_loss_pct: float | None = None,
    datetime_col: str = "entry_time",
    return_col: str = "capital_return_pct",
) -> pd.DataFrame:
    """Apply risk control filters to a trade DataFrame: max trades/day, cooldown between trades, max daily loss. Returns a DataFrame with an `allowed` column."""
    if trades.empty:
        out = trades.copy()
        out["allowed"] = pd.Series(dtype=bool)
        return out

    out = trades.copy()
    out["allowed"] = True

    if datetime_col in out.columns:
        dates = pd.to_datetime(out[datetime_col]).dt.date
        if max_trades_per_day is not None:
            out["allowed"] &= out.groupby(dates).cumcount() < max_trades_per_day
        if max_daily_loss_pct is not None and return_col in out.columns:
            daily_return = out[return_col].fillna(0).groupby(dates).cumsum()
            out["allowed"] &= daily_return >= -abs(max_daily_loss_pct)

    if cooldown_bars and "entry_index" in out.columns:
        allowed = out["allowed"].to_numpy(copy=True)
        last_allowed_index = None
        for i, idx in enumerate(out["entry_index"].to_numpy()):
            if not allowed[i]:
                continue
            if last_allowed_index is not None and idx - last_allowed_index <= cooldown_bars:
                allowed[i] = False
            else:
                last_allowed_index = idx
        out["allowed"] = allowed

    return out


def resample_ohlcv(df: pd.DataFrame, rule: str, label: str = "right", closed: str = "right") -> pd.DataFrame:
    """Resample OHLCV data to a higher timeframe (e.g. '5min', '1h'). Open=first, high=max, low=min, close=last, volume=sum."""
    required = {"datetime", "open", "high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    data = df.set_index("datetime").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in data.columns:
        agg["volume"] = "sum"
    rule_clean = rule.replace("T", "min").replace("t", "min")
    out = data.resample(rule_clean, label=label, closed=closed).agg(agg).dropna(subset=["open", "high", "low", "close"])
    return out.reset_index()


def add_higher_timeframe_indicators(df: pd.DataFrame, rule: str, add: list[str], rolling_minutes: list[int] | None = None, prefix: str | None = None) -> pd.DataFrame:
    """Compute indicators on a resampled higher timeframe and merge them back to the original DataFrame via merge_asof."""
    from mtrader.indicator_engine import add_indicators

    prefix = prefix or _timeframe_can_prefix(rule)
    higher = resample_ohlcv(df, rule)
    higher = add_indicators(higher, add=add, rolling_minutes=rolling_minutes or [])
    feature_cols = [c for c in higher.columns if c.startswith("can1_")]
    renamed = higher[["datetime"] + feature_cols].rename(columns={c: c.replace("can1_", f"{prefix}_", 1) for c in feature_cols})
    merged = pd.merge_asof(
        df.sort_values("datetime"),
        renamed.sort_values("datetime"),
        on="datetime",
        direction="backward",
    )
    return merged


def _timeframe_can_prefix(rule: str) -> str:
    rule_clean = rule.replace("T", "min").replace("t", "min")
    offset = pd.tseries.frequencies.to_offset(rule_clean)
    nanos = pd.Timedelta(offset).value
    minutes = nanos // pd.Timedelta(minutes=1).value
    if minutes <= 0 or nanos % pd.Timedelta(minutes=1).value != 0:
        raise ValueError(f"rule must resolve to whole minutes: {rule!r}")
    return f"can{minutes}"


@dataclass
class Strategy:
    name: str
    indicators: list = field(default_factory=list)
    rolling_minutes: list = field(default_factory=list)
    entry_conditions: list = field(default_factory=list)
    exit_conditions: list | None = None
    side: str = "buy"
    target_delta: float | None = None
    stoploss_delta: float | None = None
    target_delta_normalized: float | None = None
    stoploss_delta_normalized: float | None = None
    leverage: float = 1.0
    initial_capital: float = 1000.0
    risk_free_rate: float = 0.0
    trading_cost_factor: float = 0.0002
    capital_per_trade_pct: float = 1.0
    sizing_fn: Callable | None = None
    min_hold_bars: int = 0
    max_hold_bars: int | None = None
    max_trades_per_day: int | None = None
    cooldown_bars: int = 0
    max_daily_loss_pct: float | None = None

    def run(self, df: pd.DataFrame, **overrides: Any) -> Any:
        """Run a backtest using this strategy's parameters, optionally overriding any field."""
        from mtrader.backtest import run_backtest

        params = self.__dict__.copy()
        params.update(overrides)
        return run_backtest(
            df,
            entry_conditions=params["entry_conditions"],
            exit_conditions=params["exit_conditions"],
            indicators=params["indicators"],
            rolling_minutes=params["rolling_minutes"],
            buy_or_sell=params["side"],
            target_delta=params["target_delta"],
            stoploss_delta=params["stoploss_delta"],
            target_delta_normalized=params["target_delta_normalized"],
            stoploss_delta_normalized=params["stoploss_delta_normalized"],
            leverage=params["leverage"],
            initial_capital=params["initial_capital"],
            risk_free_rate=params["risk_free_rate"],
            trading_cost_factor=params["trading_cost_factor"],
            capital_per_trade_pct=params["capital_per_trade_pct"],
            sizing_fn=params.get("sizing_fn"),
            min_hold_bars=params.get("min_hold_bars", 0),
            max_hold_bars=params.get("max_hold_bars"),
            max_trades_per_day=params["max_trades_per_day"],
            cooldown_bars=params["cooldown_bars"],
            max_daily_loss_pct=params["max_daily_loss_pct"],
        )

    def to_live(self, history_df: pd.DataFrame, **overrides: Any) -> Any:
        """Convert this strategy to a live trading engine (LiveStrategyEngine)."""
        from mtrader.live import live_strategy_from_history

        params = self.__dict__.copy()
        params.update(overrides)
        return live_strategy_from_history(
            history_df,
            indicators=params["indicators"],
            periods=params["rolling_minutes"],
            entry_conditions=params["entry_conditions"],
            exit_conditions=params["exit_conditions"],
            side=params["side"],
            history_size=params.get("history_size", 512),
            warmup_batch=params.get("warmup_batch", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the strategy to a JSON-compatible dict."""
        return {
            "schema": "mtrader.strategy",
            "version": 1,
            "name": self.name,
            "indicators": self.indicators,
            "rolling_minutes": self.rolling_minutes,
            "entry_conditions": self.entry_conditions,
            "exit_conditions": self.exit_conditions,
            "side": self.side,
            "target_delta": self.target_delta,
            "stoploss_delta": self.stoploss_delta,
            "target_delta_normalized": self.target_delta_normalized,
            "stoploss_delta_normalized": self.stoploss_delta_normalized,
            "leverage": self.leverage,
            "initial_capital": self.initial_capital,
            "risk_free_rate": self.risk_free_rate,
            "trading_cost_factor": self.trading_cost_factor,
            "capital_per_trade_pct": self.capital_per_trade_pct,
            "sizing_fn": None,  # callable not serializable
            "min_hold_bars": self.min_hold_bars,
            "max_hold_bars": self.max_hold_bars,
            "max_trades_per_day": self.max_trades_per_day,
            "cooldown_bars": self.cooldown_bars,
            "max_daily_loss_pct": self.max_daily_loss_pct,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Strategy:
        """Deserialize a strategy from a dict (inverse of to_dict). Validates the schema field."""
        if data.get("schema") not in {None, "mtrader.strategy"}:
            raise ValueError(f"Unsupported strategy schema: {data.get('schema')}")
        return cls(
            name=data["name"],
            indicators=list(data.get("indicators", [])),
            rolling_minutes=list(data.get("rolling_minutes", [])),
            entry_conditions=data.get("entry_conditions", []),
            exit_conditions=data.get("exit_conditions"),
            side=data.get("side", "buy"),
            target_delta=data.get("target_delta"),
            stoploss_delta=data.get("stoploss_delta"),
            target_delta_normalized=data.get("target_delta_normalized"),
            stoploss_delta_normalized=data.get("stoploss_delta_normalized"),
            leverage=data.get("leverage", 1.0),
            initial_capital=data.get("initial_capital", 1000.0),
            risk_free_rate=data.get("risk_free_rate", 0.0),
            trading_cost_factor=data.get("trading_cost_factor", 0.0002),
            capital_per_trade_pct=data.get("capital_per_trade_pct", 1.0),
            min_hold_bars=data.get("min_hold_bars", 0),
            max_hold_bars=data.get("max_hold_bars"),
            max_trades_per_day=data.get("max_trades_per_day"),
            cooldown_bars=data.get("cooldown_bars", 0),
            max_daily_loss_pct=data.get("max_daily_loss_pct"),
        )

    def save(self, path: str) -> Path:
        """Save the strategy to a JSON file."""
        save_strategy(self, path)
        return Path(path)

    @classmethod
    def load(cls, path: str) -> Strategy:
        """Load a strategy from a JSON file."""
        return load_strategy(path)


def save_strategy(strategy: Strategy, path: str) -> Path:
    """Save a Strategy object to a JSON file."""
    path = Path(path)
    path.write_text(json.dumps(strategy.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_strategy(path: str) -> Strategy:
    """Load a Strategy object from a JSON file."""
    path = Path(path)
    return Strategy.from_dict(json.loads(path.read_text(encoding="utf-8")))


def run_portfolio(data_by_symbol: dict[str, pd.DataFrame], strategy: Strategy, initial_capital: float = 1000) -> dict[str, Any]:
    """Run a strategy across multiple symbols, distributing capital equally. Returns combined equity curve, trades, and per-symbol results."""
    if not data_by_symbol:
        raise ValueError("data_by_symbol cannot be empty")
    capital_per_symbol = initial_capital / len(data_by_symbol)
    results = {}
    equity_parts = []
    trade_parts = []
    for symbol, df in data_by_symbol.items():
        result = strategy.run(df, initial_capital=capital_per_symbol)
        results[symbol] = result
        eq = result.equity[["datetime", "equity"]].copy()
        eq = eq.rename(columns={"equity": symbol})
        equity_parts.append(eq)
        if not result.trades.empty:
            trades = result.trades.copy()
            trades.insert(0, "symbol", symbol)
            trade_parts.append(trades)

    portfolio = equity_parts[0]
    for part in equity_parts[1:]:
        portfolio = pd.merge_asof(
            portfolio.sort_values("datetime"),
            part.sort_values("datetime"),
            on="datetime",
            direction="nearest",
        )
    value_cols = [c for c in portfolio.columns if c != "datetime"]
    portfolio[value_cols] = portfolio[value_cols].ffill().fillna(capital_per_symbol)
    portfolio["equity"] = portfolio[value_cols].sum(axis=1)
    portfolio["drawdown_pct"] = (portfolio["equity"].cummax() - portfolio["equity"]) / portfolio["equity"].cummax() * 100.0
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    return {"results": results, "equity": portfolio, "trades": trades, "final_capital": float(portfolio["equity"].iloc[-1])}


def random_parameter_search(df: pd.DataFrame, strategy_factory: Callable[..., Strategy], param_space: dict[str, list[Any]], n_iter: int = 20, metric: str = "final_capital", seed: int | None = None) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    """Random search over strategy parameter space. Returns (best_params_with_result, results_df)."""
    rng = np.random.default_rng(seed)
    keys = list(param_space)
    rows = []
    best = None
    best_score = -np.inf
    for _ in range(n_iter):
        params = {}
        for key in keys:
            values = param_space[key]
            params[key] = values[int(rng.integers(0, len(values)))]
        strategy = strategy_factory(**params)
        result = strategy.run(df)
        score = result.report.get(metric, result.final_capital)
        if score is None or (isinstance(score, float) and np.isnan(score)):
            score = -np.inf
        row = dict(params)
        row.update({"score": score, "final_capital": result.final_capital, "total_trades": result.report.get("total_trades", 0)})
        rows.append(row)
        if score > best_score:
            best_score = score
            best = {"params": params, "result": result, "score": score}
    return best, pd.DataFrame(rows)


def walk_forward_optimize(df: pd.DataFrame, strategy_factory: Callable[..., Strategy], param_grid: list[dict[str, Any]], train_days: int, test_days: int, metric: str = "final_capital") -> pd.DataFrame:
    """Walk-forward optimization: train on train_days, test on test_days, stepping forward. Returns DataFrame of per-split results."""
    from mtrader.backtest import walk_forward_splits

    rows = []
    for split_no, (train_idx, test_idx) in enumerate(walk_forward_splits(df, train_days, test_days)):
        train_df = df.loc[train_idx].reset_index(drop=True)
        test_df = df.loc[test_idx].reset_index(drop=True)
        best_params = None
        best_score = -np.inf
        for params in param_grid:
            result = strategy_factory(**params).run(train_df)
            score = result.report.get(metric, result.final_capital)
            if score is not None and not pd.isna(score) and score > best_score:
                best_score = score
                best_params = params
        test_result = strategy_factory(**best_params).run(test_df)
        rows.append({
            "split": split_no,
            "train_score": best_score,
            "test_final_capital": test_result.final_capital,
            "test_trades": test_result.report.get("total_trades", 0),
            "params": best_params,
        })
    return pd.DataFrame(rows)


def grid_from_ranges(**params: Any) -> list[dict[str, Any]]:
    """Build a Cartesian product grid from parameter lists (same as parameter_grid)."""
    keys = list(params)
    vals = [v if isinstance(v, (list, tuple, np.ndarray, pd.Index)) else [v] for v in params.values()]
    return [dict(zip(keys, combo)) for combo in product(*vals)]


# ── Strategy presets ──────────────────────────────────────────────

def make_sma_crossover(fast: int = 20, slow: int = 50, side: str = "buy",
                       target_pct: float = 0.5, stop_pct: float = 0.25,
                       **kwargs) -> Strategy:
    """Create an SMA crossover strategy."""
    return Strategy(
        name=f"SMA({fast}/{slow}) Crossover",
        indicators=["sma1"], rolling_minutes=[fast, slow],
        entry_conditions=[[condition(f"can1_sma1_p{fast}", f"can1_sma1_p{slow}", lower=0)]],
        exit_conditions=[[condition(f"can1_sma1_p{slow}", f"can1_sma1_p{fast}", lower=0)]],
        side=side, target_delta_normalized=target_pct, stoploss_delta_normalized=stop_pct,
        **kwargs,
    )


def make_ema_crossover(fast: int = 9, slow: int = 21, side: str = "buy",
                       target_pct: float = 0.5, stop_pct: float = 0.25,
                       **kwargs) -> Strategy:
    """Create an EMA crossover strategy."""
    return Strategy(
        name=f"EMA({fast}/{slow}) Crossover",
        indicators=["ema1"], rolling_minutes=[fast, slow],
        entry_conditions=[[condition(f"can1_ema1_p{fast}", f"can1_ema1_p{slow}", lower=0)]],
        exit_conditions=[[condition(f"can1_ema1_p{slow}", f"can1_ema1_p{fast}", lower=0)]],
        side=side, target_delta_normalized=target_pct, stoploss_delta_normalized=stop_pct,
        **kwargs,
    )


def make_rsi_oversold(period: int = 14, threshold: float = 30,
                      target_pct: float = 0.5, stop_pct: float = 0.25,
                      **kwargs) -> Strategy:
    """Create an RSI oversold bounce (long) or overbought fade (short) strategy."""
    side = kwargs.pop("side", "buy")
    cond = [[condition(f"can1_rsi_p{period}", upper=threshold)]] if side == "buy" else \
           [[condition(f"can1_rsi_p{period}", lower=100 - threshold)]]
    return Strategy(
        name=f"RSI({period}) {'Oversold' if side=='buy' else 'Overbought'}",
        indicators=["rsi"], rolling_minutes=[period],
        entry_conditions=cond,
        side=side, target_delta_normalized=target_pct, stoploss_delta_normalized=stop_pct,
        **kwargs,
    )


def make_macd_crossover(side: str = "buy",
                        target_pct: float = 0.5, stop_pct: float = 0.25,
                        **kwargs) -> Strategy:
    """Create a MACD line/signal crossover strategy."""
    if side == "buy":
        entry = [[condition("can1_macd", "can1_macdsignal", upper=0, shift_first=1, shift_second=1),
                  condition("can1_macd", "can1_macdsignal", lower=0)]]
        exit_c = [[condition("can1_macdsignal", "can1_macd", upper=0, shift_first=1, shift_second=1),
                   condition("can1_macdsignal", "can1_macd", lower=0)]]
    else:
        entry = [[condition("can1_macdsignal", "can1_macd", upper=0, shift_first=1, shift_second=1),
                  condition("can1_macdsignal", "can1_macd", lower=0)]]
        exit_c = [[condition("can1_macd", "can1_macdsignal", upper=0, shift_first=1, shift_second=1),
                   condition("can1_macd", "can1_macdsignal", lower=0)]]
    return Strategy(
        name=f"MACD {'Bullish' if side=='buy' else 'Bearish'} Crossover",
        indicators=["macd"], rolling_minutes=[],
        entry_conditions=entry, exit_conditions=exit_c,
        side=side, target_delta_normalized=target_pct, stoploss_delta_normalized=stop_pct,
        **kwargs,
    )


def make_bollinger_rsi(bb_period: int = 20, rsi_period: int = 14,
                       target_pct: float = 0.5, stop_pct: float = 0.25,
                       **kwargs) -> Strategy:
    """Bollinger Band %B + RSI multi-condition strategy."""
    return Strategy(
        name=f"Bollinger({bb_period})+RSI({rsi_period})",
        indicators=["bbp", "rsi"], rolling_minutes=[bb_period, rsi_period],
        entry_conditions=[[
            condition(f"can1_bbp_p{bb_period}", upper=0.2),
            condition(f"can1_rsi_p{rsi_period}", upper=35),
        ]],
        side="buy", target_delta_normalized=target_pct, stoploss_delta_normalized=stop_pct,
        **kwargs,
    )


PRESET_MAP = {
    "sma_crossover": make_sma_crossover,
    "ema_crossover": make_ema_crossover,
    "rsi_oversold": make_rsi_oversold,
    "macd_crossover": make_macd_crossover,
    "bollinger_rsi": make_bollinger_rsi,
}


def strategy_from_preset(name: str, **overrides) -> Strategy:
    """Create a strategy from a named preset.

    Presets: sma_crossover, ema_crossover, rsi_oversold, macd_crossover, bollinger_rsi

    Usage:
      strat = strategy_from_preset("sma_crossover", fast=10, slow=30, side="sell")
    """
    if name not in PRESET_MAP:
        available = ", ".join(sorted(PRESET_MAP.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESET_MAP[name](**overrides)


# ── Strategy correlation analysis ─────────────────────────────────

def correlate_strategies(
    df: pd.DataFrame,
    strategies: list[Strategy],
    initial_capital: float = 10000,
    top_n: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run multiple strategies on the same data and analyze their correlation.

    Returns a dict with:
      'results': list of (Strategy, BacktestResult) pairs
      'equity_df': DataFrame with equity curves for each strategy
      'correlation': correlation matrix of equity curves
      'diverse_set': recommended subset of strategies with low correlation
      'ranking': combined ranking considering both performance and diversity

    Use 'diverse_set' to pick strategies that work well together (portfolio).
    """
    from itertools import combinations

    results_list = []
    equity_curves = {}

    for i, strat in enumerate(strategies):
        if verbose:
            print(f"  [{i+1}/{len(strategies)}] {strat.name} ... ", end="")
        try:
            r = strat.run(df, initial_capital=initial_capital)
            results_list.append((strat, r))
            eq = r.equity[["datetime", "equity"]].copy()
            eq = eq.rename(columns={"equity": strat.name})
            equity_curves[strat.name] = eq
            if verbose:
                print(f"final={r.final_capital:.0f}, trades={r.report.get('total_trades', 0)}")
        except Exception as e:
            if verbose:
                print(f"FAIL: {e}")

    if not equity_curves:
        return {"results": [], "equity_df": pd.DataFrame(), "correlation": pd.DataFrame(),
                "diverse_set": [], "ranking": pd.DataFrame()}

    # Merge equity curves
    eq_list = list(equity_curves.values())
    equity_df = eq_list[0]
    for eq in eq_list[1:]:
        equity_df = pd.merge_asof(
            equity_df.sort_values("datetime"),
            eq.sort_values("datetime"),
            on="datetime", direction="backward",
        )
    value_cols = [c for c in equity_df.columns if c != "datetime"]
    equity_df[value_cols] = equity_df[value_cols].ffill().fillna(initial_capital)

    # Correlation matrix of daily returns
    daily_returns = equity_df[value_cols].pct_change().dropna()
    corr = daily_returns.cov() if len(daily_returns) < 2 else daily_returns.corr()

    # Build ranking with diversity info
    ranking_rows = []
    for strat, r in results_list:
        ranking_rows.append({
            "name": strat.name,
            "final_capital": r.final_capital,
            "total_trades": r.report.get("total_trades", 0),
            "sharpe": r.metrics.get("Sharpe Ratio"),
        })
    ranking = pd.DataFrame(ranking_rows).sort_values("sharpe", ascending=False)

    # Find diverse set: greedy selection of top strategies with low correlation
    diverse_set = []
    if len(strategies) >= 2 and len(corr) >= 2:
        sorted_strats = ranking["name"].tolist()
        diverse_set.append(sorted_strats[0])
        for s_name in sorted_strats[1:]:
            if s_name not in corr.columns or s_name not in corr.index:
                continue
            max_corr = max(abs(corr.loc[s_name, d]) for d in diverse_set if d in corr.columns)
            if max_corr < 0.7:  # correlation threshold
                diverse_set.append(s_name)
                if top_n and len(diverse_set) >= top_n:
                    break
        if not diverse_set:
            diverse_set = sorted_strats[:1]

    # Final ranking: combined score
    ranking["in_diverse_set"] = ranking["name"].isin(diverse_set)

    return {
        "results": results_list,
        "equity_df": equity_df,
        "correlation": corr,
        "diverse_set": diverse_set,
        "ranking": ranking,
    }
