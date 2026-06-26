from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


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
    """Resample OHLCV data to a higher timeframe (e.g. '5T', '1H'). Open=first, high=max, low=min, close=last, volume=sum."""
    required = {"datetime", "open", "high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    data = df.set_index("datetime").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in data.columns:
        agg["volume"] = "sum"
    out = data.resample(rule, label=label, closed=closed).agg(agg).dropna(subset=["open", "high", "low", "close"])
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
    offset = pd.tseries.frequencies.to_offset(rule)
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
