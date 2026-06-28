"""Automated strategy discovery: generate, evaluate, rank, and validate trading strategies efficiently."""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product, combinations
from typing import Any, Callable
import numpy as np
import pandas as pd
import json
import os
import time


# ── Indicator catalog: what kinds of strategies can be built ──────
TREND_INDICATORS = {
    "sma": {"periods": [10, 20, 30, 50, 100], "label": "SMA"},
    "ema": {"periods": [9, 12, 21, 26, 50], "label": "EMA"},
    "wma": {"periods": [10, 20, 30], "label": "WMA"},
    "ssma": {"periods": [10, 20, 30], "label": "SSMA"},
}

OSCILLATOR_INDICATORS = {
    "rsi": {"periods": [7, 14, 21], "thresholds": [(30, 70), (25, 75), (20, 80)], "label": "RSI"},
    "stochk": {"periods": [14], "thresholds": [(20, 80)], "label": "Stoch%K"},
    "cci": {"periods": [20], "thresholds": [(-100, 100), (-80, 80)], "label": "CCI"},
    "willr": {"periods": [14], "thresholds": [(-80, -20)], "label": "Williams%R"},
    "mfi": {"periods": [14], "thresholds": [(20, 80)], "label": "MFI"},
}

VOLATILITY_INDICATORS = {
    "atr": {"periods": [14], "label": "ATR"},
    "bbp": {"periods": [20], "thresholds": [(0.2, 0.8)], "label": "Bollinger%B"},
}

TREND_FOLLOWING = {
    "macd": {"label": "MACD"},
    "psar": {"label": "PSAR"},
    "supertrend": {"periods": [10], "label": "SuperTrend"},
}

CANDLESTICK = {
    "inside_bar": {"label": "InsideBar"},
    "engulfing": {"label": "Engulfing"},
}

SESSION = {
    "vwap": {"label": "VWAP"},
    "prev_day": {"label": "PrevDay"},
    "gap": {"label": "Gap"},
}


def _compute_weekly_consistency(
    trades_df: pd.DataFrame,
    initial_capital: float = 10000,
) -> dict[str, float]:
    """Compute weekly performance metrics from a trade log.

    Returns dict with:
      total_weeks, profitable_weeks, consistency (profitable/total),
      weekly_sharpe (mean weekly return / std weekly return),
      avg_weekly_return_pct, weekly_volatility_pct
    """
    if trades_df.empty or len(trades_df) < 2:
        return {
            "total_weeks": 0,
            "profitable_weeks": 0,
            "consistency": 0.0,
            "weekly_sharpe": 0.0,
            "avg_weekly_return_pct": 0.0,
            "weekly_volatility_pct": 0.0,
        }

    df = trades_df.copy()
    if "exit_time" in df.columns:
        time_col = "exit_time"
    elif "entry_time" in df.columns:
        time_col = "entry_time"
    else:
        return {
            "total_weeks": 0,
            "profitable_weeks": 0,
            "consistency": 0.0,
            "weekly_sharpe": 0.0,
            "avg_weekly_return_pct": 0.0,
            "weekly_volatility_pct": 0.0,
        }

    # Group by ISO week
    df["week"] = pd.to_datetime(df[time_col]).dt.isocalendar().week.astype(int)
    df["year"] = pd.to_datetime(df[time_col]).dt.isocalendar().year.astype(int)
    df["week_key"] = df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2)

    # Profit per trade
    if "profit" in df.columns:
        profit_col = "profit"
    elif "capital_return_pct" in df.columns:
        profit_col = "capital_return_pct"
    else:
        return {
            "total_weeks": 0,
            "profitable_weeks": 0,
            "consistency": 0.0,
            "weekly_sharpe": 0.0,
            "avg_weekly_return_pct": 0.0,
            "weekly_volatility_pct": 0.0,
        }

    weekly_pnl = df.groupby("week_key")[profit_col].sum()
    total_weeks = len(weekly_pnl)
    profitable_weeks = int((weekly_pnl > 0).sum())
    consistency = profitable_weeks / total_weeks if total_weeks > 0 else 0.0

    weekly_returns_pct = (weekly_pnl / initial_capital) * 100.0
    avg_ret = float(np.mean(weekly_returns_pct)) if len(weekly_returns_pct) > 0 else 0.0
    std_ret = float(np.std(weekly_returns_pct)) if len(weekly_returns_pct) > 1 else 0.0
    weekly_sharpe = (avg_ret / std_ret) if std_ret > 0 else 0.0

    return {
        "total_weeks": total_weeks,
        "profitable_weeks": profitable_weeks,
        "consistency": consistency,
        "weekly_sharpe": weekly_sharpe,
        "avg_weekly_return_pct": avg_ret,
        "weekly_volatility_pct": std_ret,
    }


@dataclass
class StrategyCandidate:
    """A single strategy definition produced by the discovery system."""
    name: str
    indicators: list[str] = field(default_factory=list)
    rolling_minutes: list[int] = field(default_factory=list)
    entry_conditions: list = field(default_factory=list)
    exit_conditions: list | None = None
    side: str = "buy"
    target_delta_normalized: float | None = None
    stoploss_delta_normalized: float | None = None
    exit_type: str = "cross"  # cross / threshold / target_only
    score: float = 0.0
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "indicators": self.indicators,
            "rolling_minutes": self.rolling_minutes,
            "side": self.side,
            "target_delta_normalized": self.target_delta_normalized,
            "stoploss_delta_normalized": self.stoploss_delta_normalized,
            "score": self.score,
            "rank": self.rank,
        }


def _cond(first, second="zero", lower=-np.inf, upper=np.inf, s1=0, s2=0):
    return {"first_column_name": first, "second_column_name": second,
            "shift_down_first": s1, "shift_down_second": s2,
            "lower_range_of_difference": lower, "upper_range_of_difference": upper,
            "perform_normalization_of_diff": False}


# ── Candidate generators ──────────────────────────────────────────

def _generate_crossover_candidates() -> list[StrategyCandidate]:
    """Generate crossover strategies: fast MA crosses slow MA."""
    candidates = []
    base_names = ["sma", "ema", "wma"]
    for base in base_names:
        periods = TREND_INDICATORS[base]["periods"]
        for fast, slow in combinations(periods, 2):
            if fast >= slow:
                continue
            ind_name = f"{base}1"
            col_fast = f"can1_{ind_name}_p{fast}"
            col_slow = f"can1_{ind_name}_p{slow}"
            name = f"{base.upper()}({fast}/{slow}) Crossover"
            candidates.append(StrategyCandidate(
                name=name,
                indicators=[ind_name],
                rolling_minutes=[fast, slow],
                entry_conditions=[[_cond(col_fast, col_slow, lower=0)]],
                exit_conditions=[[_cond(col_slow, col_fast, lower=0)]],
            ))
            # Short version
            candidates.append(StrategyCandidate(
                name=f"{name} (Short)",
                indicators=[ind_name],
                rolling_minutes=[fast, slow],
                entry_conditions=[[_cond(col_slow, col_fast, lower=0)]],
                exit_conditions=[[_cond(col_fast, col_slow, lower=0)]],
                side="sell",
            ))
    return candidates


def _generate_threshold_candidates() -> list[StrategyCandidate]:
    """Generate threshold strategies: oscillator crosses into/out of overbought/oversold."""
    candidates = []
    for ind_name, spec in OSCILLATOR_INDICATORS.items():
        for period in spec["periods"]:
            col = f"can1_{ind_name}_p{period}"
            for lower, upper in spec["thresholds"]:
                # Oversold bounce (long)
                name = f"{spec['label']}({period}) Oversold <{lower}"
                candidates.append(StrategyCandidate(
                    name=name,
                    indicators=[ind_name],
                    rolling_minutes=[period],
                    entry_conditions=[[_cond(col, upper=lower)]],
                    exit_conditions=[[_cond(col, lower=upper)]],
                ))
                # Overbought fade (short)
                name = f"{spec['label']}({period}) Overbought >{upper} (Short)"
                candidates.append(StrategyCandidate(
                    name=name,
                    indicators=[ind_name],
                    rolling_minutes=[period],
                    entry_conditions=[[_cond(col, lower=upper)]],
                    exit_conditions=[[_cond(col, upper=lower)]],
                    side="sell",
                ))
    # Bollinger %B
    for period in VOLATILITY_INDICATORS["bbp"]["periods"]:
        col = f"can1_bbp_p{period}"
        candidates.append(StrategyCandidate(
            name=f"Bollinger%B({period}) Oversold <0.2",
            indicators=["bbp"], rolling_minutes=[period],
            entry_conditions=[[_cond(col, upper=0.2)]],
            exit_conditions=[[_cond(col, lower=0.5)]],
        ))
        candidates.append(StrategyCandidate(
            name=f"Bollinger%B({period}) Overbought >0.8 (Short)",
            indicators=["bbp"], rolling_minutes=[period],
            entry_conditions=[[_cond(col, lower=0.8)]],
            exit_conditions=[[_cond(col, upper=0.5)]],
            side="sell",
        ))
    return candidates


def _generate_macd_candidates() -> list[StrategyCandidate]:
    """Generate MACD crossover strategies."""
    candidates = []
    # MACD line crosses signal line
    candidates.append(StrategyCandidate(
        name="MACD Bullish Crossover",
        indicators=["macd"], rolling_minutes=[],
        entry_conditions=[[
            _cond("can1_macd", "can1_macdsignal", upper=0, s1=1, s2=1),
            _cond("can1_macd", "can1_macdsignal", lower=0),
        ]],
        exit_conditions=[[
            _cond("can1_macdsignal", "can1_macd", upper=0, s1=1, s2=1),
            _cond("can1_macdsignal", "can1_macd", lower=0),
        ]],
    ))
    candidates.append(StrategyCandidate(
        name="MACD Bearish Crossover (Short)",
        indicators=["macd"], rolling_minutes=[],
        entry_conditions=[[
            _cond("can1_macdsignal", "can1_macd", upper=0, s1=1, s2=1),
            _cond("can1_macdsignal", "can1_macd", lower=0),
        ]],
        exit_conditions=[[
            _cond("can1_macd", "can1_macdsignal", upper=0, s1=1, s2=1),
            _cond("can1_macd", "can1_macdsignal", lower=0),
        ]],
        side="sell",
    ))
    return candidates


def _generate_psar_candidates() -> list[StrategyCandidate]:
    """PSAR trend-following strategies."""
    return [
        StrategyCandidate(
            name="PSAR Long (Close above PSAR)",
            indicators=["psar"], rolling_minutes=[],
            entry_conditions=[[_cond("close", "can1_psar", lower=0)]],
            exit_conditions=[[_cond("can1_psar", "close", lower=0)]],
        ),
        StrategyCandidate(
            name="PSAR Short (Close below PSAR)",
            indicators=["psar"], rolling_minutes=[],
            entry_conditions=[[_cond("can1_psar", "close", lower=0)]],
            exit_conditions=[[_cond("close", "can1_psar", lower=0)]],
            side="sell",
        ),
    ]


def _generate_supertrend_candidates() -> list[StrategyCandidate]:
    """SuperTrend direction-following strategies."""
    return [
        StrategyCandidate(
            name="SuperTrend Long (Direction=1)",
            indicators=["supertrend"], rolling_minutes=[10],
            entry_conditions=[[_cond("can1_supertrend_dir_p10", lower=1)]],
            exit_conditions=[[_cond("zero", "can1_supertrend_dir_p10", lower=1)]],
        ),
        StrategyCandidate(
            name="SuperTrend Short (Direction=-1)",
            indicators=["supertrend"], rolling_minutes=[10],
            entry_conditions=[[_cond("can1_supertrend_dir_p10", upper=-1)]],
            exit_conditions=[[_cond("zero", "can1_supertrend_dir_p10", upper=-1)]],
            side="sell",
        ),
    ]


def _generate_vwap_candidates() -> list[StrategyCandidate]:
    """VWAP mean reversion strategies."""
    return [
        StrategyCandidate(
            name="VWAP Pullback Long (Close below VWAP)",
            indicators=["vwap"], rolling_minutes=[],
            entry_conditions=[[_cond("can1_vwap", "close", lower=0)]],
            exit_conditions=[[_cond("close", "can1_vwap", lower=0)]],
        ),
        StrategyCandidate(
            name="VWAP Fade Short (Close above VWAP)",
            indicators=["vwap"], rolling_minutes=[],
            entry_conditions=[[_cond("close", "can1_vwap", lower=0)]],
            exit_conditions=[[_cond("can1_vwap", "close", lower=0)]],
            side="sell",
        ),
    ]


def _generate_price_action_candidates() -> list[StrategyCandidate]:
    """Price action patterns."""
    return [
        StrategyCandidate(
            name="Bullish Engulfing",
            indicators=["engulfing"], rolling_minutes=[],
            entry_conditions=[[_cond("can1_bullish_engulfing", lower=1)]],
        ),
        StrategyCandidate(
            name="Bearish Engulfing (Short)",
            indicators=["engulfing"], rolling_minutes=[],
            entry_conditions=[[_cond("can1_bearish_engulfing", lower=1)]],
            side="sell",
        ),
        StrategyCandidate(
            name="Inside Bar Breakout Long",
            indicators=["inside_bar"], rolling_minutes=[],
            entry_conditions=[[_cond("can1_inside_bar", lower=1, s1=1),
                               _cond("close", "high", lower=0, s2=1)]],
        ),
    ]


def _generate_combined_candidates(
    base_pool: list[StrategyCandidate],
    exit_targets: list[float] | None = None,
    exit_stops: list[float] | None = None,
) -> list[StrategyCandidate]:
    """Generate target/stoploss variants of base strategies."""
    if exit_targets is None:
        exit_targets = [None, 0.5, 1.0]
    if exit_stops is None:
        exit_stops = [None, 0.25, 0.5]

    variants = []
    for base in base_pool:
        for tgt, stp in product(exit_targets, exit_stops):
            if tgt is None and stp is None and base.exit_conditions is None:
                continue
            cand = StrategyCandidate(
                name=base.name,
                indicators=list(base.indicators),
                rolling_minutes=list(base.rolling_minutes),
                entry_conditions=base.entry_conditions,
                exit_conditions=base.exit_conditions,
                side=base.side,
                target_delta_normalized=tgt,
                stoploss_delta_normalized=stp,
            )
            variants.append(cand)
    return variants


# ── Main discovery orchestrator ──────────────────────────────────

def discover_strategies(
    df: pd.DataFrame,
    train_days: int = 3,
    test_days: int = 1,
    strategy_types: list[str] | None = None,
    exit_targets: list[float] | None = None,
    exit_stops: list[float] | None = None,
    metric: str = "sharpe",
    top_n: int = 10,
    initial_capital: float = 10000,
    capital_per_trade_pct: float = 1.0,
    sizing_fn: Callable | None = None,
    min_hold_bars: int = 0,
    max_hold_bars: int | None = None,
    max_trades_per_day: int | None = None,
    cooldown_bars: int = 0,
    max_daily_loss_pct: float | None = None,
    min_trades: int = 10,
    checkpoint_path: str | None = None,
    resume: bool = False,
    checkpoint_freq: int = 50,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[StrategyCandidate]]:
    """Auto-discover profitable strategies from OHLCV data.

    Steps:
    1. Generate candidate strategies from multiple indicator families
    2. Pre-compute all required indicators ONCE
    3. Evaluate each candidate on training data
    4. Filter by min_trades (avoid low-significance strategies)
    5. Validate top candidates on out-of-sample test data
    6. Return ranked results + best StrategyCandidate objects

    Supports checkpoint/resume for long-running discovery:
      checkpoint_path='/path/to/checkpoint_dir'
      resume=True   # resume from last checkpoint

    Parameters
    ----------
    df : DataFrame with OHLCV columns
    train_days : days of training data per walk-forward fold
    test_days : days of testing data per walk-forward fold
    strategy_types : types to include: "crossover", "threshold", "macd",
                     "psar", "supertrend", "vwap", "price_action"
    exit_targets : list of target_delta_normalized values to try (None = condition-only)
    exit_stops : list of stoploss_delta_normalized values to try
    metric : ranking metric ("sharpe", "sortino", "calmar", "profit_factor",
             "final_capital", "consistency", "weekly_sharpe", "win_rate")
    top_n : number of top candidates to walk-forward validate
    initial_capital : starting capital for backtests
    capital_per_trade_pct : fraction of capital deployed per trade (0-1)
    sizing_fn : optional callable(entry_idx, capital_before, df) -> float for per-trade sizing
    min_hold_bars : minimum bars a trade must be held (shorter ones filtered)
    max_hold_bars : maximum bars a trade can be held (longer ones filtered)
    max_trades_per_day : max entries per trading day
    cooldown_bars : minimum bars between consecutive trades
    max_daily_loss_pct : stop trading for the day if loss exceeds this %
    min_trades : minimum trades required for a strategy to be considered
                 (filters out low-significance strategies with few trades)
    checkpoint_path : directory to save checkpoint files for resume support
    resume : if True, load last checkpoint and skip already-evaluated candidates
    checkpoint_freq : save checkpoint every N candidates
    verbose : print progress

    Returns
    -------
    (results_df, best_candidates)
    results_df : DataFrame with columns [name, train_score, test_score,
                  consistency, weekly_sharpe, win_rate, profit_factor, total_trades, ...]
    best_candidates : list of StrategyCandidate objects for the top strategies
    """
    from mtrader.backtest import run_backtest, walk_forward_splits, trade_log
    from mtrader.indicator_engine import add_indicators
    from mtrader.report import backtest_report

    # 1. Generate candidates
    all_candidates: list[StrategyCandidate] = []
    types = strategy_types or ["crossover", "threshold", "macd", "psar", "supertrend", "vwap", "price_action"]

    type_generators = {
        "crossover": _generate_crossover_candidates,
        "threshold": _generate_threshold_candidates,
        "macd": _generate_macd_candidates,
        "psar": _generate_psar_candidates,
        "supertrend": _generate_supertrend_candidates,
        "vwap": _generate_vwap_candidates,
        "price_action": _generate_price_action_candidates,
    }

    for t in types:
        if t in type_generators:
            base = type_generators[t]()
            variants = _generate_combined_candidates(base, exit_targets, exit_stops)
            all_candidates.extend(variants)

    if verbose:
        print(f"  Generated {len(all_candidates)} strategy candidates")

    if len(all_candidates) == 0:
        return pd.DataFrame(), []

    # 2. Pre-compute ALL needed indicators (done once!)
    all_indicators = set()
    all_periods = set()
    for c in all_candidates:
        all_indicators.update(c.indicators)
        all_periods.update(c.rolling_minutes)

    if verbose:
        print(f"  Pre-computing {len(all_indicators)} indicators across {len(all_periods)} periods")

    data = df.copy()
    data = add_indicators(data, add=list(all_indicators) + ["zero"],
                          rolling_minutes=sorted(all_periods) if all_periods else [])
    if "zero" not in data.columns:
        data["zero"] = 0.0

    # 3. Get walk-forward splits
    unique_days = data["datetime"].dt.normalize().unique()
    if test_days <= 0 or len(unique_days) < train_days + test_days:
        split_idx = int(len(data) * 0.7)
        splits = [(np.arange(split_idx), np.arange(split_idx, len(data)))]
    else:
        splits = walk_forward_splits(data, train_days, test_days)

    # ── Checkpoint setup ──────────────────────────────────────────
    _checkpoint_meta_path = None
    _checkpoint_csv_path = None
    if checkpoint_path is not None:
        os.makedirs(checkpoint_path, exist_ok=True)
        _checkpoint_csv_path = os.path.join(checkpoint_path, "discovery_results.csv")
        _checkpoint_meta_path = os.path.join(checkpoint_path, "discovery_meta.json")

    done_set: set[int] = set()
    if resume and checkpoint_path is not None and os.path.exists(_checkpoint_meta_path):
        try:
            with open(_checkpoint_meta_path) as f:
                meta = json.load(f)
            done_set = set(meta.get("done_indices", []))
            if verbose:
                print(f"  Resuming: {len(done_set)} candidates already evaluated")
        except Exception:
            done_set = set()

    def _save_checkpoint(done_indices: list[int], results_rows: list[dict],
                         phase: str = "candidates"):
        if _checkpoint_csv_path is None:
            return
        try:
            meta = {"done_indices": sorted(done_indices), "phase": phase,
                    "timestamp": time.time(), "metric": metric}
            with open(_checkpoint_meta_path, "w") as f:
                json.dump(meta, f)
            if results_rows:
                pd.DataFrame(results_rows).to_csv(_checkpoint_csv_path, index=False)
        except Exception:
            pass

    results_rows = []
    evaluated_candidates: list[StrategyCandidate] = []

    # On resume with full checkpoint, load cached results
    if resume and len(done_set) == len(all_candidates) and _checkpoint_csv_path is not None:
        if verbose:
            print(f"  All {len(all_candidates)} candidates already evaluated. Loading checkpoint...")
        try:
            cached = pd.read_csv(_checkpoint_csv_path)
            if len(cached) == len(all_candidates):
                results_rows = cached.to_dict("records")
                # Rebuild evaluated_candidates from cached scores
                for ci, cand in enumerate(all_candidates):
                    if ci < len(cached):
                        cand.score = cached.iloc[ci].get(f"{metric}_train", -999)
                    evaluated_candidates.append(cand)
                if verbose:
                    print(f"  Loaded {len(results_rows)} cached results")
        except Exception:
            results_rows = []
            evaluated_candidates = []

    # Process in batches for efficiency
    batch_size = max(1, min(checkpoint_freq, len(all_candidates) // 20))
    report_batch = max(1, len(all_candidates) // 20)
    t_start = time.time()

    for ci, cand in enumerate(all_candidates):
        # Skip if already done (resume mode)
        if ci in done_set:
            continue

        train_scores = []
        total_trades_fold = 0
        for fold_no, (train_idx, _) in enumerate(splits[:min(3, len(splits))]):
            train_data = data.iloc[train_idx].copy()
            if len(train_data) < 20:
                continue

            try:
                r = run_backtest(
                    train_data, cand.entry_conditions,
                    buy_or_sell=cand.side,
                    exit_conditions=cand.exit_conditions,
                    indicators=[], rolling_minutes=[],
                    target_delta_normalized=cand.target_delta_normalized,
                    stoploss_delta_normalized=cand.stoploss_delta_normalized,
                    initial_capital=initial_capital,
                    capital_per_trade_pct=capital_per_trade_pct,
                    sizing_fn=sizing_fn,
                    min_hold_bars=min_hold_bars,
                    max_hold_bars=max_hold_bars,
                    max_trades_per_day=max_trades_per_day,
                    cooldown_bars=cooldown_bars,
                    max_daily_loss_pct=max_daily_loss_pct,
                )
                total_trades_fold = r.report.get("total_trades", 0)

                if metric == "sharpe":
                    score = r.metrics.get("Sharpe Ratio", -999)
                elif metric == "sortino":
                    score = r.report.get("sortino_ratio", -999)
                elif metric == "calmar":
                    score = r.report.get("calmar_ratio", -999)
                elif metric == "profit_factor":
                    score = r.report.get("profit_factor", -999)
                elif metric == "win_rate":
                    score = r.report.get("win_rate_pct", -999)
                elif metric == "consistency":
                    wk = _compute_weekly_consistency(r.trades, initial_capital)
                    score = wk["consistency"]
                elif metric == "weekly_sharpe":
                    wk = _compute_weekly_consistency(r.trades, initial_capital)
                    score = wk["weekly_sharpe"]
                else:
                    score = r.final_capital

                if total_trades_fold < min_trades:
                    score = -999
                if score is None or (isinstance(score, float) and np.isnan(score)):
                    score = -999
            except Exception:
                score = -999
                total_trades_fold = 0

            train_scores.append(score)

        if train_scores:
            avg_score = float(np.mean(train_scores))
        else:
            avg_score = -999

        cand.score = avg_score
        evaluated_candidates.append(cand)
        done_set.add(ci)

        wk_metrics = _compute_weekly_consistency(r.trades, initial_capital) if 'r' in dir() else {}
        results_rows.append({
            "name": cand.name,
            "side": cand.side,
            "indicators": ",".join(cand.indicators),
            "target": cand.target_delta_normalized,
            "stoploss": cand.stoploss_delta_normalized,
            f"{metric}_train": avg_score,
            "total_trades": total_trades_fold,
            "consistency": wk_metrics.get("consistency", 0),
            "weekly_sharpe": wk_metrics.get("weekly_sharpe", 0),
            "win_rate": 0.0,
            "profit_factor": 0.0,
        })

        # Progress & checkpoint
        if verbose and (ci + 1) % report_batch == 0:
            elapsed = time.time() - t_start
            done = len(done_set)
            total = len(all_candidates)
            pct = done / total * 100 if total > 0 else 0
            if done > 0:
                eta_sec = (elapsed / done) * (total - done)
                eta_str = f", ETA: {eta_sec:.0f}s" if eta_sec < 3600 else f", ETA: {eta_sec/60:.1f}min"
            else:
                eta_str = ""
            print(f"    [{done}/{total}] ({pct:.0f}%){eta_str}")

        # Save checkpoint
        if checkpoint_path is not None and (ci + 1) % checkpoint_freq == 0:
            _save_checkpoint(list(done_set), results_rows, "candidates")

    # Save final checkpoint after candidate evaluation
    _save_checkpoint(list(done_set), results_rows, "candidates")

    # 4. Sort and pick top_n for walk-forward validation
    results_df = pd.DataFrame(results_rows)
    if results_df.empty:
        return pd.DataFrame(), []
    if "total_trades" in results_df.columns:
        results_df = results_df[results_df["total_trades"] > 0].copy()
    sort_col = f"{metric}_train" if f"{metric}_train" in results_df.columns else "total_trades"
    results_df = results_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    top_df = results_df.head(top_n).copy()
    best_candidates = sorted(evaluated_candidates, key=lambda c: c.score, reverse=True)[:top_n]

    if verbose:
        print(f"\n  Walk-forward validating top {top_n} candidates...")

    # 5. Walk-forward validate top candidates
    for i, cand in enumerate(best_candidates):
        test_scores = []
        total_trades = 0
        for fold_no, (train_idx, test_idx) in enumerate(splits[:min(5, len(splits))]):
            test_data = data.iloc[test_idx].copy()
            if len(test_data) < 5:
                continue
            try:
                r = run_backtest(
                    test_data, cand.entry_conditions,
                    buy_or_sell=cand.side,
                    exit_conditions=cand.exit_conditions,
                    indicators=[], rolling_minutes=[],
                    target_delta_normalized=cand.target_delta_normalized,
                    stoploss_delta_normalized=cand.stoploss_delta_normalized,
                    initial_capital=initial_capital,
                    capital_per_trade_pct=capital_per_trade_pct,
                    sizing_fn=sizing_fn,
                    min_hold_bars=min_hold_bars,
                    max_hold_bars=max_hold_bars,
                    max_trades_per_day=max_trades_per_day,
                    cooldown_bars=cooldown_bars,
                    max_daily_loss_pct=max_daily_loss_pct,
                )
                if metric == "sharpe":
                    s = r.metrics.get("Sharpe Ratio", -999)
                elif metric == "sortino":
                    s = r.report.get("sortino_ratio", -999)
                else:
                    s = r.final_capital
                if s is None or (isinstance(s, float) and np.isnan(s)):
                    s = -999
                test_scores.append(s)
                total_trades += r.report.get("total_trades", 0)
            except Exception:
                test_scores.append(-999)

        avg_test = float(np.mean(test_scores)) if test_scores else -999
        top_df.loc[top_df.index[i], f"{metric}_test"] = avg_test
        top_df.loc[top_df.index[i], "total_trades"] = total_trades

        if verbose:
            print(f"    [{i+1}/{top_n}] {cand.name}: train={cand.score:.3f}, test={avg_test:.3f}, trades={total_trades}")

        # Save checkpoint during validation too
        _save_checkpoint(list(done_set), results_rows, f"validate_{i+1}")

    # 6. Rank by test score, fallback to train score
    sort_col = f"{metric}_test" if f"{metric}_test" in top_df.columns else f"{metric}_train"
    top_df = top_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    for i in range(len(best_candidates)):
        best_candidates[i].rank = i + 1

    # Save final results
    if checkpoint_path is not None:
        final_path = os.path.join(checkpoint_path, "discovery_final_results.csv")
        top_df.to_csv(final_path, index=False)
        if verbose:
            print(f"  Final results saved to {final_path}")

    return top_df, best_candidates


def quick_rank_strategies(
    df: pd.DataFrame,
    strategies: list[StrategyCandidate],
    metric: str = "sharpe",
    initial_capital: float = 10000,
    capital_per_trade_pct: float = 1.0,
    sizing_fn: Callable | None = None,
    min_hold_bars: int = 0,
    max_hold_bars: int | None = None,
    max_trades_per_day: int | None = None,
    cooldown_bars: int = 0,
    max_daily_loss_pct: float | None = None,
    min_trades: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Quickly rank a list of pre-defined strategies on the given data.

    Pre-computes all indicators once, then evaluates each strategy.
    Much faster than running each strategy independently.

    Additional columns in output: consistency, weekly_sharpe, win_rate, profit_factor.
    min_trades filters out strategies with too few trades (avoids overfit).
    """
    from mtrader.backtest import run_backtest
    from mtrader.indicator_engine import add_indicators

    all_inds = set()
    all_periods = set()
    for s in strategies:
        all_inds.update(s.indicators)
        all_periods.update(s.rolling_minutes)

    data = add_indicators(df.copy(), add=list(all_inds) + ["zero"],
                          rolling_minutes=sorted(all_periods) if all_periods else [])
    if "zero" not in data.columns:
        data["zero"] = 0.0

    rows = []
    for i, s in enumerate(strategies):
        if verbose:
            print(f"  [{i + 1}/{len(strategies)}] {s.name} ... ", end="")
        try:
            r = run_backtest(
                data, s.entry_conditions, buy_or_sell=s.side,
                exit_conditions=s.exit_conditions,
                indicators=[], rolling_minutes=[],
                target_delta_normalized=s.target_delta_normalized,
                stoploss_delta_normalized=s.stoploss_delta_normalized,
                initial_capital=initial_capital,
                capital_per_trade_pct=capital_per_trade_pct,
                sizing_fn=sizing_fn,
                min_hold_bars=min_hold_bars,
                max_hold_bars=max_hold_bars,
                max_trades_per_day=max_trades_per_day,
                cooldown_bars=cooldown_bars,
                max_daily_loss_pct=max_daily_loss_pct,
            )
            total_trades = r.report.get("total_trades", 0)
            wk = _compute_weekly_consistency(r.trades, initial_capital)

            if metric == "sharpe":
                score = r.metrics.get("Sharpe Ratio", -999)
            elif metric == "sortino":
                score = r.report.get("sortino_ratio", -999)
            elif metric == "calmar":
                score = r.report.get("calmar_ratio", -999)
            elif metric == "profit_factor":
                score = r.report.get("profit_factor", -999)
            elif metric == "win_rate":
                score = r.report.get("win_rate_pct", -999)
            elif metric == "consistency":
                score = wk["consistency"]
            elif metric == "weekly_sharpe":
                score = wk["weekly_sharpe"]
            else:
                score = r.final_capital

            if total_trades < min_trades:
                score = -999

            rows.append({
                "name": s.name, "side": s.side,
                "trades": total_trades,
                "final_capital": r.final_capital,
                "sharpe": r.metrics.get("Sharpe Ratio"),
                "sortino": r.report.get("sortino_ratio"),
                "calmar": r.report.get("calmar_ratio"),
                "win_rate": r.report.get("win_rate_pct"),
                "profit_factor": r.report.get("profit_factor"),
                "consistency": wk["consistency"],
                "weekly_sharpe": wk["weekly_sharpe"],
                "profitable_weeks": wk["profitable_weeks"],
                "total_weeks": wk["total_weeks"],
            })
            if verbose:
                print(f"score={score:.3f}, trades={total_trades}, "
                      f"consistency={wk['consistency']:.0%}, weeks={wk['total_weeks']}")
        except Exception as e:
            rows.append({"name": s.name, "side": s.side,
                         "trades": 0, "final_capital": initial_capital,
                         metric: -999, "error": str(e),
                         "consistency": 0.0, "weekly_sharpe": 0.0,
                         "profitable_weeks": 0, "total_weeks": 0})
            if verbose:
                print(f"FAIL: {e}")

    results = pd.DataFrame(rows)
    if min_trades > 0 and "trades" in results.columns:
        results = results[results["trades"] >= min_trades].copy()
    sort_by = metric if metric in results.columns else "sharpe"
    if sort_by in results.columns:
        results = results.sort_values(sort_by, ascending=False).reset_index(drop=True)
    return results
