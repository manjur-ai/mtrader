import numpy as np
import pandas as pd


def _compute_trade_returns(df, initial_capital=1000):
    take_idx = np.where(df["take_trade"].to_numpy())[0]
    if len(take_idx) == 0:
        return np.array([]), np.array([]), initial_capital

    caps = df["capital_at_exit"].to_numpy()
    trade_caps = caps[take_idx]
    entry_caps = np.concatenate([[initial_capital], trade_caps[:-1]])
    trade_returns = (trade_caps - entry_caps) / entry_caps
    return trade_caps, trade_returns, trade_caps[-1]


def backtest_report(df, initial_capital=1000, risk_free_rate=0.05):
    if "take_trade" not in df.columns or "capital_at_exit" not in df.columns:
        raise ValueError("DataFrame must have 'take_trade' and 'capital_at_exit' columns. "
                         "Run precalculate_exit_time_amount_profit and take_trade_on_condition first.")

    trade_caps, trade_returns, final_capital = _compute_trade_returns(df, initial_capital)
    n_trades = len(trade_returns)

    equity_series = df["capital_at_exit"].to_numpy()
    equity_series[equity_series == 0] = np.nan
    equity_series = pd.Series(equity_series).ffill().fillna(initial_capital).to_numpy()

    log_returns = np.diff(np.log(np.clip(equity_series, 1e-10, None)))
    total_return_pct = (final_capital - initial_capital) / initial_capital * 100

    n_periods = len(log_returns)
    volatility = np.std(log_returns, ddof=1) if n_periods > 1 else 0.0
    sharpe = ((np.mean(log_returns) - risk_free_rate / 252) / volatility * np.sqrt(252)
              if volatility > 0 else np.nan)

    downside = log_returns[log_returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
    sortino = ((np.mean(log_returns) - risk_free_rate / 252) / downside_std * np.sqrt(252)
               if downside_std > 0 else np.nan)

    peak = np.maximum.accumulate(equity_series)
    drawdowns = (peak - equity_series) / peak * 100
    max_dd = np.max(drawdowns)
    max_dd_idx = np.argmax(drawdowns)
    calmar = total_return_pct / max_dd if max_dd > 0 else np.nan

    report = {
        "total_trades": n_trades,
        "total_return_pct": round(total_return_pct, 2),
        "final_capital": round(final_capital, 2),
        "volatility_annual": round(volatility * np.sqrt(252) * 100, 2),
        "sharpe_ratio": round(sharpe, 3) if not np.isnan(sharpe) else None,
        "sortino_ratio": round(sortino, 3) if not np.isnan(sortino) else None,
        "calmar_ratio": round(calmar, 3) if not np.isnan(calmar) else None,
        "max_drawdown_pct": round(max_dd, 2),
        "max_drawdown_index": int(max_dd_idx),
    }

    if n_trades > 0:
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        n_wins = len(wins)
        n_losses = len(losses)

        report["win_rate_pct"] = round(n_wins / n_trades * 100, 2)
        report["avg_win_pct"] = round(np.mean(wins) * 100, 3) if n_wins > 0 else 0.0
        report["avg_loss_pct"] = round(np.mean(losses) * 100, 3) if n_losses > 0 else 0.0
        report["best_trade_pct"] = round(np.max(wins) * 100, 3) if n_wins > 0 else 0.0
        report["worst_trade_pct"] = round(np.min(losses) * 100, 3) if n_losses > 0 else 0.0

        gross_profit = np.sum(wins)
        gross_loss = -np.sum(losses) if n_losses > 0 else 1e-10
        report["profit_factor"] = round(gross_profit / gross_loss, 3)

        cons_win = _max_consecutive(trade_returns > 0)
        cons_loss = _max_consecutive(trade_returns < 0)
        report["max_consecutive_wins"] = cons_win
        report["max_consecutive_losses"] = cons_loss
    else:
        report.update({
            "win_rate_pct": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "best_trade_pct": 0.0, "worst_trade_pct": 0.0,
            "profit_factor": 0.0, "max_consecutive_wins": 0, "max_consecutive_losses": 0,
        })

    return report


def equity_curve(df, initial_capital=1000):
    if "take_trade" not in df.columns or "capital_at_exit" not in df.columns:
        raise ValueError("DataFrame must have 'take_trade' and 'capital_at_exit' columns.")

    eq = df["capital_at_exit"].to_numpy().copy()
    eq[eq == 0] = np.nan
    eq = pd.Series(eq).ffill().fillna(initial_capital).to_numpy()

    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100

    result = df[["datetime"]].copy() if "datetime" in df.columns else pd.DataFrame(index=df.index)
    result["equity"] = eq
    result["drawdown_pct"] = dd
    result["trade"] = df["take_trade"].to_numpy()
    return result


def _max_consecutive(condition):
    if len(condition) == 0:
        return 0
    diffs = np.diff(np.concatenate([[False], condition, [False]]).astype(int))
    runs = np.where(diffs == 1)[0], np.where(diffs == -1)[0]
    if len(runs[0]) == 0:
        return 0
    return int(np.max(runs[1] - runs[0]))
