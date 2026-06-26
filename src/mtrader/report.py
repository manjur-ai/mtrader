from __future__ import annotations
import numpy as np
import pandas as pd
from html import escape
from pathlib import Path
from typing import Any


def _compute_trade_returns(df: pd.DataFrame, initial_capital: float = 1000) -> tuple[np.ndarray, np.ndarray, float]:
    take_idx = np.where(df["take_trade"].to_numpy())[0]
    if len(take_idx) == 0:
        return np.array([]), np.array([]), initial_capital

    caps = df["capital_at_exit"].to_numpy()
    trade_caps = caps[take_idx]
    entry_caps = np.concatenate([[initial_capital], trade_caps[:-1]])
    trade_returns = (trade_caps - entry_caps) / entry_caps
    return trade_caps, trade_returns, trade_caps[-1]


def backtest_report(df: pd.DataFrame, initial_capital: float = 1000, risk_free_rate: float = 0.05) -> dict[str, Any]:
    """Compute a comprehensive backtest report: total trades, returns, Sharpe/Sortino/Calmar ratios, max drawdown, win rate, profit factor, etc."""
    if "take_trade" not in df.columns or "capital_at_exit" not in df.columns:
        raise ValueError("DataFrame must have 'take_trade' and 'capital_at_exit' columns. "
                         "Run precalculate_exit_time_amount_profit and take_trade_on_condition first.")

    trade_caps, trade_returns, final_capital = _compute_trade_returns(df, initial_capital)
    n_trades = len(trade_returns)

    equity_series = df["capital_at_exit"].to_numpy(dtype=np.float64, copy=True)
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


def equity_curve(df: pd.DataFrame, initial_capital: float = 1000) -> pd.DataFrame:
    """Build an equity curve DataFrame with datetime, equity, drawdown_pct, and trade boolean columns."""
    if "take_trade" not in df.columns or "capital_at_exit" not in df.columns:
        raise ValueError("DataFrame must have 'take_trade' and 'capital_at_exit' columns.")

    eq = df["capital_at_exit"].to_numpy(dtype=np.float64, copy=True)
    eq[eq == 0] = np.nan
    eq = pd.Series(eq).ffill().fillna(initial_capital).to_numpy()

    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100

    result = df[["datetime"]].copy() if "datetime" in df.columns else pd.DataFrame(index=df.index)
    result["equity"] = eq
    result["drawdown_pct"] = dd
    result["trade"] = df["take_trade"].to_numpy()
    return result


def html_backtest_report(
    backtest: Any,
    output_path: str | None = None,
    title: str = "mtrader Backtest Report",
    strategy_name: str | None = None,
    parameters: dict[str, Any] | None = None,
    initial_capital: float = 1000,
    risk_free_rate: float = 0.05,
    max_trades: int = 50,
) -> str:
    """Generate a complete HTML backtest report with metric cards, equity/drawdown charts, trade log, monthly returns, and drawdown periods. Writes to output_path if given. Returns the HTML string."""
    df = getattr(backtest, "df", backtest)
    report = getattr(backtest, "report", None)
    if report is None:
        report = backtest_report(df, initial_capital=initial_capital, risk_free_rate=risk_free_rate)
    equity = getattr(backtest, "equity", None)
    if equity is None:
        equity = equity_curve(df, initial_capital=initial_capital)
    trades = getattr(backtest, "trades", None)
    if trades is None:
        try:
            from mtrader.backtest import trade_log
            trades = trade_log(df, initial_capital=initial_capital)
        except Exception:
            trades = pd.DataFrame()

    parameters = parameters or {}
    html = _render_html_report(
        title=title,
        strategy_name=strategy_name or "Backtest",
        report=report,
        equity=equity,
        trades=trades,
        parameters=parameters,
        max_trades=max_trades,
    )

    if output_path is not None:
        Path(output_path).write_text(html, encoding="utf-8")
    return html


def _render_html_report(title: str, strategy_name: str, report: dict[str, Any], equity: pd.DataFrame, trades: pd.DataFrame, parameters: dict[str, Any], max_trades: int) -> str:
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    cards = [
        ("Final Capital", _money(report.get("final_capital")), "ending account value"),
        ("Total Return", _pct(report.get("total_return_pct")), "net strategy return"),
        ("Sharpe", _value(report.get("sharpe_ratio")), "risk-adjusted return"),
        ("Max Drawdown", _pct(report.get("max_drawdown_pct")), "largest equity decline"),
        ("Trades", _value(report.get("total_trades")), "executed entries"),
        ("Win Rate", _pct(report.get("win_rate_pct")), "profitable trades"),
        ("Profit Factor", _value(report.get("profit_factor")), "gross profit / gross loss"),
        ("Sortino", _value(report.get("sortino_ratio")), "downside-risk return"),
    ]
    card_html = "\n".join(
        f"""
        <section class="metric-card">
          <span>{escape(label)}</span>
          <strong>{escape(str(value))}</strong>
          <small>{escape(help_text)}</small>
        </section>
        """
        for label, value, help_text in cards
    )

    param_html = _parameters_table(parameters)
    trade_html = _trades_table(trades, max_trades)
    monthly_html = _monthly_returns_table(equity)
    drawdown_periods_html = _drawdown_periods_table(equity)
    distribution_svg = _trade_distribution_svg(trades)
    best_worst_html = _best_worst_trades_table(trades)
    equity_svg = _line_chart_svg(
        equity["equity"].to_numpy(dtype=np.float64),
        stroke="#16a34a",
        fill="#dcfce7",
        label="Equity Curve",
    )
    drawdown_svg = _area_chart_svg(
        -equity["drawdown_pct"].to_numpy(dtype=np.float64),
        stroke="#dc2626",
        fill="#fee2e2",
        label="Drawdown",
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --line: #e5e7eb;
      --accent: #2563eb;
      --good: #16a34a;
      --bad: #dc2626;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-end;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.1;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 17px;
      letter-spacing: 0;
    }}
    .subtle {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      height: 28px;
      padding: 0 10px;
      border: 1px solid #bfdbfe;
      color: #1d4ed8;
      background: #eff6ff;
      border-radius: 999px;
      font-size: 13px;
      white-space: nowrap;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .metric-card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }}
    .metric-card {{ padding: 14px; min-height: 104px; }}
    .metric-card span, th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .metric-card strong {{ display: block; margin-top: 8px; font-size: 24px; line-height: 1.1; }}
    .metric-card small {{ display: block; margin-top: 8px; color: var(--muted); line-height: 1.35; }}
    .section-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.65fr);
      gap: 16px;
      margin-bottom: 16px;
    }}
    .panel {{ padding: 16px; overflow: hidden; }}
    .chart {{ width: 100%; height: auto; display: block; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: right; vertical-align: top; }}
    th:first-child, td:first-child {{ text-align: left; }}
    tr:last-child td {{ border-bottom: 0; }}
    .positive {{ color: var(--good); }}
    .negative {{ color: var(--bad); }}
    .empty {{ color: var(--muted); padding: 18px 0; }}
    @media (max-width: 900px) {{
      header, .section-grid {{ display: block; }}
      .badge {{ margin-top: 14px; }}
      .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .panel {{ margin-bottom: 16px; }}
    }}
    @media (max-width: 520px) {{
      main {{ padding: 20px 12px 32px; }}
      .metric-grid {{ grid-template-columns: 1fr; }}
      table {{ font-size: 12px; }}
      th, td {{ padding: 8px 6px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>{escape(strategy_name)}</h1>
        <div class="subtle">Generated {escape(generated_at)} by mtrader</div>
      </div>
      <div class="badge">{escape(title)}</div>
    </header>

    <section class="metric-grid">
      {card_html}
    </section>

    <section class="section-grid">
      <div class="panel">
        <h2>Equity Curve</h2>
        {equity_svg}
      </div>
      <div class="panel">
        <h2>Backtest Parameters</h2>
        {param_html}
      </div>
    </section>

    <section class="panel" style="margin-bottom:16px;">
      <h2>Drawdown</h2>
      {drawdown_svg}
    </section>

    <section class="section-grid">
      <div class="panel">
        <h2>Monthly Returns</h2>
        {monthly_html}
      </div>
      <div class="panel">
        <h2>Trade Distribution</h2>
        {distribution_svg}
      </div>
    </section>

    <section class="section-grid">
      <div class="panel">
        <h2>Drawdown Periods</h2>
        {drawdown_periods_html}
      </div>
      <div class="panel">
        <h2>Best And Worst Trades</h2>
        {best_worst_html}
      </div>
    </section>

    <section class="panel">
      <h2>Trade Log</h2>
      {trade_html}
    </section>
  </main>
</body>
</html>
"""


def _parameters_table(parameters: dict[str, Any]) -> str:
    if not parameters:
        return '<div class="empty">No parameters supplied.</div>'
    rows = []
    for key, value in parameters.items():
        rows.append(
            f"<tr><td>{escape(str(key))}</td><td>{escape(_short_value(value))}</td></tr>"
        )
    return "<table><tbody>" + "\n".join(rows) + "</tbody></table>"


def _trades_table(trades: pd.DataFrame, max_trades: int) -> str:
    if trades is None or trades.empty:
        return '<div class="empty">No trades were generated.</div>'
    cols = [
        "entry_time", "exit_time", "side", "entry_price", "exit_price",
        "return_pct", "capital_at_exit",
    ]
    available = [c for c in cols if c in trades.columns]
    head = "".join(f"<th>{escape(c.replace('_', ' ').title())}</th>" for c in available)
    rows = []
    for _, row in trades.head(max_trades).iterrows():
        cells = []
        for col in available:
            value = row[col]
            cls = ""
            if col.endswith("pct") and pd.notna(value):
                cls = "positive" if value >= 0 else "negative"
            cells.append(f'<td class="{cls}">{escape(_format_cell(value, col))}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    note = ""
    if len(trades) > max_trades:
        note = f'<div class="subtle">Showing first {max_trades} of {len(trades)} trades.</div>'
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>{note}"


def _monthly_returns_table(equity: pd.DataFrame) -> str:
    if "datetime" not in equity.columns or equity.empty:
        return '<div class="empty">No monthly return data available.</div>'
    data = equity[["datetime", "equity"]].copy()
    data["month"] = pd.to_datetime(data["datetime"]).dt.to_period("M")
    monthly = data.groupby("month")["equity"].agg(["first", "last"])
    monthly["return_pct"] = (monthly["last"] - monthly["first"]) / monthly["first"] * 100.0
    if monthly.empty:
        return '<div class="empty">No monthly return data available.</div>'
    rows = []
    for month, row in monthly.tail(24).iterrows():
        cls = "positive" if row["return_pct"] >= 0 else "negative"
        rows.append(
            f'<tr><td>{escape(str(month))}</td><td class="{cls}">{escape(_pct(row["return_pct"]))}</td></tr>'
        )
    return "<table><thead><tr><th>Month</th><th>Return</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _drawdown_periods_table(equity: pd.DataFrame, limit: int = 5) -> str:
    if equity.empty or "drawdown_pct" not in equity.columns:
        return '<div class="empty">No drawdown data available.</div>'
    dd = equity["drawdown_pct"].to_numpy(dtype=np.float64)
    in_dd = dd > 0
    starts = np.where(np.diff(np.concatenate([[False], in_dd]).astype(int)) == 1)[0]
    ends = np.where(np.diff(np.concatenate([in_dd, [False]]).astype(int)) == -1)[0]
    periods = []
    for start, end in zip(starts, ends):
        segment = dd[start:end + 1]
        trough_offset = int(np.argmax(segment))
        trough = start + trough_offset
        periods.append((float(segment[trough_offset]), start, trough, end))
    if not periods:
        return '<div class="empty">No drawdown periods.</div>'
    periods.sort(reverse=True, key=lambda x: x[0])
    rows = []
    for depth, start, trough, end in periods[:limit]:
        rows.append(
            "<tr>"
            f"<td>{escape(_date_at(equity, start))}</td>"
            f"<td>{escape(_date_at(equity, trough))}</td>"
            f"<td>{escape(_date_at(equity, end))}</td>"
            f'<td class="negative">{escape(_pct(depth))}</td>'
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Start</th><th>Trough</th><th>Recovery</th><th>Depth</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def _best_worst_trades_table(trades: pd.DataFrame) -> str:
    if trades is None or trades.empty or "capital_return_pct" not in trades.columns:
        return '<div class="empty">No trade ranking available.</div>'
    ranked = pd.concat([
        trades.nlargest(3, "capital_return_pct"),
        trades.nsmallest(3, "capital_return_pct"),
    ]).drop_duplicates()
    rows = []
    for _, row in ranked.iterrows():
        ret = row.get("capital_return_pct")
        cls = "positive" if pd.notna(ret) and ret >= 0 else "negative"
        rows.append(
            "<tr>"
            f"<td>{escape(_format_cell(row.get('entry_time'), 'entry_time'))}</td>"
            f"<td>{escape(_format_cell(row.get('exit_time'), 'exit_time'))}</td>"
            f'<td class="{cls}">{escape(_pct(ret))}</td>'
            f"<td>{escape(_money(row.get('capital_at_exit')))}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Entry</th><th>Exit</th><th>Return</th><th>Capital</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _trade_distribution_svg(trades: pd.DataFrame) -> str:
    if trades is None or trades.empty:
        return '<div class="empty">No trade distribution available.</div>'
    col = "capital_return_pct" if "capital_return_pct" in trades.columns else "return_pct"
    if col not in trades.columns:
        return '<div class="empty">No trade distribution available.</div>'
    values = trades[col].dropna().to_numpy(dtype=np.float64)
    if len(values) == 0:
        return '<div class="empty">No trade distribution available.</div>'
    counts, edges = np.histogram(values, bins=min(14, max(3, len(values))))
    width, height, pad = 420, 220, 26
    max_count = max(int(counts.max()), 1)
    bar_w = (width - 2 * pad) / len(counts)
    bars = []
    for i, count in enumerate(counts):
        x = pad + i * bar_w
        h = (height - 2 * pad) * count / max_count
        y = height - pad - h
        color = "#16a34a" if edges[i + 1] >= 0 else "#dc2626"
        bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w - 2:.2f}" height="{h:.2f}" fill="{color}" opacity="0.82"></rect>')
    return f"""
    <svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Trade return histogram">
      <rect width="{width}" height="{height}" fill="#ffffff"></rect>
      <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#e5e7eb"></line>
      {''.join(bars)}
      <text x="{pad}" y="16" fill="#6b7280" font-size="12">{escape(_pct(float(values.max())))}</text>
      <text x="{pad}" y="{height - 6}" fill="#6b7280" font-size="12">{escape(_pct(float(values.min())))}</text>
    </svg>
    """


def _line_chart_svg(values: np.ndarray, stroke: str, fill: str, label: str) -> str:
    return _chart_svg(values, stroke=stroke, fill=fill, label=label, area=False)


def _area_chart_svg(values: np.ndarray, stroke: str, fill: str, label: str) -> str:
    return _chart_svg(values, stroke=stroke, fill=fill, label=label, area=True)


def _chart_svg(values: np.ndarray, stroke: str, fill: str, label: str, area: bool) -> str:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    width, height, pad = 900, 260, 24
    if len(values) == 0:
        return '<div class="empty">No chart data available.</div>'
    if len(values) == 1:
        values = np.array([values[0], values[0]], dtype=np.float64)

    min_v = float(np.min(values))
    max_v = float(np.max(values))
    if max_v == min_v:
        max_v = min_v + 1.0
    xs = np.linspace(pad, width - pad, len(values))
    ys = height - pad - ((values - min_v) / (max_v - min_v) * (height - 2 * pad))
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
    area_path = ""
    if area:
        area_path = (
            f'<polygon points="{pad},{height - pad} {points} {width - pad},{height - pad}" '
            f'fill="{fill}" opacity="0.9"></polygon>'
        )
    else:
        area_path = (
            f'<polygon points="{pad},{height - pad} {points} {width - pad},{height - pad}" '
            f'fill="{fill}" opacity="0.55"></polygon>'
        )
    return f"""
    <svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label)}">
      <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"></rect>
      <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#e5e7eb"></line>
      <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#e5e7eb"></line>
      {area_path}
      <polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"></polyline>
      <text x="{pad}" y="16" fill="#6b7280" font-size="12">{escape(_format_number(max_v))}</text>
      <text x="{pad}" y="{height - 6}" fill="#6b7280" font-size="12">{escape(_format_number(min_v))}</text>
    </svg>
    """


def _money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.2f}"


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.2f}%"


def _value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):,.3f}".rstrip("0").rstrip(".")


def _format_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _format_cell(value: Any, col: str) -> str:
    if pd.isna(value):
        return "-"
    if "time" in col:
        try:
            return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(value)
    if col.endswith("pct"):
        return _pct(value)
    if "price" in col or "capital" in col:
        return _money(value)
    return str(value)


def _date_at(equity: pd.DataFrame, index: int) -> str:
    if "datetime" not in equity.columns:
        return str(index)
    return pd.Timestamp(equity["datetime"].iloc[index]).strftime("%Y-%m-%d")


def _short_value(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 120 else text[:117] + "..."


def _max_consecutive(condition: np.ndarray) -> int:
    if len(condition) == 0:
        return 0
    diffs = np.diff(np.concatenate([[False], condition, [False]]).astype(int))
    runs = np.where(diffs == 1)[0], np.where(diffs == -1)[0]
    if len(runs[0]) == 0:
        return 0
    return int(np.max(runs[1] - runs[0]))
