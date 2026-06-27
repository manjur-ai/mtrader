import numpy as np
import pandas as pd


def _rolling_sum_strict(values: np.ndarray, period: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if period <= 0:
        raise ValueError("period must be a positive integer")
    if n < period:
        return out

    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    csum = np.cumsum(clean, dtype=np.float64)
    counts = np.cumsum(finite.astype(np.int32))

    sums = csum[period - 1:].copy()
    sums[1:] -= csum[:-period]
    valid_counts = counts[period - 1:].copy()
    valid_counts[1:] -= counts[:-period]
    out[period - 1:] = np.where(valid_counts == period, sums, np.nan)
    return out


def _rolling_mean_strict(values: np.ndarray, period: int) -> np.ndarray:
    return _rolling_sum_strict(values, period) / period


def _rolling_minmax(values: np.ndarray, period: int, find_max: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if period <= 0:
        raise ValueError("period must be a positive integer")
    rolling = pd.Series(values, copy=False).rolling(window=period, min_periods=period)
    result = rolling.max() if find_max else rolling.min()
    return result.to_numpy(dtype=np.float64, copy=False)


def ema(source_numpy: np.ndarray, rolling_minute: int) -> np.ndarray:
    """Exponential Moving Average. EMA = (prev * (p-1) + value * 2) / (p+1) where p = period."""
    n = len(source_numpy)
    if n == 0:
        return np.array([], dtype=np.float64)

    period = rolling_minute
    out = np.zeros(n, dtype=np.float64)
    out[0] = source_numpy[0]

    if n == 1:
        return out

    warmup_end = min(period, n)
    for i in range(1, warmup_end):
        out[i] = (out[i - 1] * i + source_numpy[i] * 2.0) / (i + 2.0)

    if n > period:
        alpha = 2.0 / (period + 1.0)
        y = pd.Series(source_numpy).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=np.float64)
        diff = out[period - 1] - y[period - 1]
        if abs(diff) > 1e-15:
            decay = 1.0 - alpha
            k_arr = np.arange(1, n - period + 1, dtype=np.float64)
            out[period:] = y[period:] + diff * (decay ** k_arr)
        else:
            out[period:] = y[period:]

    return out


def evol(source_numpy: np.ndarray, rolling_minute: int) -> np.ndarray:
    """Exponential volatility: sqrt(EMA of squared deviations from EMA)."""
    source_numpy = np.asarray(source_numpy, dtype=np.float64)
    ema_values = ema(source_numpy, rolling_minute)
    sq_dev = (source_numpy - ema_values) ** 2
    evol_var = ema(sq_dev, rolling_minute)
    return np.sqrt(evol_var)


def wma(source_numpy: np.ndarray, rolling_minute: int) -> np.ndarray:
    """Weighted Moving Average with linearly decreasing weights over the period."""
    n = len(source_numpy)
    cumsum = np.cumsum(source_numpy, dtype=np.float64)
    weights_sum = rolling_minute * (rolling_minute + 1) / 2.0
    wma_values = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i == 0:
            wma_values[i] = source_numpy[i]
        elif i < rolling_minute:
            wma_values[i] = (wma_values[i - 1] * i + 2.0 * source_numpy[i]) / (i + 2.0)
        else:
            cumsum_t = cumsum[i]
            if i > rolling_minute:
                cumsum_t_minus_p_minus1 = cumsum[i - rolling_minute - 1]
            else:
                cumsum_t_minus_p_minus1 = 0.0
            wma_values[i] = (
                wma_values[i - 1]
                + ((rolling_minute + 1.0) * source_numpy[i] - cumsum_t + cumsum_t_minus_p_minus1) / weights_sum
            )

    return wma_values


def wvol(source_numpy: np.ndarray, rolling_minute: int) -> np.ndarray:
    """Weighted volatility: sqrt(WMA of squared deviations from WMA)."""
    source_numpy = np.asarray(source_numpy, dtype=np.float64)
    wma_values = wma(source_numpy, rolling_minute)
    sq_dev = (source_numpy - wma_values) ** 2
    wvol_var = wma(sq_dev, rolling_minute)
    return np.sqrt(wvol_var)


def ssma(source_numpy: np.ndarray, rolling_minute: int) -> np.ndarray:
    """Simple Smoothed Moving Average (equivalent to Wilder's smoothing)."""
    n = len(source_numpy)
    if n == 0:
        return np.array([], dtype=np.float64)

    period = rolling_minute
    out = np.zeros(n, dtype=np.float64)
    out[0] = source_numpy[0]

    if n == 1:
        return out

    warmup_end = min(period, n)
    for i in range(1, warmup_end):
        out[i] = (out[i - 1] * i + source_numpy[i]) / (i + 1.0)

    if n > period:
        alpha = 1.0 / period
        y = pd.Series(source_numpy).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=np.float64)
        diff = out[period - 1] - y[period - 1]
        if abs(diff) > 1e-15:
            decay = 1.0 - alpha
            k_arr = np.arange(1, n - period + 1, dtype=np.float64)
            out[period:] = y[period:] + diff * (decay ** k_arr)
        else:
            out[period:] = y[period:]

    return out


def ssvol(source_numpy: np.ndarray, rolling_minute: int) -> np.ndarray:
    """Smoothed volatility: sqrt(SSMA of squared deviations from SSMA)."""
    source_numpy = np.asarray(source_numpy, dtype=np.float64)
    ssma_values = ssma(source_numpy, rolling_minute)
    sq_dev = (source_numpy - ssma_values) ** 2
    ssvol_var = ssma(sq_dev, rolling_minute)
    return np.sqrt(ssvol_var)


def rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Relative Strength Index. RSI = 100 - 100 / (1 + avg_gain / avg_loss) over the given period."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if period <= 0:
        raise ValueError("period must be a positive integer")
    if n < 2:
        return out
    diffs = np.diff(close)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)

    avg_gain = ssma(gains, period)
    avg_loss = ssma(losses, period)

    gain_part = avg_gain[period - 1:]
    loss_part = avg_loss[period - 1:]
    rs = np.full(n - period, 50.0, dtype=np.float64)
    mask = loss_part > 0
    rs[mask] = 100.0 - 100.0 / (1.0 + gain_part[mask] / loss_part[mask])
    mask2 = (loss_part == 0) & (gain_part > 0)
    rs[mask2] = 100.0
    out[period:] = rs
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range. Mean of True Range (max of high-low, |high-prev_close|, |low-prev_close|) over the period."""
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    high_low = high[1:] - low[1:]
    high_prev_close = np.abs(high[1:] - close[:-1])
    low_prev_close = np.abs(low[1:] - close[:-1])
    tr[1:] = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))
    return _rolling_mean_strict(tr, period)


def stoch_k(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Stochastic %K. %K = 100 * (close - lowest_low) / (highest_high - lowest_low)."""
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    hh = _rolling_minmax(high, period, find_max=True)
    ll = _rolling_minmax(low, period, find_max=False)
    denom = hh - ll
    out = np.where(denom != 0, 100.0 * (close - ll) / denom, 50.0)
    out[:period - 1] = np.nan
    return out


def stoch_d(k_values: np.ndarray, period: int = 3) -> np.ndarray:
    """Stochastic %D (signal line) — simple moving average of %K over the given period."""
    k = np.asarray(k_values, dtype=np.float64)
    return _rolling_mean_strict(k, period)


def bollinger_b(close: np.ndarray, period: int, k: float = 2.0) -> np.ndarray:
    """Bollinger Band %B. %B = (close - lower) / (upper - lower) where upper/lower = SMA ± k * sigma."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if period <= 0:
        raise ValueError("period must be a positive integer")
    if n < period:
        return out
    rolling_sum = _rolling_sum_strict(close, period)
    rolling_sum_sq = _rolling_sum_strict(close * close, period)
    mu = rolling_sum / period
    if period == 1:
        sigma = np.full(n, np.nan, dtype=np.float64)
    else:
        var = (rolling_sum_sq - (rolling_sum * rolling_sum / period)) / (period - 1)
        sigma = np.sqrt(np.maximum(var, 0.0))
    upper = mu + k * sigma
    lower = mu - k * sigma
    denom = upper - lower
    valid = (np.arange(n) >= period - 1) & np.isfinite(denom)
    out[valid] = np.where(denom[valid] != 0, (close[valid] - lower[valid]) / denom[valid], 0.5)
    return out


def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume. Cumulative volume added on up-close days, subtracted on down-close days."""
    volume = np.asarray(volume, dtype=np.float64)
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    out[0] = volume[0]
    for i in range(1, n):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - volume[i]
        else:
            out[i] = out[i - 1]
    return out


def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD. MACD line = EMA_fast - EMA_slow; signal line = EMA of MACD line; histogram = MACD - signal."""
    close = np.asarray(close, dtype=np.float64)
    from mtrader.indicators import ema
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def willr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Williams %R. %R = -100 * (highest_high - close) / (highest_high - lowest_low)."""
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    hh = _rolling_minmax(high, period, find_max=True)
    ll = _rolling_minmax(low, period, find_max=False)
    denom = hh - ll
    out = np.where(denom != 0, -100.0 * (hh - close) / denom, -50.0)
    out[:period - 1] = np.nan
    return out


def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
    """Commodity Channel Index. CCI = (TP - SMA(TP)) / (0.015 * mean_absolute_deviation)."""
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    tp = (high + low + close) / 3.0
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period or period <= 1:
        return out
    idx = np.arange(period)[None, :] + np.arange(n - period + 1)[:, None]
    windows = tp[idx]
    mu = np.mean(windows, axis=1)
    mad = np.mean(np.abs(windows - mu[:, None]), axis=1)
    out[period - 1:] = np.where(mad != 0, (tp[period - 1:] - mu) / (0.015 * mad), 0.0)
    return out


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average Directional Index. ADX = 100 * |PDI - NDI| / (PDI + NDI). Measures trend strength."""
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out
    up = np.zeros(n, dtype=np.float64)
    down = np.zeros(n, dtype=np.float64)
    up[1:] = high[1:] - high[:-1]
    down[1:] = low[:-1] - low[1:]
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    high_low = high[1:] - low[1:]
    high_prev_close = np.abs(high[1:] - close[:-1])
    low_prev_close = np.abs(low[1:] - close[:-1])
    tr[1:] = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr_avg = _rolling_mean_strict(tr, period)
    up_avg = _rolling_mean_strict(plus_dm, period)
    down_avg = _rolling_mean_strict(minus_dm, period)
    pdi = np.divide(100.0 * up_avg, tr_avg, out=np.zeros(n, dtype=np.float64), where=tr_avg != 0)
    ndi = np.divide(100.0 * down_avg, tr_avg, out=np.zeros(n, dtype=np.float64), where=tr_avg != 0)
    denom = pdi + ndi
    out = np.divide(100.0 * np.abs(pdi - ndi), denom, out=np.zeros(n, dtype=np.float64), where=denom != 0)
    out[:period] = np.nan
    return out


def mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
    """Money Flow Index. MFI = 100 - 100 / (1 + money_flow_ratio). Volume-weighted RSI analogue."""
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out
    tp = (high + low + close) / 3.0
    raw = tp * volume
    pos_flow = np.zeros(n, dtype=np.float64)
    neg_flow = np.zeros(n, dtype=np.float64)
    pos_mask = tp[1:] > tp[:-1]
    pos_flow[1:] = np.where(pos_mask, raw[1:], 0.0)
    neg_flow[1:] = np.where(pos_mask, 0.0, raw[1:])
    pos = _rolling_sum_strict(pos_flow, period)
    neg = _rolling_sum_strict(neg_flow, period)
    mfr = np.divide(pos, neg, out=np.full(n, 1e10, dtype=np.float64), where=neg != 0)
    out = 100.0 - 100.0 / (1.0 + mfr)
    out[:period] = np.nan
    return out


def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 10, multiplier: float = 3.0):
    """Supertrend indicator. Returns (supertrend_line, direction) where direction is 1 (up) or -1 (down)."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    line = np.full(n, np.nan, dtype=np.float64)
    direction = np.zeros(n, dtype=np.float64)
    if n == 0:
        return line, direction

    atr_values = atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    upper = hl2 + multiplier * atr_values
    lower = hl2 - multiplier * atr_values
    final_upper = upper.copy()
    final_lower = lower.copy()

    first = period - 1
    if first >= n:
        return line, direction

    direction[first] = 1.0
    line[first] = final_lower[first]
    for i in range(first + 1, n):
        if upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        if line[i - 1] == final_upper[i - 1]:
            direction[i] = 1.0 if close[i] > final_upper[i] else -1.0
        else:
            direction[i] = -1.0 if close[i] < final_lower[i] else 1.0
        line[i] = final_lower[i] if direction[i] > 0 else final_upper[i]

    direction[:first] = np.nan
    return line, direction


def ichimoku(high: np.ndarray, low: np.ndarray, close: np.ndarray, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
    """Ichimoku Kinko Hyo. Returns (tenkan, kijun, senkou_span_a, senkou_span_b, chikou)."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    tenkan_line = (_rolling_minmax(high, tenkan, True) + _rolling_minmax(low, tenkan, False)) / 2.0
    kijun_line = (_rolling_minmax(high, kijun, True) + _rolling_minmax(low, kijun, False)) / 2.0
    span_a = (tenkan_line + kijun_line) / 2.0
    span_b = (_rolling_minmax(high, senkou_b, True) + _rolling_minmax(low, senkou_b, False)) / 2.0
    chikou = np.roll(close, -kijun)
    if kijun > 0:
        chikou[-kijun:] = np.nan
    return tenkan_line, kijun_line, span_a, span_b, chikou


def inside_bar(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Detect inside bars: 1 where current bar's high <= prior high and low >= prior low, else 0."""
    low = np.asarray(low, dtype=np.float64)
    out = np.zeros(len(high), dtype=np.float64)
    if len(high) > 1:
        out[1:] = ((high[1:] <= high[:-1]) & (low[1:] >= low[:-1])).astype(np.float64)
    return out


def bullish_engulfing(open_p: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Detect bullish engulfing patterns: 1 where a bearish bar is followed by a larger bullish bar, else 0."""
    close = np.asarray(close, dtype=np.float64)
    out = np.zeros(len(close), dtype=np.float64)
    if len(close) > 1:
        prev_bear = close[:-1] < open_p[:-1]
        curr_bull = close[1:] > open_p[1:]
        engulfs = (open_p[1:] <= close[:-1]) & (close[1:] >= open_p[:-1])
        out[1:] = (prev_bear & curr_bull & engulfs).astype(np.float64)
    return out


def bearish_engulfing(open_p: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Detect bearish engulfing patterns: 1 where a bullish bar is followed by a larger bearish bar, else 0."""
    close = np.asarray(close, dtype=np.float64)
    out = np.zeros(len(close), dtype=np.float64)
    if len(close) > 1:
        prev_bull = close[:-1] > open_p[:-1]
        curr_bear = close[1:] < open_p[1:]
        engulfs = (open_p[1:] >= close[:-1]) & (close[1:] <= open_p[:-1])
        out[1:] = (prev_bull & curr_bear & engulfs).astype(np.float64)
    return out


def psar(high: np.ndarray, low: np.ndarray, af: float = 0.02, max_af: float = 0.2) -> np.ndarray:
    """Parabolic SAR. Calculates trailing stop levels using acceleration factor (af) and extreme points."""
    low = np.asarray(low, dtype=np.float64)
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    bull = high[0] <= high[1]
    sar = low[0] if bull else high[0]
    ep = high[0] if bull else low[0]
    accel = af
    out[0] = sar
    for i in range(1, n):
        if bull:
            sar = sar + accel * (ep - sar)
            sar = min(sar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < sar:
                bull = False
                sar = ep
                ep = low[i]
                accel = af
            else:
                if high[i] > ep:
                    ep = high[i]
                    accel = min(accel + af, max_af)
        else:
            sar = sar + accel * (ep - sar)
            sar = max(sar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > sar:
                bull = True
                sar = ep
                ep = high[i]
                accel = af
            else:
                if low[i] < ep:
                    ep = low[i]
                    accel = min(accel + af, max_af)
        out[i] = sar
    return out


def heikin_ashi(open_p: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray):
    """Heikin-Ashi candles. Returns (ha_open, ha_high, ha_low, ha_close) using averaged price calculations."""
    open_p = np.asarray(open_p, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    ha_open = np.zeros(n, dtype=np.float64)
    ha_close = np.zeros(n, dtype=np.float64)
    ha_high = np.zeros(n, dtype=np.float64)
    ha_low = np.zeros(n, dtype=np.float64)
    ha_close[:] = (open_p + high + low + close) / 4.0
    ha_open[0] = open_p[0]
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum(np.maximum(ha_open, ha_close), high)
    ha_low = np.minimum(np.minimum(ha_open, ha_close), low)
    return ha_open, ha_high, ha_low, ha_close
