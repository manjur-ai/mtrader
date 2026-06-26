import numpy as np


def ema(source_numpy: np.ndarray, rolling_minute: int) -> np.ndarray:
    source_numpy = np.asarray(source_numpy, dtype=np.float64)
    n = len(source_numpy)
    ema_values = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i == 0:
            ema_values[i] = source_numpy[i]
        elif i < rolling_minute:
            ema_values[i] = (ema_values[i - 1] * i + source_numpy[i] * 2.0) / (i + 2.0)
        else:
            ema_values[i] = (
                ema_values[i - 1] * (rolling_minute - 1.0) + source_numpy[i] * 2.0
            ) / (rolling_minute + 1.0)

    return ema_values


def evol(source_numpy: np.ndarray, rolling_minute: int) -> np.ndarray:
    source_numpy = np.asarray(source_numpy, dtype=np.float64)
    ema_values = ema(source_numpy, rolling_minute)
    sq_dev = (source_numpy - ema_values) ** 2
    evol_var = ema(sq_dev, rolling_minute)
    return np.sqrt(evol_var)


def wma(source_numpy: np.ndarray, rolling_minute: int) -> np.ndarray:
    source_numpy = np.asarray(source_numpy, dtype=np.float64)
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
    source_numpy = np.asarray(source_numpy, dtype=np.float64)
    wma_values = wma(source_numpy, rolling_minute)
    sq_dev = (source_numpy - wma_values) ** 2
    wvol_var = wma(sq_dev, rolling_minute)
    return np.sqrt(wvol_var)


def ssma(source_numpy: np.ndarray, rolling_minute: int) -> np.ndarray:
    source_numpy = np.asarray(source_numpy, dtype=np.float64)
    n = len(source_numpy)
    ssma_values = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i == 0:
            ssma_values[i] = source_numpy[i]
        elif i < rolling_minute:
            ssma_values[i] = (ssma_values[i - 1] * i + source_numpy[i]) / (i + 1.0)
        else:
            ssma_values[i] = (ssma_values[i - 1] * (rolling_minute - 1.0) + source_numpy[i]) / rolling_minute

    return ssma_values


def ssvol(source_numpy: np.ndarray, rolling_minute: int) -> np.ndarray:
    source_numpy = np.asarray(source_numpy, dtype=np.float64)
    ssma_values = ssma(source_numpy, rolling_minute)
    sq_dev = (source_numpy - ssma_values) ** 2
    ssvol_var = ssma(sq_dev, rolling_minute)
    return np.sqrt(ssvol_var)


def rsi(close: np.ndarray, period: int) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    diffs = np.diff(close)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    cum_gain = np.cumsum(gains)
    cum_loss = np.cumsum(losses)
    for i in range(period - 1, n):
        if i == period - 1:
            avg_gain = cum_gain[i] / period
            avg_loss = cum_loss[i] / period
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0.0:
            out[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    for i in range(period - 1, n):
        out[i] = np.mean(tr[i - period + 1:i + 1])
    return out


def stoch_k(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        denom = hh - ll
        out[i] = 100.0 * (close[i] - ll) / denom if denom != 0 else 50.0
    return out


def stoch_d(k_values: np.ndarray, period: int = 3) -> np.ndarray:
    k = np.asarray(k_values, dtype=np.float64)
    n = len(k)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        out[i] = np.mean(k[i - period + 1:i + 1])
    return out


def bollinger_b(close: np.ndarray, period: int, k: float = 2.0) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        seg = close[i - period + 1:i + 1]
        mu = np.mean(seg)
        sigma = np.std(seg, ddof=1)
        upper = mu + k * sigma
        lower = mu - k * sigma
        denom = upper - lower
        out[i] = (close[i] - lower) / denom if denom != 0 else 0.5
    return out


def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
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
    close = np.asarray(close, dtype=np.float64)
    from mtrader.indicators import ema
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def willr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        denom = hh - ll
        out[i] = -100.0 * (hh - close[i]) / denom if denom != 0 else -50.0
    return out


def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    tp = (high + low + close) / 3.0
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        seg = tp[i - period + 1:i + 1]
        mu = np.mean(seg)
        mad = np.mean(np.abs(seg - mu))
        out[i] = (tp[i] - mu) / (0.015 * mad) if mad != 0 else 0.0
    return out


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out
    up = np.zeros(n, dtype=np.float64)
    down = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up[i] = high[i] - high[i - 1]
        down[i] = low[i - 1] - low[i]
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    for i in range(1, n):
        up[i] = up[i] if up[i] > down[i] and up[i] > 0 else 0.0
        down[i] = down[i] if down[i] > up[i] and down[i] > 0 else 0.0
    for i in range(period, n):
        tr_avg = np.mean(tr[i - period + 1:i + 1])
        up_avg = np.mean(up[i - period + 1:i + 1])
        down_avg = np.mean(down[i - period + 1:i + 1])
        pdi = 100.0 * up_avg / tr_avg if tr_avg != 0 else 0.0
        ndi = 100.0 * down_avg / tr_avg if tr_avg != 0 else 0.0
        dx = 100.0 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) != 0 else 0.0
        out[i] = dx
    return out


def mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out
    tp = (high + low + close) / 3.0
    raw = tp * volume
    for i in range(period, n):
        pos = 0.0
        neg = 0.0
        for j in range(i - period + 1, i + 1):
            if tp[j] > tp[j - 1]:
                pos += raw[j]
            else:
                neg += raw[j]
        mfr = pos / neg if neg != 0 else 1e10
        out[i] = 100.0 - 100.0 / (1.0 + mfr)
    return out


def psar(high: np.ndarray, low: np.ndarray, af: float = 0.02, max_af: float = 0.2) -> np.ndarray:
    high = np.asarray(high, dtype=np.float64)
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
