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
