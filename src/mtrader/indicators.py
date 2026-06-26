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
