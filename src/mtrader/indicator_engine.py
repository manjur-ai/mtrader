from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Any
from mtrader.indicators import ema, evol, wma, wvol, ssma, ssvol, rsi, atr
from mtrader.indicators import stoch_k, stoch_d, bollinger_b, obv, macd, willr, cci, adx, mfi, psar, heikin_ashi
from mtrader.indicators import supertrend, ichimoku, inside_bar, bullish_engulfing, bearish_engulfing


FEATURE_CODE: dict[str, str] = {
    "0":  "close",
    "1":  "close",
    "2":  "av2",
    "3":  "av3",
    "4":  "av4",
    "5":  "open",
    "6":  "high",
    "7":  "low",
    "8":  "dif1",
    "9":  "ret1",
    "10": "lret1",
    "11": "dif3",
    "12": "ret3",
    "13": "lret3",
    "14": "dif5",
    "15": "ret5",
    "16": "lret5",
    "17": "dif7",
    "18": "ret7",
    "19": "lret7",
    "20": "dif10",
    "21": "ret10",
    "22": "lret10",
    "23": "dif15",
    "24": "ret15",
    "25": "lret15",
    "26": "dif20",
    "27": "ret20",
    "28": "lret20",
    "29": "dif30",
    "30": "ret30",
    "31": "lret30",
    "32": "dif60",
    "33": "ret60",
    "34": "lret60",
}

BASE_CODES_ORDERED: list[str] = sorted(FEATURE_CODE.keys(), key=len, reverse=True)
BASE_NAMES_ORDERED: list[str] = sorted(FEATURE_CODE.values(), key=len, reverse=True)


def add_indicators(df: pd.DataFrame, add: list[str], rolling_minutes: list[int] | None = None, days_back: list[int] | None = None) -> pd.DataFrame:
    if rolling_minutes is None:
        rolling_minutes = []
    if days_back is None:
        days_back = [0]

    if not all(isinstance(minute, int) and minute > 0 for minute in rolling_minutes):
        raise ValueError("All rolling minutes must be positive integers.")
    if not {'high', 'low', 'datetime', 'close', 'open'}.issubset(df.columns):
        raise ValueError("The DataFrame must contain 'high', 'low', 'datetime', 'close', and 'open' columns.")

    if "vwap" in add:
        if 'volume' not in df.columns:
            raise ValueError("The DataFrame must contain a 'volume' column.")
        if df['volume'].sum() == 0:
            raise ValueError("All values in the 'volume' column are zero. VWAP cannot be calculated.")
        valid_volume_count = df['volume'].notna().sum() - (df['volume'] == 0).sum()
        if valid_volume_count < 0.5 * len(df):
            raise ValueError("More than 50% of rows of volume have invalid zero or NaN volume values.")

    temp_add = set()
    full_add = set(add)

    DIST_TO_MA = {
        "smadis": "sma",
        "emadis": "ema",
        "wmadis": "wma",
        "ssmadis": "ssma",
    }

    VOL_TO_MA = {
        "svol": "sma",
        "evol": "ema",
        "wvol": "wma",
        "ssvol": "ssma",
    }

    BASE_INDICATORS = {"sma", "ema", "wma", "ssma", "max", "min"}

    NORMALIZATION_TO_IND = {
        "SMN": "sma",
        "EMN": "ema",
        "WMN": "wma",
        "SSMN": "ssma",
        "SVN": "svol",
        "EVN": "evol",
        "WVN": "wvol",
        "SSVN": "ssvol",
        "BRN": "",
        "TMN": "",
    }

    IND_TO_LEVEL = dict()
    NIND_TO_EXTRA = dict()
    MAX_LEVEL = 0

    changed = True
    while changed:
        changed = False
        temp_add = set()

        for ind in list(full_add):
            pos = ind.find("_")
            if pos == -1:
                continue
            if ind[:pos] == "Z":
                temp_add.add(ind[pos + 1:])
                rest = ind[pos + 1:]
                level = (rest.count("_") + 1) - any(rest.endswith(name) for name in BASE_NAMES_ORDERED)
                IND_TO_LEVEL[ind] = level
                if level > MAX_LEVEL:
                    MAX_LEVEL = level

        for ind in list(full_add):
            pos = ind.find("_")
            if pos == -1:
                continue
            norm = ind[:pos]
            rest = ind[pos + 1:]

            level = (rest.count("_") + 1) - any(rest.endswith(name) for name in BASE_NAMES_ORDERED)
            IND_TO_LEVEL[ind] = level
            if level > MAX_LEVEL:
                MAX_LEVEL = level

            for norm_prefix, operation in NORMALIZATION_TO_IND.items():
                if not norm.startswith(norm_prefix):
                    continue
                temp_add.add(rest)
                suffix = norm[len(norm_prefix):]

                if suffix == "" or suffix == "F":
                    pass
                elif suffix == "P" or suffix == "0":
                    temp_add.add(f"{operation}0")
                    NIND_TO_EXTRA[ind] = f"{operation}0"
                elif suffix == "B":
                    resolved = False
                    for code in BASE_CODES_ORDERED:
                        if rest.endswith(str(code)):
                            temp_add.add(f"{operation}{code}")
                            NIND_TO_EXTRA[ind] = f"{operation}{code}"
                            resolved = True
                            break
                    if not resolved:
                        for name in BASE_NAMES_ORDERED:
                            if rest.endswith(name):
                                temp_add.add(f"{operation}_{name}")
                                NIND_TO_EXTRA[ind] = f"{operation}_{name}"
                                resolved = True
                                break
                    if not resolved:
                        raise ValueError(f"Base source could not be resolved for indicator: {ind}")
                elif suffix in BASE_CODES_ORDERED:
                    temp_add.add(f"{operation}{suffix}")
                    NIND_TO_EXTRA[ind] = f"{operation}{suffix}"
                else:
                    raise ValueError(f"Normalization base suffix not understood: {ind}")
                break

        for code, col in FEATURE_CODE.items():
            needs_base_col = False

            for dist, ma in DIST_TO_MA.items():
                if f"{dist}{code}" in full_add or f"{dist}_{col}" in full_add:
                    needs_base_col = True
                    if f"{ma}{code}" not in full_add and f"{ma}_{col}" not in full_add:
                        temp_add.add(f"{ma}{code}")

            for vol, ma in VOL_TO_MA.items():
                if f"{vol}{code}" in full_add or f"{vol}_{col}" in full_add:
                    needs_base_col = True
                    if f"{ma}{code}" not in full_add and f"{ma}_{col}" not in full_add:
                        temp_add.add(f"{ma}{code}")

            for base in BASE_INDICATORS:
                if f"{base}{code}" in full_add or f"{base}_{col}" in full_add:
                    needs_base_col = True

            if needs_base_col and col not in full_add:
                temp_add.add(col)

        new_items = temp_add - full_add
        if new_items:
            full_add |= new_items
            changed = True

    if "av2" in full_add:
        df["av2"] = (df['high'] + df['low']) / 2.0
    if "av3" in full_add:
        df["av3"] = (df['high'] + df['low'] + df['close']) / 3.0
    if "av4" in full_add:
        df["av4"] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0

    for H in [1, 3, 5, 10, 15, 20, 30, 60]:
        if f"dif{H}" in full_add:
            df[f"dif{H}"] = df['close'] - df['close'].shift(H)
        if f"ret{H}" in full_add:
            df[f"ret{H}"] = (df['close'] - df['close'].shift(H)) / df['close'].shift(H)
        if f"lret{H}" in full_add:
            df[f"lret{H}"] = np.log(df['close']).diff(H)

    group = df.groupby(df['datetime'].dt.date)

    if "zero" in full_add:
        df['zero'] = 0.0

    if "timenum" in full_add:
        df['timenum'] = (df['datetime'].dt.hour * 60) + df['datetime'].dt.minute

    if "ewap" in full_add or "iwap" in full_add or "vwap" in full_add:
        prices = (df['high'] + df['low'] + df['close']) / 3
        df['group_no'] = df.groupby(df['datetime'].dt.date).ngroup()
        df_new = pd.DataFrame(df['group_no'], columns=['group_no'])

        first_rows = df.groupby(df['datetime'].dt.date).head(1)[["group_no"]].copy()
        first_rows["first_index"] = first_rows.index.values

        for day_back in days_back:
            offset_index = (first_rows["first_index"].shift(day_back).fillna(0) - 1).astype(int).values
            first_rows["offset_index"] = offset_index
            first_rows["offset_index_plus1"] = offset_index + 1

            if "ewap" in full_add:
                eweights = pd.Series(np.ones(len(df)))
                price_eweight = prices * eweights
                eweights_cumsum = eweights.cumsum()
                price_eweight_cumsum = price_eweight.cumsum()
                first_rows["eweight_cumsum_offset"] = eweights_cumsum.reindex(offset_index).fillna(0).values
                first_rows["price_eweight_cumsum_offset"] = price_eweight_cumsum.reindex(offset_index).fillna(0).values
                df_new = df_new.merge(first_rows[['group_no', "price_eweight_cumsum_offset", 'eweight_cumsum_offset']], on='group_no', how='left')
                eweight_cumsum_offset = df_new["eweight_cumsum_offset"]
                price_eweight_cumsum_offset = df_new["price_eweight_cumsum_offset"]
                ewap = (price_eweight_cumsum - price_eweight_cumsum_offset) / (eweights_cumsum - eweight_cumsum_offset)
                if day_back == 0:
                    df['can1_ewap'] = ewap
                else:
                    df[f'can1_ewap_d{day_back}'] = ewap
                del eweights, price_eweight, eweights_cumsum, price_eweight_cumsum, eweight_cumsum_offset, price_eweight_cumsum_offset, ewap

            if "iwap" in full_add:
                iweights = pd.Series(range(1, len(df) + 1))
                price_iweight = prices * iweights
                price_cumsum = prices.cumsum()
                iweights_cumsum = iweights.cumsum()
                price_iweight_cumsum = price_iweight.cumsum()
                first_rows["price_cumsum_offset"] = price_cumsum.reindex(offset_index).fillna(0).values
                first_rows["iweight_cumsum_offset"] = iweights_cumsum.reindex(offset_index).fillna(0).values
                first_rows["price_iweight_cumsum_offset"] = price_iweight_cumsum.reindex(offset_index).fillna(0).values
                df_new = df_new.merge(first_rows[['group_no', "offset_index", "price_iweight_cumsum_offset", 'iweight_cumsum_offset', 'price_cumsum_offset', 'offset_index_plus1']], on='group_no', how='left')
                offset_index_in_dfnew = df_new["offset_index"]
                price_cumsum_offset = df_new["price_cumsum_offset"]
                iweight_cumsum_offset = df_new["iweight_cumsum_offset"]
                price_iweight_cumsum_offset = df_new["price_iweight_cumsum_offset"]
                offset_index_plus1 = df_new["offset_index_plus1"]
                iwap_sum_of_weight = ((iweights - offset_index_in_dfnew) * (iweights - offset_index_in_dfnew + 1) / 2).astype(int)
                iwap = (
                    (price_iweight_cumsum - price_iweight_cumsum_offset - (offset_index_plus1 * (price_cumsum - price_cumsum_offset)))
                    / iwap_sum_of_weight
                )
                if day_back == 0:
                    df['can1_iwap'] = iwap
                else:
                    df[f'can1_iwap_d{day_back}'] = iwap
                del iweights, price_iweight, price_cumsum, iweights_cumsum, price_iweight_cumsum, price_cumsum_offset, iweight_cumsum_offset, price_iweight_cumsum_offset, offset_index_plus1, iwap, offset_index_in_dfnew

            if "vwap" in full_add:
                volumes = df['volume']
                price_volume = prices * volumes
                volume_cumsum = volumes.cumsum()
                price_volume_cumsum = price_volume.cumsum()
                first_rows["volume_cumsum_offset"] = volume_cumsum.reindex(offset_index).fillna(0).values
                first_rows["price_volume_cumsum_offset"] = price_volume_cumsum.reindex(offset_index).fillna(0).values
                df_new = df_new.merge(first_rows[['group_no', "volume_cumsum_offset", 'price_volume_cumsum_offset']], on='group_no', how='left')
                volume_cumsum_offset = df_new["volume_cumsum_offset"]
                price_volume_cumsum_offset = df_new["price_volume_cumsum_offset"]
                vwap = (price_volume_cumsum - price_volume_cumsum_offset) / (volume_cumsum - volume_cumsum_offset)
                if day_back == 0:
                    df['can1_vwap'] = vwap
                else:
                    df[f'can1_vwap_d{day_back}'] = vwap
                del volumes, price_volume, volume_cumsum, volume_cumsum_offset, price_volume_cumsum_offset, vwap

        del prices, df_new, first_rows, offset_index
        df.drop(columns=['group_no'], inplace=True)

    for rolling_minute in rolling_minutes:
        if "min" in full_add:
            df[f'can1_min_p{rolling_minute}'] = df['low'].rolling(window=rolling_minute, min_periods=1).min()
        if "max" in full_add:
            df[f'can1_max_p{rolling_minute}'] = df['high'].rolling(window=rolling_minute, min_periods=1).max()

        oper = "sma"
        for code, col in FEATURE_CODE.items():
            ind_code = f"{oper}{code}"
            ind_col = f"{oper}_{col}"
            if ind_code in full_add or ind_col in full_add:
                metric = ind_col if ind_col in full_add else ind_code
                df[f"can1_{metric}_p{rolling_minute}"] = (
                    df[col].rolling(window=rolling_minute, min_periods=1).mean()
                )

        oper = "wma"
        for code, col in FEATURE_CODE.items():
            ind_code = f"{oper}{code}"
            ind_col = f"{oper}_{col}"
            if ind_code in full_add or ind_col in full_add:
                metric = ind_col if ind_col in full_add else ind_code
                calc_values = df[col].to_numpy()
                df[f'can1_{metric}_p{rolling_minute}'] = wma(calc_values, rolling_minute)

        oper = "ema"
        for code, col in FEATURE_CODE.items():
            ind_code = f"{oper}{code}"
            ind_col = f"{oper}_{col}"
            if ind_code in full_add or ind_col in full_add:
                metric = ind_col if ind_col in full_add else ind_code
                calc_values = df[col].to_numpy()
                df[f'can1_{metric}_p{rolling_minute}'] = ema(calc_values, rolling_minute)

        oper = "ssma"
        for code, col in FEATURE_CODE.items():
            ind_code = f"{oper}{code}"
            ind_col = f"{oper}_{col}"
            if ind_code in full_add or ind_col in full_add:
                metric = ind_col if ind_col in full_add else ind_code
                calc_values = df[col].to_numpy()
                df[f'can1_{metric}_p{rolling_minute}'] = ssma(calc_values, rolling_minute)

        oper = "svol"
        for code, col in FEATURE_CODE.items():
            ind_code = f"{oper}{code}"
            ind_col = f"{oper}_{col}"
            if ind_code in full_add or ind_col in full_add:
                metric = ind_col if ind_col in full_add else ind_code
                d_oper = "sma"
                dcol1 = f'can1_{d_oper}{code}_p{rolling_minute}'
                dcol2 = f'can1_{d_oper}_{col}_p{rolling_minute}'
                if dcol1 in df.columns:
                    mu = df[dcol1]
                elif dcol2 in df.columns:
                    mu = df[dcol2]
                else:
                    raise ValueError(f"dep ind: {dcol1} or {dcol2} should be calculated first")
                var = ((df[col] - mu) ** 2).rolling(rolling_minute, min_periods=1).mean()
                df[f'can1_{metric}_p{rolling_minute}'] = np.sqrt(var)

        oper = "wvol"
        for code, col in FEATURE_CODE.items():
            ind_code = f"{oper}{code}"
            ind_col = f"{oper}_{col}"
            if ind_code in full_add or ind_col in full_add:
                metric = ind_col if ind_col in full_add else ind_code
                calc_values = df[col].to_numpy()
                df[f'can1_{metric}_p{rolling_minute}'] = wvol(calc_values, rolling_minute)

        oper = "evol"
        for code, col in FEATURE_CODE.items():
            ind_code = f"{oper}{code}"
            ind_col = f"{oper}_{col}"
            if ind_code in full_add or ind_col in full_add:
                metric = ind_col if ind_col in full_add else ind_code
                calc_values = df[col].to_numpy()
                df[f'can1_{metric}_p{rolling_minute}'] = evol(calc_values, rolling_minute)

        oper = "ssvol"
        for code, col in FEATURE_CODE.items():
            ind_code = f"{oper}{code}"
            ind_col = f"{oper}_{col}"
            if ind_code in full_add or ind_col in full_add:
                metric = ind_col if ind_col in full_add else ind_code
                calc_values = df[col].to_numpy()
                df[f'can1_{metric}_p{rolling_minute}'] = ssvol(calc_values, rolling_minute)

        for level in range(2, MAX_LEVEL + 1):
            for ind in list(full_add):
                if IND_TO_LEVEL.get(ind, -1) != level:
                    continue

                pos = ind.find("_")
                if pos == -1:
                    continue

                norm = ind[:pos]
                rest = ind[pos + 1:]

                dcol1 = f'can1_{rest}_p{rolling_minute}'
                if dcol1 not in df.columns:
                    raise ValueError(f"Dependency indicator missing: {dcol1}")

                source = df[dcol1]
                soure_rolling = source.rolling(rolling_minute, min_periods=rolling_minute)
                source_numpy = source.to_numpy()

                if norm == "Z":
                    std = soure_rolling.std().replace(0, np.nan)
                    mean = soure_rolling.mean()
                    z = (source - mean) / std
                    df[f'can1_{ind}_p{rolling_minute}'] = z

                oper_prefix = "SMN"
                if norm.startswith(oper_prefix):
                    suffix = norm[len(oper_prefix):]
                    if suffix == "" or suffix == "F":
                        mean = soure_rolling.mean().replace(0, np.nan)
                        df[f'can1_{ind}_p{rolling_minute}'] = source / mean
                    elif suffix in {"P", "0", "B"} or suffix in BASE_CODES_ORDERED:
                        extra_ind = NIND_TO_EXTRA[ind]
                        extra_coln = f'can1_{extra_ind}_p{rolling_minute}'
                        if extra_coln in df.columns:
                            deno = df[extra_coln]
                            deno = deno.replace(0, np.nan)
                        else:
                            raise ValueError(f"Dependency indicator missing: {extra_coln}")
                        df[f'can1_{ind}_p{rolling_minute}'] = source / deno

                oper_prefix = "WMN"
                if norm.startswith(oper_prefix):
                    suffix = norm[len(oper_prefix):]
                    if suffix == "" or suffix == "F":
                        df[f'can1_{ind}_p{rolling_minute}'] = source / wma(source_numpy, rolling_minute)
                    elif suffix in {"P", "0", "B"} or suffix in BASE_CODES_ORDERED:
                        extra_ind = NIND_TO_EXTRA[ind]
                        extra_coln = f'can1_{extra_ind}_p{rolling_minute}'
                        if extra_coln in df.columns:
                            deno = df[extra_coln]
                            deno = deno.replace(0, np.nan)
                        else:
                            raise ValueError(f"Dependency indicator missing: {extra_coln}")
                        df[f'can1_{ind}_p{rolling_minute}'] = source / deno

                oper_prefix = "EMN"
                if norm.startswith(oper_prefix):
                    suffix = norm[len(oper_prefix):]
                    if suffix == "" or suffix == "F":
                        df[f'can1_{ind}_p{rolling_minute}'] = source / ema(source_numpy, rolling_minute)
                    elif suffix in {"P", "0", "B"} or suffix in BASE_CODES_ORDERED:
                        extra_ind = NIND_TO_EXTRA[ind]
                        extra_coln = f'can1_{extra_ind}_p{rolling_minute}'
                        if extra_coln in df.columns:
                            deno = df[extra_coln]
                            deno = deno.replace(0, np.nan)
                        else:
                            raise ValueError(f"Dependency indicator missing: {extra_coln}")
                        df[f'can1_{ind}_p{rolling_minute}'] = source / deno

                oper_prefix = "SSMN"
                if norm.startswith(oper_prefix):
                    suffix = norm[len(oper_prefix):]
                    if suffix == "" or suffix == "F":
                        df[f'can1_{ind}_p{rolling_minute}'] = source / ssma(source_numpy, rolling_minute)
                    elif suffix in {"P", "0", "B"} or suffix in BASE_CODES_ORDERED:
                        extra_ind = NIND_TO_EXTRA[ind]
                        extra_coln = f'can1_{extra_ind}_p{rolling_minute}'
                        if extra_coln in df.columns:
                            deno = df[extra_coln]
                            deno = deno.replace(0, np.nan)
                        else:
                            raise ValueError(f"Dependency indicator missing: {extra_coln}")
                        df[f'can1_{ind}_p{rolling_minute}'] = source / deno

                oper_prefix = "SVN"
                if norm.startswith(oper_prefix):
                    suffix = norm[len(oper_prefix):]
                    if suffix == "" or suffix == "F":
                        std = soure_rolling.std().replace(0, np.nan)
                        df[f'can1_{ind}_p{rolling_minute}'] = source / std
                    elif suffix in {"P", "0", "B"} or suffix in BASE_CODES_ORDERED:
                        extra_ind = NIND_TO_EXTRA[ind]
                        extra_coln = f'can1_{extra_ind}_p{rolling_minute}'
                        if extra_coln in df.columns:
                            deno = df[extra_coln]
                            deno = deno.replace(0, np.nan)
                        else:
                            raise ValueError(f"Dependency indicator missing: {extra_coln}")
                        df[f'can1_{ind}_p{rolling_minute}'] = source / deno

                oper_prefix = "WVN"
                if norm.startswith(oper_prefix):
                    suffix = norm[len(oper_prefix):]
                    if suffix == "" or suffix == "F":
                        df[f'can1_{ind}_p{rolling_minute}'] = source / wvol(source_numpy, rolling_minute)
                    elif suffix in {"P", "0", "B"} or suffix in BASE_CODES_ORDERED:
                        extra_ind = NIND_TO_EXTRA[ind]
                        extra_coln = f'can1_{extra_ind}_p{rolling_minute}'
                        if extra_coln in df.columns:
                            deno = df[extra_coln]
                            deno = deno.replace(0, np.nan)
                        else:
                            raise ValueError(f"Dependency indicator missing: {extra_coln}")
                        df[f'can1_{ind}_p{rolling_minute}'] = source / deno

                oper_prefix = "EVN"
                if norm.startswith(oper_prefix):
                    suffix = norm[len(oper_prefix):]
                    if suffix == "" or suffix == "F":
                        df[f'can1_{ind}_p{rolling_minute}'] = source / evol(source_numpy, rolling_minute)
                    elif suffix in {"P", "0", "B"} or suffix in BASE_CODES_ORDERED:
                        extra_ind = NIND_TO_EXTRA[ind]
                        extra_coln = f'can1_{extra_ind}_p{rolling_minute}'
                        if extra_coln in df.columns:
                            deno = df[extra_coln]
                            deno = deno.replace(0, np.nan)
                        else:
                            raise ValueError(f"Dependency indicator missing: {extra_coln}")
                        df[f'can1_{ind}_p{rolling_minute}'] = source / deno

                oper_prefix = "SSVN"
                if norm.startswith(oper_prefix):
                    suffix = norm[len(oper_prefix):]
                    if suffix == "" or suffix == "F":
                        df[f'can1_{ind}_p{rolling_minute}'] = source / ssvol(source_numpy, rolling_minute)
                    elif suffix in {"P", "0", "B"} or suffix in BASE_CODES_ORDERED:
                        extra_ind = NIND_TO_EXTRA[ind]
                        extra_coln = f'can1_{extra_ind}_p{rolling_minute}'
                        if extra_coln in df.columns:
                            deno = df[extra_coln]
                            deno = deno.replace(0, np.nan)
                        else:
                            raise ValueError(f"Dependency indicator missing: {extra_coln}")
                        df[f'can1_{ind}_p{rolling_minute}'] = source / deno

        if "min_inday" in add:
            df[f'can1_min_inday_p{rolling_minute}'] = group['low'].rolling(window=rolling_minute, min_periods=1).min().reset_index(level=0, drop=True)
        if "max_inday" in add:
            df[f'can1_max_inday_p{rolling_minute}'] = group['high'].rolling(window=rolling_minute, min_periods=1).max().reset_index(level=0, drop=True)

        for metric, col in [("ma_inday", "close"), ("ma2_inday", "av2"), ("ma3_inday", "av3"), ("ma4_inday", "av4"), ("ma5_inday", "open"), ("ma6_inday", "high"), ("ma7_inday", "low")]:
            if metric in add:
                df[f'can1_{metric}_p{rolling_minute}'] = group[col].rolling(window=rolling_minute, min_periods=1).mean().reset_index(level=0, drop=True)

        if "or_high" in add or "or_low" in add:
            def _opening_range_expanding(g: pd.DataFrame) -> pd.DataFrame:
                first_n = g.head(rolling_minute)
                if "or_high" in add:
                    g[f'can1_or_high_p{rolling_minute}'] = first_n['high'].cummax()
                if "or_low" in add:
                    g[f'can1_or_low_p{rolling_minute}'] = first_n['low'].cummin()
                return g
            df = df.groupby(df['datetime'].dt.date, group_keys=False).apply(_opening_range_expanding)
            if "or_high" in add:
                df[f'can1_or_high_p{rolling_minute}'] = df.groupby(df['datetime'].dt.date)[f'can1_or_high_p{rolling_minute}'].ffill()
            if "or_low" in add:
                df[f'can1_or_low_p{rolling_minute}'] = df.groupby(df['datetime'].dt.date)[f'can1_or_low_p{rolling_minute}'].ffill()

        if "rsi" in full_add:
            df[f'can1_rsi_p{rolling_minute}'] = rsi(df['close'].to_numpy(), rolling_minute)
        if "atr" in full_add:
            df[f'can1_atr_p{rolling_minute}'] = atr(
                df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy(), rolling_minute
            )
        if "stochk" in full_add:
            df[f'can1_stochk_p{rolling_minute}'] = stoch_k(
                df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy(), rolling_minute
            )
        if "stochd" in full_add:
            k_col = f'can1_stochk_p{rolling_minute}'
            if k_col not in df.columns:
                df[k_col] = stoch_k(
                    df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy(), rolling_minute
                )
            df[f'can1_stochd_p{rolling_minute}'] = stoch_d(df[k_col].to_numpy(), 3)
        if "bbp" in full_add:
            df[f'can1_bbp_p{rolling_minute}'] = bollinger_b(df['close'].to_numpy(), rolling_minute)
        if "willr" in full_add:
            df[f'can1_willr_p{rolling_minute}'] = willr(
                df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy(), rolling_minute)
        if "cci" in full_add:
            df[f'can1_cci_p{rolling_minute}'] = cci(
                df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy(), rolling_minute)
        if "adx" in full_add:
            df[f'can1_adx_p{rolling_minute}'] = adx(
                df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy(), rolling_minute)
        if "mfi" in full_add:
            df[f'can1_mfi_p{rolling_minute}'] = mfi(
                df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy(),
                df['volume'].to_numpy(), rolling_minute)
        if "supertrend" in full_add:
            st, st_dir = supertrend(
                df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy(), rolling_minute
            )
            df[f'can1_supertrend_p{rolling_minute}'] = st
            df[f'can1_supertrend_dir_p{rolling_minute}'] = st_dir

    if "macd" in full_add:
        ml, ms, mh = macd(df['close'].to_numpy())
        df['can1_macd'] = ml
        df['can1_macdsignal'] = ms
        df['can1_macdhist'] = mh

    if "obv" in full_add:
        df['can1_obv'] = obv(df['close'].to_numpy(), df['volume'].to_numpy())

    if "psar" in full_add:
        df['can1_psar'] = psar(df['high'].to_numpy(), df['low'].to_numpy())

    if "ha" in full_add:
        hao, hah, hal, hac = heikin_ashi(
            df['open'].to_numpy(), df['high'].to_numpy(),
            df['low'].to_numpy(), df['close'].to_numpy())
        df['can1_ha_open'] = hao
        df['can1_ha_high'] = hah
        df['can1_ha_low'] = hal
        df['can1_ha_close'] = hac

    if "ichimoku" in full_add:
        tenkan, kijun, span_a, span_b, chikou = ichimoku(
            df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy())
        df['can1_ichi_tenkan'] = tenkan
        df['can1_ichi_kijun'] = kijun
        df['can1_ichi_span_a'] = span_a
        df['can1_ichi_span_b'] = span_b
        df['can1_ichi_chikou'] = chikou

    if "prev_day" in full_add or "pivot" in full_add or "gap" in full_add:
        daily = df.groupby(df['datetime'].dt.date).agg(
            prev_high=('high', 'max'),
            prev_low=('low', 'min'),
            prev_close=('close', 'last'),
            day_open=('open', 'first'),
        ).shift(1)
        date_key = df['datetime'].dt.date
        if "prev_day" in full_add:
            df['can1_prev_day_high'] = date_key.map(daily['prev_high'])
            df['can1_prev_day_low'] = date_key.map(daily['prev_low'])
            df['can1_prev_day_close'] = date_key.map(daily['prev_close'])
        if "pivot" in full_add:
            pivot = (daily['prev_high'] + daily['prev_low'] + daily['prev_close']) / 3.0
            r1 = (2.0 * pivot) - daily['prev_low']
            s1 = (2.0 * pivot) - daily['prev_high']
            df['can1_pivot'] = date_key.map(pivot)
            df['can1_pivot_r1'] = date_key.map(r1)
            df['can1_pivot_s1'] = date_key.map(s1)
        if "gap" in full_add:
            day_open = df.groupby(df['datetime'].dt.date)['open'].transform('first')
            prev_close = date_key.map(daily['prev_close'])
            df['can1_gap_pct'] = ((day_open - prev_close) / prev_close) * 100.0

    if "inside_bar" in full_add:
        df['can1_inside_bar'] = inside_bar(df['high'].to_numpy(), df['low'].to_numpy())

    if "engulfing" in full_add:
        df['can1_bullish_engulfing'] = bullish_engulfing(df['open'].to_numpy(), df['close'].to_numpy())
        df['can1_bearish_engulfing'] = bearish_engulfing(df['open'].to_numpy(), df['close'].to_numpy())

    source_columns = {"datetime", "open", "high", "low", "close", "volume"}
    column_temp_added = [c for c in (full_add - set(add)) if c in df.columns and c not in source_columns]
    df.drop(columns=column_temp_added, inplace=True)

    return df


def add_indicators_on_group(df: pd.DataFrame, group_minutes: list[int], ma: list[int] | None = None, atr: list[int] | None = None) -> pd.DataFrame:
    if not all(isinstance(minute, int) and minute > 0 for minute in group_minutes):
        raise ValueError("All group_minutes must be positive integers.")
    if not {'open', 'high', 'low', 'close'}.issubset(df.columns):
        raise ValueError("The DataFrame must contain 'open', 'high', 'low', 'close' columns.")

    for group_minute in group_minutes:
        all_aggregated_cols = []

        open_col = f'{group_minute}min_open'
        high_col = f'{group_minute}min_high'
        low_col = f'{group_minute}min_low'
        close_col = f'{group_minute}min_close'

        if group_minute != 1:
            all_aggregated_cols.extend([open_col, high_col, low_col, close_col])

        group_index = df.index // group_minute
        aggregator = df.groupby(group_index).agg(
            **{
                open_col: ('open', 'first'),
                high_col: ('high', 'max'),
                low_col: ('low', 'min'),
                close_col: ('close', 'last'),
            }
        )

        if ma:
            for period in ma:
                ma_col = f'{group_minute}min_ma_{period}'
                all_aggregated_cols.append(ma_col)
                aggregator[ma_col] = aggregator[close_col].rolling(window=period, min_periods=1).mean()

        if atr:
            tr_col = f'{group_minute}min_tr'
            all_aggregated_cols.append(tr_col)

            aggregator['previous_close'] = aggregator[close_col].shift(1)
            aggregator[tr_col] = np.maximum.reduce([
                aggregator[high_col] - aggregator[low_col],
                np.abs(aggregator[high_col] - aggregator['previous_close']),
                np.abs(aggregator[low_col] - aggregator['previous_close'])
            ])

            for period in atr:
                atr_col = f'{group_minute}min_atr_{period}'
                all_aggregated_cols.append(atr_col)
                aggregator[atr_col] = aggregator[tr_col].rolling(window=period, min_periods=1).mean()

        aggregated = aggregator.reindex(group_index).reset_index(drop=True)
        df[all_aggregated_cols] = aggregated[all_aggregated_cols]

        mask = (df.index + 1) % group_minute == 0
        df.loc[~mask, all_aggregated_cols] = np.nan

    return df
