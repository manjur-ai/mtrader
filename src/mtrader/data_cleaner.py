import pandas as pd
import numpy as np
import re
from functools import partial
from datetime import datetime, time
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype


def detect_data_types_with_formats(df):
    data_types = {}
    formats = [
        "%Y-%m-%d %H:%M:%S","%Y-%m-%d", "%Y%m%d", "%d-%m-%Y", "%m-%d-%Y", "%d/%m/%Y", "%m/%d/%Y",
        "%H:%M:%S", "%H:%M"
    ]
    datetime_formats = "%Y-%m-%d %H:%M:%S"
    ambiguous_formats = ["%d-%m-%Y", "%m-%d-%Y", "%d/%m/%Y", "%m/%d/%Y"]

    for column in df.columns:
        non_null_values = df[column].dropna()
        if non_null_values.empty:
            data_types[column] = {'type': 'NaN', 'format': None}
            continue

        first_non_null = non_null_values.iloc[0]

        if isinstance(first_non_null, (str, int, np.integer)) and re.fullmatch(r'\d{8}', str(first_non_null)):
            try:
                parsed = pd.to_datetime(non_null_values.astype(str), format='%Y%m%d', errors='raise')
                data_types[column] = {'type': 'date', 'format': '%Y%m%d'}
                continue
            except Exception:
                pass

        if isinstance(first_non_null, (int, np.integer, float, np.floating)):
            val = pd.to_numeric(df[column].iloc[-1], errors='coerce')
            lower, upper = datetime(2000, 1, 1).timestamp(), datetime(2100, 1, 1).timestamp()
            if np.isfinite(val) and val == int(val) and lower <= val <= upper:
                data_types[column] = {'type': 'unixtimestamp', 'format': None}
            elif isinstance(first_non_null, (int, np.integer)):
                data_types[column] = {'type': 'int', 'format': None}
            else:
                data_types[column] = {'type': 'float', 'format': None}
            continue

        elif isinstance(first_non_null, (bool, np.bool_)):
            data_types[column] = {'type': 'bool', 'format': None}
            continue
        elif isinstance(first_non_null, (np.datetime64, datetime, pd.Timestamp)):
            data_types[column] = {'type': 'datetime', 'format': "%Y-%m-%d %H:%M:%S"}
            continue

        elif isinstance(first_non_null, str):
            if all(value == first_non_null for value in non_null_values):
                data_types[column] = {'type': 'str', 'format': 'repeat'}
                continue
            if all(value.strip() == '' for value in non_null_values):
                data_types[column] = {'type': 'str', 'format': 'blank'}
                continue

            detected_format = None
            is_ambiguous = False

            for fmt in formats:
                try:
                    pd.to_datetime(first_non_null, format=fmt)
                    detected_format = fmt
                    if fmt in ambiguous_formats:
                        is_ambiguous = True
                    break
                except ValueError:
                    continue

            if is_ambiguous:
                day_first_count, month_first_count = 0, 0
                for value in non_null_values:
                    try:
                        day, month = map(int, value.split(detected_format[2])[:2])
                        if day > 12:
                            day_first_count += 1
                        elif month > 12:
                            month_first_count += 1
                    except ValueError:
                        continue
                if day_first_count > month_first_count:
                    detected_format = "%d/%m/%Y"
                elif month_first_count > day_first_count:
                    detected_format = "%m/%d/%Y"
                else:
                    detected_format = "ambiguous"

            if detected_format:
                if "H" in detected_format:
                    if "Y" in detected_format:
                        data_types[column] = {'type': 'datetime', 'format': detected_format}
                    else:
                        data_types[column] = {'type': 'time', 'format': detected_format}
                else:
                    if len(value.strip()) > 10:
                        try:
                            pd.to_datetime(first_non_null, format=datetime_formats)
                            detected_format = datetime_formats
                            data_types[column] = {'type': 'datetime', 'format': detected_format}
                        except ValueError:
                            data_types[column] = {'type': 'date', 'format': detected_format}
                    else:
                        data_types[column] = {'type': 'date', 'format': detected_format}
            else:
                data_types[column] = {'type': 'str', 'format': None}
        else:
            data_types[column] = {'type': 'unknown', 'format': None}

    return data_types


def fill_missing_rows(group, start_time, end_time):
    start_time = pd.Timestamp(group['datetime'].iloc[0].date()).replace(hour=start_time.hour, minute=start_time.minute)
    end_time = pd.Timestamp(group['datetime'].iloc[0].date()).replace(hour=end_time.hour, minute=end_time.minute)
    full_range = pd.date_range(start=start_time, end=end_time, freq='min')
    group = group.set_index('datetime').reindex(full_range)
    group.index.name = 'datetime'
    group = group.ffill().bfill().infer_objects()
    group = group.reset_index()
    return group


def clean_data(df_input, start_time=None, end_time=None, min_rec_perday=1, fill_gap=False,
               start_date="0001-01-01", end_date="9999-12-31", share_name="keep_original", create_copy=True,
               column_report=False, multiplier=1, stopprint=False, adjustsplit=False):
    if create_copy:
        df = df_input.copy()
    else:
        df = df_input

    if start_time and end_time:
        try:
            start_time_fmt = pd.to_datetime(start_time, format="%H:%M:%S").time()
            end_time_fmt = pd.to_datetime(end_time, format="%H:%M:%S").time()
        except ValueError:
            start_time_fmt = pd.to_datetime(start_time, format="%H:%M").time()
            end_time_fmt = pd.to_datetime(end_time, format="%H:%M").time()

    if start_date:
        start_date_fmt = datetime.strptime(start_date, "%Y-%m-%d").date()
    if end_date:
        end_date_fmt = datetime.strptime(end_date, "%Y-%m-%d").date()

    data_type_info = detect_data_types_with_formats(df)

    column_tracking = []
    col_id = 0

    for col, info in data_type_info.items():
        column_tracking.append({
            'col_id': col_id,
            'old_col_name': col,
            'col_type': info['type'],
            'col_format': info['format'],
            'new_col_name': None,
            'final_status': 'pending'
        })
        col_id += 1

    new_df = pd.DataFrame()
    name_col, datetime_col, date_col, time_col = None, None, None, None

    if share_name is not None:
        if share_name == "keep_original":
            for item in column_tracking:
                if item['col_type'] == 'str' and item['col_format'] == 'repeat':
                    new_df['name'] = df[item['old_col_name']].astype('string')
                    item['new_col_name'] = 'name'
                    item['final_status'] = 'kept'
                    name_col = item['old_col_name']
                    break

        if not name_col:
            if share_name != "keep_original":
                new_df = pd.DataFrame({'name': [share_name] * len(df)}, dtype='string')
                column_tracking.append({
                    'col_id': col_id,
                    'old_col_name': 'NA',
                    'col_type': 'str',
                    'col_format': 'added',
                    'new_col_name': 'name',
                    'final_status': 'added'
                })
                col_id += 1

    for item in column_tracking:
        if item['col_type'] == 'datetime':
            new_df['datetime'] = pd.to_datetime(df[item['old_col_name']], errors='coerce').astype('datetime64[ns]')
            item['new_col_name'] = 'datetime'
            item['final_status'] = 'kept'
            datetime_col = item['old_col_name']
            break

    if not datetime_col:
        for item in column_tracking:
            if item['col_type'] == 'unixtimestamp':
                new_df['datetime'] = pd.to_datetime(df[item['old_col_name']], unit='s')
                item['new_col_name'] = 'datetime'
                item['final_status'] = 'kept'
                datetime_col = item['old_col_name']
                break

    if not datetime_col:
        for item in column_tracking:
            if item['col_type'] == 'date':
                date_col = item['old_col_name']
                date_format = item['col_format']
            elif item['col_type'] == 'time':
                time_col = item['old_col_name']
                time_format = item['col_format']

        if date_col and time_col:
            combined_format = f"{date_format} {time_format}" if date_format and time_format else None
            df[date_col] = df[date_col].astype(str)
            df[time_col] = df[time_col].astype(str)
            new_df['datetime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col], format=combined_format)
            column_tracking.append({
                'col_id': col_id,
                'old_col_name': 'NA',
                'col_type': 'datetime',
                'col_format': 'added',
                'new_col_name': 'datetime',
                'final_status': 'added'
            })
            col_id += 1
        elif date_col:
            new_df['datetime'] = pd.to_datetime(df[date_col].astype(str) + ' 00:00:00', format=f"{date_format} %H:%M:%S", errors='coerce')
            column_tracking.append({
                'col_id': col_id,
                'old_col_name': 'NA',
                'col_type': 'datetime',
                'col_format': 'added',
                'new_col_name': 'datetime',
                'final_status': 'added'
            })
            col_id += 1

    numerical_cols = [item for item in column_tracking if item['col_type'] in ['int', 'float']]
    if numerical_cols:
        open_col = numerical_cols.pop(0)
        new_df['open'] = df[open_col['old_col_name']].astype('float64')
        open_col['new_col_name'] = 'open'
        open_col['final_status'] = 'kept'

        remaining = numerical_cols[:3]
        if len(remaining) == 3:
            high = max(remaining, key=lambda x: df[x['old_col_name']].sum())
            low = min(remaining, key=lambda x: df[x['old_col_name']].sum())
            close = [x for x in remaining if x != high and x != low][0]

            new_df['high'] = df[high['old_col_name']].astype('float64')
            high['new_col_name'] = 'high'
            high['final_status'] = 'kept'

            new_df['low'] = df[low['old_col_name']].astype('float64')
            low['new_col_name'] = 'low'
            low['final_status'] = 'kept'

            new_df['close'] = df[close['old_col_name']].astype('float64')
            close['new_col_name'] = 'close'
            close['final_status'] = 'kept'

    volume_found = False
    if numerical_cols:
        if len(numerical_cols) >= 4:
            volume_col = numerical_cols[3]
            new_df['volume'] = df[volume_col['old_col_name']].astype('float64')
            item['new_col_name'] = 'volume'
            item['final_status'] = 'kept'
            volume_found = True

    if not volume_found:
        new_df['volume'] = 0.0
        column_tracking.append({
            'col_id': col_id,
            'old_col_name': 'NA',
            'col_type': 'int',
            'col_format': 'added',
            'new_col_name': 'volume',
            'final_status': 'added'
        })
        col_id += 1

    for item in column_tracking:
        if item['final_status'] == 'pending':
            item['final_status'] = 'deleted'

    new_df['datetime'] = new_df['datetime'].dt.floor('min')
    new_df = new_df.drop_duplicates(subset='datetime', keep='first')

    non_zero_mask = new_df['close'] != 0
    cols_to_fill = ['open', 'high', 'low', 'close']
    new_df.loc[~non_zero_mask, cols_to_fill] = new_df.loc[non_zero_mask, cols_to_fill].ffill().reindex(new_df.index)[~non_zero_mask]

    if adjustsplit:
        new_df = new_df.reset_index(drop=True)
        splits = []

        for i in range(1, len(new_df)):
            prev = new_df.loc[i - 1, 'close']
            curr = new_df.loc[i, 'open']

            if prev == 0 or curr == 0:
                continue

            pct_change = abs(curr - prev) / prev

            if pct_change >= 0.30:
                raw_ratio = curr / prev
                ratio = round(raw_ratio)

                if raw_ratio <= 1:
                    inverse_ratio = round(prev / curr)
                    if abs((prev / curr) - inverse_ratio) < 0.1 and inverse_ratio > 1 and inverse_ratio < 5:
                        if not any(i == s['index'] for s in splits):
                            print(f"Split detected at index {i} on {new_df.loc[i, 'datetime']} \u2192 ratio {inverse_ratio}:1")
                            new_df.loc[:i - 1, ['open', 'high', 'low', 'close']] /= inverse_ratio
                            new_df.loc[:i - 1, 'volume'] *= inverse_ratio
                            splits.append({'index': i, 'datetime': new_df.loc[i, 'datetime'], 'ratio': inverse_ratio, 'type': 'split'})
                    elif abs((prev / curr) - inverse_ratio) < 0.2 and inverse_ratio > 4:
                        if not any(i == s['index'] for s in splits):
                            print(f"Split detected at index {i} on {new_df.loc[i, 'datetime']} \u2192 ratio {inverse_ratio}:1")
                            new_df.loc[:i - 1, ['open', 'high', 'low', 'close']] /= inverse_ratio
                            new_df.loc[:i - 1, 'volume'] *= inverse_ratio
                            splits.append({'index': i, 'datetime': new_df.loc[i, 'datetime'], 'ratio': inverse_ratio, 'type': 'split'})
                    elif abs((prev / curr) - (2.0/3.0)) < 0.1 and raw_ratio < 1:
                        if not any(i == s['index'] for s in splits):
                            print(f"Merger detected at index {i} on {new_df.loc[i, 'datetime']} \u2192 reverse split ratio 1:{(2.0/3.0)}")
                            new_df.loc[:i - 1, ['open', 'high', 'low', 'close']] *= (2.0/3.0)
                            new_df.loc[:i - 1, 'volume'] /= (2.0/3.0)
                            splits.append({'index': i, 'datetime': new_df.loc[i, 'datetime'], 'ratio': (2.0/3.0), 'type': 'merger'})
                elif abs(raw_ratio - ratio) < 0.1 and ratio > 1 and ratio < 5:
                    if not any(i == s['index'] for s in splits):
                        print(f"Merger detected at index {i} on {new_df.loc[i, 'datetime']} \u2192 reverse split ratio 1:{ratio}")
                        new_df.loc[:i - 1, ['open', 'high', 'low', 'close']] *= ratio
                        new_df.loc[:i - 1, 'volume'] /= ratio
                        splits.append({'index': i, 'datetime': new_df.loc[i, 'datetime'], 'ratio': ratio, 'type': 'merger'})
                elif abs(raw_ratio - ratio) < 0.2 and ratio > 4:
                    if not any(i == s['index'] for s in splits):
                        print(f"Merger detected at index {i} on {new_df.loc[i, 'datetime']} \u2192 reverse split ratio 1:{ratio}")
                        new_df.loc[:i - 1, ['open', 'high', 'low', 'close']] *= ratio
                        new_df.loc[:i - 1, 'volume'] /= ratio
                        splits.append({'index': i, 'datetime': new_df.loc[i, 'datetime'], 'ratio': ratio, 'type': 'merger'})
                elif abs(raw_ratio - (3.0/2.0)) < 0.1 and raw_ratio > 1 and raw_ratio < 2:
                    if not any(i == s['index'] for s in splits):
                        print(f"Merger detected at index {i} on {new_df.loc[i, 'datetime']} \u2192 reverse split ratio 1:{(3.0/2.0)}")
                        new_df.loc[:i - 1, ['open', 'high', 'low', 'close']] *= (3.0/2.0)
                        new_df.loc[:i - 1, 'volume'] /= (3.0/2.0)
                        splits.append({'index': i, 'datetime': new_df.loc[i, 'datetime'], 'ratio': (3.0/2.0), 'type': 'merger'})

        if splits:
            new_df[['open', 'high', 'low', 'close', 'volume']] = new_df[['open', 'high', 'low', 'close', 'volume']].round(2)

    if start_date:
        new_df = new_df[(new_df['datetime'].dt.date >= start_date_fmt)]
    if end_date:
        new_df = new_df[(new_df['datetime'].dt.date <= end_date_fmt)]

    if start_time and end_time:
        new_df = new_df[
            (new_df['datetime'].dt.time >= start_time_fmt) &
            (new_df['datetime'].dt.time <= end_time_fmt)
        ]

        record_count = new_df.groupby(new_df['datetime'].dt.date).size()
        dates_to_discard = record_count[record_count < min_rec_perday].index
        new_df = new_df[~new_df['datetime'].dt.date.isin(dates_to_discard)]

        if fill_gap:
            original_columns = new_df.columns.tolist()
            fill_missing_rows_partial = partial(fill_missing_rows, start_time=start_time_fmt, end_time=end_time_fmt)
            new_df = new_df.groupby(new_df['datetime'].dt.date, group_keys=False).apply(fill_missing_rows_partial).reset_index(drop=True)
            new_df = new_df[original_columns]

    if multiplier:
        if multiplier != 1:
            new_df[['open', 'high', 'low', 'close']] = new_df[['open', 'high', 'low', 'close']] * multiplier
            new_df[['open', 'high', 'low', 'close']] = new_df[['open', 'high', 'low', 'close']].apply(lambda col: np.round(col, 4))

    new_df = new_df.reset_index(drop=True)

    if not stopprint:
        from inspecty import inspect as mprint
        mprint(new_df)

        record_count = new_df.groupby(new_df['datetime'].dt.date).size()
        min_count = record_count.min()
        max_count = record_count.min()

        if min_count == max_count:
            print(f"all date have same record count :{min_count}")
        else:
            print("\ndate group info sort by count:")
            per_day_stats = new_df.groupby(new_df['datetime'].dt.date).agg(
                count=('datetime', 'size'),
                min_time=('datetime', lambda x: x.min().time()),
                max_time=('datetime', lambda x: x.max().time())
            ).sort_values('count', ascending=True)
            per_day_stats = per_day_stats.rename_axis('date')
            print(per_day_stats)

        if column_report:
            df_report = pd.DataFrame(column_tracking)
            print("\nColumn Change Report:")
            print(df_report[['old_col_name', 'new_col_name', 'final_status']].
                  rename(columns={'old_col_name': 'old_name', 'new_col_name': 'new_name', 'final_status': 'status'}).to_string(index=False, col_space=15))

    return new_df
