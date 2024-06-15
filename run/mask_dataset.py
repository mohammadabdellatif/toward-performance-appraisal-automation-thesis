from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas import Series

from sys_utils.masking import Masking


def mask_cols(row: Series, features: dict[str, object], masking: Masking):
    for col in features:
        f_call = features[col]
        if pd.isna(row[col]):
            continue
        row[col] = f_call(col, str(row[col]), row, masking)
    return row


def dataset_output_path(dataset_path):
    ds_path = Path(dataset_path)
    output_path = ds_path.parent / f'masked_{ds_path.name}'
    return output_path


def mask_csv_dataset(dataset_path: str,
                     index_features: list[str],
                     features: Iterable,
                     mask_pin: int,
                     mask_r_seed: int) -> None:
    df = pd.read_csv(dataset_path, index_col=index_features)
    df = mask_dataset(df, features, mask_pin, mask_r_seed)
    df.to_csv(dataset_output_path(dataset_path))


def mask_excel_dataset(dataset_path: str,
                       index_features: list[int],
                       cols: list[int],
                       features: Iterable,
                       mask_pin: int,
                       mask_r_seed: int) -> None:
    df = pd.read_excel(dataset_path, index_col=index_features, usecols=cols)
    index_names = df.index.names
    df = df.reset_index()
    df = mask_dataset(df, features, mask_pin, mask_r_seed)
    df = df.set_index(index_names)
    df.to_excel(dataset_output_path(dataset_path))


def mask_dataset(df: pd.DataFrame,
                 features: Iterable,
                 mask_pin: int,
                 mask_r_seed: int):
    features_funcs = {}
    for feature in features:
        if type(feature) is str:
            features_funcs[feature] = lambda col, col_val, row, masking: masking.mask(col_val)
        else:
            features_funcs[feature[0]] = feature[1]
    df = df.apply(func=lambda r: mask_cols(r, features_funcs, Masking(pin_code=mask_pin, random_seed=mask_r_seed)),
                  axis='columns')
    return df


def mask_issue_proj(col: str, value, row, masking: Masking):
    # take the first two characters as is as the country code of the project and mask the other values
    if value is None:
        return None
    return masking.mask(value) if len(value) < 6 else value[:2] + masking.mask(value[2:])


def mask_change_history_value(col: str, value, row, masking: Masking):
    if value is None:
        return None
    if row['field'] != 'assignee':
        return value
    return masking.mask(value)


if __name__ == '__main__':
    pin: int = int(input('enter pin: '))
    r_seed: int = int(input('enter random seed: '))
    print('mask issues')
    mask_csv_dataset('../temp_data/issues.csv',
                     ['id'],
                     [('issue_proj', mask_issue_proj), 'issue_reporter', 'issue_assignee'],
                     mask_pin=pin,
                     mask_r_seed=r_seed)
    print('mask issues snapshots')
    mask_csv_dataset('../temp_data/issues_snapshot.csv',
                     ['idx'],
                     [('issue_proj', mask_issue_proj), 'issue_reporter', 'issue_assignee'],
                     mask_pin=pin,
                     mask_r_seed=r_seed)
    print('mask issues change history')
    mask_csv_dataset('../temp_data/issues_change_history.csv',
                     ['id'],
                     [('value', mask_change_history_value)],
                     mask_pin=pin,
                     mask_r_seed=r_seed)
    print('mask sample data')
    mask_excel_dataset('../temp_data/issues_snapshot_sample.xlsx',
                       index_features=[i for i in range(0, 8)],
                       cols=[i for i in range(0, 19)],
                       features=[('project', mask_issue_proj), 'reporter', 'assignee'],
                       mask_pin=pin,
                       mask_r_seed=r_seed
                       )
    print('done')
