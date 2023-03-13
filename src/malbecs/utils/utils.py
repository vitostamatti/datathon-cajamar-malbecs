import pandas as pd
from typing import List
import numpy as np


def fillna_by_group(data: pd.DataFrame, cols: List[str], group: List[str]):
    for col in cols:
        data[col] = data.groupby(group)[col].transform(
            lambda x: x.fillna(x.mean()))
    return data


def replace_zeros_with_na(data: pd.DataFrame, cols: List[str]):
    for col in cols:
        data[col] = data[col].replace(0, np.nan)
    return data


def fillna_by_value(data: pd.DataFrame, cols: List[str], value=0):
    data[cols] = data[cols].fillna(value)
    return data
