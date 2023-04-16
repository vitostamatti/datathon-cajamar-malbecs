import pandas as pd
from typing import List
import numpy as np


def fillna_by_group(data: pd.DataFrame, cols: List[str], group: List[str]):
    """
    Fills missing values in the specified columns of a pandas DataFrame by taking the mean of the non-missing values
    within groups defined by the specified group columns.

    Args:
        data (pd.DataFrame): The input DataFrame.
        cols (List[str]): A list of column names to fill missing values for.
        group (List[str]): A list of column names to group by.

    Returns:
        pd.DataFrame: The input DataFrame with the specified columns filled with the mean of non-missing values within
            groups defined by the specified group columns.
    """
    for col in cols:
        data[col] = data.groupby(group)[col].transform(
            lambda x: x.fillna(x.mean()))
    return data


def replace_zeros_with_na(data: pd.DataFrame, cols: List[str]):
    """
    Replaces zeros in the specified columns of a pandas DataFrame with NaNs.

    Args:
        data (pd.DataFrame): The input DataFrame.
        cols (List[str]): A list of column names to replace zeros in.

    Returns:
        pd.DataFrame: The input DataFrame with zeros in the specified columns replaced with NaNs.
    """
    for col in cols:
        data[col] = data[col].replace(0, np.nan)
    return data


def fillna_by_value(data: pd.DataFrame, cols: List[str], value=0):
    """
    Fills missing values in the specified columns of a pandas DataFrame with the specified value.

    Args:
        data (pd.DataFrame): The input DataFrame.
        cols (List[str]): A list of column names to fill missing values for.
        value (float, int, optional): The value to fill missing values with. Defaults to 0.

    Returns:
        pd.DataFrame: The input DataFrame with the specified columns filled with the specified value.
    """
    data[cols] = data[cols].fillna(value)
    return data
