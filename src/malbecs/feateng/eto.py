import pandas as pd
import os


def get_mean_features(eto_data, features, name):
    """
    Calculates the mean value of a set of features for each row in a pandas DataFrame.

    Args:
        eto_data (pandas.DataFrame): the input DataFrame.
        features (list): a list of column names to be used to calculate the mean.
        name (str): the name of the new column that will store the mean value.

    Returns:
        pandas.DataFrame: a copy of the input DataFrame with an additional column storing the mean value of the specified features.
    """
    eto_data[name] = eto_data[list(features)].mean(axis=1)
    return eto_data


def get_total_features(eto_data, features, name):
    """
    Calculates the total value of a set of features for each row in a pandas DataFrame.

    Args:
        eto_data (pandas.DataFrame): the input DataFrame.
        features (list): a list of column names to be used to calculate the total.
        name (str): the name of the new column that will store the total value.

    Returns:
        pandas.DataFrame: a copy of the input DataFrame with an additional column storing the total value of the specified features.
    """
    eto_data[name] = eto_data[list(features)].sum(axis=1)
    return eto_data


def get_std_features(eto_data, features, name):
    """
    Calculates the standard deviation of a set of features for each row in a pandas DataFrame.

    Args:
        eto_data (pandas.DataFrame): the input DataFrame.
        features (list): a list of column names to be used to calculate the standard deviation.
        name (str): the name of the new column that will store the standard deviation value.

    Returns:
        pandas.DataFrame: a copy of the input DataFrame with an additional column storing the standard deviation value of the specified features.
    """
    eto_data[name] = eto_data[list(features)].std(axis=1)
    return eto_data


def feateng_eto(eto_data, output_path=None):
    """
    Performs feature engineering on a pandas DataFrame containing ETO data.

    Args:
        eto_data: The input DataFrame containing ETO data.
        output_path: Optional string representing the path to save the output DataFrame as a CSV file.

    Returns:
        A pandas DataFrame with engineered features for precipitation and snow data. The new columns include
        'MeanPrecip', 'TotalPrecip', 'StdlPrecip', 'MeanSnow', and 'TotalSnow'.
    """

    precip_feats = eto_data.filter(like="SumTotalPrecip").columns
    eto_data = get_mean_features(eto_data, precip_feats, name="MeanPrecip")
    eto_data = get_total_features(eto_data, precip_feats, name="TotalPrecip")
    eto_data = get_std_features(eto_data, precip_feats, name="StdlPrecip")

    snow_feats = eto_data.filter(like="SumTotalSnow").columns
    eto_data = get_mean_features(eto_data, snow_feats, name="MeanSnow")
    eto_data = get_total_features(eto_data, snow_feats, name="TotalSnow")

    if output_path:
        eto_data.to_csv(output_path, index=False)

        dirname = os.path.dirname(output_path)
        with open(os.path.join(dirname, "eto_features.txt"), "w") as f:
            f.write("\n".join(eto_data.columns.to_list()[2:]))

    return eto_data
