from malbecs.utils import fillna_by_group, replace_zeros_with_na
import pandas as pd
import numpy as np

def load_wine_dataset(path: str) -> pd.DataFrame:
    """
    Loads a wine dataset from a file and returns it as a pandas DataFrame.

    Args:
        path (str): The path to the wine dataset file to load.

    Returns:
        pd.DataFrame: The loaded wine dataset as a pandas DataFrame.
    """
    # Load the wine dataset using pandas and set the separator parameter.
    wine_dataset = pd.read_csv(path, sep='|')

    # Return the loaded wine dataset as a pandas DataFrame.
    return wine_dataset


def norm_columns(wine: pd.DataFrame) -> pd.DataFrame:
    """
    Renames the columns of a wine dataset to a standard set of column names.

    Args:
        wine (pd.DataFrame): The wine dataset to normalize.

    Returns:
        pd.DataFrame: The normalized wine dataset with standardized column names.
    """
    # Define the new column names to use.
    new_cols = [
        'campaña', 'id_finca', 'id_zona',
        'id_estacion', 'altitud', 'variedad',
        'modo', 'tipo', 'color', 'superficie', 'produccion'
    ]
    # Rename the columns of the wine dataset to the new column names.
    wine.columns = new_cols

    # Return the normalized wine dataset with standardized column names.
    return wine


def process_altitud(data):
    """
    Takes in a pandas dataframe and processes the 'altitud' column by converting any altitude ranges
    (represented as a string with a dash) into the average of the range as a float.

    Args:
        data (pandas.DataFrame): The input dataframe to process.

    Returns:
        pandas.DataFrame: The processed dataframe with the 'altitud' column transformed.
    """

    def transform_altitud(alt):
        """
        Helper function that takes in an altitude value and converts it to the average value if it's a
        range (represented as a string with a dash) or returns the value as is if it's already a float.

        Args:
            alt (float or str): The altitude value to transform.

        Returns:
            float: The transformed altitude value.
        """

        if type(alt) is str:
            alt_list = alt.split("-")
            alt_list = list(map(float, alt_list))
            return np.mean(alt_list)

        return alt

    # apply the transformation to the 'altitud' column
    data['altitud'] = data['altitud'].apply(lambda alt: transform_altitud(alt))

    return data

    
def add_std_superficie(wine_data):
    """
    Computes the standard deviation of the 'superficie' column for each combination of 'id_finca', 'variedad',
    and 'modo' in the input dataframe and adds it as a new column 'std_superficie' to the dataframe. Also adds
    a new boolean column 'std_superficie_null' which indicates if the 'std_superficie' value is 0.

    Args:
        wine_data (pandas.DataFrame): The input dataframe to add the 'std_superficie' column to.

    Returns:
        pandas.DataFrame: The input dataframe with the 'std_superficie' and 'std_superficie_null' columns added.
    """

    # compute the standard deviation of 'superficie' for each combination of 'id_finca', 'variedad', and 'modo'
    std_sup = wine_data.dropna(subset=['superficie']).groupby(
        ['id_finca','variedad','modo']
    )['superficie'].std().fillna(0).rename("std_superficie")

    # merge the computed standard deviation with the input dataframe
    wine_data = wine_data.merge(
        std_sup,
        left_on=['id_finca','variedad','modo'],
        right_on=['id_finca','variedad','modo'],
        how='left'
    )

    # add a boolean column 'std_superficie_null' which indicates if the 'std_superficie' value is 0
    wine_data['std_superficie_null'] = wine_data['std_superficie'] == 0

    return wine_data


def preproces_wine_data(wine_data, fillna_alt=True, fillna_sup=True, output_path=None):
    """
    Preprocesses the input wine data by performing various data cleaning and imputation operations.

    Args:
        wine_data (pandas.DataFrame): The input wine data to preprocess.
        fillna_alt (bool): Whether to impute missing values in the 'altitud' column using group-wise mean imputation.
        fillna_sup (bool): Whether to impute missing values in the 'superficie' column using group-wise mean imputation.
        output_path (str): The file path to save the preprocessed data to. Defaults to None.

    Returns:
        pandas.DataFrame: The preprocessed wine data.
    """

    # normalize the columns of the input data
    wine_data = norm_columns(wine_data)

    # process the 'altitud' column to convert strings to floats and compute mean values
    wine_data = process_altitud(wine_data)

    # replace zeros with NaN values in the 'superficie' and 'altitud' columns
    wine_data = replace_zeros_with_na(wine_data, cols=['superficie', 'altitud'])

    # impute missing values in the 'altitud' column using group-wise mean imputation
    if fillna_alt:
        wine_data = fillna_by_group(wine_data, cols=['altitud'], group=['id_estacion'])

    # add a 'std_superficie' column to the dataframe which represents the standard deviation of the 'superficie'
    # column for each combination of 'id_finca', 'variedad', and 'modo'
    wine_data = add_std_superficie(wine_data)

    # impute missing values in the 'superficie' column using group-wise mean imputation
    if fillna_sup:
        # create a new column 'sup_is_nan' which is 0 if 'superficie' is not NaN, and 1 otherwise
        wine_data['sup_is_nan'] = wine_data['superficie'].apply(lambda x: 0 if x > 0 else x)
        wine_data['sup_is_nan'] = wine_data['sup_is_nan'].replace(np.nan, 1)

        # impute missing values in 'superficie' using group-wise mean imputation for various groups
        wine_data = fillna_by_group(wine_data, cols=['superficie'], group=['id_finca', 'variedad', 'modo'])
        wine_data = fillna_by_group(wine_data, cols=['superficie'], group=['id_zona', 'variedad', 'modo'])
        wine_data = fillna_by_group(wine_data, cols=['superficie'], group=['id_estacion', 'variedad', 'modo'])
        wine_data = fillna_by_group(wine_data, cols=['superficie'], group=['variedad', 'modo'])
        wine_data = fillna_by_group(wine_data, cols=['superficie'], group=['variedad'])

    # save the preprocessed data to a file if an output path is provided
    if output_path:
        wine_data.to_csv(output_path, index=False)

    return wine_data