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

    std_sup = wine_data.dropna(subset=['superficie']).groupby(
        ['id_finca','variedad','modo']
    )['superficie'].std().fillna(0).rename("std_superficie")
    wine_data = wine_data.merge(
            std_sup,
            left_on=['id_finca','variedad','modo'],
            right_on=['id_finca','variedad','modo'],
            how='left'
        )
    wine_data['std_superficie_null'] = wine_data['std_superficie'] == 0
    return wine_data

def preproces_wine_data(wine_data, fillna_alt=True, fillna_sup=True, output_path=None):
    # load data
    wine_data = norm_columns(wine_data)
    wine_data = process_altitud(wine_data)
    wine_data = replace_zeros_with_na(
        wine_data, cols=['superficie', 'altitud'])

    # fill nulls
    if fillna_alt:
        wine_data = fillna_by_group(
            wine_data, cols=['altitud'], group=['id_estacion'])

    wine_data = add_std_superficie(wine_data)

    if fillna_sup:
        wine_data['sup_is_nan'] = wine_data['superficie'].apply(lambda x: 0 if x > 0 else x)
        wine_data['sup_is_nan'] = wine_data['sup_is_nan'].replace(np.nan, 1)
        
        
        wine_data = fillna_by_group(wine_data, cols=['superficie'], group=['id_finca', 'variedad', 'modo'])
        # wine_data['superficie'] = wine_data['superficie'].fillna(-1)
        wine_data = fillna_by_group(wine_data, cols=['superficie'], group=['id_zona', 'variedad', 'modo'])
        wine_data = fillna_by_group(wine_data, cols=['superficie'], group=['id_estacion', 'variedad', 'modo'])
        wine_data = fillna_by_group(wine_data, cols=['superficie'], group=['variedad', 'modo'])
        wine_data = fillna_by_group(wine_data, cols=['superficie'], group=['variedad'])

    # save
    if output_path:
        wine_data.to_csv(output_path, index=False)

    return wine_data
