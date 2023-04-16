#Denis functions 
import pandas as pd


def df_for_fe(data):
    """
    Filter out rows in a pandas DataFrame where the 'campaña' column equals 21.

    Args:
        data (pandas.DataFrame): The DataFrame to filter.

    Returns:
        pandas.DataFrame: The filtered DataFrame with rows where 'campaña' equals 21 removed.
    """
    transf = data[data.campaña != 21]
    return transf

def above_mean_col(data,col):
    """
    Creates a new binary column in a pandas DataFrame indicating whether the values in a given column
    are above or below the mean production value of the rows in the DataFrame where the 'campaña' column
    does not equal 21.

    Args:
        data (pandas.DataFrame): The DataFrame to use for creating the new column.
        col (str): The name of the column to use for calculating the new binary column.

    Returns:
        pandas.DataFrame: The original DataFrame with a new binary column indicating whether the values in
        the specified column are above or below the mean production value of the rows where 'campaña' does not
        equal 21.
    """
    df = df_for_fe(data)
    above_mean = pd.DataFrame(df.groupby(col)['produccion'].mean() > df.produccion.mean())
    above_mean = list(above_mean[above_mean['produccion']].index)
    data[f'{col}_above_mean'] = [1 if x in above_mean else 0 for x in data[col]]
    return data

def quantiles_col(data,col):
    """
    Creates a new categorical column in a pandas DataFrame based on the quartiles of the production values
    for each unique value in a given column.

    Args:
        data (pandas.DataFrame): The DataFrame to use for creating the new column.
        col (str): The name of the column to use for grouping and calculating quartiles.

    Returns:
        pandas.DataFrame: The original DataFrame with a new categorical column indicating which quartile
        the production value for each row falls into, based on the quartiles of the production values for
        each unique value in the specified column.
    """
  
    df = df_for_fe(data)
    quantiles_df = pd.DataFrame(df.groupby(col)['produccion'].mean()).reset_index()
    Q1 = quantiles_df.produccion.quantile(0.25)
    Q2 = quantiles_df.produccion.quantile(0.5)
    Q3 = quantiles_df.produccion.quantile(0.75)
    
    quantiles_df[f'{col}_percentiles'] = [0 if x < Q1 else 1 if x < Q2 else 2 if x < Q3 else 3 for x in quantiles_df.produccion]

    data = data.merge(quantiles_df[[col,f'{col}_percentiles']], how='left', on = col) #Joineamos con el df transformado
    
    return data

#Eto annuals

def eto_dict_generator(df):
    """
    Generates a dictionary of column names for various weather variables based on a DataFrame containing
    weather data.

    Args:
        df (pandas.DataFrame): The DataFrame containing weather data.

    Returns:
        dict: A dictionary where each key is a string indicating a weather variable and each value is a list
        of column names in the DataFrame that contain data for that weather variable.
    """
    temp_strings = ['gust','mslp','humidity','uvindex','visibility','windspeed',
                    'lluvia_daytime','lluvia_night','nieve_day','nieve_night']
    dic_temp = {}
    for string in temp_strings:
        columns = [col for col in df.columns if string in col]
        dic_temp[string] = columns
    return dic_temp


def eto_annual_sum(df,cols_list):
    """
    Calculates the annual sum of various weather variables based on a DataFrame containing weather data.

    Args:
        df (pandas.DataFrame): The DataFrame containing weather data.
        cols_list (list): A list of strings indicating which weather variables to sum.

    Returns:
        pandas.DataFrame: The original DataFrame with additional columns representing the annual sum
        of the specified weather variables, and with the original columns for those variables dropped.
    """
    dictionary = eto_dict_generator(df)
    dictionary = {key: dictionary[key] for key in cols_list}
    for name, data in dictionary.items():
        df[name] = df[data].sum(axis=1)
        df = df.drop(columns=data)
    return df

def eto_annual_mean(df,cols_list):
    """
    Calculates the annual mean of various weather variables based on a DataFrame containing weather data.

    Args:
        df (pandas.DataFrame): The DataFrame containing weather data.
        cols_list (list): A list of strings indicating which weather variables to average.

    Returns:
        pandas.DataFrame: The original DataFrame with additional columns representing the annual mean
        of the specified weather variables, and with the original columns for those variables dropped.
    """
    dictionary = eto_dict_generator(df)
    dictionary = {key: dictionary[key] for key in cols_list}
    for name, data in dictionary.items():
            df[name] = df[data].mean(axis=1)
            df = df.drop(columns=data)
    return df