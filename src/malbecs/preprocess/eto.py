from typing import List
import pandas as pd
import os
from malbecs.utils import fillna_by_group, fillna_by_value

avg_cols = [
    'DewpointLocalDayAvg',
    'EvapotranspirationLocalDayAvg',
    'FeelsLikeLocalDayAvg',
    'GlobalHorizontalIrradianceLocalDayAvg',
    'GustLocalDayAvg',
    'MSLPLocalDayAvg',
    'PrecipAmountLocalDayAvg',
    'RelativeHumidityLocalDayAvg',
    'SnowAmountLocalDayAvg',
    'TemperatureLocalDayAvg',
    'UVIndexLocalDayAvg',
    'VisibilityLocalDayAvg',
    'WindSpeedLocalDayAvg'
]

max_cols = [
    'DewpointLocalDayMax',
    'EvapotranspirationLocalDayMax',
    'FeelsLikeLocalDayMax',
    'GlobalHorizontalIrradianceLocalDayMax',
    'GustLocalDayMax',
    'MSLPLocalDayMax',
    'PrecipAmountLocalDayMax',
    'RelativeHumidityLocalDayMax',
    'SnowAmountLocalDayMax',
    'TemperatureLocalDayMax',
    'UVIndexLocalDayMax',
    'VisibilityLocalDayMax',
    'WindSpeedLocalDayMax',
]

min_cols = [
    'DewpointLocalDayMin',
    'FeelsLikeLocalDayMin',
    'GustLocalDayMin',
    'MSLPLocalDayMin',
    'RelativeHumidityLocalDayMin',
    'TemperatureLocalDayMin',
    'VisibilityLocalDayMin',
    'WindSpeedLocalDayMin'
]

cols_sum = [
    'PrecipAmountLocalDayAvg',
    'SnowAmountLocalDayAvg'
]

cols_mean = avg_cols + max_cols + min_cols + \
    ['TemperatureLocalAfternoonAvg', 'TemperatureLocalOvernightAvg']


def load_eto_dataset(path: str) -> pd.DataFrame:
    """
    Loads a dataset from a file and returns it as a pandas DataFrame.

    Args:
        path (str): The path to the dataset file to load.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    # Load the dataset using pandas and set the separator and header parameters.
    eto = pd.read_csv(path, sep='|', header=0)

    # Call the 'parse_date' function to convert the date column to a datetime object.
    eto = parse_date(eto)

    # Return the loaded dataset as a pandas DataFrame.
    return eto

def parse_date(eto: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the date column in a pandas DataFrame to a datetime object.

    Args:
        eto (pd.DataFrame): The dataset containing the date column.

    Returns:
        pd.DataFrame: The dataset with the date column converted to a datetime object.
    """
    # Convert the date column to a datetime object using the 'pd.to_datetime' method.
    eto['date'] = pd.to_datetime(
        eto['date'].astype(str).apply(
            lambda x: "{}/{}/{}".format(x[4:6], x[6:], x[0:4])
        )
    )
    return eto


def add_year_and_month(eto: pd.DataFrame):
    """
    Adds the 'year' and 'month' columns to a dataframe with a 'date' column.
    
    Args:
        eto: A pandas DataFrame containing a 'date' column.
    
    Returns:
        A pandas DataFrame with added 'year' and 'month' columns.
    """
    eto['month'] = eto.date.dt.month
    eto['year'] = eto.date.dt.year.astype("int32")
    return eto



def get_totals_by_daytime_and_nighttime(eto: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Calculates the totals for a list of columns for daytime and nighttime readings in a dataframe.
    
    Args:
        eto: A pandas DataFrame containing columns for daytime and nighttime readings.
        cols: A list of column names in the dataframe.
    
    Returns:
        A pandas DataFrame with added columns containing the total values for daytime and nighttime readings.
    """
    new_cols = [f"Total{c[:-3]}" for c in cols]
    eto[new_cols] = (eto[cols]*12)
    return eto[new_cols]


def get_totals_by_day(eto, cols):
    """
    Get the total values of the given columns in eto dataframe by day.

    Args:
        eto (pd.DataFrame): The dataframe containing the data to be processed.
        cols (list of str): The list of column names for which the total is to be calculated.

    Returns:
        pd.DataFrame: A new dataframe containing the total values for the given columns, multiplied by 24.

    """
    new_cols = [f"Total{c[:-3]}" for c in cols]
    eto[new_cols] = (eto[cols]*24)
    return eto[new_cols]


def get_data_for_sum_group(eto, cols_sum, cols_ids):
    """
    Return a pandas dataframe with the sum of the specified columns by day, along with the columns used to identify each row.

    Args:
        eto (pandas.DataFrame): A pandas dataframe with the ETo data and columns to be summed by day.
        cols_sum (list): A list of strings with the names of the columns to be summed by day.
        cols_ids (list): A list of strings with the names of the columns to identify each row.

    Returns:
        pandas.DataFrame: A pandas dataframe with the sum of the specified columns by day, along with the columns used to identify each row.
    """
    eto_sum = pd.concat([
        eto[cols_ids],
        get_totals_by_day(eto, cols_sum)
    ], axis=1)
    return eto_sum


def get_data_for_mean_group(eto: pd.DataFrame, cols_mean: List[str], cols_ids: List[str]) -> pd.DataFrame:
    """
    Returns a new DataFrame with columns for the specified `cols_mean` that have been aggregated by mean for each group
    of columns specified in `cols_ids`.

    Args:
        eto (pd.DataFrame): The DataFrame containing the ETO data.
        cols_mean (List[str]): A list of column names to aggregate by mean.
        cols_ids (List[str]): A list of column names that should be used to group by.

    Returns:
        pd.DataFrame: A new DataFrame containing the mean aggregated data for each group of `cols_ids`.
    """
    eto_mean = eto[cols_ids+cols_mean]
    return eto_mean


def get_monthly_data(eto: pd.DataFrame, cols_mean: list, cols_sum: list) -> pd.DataFrame:
    """Returns monthly data by calculating mean and sum of specified columns.

    Args:
        eto (pd.DataFrame): DataFrame with data to be grouped by year and month.
        cols_mean (list): List of column names to calculate mean.
        cols_sum (list): List of column names to calculate sum.

    Returns:
        pd.DataFrame: DataFrame with calculated mean and sum for specified columns grouped by year and month.
    """

    cols_ids = ['ID_ESTACION', 'year', 'month']

    eto_mean = get_data_for_mean_group(eto, cols_mean, cols_ids)
    eto_sum = get_data_for_sum_group(eto, cols_sum, cols_ids)

    grouped_sum = eto_sum.groupby(cols_ids).sum()
    grouped_sum.columns = [f"Sum{c}" for c in grouped_sum.columns]

    grouped_mean = eto_mean.groupby(cols_ids).mean()
    grouped_mean.columns = [f"Mean{c}" for c in grouped_mean.columns]

    eto_month = pd.concat([grouped_mean, grouped_sum], axis=1).reset_index()

    return eto_month


def filter_relevant_months(eto_month: pd.DataFrame, months: list=[1, 2, 3, 4, 5, 6]) -> pd.DataFrame:
    """Filters relevant months from eto_month DataFrame.

    Args:
        eto_month (pd.DataFrame): DataFrame with monthly data.
        months (list): List of months to filter (default is [1, 2, 3, 4, 5, 6]).

    Returns:
        pd.DataFrame: DataFrame with relevant months.
    """
    return eto_month[eto_month['month'].isin(months)]


def flatten_pivot_columns(eto_pivot: pd.DataFrame) -> pd.DataFrame:
    """Flattens the columns of a pivot table DataFrame.

    Args:
        eto_pivot (pd.DataFrame): Pivot table DataFrame to flatten columns.

    Returns:
        pd.DataFrame: Flattened DataFrame with new column names.
    """

    eto_pivot.columns = [
        x + 'Month' + str(y) if y != '' else x for x, y in eto_pivot.columns.to_flat_index()]
    return eto_pivot



def pivot_monthly_data(eto_month: pd.DataFrame) -> pd.DataFrame:
    """Pivots a DataFrame with monthly ETO data.

    The function pivots the input DataFrame by year, ID_ESTACION, and month to create a new DataFrame where each row
    corresponds to a combination of year, station ID, and month, and each column corresponds to a specific ETO value.

    Args:
        eto_month (pd.DataFrame): A pandas DataFrame containing monthly ETO data with the following columns:
            - year (int): The year the data corresponds to.
            - month (int): The month the data corresponds to.
            - ID_ESTACION (int): The ID of the station the data corresponds to.
            - Other columns: Columns with the monthly ETO values.

    Returns:
        pd.DataFrame: A pivoted DataFrame with the following columns:
            - year (int): The year the data corresponds to.
            - ID_ESTACION (int): The ID of the station the data corresponds to.
            - Columns with the format ETO{MonthX}: Columns containing the ETO values for each month X (1-12).
    """
    index = ['year', 'ID_ESTACION']
    columns = ['month']
    values = eto_month.drop(columns=index+columns).columns.tolist()
    eto_pivot = eto_month.pivot(
        index=index, columns=columns, values=values).reset_index()
    eto_pivot = flatten_pivot_columns(eto_pivot)
    return eto_pivot


def get_mean_and_std_by_month(eto_data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Calculates the mean and standard deviation of a column in a DataFrame by month and station.

    The function calculates the mean and standard deviation of a specific column in a DataFrame by grouping the data by
    month and station.

    Args:
        eto_data (pd.DataFrame): A pandas DataFrame containing ETO data with the following columns:
            - ID_ESTACION (int): The ID of the station the data corresponds to.
            - date (pd.Timestamp): The date the data corresponds to.
            - Other columns: Columns with the ETO values.

        column (str): The name of the column for which to calculate the mean and standard deviation.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - ID_ESTACION (int): The ID of the station the data corresponds to.
            - month (int): The month the data corresponds to.
            - mean (float): The mean value of the column for the given station and month.
            - std (float): The standard deviation of the column for the given station and month.
    """
    return (
        eto_data
        [['ID_ESTACION', 'date', column]]
        .groupby(
            ["ID_ESTACION", eto_data.date.dt.month]
        )
        .agg(
            mean=(column, 'mean'),
            std=(column, 'std'),
        )
        .reset_index()
        .rename(columns={
            "date": "month",
        })
    )



def get_days_over_and_under_mean(
        eto_data: pd.DataFrame, 
        column: str, 
        out_column_name: str, 
        over: bool = True, 
        under: bool = True
        ) -> pd.DataFrame:
    """
    Get number of days over and/or under 1 and 2 standard deviations from the monthly mean for a given column.

    Args:
        eto_data: pandas DataFrame containing data for the column of interest (e.g. ETo)
        column: str, name of the column of interest
        out_column_name: str, prefix for the output column names
        over: bool, whether to include count of days over 1 and 2 standard deviations (default: True)
        under: bool, whether to include count of days under 1 and 2 standard deviations (default: True)

    Returns:
        pandas DataFrame with the following columns:
            - ID_ESTACION: station ID
            - year: year of the data
            - month: month of the data
            - Under1Std: number of days under 1 standard deviation from the monthly mean
            - Over1Std: number of days over 1 standard deviation from the monthly mean
            - Under2Std: number of days under 2 standard deviations from the monthly mean
            - Over2Std: number of days over 2 standard deviations from the monthly mean
    """

    def rename(df):
        return df.rename(columns={c: f"{out_column_name}{c}" for c in df.columns})

    def select(df):
        overcols = [c for c in df.columns if 'Over' in c] if over else []
        undercols = [c for c in df.columns if 'Under' in c] if under else []
        return df[overcols + undercols]

    month_data = get_mean_and_std_by_month(eto_data, column)

    return (eto_data
            [['ID_ESTACION', 'date', column]]
            .assign(
                month=eto_data['date'].dt.month,
                year=eto_data['date'].dt.year,
            )
            .merge(
                month_data,
                left_on=['ID_ESTACION', 'month'],
                right_on=['ID_ESTACION', 'month']
            )
            .assign(
                Diff=lambda df: df[column] - df['mean']
            )
            .assign(
                Over1Std=lambda df: (df['Diff'] > df["std"]).astype(int),
                Over2Std=lambda df: (df['Diff'] > df["std"]*2).astype(int),
                Under1Std=lambda df: (df['Diff'] < -df["std"]).astype(int),
                Under2Std=lambda df: (df['Diff'] < -df["std"]*2).astype(int),
            )
            .groupby(["ID_ESTACION", "year", 'month'])
            [['Under1Std', 'Over1Std', 'Over2Std', 'Under2Std']]
            .sum()
            .pipe(select)
            .pipe(rename)
            )


def get_days_over_and_under_features(eto_data):
    """
    This function extracts and aggregates features from the ETO data using the get_days_over_and_under_mean function, 
    and then applies some additional data processing steps.

    Args:
        eto_data (pandas.DataFrame): The ETO data containing weather information.

    Returns:
        pandas.DataFrame extracted and aggregated features from the ETO data.Each row represents a unique combination of year and month, 
            and each column represents a different feature. The features are as follows.
            - TempOverMean: number of days in the month where TemperatureLocalDayAvg was above the monthly mean
            - TempUnderMean: number of days in the month where TemperatureLocalDayAvg was below the monthly mean
            - PrecipUnderMean: number of days in the month where PrecipAmountLocalDayAvg was below the monthly mean
            - SnowUnderMean: number of days in the month where SnowAmountLocalDayAvg was below the monthly mean
            - WindUnderMean: number of days in the month where WindSpeedLocalDayAvg was below the monthly mean
            - GustUnderMean: number of days in the month where GustLocalDayAvg was below the monthly mean
            - PrecipOverZero: number of days in the month where PrecipAmountLocalDayAvg was greater than zero
            - SnowOverZero: number of days in the month where SnowAmountLocalDayAvg was greater than zero
            - WindOverZero: number of days in the month where WindSpeedLocalDayAvg was greater than zero
            - GustOverZero: number of days in the month where GustLocalDayAvg was greater than zero
    """
    features = get_days_over_and_under_mean(
        eto_data,
        column="TemperatureLocalDayAvg",
        out_column_name="Temp",
        over=True,
        under=True
    ).join(
        get_days_over_and_under_mean(
            eto_data,
            column="PrecipAmountLocalDayAvg",
            out_column_name="Precip",
            under=False
        )
    ).join(
        get_days_over_and_under_mean(
            eto_data,
            column="SnowAmountLocalDayAvg",
            out_column_name="Snow",
            under=False
        )
    ).join(
        get_days_over_and_under_mean(
            eto_data,
            column="WindSpeedLocalDayAvg",
            out_column_name="Wind",
            under=False
        )
    ).join(
        get_days_over_and_under_mean(
            eto_data,
            column="GustLocalDayAvg",
            out_column_name="Gust",
            under=False
        )
    )
    features = filter_relevant_months(features.reset_index())
    features = pivot_monthly_data(features)
    features = fillna_by_value(features, cols=features.columns, value=-1)
    return features


def preprocess_eto_dataset(eto_data, cols_mean, cols_sum, output_path=None):
    """
    Preprocess the ETO dataset by performing the following steps:
        1. Extract features related to the number of days over/under a mean value for temperature, precipitation, snow, wind, and gusts using the function `get_days_over_and_under_features`.
        2. Add columns for the year and month of each data point using the function `add_year_and_month`.
        3. Get monthly aggregated data for the columns specified in `cols_mean` and `cols_sum` using the function `get_monthly_data`.
        4. Filter out data points from irrelevant months using the function `filter_relevant_months`.
        5. Fill in missing values for gust, snow, and precipitation columns with 0 using the function `fillna_by_value`.
        6. Fill in missing values for all other columns using the function `fillna_by_group`, grouping by station ID and month.
        7. Pivot the monthly data using the function `pivot_monthly_data`.
        8. Fill in missing values for the pivoted data using `fillna_by_group`, grouping by station ID.
        9. Merge the over/under features from step 1 with the pivoted data, using station ID and year as keys.
        10. If specified, save the final dataset as a CSV file at the `output_path` location.

    Args:
        eto_data (pd.DataFrame): A pandas DataFrame containing ETO data for one or more stations.
        cols_mean (list): A list of columns to calculate the mean for in the monthly aggregated data.
        cols_sum (list): A list of columns to calculate the sum for in the monthly aggregated data.
        output_path (str): The file path where the final preprocessed dataset should be saved. If None, the dataset will not be saved.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the preprocessed ETO data.
    """
    over_under_features = get_days_over_and_under_features(eto_data)
    eto_data = add_year_and_month(eto_data)
    df_month = get_monthly_data(
        eto_data,
        cols_mean,
        cols_sum
    )

    df_month = filter_relevant_months(df_month)
    gust_cols = df_month.filter(like="Gust").columns.to_list()
    snow_cols = df_month.filter(like="Snow").columns.to_list()
    precip_cols = df_month.filter(like="Precip").columns.to_list()

    df_month = fillna_by_value(
        df_month, cols=gust_cols+snow_cols+precip_cols, value=0)

    df_month = fillna_by_group(
        df_month,
        cols=df_month.columns,
        group=['ID_ESTACION', 'month']
    )

    df_pivot = pivot_monthly_data(df_month)

    df_pivot = fillna_by_group(
        df_pivot,
        cols=df_pivot.columns,
        group=['ID_ESTACION']
    )

    df_pivot = df_pivot.merge(
        over_under_features,
        left_on=['ID_ESTACION', 'year'],
        right_on=['ID_ESTACION', 'year']
    )

    if output_path:
        df_pivot.to_csv(output_path, index=False)

    return df_pivot
