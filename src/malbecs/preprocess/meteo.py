from typing import Optional
import pandas as pd
from malbecs.utils import fillna_by_group
import os


def load_meteo_data(path):
    """
    Loads meteo data from a file.

    Args:
        path (str): The path to the file.

    Returns:
        pandas.DataFrame: The meteo data.
    """
    meteo_data = pd.read_csv(path, sep='|')
    meteo_data['validTimeUtc'] = pd.to_datetime(meteo_data['validTimeUtc'])
    return meteo_data


def get_daily_mean_by_hour_range(meteo_data, col='temperature', from_hour=0, to_hour=24):
    """
    Computes the daily mean of a meteo variable within a specified hour range.

    Args:
        meteo_data (pandas.DataFrame): The meteo data.
        col (str): The name of the column to compute the daily mean of.
        from_hour (int): The starting hour of the range (inclusive).
        to_hour (int): The ending hour of the range (exclusive).

    Returns:
        pandas.DataFrame: The daily mean of the specified variable within the specified hour range, grouped by station and date.
    """
    meteo_data_filtered = meteo_data.loc[
        meteo_data['validTimeUtc'].dt.hour.between(from_hour, to_hour),
        ['ID_ESTACION', 'validTimeUtc', col]
    ]

    return meteo_data_filtered.groupby([
        'ID_ESTACION',
        pd.to_datetime(meteo_data_filtered['validTimeUtc'].dt.date)
    ]).agg({
        col: "mean"
    }).reset_index().rename(columns={
        col: f"{col}_avg_from_{from_hour}_to_{to_hour}",
        "validTimeUtc": "date"
    })


def flatten_pivot_columns(data):
    """
    Flattens the columns of a pivot table.

    Args:
        data (pandas.DataFrame): The pivot table.

    Returns:
        pandas.DataFrame: The flattened pivot table.
    """
    data.columns = [
        x + '_month_' + str(y) if y != '' else x for x, y in data.columns.to_flat_index()]
    return data



def get_temp_features(meteo_data):
    """
    Generates temperature-related features from the `meteo_data`.

    Args:
        meteo_data: a pandas DataFrame containing weather data. It must have the following columns:
        - 'validTimeUtc': the timestamp of the measurement;
        - 'ID_ESTACION': the ID of the weather station;
        - 'temperature': the temperature value.

    Returns:
        A pandas DataFrame with the following columns:
        - 'ID_ESTACION': the ID of the weather station;
        - 'year': the year of the measurement;
        - 'month_<n>_temp_avg_daytime': the average daytime temperature for month <n> (where <n> is a number between 1 and 6);
        - 'month_<n>_temp_max_daytime': the maximum daytime temperature for month <n>;
        - 'month_<n>_temp_avg_nighttime': the average nighttime temperature for month <n>;
        - 'month_<n>_temp_min_nighttime': the minimum nighttime temperature for month <n>.

    Note: missing values are filled using the 'fillna_by_group' function.
    """

    daytime_temp = get_daily_mean_by_hour_range(
        meteo_data, 'temperature', 12, 16)
    nighttime_temp = get_daily_mean_by_hour_range(
        meteo_data, 'temperature', 1, 5)

    temp_features = daytime_temp.merge(
        nighttime_temp,
        left_on=['ID_ESTACION', 'date'],
        right_on=['ID_ESTACION', 'date']
    )

    monthly_temp_features = temp_features.groupby([
        'ID_ESTACION',
        temp_features.date.dt.year,
        temp_features.date.dt.month
    ]).agg(
        temp_avg_daytime=('temperature_avg_from_12_to_16', 'mean'),
        temp_max_daytime=('temperature_avg_from_12_to_16', 'max'),
        temp_avg_nighttime=('temperature_avg_from_12_to_16', 'mean'),
        temp_min_nighttime=('temperature_avg_from_12_to_16', 'min'),
    )

    monthly_temp_features.index.names = ['ID_ESTACION', 'year', 'month']
    monthly_temp_features = monthly_temp_features.reset_index()
    months = [1, 2, 3, 4, 5, 6]
    monthly_temp_features = monthly_temp_features[monthly_temp_features['month'].isin(
        months)]

    monthly_temp_pivot = monthly_temp_features.pivot(
        index=['ID_ESTACION', "year"],
        columns=['month'],
        values=['temp_avg_daytime', 'temp_max_daytime',
                'temp_avg_nighttime', 'temp_min_nighttime']
    ).reset_index()

    monthly_temp_pivot = flatten_pivot_columns(monthly_temp_pivot)

    monthly_temp_pivot = fillna_by_group(
        monthly_temp_pivot,
        cols=monthly_temp_pivot.columns,
        group=['ID_ESTACION']
    )

    return monthly_temp_pivot


def get_light_fetaures(meteo_data):
    """
    Extracts features related to light hours from the given meteo_data.

    Args:
        meteo_data (pandas.DataFrame): The input meteorological data.

    Returns:
        pandas.DataFrame: A dataframe containing light hour features. The dataframe has columns "ID_ESTACION", "year",
        "MeanLightHours_month_1", "MeanLightHours_month_2", ..., "MeanLightHours_month_6".
    """

    meteo_data['LightHours'] = meteo_data['uvIndex'] > 0

    light_hours = meteo_data.groupby([
        'ID_ESTACION',
        pd.to_datetime(meteo_data.validTimeUtc.dt.date)]
    )['LightHours'].sum().reset_index()

    light_hours = light_hours.groupby([
        'ID_ESTACION',
        light_hours.validTimeUtc.dt.year,
        light_hours.validTimeUtc.dt.month
    ]).agg(
        MeanLightHours=('LightHours', 'mean')
    )

    light_hours.index.names = ['ID_ESTACION', 'year', 'month']

    light_hours = light_hours.reset_index()

    months = [1, 2, 3, 4, 5, 6]
    light_hours = light_hours[light_hours['month'].isin(months)]

    light_hours_pivot = light_hours.pivot(
        index=['ID_ESTACION', "year"],
        columns=['month'],
        values=['MeanLightHours']
    ).reset_index()

    light_hours_pivot = flatten_pivot_columns(light_hours_pivot)

    light_hours_pivot = fillna_by_group(
        light_hours_pivot,
        cols=light_hours_pivot.columns,
        group=['ID_ESTACION']
    )
    return light_hours_pivot


def preproces_meteo_data(meteo_data: pd.DataFrame, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Preprocesses the raw meteorological data by extracting features related to temperature and light.

    Args:
        meteo_data: A DataFrame containing the raw meteorological data.
        output_path: An optional string indicating the file path to save the preprocessed data to.

    Returns:
        A DataFrame containing the preprocessed meteorological data.
    """
    temp_features = get_temp_features(meteo_data)
    light_features = get_light_fetaures(meteo_data)
    meteo_pro = temp_features.merge(
        light_features,
        left_on=['ID_ESTACION', 'year'],
        right_on=['ID_ESTACION', 'year']
    )

    # save
    if output_path:
        meteo_pro.to_csv(output_path, index=False)

        dirname = os.path.dirname(output_path)

        with open(os.path.join(dirname, "meteo_features.txt"), "w") as f:
            f.write("\n".join(meteo_pro.columns.to_list()[2:]))

    return meteo_pro
