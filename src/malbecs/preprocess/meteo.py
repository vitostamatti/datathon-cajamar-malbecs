import pandas as pd
from malbecs.utils import fillna_by_group
import os


def load_meteo_data(path):
    meteo_data = pd.read_csv(path, sep='|')
    meteo_data['validTimeUtc'] = pd.to_datetime(meteo_data['validTimeUtc'])
    return meteo_data


def get_daily_mean_by_hour_range(meteo_data, col='temperature', from_hour=0, to_hour=24):
    # 12,16
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
    data.columns = [
        x + '_month_' + str(y) if y != '' else x for x, y in data.columns.to_flat_index()]
    return data


def get_temp_features(meteo_data):
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


def preproces_meteo_data(meteo_data, output_path=None):
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
