import pandas as pd
from .common import fillna_by_group

def load_meteo_data(path):
    meteo = pd.read_csv(path, sep='|')
    meteo['validTimeUtc'] = pd.to_datetime(meteo['validTimeUtc'])
    return meteo

def get_daily_mean_by_hour_range(meteo, col='temperature', from_hour=0, to_hour=24):
    # 12,16
    meteo_filtered = meteo.loc[
        meteo['validTimeUtc'].dt.hour.between(from_hour,to_hour),
        ['ID_ESTACION','validTimeUtc',col]
    ]

    return meteo_filtered.groupby([
        'ID_ESTACION',
        pd.to_datetime(meteo_filtered['validTimeUtc'].dt.date)
    ]).agg({
        col:"mean"
    }).reset_index().rename(columns={
        col:f"{col}_avg_from_{from_hour}_to_{to_hour}",
        "validTimeUtc":"date"
    })


def flatten_pivot_columns(data):
    data.columns = [x +'_month_'+ str(y) if y != '' else x for x,y in data.columns.to_flat_index()]
    return data


def preproces_meteo_data(path):
    meteo = load_meteo_data(path)
    daytime_temp = get_daily_mean_by_hour_range(meteo, 'temperature', 12, 16)
    nighttime_temp = get_daily_mean_by_hour_range(meteo, 'temperature', 1, 5)
    temp_features = daytime_temp.merge(
        nighttime_temp, 
        left_on=['ID_ESTACION','date'], 
        right_on=['ID_ESTACION','date']
    )
    temp_features.groupby([
        'ID_ESTACION',
        temp_features.date.dt.year,
        temp_features.date.dt.month
    ]).agg({
        "temperature_avg_from_12_to_16":"max",
        
        "temperature_avg_from_1_to_5":"min"
    })