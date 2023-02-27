import dataclasses
import pandas as pd
import numpy as np
import os

from dataclasses import dataclass

@dataclass()
class ETOPreprocessConfig:
    cols_ids:list
    cols_sum:list
    cols_mean:list
    path:str


def load_eto_dataset(path:str):
    # 'DATOS_ETO.txt'
    eto = pd.read_csv(path, sep= '|', header=0 )
    eto = parse_date(eto)
    return eto


def parse_date(eto:pd.DataFrame):
    eto['date'] =  pd.to_datetime(
        eto['date'].astype(str).apply(
        lambda x: "{}/{}/{}".format(x[4:6],x[6:], x[0:4])
        )
    )
    return eto


def add_year_and_month(eto:pd.DataFrame):
    eto['month'] =  eto.date.dt.month
    eto['year'] = eto.date.dt.year.astype("int32")
    return eto


def get_totals_by_daytime_and_nighttime(eto, cols):
    new_cols = [f"Total{c[:-3]}" for c in cols]
    eto[new_cols] = (eto[cols]*12)
    return eto[new_cols]



def get_totals_by_daytime_and_nighttime(eto, cols, rename=True):
    if rename:
        new_cols = [f"Total{c[:-3]}" for c in cols]
    else:
        new_cols = cols

    eto[new_cols] = (eto[cols]*12)    
    return eto[new_cols]


def get_data_for_sum_group(eto, cols_sum, cols_ids):
    eto_sum = pd.concat([
        eto[cols_ids],
        get_totals_by_daytime_and_nighttime(eto, cols_sum)
    ], axis=1)
    return eto_sum


def get_data_for_mean_group(eto, cols_mean, cols_ids):
    eto_mean = eto[cols_ids+cols_mean]
    return eto_mean


def get_monthly_datat(eto, cols_mean, cols_sum, cols_ids):
    
    eto_mean = get_data_for_mean_group(eto, cols_mean, cols_ids)
    eto_sum = get_data_for_sum_group(eto, cols_sum, cols_ids)

    grouped_sum = eto_sum.groupby(cols_ids).sum()
    grouped_mean = eto_mean.groupby(cols_ids).mean()
    
    eto_month = pd.concat([grouped_mean,grouped_sum],axis=1).reset_index()
    
    return eto_month


def filter_relevant_months(eto_month, months=[1,2,3,4,5,6]):
    return eto_month[eto_month['month'].isin(months)]


def fillna_by_group(eto_month, cols=['MSLPLocalDayAvg'], group=['ID_ESTACION','month']):
    for col in cols:
        eto_month[col] = eto_month.groupby(group)[col].transform(lambda x: x.fillna(x.mean()))
    return eto_month


def fillna_by_value(eto_month, cols=['GustLocalDayAvg'] ):
    eto_month[cols] = eto_month[cols].fillna(0)
    return eto_month


def flatten_pivot_columns(eto_pivot):
    eto_pivot.columns = [x +'Month'+ str(y) if y != '' else x for x,y in eto_pivot.columns.to_flat_index()]
    return eto_pivot


def pivot_monthly_data(eto_month, index=['year','ID_ESTACION'], columns=['month']):
    values = eto_month.drop(columns=index+columns).columns.tolist()
    eto_pivot = eto_month.pivot(index=index, columns=columns, values=values).reset_index()
    eto_pivot = flatten_pivot_columns(eto_pivot)
    return eto_pivot




def preprocess_eto_dataset(path):
    
    eto_prepro_config = ETOPreprocessConfig(
        cols_ids = ['ID_ESTACION','year','month'],

        cols_sum = [
            'PrecipAmountLocalDaytimeAvg','PrecipAmountLocalNighttimeAvg',
            'SnowAmountLocalDaytimeAvg','SnowAmountLocalNighttimeAvg'
        ],

        cols_mean = [
            'GustLocalDayAvg', 'MSLPLocalDayAvg', 'RelativeHumidityLocalDayAvg',
            'UVIndexLocalDayAvg', 'VisibilityLocalDayAvg', 'WindSpeedLocalDayAvg',
            'TemperatureLocalAfternoonAvg','TemperatureLocalOvernightAvg'
        ],
        path = path 
    )

    eto = load_eto_dataset(eto_prepro_config.path)
    
    eto = add_year_and_month(eto)

    df_month = get_monthly_datat(
        eto, eto_prepro_config.cols_mean, 
        eto_prepro_config.cols_sum, 
        eto_prepro_config.cols_ids)

    df_month = filter_relevant_months(df_month)

    df_month = fillna_by_group(df_month)

    df_month = fillna_by_value(df_month)

    df_pivot = pivot_monthly_data(df_month)

    df_pivot = fillna_by_group(df_pivot, cols=df_pivot.columns, group=['ID_ESTACION'])

    return df_pivot