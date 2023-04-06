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


def load_eto_dataset(path: str):
    # 'DATOS_ETO.txt'
    eto = pd.read_csv(path, sep='|', header=0)
    eto = parse_date(eto)
    return eto


def parse_date(eto: pd.DataFrame):
    eto['date'] = pd.to_datetime(
        eto['date'].astype(str).apply(
            lambda x: "{}/{}/{}".format(x[4:6], x[6:], x[0:4])
        )
    )
    return eto


def add_year_and_month(eto: pd.DataFrame):
    eto['month'] = eto.date.dt.month
    eto['year'] = eto.date.dt.year.astype("int32")
    return eto


def get_totals_by_daytime_and_nighttime(eto, cols):
    new_cols = [f"Total{c[:-3]}" for c in cols]
    eto[new_cols] = (eto[cols]*12)
    return eto[new_cols]


def get_totals_by_day(eto, cols):
    new_cols = [f"Total{c[:-3]}" for c in cols]
    eto[new_cols] = (eto[cols]*24)
    return eto[new_cols]


def get_data_for_sum_group(eto, cols_sum, cols_ids):
    eto_sum = pd.concat([
        eto[cols_ids],
        get_totals_by_day(eto, cols_sum)
    ], axis=1)
    return eto_sum


def get_data_for_mean_group(eto, cols_mean, cols_ids):
    eto_mean = eto[cols_ids+cols_mean]
    return eto_mean


def get_monthly_data(eto, cols_mean, cols_sum):

    cols_ids = ['ID_ESTACION', 'year', 'month']

    eto_mean = get_data_for_mean_group(eto, cols_mean, cols_ids)
    eto_sum = get_data_for_sum_group(eto, cols_sum, cols_ids)

    grouped_sum = eto_sum.groupby(cols_ids).sum()
    grouped_sum.columns = [f"Sum{c}" for c in grouped_sum.columns]

    grouped_mean = eto_mean.groupby(cols_ids).mean()
    grouped_mean.columns = [f"Mean{c}" for c in grouped_mean.columns]

    eto_month = pd.concat([grouped_mean, grouped_sum], axis=1).reset_index()

    return eto_month


def filter_relevant_months(eto_month, months=[1, 2, 3, 4, 5, 6]):
    return eto_month[eto_month['month'].isin(months)]


def flatten_pivot_columns(eto_pivot):
    eto_pivot.columns = [
        x + 'Month' + str(y) if y != '' else x for x, y in eto_pivot.columns.to_flat_index()]
    return eto_pivot


def pivot_monthly_data(eto_month):
    index = ['year', 'ID_ESTACION']
    columns = ['month']
    values = eto_month.drop(columns=index+columns).columns.tolist()
    eto_pivot = eto_month.pivot(
        index=index, columns=columns, values=values).reset_index()
    eto_pivot = flatten_pivot_columns(eto_pivot)
    return eto_pivot



def get_mean_and_std_by_month(eto_data, column):
    return (
        eto_data
        [['ID_ESTACION','date',column]]
        .groupby(
        ["ID_ESTACION",eto_data.date.dt.month]
        )
        .agg(
            mean = (column,'mean'),
            std = (column,'std'),
        )
        .reset_index()
        .rename(columns={
            "date":"month",
        })
    )


def get_days_over_and_under_mean(eto_data, column, out_column_name, over=True, under=True):

    def rename(df):
        return df.rename(columns={c:f"{out_column_name}{c}" for c in df.columns})
    
    def select(df):
        overcols = [c for c in df.columns if 'Over' in c] if over else []
        undercols = [c for c in df.columns if 'Under' in c] if under else []
        return df[['Diff']+overcols + undercols]

    month_data = get_mean_and_std_by_month(eto_data, column)
    return (eto_data
     [['ID_ESTACION','date',column]]
        .assign(
            month = eto_data['date'].dt.month,
            year = eto_data['date'].dt.year,
        )
        .merge(
            month_data,
            left_on=['ID_ESTACION','month'],
            right_on=['ID_ESTACION','month']
        )
        .assign(
            Diff = lambda df: df[column] - df['mean']
        )
        .assign(
            Over1Std = lambda df: (df['Diff'] > df["std"]).astype(int),
            Over2Std = lambda df: (df['Diff'] > df["std"]*2).astype(int),
            Under1Std = lambda df: (df['Diff']< -df["std"]).astype(int),
            Under2Std = lambda df: (df['Diff']< -df["std"]*2).astype(int),
        )
        .groupby(["ID_ESTACION","year",'month'])
        [["Diff",'Under1Std','Over1Std','Over2Std','Under2Std']]
        .sum()
        .pipe(select)
        .pipe(rename)
    )



def get_days_over_and_under_features(eto_data):
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
    feat_cols = features.columns.to_list()
    features = filter_relevant_months(features.reset_index())
    features = pivot_monthly_data(features)
    features = fillna_by_value(features, cols = features.columns, value=-1) 
    return features


def preprocess_eto_dataset(eto_data, cols_mean, cols_sum, output_path=None):

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

    df_pivot=df_pivot.merge(
        over_under_features,
        left_on=['ID_ESTACION', 'year'],
        right_on=['ID_ESTACION', 'year']
    )


    if output_path:
        df_pivot.to_csv(output_path, index=False)

    return df_pivot
