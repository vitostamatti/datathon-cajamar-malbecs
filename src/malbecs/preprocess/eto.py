import pandas as pd

from malbecs.utils import fillna_by_group, fillna_by_value

from dataclasses import dataclass

@dataclass()
class ETOPreprocessConfig:
    path:str
    cols_sum:list
    cols_mean:list
    output_path:str=None


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
    
    cols_ids = ['ID_ESTACION','year','month']

    eto_mean = get_data_for_mean_group(eto, cols_mean, cols_ids)
    eto_sum = get_data_for_sum_group(eto, cols_sum, cols_ids)

    grouped_sum = eto_sum.groupby(cols_ids).sum()
    grouped_sum.columns = [f"Sum{c}" for c in grouped_sum.columns]
    
    grouped_mean = eto_mean.groupby(cols_ids).mean()
    grouped_mean.columns = [f"Mean{c}" for c in grouped_mean.columns]

    eto_month = pd.concat([grouped_mean,grouped_sum],axis=1).reset_index()
    
    return eto_month


def filter_relevant_months(eto_month, months=[1,2,3,4,5,6]):
    return eto_month[eto_month['month'].isin(months)]



def flatten_pivot_columns(eto_pivot):
    eto_pivot.columns = [x +'Month'+ str(y) if y != '' else x for x,y in eto_pivot.columns.to_flat_index()]
    return eto_pivot


def pivot_monthly_data(eto_month):
    index=['year','ID_ESTACION']
    columns=['month']
    values = eto_month.drop(columns=index+columns).columns.tolist()
    eto_pivot = eto_month.pivot(index=index, columns=columns, values=values).reset_index()
    eto_pivot = flatten_pivot_columns(eto_pivot)
    return eto_pivot



def preprocess_eto_dataset(eto_data, cols_mean, cols_sum, output_path=None):
    
    eto_data = add_year_and_month(eto_data)

    df_month = get_monthly_data(
        eto_data, cols_mean, 
        cols_sum
    )

    df_month = filter_relevant_months(df_month)

    gust_cols = df_month.filter(like="Gust").columns.to_list()
    snow_cols = df_month.filter(like="Snow").columns.to_list()
    precip_cols  = df_month.filter(like="Precip").columns.to_list()
    df_month = fillna_by_value(df_month, cols=gust_cols+snow_cols+precip_cols, value=0)

    df_month = fillna_by_group(
        df_month,
        cols=df_month.columns,
        group=['ID_ESTACION','month']
    )

    df_pivot = pivot_monthly_data(df_month)

    df_pivot = fillna_by_group(
        df_pivot, 
        cols=df_pivot.columns, 
        group=['ID_ESTACION']
    )

    if output_path:
        df_pivot.to_csv(output_path, index=False)

    return df_pivot



