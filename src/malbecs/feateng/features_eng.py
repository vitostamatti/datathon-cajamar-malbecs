#Denis functions 
import pandas as pd


def df_for_fe(data):
    transf = data[data.campaña != 21]
    return transf

def above_mean_col(data,col):
    df = df_for_fe(data)
    above_mean = pd.DataFrame(df.groupby(col)['produccion'].mean() > df.produccion.mean())
    above_mean = list(above_mean[above_mean['produccion']].index)
    data[f'{col}_above_mean'] = [1 if x in above_mean else 0 for x in data[col]]
    return data

def quantiles_col(data,col):
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
    temp_strings = ['gust','mslp','humidity','uvindex','visibility','windspeed',
                    'lluvia_daytime','lluvia_night','nieve_day','nieve_night']
    dic_temp = {}
    for string in temp_strings:
        columns = [col for col in df.columns if string in col]
        dic_temp[string] = columns
    return dic_temp


def eto_annual_sum(df,cols_list):
    dictionary = eto_dict_generator(df)
    dictionary = {key: dictionary[key] for key in cols_list}
    for name, data in dictionary.items():
            df[name] = df[data].sum(axis=1)
            df = df.drop(columns=data)
    return df

def eto_annual_mean(df,cols_list):
    dictionary = eto_dict_generator(df)
    dictionary = {key: dictionary[key] for key in cols_list}
    for name, data in dictionary.items():
            df[name] = df[data].mean(axis=1)
            df = df.drop(columns=data)
    return df