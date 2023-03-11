from malbecs.utils import fillna_by_group, replace_zeros_with_na
import pandas as pd
import numpy as np

from dataclasses import dataclass


@dataclass()
class WinePreprocessConfig:
    path:str
    fillna_sup:bool
    fillna_alt:bool
    output_path:str=None


def load_wine_dataset(path:str):
    return pd.read_csv(path, sep='|')


def norm_columns(wine):
    new_cols = [
        'campaña','id_finca','id_zona',
        'id_estacion','altitud','variedad',
        'modo','tipo','color','superficie','produccion'
    ]
    wine.columns = new_cols   
    return wine


def process_altitud(data):

    def transform_altitud(alt):
        if type(alt) is str:
            alt_list = alt.split("-")
            alt_list = list(map(float, alt_list))
            return np.mean(alt_list) 
        return alt
    
    data['altitud'] = data['altitud'].apply(lambda alt: transform_altitud(alt))

    return data



def preproces_wine_data(wine_data, fillna_alt=True, fillna_sup=True, output_path=None):
    # load data
    wine_data = norm_columns(wine_data)
    wine_data = process_altitud(wine_data)
    wine_data = replace_zeros_with_na(wine_data, cols=['superficie','altitud'])

    # fill nulls
    if fillna_alt:
        wine_data = fillna_by_group(wine_data, cols = ['altitud'], group = ['id_estacion'])

    if fillna_sup:
        wine_data = fillna_by_group(wine_data,cols=['superficie'], group=['id_finca','variedad','modo'])
        wine_data = fillna_by_group(wine_data,cols=['superficie'], group=['id_zona','variedad','modo'])
        wine_data = fillna_by_group(wine_data,cols=['superficie'], group=['id_estacion','variedad','modo'])
        wine_data = fillna_by_group(wine_data,cols=['superficie'], group=['variedad','modo'])
        wine_data = fillna_by_group(wine_data,cols=['superficie'], group=['variedad'])

    #save
    if output_path:
        wine_data.to_csv(output_path, index=False)

    return wine_data
