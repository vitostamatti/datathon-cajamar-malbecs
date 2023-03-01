from .common import fillna_by_group, replace_zeros_with_na, fillna_by_value
import pandas as pd


def load_wine_dataset(path:str):
    return pd.read_csv(path, sep='|')


def norm_columns(wine):
    new_cols = [
        'campana','id_finca','id_zona',
        'id_estacion','altitud','variedad',
        'modo','tipo','color','superficie','produccion'
    ]
    wine.columns = new_cols   
    return wine


def process_altitud(data):
    data['altitud'] = (
        data['altitud'].str.split("-",expand=True)
        .fillna(0).astype(float)
        .apply(
            lambda row: (row[0]+row[1])/2 if (row[0] > 0 and row[1] > 0) else row[0], 
            axis=1
        )
    )
    return data



def get_sup_tot_camp_finca(data):
    tot_sup_camp_finca = (
        data.groupby(['id_finca','campana'])
        .agg({"superficie":'sum'})
        .rename(columns={"superficie":"sup_tot_camp_finca"})
        .reset_index()
    )
    return data.merge(
        tot_sup_camp_finca,
        left_on=['id_finca','campana'],
        right_on=['id_finca','campana'],
        how='left'
    )   

def get_sup_tot_finca(data):
    tot_sup_finca = (
        data.groupby(['id_finca','campana'])
        .agg({"superficie":'sum'})
        .rename(columns={"superficie":"superficie_total"})
        .groupby('id_finca')
        .agg({"superficie_total":"mean"})
        .reset_index()
    )
    return data.merge(
        tot_sup_finca,
        left_on=['id_finca'],
        right_on=['id_finca'],
        how='left'
    )   


def get_n_var_finca_camp(data):
    n_variedad_finca = (
        data.groupby(['id_finca','campana'])
        .agg({"variedad":'nunique'})
        .rename(columns={"variedad":"n_var_camp_finca"})
    )
    return data.merge(
        n_variedad_finca,
        left_on=['id_finca','campana'],
        right_on=['id_finca','campana'],
        how='left'
    )


def preproces_wine_data(path):
    wine = load_wine_dataset(path)
    wine = norm_columns(wine)
    wine = process_altitud(wine)
    wine = replace_zeros_with_na(wine, cols=['superficie','altitud'])
    wine = fillna_by_group(wine,cols=['superficie'], group=['id_finca','variedad','modo'])
    wine = fillna_by_group(wine, cols = ['altitud'], group = ['id_estacion'])
    wine = get_sup_tot_camp_finca(wine)
    wine = get_sup_tot_finca(wine)
    wine = get_n_var_finca_camp(wine)
    wine = replace_zeros_with_na(
        wine, 
        cols=['sup_tot_camp_finca','superficie_total']
    )
    wine = fillna_by_value(
        wine, 
        cols=['superficie','sup_tot_camp_finca','superficie_total'],
        value=-1
    )
    return wine
