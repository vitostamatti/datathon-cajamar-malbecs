
# random-forest
from dataclasses import dataclass, asdict
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, StandardScaler
import pickle as pkl
from typing import List
import malbecs.modeling.transformers as mt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

seed = 42


def get_final_model():
    return get_model_fase_nacional()


def save_trained_model(model, path):
    with open(path, "wb") as file:
        pkl.dump(model, file)


def load_trained_model(path):
    with open(path,  "rb") as file:
        model = pkl.load(file)
    return model


def get_final_model_eto_features():
    #Modelo fase nacional
    precipuo = ['PrecipOver2StdMonth1',
                'PrecipOver2StdMonth2',
                'PrecipOver2StdMonth3',
                'PrecipOver2StdMonth4',
                'PrecipOver2StdMonth5',
                'PrecipOver2StdMonth6']
    
    precip_cols = ['SumTotalPrecipAmountLocalDayMonth1',
                    'SumTotalPrecipAmountLocalDayMonth2',
                    'SumTotalPrecipAmountLocalDayMonth3',
                    'SumTotalPrecipAmountLocalDayMonth4',
                    'SumTotalPrecipAmountLocalDayMonth5',
                    'SumTotalPrecipAmountLocalDayMonth6']
    
    snow_cols = ['SumTotalSnowAmountLocalDayMonth1', 'SumTotalSnowAmountLocalDayMonth2']

    return {"precipuo":precipuo,"precip_cols":precip_cols,"snow_cols":snow_cols}

def get_final_model_num_features():
    return  [
        'campaña',
        'superficie',
        'prod_shift_max',
        'prod_shift_change',
        'prod_shift_avg',
        'prod_he_var_zone_mean_hist_total',
        'prod_he_var_zone_std_hist_total',
        'prod_he_var_modo_zona_mean_shift1_total',
        "prod_he_var_modo_zona_change_total"
    ]


def get_preprocesing_fase_nacional():

    """Pipeline final para fase nacional"""

    model_num_cols = get_final_model_num_features()
    eto_feat = get_final_model_eto_features()

    precipuo = eto_feat['precipuo']
    precip_cols = eto_feat['precip_cols']
    snow_cols = eto_feat['snow_cols']
    
    return ColumnTransformer([

        ('Flag',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1), ['sup_is_nan']),
        ('Zona_encoder',mt.TargetEncoder(), ['id_zona']),
        ('Zona_encoder_2',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1), ['id_zona']),
        ('Variedad_encoder',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1), ['variedad']),
        ('Modo_encoder',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1), ['modo']),
        ('scaler',StandardScaler(), model_num_cols),
        ('Under_over_scaler',StandardScaler(),precipuo),
        ('Precip_PCA',make_pipeline(StandardScaler(),PCA(n_components=2, random_state=seed)),precip_cols),
        ("Snow",StandardScaler(),snow_cols),
        ],

        remainder='drop'
    )


def get_model_fase_nacional():
    """Modelo final para fase nacional"""

    prep = get_preprocesing_fase_nacional()

    model = RandomForestRegressor(
        random_state=seed,
        n_estimators=500,
        min_samples_leaf=4,
        n_jobs=-1,
        max_features=0.20,
        max_samples=0.8
    )

    m = make_pipeline(
        prep,
        model
    )

    return m