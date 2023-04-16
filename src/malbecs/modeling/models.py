
# random-forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pickle as pkl
import malbecs.modeling.transformers as mt
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

seed = 42


def get_final_model():
    """
    Returns the final model for the challenge.

    Returns:
        ipeline: The final model pipeline returned by get_model_fase_nacional().
    """
    return get_model_fase_nacional()


def save_trained_model(model, path):
    """Save a trained model to disk using pickle serialization.

    Args:
        model (object): The trained model object to be saved.
        path (str): The path where the model will be saved.

    Returns:
        None
    """
    with open(path, "wb") as file:
        pkl.dump(model, file)


def load_trained_model(path):
    """Load a trained model from disk using pickle deserialization.

    Args:
        path (str): The path where the trained model is located.

    Returns:
        object: The loaded model object.
    """
    with open(path,  "rb") as file:
        model = pkl.load(file)
    return model


def get_final_model_eto_features():
    """
    Returns a dictionary with feature names for the final ETO model.

    Returns:
        dict: A dictionary with the following keys:
            - precipuo: a list of strings representing the names of the features indicating whether there was precipitation 
                        over two standard deviations from the monthly mean for each of the first six months of the year.
            - precip_cols: a list of strings representing the names of the features containing the total precipitation amount 
                        for each day of the first six months of the year.
            - snow_cols: a list of strings representing the names of the features containing the total snow amount for each day 
                        of the first two months of the year.
    """
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
    """
    Returns a list of strings representing the names of the numeric features used in the final model.

    Returns:
        list: A list of strings representing the names of the following numeric features:
            - campaña: an integer representing the year of the campaign.
            - superficie: a float representing the size of the crop field.
            - prod_shift_max: a float representing the maximum production shift in the field.
            - prod_shift_change: a float representing the change in production shift over time.
            - prod_shift_avg: a float representing the average production shift.
            - prod_he_var_zone_mean_hist_total: a float representing the mean historical production variance 
                                                of the field divided by the variance of the zone.
            - prod_he_var_zone_std_hist_total: a float representing the standard deviation of the historical production variance 
                                                of the field divided by the variance of the zone.
            - prod_he_var_modo_zona_mean_shift1_total: a float representing the mean shift in the mode of the zone's historical 
                                                        production variance compared to the current year's production variance.
            - prod_he_var_modo_zona_change_total: a float representing the change in the mode of the zone's historical 
                                                production variance over time.
    """
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
    """
    Returns a ColumnTransformer object representing the final preprocessing pipeline for the national phase.

    Returns:
        ColumnTransformer: A ColumnTransformer object with the following steps:
            - Flag: An OrdinalEncoder to encode the 'sup_is_nan' column as an ordinal variable, with unknown values encoded as -1.
            - Zona_encoder: A TargetEncoder to encode the 'id_zona' column as a target variable.
            - Zona_encoder_2: An OrdinalEncoder to encode the 'id_zona' column as an ordinal variable, with unknown values encoded as -1.
            - Variedad_encoder: An OrdinalEncoder to encode the 'variedad' column as an ordinal variable, with unknown values encoded as -1.
            - Modo_encoder: An OrdinalEncoder to encode the 'modo' column as an ordinal variable, with unknown values encoded as -1.
            - scaler: A StandardScaler to scale the numeric features returned by get_final_model_num_features().
            - Under_over_scaler: A StandardScaler to scale the precipitation features in precipuo.
            - Precip_PCA: A pipeline with a StandardScaler and a PCA with 2 components to reduce the dimensionality of the precipitation features in precip_cols.
            - Snow: A StandardScaler to scale the snow features in snow_cols.
    """

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
    """
    Returns a pipeline object representing the final model for the national phase.

    Returns:
        Pipeline: A pipeline object with the following steps:
            - prep: The preprocessing pipeline returned by get_preprocesing_fase_nacional().
            - model: A RandomForestRegressor with the following hyperparameters:
                - random_state: The seed for the random number generator.
                - n_estimators: The number of trees in the forest.
                - min_samples_leaf: The minimum number of samples required to be at a leaf node.
                - n_jobs: The number of jobs to run in parallel for both fit and predict.
                - max_features: The maximum number of features to consider when looking for the best split.
                - max_samples: The maximum number of samples to draw from X to train each base estimator.
    """

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