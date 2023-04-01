
# random-forest
from dataclasses import dataclass, asdict
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, StandardScaler
import pickle as pkl
from typing import List
import malbecs.modeling.transformers as mt

seed = 42


def get_final_model():

    model_num_cols = [
        'superficie',
        'prod_shift_max',
        'prod_shift_change',
        'prod_shift_avg',
    ]

    m = make_pipeline(
        make_column_transformer(

            (mt.BaseNEncoder(), ['id_finca']),

            (mt.TargetEncoder(), ['id_zona']),

            (OrdinalEncoder(handle_unknown='use_encoded_value',
             unknown_value=-1), ['id_estacion']),

            (mt.BaseNEncoder(), ['variedad']),

            (OrdinalEncoder(handle_unknown='use_encoded_value',
             unknown_value=-1), ['modo']),

            (KBinsDiscretizer(n_bins=2), ['altitud']),

            (StandardScaler(), model_num_cols),

            remainder='drop'
        ),
        RandomForestRegressor(
            random_state=seed,
            n_estimators=200,
            min_samples_leaf=4,
            n_jobs=-1,
            max_features='sqrt',
            max_samples=0.8

        )
    )
    return m


def save_trained_model(model, path):
    with open(path, "wb") as file:
        pkl.dump(model, file)


def load_trained_model(path):
    with open(path,  "rb") as file:
        model = pkl.load(file)
    return model


# catboost
@dataclass()
class CatBoostRegressorParams:
    iterations: int = None
    learning_rate: float = None
    depth: int = None
    l2_leaf_reg = None
    model_size_reg = None
    random_seed = 42
    use_best_model = None
    verbose: int = 0
    max_depth: int = None
    n_estimators: int = None
    num_boost_round: int = None
    num_trees = None
    colsample_bylevel = None
    random_state: int = 46
    reg_lambda: float = None
    early_stopping_rounds = None
    cat_features: List[str] = None
    grow_policy = None
    min_data_in_leaf: int = None


default_catboost_params = CatBoostRegressorParams(
    random_state=42,
    iterations=500,
    cat_features=[
        'id_finca',
        'id_zona',
        "id_estacion",
        'variedad',
        "modo",
        "tipo",
        "color",
        "prod_shift1_gt_shift2"
    ],
    max_depth=4
)


def get_catboost_model(catboost_params: CatBoostRegressorParams = default_catboost_params, **cbkwargs):
    import catboost as cb
    m = cb.CatBoostRegressor(
        random_seed=42, **asdict(catboost_params), **cbkwargs)
    return m

# xgboost

# knn

# Lasso

# hist-forest
