# script to train the final model
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate

from sklearn.model_selection._split import _BaseKFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection._split import _num_samples
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from malbecs.modeling.transformers import QuantileFeatureEncoder, ThresholdFeatureEncoder


def load_final_data(wine_path, eto_path=None, meteo_path=None):
    wine_data = pd.read_csv(wine_path)
    if eto_path:
        eto_data = pd.read_csv(eto_path)
        wine_data = merge_data(wine_data, eto_data=eto_data)
    if meteo_path:
        meteo_data = pd.read_csv(meteo_path)
        wine_data = merge_data(wine_data, meteo_data=meteo_data)
    return wine_data


def merge_data(wine_data, eto_data=None, meteo_data=None):
    if eto_data is not None:
        eto_data['year'] = eto_data['year'] % 2000
        wine_data = wine_data.merge(
            eto_data,
            left_on=['id_estacion', 'campaña'],
            right_on=['ID_ESTACION', 'year'],
            how='left',
        )
    if meteo_data is not None:
        meteo_data['year'] = meteo_data['year'] % 2000
        wine_data = wine_data.merge(
            meteo_data,
            left_on=['id_estacion', 'campaña'],
            right_on=['ID_ESTACION', 'year'],
            how='left'
        )
    return wine_data


def filter_camp(data, min_camp, max_camp):
    return data[(data['campaña'] >= min_camp) & (data['campaña'] <= max_camp)]


def train_test_split(data, test_camp=21):
    return data[data['campaña'] != test_camp], data[data['campaña'] == test_camp]


def xy_split(data):
    return data.drop(columns=['produccion']), data['produccion']


class CampKFold():
    def __init__(self, train_idxs, test_idxs):
        self.train_idxs = train_idxs
        self.test_idxs = test_idxs
        self.n_splits = len(test_idxs)

    def get_train_test(X_camps, from_camp, to_camp):
        test_idxs = []
        train_idxs = []
        for i in range(from_camp, to_camp+1):
            train_idxs.append(X_camps < i)
            test_idxs.append(X_camps == i)
        return train_idxs, test_idxs

    def split(self, X, y=None, groups=None):

        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        for train, test in zip(self.train_idxs, self.test_idxs):
            yield (
                indices[train],
                indices[test],
            )

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits


def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


rmse_scorer = make_scorer(rmse_score, greater_is_better=False)


def save_trained_model(model, path):
    with open(path, "wb") as file:
        pkl.dump(model, file)


def load_trained_model(path):
    with open(path,  "rb") as file:
        model = pkl.load(file)
    return model


def load_xy(wine_path, eto_path, meteo_path, min_camp=14, max_camp=21):

    data = load_final_data(
        wine_path=wine_path,
        eto_path=eto_path,
        meteo_path=meteo_path
    )

    data_train = filter_camp(data.copy(), min_camp=min_camp, max_camp=max_camp)

    X, y = xy_split(data_train)

    cat_cols = [
        'id_finca',
        'id_zona',
        'id_estacion',
        'variedad',
        "modo",
        "tipo",
        "color",
        "prod_shift1_gt_shift2"
    ]

    X[cat_cols] = X[cat_cols].astype('category')

    return X, y


def evaluate_model(model, X, y):

    train_idxs, test_idxs = CampKFold.get_train_test(
        X['campaña'], from_camp=19, to_camp=21
    )

    cv = CampKFold(train_idxs, test_idxs)

    res = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        n_jobs=-1,
        scoring=rmse_scorer,
        return_train_score=True,
        return_estimator=False,
        verbose=2
    )

    print(f"Train RMSE: {res.get('train_score')}")
    print(f"Test RMSE: {res.get('test_score')}")

    return res


def search_model_params(model, X, y, param_grid):

    train_idxs, test_idxs = CampKFold.get_train_test(
        X['campaña'], from_camp=19, to_camp=21
    )
    cv = CampKFold(train_idxs, test_idxs)

    gsm = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=cv,
        verbose=2,
        scoring=rmse_scorer
    )

    gsm.fit(X, y)
    print("Best Params: ", gsm.best_params_)
    print("Best Mean Score", gsm.best_score_)
    return gsm
