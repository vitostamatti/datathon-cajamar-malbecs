# script to train the final model
import pandas as pd
import numpy as np

from sklearn.model_selection._split import _BaseKFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection._split import _num_samples


def load_final_data(wine_path, eto_path=None, meteo_path=None):
    wine_data = pd.read_csv(wine_path)
    if eto_path:
        eto_data = pd.read_csv(eto_path)
        eto_data['year'] = eto_data['year'] % 2000
        wine_data = wine_data.merge(
            eto_data,
            left_on=['id_estacion', 'campaña'],
            right_on=['ID_ESTACION', 'year'],
            how='left',
        )
    if meteo_path:
        meteo_data = pd.read_csv(meteo_path)
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


class IndexKFold(_BaseKFold):
    def __init__(self, test_idxs):
        self.test_idxs = test_idxs
        self.n_splits = len(test_idxs)

    def _iter_test_indices(self, X: pd.DataFrame, y=None, groups=None):
        for idx in self.test_idxs:
            yield idx


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
