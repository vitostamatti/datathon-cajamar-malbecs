from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin, ClassNamePrefixFeaturesOutMixin
import pandas as pd
from typing import List
import numpy as np
import category_encoders as ce
from typing import List,Union

# create some warppers around category encoders to support latest scikit-learn api.

class TargetEncoder(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    
    def __init__(self,
            verbose: int = 0,
            cols:Union[List[str],None] = None,
            min_samples_leaf: int = 20,
        ):
        self.verbose=verbose
        self.cols=cols
        self.min_samples_leaf=min_samples_leaf
        

    def fit(self, X, y=None):
        self.encoder_ = ce.TargetEncoder(
            cols=self.cols,
            verbose=self.verbose,
            min_samples_leaf=self.min_samples_leaf
        )
        self.encoder_.fit(X, y)
        return self
    
    def transform(self, X):
        return self.encoder_.transform(X)



class CatBoostEncoder(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self, verbose: int = 0):
        self.verbose = verbose

    def fit(self, X, y=None):
        self.encoder_ = ce.CatBoostEncoder(self.verbose)
        self.encoder_.fit(X, y)
        return self
    
    def transform(self, X):
        return self.encoder_.transform(X)



class BaseNEncoder(BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin):
    def __init__(self, verbose:int = 0, base:int = 2):
        self.verbose = verbose
        self.base = base


    def fit(self, X, y=None):
        self.encoder_ = ce.BaseNEncoder(
            verbose = self.verbose,
            base = self.base
        )
        self.encoder_.fit(X, y)
        self._n_features_out = len(self.encoder_.get_feature_names_out())
        return self
    
    def transform(self, X):
        return self.encoder_.transform(X)





class QuantileFeatureEncoder(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self, col: List[str], qs=[0.25, 0.5, 0.75], scale=True):
        self.col = col
        self.qs = qs
        self.scale = scale
        

    def fit(self, X: pd.DataFrame, y: pd.Series):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            return Exception("X must be of type pd.DataFrame")
        if not isinstance(y, pd.Series):
            return Exception("y must be of type pd.Series")

        category_means_ = pd.concat([X[self.col], y], axis=1).groupby(self.col)[
            y.name].mean()

        qs_ = [category_means_.quantile(q) for q in self.qs]

        def encode_qs(x, qs):
            for i, q in enumerate(qs):
                if x < q:
                    return i
            return len(qs)

        self.category_encodings_ = category_means_.apply(
            lambda x: encode_qs(x, qs_)).to_dict()

        self.mean_ = np.mean(category_means_)

        return self

    def transform(self, X):
        X = X.copy()
        X[self.col] = X[self.col].map(self.category_encodings_)
        X[self.col] = X[self.col].fillna(self.mean_)
        return X



class GroupedQuantileFeatureEncoder(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):

    def __init__(self, col: str, groups:List[str], value:str, qs=[0.25, 0.5, 0.75], scale=True):
        self.col = col
        self.groups = groups
        self.value = value
        self.qs = qs
        self.scale = scale
        

    def fit(self, X: pd.DataFrame, y: pd.Series):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            return Exception("X must be of type pd.DataFrame")
        if not isinstance(y, pd.Series):
            return Exception("y must be of type pd.Series")

    
        category_means_ = X.groupby(self.groups)[self.value].mean()
        # category_means_ = pd.concat([X[self.col], y], axis=1).groupby(self.col)[
            # y.name].mean()

        qs_ = [category_means_.quantile(q) for q in self.qs]

        def encode_qs(x, qs):
            for i, q in enumerate(qs):
                if x < q:
                    return i
            return len(qs)

        self.category_encodings_ = category_means_.apply(
            lambda x: encode_qs(x, qs_)).to_dict()

        self.mean_ = np.mean(category_means_)

        return self

    def transform(self, X):
        X = X.copy()
        X[self.col] = X[self.col].map(self.category_encodings_)
        X[self.col] = X[self.col].fillna(self.mean_)
        return X




class ThresholdFeatureEncoder(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    methods = ['mean', 'median']

    def __init__(self, col, method='mean'):
        self.col = col
        if method not in self.methods:
            raise ValueError(f"method must be one of {self.methods}")
        self.method = method

    def fit(self, X, y):

        if not isinstance(X, pd.DataFrame):
            return Exception("X must be of type pd.DataFrame")
        if not isinstance(y, pd.Series):
            return Exception("y must be of type pd.Series")

        if self.method == 'mean':
            category_measure_ = pd.concat([X[self.col], y], axis=1).groupby(self.col)[
                y.name].mean()
            target_measure_ = np.mean(y)
        elif self.method == 'median':
            category_measure_ = pd.concat([X[self.col], y], axis=1).groupby(self.col)[
                y.name].median()
            target_measure_ = np.median(y)

        self.category_encodings_ = category_measure_.apply(
            lambda x: 0 if x < target_measure_ else 1).to_dict()

        return self

    def transform(self, X):
        X = X.copy()
        X[self.col] = X[self.col].map(self.category_encodings_)
        X[self.col] = X[self.col].fillna(0)
        return X
