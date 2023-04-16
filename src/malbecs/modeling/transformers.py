from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin, ClassNamePrefixFeaturesOutMixin
import pandas as pd
from typing import List
import numpy as np
import category_encoders as ce
from typing import List,Union

# create some warppers around category encoders to support latest scikit-learn api.

class TargetEncoder(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """
    A transformer that encodes categorical features using target encoding.

    Args:
        verbose (int, optional): Verbosity level. Defaults to 0.
        cols (List[str] or None, optional): The names of the categorical columns to encode. If None, all columns will be encoded. Defaults to None.
        min_samples_leaf (int, optional): The minimum number of samples required to perform a split when constructing the encoding. Defaults to 20.
    """
    def __init__(self,
            verbose: int = 0,
            cols:Union[List[str],None] = None,
            min_samples_leaf: int = 20,
        ):
        self.verbose=verbose
        self.cols=cols
        self.min_samples_leaf=min_samples_leaf
        

    def fit(self, X, y=None):
        """Fit the target encoder to the given data.
        
        Args:
            X (pandas.DataFrame): The input data to encode.
            y (pandas.Series, optional): The target variable. If None, the encoding will be computed using the mean of the target variable. Defaults to None.

        Returns:
            TargetEncoder: Returns self.
        """
        self.encoder_ = ce.TargetEncoder(
            cols=self.cols,
            verbose=self.verbose,
            min_samples_leaf=self.min_samples_leaf
        )
        self.encoder_.fit(X, y)
        return self
    
    def transform(self, X):
        """Transform the input data using the fitted encoder.
        
        Args:
            X (pandas.DataFrame): The input data to transform.

        Returns:
            pandas.DataFrame: The transformed data.
        """
        return self.encoder_.transform(X)



class CatBoostEncoder(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """A transformer that encodes categorical features using CatBoost encoding.
    
    Args:
        verbose (int, optional): Verbosity level. Defaults to 0.
    """

    def __init__(self, verbose: int = 0):
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit the CatBoost encoder to the given data.
        
        Args:
            X (pandas.DataFrame): The input data to encode.
            y (pandas.Series, optional): The target variable. If None, the encoding will be computed using the mean of the target variable. Defaults to None.

        Returns:
            CatBoostEncoder: Returns self.
        """
        self.encoder_ = ce.CatBoostEncoder(self.verbose)
        self.encoder_.fit(X, y)
        return self
    
    def transform(self, X):
        """Transform the input data using the fitted encoder.
        
        Args:
            X (pandas.DataFrame): The input data to transform.

        Returns:
            pandas.DataFrame: The transformed data.
        """
        return self.encoder_.transform(X)



class BaseNEncoder(BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin):
    """A transformer that encodes categorical features using Base-N encoding.
    
    Args:
        verbose (int, optional): Verbosity level. Defaults to 0.
        base (int, optional): The base of the encoding. Defaults to 2.
        
    """

    def __init__(self, verbose:int = 0, base:int = 2):
        self.verbose = verbose
        self.base = base


    def fit(self, X, y=None):
        """Fit the Base-N encoder to the given data.
        
        Args:
            X (pandas.DataFrame): The input data to encode.
            y (pandas.Series, optional): The target variable. If None, the encoding will be computed using the mean of the target variable. Defaults to None.

        Returns:
            BaseNEncoder: Returns self.
        """
        self.encoder_ = ce.BaseNEncoder(
            verbose = self.verbose,
            base = self.base
        )
        self.encoder_.fit(X, y)
        self._n_features_out = len(self.encoder_.get_feature_names_out())
        return self
    
    def transform(self, X):
        """Transform the input data using the fitted encoder.
        
        Args:
            X (pandas.DataFrame): The input data to transform.

        Returns:
            pandas.DataFrame: The transformed data.
        """

        return self.encoder_.transform(X)





class QuantileFeatureEncoder(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """Transforms a categorical column by encoding each category with the quantile
    of the target variable.
    
    Args:
        col : list of str
            The name of the column(s) to encode.
        qs : list of float, optional (default=[0.25, 0.5, 0.75])
            The quantiles to use for encoding each category.
        scale : bool, optional (default=True)
            If True, the encoded values are scaled to have zero mean and unit variance.
    """
    def __init__(self, col: List[str], qs=[0.25, 0.5, 0.75], scale=True):
        self.col = col
        self.qs = qs
        self.scale = scale
        

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits the encoder to the data.
        
        Args:            
            X : pandas DataFrame
                The input data to fit.
            y : pandas Series
                The target variable to use for encoding.
        """
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

    def transform(self, X:pd.DataFrame):
        """
        Encodes the categorical variable specified in `self.col` using the precomputed category encodings
        and fills any missing values with -1.

        Args:
            X: A pandas DataFrame with the categorical column to encode.

        Returns:
            A pandas DataFrame with the categorical column encoded and any missing values filled with -1.

        Raises:
            None.
        """
        X = X.copy()
        X[self.col] = X[self.col].map(self.category_encodings_)
        X[self.col] = X[self.col].fillna(-1)
        return X



class ThresholdFeatureEncoder(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """
    Encodes a categorical variable based on the threshold of a target variable.

    Args:
        col (str): The name of the categorical column to encode.
        method (str): The method used to calculate the threshold of the target variable. Must be one of 'mean' or 'median'.

    """
    methods = ['mean', 'median']

    def __init__(self, col, method='mean'):
        self.col = col
        if method not in self.methods:
            raise ValueError(f"method must be one of {self.methods}")
        self.method = method

    def fit(self, X, y):
        """
        Fits the encoder to the data by calculating the threshold of the target variable.

        Args:
            X: A pandas DataFrame with the categorical column to encode.
            y: A pandas Series with the target variable.

        Returns:
            self: The fitted encoder.

        Raises:
            TypeError: If `X` is not a pandas DataFrame or if `y` is not a pandas Series.
        """

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
        """
        Encodes the categorical variable in `X` based on the precomputed category encodings and fills any missing values with 0.

        Args:
            X: A pandas DataFrame with the categorical column to encode.

        Returns:
            A pandas DataFrame with the categorical column encoded and any missing values filled with 0.

        Raises:
            None.
        """
        X = X.copy()
        X[self.col] = X[self.col].map(self.category_encodings_)
        X[self.col] = X[self.col].fillna(0)
        return X
