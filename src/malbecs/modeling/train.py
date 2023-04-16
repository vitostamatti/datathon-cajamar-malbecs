# script to train the final model
import pickle as pkl
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection._split import _num_samples


def load_final_data(wine_path:str, eto_path:str=None, meteo_path:str=None) -> pd.DataFrame:
    """Load and merge data from CSV files to create a final dataframe of wine data with optional meteorological and 
    ETO data.
    
    Args:
        wine_path (str): Path to the wine data CSV file.
        eto_path (str, optional): Path to the ETo data CSV file. Defaults to None.
        meteo_path (str, optional): Path to the meteorological data CSV file. Defaults to None.
    
    Returns:
        pd.DataFrame: A pandas dataframe containing merged wine data with optional ETo and meteorological data.
    
    """
    wine_data = pd.read_csv(wine_path)
    if eto_path:
        eto_data = pd.read_csv(eto_path)
        wine_data = merge_data(wine_data, eto_data=eto_data)
    if meteo_path:
        meteo_data = pd.read_csv(meteo_path)
        wine_data = merge_data(wine_data, meteo_data=meteo_data)
    return wine_data


def merge_data(wine_data: pd.DataFrame, eto_data: pd.DataFrame = None, meteo_data: pd.DataFrame = None) -> pd.DataFrame:
    """Merge wine data with optional ETO and meteorological data using left join on 'id_estacion' and 'campaña'
    columns, and 'ID_ESTACION' and 'year' columns respectively.
    
    Args:
        wine_data (pd.DataFrame): The pandas dataframe containing wine data.
        eto_data (pd.DataFrame, optional): The pandas dataframe containing ETo data. Defaults to None.
        meteo_data (pd.DataFrame, optional): The pandas dataframe containing meteorological data. Defaults to None.
    
    Returns:
        pd.DataFrame: A pandas dataframe containing merged wine data with optional ETo and meteorological data.
    
    """
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


def filter_camp(data: pd.DataFrame, min_camp: int, max_camp: int) -> pd.DataFrame:
    """Filter wine data between two campaign years.
    
    Args:
    data (pd.DataFrame): The pandas dataframe containing wine data.
    min_camp (int): The minimum campaign year.
    max_camp (int): The maximum campaign year.
    
    Returns:
    pd.DataFrame: A pandas dataframe containing wine data between two campaign years.
    
    """
    return data[(data['campaña'] >= min_camp) & (data['campaña'] <= max_camp)]


def train_test_split(data: pd.DataFrame, test_camp: int = 21) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split wine data into training and testing sets.
    
    Args:
        data (pd.DataFrame): The pandas dataframe containing wine data.
        test_camp (int, optional): The campaign year to be used as the testing set. Defaults to 21.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two pandas dataframes, one for training data and another for testing data.
    
    """
    return data[data['campaña'] != test_camp], data[data['campaña'] == test_camp]


def xy_split(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split wine data into features and target variables.
    
    Args:
        data (pd.DataFrame): The pandas dataframe containing wine data.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two pandas dataframes, one for features and another for target variables.
    
    """
    return data.drop(columns=['produccion']), data['produccion']



class CampKFold():
    """
    Custom kfold based on given train and test indexes.

    
    Args:
        train_idxs (list): A list of training indices for each fold.
        test_idxs (list): A list of testing indices for each fold.
    """
    def __init__(self, train_idxs, test_idxs):
        self.train_idxs = train_idxs
        self.test_idxs = test_idxs
        self.n_splits = len(test_idxs)

    def get_train_test(X_camps, from_camp, to_camp):
        """Get the training and testing indices for the given campaign years.
        
        Args:
            X_camps (np.array): An array of campaign years.
            from_camp (int): The first campaign year to be used in testing.
            to_camp (int): The last campaign year to be used in testing.
        
        Returns:
            list: A list of training indices and a list of testing indices.
        
        """
        test_idxs = []
        train_idxs = []
        for i in range(from_camp, to_camp+1):
            train_idxs.append(X_camps < i)
            test_idxs.append(X_camps == i)
        return train_idxs, test_idxs

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and testing folds.
        
        Args:
            X (np.array): An array of samples.
            y (np.array, optional): An array of target labels. Defaults to None.
            groups (np.array, optional): An array of groups. Defaults to None.
        
        Yields:
            tuple: A tuple of indices for the training set and the testing set.
        
        """
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        for train, test in zip(self.train_idxs, self.test_idxs):
            yield (
                indices[train],
                indices[test],
            )

    def get_n_splits(self, X, y=None, groups=None):
        """Get the number of splitting iterations.
        
        Args:
            X (np.array): An array of samples.
            y (np.array, optional): An array of target labels. Defaults to None.
            groups (np.array, optional): An array of groups. Defaults to None.
        
        Returns:
            int: The number of splitting iterations.
        
        """
        return self.n_splits


def rmse_score(y_true, y_pred):
    """
    Calculates the root mean squared error (RMSE) between two arrays of values.

    Args:
        y_true (array-like of shape (n_samples,)): Ground truth (correct) target values.
        y_pred (array-like of shape (n_samples,)): Estimated target values.

    Returns:
        float: The RMSE between `y_true` and `y_pred`.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


rmse_scorer = make_scorer(rmse_score, greater_is_better=False)


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


def convert_cat_features(X):
    """Convert categorical features of a DataFrame to the 'category' data type.

    Args:
        X (pandas.DataFrame): The input DataFrame with categorical features.

    Returns:
        pandas.DataFrame: The transformed DataFrame with categorical features in 'category' data type.
    """
    cat_cols = [
        'id_finca',
        'id_zona',
        'id_estacion',
        'variedad',
        "modo",
        "tipo",
        "color",
        "prod_shift1_gt_shift2",
        "sup_is_nan",
    ]

    X[cat_cols] = X[cat_cols].astype('category')
    return X


def load_xy(wine_path, eto_path, meteo_path, min_camp=14, max_camp=21):
    """Load and preprocess data from different sources and returns training features and targets.

    Args:
        wine_path (str): The path where the wine data is located.
        eto_path (str): The path where the ETo data is located.
        meteo_path (str): The path where the meteorological data is located.
        min_camp (int, optional): The minimum number of days of the campaign to consider in the training data. Defaults to 14.
        max_camp (int, optional): The maximum number of days of the campaign to consider in the training data. Defaults to 21.

    Returns:
        tuple(pandas.DataFrame, pandas.DataFrame): A tuple containing the training features DataFrame (X) and the target DataFrame (y).
    """
    data = load_final_data(
        wine_path=wine_path,
        eto_path=eto_path,
        meteo_path=meteo_path
    )

    data_train = filter_camp(data.copy(), min_camp=min_camp, max_camp=max_camp)

    X, y = xy_split(data_train)

    X = convert_cat_features(X)

    return X, y


def evaluate_model(model, X, y):
    """Evaluate a machine learning model using nested cross-validation.

    Args:
        model (estimator object): The model to be evaluated.
        X (pandas.DataFrame): The input DataFrame with the features.
        y (pandas.DataFrame): The input DataFrame with the target variable.

    Returns:
        dict: A dictionary containing the evaluation metrics (train and test RMSE).
    """
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
    """Search for the best hyperparameters of a machine learning model using nested cross-validation and grid search.

    Args:
        model (estimator object): The model to be tuned.
        X (pandas.DataFrame): The input DataFrame with the features.
        y (pandas.DataFrame): The input DataFrame with the target variable.
        param_grid (dict): The hyperparameters to be tuned.

    Returns:
        estimator object: The best estimator found.
    """
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
