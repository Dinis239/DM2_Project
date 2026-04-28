import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols:list):
        self.num_cols = num_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.replace('?', np.nan)
        X[self.num_cols] = X[self.num_cols].clip(lower=0)
        X[self.num_cols] = np.round(X[self.num_cols])
        return X
        

def run_gridsearch(grid: dict,
                   cv: any,
                   X: any,
                   y: any, 
                   model: any) -> pd.DataFrame:
    """
    Perform a manual grid search with cross-validation to tune model
    hyperparameters.

    Parameters:
        ----------
         - grid (dict): Dictionary with parameters names as keys and lists
           of settings to try.
         - cv (int/generator): Determines the cross-validation splitting
         strategy.
         - X (array-like): The training input samples.
         - y (array-like): The target values.
         - model (estimator): The object to use to fit the data.

    Returns:
        ----------
         pd.DataFrame: A DataFrame containing the parameters, mean/std scores
         for training and validation, and execution status.
    """
    params = ParameterGrid(grid)
    results = []
    for param in tqdm(params, desc="Tuning Hyperparameters"):
        try:
            model.set_params(**param)
            cv_results = cross_validate(model, X, y, cv=cv,
                                        return_train_score=True,
                                        scoring='f1', n_jobs=-1)
            param['params'] = param.copy()
            param['mean_val_f1'] = np.mean(cv_results['test_score'])
            param['std_val_f1'] = np.std(cv_results['test_score'])
            param['mean_train_f1'] = np.mean(cv_results['train_score'])
            param['std_train_f1'] = np.std(cv_results['train_score'])
            param['status'] = 'Success'
        except Exception as e:
            param['mean_val_f1'] = np.nan
            param['std_val_f1'] = np.nan
            param['mean_train_f1'] = np.nan
            param['std_train_f1'] = np.nan
            param['status'] = f'Failed: {str(e)[:50]}'
        results.append(param)
    return pd.DataFrame(results).sort_values('mean_val_f1')
