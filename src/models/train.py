from typing import Union
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_lightgbm(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray]
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        num_leaves = 31,
        max_depth = -1,
        learning_rate = 0.05,
        n_estimators = 600,
        min_split_gain = 0,
        min_child_weight = 0.001,
        min_child_samples = 20
    )

    model.fit(X, y)

    return model

def train_ridge(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray]
) -> Pipeline:
    ridge_params = {
        'alpha': 10.0
    }

    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(*ridge_params))
    ])

    ridge_pipe.fit(X, y)

    return ridge_pipe
