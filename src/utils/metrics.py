from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def rmse(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
) -> float:
    return r2_score(y_true, y_pred)


def mape(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def smape(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
) -> float:
    return (
        100
        / len(y_true)
        * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    )


def wape(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
) -> float:
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
