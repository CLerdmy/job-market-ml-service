from typing import Optional

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.feature_engineering import build_features
from src.data.load import load_job_market_dataset
from src.data.preprocessing import preprocess
from src.models.train import train_lightgbm


def build_salary_prediction_model(
    use_split: bool = True, test_size: float = 0.2, random_state: int = 42
) -> tuple[
    lgb.LGBMRegressor,
    pd.DataFrame,
    pd.Series,
    Optional[pd.DataFrame],
    Optional[pd.Series],
]:
    dataset = "job_market"
    data = load_job_market_dataset()
    data = preprocess(data, dataset)
    data = build_features(data, dataset)

    data = data.to_pandas()

    target = "salary_mean"
    X = data.drop(columns=[target])
    y = data[target]

    if use_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        model = train_lightgbm(X_train, y_train)

    else:
        model = train_lightgbm(X, y)
        X_test, y_test = None, None

    return model, X, y, X_test, y_test
