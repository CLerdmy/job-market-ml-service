from typing import Any, Dict

import polars as pl

from src.data.feature_engineering import build_features
from src.data.preprocessing import preprocess, use_salary_predictor_columns


def build_features_for_request(
    features: Dict[str, Any], mte: Dict[str, Dict[Any, float]]
) -> pl.DataFrame:
    df = pl.DataFrame([features])

    df_processed = preprocess(df, "job_market", mte=mte, train=False)

    df_features = build_features(df_processed, "job_market")

    df_features = use_salary_predictor_columns(df_features)

    return df_features
