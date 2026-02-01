from pathlib import Path

import polars as pl

from src.data.feature_engineering import build_features
from src.data.preprocessing import preprocess


def load_job_market_dataset() -> pl.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "job_market.csv"
    return pl.read_csv(data_path)


def load_prepared_job_market_dataset() -> pl.DataFrame:
    data = load_job_market_dataset()
    data = preprocess(data, "job_market")
    data = build_features(data, "job_market")
    return data
