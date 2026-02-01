import os
from typing import Any, Dict

import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def _with_pkl(name: str) -> str:
    if not name.endswith(".pkl"):
        return f"{name}.pkl"
    return name


def save_model(model: Any, model_name: str) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    filename = _with_pkl(model_name)
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)


def load_model(model_name: str) -> Any:
    filename = _with_pkl(model_name)
    path = os.path.join(MODEL_DIR, filename)
    return joblib.load(path)


def save_mte(df_pd: pd.DataFrame, dataset: str) -> None:
    config = {"job_market": [["salary_mean"], ["job_title", "company", "location"]]}
    target = config.get(dataset, [[], []])[0][0]
    mte_cols = config.get(dataset, [[], []])[1]

    mte = {}

    for col in mte_cols:
        mte[col] = df_pd.groupby(col)[target].mean().to_dict()

    os.makedirs(MODEL_DIR, exist_ok=True)
    filename = f"mte_{dataset}.pkl"
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(mte, path)


def load_mte(mte_name: str) -> Dict[str, Dict[Any, float]]:
    filename = _with_pkl(mte_name)
    path = os.path.join(MODEL_DIR, filename)
    return joblib.load(path)
