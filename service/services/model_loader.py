from src.models.registry import load_model, load_mte
from typing import Dict, Any

class SalaryPredictor:
    def __init__(self, model_name: str, mte_name: str):
        self.model = load_model(model_name)
        self.mte: Dict[str, Dict[Any, float]] = load_mte(mte_name)

    def get_model(self):
        return self.model

    def get_mte(self):
        return self.mte

salary_predictor = SalaryPredictor("salary_predictor_full.pkl", "mte_job_market.pkl")
