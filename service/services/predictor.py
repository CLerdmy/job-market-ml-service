from service.services.model_loader import salary_predictor
from service.services.feature_builder import build_features_for_request

def predict_salary(features: dict) -> float:
    X = build_features_for_request(features, salary_predictor.get_mte())
    return float(salary_predictor.get_model().predict(X)[0])