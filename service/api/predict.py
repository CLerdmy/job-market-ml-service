from fastapi import APIRouter

from service.schemas.request import JobFeatures
from service.schemas.response import SalaryPrediction
from service.services.predictor import predict_salary

router = APIRouter()


@router.post("/", response_model=SalaryPrediction)
def predict(features: JobFeatures):
    prediction = predict_salary(features.model_dump())
    return SalaryPrediction(predicted_salary=prediction)
