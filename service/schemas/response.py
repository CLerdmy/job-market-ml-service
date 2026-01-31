from pydantic import BaseModel

class SalaryPrediction(BaseModel):
    predicted_salary: float