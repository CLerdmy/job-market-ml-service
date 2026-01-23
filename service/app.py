from fastapi import FastAPI
from service.api.predict import router as predict_router

app = FastAPI(
    title="Job Market ML Service",
    version="0.0.1"
)

app.include_router(predict_router, prefix="/predict", tags=["prediction"])

@app.get("/health")
def health_check():
    return {"status": "ok"}