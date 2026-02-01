# Project Overview

**job-market-ml-service** is a machine learning service for analyzing the job market.  
The project is designed to predict **various job-related characteristics**, including salaries, job competitiveness, and other potential indicators. The architecture allows integration of **multiple ML models**, making it easy to expand the service to new types of predictions in the future.

---

## Notebooks Workflow

The `notebooks/` directory contains the full exploratory and experimental workflow:

- **01_eda.ipynb:** exploratory data analysis and data understanding.
- **02_data_preprocessing.ipynb:** data cleaning, handling missing values, and basic transformations.
- **03_feature_engineering.ipynb:** creating new features, aggregation and transformation of existing ones.
- **04_model_training.ipynb:** model training and hyperparameter tuning.
- **05_evaluation.ipynb:** model validation and performance analysis.

---

## Salary Prediction

The currently implemented component of the project is **Salary Prediction**:

- Predicts the **mean salary** for a job posting using structured job features.
- Implemented using a **LightGBM regression model**.
- Trained and tested on the **Job Market Dataset** from Kaggle: [Job Dataset](https://www.kaggle.com/datasets/shaistashahid/job-market-insight).

**Input features description:**

The model uses the following job attributes as input:

- **Job Title:** the name or designation of the job position.
- **Company:** the name of the hiring organization.
- **Location:** the city or area where the job is based.
- **Work Type:** type of employment (e.g., full-time, part-time).
- **Category:** the general job category or role type (e.g., Software, Marketing).
- **Experience:** required or preferred years of experience for the role.
- **Skills:** a list of required skills or qualifications needed for the position.

**API Endpoints:**

- `POST /predict`: returns predicted salary
- `GET /health`: service health check

---

## Key Features

- **Full ML pipeline:** EDA, preprocessing, feature engineering, training, and evaluation  
- **FastAPI service** for inference with structured request/response schemas  
- **Modular architecture:** `service/` for backend, `src/` for reusable ML code  
- **Extensible:** easy to integrate additional models for predicting other job-related metrics

---

## Technologies Used

The project uses the following main dependencies and technologies:

- **FastAPI** for REST API backend
- **Pydantic** for data validation and schemas
- **Uvicorn** as ASGI server
- **NumPy, Pandas, Polars, PyArrow** for data processing
- **Matplotlib** for visualization and EDA
- **Scikit-learn** for preprocessing and metrics
- **LightGBM** for salary prediction model
- **Loguru** for logging
