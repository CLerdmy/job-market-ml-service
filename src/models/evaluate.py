from sklearn.model_selection import KFold, cross_val_score

from src.utils.metrics import mae, mape, r2, rmse, smape, wape


def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print(
        f"RMSE: {rmse(y_test, y_pred)}",
        f"MAE: {mae(y_test, y_pred)}",
        f"R2: {r2(y_test, y_pred)}",
        f"MAPE: {mape(y_test, y_pred)}",
        f"SMAPE: {smape(y_test, y_pred)}",
        f"WAPE: {wape(y_test, y_pred)}",
        sep="\n",
    )


def calculate_cv_rmse(model, X, y, cv: int = 5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        model, X, y, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1
    )

    rmse_scores = -cv_scores

    print(f"CV RMSE folds: {rmse_scores}")
    print(f"CV RMSE mean: {rmse_scores.mean():.2f}")
