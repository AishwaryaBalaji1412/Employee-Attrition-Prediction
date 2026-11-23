
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


MODEL_FILES = {
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl",
    "Ensemble": "ensemble_model.pkl"
}


def load_model(model_name):
    """Load a previously trained model."""

    if model_name not in MODEL_FILES:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from {list(MODEL_FILES.keys())}"
        )

    model_path = MODEL_DIR / MODEL_FILES[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Please train the models first."
        )

    return joblib.load(model_path)


def prepare_employee_data(employee_data, model):
    """Validate and arrange employee input features."""

    if isinstance(employee_data, dict):
        employee_data = pd.DataFrame([employee_data])

    if not isinstance(employee_data, pd.DataFrame):
        raise TypeError(
            "Employee data must be a dictionary or pandas DataFrame."
        )

    if employee_data.empty:
        raise ValueError("Employee data cannot be empty.")

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "The trained model does not contain feature names."
        )

    expected_columns = list(model.feature_names_in_)

    missing_columns = [
        col for col in expected_columns
        if col not in employee_data.columns
    ]

    extra_columns = [
        col for col in employee_data.columns
        if col not in expected_columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing employee features: {missing_columns}"
        )

    if extra_columns:
        raise ValueError(
            f"Unexpected employee features: {extra_columns}"
        )

    employee_data = employee_data[expected_columns].copy()

    if employee_data.isnull().any().any():
        missing_values = (
            employee_data.columns[
                employee_data.isnull().any()
            ].tolist()
        )

        raise ValueError(
            f"Missing values found in columns: {missing_values}"
        )

    return employee_data


def predict_attrition(
    employee_data,
    model_name="Random Forest",
    threshold=0.5
):
    """Predict attrition for one or multiple employees."""

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    model = load_model(model_name)

    employee_data = prepare_employee_data(
        employee_data,
        model
    )

    probabilities = model.predict_proba(employee_data)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    results = pd.DataFrame({
        "Attrition Probability": probabilities,
        "Prediction": predictions,
        "Result": [
            "Likely to Leave" if prediction == 1
            else "Likely to Stay"
            for prediction in predictions
        ]
    }, index=employee_data.index)

    results["Attrition Probability"] = (
        results["Attrition Probability"].round(4)
    )

    return results


def predict_single_employee(
    employee_data,
    model_name="Random Forest",
    threshold=0.5
):
    """Predict attrition for a single employee."""

    if isinstance(employee_data, pd.DataFrame):
        if len(employee_data) != 1:
            raise ValueError(
                "Single employee prediction requires exactly one row."
            )

    results = predict_attrition(
        employee_data,
        model_name,
        threshold
    )

    return results.iloc[0].to_dict()
