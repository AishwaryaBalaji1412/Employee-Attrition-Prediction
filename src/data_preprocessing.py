
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

TARGET_COLUMN = "Attrition"

# Columns that should not be used for prediction
DROP_COLUMNS = [
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours",
    "Over18"
]


def load_data():
    """Load the employee attrition dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df


def preprocess_data(df):
    """Validate and preprocess the employee dataset."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    if df[TARGET_COLUMN].isnull().any():
        raise ValueError(
            "The target column contains missing values."
        )

    # Validate target values
    valid_targets = {"Yes", "No"}

    actual_targets = set(df[TARGET_COLUMN].unique())

    if not actual_targets.issubset(valid_targets):
        raise ValueError(
            f"Unexpected target values: {actual_targets}"
        )

    # Convert target labels into numeric values
    y = df[TARGET_COLUMN].map({
        "No": 0,
        "Yes": 1
    })

    # Separate input features
    X = df.drop(columns=[TARGET_COLUMN])

    # Remove unnecessary identifier and metadata columns
    X = X.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    # Remove constant features
    constant_columns = [
        column
        for column in X.columns
        if X[column].nunique(dropna=False) <= 1
    ]

    X = X.drop(columns=constant_columns)

    # Validate missing values
    if X.isnull().any().any():
        missing_columns = (
            X.columns[X.isnull().any()].tolist()
        )

        raise ValueError(
            f"Missing values found in features: {missing_columns}"
        )

    # Identify categorical features
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return X, y, categorical_cols


def split_data(X, y):
    """Split the dataset into stratified training and testing sets."""

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


def prepare_data():
    """Run the complete data preparation workflow."""

    df = load_data()

    X, y, categorical_cols = preprocess_data(df)

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(X, y)

    return (
        df,
        X,
        y,
        categorical_cols,
        X_train,
        X_test,
        y_train,
        y_test
    )
