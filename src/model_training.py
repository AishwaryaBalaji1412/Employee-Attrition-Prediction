
from pathlib import Path

import joblib
import numpy as np

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


def create_pipeline(classifier, categorical_cols, feature_cols):
    """Create a preprocessing and model-training pipeline."""

    categorical_indices = [
        feature_cols.get_loc(col)
        for col in categorical_cols
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_indices
            )
        ],
        remainder="passthrough"
    )

    return Pipeline([
        (
            "smote",
            SMOTENC(
                categorical_features=categorical_indices,
                random_state=42
            )
        ),
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


def find_best_threshold(y_true, y_prob):
    """Find the threshold that maximizes F1-score."""

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_prob
    )

    if len(thresholds) == 0:
        raise ValueError("Unable to calculate classification thresholds.")

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    return float(thresholds[np.argmax(f1_scores)])


def get_oof_probabilities(model, X_train, y_train, cv):
    """Generate out-of-fold probabilities for threshold selection."""

    probabilities = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    return probabilities


def get_model_configs():
    """Define classifiers and their hyperparameter search spaces."""

    return {
        "Random Forest": (
            RandomForestClassifier(
                random_state=42,
                n_jobs=1
            ),
            {
                "classifier__n_estimators": [200, 300],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2]
            }
        ),

        "XGBoost": (
            XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=1
            ),
            {
                "classifier__n_estimators": [200, 300],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__subsample": [0.8, 1.0]
            }
        )
    }


def tune_models(X_train, y_train, categorical_cols):
    """Tune model hyperparameters using stratified cross-validation."""

    model_configs = get_model_configs()

    best_models = {}
    best_parameters = {}

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    for name, (classifier, params) in model_configs.items():

        print(f"\nTuning {name}...")

        pipeline = create_pipeline(
            classifier,
            categorical_cols,
            X_train.columns
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            refit=True,
            error_score="raise"
        )

        grid.fit(X_train, y_train)

        best_models[name] = grid.best_estimator_
        best_parameters[name] = grid.best_params_

        print(f"Best {name} parameters:")
        print(grid.best_params_)

        print(
            f"Best cross-validation F1: "
            f"{grid.best_score_:.4f}"
        )

    return best_models, best_parameters


def tune_thresholds(best_models, X_train, y_train):
    """Select thresholds using out-of-fold training probabilities."""

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    oof_probabilities = {}

    for name, model in best_models.items():

        print(f"\nGenerating OOF probabilities for {name}...")

        oof_probabilities[name] = get_oof_probabilities(
            model,
            X_train,
            y_train,
            cv
        )

    ensemble = VotingClassifier(
        estimators=[
            ("rf", clone(best_models["Random Forest"])),
            ("xgb", clone(best_models["XGBoost"]))
        ],
        voting="soft"
    )

    print("\nGenerating OOF probabilities for Ensemble...")

    oof_probabilities["Ensemble"] = get_oof_probabilities(
        ensemble,
        X_train,
        y_train,
        cv
    )

    thresholds = {
        name: find_best_threshold(y_train, probabilities)
        for name, probabilities in oof_probabilities.items()
    }

    return thresholds


def train_final_models(best_models, X_train, y_train):
    """Fit individual models and create the soft voting ensemble."""

    models = {}

    for name, model in best_models.items():

        final_model = clone(model)
        final_model.fit(X_train, y_train)

        models[name] = final_model

    ensemble = VotingClassifier(
        estimators=[
            ("rf", clone(models["Random Forest"])),
            ("xgb", clone(models["XGBoost"]))
        ],
        voting="soft"
    )

    ensemble.fit(X_train, y_train)

    models["Ensemble"] = ensemble

    return models


def save_models(models):
    """Save trained models to the models directory."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    filenames = {
        "Random Forest": "rf_model.pkl",
        "XGBoost": "xgb_model.pkl",
        "Ensemble": "ensemble_model.pkl"
    }

    for name, model in models.items():

        joblib.dump(
            model,
            MODEL_DIR / filenames[name]
        )

        print(f"Saved {name} model.")


def train_all_models(X_train, y_train, categorical_cols):
    """Execute the complete model training workflow."""

    best_models, best_parameters = tune_models(
        X_train,
        y_train,
        categorical_cols
    )

    thresholds = tune_thresholds(
        best_models,
        X_train,
        y_train
    )

    models = train_final_models(
        best_models,
        X_train,
        y_train
    )

    save_models(models)

    return models, thresholds, best_parameters
