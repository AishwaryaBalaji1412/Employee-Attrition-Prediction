
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = PROJECT_ROOT / "outputs" / "plots"


def transform_test_data(model, X_test):
    """Apply the fitted preprocessing step to test data."""

    preprocessor = model.named_steps["preprocessor"]

    transformed = preprocessor.transform(X_test)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    return pd.DataFrame(
        transformed,
        columns=feature_names,
        index=X_test.index
    )


def get_positive_class_shap_values(explainer, X_data):
    """Extract SHAP values for the positive class (Attrition = 1)."""

    output = explainer.shap_values(X_data)

    if isinstance(output, list):
        values = np.asarray(output[1])
    else:
        values = np.asarray(
            output.values if hasattr(output, "values") else output
        )

        if values.ndim == 3:
            if values.shape[:2] == (
                len(X_data), X_data.shape[1]
            ):
                values = values[:, :, 1]

            elif values.shape[1:] == (
                len(X_data), X_data.shape[1]
            ):
                values = values[1, :, :]

    if values.shape != (len(X_data), X_data.shape[1]):
        raise ValueError(
            f"Unexpected SHAP shape: {values.shape}"
        )

    return values


def explain_model(model, X_test, model_name, max_display=15):
    """Generate SHAP feature importance and summary plots."""

    classifier = model.named_steps["classifier"]

    X_transformed = transform_test_data(model, X_test)

    explainer = shap.TreeExplainer(classifier)

    shap_values = get_positive_class_shap_values(
        explainer,
        X_transformed
    )
    max_display = min(max_display, X_transformed.shape[1])

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    figures = {}

    plot_configs = [
        ("bar", "Feature Importance"),
        ("dot", "SHAP Summary")
    ]

    for plot_type, title in plot_configs:

        plt.figure(figsize=(10, 6))

        shap.summary_plot(
            shap_values,
            X_transformed,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )

        plt.title(f"{model_name} - {title}")
        plt.tight_layout()

        filename = (
            f"{model_name.lower().replace(' ', '_')}_"
            f"shap_{plot_type}.png"
        )

        plt.savefig(
            PLOT_DIR / filename,
            dpi=300,
            bbox_inches="tight"
        )

        figures[plot_type] = plt.gcf()
        plt.close(figures[plot_type])

    return figures


def explain_all_models(models, X_test):
    """Generate SHAP explanations for Random Forest and XGBoost."""

    explanations = {}

    for name in ["Random Forest", "XGBoost"]:

        if name not in models:
            continue

        print(f"\nGenerating SHAP explanations for {name}...")

        explanations[name] = explain_model(
            models[name],
            X_test,
            name
        )

    return explanations
