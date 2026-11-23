
import json
from pathlib import Path

from src.data_preprocessing import prepare_data
from src.exploratory_analysis import generate_eda_plots
from src.model_training import train_all_models
from src.model_evaluation import evaluate_all_models
from src.model_explainability import explain_all_models


PROJECT_ROOT = Path(__file__).resolve().parent
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


def make_json_serializable(value):
    """Convert NumPy values and nested structures into JSON-safe values."""
    if isinstance(value, dict):
        return {
            key: make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]

    if hasattr(value, "item"):
        return value.item()

    return value


def main():
    print("=" * 60)
    print("EMPLOYEE ATTRITION PREDICTION - MODEL TRAINING")
    print("=" * 60)

    print("\n1. Loading and preprocessing dataset...")

    (
        df, X, y, categorical_cols,
        X_train, X_test, y_train, y_test
    ) = prepare_data()

    print(f"Dataset shape: {df.shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Categorical features: {len(categorical_cols)}")

    print("\n2. Generating EDA visualizations...")
    generate_eda_plots(df, categorical_cols)

    print("\n3. Training machine learning models...")

    models, thresholds, best_parameters = train_all_models(
        X_train,
        y_train,
        categorical_cols
    )

    print("\nOptimized thresholds:")
    for name, threshold in thresholds.items():
        print(f"{name}: {threshold:.4f}")

    print("\n4. Evaluating trained models...")

    evaluate_all_models(
        models,
        X_test,
        y_test,
        thresholds
    )

    print("\n5. Generating SHAP explanations...")
    explain_all_models(models, X_test)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n6. Saving optimized thresholds...")

    with open(METRICS_DIR / "thresholds.json", "w") as file:
        json.dump(
            make_json_serializable(thresholds),
            file,
            indent=4
        )

    print("\n7. Saving best model parameters...")

    with open(METRICS_DIR / "best_parameters.json", "w") as file:
        json.dump(
            make_json_serializable(best_parameters),
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("TRAINING WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("\nTrained models:")
    for name in models:
        print(f"- {name}")

    print("\nSaved files:")
    print("- Models: models/")
    print("- Thresholds: outputs/metrics/thresholds.json")
    print("- Best parameters: outputs/metrics/best_parameters.json")
    print("- Evaluation results: outputs/metrics/")
    print("- Visualizations: outputs/plots/")


if __name__ == "__main__":
    main()
