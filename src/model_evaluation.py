
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLOT_DIR = PROJECT_ROOT / "outputs" / "plots"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate a model using classification metrics."""

    y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= threshold).astype(int)

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1-score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Average Precision": average_precision_score(
            y_test, y_prob
        )
    }

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    return {
        "metrics": results,
        "classification_report": report,
        "confusion_matrix": matrix,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "threshold": threshold
    }


def plot_confusion_matrix(
    matrix,
    model_name,
    threshold_type="Default"
):
    """Plot and save the confusion matrix."""

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stayed", "Left"],
        yticklabels=["Stayed", "Left"],
        ax=ax
    )

    ax.set_title(
        f"{model_name} - {threshold_type} Confusion Matrix"
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    fig.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{model_name.lower().replace(' ', '_')}_"
        f"{threshold_type.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    fig.savefig(
        PLOT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_roc_curve(y_test, y_prob, model_name):
    """Plot and save the ROC curve."""

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc:.3f})"
    )

    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_title(f"{model_name} ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    fig.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        PLOT_DIR / f"{model_name.lower().replace(' ', '_')}_roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_precision_recall_curve(y_test, y_prob, model_name):
    """Plot and save the precision-recall curve."""

    precision, recall, _ = precision_recall_curve(
        y_test,
        y_prob
    )

    average_precision = average_precision_score(
        y_test,
        y_prob
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        recall,
        precision,
        label=f"AP = {average_precision:.3f}"
    )

    ax.set_title(f"{model_name} Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()

    fig.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        PLOT_DIR / f"{model_name.lower().replace(' ', '_')}_pr_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_evaluation_results(all_results):
    """Save model comparison results as CSV and JSON."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_rows = []

    for model_name, evaluations in all_results.items():

        for threshold_type, result in evaluations.items():

            row = {
                "Model": model_name,
                "Threshold Type": threshold_type,
                "Threshold": result["threshold"]
            }

            row.update(result["metrics"])

            comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df.to_csv(
        METRICS_DIR / "model_comparison.csv",
        index=False
    )

    with open(
        METRICS_DIR / "model_comparison.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            comparison_rows,
            file,
            indent=4
        )

    print("\nModel comparison saved successfully.")
    print(comparison_df.to_string(index=False))


def evaluate_all_models(models, X_test, y_test, thresholds):
    """Evaluate every model using default and tuned thresholds."""

    all_results = {}

    for name, model in models.items():

        all_results[name] = {}

        for threshold_type, threshold in [
            ("Default", 0.5),
            ("Tuned", thresholds[name])
        ]:

            result = evaluate_model(
                model,
                X_test,
                y_test,
                threshold
            )

            all_results[name][threshold_type] = result

            print(f"\n{name} - {threshold_type} Threshold")

            for metric, value in result["metrics"].items():
                print(f"{metric}: {value:.4f}")

            print("\nClassification Report:")
            print(result["classification_report"])

            plot_confusion_matrix(
                result["confusion_matrix"],
                name,
                threshold_type
            )

        # ROC and PR curves use probabilities,
        # so they do not depend on the classification threshold.

        y_prob = all_results[name]["Default"]["y_prob"]

        plot_roc_curve(
            y_test,
            y_prob,
            name
        )

        plot_precision_recall_curve(
            y_test,
            y_prob,
            name
        )

    save_evaluation_results(all_results)

    return all_results
