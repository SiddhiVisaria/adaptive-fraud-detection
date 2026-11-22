from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_model(y_true, predictions, probabilities, model_name="Model"):
    """
    Evaluate a binary classification model.

    Returns:
        metrics: dictionary containing evaluation metrics
        cm: confusion matrix
        report: classification report dictionary
    """

    metrics = {
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),
    }

    cm = confusion_matrix(y_true, predictions)

    report = classification_report(
        y_true,
        predictions,
        zero_division=0,
        output_dict=True,
    )

    print(f"\n--- {model_name.upper()} RESULTS ---")

    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1-score:", metrics["f1"])
    print("PR-AUC:", metrics["pr_auc"])
    print("ROC-AUC:", metrics["roc_auc"])

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0,
        )
    )

    return metrics, cm, report