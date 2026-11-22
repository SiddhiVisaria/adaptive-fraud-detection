import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

from src.experiment import (
    load_and_split_data,
    prepare_experiment_data,
    preprocess_experiment_data,
    train_model,
)


# ============================================================
# 1. LOAD DATA AND CHRONOLOGICAL SPLIT
# ============================================================

train, validation, test = load_and_split_data()

print("Dataset sizes:")
print(f"Train:      {len(train):,}")
print(f"Validation: {len(validation):,}")
print(f"Test:       {len(test):,}")


# ============================================================
# 2. PREPARE EXPERIMENT A: TRANSACTION FEATURES ONLY
# ============================================================

(
    X_train,
    y_train,
    X_validation,
    y_validation,
    preprocessor,
) = prepare_experiment_data(
    train,
    validation,
    experiment="A",
)


# ============================================================
# 3. PREPROCESS
# ============================================================

(
    X_train_processed,
    X_validation_processed,
) = preprocess_experiment_data(
    X_train,
    X_validation,
    preprocessor,
)


# ============================================================
# 4. TRAIN STATIC MODEL ONCE
# ============================================================

model = train_model(
    X_train_processed,
    y_train,
)

print("\nStatic Experiment A model trained.")


# ============================================================
# 5. GENERATE VALIDATION PREDICTIONS
# ============================================================

validation_predictions = model.predict(
    X_validation_processed
)

validation_probabilities = model.predict_proba(
    X_validation_processed
)[:, 1]


# ============================================================
# 6. CREATE CHRONOLOGICAL WINDOWS
# ============================================================

time_column = "TransactionDT"

validation_start = validation[time_column].min()

window_seconds = 7 * 24 * 60 * 60

validation = validation.copy()

validation["time_window"] = (
    (validation[time_column] - validation_start)
    // window_seconds
).astype(int)


# ============================================================
# 7. EVALUATE MODEL PERFORMANCE BY WINDOW
# ============================================================

results = []

for window_number in sorted(validation["time_window"].unique()):

    mask = validation["time_window"] == window_number

    y_true = y_validation.loc[mask]

    y_pred = validation_predictions[mask]
    y_prob = validation_probabilities[mask]

    window_data = validation.loc[mask]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "time_window": window_number,
        "transactions": len(window_data),
        "fraud_cases": int(y_true.sum()),
        "fraud_rate": y_true.mean(),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "pr_auc": average_precision_score(
            y_true,
            y_prob,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_prob,
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "start_time": window_data[time_column].min(),
        "end_time": window_data[time_column].max(),
    }

    results.append(metrics)


# ============================================================
# 8. CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("STATIC MODEL — TEMPORAL VALIDATION PERFORMANCE")
print("=" * 80)

print(
    results_df[
        [
            "time_window",
            "transactions",
            "fraud_cases",
            "fraud_rate",
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "roc_auc",
            "false_positives",
            "false_negatives",
        ]
    ].to_string(index=False)
)


# ============================================================
# 9. VALIDATION CHECKS
# ============================================================

assert results_df["transactions"].sum() == len(validation)

assert results_df["fraud_cases"].sum() == validation["isFraud"].sum()

assert len(results_df) == validation["time_window"].nunique()

print("\nAll temporal evaluation checks passed.")