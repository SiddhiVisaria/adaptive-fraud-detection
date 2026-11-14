import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

# Load transaction data
transactions = pd.read_csv("data/raw/train_transaction.csv")

# Sort chronologically
transactions = transactions.sort_values(
    "TransactionDT"
).reset_index(drop=True)

# Recreate chronological boundaries
train_end = transactions["TransactionDT"].iloc[
    int(len(transactions) * 0.60)
]

validation_end = transactions["TransactionDT"].iloc[
    int(len(transactions) * 0.80)
]

train = transactions[
    transactions["TransactionDT"] <= train_end
]

validation = transactions[
    (transactions["TransactionDT"] > train_end)
    & (transactions["TransactionDT"] <= validation_end)
]

X_train = train.drop(
    columns=["isFraud", "TransactionID", "TransactionDT"]
)
y_train = train["isFraud"]

X_validation = validation.drop(
    columns=["isFraud", "TransactionID", "TransactionDT"]
)
y_validation = validation["isFraud"]

# Establish a majority-class benchmark
dummy = DummyClassifier(strategy="most_frequent")

dummy.fit(X_train, y_train)

predictions = dummy.predict(X_validation)
probabilities = dummy.predict_proba(X_validation)[:, 1]

print("Confusion Matrix:")
print(confusion_matrix(y_validation, predictions))

print("\nClassification Report:")
print(classification_report(
    y_validation,
    predictions,
    zero_division=0
))

print("PR-AUC:", average_precision_score(
    y_validation, probabilities
))

print("ROC-AUC:", roc_auc_score(
    y_validation, probabilities
))