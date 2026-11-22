import pandas as pd

from sklearn.dummy import DummyClassifier
from src.data_split import chronological_split
from src.evaluation import evaluate_model

# Load transaction data
transactions = pd.read_csv("data/raw/train_transaction.csv")

# Chronological split
train, validation, test = chronological_split(transactions)

print("Training rows:", len(train))
print("Training fraud:", train["isFraud"].sum())

print("Validation rows:", len(validation))
print("Validation fraud:", validation["isFraud"].sum())

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

# Evaluate model
metrics, cm, report = evaluate_model(
    y_true=y_validation,
    predictions=predictions,
    probabilities=probabilities,
    model_name="DummyClassifier",
)