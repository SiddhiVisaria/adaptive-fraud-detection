
import pandas as pd
import sys
from src.data_preprocessing import prepare_features

transactions = pd.read_csv(
    "data/raw/train_transaction.csv"
)

identity = pd.read_csv(
    "data/raw/train_identity.csv"
)

data = transactions.merge(
    identity,
    on="TransactionID",
    how="left",
    validate="one_to_one",
    indicator=True
)

data["has_identity"] = (
    data["_merge"] == "both"
).astype(int)


experiments = {
    "Transaction only": {
        "include_identity": False,
        "include_identity_indicator": False
    },
    "Transaction + identity": {
        "include_identity": True,
        "include_identity_indicator": False
    },
    "Transaction + identity + indicator": {
        "include_identity": True,
        "include_identity_indicator": True
    }
}

for name, settings in experiments.items():

    X, y, preprocessor = prepare_features(
        data,
        **settings
    )

    print(f"\n--- {name} ---")
    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("Fraud cases:", y.sum())
    print("Categorical features:",
          len(X.select_dtypes(
              include=["object", "category", "string"]
          ).columns))
    print("Numerical features:",
          len(X.select_dtypes(
              include=["number"]
          ).columns))

    print("TransactionID excluded:", "TransactionID" not in X.columns)
    print("Target excluded:", "isFraud" not in X.columns)
    print("TransactionDT excluded:", "TransactionDT" not in X.columns)
    print("_merge excluded:", "_merge" not in X.columns)
# Fit preprocessing on a small sample first
sample_size = 10000

X_sample = X.iloc[:sample_size]
y_sample = y.iloc[:sample_size]

X_transformed = preprocessor.fit_transform(X_sample, y_sample)

print("Transformed shape:", X_transformed.shape)
print("Sparse matrix:", hasattr(X_transformed, "toarray"))
print("Output data type:", X_transformed.dtype)

if hasattr(X_transformed, "data"):
    memory_bytes = (
        X_transformed.data.nbytes
        + X_transformed.indices.nbytes
        + X_transformed.indptr.nbytes
    )
else:
    memory_bytes = X_transformed.nbytes

memory_mb = memory_bytes / (1024 ** 2)

print(f"Transformed matrix memory: {memory_mb:.2f} MB")