
import pandas as pd

# Load transaction dataset
transactions = pd.read_csv("data/raw/train_transaction.csv")

# Load identity dataset
identity = pd.read_csv("data/raw/train_identity.csv")

# Merge identity data without losing transactions
data = transactions.merge(
    identity,
    on="TransactionID",
    how="left",
    validate="one_to_one"
)

print("Dataset shape:", data.shape)

# Target and identifier columns
target = "isFraud"
excluded_columns = ["TransactionID", target]

# Inspect feature cardinality
print("\n--- FEATURE CARDINALITY ---")

feature_summary = []

for column in data.columns:
    if column in excluded_columns:
        continue

    feature_summary.append({
        "feature": column,
        "dtype": str(data[column].dtype),
        "unique_values": data[column].nunique(),
        "missing_percentage": (
            data[column].isna().mean() * 100
        )
    })

feature_summary = pd.DataFrame(feature_summary)

feature_summary = feature_summary.sort_values(
    "unique_values",
    ascending=False
)

print("\nTop 30 features by unique values:")
print(feature_summary.head(30).to_string(index=False))

# Inspect categorical columns
categorical_features = data.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nCategorical features:", len(categorical_features))

# Inspect numerical columns
numerical_features = data.select_dtypes(
    include=["number"]
).columns.tolist()

print("Numerical features:", len(numerical_features))

# Identify low, medium and high cardinality features
print("\n--- CARDINALITY GROUPS ---")

print(
    "Low cardinality (<= 10):",
    (feature_summary["unique_values"] <= 10).sum()
)

print(
    "Medium cardinality (11-100):",
    (
        (feature_summary["unique_values"] > 10)
        & (feature_summary["unique_values"] <= 100)
    ).sum()
)

print(
    "High cardinality (> 100):",
    (feature_summary["unique_values"] > 100).sum()
)
