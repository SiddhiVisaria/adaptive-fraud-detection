from src.data_loading import load_transaction_identity_data


# Load merged dataset
data = load_transaction_identity_data()


# Basic dataset checks
print("Merged dataset shape:", data.shape)

print(
    "Unique TransactionIDs:",
    data["TransactionID"].nunique()
)

print(
    "Duplicate TransactionIDs:",
    data["TransactionID"].duplicated().sum()
)

print("\nIdentity availability:")
print(data["has_identity"].value_counts())

print(
    "\nIdentity coverage:",
    round(data["has_identity"].mean() * 100, 2),
    "%"
)


# Verify data integrity
assert data["TransactionID"].is_unique

assert len(data) == 590540

assert data["has_identity"].isin([0, 1]).all()

assert data["isFraud"].nunique() == 2

print("\nAll data loading checks passed.")