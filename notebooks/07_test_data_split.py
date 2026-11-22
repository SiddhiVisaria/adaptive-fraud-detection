import pandas as pd

from src.data_split import chronological_split


# Load dataset
transactions = pd.read_csv(
    "data/raw/train_transaction.csv"
)

# Apply chronological split
train, validation, test = chronological_split(
    transactions
)


# Print partition statistics
for name, dataset in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:
    print(f"\n{name}")
    print("-" * 30)
    print("Rows:", len(dataset))
    print("Fraud cases:", dataset["isFraud"].sum())
    print("Fraud rate:", dataset["isFraud"].mean())
    print(
        "Start TransactionDT:",
        dataset["TransactionDT"].min()
    )
    print(
        "End TransactionDT:",
        dataset["TransactionDT"].max()
    )


# Verify split integrity
assert len(train) + len(validation) + len(test) == len(transactions)

assert train["TransactionDT"].max() < validation["TransactionDT"].min()
assert validation["TransactionDT"].max() < test["TransactionDT"].min()

assert train["isFraud"].nunique() == 2
assert validation["isFraud"].nunique() == 2
assert test["isFraud"].nunique() == 2

print("\nAll split integrity checks passed.")