
import pandas as pd

# Load transaction data
transactions = pd.read_csv("data/raw/train_transaction.csv")

# Sort chronologically
transactions = transactions.sort_values(
    "TransactionDT"
).reset_index(drop=True)

# Recreate the original timestamp boundaries.
# Use the last row of each target partition as the boundary.
train_end = transactions["TransactionDT"].iloc[
    int(len(transactions) * 0.60) - 1
]

validation_end = transactions["TransactionDT"].iloc[
    int(len(transactions) * 0.80) - 1
]

# Assign partitions without splitting identical timestamps
train = transactions[
    transactions["TransactionDT"] <= train_end
]

validation = transactions[
    (transactions["TransactionDT"] > train_end)
    & (transactions["TransactionDT"] <= validation_end)
]

test = transactions[
    transactions["TransactionDT"] > validation_end
]

# Report partition sizes and fraud prevalence
for name, partition in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:
    print(f"\n--- {name} ---")
    print("Rows:", len(partition))
    print("Fraud cases:", int(partition["isFraud"].sum()))
    print(
        "Fraud prevalence:",
        round(partition["isFraud"].mean() * 100, 3),
        "%"
    )
    print(
        "TransactionDT range:",
        partition["TransactionDT"].min(),
        "to",
        partition["TransactionDT"].max()
    )

# Validate chronological ordering and row coverage
print("\n--- SPLIT VALIDATION ---")

print("All rows accounted for:",
      len(train) + len(validation) + len(test) == len(transactions))

print("Train before validation:",
      train["TransactionDT"].max() < validation["TransactionDT"].min())

print("Validation before test:",
      validation["TransactionDT"].max() < test["TransactionDT"].min())

print("Train size matches previous split:", len(train) == 354324)
print("Validation size matches previous split:", len(validation) == 118108)
print("Test size matches previous split:", len(test) == 118108)
