
import pandas as pd
from pathlib import Path

# Dataset locations
DATA_DIR = Path("data/raw")

transaction_file = DATA_DIR / "train_transaction.csv"
identity_file = DATA_DIR / "train_identity.csv"

# Load datasets
transactions = pd.read_csv(transaction_file)
identity = pd.read_csv(identity_file)

# 1. Dataset dimensions
print("\n--- DATASET DIMENSIONS ---")
print("Transactions:", transactions.shape)
print("Identity:", identity.shape)

# 2. Target distribution
print("\n--- FRAUD DISTRIBUTION ---")
print(transactions["isFraud"].value_counts())
print("\nFraud percentage:")
print(transactions["isFraud"].value_counts(normalize=True) * 100)

# 3. Time coverage
print("\n--- TIME COVERAGE ---")
print("Earliest TransactionDT:", transactions["TransactionDT"].min())
print("Latest TransactionDT:", transactions["TransactionDT"].max())

# 4. Missing values
print("\n--- MISSING VALUES: TRANSACTIONS ---")
missing = transactions.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False).head(20))

print("\n--- MISSING VALUES: IDENTITY ---")
missing_identity = identity.isnull().sum()
print(missing_identity[missing_identity > 0].sort_values(ascending=False).head(20))

# 5. Feature overview
print("\n--- COLUMN INFORMATION ---")
print(transactions.dtypes.value_counts())
print("\nTransaction columns:", len(transactions.columns))
print("Identity columns:", len(identity.columns))

print("\n--- FRAUD DISTRIBUTION OVER TIME ---")

transactions["period"] = (
    (transactions["TransactionDT"] - transactions["TransactionDT"].min())
    // (14 * 24 * 60 * 60)
) + 1

temporal_analysis = transactions.groupby("period").agg(
    total_transactions=("isFraud", "count"),
    fraud_cases=("isFraud", "sum"),
    start_time=("TransactionDT", "min"),
    end_time=("TransactionDT", "max")
)

temporal_analysis["fraud_rate_percent"] = (
    temporal_analysis["fraud_cases"]
    / temporal_analysis["total_transactions"]
) * 100

print(
    temporal_analysis[
        [
            "total_transactions",
            "fraud_cases",
            "fraud_rate_percent",
            "start_time",
            "end_time"
        ]
    ].round(3).to_string()
)

print("\n--- TRANSACTION AMOUNT OVER TIME ---")

amount_analysis = transactions.groupby("period").agg(
    median_amount=("TransactionAmt", "median"),
    average_amount=("TransactionAmt", "mean"),
    fraud_median_amount=(
        "TransactionAmt",
        lambda x: x[transactions.loc[x.index, "isFraud"] == 1].median()
    ),
    legitimate_median_amount=(
        "TransactionAmt",
        lambda x: x[transactions.loc[x.index, "isFraud"] == 0].median()
    )
)

print(amount_analysis.round(2).to_string())

print("\n--- FRAUD RATE BY TRANSACTION AMOUNT BAND ---")

transactions["amount_band"] = pd.qcut(
    transactions["TransactionAmt"],
    q=5,
    duplicates="drop"
)

amount_band_analysis = (
    transactions.groupby("amount_band", observed=False)
    .agg(
        total_transactions=("isFraud", "size"),
        fraud_cases=("isFraud", "sum"),
        fraud_rate=("isFraud", "mean")
    )
)

amount_band_analysis["fraud_rate_percent"] = (
    amount_band_analysis["fraud_rate"] * 100
)

print(
    amount_band_analysis[
        ["total_transactions", "fraud_cases", "fraud_rate_percent"]
    ].round(3).to_string()
)

print("\n--- CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT ---")

# Sort transactions by elapsed transaction time
sorted_transactions = transactions.sort_values(
    "TransactionDT"
).reset_index(drop=True)

total_rows = len(sorted_transactions)

# Calculate proposed split positions
train_position = int(total_rows * 0.60)
validation_position = int(total_rows * 0.80)

# Use timestamp boundaries so identical timestamps stay together
train_end_time = sorted_transactions.loc[
    train_position, "TransactionDT"
]

validation_end_time = sorted_transactions.loc[
    validation_position, "TransactionDT"
]

# Create chronological partitions
train_data = sorted_transactions[
    sorted_transactions["TransactionDT"] < train_end_time
]

validation_data = sorted_transactions[
    (sorted_transactions["TransactionDT"] >= train_end_time)
    & (sorted_transactions["TransactionDT"] < validation_end_time)
]

test_data = sorted_transactions[
    sorted_transactions["TransactionDT"] >= validation_end_time
]

# Summarise each partition
for name, data in [
    ("TRAIN", train_data),
    ("VALIDATION", validation_data),
    ("TEST", test_data)
]:
    print(f"\n{name}")
    print("Transactions:", len(data))
    print("Fraud cases:", int(data["isFraud"].sum()))
    print("Fraud rate:", round(data["isFraud"].mean() * 100, 3), "%")
    print("Start time:", data["TransactionDT"].min())
    print("End time:", data["TransactionDT"].max())

print("\nTotal partitioned transactions:",
      len(train_data) + len(validation_data) + len(test_data))
print("Original transactions:", total_rows)

print("\n--- SPLIT VALIDATION ---")

# Check that all records have been allocated
print("All records accounted for:",
      len(train_data) + len(validation_data) + len(test_data)
      == len(transactions))

# Check chronological ordering
print("Training before validation:",
      train_data["TransactionDT"].max()
      < validation_data["TransactionDT"].min())

print("Validation before testing:",
      validation_data["TransactionDT"].max()
      < test_data["TransactionDT"].min())

# Check that all three partitions contain both classes
for name, data in [
    ("TRAIN", train_data),
    ("VALIDATION", validation_data),
    ("TEST", test_data)
]:
    print(
        f"{name}:",
        data["isFraud"].nunique(),
        "classes present"
    )
# --------------------------------------------------
# FEATURE AND DATA QUALITY INSPECTION
# --------------------------------------------------

print("\n--- FEATURE AND DATA QUALITY INSPECTION ---")

# 1. Identify identifier and target columns
print("\nIdentifier and target columns:")
print([
    column for column in transactions.columns
    if column.lower() in ["transactionid", "isfraud"]
])

# 2. Inspect categorical/object columns
categorical_columns = transactions.select_dtypes(
    include=["object"]
).columns

print("\nCategorical columns:")
for column in categorical_columns:
    print(
        column,
        "| Unique values:", transactions[column].nunique(),
        "| Missing:", transactions[column].isna().sum()
    )

# 3. Inspect columns with high missingness
missing_percentage = (
    transactions.isnull().mean() * 100
).sort_values(ascending=False)

print("\nTop 20 columns by missing percentage:")
print(missing_percentage.head(20).round(2))

# 4. Identify constant columns
constant_columns = [
    column for column in transactions.columns
    if transactions[column].nunique(dropna=False) <= 1
]

print("\nConstant columns:")
print(constant_columns)

# 5. Check duplicate transaction IDs
print("\nDuplicate TransactionID count:")
print(transactions["TransactionID"].duplicated().sum())

# --------------------------------------------------
# IDENTITY DATA AND FEATURE STRUCTURE
# --------------------------------------------------

print("\n--- IDENTITY DATA INSPECTION ---")

# Check transaction ID uniqueness in identity data
print(
    "Duplicate identity TransactionIDs:",
    identity["TransactionID"].duplicated().sum()
)

# Identity dataset coverage
identity_ids = set(identity["TransactionID"])
transaction_ids = set(transactions["TransactionID"])

matched_ids = transaction_ids.intersection(identity_ids)

print(
    "Transactions with identity records:",
    len(matched_ids)
)
print(
    "Identity coverage (%):",
    round(len(matched_ids) / len(transaction_ids) * 100, 2)
)

# Inspect identity columns
print("\nIdentity columns:")
for column in identity.columns:
    if column != "TransactionID":
        print(
            column,
            "| dtype:", identity[column].dtype,
            "| unique:", identity[column].nunique(),
            "| missing:", identity[column].isna().sum()
        )

# Check whether identity records have matching transactions
print(
    "\nIdentity records without matching transactions:",
    len(identity_ids - transaction_ids)
)

# Inspect transaction feature groups
print("\n--- TRANSACTION FEATURE GROUPS ---")

feature_groups = {
    "Card": ["card1", "card2", "card3", "card4", "card5", "card6"],
    "Address": ["addr1", "addr2"],
    "Email": ["P_emaildomain", "R_emaildomain"],
    "Transaction": ["TransactionAmt", "ProductCD", "TransactionDT"],
    "Distance": ["dist1", "dist2"],
}

for group, columns in feature_groups.items():
    available = [
        column for column in columns
        if column in transactions.columns
    ]
    print(f"{group}: {available}")

# --------------------------------------------------
# IDENTITY DATA JOIN AND AVAILABILITY ANALYSIS
# --------------------------------------------------

print("\n--- IDENTITY DATA JOIN ---")

transactions_with_identity = transactions.merge(
    identity,
    on="TransactionID",
    how="left",
    validate="one_to_one",
    indicator=True
)

# Create an indicator for identity record availability
transactions_with_identity["has_identity"] = (
    transactions_with_identity["_merge"] == "both"
).astype(int)

print("Original transaction shape:", transactions.shape)
print("Joined dataset shape:", transactions_with_identity.shape)

print(
    "Identity records matched:",
    (transactions_with_identity["_merge"] == "both").sum()
)

print(
    "Transactions without identity:",
    (transactions_with_identity["_merge"] == "left_only").sum()
)

print(
    "All transactions preserved:",
    len(transactions_with_identity) == len(transactions)
)

# Identity availability by fraud label
print("\n--- IDENTITY AVAILABILITY BY FRAUD LABEL ---")

identity_by_fraud = (
    transactions_with_identity
    .groupby("isFraud")["has_identity"]
    .agg(["count", "sum", "mean"])
)

identity_by_fraud["availability_percent"] = (
    identity_by_fraud["mean"] * 100
)

print(identity_by_fraud)

# Identity availability over time
print("\n--- IDENTITY AVAILABILITY OVER TIME ---")

identity_by_period = (
    transactions_with_identity
    .groupby("period")
    .agg(
        total_transactions=("TransactionID", "count"),
        transactions_with_identity=("has_identity", "sum"),
        fraud_cases=("isFraud", "sum")
    )
)

identity_by_period["identity_coverage_percent"] = (
    identity_by_period["transactions_with_identity"]
    / identity_by_period["total_transactions"] * 100
)

print(identity_by_period.round(2))

# Confirm join did not introduce duplicate transaction IDs
print(
    "\nDuplicate IDs after join:",
    transactions_with_identity["TransactionID"].duplicated().sum()
)

# --------------------------------------------------
# IDENTITY AVAILABILITY AND FRAUD RELATIONSHIP
# --------------------------------------------------

print("\n--- FRAUD RATE BY IDENTITY AVAILABILITY ---")

availability_analysis = (
    transactions_with_identity
    .groupby("has_identity")
    .agg(
        transactions=("isFraud", "size"),
        fraud_cases=("isFraud", "sum"),
        fraud_rate=("isFraud", "mean")
    )
)

availability_analysis["fraud_rate_percent"] = (
    availability_analysis["fraud_rate"] * 100
)

print(availability_analysis)

print("\n--- FRAUD RATE BY IDENTITY AVAILABILITY AND PERIOD ---")

availability_over_time = (
    transactions_with_identity
    .groupby(["period", "has_identity"])
    .agg(
        transactions=("isFraud", "size"),
        fraud_cases=("isFraud", "sum"),
        fraud_rate=("isFraud", "mean")
    )
    .reset_index()
)

availability_over_time["fraud_rate_percent"] = (
    availability_over_time["fraud_rate"] * 100
)

print(
    availability_over_time[
        [
            "period",
            "has_identity",
            "transactions",
            "fraud_cases",
            "fraud_rate_percent"
        ]
    ].round(3).to_string(index=False)
)

# Check identity availability in our chronological partitions
print("\n--- IDENTITY COVERAGE BY DATA PARTITION ---")

for name, partition in [
    ("TRAIN", train_data),
    ("VALIDATION", validation_data),
    ("TEST", test_data)
]:
    coverage = partition["TransactionID"].isin(identity_ids).mean() * 100
    print(f"{name}: {coverage:.2f}%")
