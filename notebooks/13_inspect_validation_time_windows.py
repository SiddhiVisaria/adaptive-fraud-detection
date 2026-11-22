import pandas as pd

from src.data_loading import load_transaction_identity_data
from src.data_split import chronological_split


# ============================================================
# LOAD DATA AND SPLIT
# ============================================================

data = load_transaction_identity_data()

train, validation, test = chronological_split(data)


# ============================================================
# INSPECT VALIDATION TIME RANGE
# ============================================================

time_column = "TransactionDT"

validation_start = validation[time_column].min()
validation_end = validation[time_column].max()

duration_days = (
    validation_end - validation_start
) / (24 * 60 * 60)

print("Validation time range")
print(f"Start: {validation_start}")
print(f"End:   {validation_end}")
print(f"Duration: {duration_days:.2f} days")


# ============================================================
# CREATE 7-DAY TIME WINDOWS
# ============================================================

window_seconds = 7 * 24 * 60 * 60

validation = validation.copy()

validation["time_window"] = (
    (validation[time_column] - validation_start)
    // window_seconds
).astype(int)


# ============================================================
# SUMMARISE EACH WINDOW
# ============================================================

summary = (
    validation.groupby("time_window")
    .agg(
        transactions=("TransactionID", "count"),
        fraud_cases=("isFraud", "sum"),
        start_time=(time_column, "min"),
        end_time=(time_column, "max"),
    )
)

summary["fraud_rate"] = (
    summary["fraud_cases"] / summary["transactions"]
)

summary["duration_days"] = (
    summary["end_time"] - summary["start_time"]
) / (24 * 60 * 60)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nValidation time-window summary:")
print(summary.to_string())

print("\nTotal validation transactions:", len(validation))
print("Total validation fraud cases:", validation["isFraud"].sum())

assert summary["transactions"].sum() == len(validation)
assert summary["fraud_cases"].sum() == validation["isFraud"].sum()

print("\nTime-window inspection checks passed.")