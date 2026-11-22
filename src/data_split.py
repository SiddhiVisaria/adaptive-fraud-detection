import pandas as pd


def chronological_split(
    data,
    train_ratio=0.60,
    validation_ratio=0.20,
    time_column="TransactionDT",
):
    """
    Split data chronologically into train, validation, and test sets.

    Timestamp ties are kept together to prevent transactions with the
    same timestamp from being split across partitions.
    """

    data = data.sort_values(time_column).reset_index(drop=True)

    train_end = data[time_column].iloc[
        int(len(data) * train_ratio) - 1
    ]

    validation_end = data[time_column].iloc[
        int(len(data) * (train_ratio + validation_ratio)) - 1
    ]

    train = data[
        data[time_column] <= train_end
    ].copy()

    validation = data[
        (data[time_column] > train_end)
        & (data[time_column] <= validation_end)
    ].copy()

    test = data[
        data[time_column] > validation_end
    ].copy()

    return train, validation, test