import pandas as pd


def load_transaction_identity_data(
    transaction_path="data/raw/train_transaction.csv",
    identity_path="data/raw/train_identity.csv",
):
    """
    Load transaction and identity datasets and perform a validated merge.

    Returns
    -------
    pandas.DataFrame
        Merged dataset with a has_identity indicator.
    """

    transactions = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)

    transactions_with_identity = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    transactions_with_identity["has_identity"] = (
        transactions_with_identity["_merge"] == "both"
    ).astype(int)

    if len(transactions_with_identity) != len(transactions):
        raise ValueError(
            "Identity merge changed the number of transaction rows."
        )

    return transactions_with_identity