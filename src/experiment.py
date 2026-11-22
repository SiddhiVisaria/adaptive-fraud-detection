from src.data_loading import load_transaction_identity_data
from src.data_split import chronological_split
from src.data_preprocessing import prepare_features
from sklearn.linear_model import LogisticRegression
from src.evaluation import evaluate_model

def load_and_split_data():
    """
    Load transaction and identity data, then split chronologically
    into training, validation, and test datasets.
    """

    data = load_transaction_identity_data()

    train, validation, test = chronological_split(data)

    return train, validation, test

def get_feature_config(experiment):
    """
    Return feature configuration for a given experiment.
    """

    configs = {
        "A": {
            "include_identity": False,
            "include_identity_indicator": False,
        },
        "B": {
            "include_identity": True,
            "include_identity_indicator": False,
        },
        "C": {
            "include_identity": True,
            "include_identity_indicator": True,
        },
    }

    if experiment not in configs:
        raise ValueError(f"Unknown experiment: {experiment}")

    return configs[experiment]

def prepare_experiment_data(train, validation, experiment):
    """
    Prepare training and validation features for a given experiment.
    """

    config = get_feature_config(experiment)

    X_train, y_train, preprocessor = prepare_features(
        train,
        **config,
    )

    X_validation, y_validation, _ = prepare_features(
        validation,
        **config,
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        preprocessor,
    )

def preprocess_experiment_data(
    X_train,
    X_validation,
    preprocessor,
):
    """
    Fit preprocessing on training data and transform
    training and validation data.
    """

    preprocessor.fit(X_train)

    X_train_processed = preprocessor.transform(X_train)
    X_validation_processed = preprocessor.transform(X_validation)

    return X_train_processed, X_validation_processed

def train_model(X_train_processed, y_train):
    """
    Train Logistic Regression using the processed training data.
    """

    model = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
        tol=1e-4,
    )

    model.fit(X_train_processed, y_train)

    return model

def evaluate_experiment(model, X_validation_processed, y_validation, experiment):
    """
    Evaluate a trained model on validation data.
    """

    predictions = model.predict(X_validation_processed)
    probabilities = model.predict_proba(X_validation_processed)[:, 1]

    metrics, confusion_matrix, report = evaluate_model(
        y_validation,
        predictions,
        probabilities,
        model_name=f"Experiment {experiment}",
    )

    return metrics, confusion_matrix, report

def run_experiment(experiment):
    """
    Run a complete feature experiment:
    load data, split, prepare features, preprocess,
    train, and evaluate.
    """

    print(f"\n{'=' * 60}")
    print(f"RUNNING EXPERIMENT {experiment}")
    print(f"{'=' * 60}")

    # 1. Load data and split chronologically
    train, validation, test = load_and_split_data()

    # 2. Prepare features and labels
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        preprocessor,
    ) = prepare_experiment_data(
        train,
        validation,
        experiment,
    )

    # 3. Preprocess features
    (
        X_train_processed,
        X_validation_processed,
    ) = preprocess_experiment_data(
        X_train,
        X_validation,
        preprocessor,
    )

    # 4. Train model
    model = train_model(
        X_train_processed,
        y_train,
    )

    # 5. Evaluate on validation data
    metrics, confusion_matrix, report = evaluate_experiment(
        model,
        X_validation_processed,
        y_validation,
        experiment,
    )

    # 6. Return all relevant outputs
    return {
        "experiment": experiment,
        "model": model,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix,
        "classification_report": report,
    }