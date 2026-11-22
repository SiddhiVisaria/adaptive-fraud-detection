
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from src.data_split import chronological_split
from src.data_preprocessing import prepare_features
from src.evaluation import evaluate_model

# 1. Load and sort transaction data
transactions = pd.read_csv("data/raw/train_transaction.csv")

# 2. Chronological split
train, validation, test = chronological_split(transactions)

print("Training rows:", len(train))
print("Training fraud:", train["isFraud"].sum())

print("Validation rows:", len(validation))
print("Validation fraud:", validation["isFraud"].sum())

print(
    "Validation fraud prevalence:",
    validation["isFraud"].mean()
)

# 3. Prepare transaction-only features
X_train, y_train, preprocessor = prepare_features(
    train,
    include_identity=False
)

X_validation, y_validation, _ = prepare_features(
    validation,
    include_identity=False
)


# 4. Fit preprocessing on training data only
X_train_transformed = preprocessor.fit_transform(X_train)

X_validation_transformed = preprocessor.transform(X_validation)


# 5. Train Logistic Regression
model = LogisticRegression(
    class_weight="balanced",
    solver="lbfgs",
    max_iter=2000,
    random_state=42,
    tol=1e-4
)

model.fit(X_train_transformed, y_train)


# 6. Generate validation predictions
probabilities = model.predict_proba(
    X_validation_transformed
)[:, 1]

predictions = model.predict(X_validation_transformed)

# 7. Evaluate
metrics, cm, report = evaluate_model(
    y_true=y_validation,
    predictions=predictions,
    probabilities=probabilities,
    model_name="Logistic Regression",
)

print("\nIterations:", model.n_iter_)
print("Converged:", model.n_iter_[0] < model.max_iter)

# 8. Save trained model and preprocessor

artifacts_dir = Path("artifacts/logistic_regression")
artifacts_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    model,
    artifacts_dir / "model.joblib"
)

joblib.dump(
    preprocessor,
    artifacts_dir / "preprocessor.joblib"
)

print("\nModel and preprocessor saved successfully.")