# Experiment Log

## Purpose
Record experiments in a reproducible and consistent manner.

No experiments have been conducted yet.

## Experiment template

### Experiment ID:
### Date:
### Objective:
### Dataset and time period:
### Model:
### Features:
### Training configuration:
### Evaluation method:

### Results
- Precision:
- Recall:
- F1-score:
- PR-AUC:
- Confusion matrix:

### Observations

### Conclusion

### Next action:

---

## Current status

**Experiments completed:** 0

The first experiment will establish a static baseline model after dataset selection and validation.

### Experiment 1 — Majority-Class Dummy Baseline

**Objective:** Establish a minimum-performance benchmark for fraud detection.

**Model:** `DummyClassifier(strategy="most_frequent")`

**Evaluation:** Chronological validation partition.

**Validation results:**
- Transactions: 118,108
- Legitimate: 113,496
- Fraudulent: 4,612
- Accuracy: approximately 96.10%
- Fraud precision: 0.00
- Fraud recall: 0.00
- Fraud F1-score: 0.00
- PR-AUC: 0.03905
- ROC-AUC: 0.500

**Confusion matrix:**

| | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| Actual legitimate | 113,496 | 0 |
| Actual fraud | 4,612 | 0 |

**Interpretation:** The majority-class classifier predicts every transaction as legitimate. Despite approximately 96% accuracy, it fails to detect any fraudulent transactions. Its PR-AUC equals approximately the validation fraud prevalence, confirming that it provides no meaningful ranking ability.

**Conclusion:** Accuracy is insufficient as a primary evaluation metric for this imbalanced classification problem. Subsequent models must demonstrate their ability to identify fraudulent transactions while managing false positives.

### Experiment 2 — Transaction-Only Logistic Regression

**Objective:** Establish the first learned fraud classification model using transaction features.

**Model:** Logistic Regression

**Solver:** LBFGS

**Configuration:**
- `class_weight="balanced"`
- `max_iter=2000`
- `tol=1e-4`
- `random_state=42`

**Data:**
- Training: 354,324 transactions, 11,988 fraud cases
- Validation: 118,108 transactions, 4,611 fraud cases
- Test: Reserved and not used

**Validation results:**
- Precision: 0.1643
- Recall: 0.7038
- F1-score: 0.2664
- PR-AUC: 0.4622
- ROC-AUC: 0.8606
- Iterations: 1,199
- Convergence: Achieved

**Confusion matrix:**

| | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| Actual legitimate | 96,990 | 16,507 |
| Actual fraud | 1,366 | 3,245 |

**Interpretation:** The model identifies approximately 70.38% of fraudulent transactions at the default classification threshold. However, precision is relatively low, indicating a substantial false-positive burden.

**Conclusion:** Transaction features provide meaningful predictive information. This model establishes our first converged learned baseline. Further investigation will examine threshold selection, model performance over time, and the contribution of identity features.

**Limitations:** Results are based on one chronological validation partition. No conclusions about concept drift or adaptive model performance can yet be drawn.

The above confusion matrix was generated during the initial run where we had a boundary discrepancy, but I'd keep the experiment log clean and record the corrected run as the official benchmark below.
| | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| Actual legitimate | 113,497 | 0 |
| Actual fraud | 4,611 | 0 |