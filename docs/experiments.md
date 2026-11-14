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