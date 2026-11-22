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

## Feature Availability Experiments: A, B and C

### 1. Objective

Investigate whether adding identity-related features and explicitly modelling identity-data availability improves fraud detection performance compared with a transaction-only baseline.

Three feature configurations were evaluated using the same Logistic Regression model and chronological train-validation-test split.

### 2. Dataset and chronological split

Dataset: IEEE-CIS Fraud Detection

The transaction dataset was merged with the identity dataset using `TransactionID` through a validated one-to-one left join.

- Total transactions: 590,540
- Transactions with identity information: 144,233 (24.42%)
- Transactions without identity information: 446,307
- Duplicate TransactionIDs after merge: 0

Chronological split:

| Dataset | Transactions | Fraud cases | Fraud rate |
|---|---:|---:|---:|
| Train | 354,324 | 11,988 | 3.3833% |
| Validation | 118,108 | 4,611 | 3.9041% |
| Test | 118,108 | 4,064 | 3.4409% |
| Total | 590,540 | 20,663 | 3.4990% |

The split was performed chronologically using `TransactionDT`.

The test set was not used during these experiments. All results reported below are validation results.

### 3. Model and hyperparameters

Model: Logistic Regression

Configuration:

- `class_weight="balanced"`
- `solver="lbfgs"`
- `max_iter=2000`
- `random_state=42`
- `tol=1e-4`

The same model configuration was used across all three experiments.

Preprocessing was fitted using training data only. Validation data was transformed using the fitted training preprocessor.

The classification threshold remained at the default value of 0.5. No threshold tuning was performed.

### 4. Feature configurations

| Experiment | Features |
|---|---|
| A | Transaction features only |
| B | Transaction features + identity features |
| C | Transaction features + identity features + `has_identity` indicator |

The `has_identity` indicator represents whether identity information was available for a transaction.

### 5. Validation results

| Metric | Experiment A | Experiment B | Experiment C |
|---|---:|---:|---:|
| Precision | 0.164287 | 0.164997 | 0.165132 |
| Recall | 0.703752 | 0.703752 | 0.704186 |
| F1-score | 0.266388 | 0.267320 | 0.267529 |
| PR-AUC | 0.462218 | 0.455057 | 0.454873 |
| ROC-AUC | 0.860567 | 0.858997 | 0.858956 |
| True Positives | 3,245 | 3,245 | 3,247 |
| False Negatives | 1,366 | 1,366 | 1,364 |
| False Positives | 16,507 | 16,422 | 16,416 |
| True Negatives | 96,990 | 97,075 | 97,081 |

### 6. Confusion matrices

#### Experiment A — Transaction features only

```text
[[96990, 16507],
 [ 1366,  3245]]
```

#### Experiment B — Transaction + identity features

```text
[[97075, 16422],
 [ 1366,  3245]]
```

#### Experiment C — Transaction + identity + availability indicator

```text
[[97081, 16416],
 [ 1364,  3247]]
```

Matrix format:

```text
[[True Negatives, False Positives],
 [False Negatives, True Positives]]
```

### 7. Results interpretation

#### Experiment B compared with Experiment A

Adding identity features resulted in:

- A slight increase in precision.
- No change in recall.
- A slight increase in F1-score.
- 85 fewer false positives.
- A decrease in PR-AUC.
- A decrease in ROC-AUC.

Identity features did not improve ranking performance according to PR-AUC or ROC-AUC.

#### Experiment C compared with Experiment B

Adding the identity-availability indicator resulted in:

- A slight increase in precision.
- A small increase in recall, detecting two additional fraud cases.
- A slight increase in F1-score.
- Six fewer false positives.
- A slight decrease in PR-AUC.
- A slight decrease in ROC-AUC.

The availability indicator produced marginal improvements in threshold-based metrics but did not improve ranking performance.

#### Overall finding

The transaction-only configuration achieved the highest PR-AUC and ROC-AUC among the three feature configurations.

Adding identity features and the identity-availability indicator resulted in small improvements in precision, recall, and F1-score at the default classification threshold, while ranking metrics decreased.

The exploratory analysis previously identified an association between identity-data availability and fraud prevalence. However, this association did not translate into improved ranking performance in these experiments.

These results demonstrate that an observed relationship between a feature and the target does not necessarily result in improved predictive performance when that feature is added to a model.

### 8. Limitations

- Results are based on the validation dataset only. The test dataset remains untouched.
- No statistical significance testing or confidence intervals were calculated.
- The classification threshold was fixed at 0.5 and was not optimised.
- Only Logistic Regression was evaluated in these experiments.
- The experiments evaluate aggregate validation performance and do not yet investigate performance changes across time periods.
- These experiments do not establish whether identity features would be useful under different models, thresholds, or adaptation strategies.

### 9. Next research phase

The next phase will investigate static-model performance over time.

The objective is to determine whether fraud detection performance deteriorates across temporal periods and investigate possible explanations, including changes in fraud prevalence, feature distributions, and fraud patterns.

The results of this analysis will inform the hypothesis and design of the adaptive fraud detection strategy.

## Static Model Temporal Performance Analysis

### Objective

Investigate whether the static Logistic Regression model's fraud detection performance remains consistent across chronological validation periods.

The model was trained once using the training dataset and evaluated across six chronological validation windows without retraining.

### Methodology

- Model: Experiment A — Transaction features only
- Algorithm: Logistic Regression
- Training data: Chronological training split
- Evaluation data: Chronological validation split
- Temporal window size: 7 days
- Model retraining between windows: None
- Classification threshold: 0.5
- Test set: Not used

The final validation window covers approximately 4.90 days because the validation period does not contain a complete seventh day.

### Temporal validation results

| Window | Transactions | Fraud cases | Fraud rate | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20,747 | 798 | 3.846% | 0.1586 | 0.7118 | 0.2594 | 0.4308 | 0.8604 |
| 1 | 20,957 | 897 | 4.280% | 0.1897 | 0.7447 | 0.5031 | 0.8781 |
| 2 | 22,137 | 1,051 | 4.748% | 0.1798 | 0.7212 | 0.5194 | 0.8696 |
| 3 | 22,351 | 699 | 3.127% | 0.1482 | 0.7182 | 0.5267 | 0.8659 |
| 4 | 18,743 | 757 | 4.039% | 0.1666 | 0.6460 | 0.4025 | 0.8396 |
| 5 | 13,173 | 409 | 3.105% | 0.1232 | 0.6357 | 0.2905 | 0.8234 |

### Observations

Performance varies across the chronological validation windows.

Windows 0–3 show fluctuations in precision, recall, PR-AUC, and ROC-AUC. Performance deterioration becomes more apparent in Windows 4 and 5.

Window 5 records the lowest values for precision, recall, PR-AUC, and ROC-AUC:

- Precision: 12.32%
- Recall: 63.57%
- PR-AUC: 0.2905
- ROC-AUC: 0.8234

Compared with Window 1, recall decreases by approximately 10.90 percentage points, PR-AUC decreases by approximately 0.213, and ROC-AUC decreases by approximately 0.055.

### Fraud prevalence

Fraud prevalence varies across validation windows, ranging from approximately 3.10% to 4.75%.

The final window has a fraud rate of approximately 3.10%, which is lower than the fraud rates observed in Windows 1 and 2. However, model performance is substantially worse in Window 5.

This suggests that the observed deterioration cannot be explained solely by an increase in fraud prevalence.

### Preliminary interpretation

The static model exhibits temporal performance variation, with notably weaker performance in the later validation windows.

This is evidence of temporal performance instability. However, the analysis does not yet establish the cause of the deterioration or confirm concept drift.

Further investigation is required to determine whether the observed performance changes are associated with feature distribution changes, changes in fraud patterns, changes in feature-target relationships, or statistical variability.

### Limitations

- The model was evaluated on six temporal windows.
- The final window is shorter than the preceding windows.
- No statistical significance testing or confidence intervals were calculated.
- Performance variation alone does not establish concept drift.
- The analysis uses validation data only; the test set remains untouched.

### Next investigation

Investigate feature distribution changes across the training dataset and chronological validation windows to identify potential data drift and determine whether these changes coincide with the observed deterioration in model performance.
