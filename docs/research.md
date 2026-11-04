# Research Log: Adaptive Fraud Detection Under Concept Drift

## 1. Research objective

Determine whether available fraud-detection datasets support a credible
experiment into how model performance changes over time and whether
adaptation can improve performance relative to a static model.

The research question is:

> Can a fraud detection system detect changing fraud patterns and adapt
> more effectively than a conventional static model?

Adaptation is a hypothesis to test, not a predetermined outcome.

## 2. Dataset selection criteria

Each candidate will be assessed against the following criteria:

-   Temporal coverage and chronological ordering.
-   Availability and reliability of fraud labels.
-   Number and proportion of fraudulent transactions.
-   Number and interpretability of input features.
-   Ability to simulate a realistic past-to-future evaluation.
-   Potential feature leakage and availability of features at prediction
    time.
-   Evidence of changing transaction or fraud patterns.
-   Access conditions, licensing, and reproducibility.
-   Practical size and complexity for this project.

## 3. Candidate A: IEEE-CIS Fraud Detection

**Source:**
https://www.kaggle.com/competitions/ieee-fraud-detection/data

### Published information

-   The task is to predict whether an online transaction is fraudulent.
-   The target variable is `isFraud`.
-   Transaction and identity data are provided separately and can be
    joined using `TransactionID`.
-   `TransactionDT` represents elapsed time from a reference point,
    rather than a real-world timestamp.
-   The dataset includes transaction attributes and, for a subset of
    transactions, identity and device-related information.

### Potential strengths

-   Real-world transaction data rather than entirely simulated data.
-   A substantial labelled dataset with multiple feature groups.
-   Relative time information that may support chronological evaluation.

### Risks and unanswered questions

-   The exact temporal coverage and distribution of labelled
    transactions must be inspected.
-   The meaning, availability, and leakage risk of engineered count and
    time-delta features need review.
-   Identity information is missing for many transactions.
-   Changes in model performance over time would not, by themselves,
    prove concept drift.
-   Dataset access is subject to Kaggle's competition rules.

### Next validation steps

1.  Obtain access to the data and review its permitted use.
2.  Inspect file sizes, columns, data types, missing values, and target
    distribution.
3.  Establish the observed time range and fraud prevalence across
    chronological periods.
4.  Audit features for leakage and prediction-time availability.
5.  Decide whether the data supports a defensible temporal
    train-validation-test split.

## 4. Candidate B: Fraud Detection Handbook simulator

**Source:**
https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html

### Published information

-   The simulator generates legitimate and fraudulent payment-card
    transactions.
-   It includes transaction timestamps, customer and terminal
    identifiers, amounts, and fraud labels.
-   The documented example covers 183 simulated days.
-   It includes several fraud scenarios, including temporarily
    compromised terminals and customers.
-   The handbook explicitly notes that the simulator simplifies
    real-world transaction dynamics.

### Potential strengths

-   Controlled and reproducible data generation.
-   Time-dependent fraud scenarios suitable for studying adaptation.
-   Interpretable scenarios that can help explain why a model succeeds
    or fails.

### Risks and unanswered questions

-   Simulated fraud is not evidence of real-world fraud behaviour.
-   A model may exploit simple simulator rules rather than learn robust
    fraud patterns.
-   We must check feature construction for information that would not be
    available at prediction time.
-   The simulator's complexity and generation requirements need to be
    assessed before adoption.

### Next validation steps

1.  Review the generation process and documented fraud scenarios.
2.  Confirm data access, implementation requirements, and applicable
    licence.
3.  Determine whether the scenarios support a fair comparison between
    static and adaptive models.
4.  Consider using it as a controlled secondary experiment rather than a
    replacement for real-world data.

## 5. Preliminary comparison

IEEE-CIS is the leading candidate for the main investigation because it
contains real-world e-commerce transaction data and relative temporal
information.

The Fraud Detection Handbook simulator is a potential secondary
candidate because its fraud scenarios are deliberately designed and
documented, making controlled tests possible.

Neither candidate is finally selected. The decision depends on direct
data inspection, feature-leakage analysis, chronological evaluation
feasibility, and project scope.

## 6. Planned evaluation design

Subject to dataset validation, the intended design is:

1.  Train on an earlier period.
2.  Validate on a later period.
3.  Reserve a future period for final testing.
4.  Measure performance across time using metrics appropriate for
    imbalanced classification.
5.  Investigate performance degradation and potential causes.
6.  Compare a static baseline with a justified adaptation strategy using
    equivalent evaluation conditions.

Candidate metrics include precision, recall, F1-score, PR-AUC, confusion
matrices, and threshold analysis.

## 7. Evidence status

This document records an initial review of published dataset
documentation. It does not record an independent analysis of downloaded
data.

No dataset has been finally selected, no model has been trained, and no
experimental results or concept-drift conclusions have been established.

## 8. References

1.  IEEE-CIS Fraud Detection, Kaggle dataset description:
    https://www.kaggle.com/competitions/ieee-fraud-detection/data
2.  Fraud Detection Handbook, Transaction data simulator:
    https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html
3.  Fraud Detection Handbook, introduction:
    https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/Introduction.html

## 9. Next actions

-   [ ] Review dataset access and use conditions.
-   [ ] Inspect IEEE-CIS data structure and time coverage.
-   [ ] Inspect class balance across time periods.
-   [ ] Audit candidate features for leakage.
-   [ ] Update the dataset decision based on observed evidence.
