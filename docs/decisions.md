# Technical Decision Log

## Decision 001: Initial dataset shortlist

**Status:** Provisional  
**Date:** 4 November 2026

### Context
The project requires transaction data with fraud labels and temporal information to investigate model performance over time.

### Candidates considered
- IEEE-CIS Fraud Detection
- Fraud Detection Handbook simulator

### Current position
IEEE-CIS is the preferred candidate for the primary investigation because it contains real-world e-commerce transaction data.

The Fraud Detection Handbook simulator remains a possible secondary dataset for controlled experiments.

### Alternatives and trade-offs
- IEEE-CIS: Real-world data, but complex features and potential leakage require investigation.
- Handbook simulator: Controlled scenarios, but synthetic data may not represent real-world fraud.

### Final decision
No dataset has been selected yet. Final selection depends on direct data inspection, temporal coverage, feature leakage, and access conditions.

### Next action
Inspect the IEEE-CIS dataset before confirming the selection.

## Decision: Chronological Dataset Partitioning

**Status:** Accepted — initial experimental design

### Context

The IEEE-CIS dataset contains 590,540 labelled transactions covering approximately 182 days.

The project investigates whether fraud detection models experience performance degradation as transaction behaviour evolves. A random train/validation/test split could mix historical and future transactions, making temporal evaluation less representative of a real-world deployment scenario.

### Decision

Use chronological partitioning based on `TransactionDT`.

The initial allocation is:

- Training: approximately 60% of transactions.
- Validation: approximately 20% of transactions.
- Testing: approximately 20% of transactions.

Transactions are sorted by `TransactionDT`. Timestamp boundaries are used to ensure that transactions sharing the same timestamp are not separated across partitions.

### Initial Results

| Partition | Transactions | Fraud cases | Fraud rate |
|---|---:|---:|---:|
| Training | 354,324 | 11,988 | 3.383% |
| Validation | 118,108 | 4,611 | 3.904% |
| Testing | 118,108 | 4,064 | 3.441% |

All 590,540 transactions are accounted for.

### Rationale

This approach simulates a realistic temporal evaluation:

- Train using historical transactions.
- Validate using later transactions.
- Evaluate using the latest unseen transactions.

It also allows investigation of changes in fraud prevalence and model performance across time.

### Limitations and Considerations

- Fraud prevalence differs across partitions.
- The split does not itself establish concept drift.
- The test partition must remain untouched during model selection and threshold tuning.
- Any preprocessing, feature selection or encoding must be fitted using training data only.
- The final split strategy may be revisited if subsequent analysis identifies a methodological issue.