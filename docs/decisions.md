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