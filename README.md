# adaptive-fraud-detection
Investigating the robustness and adaptation of fraud detection under concept drift.

# Adaptive Fraud Detection Under Concept Drift

## Overview

This project investigates whether a fraud detection system can maintain its effectiveness as transaction patterns and fraudulent behaviour change over time.

The project will establish a baseline model, evaluate its performance on later unseen data, investigate potential causes of performance degradation, and compare justified adaptation strategies against a static model.

**The goal is to evaluate whether adaptation helps, under what conditions it helps, and what trade-offs it introduces.** Adaptation is not assumed to be better.

## Research Question

Can a fraud detection system detect changing fraud patterns and adapt more effectively than a conventional static model?

## Objectives

- Assess suitable fraud detection datasets for chronological evaluation.
- Establish a reproducible baseline fraud detection model.
- Evaluate model performance across different time periods.
- Investigate performance changes and potential evidence of concept drift.
- Compare selected adaptation strategies with a static baseline.
- Analyse false positives, false negatives, and performance trade-offs.
- Document findings, limitations, and recommendations.

## Proposed Methodology

1. **Research:** assess dataset suitability, temporal coverage, labels, features, and limitations.
2. **Baseline:** train and evaluate a conventional model using a chronological split.
3. **Failure analysis:** examine performance across time periods and investigate possible causes of degradation.
4. **Adaptation:** implement an evidence-based adaptation strategy if justified.
5. **Comparison:** evaluate the static and adaptive approaches under comparable conditions.
6. **Review:** document results, limitations, reproducibility, and lessons learned.

## Evaluation

Evaluation will consider appropriate metrics for imbalanced fraud detection, including precision, recall, F1-score, PR-AUC, confusion matrices, and threshold trade-offs.

Results will be analysed over time, not only as a single aggregate score. The final metrics and evaluation protocol will be justified based on the selected dataset and experimental design.

## Dataset

The dataset has not yet been finalised. Candidate datasets will be assessed for temporal coverage, label quality, class imbalance, feature availability, leakage risks, licensing, and suitability for chronological evaluation.

Raw datasets will not be committed to this repository.

## Repository Status

**Current stage:** Problem formulation and dataset research.

No model performance results or conclusions have been established yet.

## Reproducibility and Project Quality

The project will aim to include documented data preparation, reproducible experiments, configuration management, appropriate tests, and clear records of technical decisions.

## Limitations

Findings will be interpreted in the context of the chosen dataset. Evidence of changing performance alone will not automatically be treated as proof of concept drift, and results from synthetic data will not be presented as proof of real-world performance.

## Project Documentation

- Research and dataset assessment: `docs/research.md`
- Technical decision log: `docs/decisions.md`
- Experiment log: `docs/experiments.md`
- Work and learning log: `docs/project-log.md`

These documents will be developed as the project progresses.