# Explainable Fraud Detection with Temporal Validation and Velocity-Based Features

## Overview

This project presents a robust and production-aware fraud detection pipeline designed for highly imbalanced credit card transaction data.

The study emphasizes realistic evaluation through temporal validation, focusing on behavioral anomaly detection rather than static transaction attributes. Unlike traditional approaches that rely on random train-test splits, this implementation uses chronological separation to simulate real-world deployment conditions.

Model evaluation prioritizes Precision-Recall metrics and incorporates SHAP-based explainability.


---

## Problem Statement

Credit card fraud detection is a highly imbalanced binary classification problem:

- Fraud transactions represent less than 1% of total observations.
- Accuracy is misleading.
- Random splits may introduce data leakage.
- Behavioral fraud patterns evolve over time.

The objective is to build a robust, leakage-free model capable of detecting fraudulent behavior using temporal validation and behavioral feature engineering.

---

## Key Methodological Decisions

### 1️⃣ Temporal Train-Test Split

Transactions are split chronologically:

- First 80% → Training
- Last 20% → Test

This prevents future information leakage and simulates production deployment.

---

### 2️⃣ Handling Class Imbalance

To address severe imbalance:

- Controlled undersampling (1:10 ratio)
- Class weighting using `scale_pos_weight`
- Precision-Recall focused evaluation

The test set retains its original distribution to reflect real-world conditions.

---

### 3️⃣ Feature Engineering Strategy

The model leverages behavioral features:

- 1-hour and 24-hour transaction counts
- 1-hour and 24-hour transaction sums
- 24-hour mean transaction amount
- Log-transformed amount
- Time-of-day indicators
- Leakage-free merchant and category risk encoding

Velocity-based features significantly improve detection performance.

---

## Model Architecture

Hyperparameters were selected to balance model complexity and generalization, preventing overfitting under severe class imbalance.

We employ a regularized XGBoost classifier with:

- Limited tree depth
- Reduced learning rate
- Subsampling
- L1 & L2 regularization
- Early stopping (based on PR-AUC)

Evaluation metric: **Precision-Recall AUC (PR-AUC)**

---

## Final Performance
Performance is reported on a temporally separated test set reflecting real-world transaction distribution.

| Metric | Value |
|--------|-------|
| PR-AUC | 0.77 |
| Test ROC-AUC | 0.996 |
| Fraud Precision | 0.73 |
| Fraud Recall | 0.72 |

These results demonstrate strong separation capability while maintaining balanced fraud detection performance under real-world class imbalance conditions.

The model achieves balanced fraud detection while maintaining interpretability.

---

## Explainability (SHAP)

SHAP analysis reveals:

- 24-hour mean transaction amount is the most influential feature.
- Short-term spending spikes strongly increase fraud probability.
- Merchant and category risk contribute meaningfully.
- Night-time transactions show elevated risk.

This aligns with established fraud patterns where abrupt behavioral deviations serve as primary risk indicators.

These findings confirm that the model captures behavioral anomalies rather than static transaction attributes.


---

## Repository Structure
    fraud-detection-temporal-xgboost/
    │
    ├── Fraud_Detection_Temporal_XGBoost.ipynb
    ├── feature_engineering.py
    ├── requirements.txt
    └── README.md

---
## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

## Future Work

- Cost-sensitive optimization
- Probability calibration
- Real-time risk scoring API
- Production deployment as a SaaS fraud monitoring system

---

## Author

Mehmet Ersolak  
Computer Engineering Student  
Interests: Machine Learning, Fraud Detection, Behavioral Modeling  

