# 🚨 Credit Card Fraud Detection Model

A machine learning pipeline to detect fraudulent credit card transactions using **Random Forest** and **XGBoost**, with a strong focus on **time-aware evaluation, probability calibration, and business-aligned decision thresholds**.

Dataset used: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 📌 Project Overview

Less than **0.2% of transactions are fraudulent** — yet they cost banks billions globally every year. Detecting fraud is like finding a **needle in a haystack**: only **492 frauds out of 284,807 transactions** in this dataset.

This project simulates how a fraud analytics team would approach the problem:

* Building **trustworthy, calibrated models**
* Avoiding **data leakage with time-aware splits**
* Balancing **catching fraudsters vs. minimizing false alarms**

---

## ⚡ Pipeline Workflow

### 1. Exploratory Data Analysis (EDA)

* Distribution of **transaction amounts**
* **Temporal patterns**: hour of day, day-parts
* Class imbalance visualization

### 2. Feature Engineering

* `Hour` (transaction time)
* `Night_transaction` (binary feature)
* `Business_hours` (binary feature)
* `log(Amount)` (scale correction)

### 3. Modeling

* **Logistic Regression (baseline)**
* **Random Forest**
* **XGBoost**
* **Probability Calibration**: Isotonic regression with `TimeSeriesSplit(cv=3)`

### 4. Evaluation Strategy

* **Time-aware split** (train → validation → test)
* Metrics:

  * **AUPRC (Average Precision)** → main metric for imbalanced data
  * ROC-AUC
  * Brier Score (calibration quality)
  * Expected Calibration Error (ECE)

### 5. Threshold Tuning

Two modes of decision-making:

1. **Precision-first** → target ≥90% precision (minimize false positives, better customer experience)
2. **Cost-based** → minimize financial loss using:

   * False Negative (FN) = \$200
   * False Positive (FP) = \$5
     *(values are illustrative — replace with business-specific costs)*

---

## 💡 Why AUPRC over ROC-AUC?

* With **severe imbalance**, ROC-AUC looks high because most negatives are easy.
* **AUPRC** emphasizes precision-recall trade-off, which aligns better with fraud detection (missed fraud vs. false alarms).

---

## 🔒 Reproducibility & Leakage Prevention

* Strict **time-based split** (train/validation/test in chronological order).
* **Thresholds tuned on validation only** → reported on **held-out test window**.
* Prevents **look-ahead bias** and ensures fair real-world evaluation.

---

## 📊 Results (Example Summary)

* **Random Forest** and **XGBoost** outperform baseline Logistic Regression.
* **Calibrated probabilities** → more reliable fraud scores.
* **Precision-first mode** yields fewer false positives.
* **Cost-minimization mode** adapts depending on FN/FP assumptions.

---

## 🔧 How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook step by step:

   * **EDA → Feature Engineering → Modeling → Calibration → Threshold Tuning → Cost Analysis → Summary**

---


## 🎯 Key Takeaways

* **Leakage-safe, time-aware evaluation** is critical.
* **Probability calibration** improves trust in model outputs.
* **Business-aligned thresholds** (precision-first vs. cost-minimum) guide deployment.
* **Cost assumptions matter** — small changes can flip the preferred model/threshold.

