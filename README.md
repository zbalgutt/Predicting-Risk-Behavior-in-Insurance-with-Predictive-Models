# 📊 Predicting Risk Behavior in Insurance with Predictive Models

## Overview

This project develops a predictive modeling framework to classify high-risk insurance policyholders using structured underwriting and claims-related data. The objective is to evaluate whether modern machine learning techniques can enhance traditional actuarial approaches for insurance risk segmentation.

As an Associate of the Society of Actuaries (ASA) and MS in Data Science candidate at Columbia University, this project bridges classical actuarial modeling with modern machine learning methods.

---

## 🎯 Objective

Can machine learning models improve the identification and segmentation of high-risk insurance policyholders compared to traditional statistical models?

The project compares:

- Logistic Regression (actuarial baseline)
- Decision Tree
- Random Forest
- XGBoost
- Deep Learning (PyTorch neural network)

---

## 📁 Dataset

**Source:** Prudential Life Insurance Assessment (Kaggle)

The dataset contains:
- Demographic features
- Policy characteristics
- Insurance & medical history variables
- Underwriting response (transformed into binary risk classification)

**Target Variable:**
- 0 → Lower Risk  
- 1 → Higher Risk  

---

## 🛠 Methodology

### 1️⃣ Data Preprocessing
- Missing value handling (deletion + domain-informed imputation)
- Feature aggregation (insurance history score, family history score)
- One-hot encoding for categorical variables
- Min–max scaling for numerical stability
- 80–20 train–validation split

### 2️⃣ Modeling Framework
- Logistic Regression (interpretable baseline)
- Decision Tree (nonlinear threshold modeling)
- Random Forest (variance reduction via bagging)
- XGBoost (boosted ensemble learning)
- Feedforward Neural Network (PyTorch, BCE loss, Adam optimizer)

### 3️⃣ Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score (primary metric)

Recall is emphasized due to the asymmetric cost of failing to identify high-risk policyholders.

---

## 📈 Results

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.7097   | 0.6763    | 0.6885 | 0.6824   |
| Decision Tree       | 0.6596   | 0.6190    | 0.6462 | 0.6323   |
| Random Forest       | 0.7273   | 0.6808    | 0.7492 | 0.7133   |
| **XGBoost**         | **0.7344** | **0.6861** | **0.7627** | **0.7224** |
| Deep Learning       | 0.7043   | 0.6815    | 0.6517 | 0.6663   |

### Key Findings
- XGBoost achieved the highest F1-score and recall.
- Logistic Regression remained competitive and highly interpretable.
- Ensemble models captured nonlinear interactions effectively.
- Deep learning did not outperform boosting on structured tabular data.

---

## 🔎 Model Interpretability

To address regulatory and governance considerations:

- Feature importance analysis (XGBoost)
- Partial dependence plots
- Risk driver identification (BMI, insurance history, policy features)

This supports a hybrid modeling approach:
Use complex models for prediction and interpretable models/explanations for governance.

---

## 💼 Business Impact

This framework demonstrates how insurers can:
- Improve underwriting precision
- Reduce cross-subsidization
- Identify high-risk policyholders earlier
- Balance predictive accuracy with regulatory transparency

The project highlights the trade-off between interpretability (logistic regression) and performance (ensemble methods), a critical consideration in insurance analytics.

---

## ⚙️ Tech Stack

- Python
- Pandas & NumPy
- Scikit-learn
- XGBoost
- PyTorch
- Matplotlib

---

## 🚀 Future Improvements

- Incorporate severity modeling (loss prediction)
- Apply SHAP for deeper explainability
- Introduce temporal modeling of policyholder behavior
- Fairness analysis across demographic groups
- Hyperparameter tuning and cross-validation

---

## 👤 Author

**Zachary Balgut Tan**  
MS in Data Science — Columbia University (Expected Dec 2026)  
Associate of the Society of Actuaries (ASA)
