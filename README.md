# 🦟 Malaria Severity Prediction Using Machine Learning

> An Explainable AI-based Machine Learning system for predicting malaria severity and comparing multiple classification models.

## 🚀 Live Demo

**Try the deployed application:**

[Malaria Severity Prediction – Live Demo](https://malaria-severity-prediction.streamlit.app/)

The application provides:

- Patient severity prediction
- Probability of Non-Severe and Severe Malaria
- Comparison of multiple Machine Learning models
- Confusion matrix and ROC curves
- SHAP-based global feature importance
- LIME-based individual prediction explanation
- Viva / project demonstration guide

---

## 📌 Project Overview

Malaria can present with different levels of severity, and early identification of severe cases can support timely clinical attention.

This project develops a Machine Learning classification system to predict whether a malaria case belongs to:

- **Non-Severe Malaria**
- **Severe Malaria**

The project compares six Machine Learning classification algorithms and evaluates them using multiple performance metrics.

An Explainable AI component using **SHAP** and **LIME** is also included to make model predictions easier to interpret.

> **Educational / research project:** This system is intended for academic demonstration and is not a clinical diagnostic tool.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze a malaria patient dataset
- Perform data preprocessing and exploratory data analysis
- Identify relevant input features
- Prepare the data for Machine Learning
- Train multiple classification models
- Compare model performance using several evaluation metrics
- Investigate class imbalance
- Apply Random Oversampling to the training data
- Train a Balanced Random Forest model
- Predict malaria severity for new patient inputs
- Provide class probabilities
- Explain model predictions using SHAP and LIME
- Deploy the final demonstration application using Streamlit

---

## 🧠 Machine Learning Models

Six classification models were implemented and evaluated:

1. Random Forest
2. AdaBoost
3. Gradient Boosting
4. XGBoost
5. CatBoost
6. Balanced Random Forest

---

## 📊 Dataset

The current project demonstration uses:

| Dataset Property | Value |
|---|---:|
| Total records | **337** |
| Input features | **17** |
| Test records | **68** |
| Models evaluated | **6** |
| Target classes | **2** |

### Target Classes

- Non-Severe Malaria
- Severe Malaria

---

## 🔬 Project Workflow

```text
Raw Dataset
     ↓
Data Preprocessing
     ↓
Exploratory Data Analysis
     ↓
Feature Preparation
     ↓
Train / Test Split
     ↓
Feature Scaling
     ↓
Multiple ML Models
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Severity Prediction
     ↓
SHAP + LIME Explainability
     ↓
Streamlit Web Application
