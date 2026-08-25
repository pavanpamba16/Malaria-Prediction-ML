# Malaria Prediction Using Machine Learning

## Project Overview

This project focuses on predicting malaria severity using Machine Learning classification techniques.

The project includes data preprocessing, exploratory data analysis, multiple Machine Learning models, class imbalance handling, model comparison, and model explainability using LIME and SHAP.

## Objectives

- Analyze the malaria dataset
- Perform data preprocessing and exploratory data analysis
- Identify important features
- Train multiple Machine Learning classification models
- Compare model performance
- Handle class imbalance using Random Oversampling
- Train a Balanced Random Forest model
- Explain predictions using LIME and SHAP
- Save and load the trained Machine Learning model
- Predict malaria severity for a new patient

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- CatBoost
- Imbalanced-learn
- LIME
- SHAP
- Joblib
- Jupyter Notebook

## Machine Learning Models

The following models were implemented and compared:

1. Random Forest
2. AdaBoost
3. Gradient Boosting
4. XGBoost
5. CatBoost
6. Balanced Random Forest

## Data Processing

The notebook performs:

- Dataset loading
- Missing-value analysis
- Exploratory Data Analysis
- Correlation analysis
- Feature preparation
- Train-test splitting
- Feature scaling
- Class distribution analysis

## Handling Class Imbalance

Random Oversampling was applied to the training dataset to address class imbalance.

The testing dataset was kept unchanged so that model performance could be evaluated on unseen data.

A Balanced Random Forest model was then trained using the balanced training data.

## Model Explainability

### LIME

LIME was used to explain individual patient predictions and identify which features contributed positively or negatively to a prediction.

### SHAP

SHAP was used for:

- Global feature importance
- Feature contribution analysis
- Individual patient explanations

These techniques help make the Machine Learning predictions easier to interpret.

## Prediction

The final model can be used to predict:

- Non-Severe Malaria
- Severe Malaria

The prediction system also provides the probability associated with each class.

## Project Structure

```text
Malaria-Prediction-ML/
│
├── malaria_prediction.ipynb
│
└── README.md
