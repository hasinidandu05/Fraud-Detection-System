# Fraud Detection using Machine Learning

This project builds a machine learning pipeline to detect fraudulent transactions using Random Forest, XGBoost, and LightGBM. The process includes data preprocessing, handling class imbalance with SMOTE, training and evaluating multiple models, model interpretability using SHAP, and saving the final pipeline.

## Features

- Data cleaning and feature engineering
- Handling imbalanced dataset using SMOTE
- Model training with Random Forest, XGBoost, and LightGBM
- Evaluation using classification metrics and precision-recall curves
- Feature importance and SHAP-based explainability
- Saving the trained model pipeline using Joblib

## Technologies Used

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- imbalanced-learn
- SHAP
- Joblib

## Dataset

The dataset used contains financial transaction records with features such as:

- Transaction type
- Amount
- Old and new balances
- Time step
- isFraud (target variable)

Feature engineering includes balance changes, amount ratios, and extracting hour/day from transaction step.

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn shap joblib
