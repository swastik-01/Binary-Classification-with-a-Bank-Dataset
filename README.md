# Binary-Classification-with-a-Bank-Dataset

# Customer Subscription Prediction

This repository contains a machine learning pipeline for predicting customer subscriptions to a bank's term deposit product. The project leverages advanced feature engineering and a tuned ensemble of gradient boosting models to achieve high predictive accuracy (AUC).

---

## 🎯 Project Overview

The goal is to build a robust model that accurately identifies customers who are likely to subscribe to a term deposit. The solution automates the most critical and time-consuming parts of the modeling process, including feature creation and hyperparameter tuning.

### Key Features:
* **Advanced Feature Engineering:** Creates cyclical time-based features (month, day), safely handles log transformations, and generates insightful interaction terms.
* **Hyperparameter Tuning:** Utilizes **Optuna** to automatically search for the optimal parameters for both LightGBM and CatBoost, maximizing their performance.
* **Weighted Ensembling:** Combines predictions from both LightGBM and CatBoost using a data-driven weighting scheme based on each model's cross-validation score.
* **Efficient Modeling:** Uses LightGBM and CatBoost, which are known for their speed and top-tier performance, especially with tabular data.

---


## 📂 Project Structure

├── data/                  # (Optional) Directory for input data
│   ├── train.csv
│   └── test.csv
├── main.py                # Main script for the entire pipeline
├── requirements.txt         # List of Python dependencies
└── README.md              # Project documentation

I have done this in Kaggle notebook so it was just easy with loading the dataset and also for the computational power . I recommend to use Kaggle Notebook as it provides you with good GPUs and TPUs .
