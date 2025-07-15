# 💓 Heart Disease Prediction - Full ML Pipeline

This project performs a complete Machine Learning workflow to analyze and predict heart disease using the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).

## 📌 Features
- Data Cleaning & Preprocessing
- PCA for Dimensionality Reduction
- Feature Selection (Chi-Square, RFE, Feature Importance)
- Supervised Models: Logistic Regression, Decision Tree, Random Forest, SVM
- Unsupervised Models: K-Means, Hierarchical Clustering
- Hyperparameter Tuning (GridSearchCV, RandomizedSearchCV)
- Streamlit Web UI for Real-time Predictions
- Ngrok deployment for online access

---

## 🗂 Project Structure

Heart_Disease_Project/
│
├── heart+disease/ # Original dataset
│ └── heart_disease.csv
│
├── notebooks/ # Jupyter Notebooks for each stage
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_pca_analysis.ipynb
│ ├── 03_feature_selection.ipynb
│ ├── 04_supervised_learning.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ └── 06_hyperparameter_tuning.ipynb
│
├── models/
│ └── final_model.pkl # Exported ML model with pipeline
│
├── ui/
│ └── app.py # Streamlit UI App
│
├── deployment/
│ └── ngrok_setup.txt # Notes on running Ngrok
│
├── results/
│ └── evaluation_metrics.txt # Model performance logs
│
├── requirements.txt # List of dependencies
└── README.md
