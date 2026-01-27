# 🏠 House Price Prediction – Advanced Regression Models ( version(2) )

This project predicts house prices using multiple regression models and compares their performance using standard evaluation metrics.  
It is based on the **Ames Housing Dataset** and follows a clean, modular Machine Learning workflow.

---

## 📌 Project Features
- Data loading and inspection
- Data preprocessing:
  - Handling missing values
  - One-hot encoding for categorical features
  - Train/Test split
- Multiple regression models:
  - Linear Regression
  - Polynomial Regression
  - Ridge Regression
  - Lasso Regression
- Model evaluation using:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score
- Clear comparison of model performance

---

## 🗂 Project Structure
```
/house_price_prediction/
│
├── src/
│ ├── main.py # Entry point of the project
│ ├── data_loader.py # Load dataset
│ ├── preprocessing.py # Data cleaning & feature engineering
│ └── model.py # Training & evaluation of models
│
├── data/
│ └── train.csv # Dataset (not included in repo)
│
├── README.md
└── requirements.txt
```

## ⚙️ How to Run the Project
```
pip install -r requirements.txt
```
## Dont forget to download the dataset from Kaggle 
- https://www.kaggle.com/code/ryanholbrook/feature-engineering-for-house-prices?select=train.csv
### run the main 
```
python src/main.py
```

## key Lernings 
- Importance of handling missing data before training
- Effect of regularization (Ridge & Lasso) on model performance
- Polynomial regression can easily overfit without proper scaling
- Clean code structure improves readability and maintainability

## 🚀 Future Improvements
- Add feature scaling using Pipeline
- Hyperparameter tuning with GridSearchCV
- Feature importance visualization
- Model persistence using joblib

---
## Authors 
### Alaa Madi


