from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np

def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2

def train_models(x_train, x_test, y_train, y_test):
    results = []

    # Linear Regression
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)


    mae, rmse , r2 = evaluate_model(y_test, y_pred)
    results.append(("Linear Regression", mae, rmse, r2))


    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    poly_model = LinearRegression()
    poly_model.fit(x_train_poly, y_train)
    y_pred_poly = poly_model.predict(x_test_poly)

    mae, rmse, r2 = evaluate_model(y_test, y_pred_poly)
    results.append(("Polynomial Regression", mae, rmse, r2))


    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train, y_train)
    y_pred_ridge = ridge.predict(x_test)
    mae, rmse, r2 = evaluate_model(y_test,y_pred_ridge)
    results.append(("Ridge Regression", mae, rmse,r2))


    # Lasso Regression
    lasso = Lasso(alpha=0.001)
    lasso.fit(x_train,y_train)
    y_pred_lasso = lasso.predict(x_test)

    mae, rmse, r2 = evaluate_model(y_test, y_pred_lasso)
    results.append(("Lasso Regression ", mae , rmse , r2))



    # print comparison
    print("\nModel Performance Comparison")
    for r in results:
        print(r)


