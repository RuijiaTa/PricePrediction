import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
from scipy import stats
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.linear_model import Lasso

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

'''
Step1: Prepare the Data.
'''
### Open the data set, and read the content of the house prices.
df = pd.read_fwf("boston.csv")
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

### Divide the whole data into train set and test set. Pick 20% to be test set and the rest to be trained by models.
print("Get the shape of Train Set and Test Set.(X:Inputs; Y:MEDV)")
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
print("X_train.shape:", X_train.shape)
print("Y_train.shape:", Y_train.shape)
print("X_test.shape:", X_test.shape)
print("Y_test.shape:", Y_test.shape)


##############################################################

'''
Step3: Analyse the models.
'''

'''1. Linear Regression'''
def linear_regression(x, y):
    ### Use the model of OLS to pridicate the house price.
    reg1 = sm.OLS(y, sm.add_constant(x)).fit()
    result_OLS = reg1.summary()

    ### Using OLS to Define the sequence factors's importance'.

    ### Get the Linear Regression's RMSE and Score the model of Linear Regression.
    pred1 = reg1.predict(sm.add_constant(X_test))
    # The fitting standard deviation of regression system is a parameter corresponding to
    # the original data and the predicted data in linear regression model.
    X = df.drop(['MEDV'], axis=1).values
    Y = np.log(df['MEDV'].values)
    reg1 = LinearRegression(fit_intercept=True).fit(X, Y)

    ### Pick Losso Regression as the optimization model
    # Lasso Regression （Least Absolute Shrinkage and Selection Operator）
    # Lasso also penalizes the absolute value of its regression coefficient.
    # This makes it possible for the value of the punishment to go to zero
    lasso = Lasso()
    lasso.fit(x, y)
    y_predict_lasso = lasso.predict(X_test)
    return pred1

# linear_regression(X_train,Y_train)

'''2. KNN Regression'''


def knn_regression(x, y):
    ### Use the KNN Regression as the model to predict. And judge the result of it by RMSE and Score.
    reg2 = KNeighborsRegressor(n_neighbors=2).fit(x, y)
    pred2 = reg2.predict(X_test)

    ### Use MSE which can better reflect the actual situation of predicted value error. To compare with the neighbors.
    knn_res = []
    for idx in range(2, 20):
        reg2 = KNeighborsRegressor(n_neighbors=idx).fit(x, y)
        pred2 = reg2.predict(X_test)
        knn_res.append(np.abs((Y_test - pred2)).mean())
    return pred2


# knn_regression(X_train,Y_train)

'''3. SVM Regression'''


def svm_regression(x, y):
    ### Use the SVM Regression as the model to predict. And judge the result of it by RMSE and Score.
    reg3 = SVR().fit(x, y)
    pred3 = reg3.predict(X_test)
    return pred3


# svm_regression(X_train,Y_train)

'''4. RandomForest'''


def random_forest_regression(x, y):
    ### Use the RandomForest Regression as the model to predict. And judge the result of it by RMSE and Score.
    reg4 = RandomForestRegressor().fit(x, y)
    pred4 = reg4.predict(X_test)

    ### Use MSE which can better reflect the actual situation of predicted value error. To compare with the max_depth.
    rf_res = []
    for idx in range(2, 20):
        reg4 = RandomForestRegressor(max_depth=idx).fit(x, y)
        pred4 = reg4.predict(X_test)
        rf_res.append(np.abs((Y_test - pred4)).mean())

    ### Use MSE which can better reflect the actual situation of predicted value error. To compare with the number of estimator.
    rf_res = []
    for idx in range(20, 100, 10):
        reg4 = RandomForestRegressor(n_estimators=idx).fit(x, y)
        pred4 = reg4.predict(X_test)
        rf_res.append(np.abs((Y_test - pred4)).mean())

    return pred4


##############################################################

'''
Step4: Comparing 4 Regression Models
'''
def compare_4_models(p1, p2, p3, p4):
    plt.plot(Y_test, p1, 'o', label="Linear Regression")
    plt.plot(Y_test, p2, 'o', label="KNN")
    plt.plot(Y_test, p3, 'o', label="SVM")
    plt.plot(Y_test, p4, 'o', label="RF")
    plt.plot([0, 50], [0, 50], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    plt.legend()
    plt.title("Comparison of 4 models")
    plt.show()
    print("Finish the Boston House Price Prediction!")
    print("The Best Choice is RandomForest Regression")


compare_4_models(linear_regression(X_train, Y_train), knn_regression(X_train, Y_train),svm_regression(X_train, Y_train), random_forest_regression(X_train, Y_train))
