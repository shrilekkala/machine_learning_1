import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot, normaltest

# sklearn imports
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

ale_data = pd.read_csv('mcs_ds_edited_iter_shuffled.csv')


# Split into features and labels
feature_cols = ["anchor_ratio", "trans_range", "node_density", "iterations"]
label = ["ale"]
x, y = ale_data[feature_cols], ale_data[label]

# MODEL 1: LINEAR
linear_model = LinearRegression()
linear_model.fit(x, y)
y_pred = linear_model.predict(x)
residuals_linear = y_pred - y

# qq plot
probplot(residuals_linear['ale'], dist="norm", plot=plt)
plt.title("Probability plot for linear regression residuals vs normal distribution")
plt.show()
# normal test
z, p = normaltest(residuals_linear['ale'])
if p < 0.05:
    print(f"The p value is: {p}")
    print("The p value of the normal test is less than 0.05"
          "\n so we reject the null hypothesis that the residuals are normally distributed.")

# MODEL 2: Support Vector Machine
svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_rbf.fit(x, y.values.ravel())
y_pred_svm = svr_rbf.predict(x)
residuals_svm = y_pred_svm - np.array(y).T[0]

# qq plot
probplot(residuals_svm, dist="norm", plot=plt)
plt.title("Probability plot for SVM residuals vs normal distribution")
plt.show()
# normal test
z, p = normaltest(residuals_svm)
if p < 0.05:
    print(f"The p value is: {p}")
    print("The p value of the normal test is less than 0.05"
          "\n so we reject the null hypothesis that the residuals are normally distributed.")

# MODEL 3: Neural Network
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

# specify the model
regr = MLPRegressor(hidden_layer_sizes=10,
                                          activation='tanh',
                                          solver='lbfgs', max_iter=1000, learning_rate_init=0.001)

# train the model
regr.fit(X_train, y_train.values.ravel())

# evaluate the model on the whole dataset
y_pred_MLP = regr.predict(x)

residuals_MLP = y_pred_MLP - np.array(y).T[0]

# evaluate the model
regr.score(X_test, y_test)

# qq plot
probplot(residuals_MLP, dist="norm", plot=plt)
plt.title("Probability plot for neural network (MLP) residuals vs normal distribution")
plt.show()
# normal test
z, p = normaltest(residuals_MLP)
if p < 0.05:
    print(f"The p value is: {p}")
    print("The p value of the normal test is less than 0.05"
          "\n so we reject the null hypothesis that the residuals are normally distributed.")