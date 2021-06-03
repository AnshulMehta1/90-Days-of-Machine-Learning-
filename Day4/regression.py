from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

boston=datasets.load_boston()
# There are 1 label and 13 features in the dataset
# Same step for train test split
x=boston.data
y=boston.target
# Linear regression
linear_regr=linear_model.LinearRegression()
# There can be multiple features in the dataset
# But there can one relationship between feature and label
# Transpose
plt.scatter(x.T[0], y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
reg=linear_regr.fit(x_train,y_train)

predictions=reg.predict(x_test)
print("predictions: ", predictions)
print("R^2: ", linear_regr.score(x, y))
print("coeff: ", linear_regr.coef_)
print("intercept: ", linear_regr.intercept_)