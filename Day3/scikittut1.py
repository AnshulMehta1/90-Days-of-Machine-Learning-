# Classification
# n features have n dimensions 
# Classification algorithms might seprate the data points in the graph
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import neighbors,metrics
from sklearn.preprocessing import  LabelEncoder
iris=datasets.load_iris()
#  Splitting
# x=iris.data
# y=iris.target
# print(x,y)
# print(x.shape)
# print(y.shape)
# #  The Shape 
# (150, 4)
# (150,)
# An Example
# Training The Model
# Training with 8 sample
# Hours of Study with grades
#  We'll Predict accuracy for other 2 out of 10

# Results
# (120, 4)
# (30, 4)
# (120,)
# (30,)

# K NN means calculating distance from a point to k nearest points which will be then labelled into a label/region
# k value higher whenever a bigger datasets
# k generally odd number 

data=pd.read_csv(r'D:\Desktop\Technologies\MachineLearning\Machine-Learning-in-90-days\Day3\car.data')
print(data.head())
# Lables
X = data[['buying', 'maintain', 'safety']].values
y = data[['class']]
X = np.array(X)
print(X)
#converting data
##  Converting the data as String do not directly fit in the model
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])
print(X)
# Mapping Lables
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class'] = y['class'].map(label_mapping)
y = np.array(y)
#  Creating the Model 
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#  Mention that we'll be testing 20% of the data and training 80% of the data
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

KNN=neighbors.KNeighborsClassifier(n_neighbors=25,weights='uniform')
# Training the Lables
KNN.fit(x_train,y_train)
predict=KNN.predict(x_test)
# Predicting the value of 2 based on the model
# For accuract 
accuracy=metrics.accuracy_score(y_test,predict)

#  [Printing Prediction and Accuracy]
print(predict)
print(accuracy)

# Testing With some values 

a = 100
b= 963
c=894

print("A real ", y[a])
print("A predictes", KNN.predict(X)[a])
print("A real ", y[b])
print("A predictes", KNN.predict(X)[b])
print("C real ", y[c])
print("C predictes", KNN.predict(X)[c])

