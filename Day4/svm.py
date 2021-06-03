# Support Vector Machine
#  Effective for high features-more features 
#  Classification as well as regression
# Hyper Planes are suport vectors
# n-1 dimesions of Planes
# Margin Lines must be the greatest possible 
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import neighbors,metrics
from sklearn import  svm
from sklearn.preprocessing import  LabelEncoder
from sklearn.metrics import accuracy_score
iris=datasets.load_iris()
classes=['Iris Setosa','Iris Versicolour','Iris Virginica']
x=iris.data
y=iris.target
# print(x,y)
# print(x.shape)
# print(y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=svm.SVC()
model.fit(x_train,y_train)
print(model)
predictions = model.predict(x_test)
acc=accuracy_score(y_test,predictions)
print(acc)
print(y_test)
print(predictions)