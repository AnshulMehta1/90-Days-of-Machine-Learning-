from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd

bc=load_breast_cancer()
print(bc)
x=bc.data
y=bc.target
x_train,x_test,y_train,y_test=train_test_split(x, y,  test_size=0.1)

model=KMeans(n_clusters=2,random_state=0)
# Randomization in tge algorithm will change every time

model.fit(x_train)
predictions=model.predict(x_test)
labels=model.labels_
print(labels)
print(predictions)
print(accuracy_score(y_test,predictions))
# Testing
# Cross Tabulations Testing


print(pd.crosstab(y_train,labels))
