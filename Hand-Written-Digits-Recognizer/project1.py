import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import mnist
#  Training the Model
x_train=mnist.train_images()
y_train=mnist.train_labels()
x_test=mnist.test_images()
y_test=mnist.test_labels()
# print(x_train)
# print(x_test)

print(x_train.ndim)
print(y_train.ndim)
# Imlementing a Neural Network
x_train=x_train.reshape((-1,28*28))
x_test=x_train.reshape((-1,28*28))
#  Reahping for fitting in the Neural Network
# The range in the Pixels is 0 to 256 which needs to be bought down by 0 to 1
# In order to fit in Neural Network
x_test=np.array(x_test/256)
x_train=np.array(x_train/256)
clf=MLPClassifier(solver='adam',activation='relu',hidden_layer_sizes='(64,64)')
# Building the classifier
# Model Training
clf.fit(x_train,y_train)
predictions=classifier.predict(x_test)
accuracy=confusion_matrix(y_test,predictions)
print(accuracy)
#  Trace divided by all of the elements
def acc(cm):
    diag=cm.trace()
    elements=cm.sum()
    return diag/elements

print(acc(accuracy))





