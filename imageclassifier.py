# -*- coding: utf-8 -*-
"""ImageClassifier (1).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KPdBcMZ5w6vV__riRvGBF3G5Igp_PibW
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os

print ("tensorflow version: ",tf.__version__)
print("is using gpu: ", len(tf.config.list_physical_devices('GPU')) > 0) # Changed this line to use the correct function

"""## **Load Data**"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)
print("x_test shape: ", x_test.shape)
print("y_test shape: ", y_test.shape)

x_train[1,:]

"""## **Normalize Values**"""

x_train = x_train/255.0
x_test = x_test/255.0

"""## **Reshape Data**"""

y_train[:5]

y_train = y_train.reshape(-1,)
y_train[:5]

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

"""## **Show Data**"""

idx = random.randint(0,len(x_train))
plt.figure(figsize = (15,2))
plt.imshow(x_train[idx,:])
plt.xlabel(classes[y_train[idx]])
plt.show

"""# **Build Model**"""

from tensorflow.keras import models, layers

ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(x_train, y_train, epochs=10)

from sklearn.metrics import confusion_matrix , classification_report
y_pred = ann.predict(x_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=None)  # Keeps the original colormap
plt.title("Confusion Matrix")
plt.colorbar()

# Add labels for axes
classes = [str(i) for i in range(10)]  # Assuming 10 classes labeled as 0, 1, ..., 9
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Add values to cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(x_train, y_train, epochs=10)

cnn.evaluate(x_test,y_test)

y_pred = cnn.predict(x_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=None)  # Keeps the original colormap
plt.title("Confusion Matrix")
plt.colorbar()

# Add labels for axes
classes = [str(i) for i in range(10)]  # Assuming 10 classes labeled as 0, 1, ..., 9
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Add values to cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

!nvidia-smi