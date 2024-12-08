# AI CNN Project 4
Google Colab link: https://colab.research.google.com/drive/1Xw-mPPhHEyIdskrkorfZe6oXQXiAQyV8?usp=sharing 
# **Step1: Import packages**


```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os

print("Packages imported!")
print ("tensorflow version: ",tf.__version__)
print("is using gpu: ", len(tf.config.list_physical_devices('GPU')) > 0)

```

# **Step2: Load Data**


```python
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# One-hot encode the labels for training
y_train = to_categorical(y_train, 10)  # 10 classes in CIFAR-10
y_test = to_categorical(y_test, 10)

# Define the class names for CIFAR-10
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Normalize the images
x_train = x_train / 255.0
x_test = x_test / 255.0

print("Data loaded!")
```

# **Step3: Display Data**


```python
# Function to show random examples with true and predicted labels
def show_random_examples(x, y, p):
    indices = np.random.choice(len(x), size=10, replace=False)

    x = x[indices]
    y = y[indices]
    p = p[indices]

    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i])
        plt.xticks([])
        plt.yticks([])
        col = 'green' if np.argmax(y[i]) == np.argmax(p[i]) else 'red'
        plt.xlabel(f"Pred: {classes[np.argmax(p[i])]}\nTrue: {classes[np.argmax(y[i])]}", color=col)
    plt.show()

# Show random examples with true labels
show_random_examples(x_train, y_train, y_train)
```

# **Step3: Build Models**


```python
from tensorflow.keras import models, layers
from sklearn.metrics import confusion_matrix , classification_report
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation, Dropout, Input
from tensorflow.keras.optimizers import Adam, Nadam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint

print("Packages imported!")
```

### Model 1: Bad Model, 50% accuracy & Overfitting


```python
## Create the model
# Defining the model
model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),  # Flatten input for CNN input shape (32x32x3)
    layers.Dense(3000, activation='relu'),    # Dense hidden layer with 3000 neurons
    layers.Dense(1000, activation='relu'),    # Dense hidden layer with 1000 neurons
    layers.Dense(10, activation='softmax')    # Output layer with 10 neurons for classification
])
# Model Summary
model.summary()
```

##### *Trainning Model 1*


```python
## Train the model
# Compiling the model
model.compile(optimizer='SGD',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
# Fiting the model with validation split (to monitor validation accuracy)
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# Get the predictions (probabilities)
y_pred = model.predict(x_test)
# Convert the predicted probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)
# Convert the true labels from one-hot encoding to class labels
y_test_classes = np.argmax(y_test, axis=1)
## Display Data
!nvidia-smi
# Show Classification Report
print("Classification Report: \n", classification_report(y_test_classes, y_pred_classes))
# Show predicted labels
show_random_examples(x_test, y_test, y_pred)
# Plot for training and validation accuracy
plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# Plot for training and validation loss as well
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# Show Plots
plt.show()
# Compute confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
# Plot the confusion matrix
plt.figure(figsize=(5, 5))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
# Add labels for axes
classes = [str(i) for i in range(10)]
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
# Show confusion matrix
plt.show()
```

## Model 2: Good Model, 81% accuracy, Well-Generalized + Early Stopping


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)
# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
# Model Summary
model.summary()
```

##### *Trainning Model 2*


```python
import json
# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Define model checkpoint callback to save the best model based on validation accuracy
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max',
                             save_best_only=True, verbose=1)

# Train the model with data augmentation
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

print("Model saved at:", os.path.abspath("best_model.keras"))
# Save the training history as a JSON file
with open('history.json', 'w') as f:
    json.dump(history.history, f)
```

## Dispay Results



```python
from tensorflow.keras.models import load_model
## Load Data
model = load_model("best_model.keras")
print("Model loaded successfully!")
# Load the history from the saved JSON file
with open('history.json', 'r') as f:
    history_data = json.load(f)
print("History loaded successfully!")

# Get the predictions (probabilities)
y_pred = model.predict(x_test)
# Convert the predicted probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)
# Convert the true labels from one-hot encoding to class labels
y_test_classes = np.argmax(y_test, axis=1)
## Display Data
!nvidia-smi
# Show Classification Report
print("Classification Report: \n", classification_report(y_test_classes, y_pred_classes))
# Show predicted labels
show_random_examples(x_test, y_test, y_pred)
# Plot for training and validation accuracy
plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.plot(history_data['accuracy'], label='Training Accuracy')
plt.plot(history_data['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# Plot for training and validation loss as well
plt.subplot(1, 2, 2)
plt.plot(history_data['loss'], label='Training Loss')
plt.plot(history_data['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# Show Plots
plt.show()
# Compute confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
# Plot the confusion matrix
plt.figure(figsize=(5, 5))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
# Add labels for axes
classes = [str(i) for i in range(10)]
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
# Show confusion matrix
plt.show()
```

