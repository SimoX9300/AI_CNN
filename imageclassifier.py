
# **Step1: Setup & Import packages**
# !! must have python 3.9 or 3.10 for tensorflow to work, higher versions are not supported
# !! install the following packages:
# python3.9 -m pip install tensorflow==2.17.1
# python3.9 -m pip install numpy 
# python3.9 -m pip install scikit-learn
# python3.9 -m pip install matplotlib
# ?? run the project with the following command in the terminal oppened at the file location: 
# python3.9 imageclassifier.py

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation, Dropout, Input
from tensorflow.keras.optimizers import Adam, Nadam, SGD, RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix , classification_report
import matplotlib.pyplot as plt
import numpy as np
import random, os, json

print("---\nPackages imported!",
      "\ntensorflow version: ",tf.__version__, 
      "\nusing gpu: ", len(tf.config.list_physical_devices('GPU')) > 0, 
      "\n---")

"""# **Step2: Load & Display Data**"""
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
print("---\n","Data loaded!","\n---")
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
show_random_examples(x_train, y_train, y_train)

"""# **Step3: Build Model**"""
# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)
print("---\n","Data Augmentation Applied!","\n---")

# Define the model structures as functions
# Model 1
def model1(dropout_rate=0.3):
    model = models.Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(96, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])
    # Model Summary
    model.summary()
    return model

# Model 2
def model2(dropout_rate=0.3):
    model = models.Sequential([
        # Layer 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        # Layer 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate + 0.1),
        # Layer 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate + 0.2),
        # Layer 4
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate + 0.2),
        Dense(10, activation='softmax')
    ])
    # Model Summary
    model.summary()
    return model

# Function that Train one Model with specific parameters
def train_model(model, model_name="model", batch=128, epochs=10, patience=3, optimizer='adam', dropout_rate=0.3, learning_rate=0.002):
    model_ = model
    if optimizer == "Adam":
        optimizer_ = Adam(learning_rate=learning_rate)
    elif optimizer == "Nadam":
        optimizer_ = Nadam(learning_rate=learning_rate)
    elif optimizer == "RMSprop":
        optimizer_ = RMSprop(learning_rate=learning_rate)
    elif optimizer == "SGD":
        optimizer_ = SGD(learning_rate=learning_rate)
    else:
        optimizer_ = 'adam' # adam with no learning-rate if no value given
    # Compile the model
    model_.compile(
        optimizer=optimizer_,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("---\n","Model Compiled!","\n---")
    # Callbacks
    early_stopping_ = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    lr_scheduler_ = LearningRateScheduler(lambda epoch: learning_rate * (0.5 ** (epoch // 10)))
    checkpoint_path = f'{model_name}_{batch}b_{epochs}e_{optimizer}.keras'
    checkpoint_ = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    # Train
    print("---\n", f"Trainning: {model_name}", 
          "\n", f"batch_size={batch}", 
          "\n", f"epochs={epochs}", 
          "\n", f"patience={patience}", 
          "\n", f"optimizer={optimizer}", 
          "\n", f"learning_rate={learning_rate}", 
          "\n", f"dropout_rate={dropout_rate}", "\n---")
    history_ = model_.fit(
        datagen.flow(x_train, y_train, batch_size=batch),
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[lr_scheduler_, early_stopping_, checkpoint_],
        verbose=1
    )
    print("---\n","Model trained!","\n---")
    # Save model
    print("---\n", 
          f"Model saved at: {os.path.abspath(f'{model_name}_{batch}b_{epochs}e_{optimizer}.keras')}",
          "\n---")
    # Save the history
    with open(f'{model_name}_{batch}b_{epochs}e_{optimizer}.json', 'w') as f:
        json.dump(history_.history, f)
    print("---\n",
          f"History saved at: {os.path.abspath(f'{model_name}_{batch}b_{epochs}e_{optimizer}.json ')}",
          "\n---")
    # Return the trained model and its validation accuracy for comparison
    val_accuracy = max(history_.history['val_accuracy'])
    return model_, history_.history, val_accuracy

"""# **Step4: Find the best Hyper-parameters**"""
# Function to find and save the best model and its training history
def find_best_model():
    # model 1
    # 10 epochs
    train_model(model1(), model_name="model1", batch=128, epochs=10) # 68%
    train_model(model1(), model_name="model1", batch=200, epochs=10) # 69%
    # 20 epochs
    train_model(model1(), model_name="model1", batch=32, epochs=20) # 71%
    train_model(model1(), model_name="model1", batch=64, epochs=20) # 73%
    # model 2
    # 10 epochs
    train_model(model2(), model_name="model2", batch=128, epochs=10) # 79%
    train_model(model2(), model_name="model2", batch=200, epochs=10) # 70% (overfitting)
    # 20 epochs
    train_model(model2(), model_name="model2", batch=64, epochs=20) # 83% - best model
    train_model(model2(), model_name="model2", batch=128, epochs=20) # 81%
# find_best_model()

train_model(model2(), model_name="model2", batch=64, epochs=20, optimizer='Adam', learning_rate=0.010) # 83% - best model

# Function to compare different models
def compare_data():
    models10_names = [
        "model1_128b_10e_adam",
        "model1_200b_10e_adam",
        "model2_128b_10e_adam",
        "model2_200b_10e_adam"
    ]
    histories_ = []
    for name in models10_names:
        with open(f'models/{name}.json', 'r') as f:
            histories_.append(json.load(f))
    print("---\n","All histories loaded successfully!","\n---")
    colors = plt.cm.tab10(range(len(models10_names)))
    plt.figure(figsize=(12, 6))
    # Plot training and validation accuracy
    plt.subplot(1, 1, 1)
    for history, name, color in zip(histories_, models10_names, colors):
        plt.plot(history['accuracy'], linestyle='--', color=color, label=f'{name} - Training')
        plt.plot(history['val_accuracy'], linestyle='-', color=color, label=f'{name} - Validation')
    plt.title('Training and Validation Accuracy for 10 epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    models20_names = [
        "model1_32b_20e_adam",
        "model1_64b_20e_adam",
        "model2_64b_20e_adam",
        "model2_128b_20e_adam"
    ]
    # Load all histories
    histories_ = []
    for name in models20_names:
        with open(f'models/{name}.json', 'r') as f:
            histories_.append(json.load(f))
    print("---\n","All histories loaded successfully!","\n---")
    colors = plt.cm.tab10(range(len(models20_names)))
    plt.figure(figsize=(12, 6))
    # Plot training and validation accuracy
    plt.subplot(1, 1, 1)
    for history, name, color in zip(histories_, models20_names, colors):
        plt.plot(history['accuracy'], linestyle='--', color=color, label=f'{name} - Training')
        plt.plot(history['val_accuracy'], linestyle='-', color=color, label=f'{name} - Validation')
    plt.title('Training and Validation Accuracy for 20 epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

compare_data()

# best model & hyper parameters:
best_model = "model2_64b_20e_adam"
print("---\n",
      "Best Model Saved!",
      "\nData Augmentation",
      "\nmodel 2",
      "\nepochs 20",
      "\nbatch 64",
      "\npatience 3",
      "\noptimizer adam",
      "\ndrop out rate 0.3",
      "\nlearning rate 0.010",
      "\n=> accuracy rate 83%",
      "\n---")

"""# **Step5: Display The best model Results**"""
# function that display relevent statistics
def display_data(model_name="model"):
    ## Load Data
    model = load_model(f'models/{model_name}.keras')
    print("---\n","Model loaded successfully!","\n---")
    # Load the history from the saved JSON file
    with open(f'models/{model_name}.json', 'r') as f:
        history_data = json.load(f)
    print("---\n","History loaded successfully!","\n---")
    ## Get data
    # Get the predictions (probabilities)
    y_pred = model.predict(x_test)
    # Convert the predicted probabilities to class labels
    y_pred_classes = np.argmax(y_pred, axis=1)
    # Convert the true labels from one-hot encoding to class labels
    y_test_classes = np.argmax(y_test, axis=1)
    ## Display Data
    # Show Classification Report
    print("---\n",
          "Classification Report:\n", 
          classification_report(y_test_classes, y_pred_classes),
          "\n---")
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
display_data(model_name=best_model)