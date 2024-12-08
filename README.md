---

Google Colab link: https://colab.research.google.com/drive/1Xw-mPPhHEyIdskrkorfZe6oXQXiAQyV8?usp=sharing 

---

### **AI CNN Project Report: CIFAR-10 Image Classification with Convolutional Neural Networks**

---

#### **1. Overview**
This project focuses on classifying images from the CIFAR-10 dataset using Convolutional Neural Networks (CNNs). The dataset consists of 60,000 images in 10 classes, with 6,000 images per class. The goal is to train a CNN to accurately classify the images and compare two models—one underperforming and another optimized with techniques like early stopping and data augmentation.

---

### **2. Environment Setup**

#### **Packages Imported**
- **TensorFlow**: for building and training the neural network.
- **NumPy**: for numerical operations.
- **Matplotlib**: for data visualization.
- **Random and OS**: for random sample selection and file management.
- **Keras Components**: such as layers and callbacks for model building.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
```

---

### **3. Data Loading and Preprocessing**

#### **Loading the CIFAR-10 Dataset**
The CIFAR-10 dataset is loaded using TensorFlow’s built-in dataset loader. The training and testing sets are separated, and labels are one-hot encoded for compatibility with the softmax output layer of the model.

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

#### **Normalization**
All pixel values are scaled to the range [0, 1] by dividing by 255. This improves model convergence.

```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

#### **Displaying Random Data Samples**
A function displays random images with their true and predicted labels.

```python
def show_random_examples(x, y, p):
    # Displays 10 random images with true and predicted labels
```

---

### **4. Model 1: Bad Model (50% Accuracy and Overfitting)**

#### **Model Architecture**
- **Flatten Layer**: Flattens the input images to 1D arrays for fully connected layers.
- **Dense Layers**: Two dense layers with 3000 and 1000 neurons respectively.
- **Output Layer**: A softmax output layer with 10 neurons, corresponding to the 10 classes.

```python
model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

#### **Training the Model**
- **Optimizer**: SGD (Stochastic Gradient Descent).
- **Loss Function**: Categorical Crossentropy, as this is a multi-class classification task.
- **Metrics**: Accuracy.

```python
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)
```

#### **Evaluation**
- **Classification Report**: Displays precision, recall, and F1-score for each class.
- **Confusion Matrix**: Visualizes the model's performance in terms of true vs predicted labels.

```python
print("Classification Report: \n", classification_report(y_test_classes, y_pred_classes))
```

---

### **Model 2: Improved Model with Better Generalization**

#### **Model Architecture**

**1. Convolutional Layers:**
Convolutional layers are the core of a CNN. They allow the model to automatically learn spatial hierarchies in the data, such as edges, textures, and object shapes. In this model, multiple convolutional layers are stacked to progressively learn complex patterns in the CIFAR-10 images.

```python
model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),

    # Fully Connected Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

**Explanation of Layers:**

- **Conv2D Layers:** These layers apply convolutional filters to the input image, extracting important features. Each convolutional layer has an associated filter size (3x3), with a ReLU activation function to introduce non-linearity.
- **BatchNormalization:** This layer normalizes the output of each convolutional layer to stabilize training by reducing internal covariate shift. It helps the model learn faster and generalize better.
- **MaxPooling2D:** Max pooling is used to downsample the feature maps and reduce the spatial dimensions of the output. This helps reduce computational cost and improves the model's ability to generalize.
- **Dropout:** Dropout is used as a regularization technique to prevent overfitting. It randomly drops a fraction of the neurons during training, forcing the model to learn more robust features. Here, different dropout rates (0.3, 0.4, 0.5) are applied in various layers.
- **Dense Layer:** The fully connected layers at the end of the network serve as classifiers. The last dense layer has 10 neurons, corresponding to the 10 classes in the CIFAR-10 dataset. The output layer uses the softmax activation function to produce probabilities for each class.

---

#### **Data Augmentation**

One of the key improvements in Model 2 is the use of **data augmentation** to artificially increase the diversity of the training dataset. This is achieved by applying random transformations to the images during training, such as:

- **Rotation**: The images are rotated by up to 15 degrees.
- **Width/Height Shifting**: The images are randomly shifted horizontally and vertically by 10% of their dimensions.
- **Horizontal Flip**: The images are flipped horizontally with a probability of 50%.

```python
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)
```

By applying these transformations, the model is exposed to more variations of the images, which helps it generalize better to unseen data. The model becomes more robust to changes in the input images, such as slight rotations or translations.

---

#### **Early Stopping**

To further improve the model's generalization and prevent overfitting, **early stopping** is used. Early stopping halts training when the validation loss stops improving for a set number of epochs (patience). This helps prevent the model from continuing to train after it has already learned the most relevant patterns, avoiding unnecessary overfitting.

```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

Here, the model will stop training if the validation loss doesn’t improve for 5 consecutive epochs, and the weights of the best-performing model (lowest validation loss) will be restored.

---

#### **Model Checkpoints**

**Model Checkpoints** are used to save the best version of the model during training. This is useful when using early stopping, as the model can be restored to the point where it performed the best on the validation set.

```python
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
```

With this, the model with the highest validation accuracy is saved during the training process. After training, this saved model can be loaded to make predictions or further evaluate its performance.

---

#### **Compiling and Training the Model**

The model is compiled with the **Adam optimizer** (which adapts the learning rate during training) and **categorical crossentropy** loss function (suitable for multi-class classification).

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

Then, the model is trained with the augmented data using the **datagen.flow()** method, which applies data augmentation on the fly.

```python
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, checkpoint],
    verbose=1
)
```

- **Batch size**: 128 (number of samples processed before the model is updated).
- **Epochs**: 20 (the number of times the model will iterate over the entire training dataset).

The training also uses **early stopping** and **model checkpoints** to monitor validation performance and avoid overfitting.

---

#### **Model Evaluation**

After training, the best model is loaded, and predictions are made on the test set.

```python
model = load_model("best_model.keras")
```

The **classification report** and **confusion matrix** are generated to evaluate the model’s performance.

```python
print("Classification Report: \n", classification_report(y_test_classes, y_pred_classes))
```

The **confusion matrix** provides a detailed analysis of how well the model classifies each of the 10 CIFAR-10 classes.

```python
cm = confusion_matrix(y_test_classes, y_pred_classes)
```

---

### **Performance Analysis of Model 2**

- **Training and Validation Accuracy**: The model shows steady improvement in both training and validation accuracy. This indicates that the model is learning the relevant features without overfitting.
- **Confusion Matrix**: The confusion matrix shows how well the model performs on each class, helping to identify any classes where the model might still struggle.
- **Classification Report**: The precision, recall, and F1-score for each class will provide more detailed insights into the model's performance.

---

### **Conclusion**

**Model 2** significantly outperforms **Model 1** by incorporating:
- **Convolutional layers** for automatic feature extraction.
- **Data augmentation** to create more training data and help the model generalize better.
- **Dropout** and **batch normalization** to prevent overfitting and ensure stable training.
- **Early stopping** and **model checkpoints** to stop training when the model has reached its best performance and avoid overfitting.

With these improvements, Model 2 achieves a much higher accuracy (81%) and is better generalized compared to Model 1, which suffered from overfitting and poor performance.

---

---

### **6. Results and Analysis**

#### **Training and Validation Curves**
Plots of training and validation accuracy and loss show the progress of the model during training and the effect of early stopping.

#### **Confusion Matrix**
The confusion matrix is visualized to identify which classes the model is confusing most frequently.

---

### **7. Conclusion**
- **Model 1** demonstrated poor performance due to overfitting and lack of regularization.
- **Model 2** achieved significantly better accuracy (81%) by incorporating advanced techniques like convolutional layers, data augmentation, dropout, and early stopping.
- Future improvements could include further hyperparameter tuning, exploring different architectures, and incorporating more advanced regularization techniques.

---
