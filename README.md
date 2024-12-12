
### **AI CNN Project: CIFAR-10 Image Classification with Convolutional Neural Networks**

---

#### **Overview**
This project focuses on classifying images from the CIFAR-10 dataset using Convolutional Neural Networks (CNNs). The dataset consists of 60,000 images in 10 classes, with 6,000 images per class. The goal is to train a CNN to accurately classify the images and compare two models—one underperforming and another optimized with techniques like early stopping and data augmentation.

---

Google Colab link: https://colab.research.google.com/drive/17LbsMVHmQ2Ums28TQsUy0JKLLWwftZst?usp=sharing 
Video Demo: https://youtu.be/OdLN-6J_ZAU 

---

## **Step 1: Project Environment**
- The first step of our project was to import the necessary packages to build and train the models effectively.
- We used **TensorFlow** and **Keras** as the main frameworks for constructing and training the Convolutional Neural Networks.
  - These tools provide pre-built functions for defining layers, compiling models, and optimizing training.
- For numerical operations, we relied on **NumPy**, which is essential for handling arrays and performing calculations on the image data.
- To visualize the training process and results, we used **Matplotlib**, a plotting library that allowed us to create graphs showing accuracy and loss over time.
- Additionally, packages like **random** (for sampling) and **os** (for managing file paths) ensured smooth handling of the dataset and results.

---

## **Step 2: Data Loading**
- Before training our models, we processed the data to ensure it was ready for learning.
  - The dataset was split into two parts:
    - **Training set**: Used to teach the model how to recognize patterns.
    - **Testing set**: Used to evaluate how well the model can classify new, unseen images.
- The pixel values of the images were normalized:
  - Original values ranged from 0 to 255.
  - These were scaled down to a range of 0 to 1 by dividing by 255, making the data easier to work with and improving learning speed.
- A function was created to visualize random samples from the dataset along with their true labels and predicted labels:
  - This step ensured the data was loaded correctly and provided a way to check model performance visually.
- **Data augmentation** was applied to enhance model performance:
  - Random transformations such as flipping, rotating, and shifting were applied to create variations of the images.
  - This exposed the model to a wider variety of data, improving robustness.

---

## **Step 3: Model Building**
- After processing the dataset, we built and tested two CNN architectures:
  ### **Model 1**
  - A simple design with a few convolutional layers that extract basic features like edges and shapes.
  - Pooling layers reduced the size of the image data, making computations more efficient.
  - Dropout layers were added to prevent overfitting by randomly disabling some neurons during training, encouraging the model to learn generalized patterns.
  
  ### **Model 2**
  - A more advanced design with additional features:
    - **Batch normalization** was used to adjust the data flow, stabilizing learning and speeding up the training process.
    - More layers allowed it to learn more complex features.
  - Similar to Model 1, it ended with a dense layer to assign probabilities for the 10 categories and make predictions.

- Both models were trained using the **Adam optimizer**, which adjusts the learning rate during training to find the optimal weights more efficiently.
- **Early stopping** was implemented:
  - Training stopped when validation loss did not improve for a set number of epochs to avoid overfitting and save time.
- **Model checkpoints** were used to save the best model whenever it achieved the highest validation accuracy.
- The models were trained for several epochs with various settings to identify the best parameter combination.
- Finally, the best model and its training history were saved for evaluation in later stages.

---

## **Step 4: Hyperparameter Comparison**
- To find the best training conditions, we trained eight models with different hyperparameter combinations:
  - Parameters included batch size, learning rate, number of epochs, and dropout rates.
- Multiple combinations were tested:
  - Batch sizes: 64, 128, 200.
  - Learning rates: 0.002, 0.01.
  - Dropout rates: 0.3.
- Results:
  - **Model 2** performed best with:
    - Batch size: 64.
    - Learning rate: 0.01.
    - Training epochs: 20.
  - This configuration achieved the highest validation accuracy.
- **Model Comparison**:
  - Model 2 outperformed Model 1, particularly in longer training sessions, due to its advanced architecture.
  - However, Model 1 was better at avoiding overfitting during shorter training sessions, making it a good option for quick tasks with limited resources.
- In the end, Model 2 was chosen as the better option after fine-tuning its hyperparameters.

---

## **Step 5: Results of the Best Model**
- The best-performing model, **Model 2**, achieved an accuracy of **83%** on the test set:
  - It correctly classified 83% of unseen images.
- Detailed evaluation metrics, such as **precision**, **recall**, and **F1-score**, provided insights into the model’s performance for each category.
- When tested on 10 random images, the model correctly predicted all of them, demonstrating strong generalization to new data.
- The **confusion matrix** revealed areas of improvement:
  - The model struggled with visually similar categories like cats and dogs.
- Although the model showed strong performance, further refinements could improve its ability to distinguish between similar categories.

---

## **Future Work**
- Explore advanced architectures like **ResNet** or **EfficientNet** for better feature extraction.
- Apply **transfer learning** to leverage pre-trained models, saving time and enhancing accuracy.
- Focus on improving the model’s ability to distinguish between challenging categories such as cats and dogs.

