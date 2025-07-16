# **Dog Breed Classification using Custom-Built CNNs**

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbDB2d250c25oZzJqYjR0c2NlM3c0b3N6aHl0d2Z0b3ZkYjZqdjZqdyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7527pa7qs9kCG78A/giphy.gif" width="400">
</p>

This project explores the fascinating challenge of fine-grained image classification by building, training, and evaluating multiple Convolutional Neural Network (CNN) architectures from scratch to identify 70 different dog breeds. The primary goal was to gain a deep, practical understanding of how architectural choices impact a model's ability to learn subtle visual distinctions.

---

## 🚀 **Project Overview**

Classifying dog breeds is notoriously difficult due to high inter-class similarity (e.g., Alaskan Malamute vs. Siberian Husky) and high intra-class variability (e.g., different grooming styles of a Poodle). This project tackles this challenge head-on by demonstrating the process of building robust deep learning models capable of learning hierarchical features directly from images.

### **Key Features**
* **Built from Scratch:** Three distinct CNN architectures were developed without relying on pre-trained models to analyze the impact of design choices.
* **Comparative Analysis:** A systematic comparison of a baseline CNN, a VGG-style network, and a ResNet-inspired network.
* **Data Augmentation:** Implemented a robust data augmentation pipeline to improve model generalization and prevent overfitting.
* **In-depth Evaluation:** Analyzed model performance using accuracy, loss curves, and confusion matrices to identify common misclassifications.

---

## 🛠️ **Tech Stack**

This project was developed using the following technologies:

* **Language:** Python
* **Frameworks:** TensorFlow 2.x, Keras
* **Libraries:** Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn
* **Environment:** Google Colaboratory (Colab) with GPU acceleration.

<p>
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"/>
<img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
<img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=google-colab&logoColor=black" alt="Colab"/>
</p>

---

## 📂 **Dataset**
The model was trained on a curated subset of the famous [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset), containing images for 70 distinct dog breeds.

The data was preprocessed and structured as follows:

* **Image Size:** Resized to 224x224 pixels.
* **Data Split:** Divided into training, validation, and test sets.
* **Augmentation:** The training data was augmented with random rotations, zooms, flips, and shifts to create a more diverse dataset.

---

## 🧠 **Model Architectures**
Three different CNN models were designed and trained to compare their performance.

### **1. Model Architecture 1: Baseline CNN**
A simple, shallow network with three convolutional blocks to establish a performance baseline.
* **Structure:** `Conv -> Pool -> Conv -> Pool -> Conv -> Pool -> Flatten -> Dense`
* **Purpose:** To demonstrate the foundational feature extraction capabilities of a basic CNN.

### **2. Model Architecture 2: VGG-Inspired Network**
A deeper network inspired by the VGG architecture, using stacked `3x3` convolutions and Batch Normalization to learn more complex features and stabilize training.
* **Structure:** Deeper stacks of `Conv` layers with `BatchNormalization` after each block.
* **Purpose:** To test if increasing depth directly improves performance on this fine-grained task.

### **3. Model Architecture 3: ResNet-Inspired Network**
An efficient, deep network that incorporates **Global Average Pooling** instead of a large, dense Flatten layer. This design is inspired by modern architectures like ResNet and helps significantly in reducing overfitting and the number of parameters.
* **Structure:** Deep convolutional blocks followed by `GlobalAveragePooling2D`.
* **Purpose:** To build a more parameter-efficient and robust model that can generalize better.

---

## 📈 **Results & Conclusion**
After training and evaluation, the performance of the three models was as follows:

* 🥇 **Model 3: ResNet-Inspired Network (Test Accuracy: ~50%)**
    This was the best-performing model. The use of **Global Average Pooling** proved highly effective in improving generalization and achieving the highest accuracy.

* 🥈 **Model 1: Baseline CNN (Test Accuracy: ~29%)**
    Performed reasonably for a simple model but struggled to differentiate between visually similar breeds.

* 🥉 **Model 2: VGG-Inspired Network (Test Accuracy: ~15%)**
    Performed poorly. This provided a key insight: simply increasing network depth without proper architectural design (like skip connections) can hurt performance.

### **Key Takeaway**
This project successfully demonstrates that for complex, fine-grained classification tasks, a **well-thought-out architecture** (like the ResNet-inspired model) is far more effective than simply increasing network depth. The results highlight the practical importance of modern design principles like **Global Average Pooling** in creating efficient and powerful deep learning models.

---

## 🚀 **How to Run**

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-link]
    cd [your-repo-name]
    ```

2.  **Open in Google Colab:**
    * Upload the `.ipynb` notebook file to your Google Drive.
    * Open it with Google Colaboratory.

3.  **Set up the dataset:**
    * Download the dataset and upload it to your Google Drive or use the Kaggle API within the notebook to fetch it.
    * Make sure to update the file paths in the notebook to point to your dataset location.

4.  **Run the cells:**
    * Ensure the runtime is set to **GPU** for faster training.
    * Execute the cells sequentially to preprocess the data, build the models, train them, and evaluate their performance.

<br>

<p align="center">
Developed with ❤️ by Sriram Chowdary Alasakani
</p>
