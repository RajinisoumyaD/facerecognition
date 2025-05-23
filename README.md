# 🧠 Face Recognition with PCA, Fisherfaces, and MLP

This project implements **face recognition** by combining dimensionality reduction techniques (PCA and Fisherfaces) with a **Multi-layer Perceptron (MLP)** classifier.

---

## 📁 Dataset

- **Dataset Used**: ORL Face Dataset (or your own custom dataset inside `dataset/att_faces`)
- **Total Images**: 450
- **Number of Classes**: 10 (example: Aamir, Ajay, Akshay, Alia, etc.)
- **Image Size**: Resized to 100x100 pixels in grayscale

---

## 🖼️ Sample Faces

Displays 10 randomly selected face images with corresponding labels from the dataset to ensure data integrity.

---

## 🔢 Dimensionality Reduction

- Principal Component Analysis (**PCA**) for feature extraction (Eigenfaces)
- Top 150 Eigenfaces computed from 337 total eigenfaces
- Linear Discriminant Analysis (**Fisherfaces**) applied after PCA for better class separation

---

## 🧪 Model Evaluation

### Train-Test Split:
- **Training Samples**: 270  
- **Testing Samples**: 180  

### Accuracy with Different PCA Components:

| PCA Components | Accuracy (%) |
|----------------|--------------|
| 20             | 41.11        |
| 30             | 40.00        |
| 40             | 17.22        |
| 50             | 46.11        |
| 60             | 62.22 ✅     |
| 70             | 28.89        |
| 80             | 48.89        |

> Best performance at 60 PCA components with **62.22% accuracy**

---

## 🧠 Model Architecture

- **Classifier**: Multi-layer Perceptron (MLP)
- **Structure**:
  - One hidden layer with 100 neurons
  - Activation function: ReLU
  - Optimizer: Adam
  - Maximum iterations: 1000
  - Early stopping enabled to prevent overfitting

---


