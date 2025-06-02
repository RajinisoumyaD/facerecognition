# 🧠 Face Recognition with PCA, Fisherfaces, and MLP

This project implements **face recognition** on grayscale face images using a combination of dimensionality reduction (PCA/Eigenfaces and Fisherfaces/LDA) and classification via a Multi-layer Perceptron (MLP/ANN). The code is written in Python using scikit-learn, OpenCV, and matplotlib, and is suitable for experimentation with standard datasets (like ORL) or your own.

---

## 📁 Dataset

- **Dataset Used**: ORL Face Dataset (or your own custom dataset inside `dataset/att_faces`)
- **Total Images**: 450
- **Number of Classes**: 10 (example: Aamir, Ajay, Akshay, Alia, etc.)
- **Image Size**: All images are resized to 100x100 pixels in grayscale by default.
- **Format**: Each person's images should reside in a separate subdirectory under `dataset/att_faces`, and should be in `.pgm` format.

---

## 🖼️ Sample Faces

The code randomly displays 10 face images with their labels to verify the dataset integrity and class balance.

---

## 🏗️ How It Works

1. **Data Loading**
   - Loads all grayscale images, resizes as needed, and flattens them into vectors.
   - Each image is assigned a numeric label according to its containing directory.

2. **Dimensionality Reduction**
   - **PCA (Eigenfaces)**: Reduces dimensionality by extracting the principal components of the face data. Number of components (`k`) is tunable; best results are found empirically.
   - **Fisherfaces (LDA)**: Optionally, Fisherfaces are computed after PCA for improved class separation.

3. **Model Training**
   - The dataset is split into train and test sets (typically 60% train, 40% test).
   - An MLP classifier is trained on the PCA-reduced feature vectors.

4. **Imposter Testing**
   - Imposter images (random noise or mixtures of faces) are generated to test false positive rates and model robustness.

5. **Evaluation and Visualization**
   - Accuracy is measured for multiple values of `k` (number of PCA components).
   - Eigenfaces and the mean face are visualized and saved as PNG images.

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

## 🚀 Usage

1. **Install Dependencies**
   ```sh
   pip install numpy opencv-python scikit-learn matplotlib
   ```

2. **Prepare Dataset**
   - Place the ORL or your own dataset in `dataset/att_faces/`, with each class in its own folder.

3. **Run the Code**
   - You can use either the script or the Jupyter notebook:
   ```sh
   python facerecognition.py
   ```
   - Or open and run `facerecognition.ipynb` in Jupyter.

4. **Output**
   - The script will print accuracy scores, display sample images, and save plots for eigenfaces and mean face.

---

## 📊 Visualizations

- **Eigenfaces**: The most significant features extracted from the dataset, shown as grayscale images.
- **Mean Face**: The average of all faces in the dataset for reference.
- **Accuracy vs. PCA Component Plot**: Shows how accuracy changes as you vary the number of principal components.

---

## 📝 Notes

- You can experiment with different PCA component values, MLP hyperparameters, or datasets.
- The code includes both class and function-based organization for clarity and extensibility.
- Imposter testing ensures that the model isn't easily fooled by non-face or outlier images.

---

## 📂 File Structure

```
facerecognition/
├── facerecognition.py        # Main script with all logic
├── facerecognition.ipynb     # Jupyter Notebook version
├── dataset/
│   └── att_faces/            # Place dataset here
├── eigenfaces.png            # Visualization output
├── meanface.png              # Visualization output
└── README.md
```

---

## 🙏 Acknowledgements

- ORL Face Dataset (AT&T Laboratories Cambridge)
- scikit-learn, OpenCV, matplotlib

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to open issues or contribute to this project!
