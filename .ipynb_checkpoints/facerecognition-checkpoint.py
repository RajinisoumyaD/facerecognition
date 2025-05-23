import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Function to load images and create face database
def load_face_database(dataset_path, img_height=112, img_width=92):
    face_db = []
    labels = []
    person_id = 0
    
    for person_folder in sorted(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person_folder)
        
        # Skip if not a directory
        if not os.path.isdir(person_path):
            continue
            
        print(f"Loading images from {person_folder}")
        
        for img_file in sorted(os.listdir(person_path)):
            if img_file.endswith('.pgm'):
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize if needed
                    if img.shape != (img_height, img_width):
                        img = cv2.resize(img, (img_width, img_height))
                    
                    # Flatten the image to a column vector
                    img_vector = img.flatten()
                    face_db.append(img_vector)
                    labels.append(person_id)
        
        person_id += 1
    
    return np.array(face_db, dtype=np.float32), np.array(labels)

# PCA implementation
class PCA_FaceRecognition:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_face = None
        self.eigenfaces = None
        self.feature_vector = None
        self.face_signatures = None
    
    def fit(self, face_db):
        # Step 2: Mean Calculation
        self.mean_face = np.mean(face_db, axis=0)
        
        # Step 3: Mean Zero
        mean_aligned_faces = face_db - self.mean_face
        
        # Step 4: Calculate surrogate covariance matrix
        # C = (mean_aligned_faces.T @ mean_aligned_faces) / (mean_aligned_faces.shape[0] - 1)
        # Normalized dot product for numerical stability
        C = np.dot(mean_aligned_faces, mean_aligned_faces.T) / (mean_aligned_faces.shape[0] - 1)
        
        # Step 5: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Step 6: Select top k eigenvectors
        if self.n_components is None:
            # Use enough components to explain 95% of the variance
            explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
            cumulative_variance = np.cumsum(explained_variance_ratio)
            self.n_components = np.argmax(cumulative_variance >= 0.95) + 1
        
        self.feature_vector = eigenvectors[:, :self.n_components]
        
        # Step 7: Generate eigenfaces
        # Project the mean-aligned faces onto the feature vector
        self.eigenfaces = np.dot(mean_aligned_faces.T, self.feature_vector)
        
        # Normalize eigenfaces
        for i in range(self.eigenfaces.shape[1]):
            self.eigenfaces[:, i] = self.eigenfaces[:, i] / np.linalg.norm(self.eigenfaces[:, i])
        
        # Step 8: Generate signatures for each face
        self.face_signatures = np.dot(mean_aligned_faces, self.eigenfaces)
        
        return self.face_signatures
    
    def transform(self, faces):
        # For new face images:
        # Step 1-2: Subtract mean face
        mean_aligned_faces = faces - self.mean_face
        
        # Step 3-4: Project onto eigenfaces
        return np.dot(mean_aligned_faces, self.eigenfaces)

# Evaluate model with different k values
def evaluate_model_with_different_k(face_db, labels, k_values, test_size=0.4, random_state=42):
    accuracies = []
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        face_db, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    for k in k_values:
        print(f"Training with k = {k}")
        
        # Apply PCA
        pca = PCA_FaceRecognition(n_components=k)
        X_train_pca = pca.fit(X_train)
        X_test_pca = pca.transform(X_test)
        
        # Train ANN
        ann = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=random_state
        )
        
        ann.fit(X_train_pca, y_train)
        
        # Test the model
        y_pred = ann.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        
        print(f"Accuracy with k={k}: {accuracy:.4f}")
    
    return k_values, accuracies

# Test with imposters
def test_with_imposters(pca, ann, imposters, threshold=None):
    # Apply PCA transformation to imposters
    imposter_features = pca.transform(imposters)
    
    if threshold is None:
        # If no threshold is provided, just use the classifier's prediction
        predictions = ann.predict(imposter_features)
        # All should be classified as "unknown"
        return predictions
    else:
        # Calculate distances to all known faces and use a threshold
        probabilities = ann.predict_proba(imposter_features)
        max_probs = np.max(probabilities, axis=1)
        # If max probability is below threshold, classify as "unknown"
        predictions = np.where(max_probs < threshold, -1, ann.predict(imposter_features))
        return predictions, max_probs

# Main function
def main():
    # Check if dataset exists, if not download it
    if not os.path.exists("face_dataset"):
        download_and_extract_dataset()
    
    # Load the dataset
    dataset_path = "face_dataset/att_faces"
    face_db, labels = load_face_database(dataset_path)
    
    print(f"Loaded {face_db.shape[0]} images with shape {face_db.shape[1]}")
    
    # a) Evaluate with different k values
    k_values = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70]
    k_values, accuracies = evaluate_model_with_different_k(face_db, labels, k_values)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o')
    plt.title('Face Recognition Accuracy vs. Number of Components (k)')
    plt.xlabel('Number of Components (k)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('accuracy_vs_k.png')
    plt.show()
    
    # b) Test with imposters
    # For this part, we'll use the best k value from previous experiment
    best_k = k_values[np.argmax(accuracies)]
    print(f"Best k value: {best_k} with accuracy: {max(accuracies):.4f}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        face_db, labels, test_size=0.4, random_state=42, stratify=labels
    )
    
    # Apply PCA with best k
    pca = PCA_FaceRecognition(n_components=best_k)
    X_train_pca = pca.fit(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Train ANN
    ann = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    
    ann.fit(X_train_pca, y_train)
    
    # Create synthetic imposters by averaging faces or using random noise
    num_imposters = 20
    np.random.seed(42)
    
    # Method 1: Random noise
    noise_imposters = np.random.rand(num_imposters, face_db.shape[1]) * 255
    
    # Method 2: Mixtures of existing faces with distortions
    mixtures = []
    for _ in range(num_imposters):
        idx1, idx2 = np.random.choice(range(len(X_train)), 2, replace=False)
        mixture = (X_train[idx1] + X_train[idx2]) / 2
        # Add some random noise
        mixture += np.random.randn(*mixture.shape) * 20
        mixtures.append(mixture)
    
    mixture_imposters = np.array(mixtures)
    
    # Combine both types of imposters
    imposters = np.vstack([noise_imposters, mixture_imposters])
    
    # Find a suitable threshold using the training data
    train_probs = ann.predict_proba(X_train_pca)
    max_train_probs = np.max(train_probs, axis=1)
    threshold = np.percentile(max_train_probs, 5)  # 5th percentile as threshold
    
    print(f"Using confidence threshold: {threshold:.4f}")
    
    # Test with genuine test samples
    test_probs = ann.predict_proba(X_test_pca)
    max_test_probs = np.max(test_probs, axis=1)
    y_pred_with_threshold = np.where(max_test_probs < threshold, -1, ann.predict(X_test_pca))
    
    # Count how many genuine samples are rejected as imposters
    false_rejects = np.sum(y_pred_with_threshold == -1)
    false_reject_rate = false_rejects / len(y_test)
    print(f"False Reject Rate: {false_reject_rate:.4f} ({false_rejects} out of {len(y_test)})")
    
    # Test with imposters
    imposter_preds, imposter_probs = test_with_imposters(pca, ann, imposters, threshold)
    
    # Count how many imposters are correctly rejected
    true_rejects = np.sum(imposter_preds == -1)
    true_reject_rate = true_rejects / len(imposters)
    print(f"True Reject Rate: {true_reject_rate:.4f} ({true_rejects} out of {len(imposters)})")
    
    # Visualize some eigenfaces
    plt.figure(figsize=(12, 4))
    for i in range(min(5, best_k)):
        plt.subplot(1, 5, i+1)
        eigenface = pca.eigenfaces[:, i].reshape(112, 92)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f'Eigenface {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('eigenfaces.png')
    plt.show()
    
    # Visualize the mean face
    plt.figure(figsize=(4, 4))
    plt.imshow(pca.mean_face.reshape(112, 92), cmap='gray')
    plt.title('Mean Face')
    plt.axis('off')
    plt.savefig('mean_face.png')
    plt.show()
    
    # Plot imposter vs genuine confidence scores
    plt.figure(figsize=(10, 6))
    plt.hist(max_test_probs, bins=20, alpha=0.5, label='Genuine')
    plt.hist(imposter_probs, bins=20, alpha=0.5, label='Imposters')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
    plt.xlabel('Confidence Score (Max Probability)')
    plt.ylabel('Count')
    plt.title('Genuine vs Imposter Confidence Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig('confidence_scores.png')
    plt.show()

if __name__ == "__main__":
    main()