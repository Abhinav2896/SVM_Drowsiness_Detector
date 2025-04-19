import cv2
import dlib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib  # for saving the model

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate EAR (Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to extract features from a single frame (image)
def extract_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    features = []

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        
        # Calculate EAR for both eyes
        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        
        # Combine EAR values (you could add more features here)
        features.append([ear_left, ear_right])
        
    return features

# Function to extract features from all images in the dataset
def extract_all_features(csv_path):
    # Load CSV data (assuming it has 'EAR' and 'label' columns)
    data = pd.read_csv(csv_path)
    
    X = []
    labels = []
    
    for i, row in data.iterrows():
        # For simplicity, assume 'frame' is the image loaded from the file paths
        frame = cv2.imread(row['file_path'])  # Assuming you have the file paths in your CSV
        features = extract_features(frame)
        
        if features:
            X.append(np.mean(features, axis=0))  # Averaging the features per face (you could improve this)
            labels.append(row['label'])
    
    # Flip labels to correct the inverted classification issue
    labels = [1 - label for label in labels]  # Flip 0 to 1 and 1 to 0
    
    return np.array(X), np.array(labels)

# Main execution
if __name__ == "__main__":
    csv_path = "C:/Users/hp/Desktop/Drowsiness-1/drowsiness_dataset.csv"  # Path to your dataset CSV
    X, y = extract_all_features(csv_path)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize and train the SVM model
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    
    print("[INFO] Accuracy:", accuracy_score(y_test, y_pred))
    print("[INFO] Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, 'drowsiness_svm_model.pkl')
    print("[INFO] Model saved as drowsiness_svm_model.pkl")
