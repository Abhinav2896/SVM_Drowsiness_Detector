import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load your dataset
data = pd.read_csv("C:/Users/hp/Desktop/Drowsiness-1/drowsiness_dataset.csv")

# Show column names for verification
print("[INFO] Dataset columns:", data.columns)

# Use correct column names from your CSV
X = data[['EAR']]  # EAR as the feature
y = data['label']  # Label (awake/drowsy)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("[INFO] Accuracy:", accuracy_score(y_test, y_pred))
print("[INFO] Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "drowsiness_svm_model.pkl")
print("[INFO] Model saved as drowsiness_svm_model.pkl")