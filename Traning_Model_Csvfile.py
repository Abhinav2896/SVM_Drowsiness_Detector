import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
csv_path = 'C:/Users/hp/Desktop/Drowsiness-1/drowsiness_dataset.csv'
data = pd.read_csv(csv_path)

# Features and labels
X = data[['EAR']].values  # EAR as feature
y = data['label'].values  # Labels (Awake = 1, Drowsy = 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)

# Print accuracy and classification report
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'drowsiness_svm_model.pkl')
