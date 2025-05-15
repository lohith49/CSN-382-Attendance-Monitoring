import os
import pickle
import numpy as np
import face_recognition
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load stored images
data = []
labels = []

for student in os.listdir(r"C:\Users\lohit\OneDrive\Desktop\ML_Project_CSN_382\ML_Project_CSN_382\Automated_Checkin_ML_Project\backend\images"):
    folder_path = os.path.join(r"C:\Users\lohit\OneDrive\Desktop\ML_Project_CSN_382\ML_Project_CSN_382\Automated_Checkin_ML_Project\backend\images", student)

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = face_recognition.load_image_file(img_path)

        face_encodings = face_recognition.face_encodings(img)
        if face_encodings:
            data.append(face_encodings[0])
            labels.append(student)

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split the data for evaluation
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Train the SVM model
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
with open("trained_svm.pkl", "wb") as f:
    pickle.dump(svm_model, f)

print("\nTraining complete. Model saved as trained_svm.pkl")
