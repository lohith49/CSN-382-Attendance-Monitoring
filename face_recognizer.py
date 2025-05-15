import cv2
import face_recognition
import pickle
import numpy as np
import json
import csv
import os
from datetime import datetime, timedelta

# Load trained SVM model
with open("trained_svm.pkl", "rb") as f:
    svm_model = pickle.load(f)

# Load stored names
with open(r"C:\Users\lohit\OneDrive\Desktop\Automated_Checkin_ML_Project\backend\names.json", "r") as f:
    name_map = json.load(f)

# CSV file to track attendance
CSV_FILE = "attendance.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Enrollment No.", "Timestamp", "Status"])

# Open webcam with DirectShow to avoid buffer lag
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

previous_status = {}  # Tracks "IN"/"OUT" status
last_seen_time = {}  # Tracks last timestamp for each person

while cap.isOpened():  # Ensures continuous loop
    ret, frame = cap.read()
    if not ret:
        continue  # Skip frame if not captured

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Use "cnn" if needed
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Predict using SVM
        probas = svm_model.predict_proba([face_encoding])[0]
        max_similarity = np.max(probas)
        best_match = svm_model.classes_[np.argmax(probas)]

        # Apply threshold
        if max_similarity < 0.5:
            name = "Unknown"
        else:
            name = name_map.get(best_match, "Unknown")

        # Get current timestamp
        current_time = datetime.now()

        # Default status is last known status, else "OUT"
        current_status = previous_status.get(name, "OUT")

        # Toggle only if more than 1 minute has passed
        if name in last_seen_time:
            time_difference = (current_time - last_seen_time[name]).total_seconds()
            if time_difference >= 60:  # More than 1 min, toggle
                current_status = "IN" if previous_status.get(name, "OUT") == "OUT" else "OUT"
                previous_status[name] = current_status
                last_seen_time[name] = current_time

                # Log attendance
                if name != "Unknown":
                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([name, current_time.strftime("%Y-%m-%d %H:%M:%S"), current_status])
        else:
            last_seen_time[name] = current_time  # First-time detection

        # Draw bounding box and label
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"{name} ({current_status})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show frame
    cv2.imshow("Face Recognition", frame)

    # Break on 'ESC' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
