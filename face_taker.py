import cv2
import dlib
import numpy as np
import face_recognition
import os
import json
import sys
import time

# Define file path for names.json
file_path = "names.json"

# Load Dlib's HOG-based face detector
detector = dlib.get_frontal_face_detector()

# Ensure storage folders exist
if not os.path.exists("images"):
    os.makedirs("images")

# Ensure names.json exists
if not os.path.exists(file_path):
    with open(file_path, "w") as f:
        json.dump({}, f)

# Read name-ID mapping
try:
    with open(file_path, "r") as f:
        name_map = json.load(f)
except json.JSONDecodeError:
    name_map = {}  # Handle corrupted JSON file

# Get enrollment number from command-line argument
if len(sys.argv) > 1:
    enrollment_number = sys.argv[1]
else:
    # Fallback to input if not provided as argument
    enrollment_number = input("Enter your enrollment number: ")

if not enrollment_number:
    print("Error: Enrollment number cannot be empty.")
    sys.exit(1)

folder_path = os.path.join("images", enrollment_number)
os.makedirs(folder_path, exist_ok=True)

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    sys.exit(1)

print(f"Starting face capture for enrollment number: {enrollment_number}")
print(f"Will capture 120 images. Please look at the camera and move your head slightly.")

# Allow camera to initialize
time.sleep(1)

count = 0
while count < 120:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Ensure face coordinates are within frame bounds
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            continue

        face_img = frame[y:y+h, x:x+w]
        filename = os.path.join(folder_path, f"{count}.jpg")
        cv2.imwrite(filename, face_img)
        count += 1

        # Print progress
        print(f"Captured image {count}/120", end="\r")

        # Draw bounding box (only in interactive mode)
        if not len(sys.argv) > 1:  # Only display if not running in script mode
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing {count}/120", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Face Capture", frame)

    # Only show window and check for ESC key if running interactively
    if not len(sys.argv) > 1:
        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit early
            break
    
    # Slow down capture rate slightly for better diversity of images
    time.sleep(0.05)

cap.release()
if not len(sys.argv) > 1:  # Only if running interactively
    cv2.destroyAllWindows()

# Save enrollment number to JSON
name_map[enrollment_number] = enrollment_number
with open(file_path, "w") as f:
    json.dump(name_map, f, indent=4)

print(f"\n[OK] Saved {count} images for enrollment number {enrollment_number}.")
sys.exit(0)  # Exit with success code