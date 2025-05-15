import os

# Paths
IMAGE_DIR = "images"
ENCODINGS_FILE = "encodings.pkl"
SVM_MODEL_FILE = "trained_svm.pkl"
ATTENDANCE_CSV = "attendance.csv"

# Face Recognition Threshold
SIMILARITY_THRESHOLD = 0.5  # If similarity < 0.5, mark as "Unknown"

# Camera Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
