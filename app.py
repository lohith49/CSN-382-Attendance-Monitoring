from flask import Flask, render_template, Response, request, jsonify
import cv2
import face_recognition
import numpy as np
import pickle
import os
import base64
from flask_socketio import SocketIO
from datetime import datetime
import csv
import re
import json
import dlib
from sklearn.svm import SVC

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize CSV file if it doesn't exist
CSV_FILE = "attendance.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Enrollment No.", "Timestamp", "Status"])

# Load trained SVM model (global variable)
try:
    with open("trained_svm.pkl", "rb") as f:
        svm_model = pickle.load(f)
except FileNotFoundError:
    svm_model = None  # If the model doesn't exist yet, set to None

# Load known names from names.json
try:
    with open("names.json", "r") as f:
        name_map = json.load(f)
except FileNotFoundError:
    name_map = {}  # If names.json doesn't exist, initialize as empty dict

# Global variables for tracking presence
entry_times = {}  # enrollment_number -> entry timestamp when "IN"
last_detection_times = {}  # enrollment_number -> last detection timestamp
LEAVING_THRESHOLD = 30  # seconds to consider a user has left

# Track capture progress
capture_progress = {}  # Dictionary to store progress for each enrollment number

# Load Dlib's HOG-based face detector for capturing images
detector = dlib.get_frontal_face_detector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_student')
def add_student():
    return render_template('add_student.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        camera = cv2.VideoCapture(0)
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        camera.release()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    data = request.get_json()
    enrollment_number = data.get("enrollment_number")
    if not enrollment_number:
        return jsonify({"status": "error", "message": "Enrollment number required"}), 400

    # Ensure the images folder exists
    if not os.path.exists("images"):
        os.makedirs("images")

    # Create a folder for the enrollment number
    folder_path = os.path.join("images", enrollment_number)
    os.makedirs(folder_path, exist_ok=True)

    # Initialize capture progress
    capture_progress[enrollment_number] = {
        "count": 0,
        "total": 120,
        "status": "running",
        "complete": False,
        "progress": 0,
        "folder_path": folder_path,
        "error_reported": False
    }

    return jsonify({"status": "success", "message": "Capture started"})

@app.route('/capture_status')
def capture_status():
    enrollment_number = request.args.get("enrollment_number")
    if not enrollment_number or enrollment_number not in capture_progress:
        return jsonify({"status": "error", "error": "Invalid enrollment number or capture not started"}), 400

    progress_data = capture_progress[enrollment_number]
    return jsonify({
        "status": progress_data.get("status", "running"),
        "progress": progress_data.get("progress", 0),
        "complete": progress_data.get("complete", False),
        "error": progress_data.get("error", None)
    })

@app.route('/check_enrollment/<enrollment_number>', methods=['GET'])
def check_enrollment(enrollment_number):
    return jsonify({'exists': enrollment_number in name_map})

@socketio.on('capture_frame')
def handle_capture_frame(data):
    enrollment_number = data.get("enrollment_number")
    frame_data = data.get("frame")

    if not enrollment_number or enrollment_number not in capture_progress:
        socketio.emit('capture_progress', {
            "status": "error",
            "error": "Invalid enrollment number or capture not started",
            "complete": False
        })
        return

    progress_data = capture_progress[enrollment_number]
    if progress_data["complete"]:
        return
    
    # Skip processing if an error has already been reported
    if progress_data.get("error_reported", False):
        return

    try:
        # Decode the base64 frame
        base64_data = re.sub('^data:image/.+;base64,', '', frame_data)
        image_bytes = base64.b64decode(base64_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert BGR to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        face_detected = False
        face_location = None

        # If this is the first frame, check if face exists in the system
        if progress_data["count"] == 0 and svm_model is not None:
            face_encodings = face_recognition.face_encodings(rgb_frame)
            if face_encodings:
                # Check against the existing face encodings
                face_encoding = face_encodings[0]
                probas = svm_model.predict_proba([face_encoding])[0]
                max_similarity = np.max(probas)
                best_match = svm_model.classes_[np.argmax(probas)]
                
                # If similarity is high (>=0.6), then face already exists
                if max_similarity >= 0.8:
                    progress_data["error_reported"] = True
                    socketio.emit('capture_progress', {
                        "status": "error",
                        "error": f"Face matches existing student: {best_match}",
                        "complete": False
                    })
                    return

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            # Ensure face coordinates are within frame bounds
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue

            face_detected = True
            face_location = [y, x + w, y + h, x]  # [top, right, bottom, left]

            # Save the face image
            face_img = frame[y:y+h, x:x+w]
            filename = os.path.join(progress_data["folder_path"], f"{progress_data['count']}.jpg")
            cv2.imwrite(filename, face_img)

            # Update progress
            progress_data["count"] += 1
            progress_data["progress"] = (progress_data["count"] / progress_data["total"]) * 100

            # Emit progress update
            socketio.emit('capture_progress', {
                "status": "running",
                "progress": progress_data["progress"],
                "complete": False,
                "face_detected": face_detected,
                "face_location": face_location
            })

            # Check if capture is complete
            if progress_data["count"] >= progress_data["total"]:
                progress_data["status"] = "complete"
                progress_data["complete"] = True

                # Update names.json
                name_map[enrollment_number] = enrollment_number
                with open("names.json", "w") as f:
                    json.dump(name_map, f, indent=4)

                # Train the model
                # train_model()

                # Emit completion update
                socketio.emit('capture_progress', {
                    "status": "complete",
                    "progress": 100,
                    "complete": True,
                    "face_detected": face_detected,
                    "face_location": face_location
                })
                break

        if not face_detected:
            # Emit progress update even if no face is detected
            socketio.emit('capture_progress', {
                "status": "running",
                "progress": progress_data["progress"],
                "complete": False,
                "face_detected": False,
                "face_location": None
            })

    except Exception as e:
        progress_data["status"] = "error"
        progress_data["error"] = str(e)
        progress_data["error_reported"] = True
        socketio.emit('capture_progress', {
            "status": "error",
            "error": str(e),
            "complete": False
        })

def train_model():
    global svm_model
    # Load stored images
    data = []
    labels = []

    for student in os.listdir("images"):
        folder_path = os.path.join("images", student)
        
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = face_recognition.load_image_file(img_path)
            
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                data.append(face_encodings[0])
                labels.append(student)

    # Train SVM model
    new_svm_model = SVC(kernel="linear", probability=True)
    new_svm_model.fit(data, labels)

    # Save model
    with open("trained_svm.pkl", "wb") as f:
        pickle.dump(new_svm_model, f)

    # Reload the model into the global svm_model variable
    with open("trained_svm.pkl", "rb") as f:
        svm_model = pickle.load(f)

@app.route('/status')
def check_status():
    return jsonify({"status": "Live monitoring active"})

@app.route('/status_page')
def status_page():
    # Get current status of all students
    current_time = datetime.now()
    student_statuses = {}

    for student_id in name_map.keys():
        if student_id in last_detection_times:
            if (student_id in entry_times and 
                (current_time - last_detection_times[student_id]).total_seconds() < LEAVING_THRESHOLD):
                status = "IN"
                timestamp = entry_times[student_id].strftime("%Y-%m-%d %H:%M:%S")
            else:
                status = "OUT"
                timestamp = last_detection_times[student_id].strftime("%Y-%m-%d %H:%M:%S")
        else:
            status = "OUT"
            timestamp = "-"
        
        student_statuses[student_id] = {
            "status": status,
            "timestamp": timestamp
        }

    return render_template('status_page.html', student_statuses=student_statuses)

@socketio.on('video_frame')
def handle_frame(data):
    try:
        # Extract the base64 image data from the data URI
        frame_data = data.get("frame")
        if not frame_data or not isinstance(frame_data, str):
            return
        
        # Parse base64 data - remove the data:image/jpeg;base64, prefix
        base64_data = re.sub('^data:image/.+;base64,', '', frame_data)
        
        # Decode base64 image
        image_bytes = base64.b64decode(base64_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Get current time
        current_time = datetime.now()
        timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        detected_users = set()

        # Process each detected face
        for face_encoding in face_encodings:
            # Use SVM to predict the person
            if svm_model is None:
                print("SVM model not loaded yet.")
                continue

            probas = svm_model.predict_proba([face_encoding])[0]
            max_similarity = np.max(probas)
            best_match = svm_model.classes_[np.argmax(probas)]
            
            # Apply threshold (0.6 confidence)
            if max_similarity < 0.6:
                continue  # Skip unknown faces
                
            detected_users.add(best_match)
            
            # Update last detection time
            last_detection_times[best_match] = current_time
            
            # If not already "IN", set to "IN" and log
            if best_match not in entry_times:
                entry_times[best_match] = current_time
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([best_match, timestamp_str, "IN"])
                # Emit notification
                socketio.emit('notification', {
                    "enrollment_number": best_match,
                    "status": "IN",
                    "timestamp": timestamp_str
                })

        # Check for users who are "IN" but not detected recently
        for user in list(entry_times.keys()):
            if user not in detected_users:
                if (current_time - last_detection_times[user]).total_seconds() > LEAVING_THRESHOLD:
                    duration = (current_time - entry_times[user]).total_seconds() / 60  # in minutes
                    # Log "OUT"
                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([user, timestamp_str, "OUT"])
                    # Emit notification
                    socketio.emit('notification', {
                        "enrollment_number": user,
                        "status": "OUT",
                        "timestamp": timestamp_str
                    })
                    # Mark attendance if duration > 5 minutes
                    if duration > 5:
                        with open(CSV_FILE, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([user, timestamp_str, "ATTENDED"])
                    # Remove from entry_times
                    del entry_times[user]
                
    except Exception as e:
        print(f"Error processing frame: {str(e)}")

if __name__ == '__main__':
    socketio.run(app, debug=True)