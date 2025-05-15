# Automated Check-in ML Project

This project provides an automated check-in system using facial recognition. It includes a simple web interface to interact with the system and a training module to build the face recognition model.

## Features

- Train a face recognition model using `face_trainer.py`
- Launch a web-based interface for check-in using `app.py`
- Easy-to-use and lightweight setup

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- OpenCV (`cv2`)
- Flask
- NumPy
- PIL (Pillow)
- Any other libraries mentioned in `requirements.txt` (if available)

You can install dependencies with:

```bash
pip install -r requirements.txt

If requirements.txt is not available, install manually:

pip install flask opencv-python numpy pillow

## Folder Structure
Automated_Checkin_ML_Project/
├── app.py               # Web interface
├── face_trainer.py      # Model training script
├── dataset/             # Folder for storing face images
├── trainer/             # Folder where trained model gets saved
├── static/              # Static files (e.g., CSS, JS)
├── templates/           # HTML templates
└──

```

To Train the Model
Run the following command to start the face data training process:

python face_trainer.py

To run the Web app:
python app.py


