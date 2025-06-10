from keras.models import load_model
import mediapipe as mp
import cv2
import numpy as np

# Load pre-trained emotion model

emotion_model = load_model(r"C:\Users\elaki\Downloads\EngagenmentPrediction (2)\EngagenmentPrediction\EngagenmentPrediction\models\emotion_model.h5")


emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

# Function to classify engagement based on emotion
def classify_engagement(emotion):
    if emotion in ["Happy", "Neutral"]:
        return "Listening"
    elif emotion in ["Sad", "Disgust", "Fear"]:
        return "Distracted"
    elif emotion == "Angry":
        return "Frustrated"
    else:
        return "Sleeping"


# Function to detect faces and classify emotions
def detect_faces(frame):
    # Detect faces using Mediapipe or any other method
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Crop the face from the frame
            face = frame[y:y + h_box, x:x + w_box]
            return face  # Return the detected face

    return None  # If no face is detected, return None