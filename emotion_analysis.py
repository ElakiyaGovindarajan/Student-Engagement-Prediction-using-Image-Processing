import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import csv
from datetime import datetime

# Load the pre-trained emotion detection model
emotion_model_path = "models/emotion_model.h5"  # Update this path if needed
emotion_model = tf.keras.models.load_model(emotion_model_path)

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# File paths
csv_file = "engagement_log.csv"
excel_file = "engagement_log.xlsx"

# Function to preprocess face for model
def preprocess_face(face):
    try:
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=(0, -1))
        return face_input
    except Exception as e:
        print(f"Error in face preprocessing: {e}")
        return None

# Function to log engagement data
def log_engagement(person_id, emotion_label, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to CSV
    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, person_id, emotion_label, confidence])

    # Save to Excel
    try:
        new_data = pd.DataFrame([[timestamp, person_id, emotion_label, confidence]],
                                columns=["Timestamp", "Person_ID", "Emotion", "Confidence"])
        try:
            existing_df = pd.read_excel(excel_file)
            df = pd.concat([existing_df, new_data], ignore_index=True)
        except FileNotFoundError:
            df = new_data

        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"✅ Logged: {timestamp}, Person {person_id}, {emotion_label}, Confidence: {confidence:.2f}")
    except Exception as e:
        print(f"❌ Error saving to Excel: {e}")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))

    if len(faces) == 0:
        print("⚠ No face detected → Marking as 'Not Listening'")
        cv2.putText(frame, "Not Listening", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        log_engagement("None", "Not Listening", 0.0)
    else:
        print(f"📸 Detected {len(faces)} face(s)")

        for i, (x, y, w, h) in enumerate(faces):
            person_id = i + 1  # Unique person ID per face in current frame
            face = frame[y:y + h, x:x + w]
            face_input = preprocess_face(face)

            if face_input is not None:
                try:
                    predictions = emotion_model.predict(face_input)
                    emotion_idx = np.argmax(predictions)
                    emotion_label = emotion_labels[emotion_idx]
                    confidence = predictions[0][emotion_idx]

                    if confidence < 0.5:
                        emotion_label = "Uncertain"

                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f"Person {person_id}: {emotion_label} ({confidence:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # Log engagement
                    log_engagement(person_id, emotion_label, confidence)

                except Exception as e:
                    print(f"❌ Error in model prediction: {e}")

    cv2.imshow("Emotion Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
