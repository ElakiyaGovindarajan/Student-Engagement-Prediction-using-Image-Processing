from flask import Flask, render_template, Response, request, redirect, url_for, session
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import csv
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # For session management

# Hardcoded user credentials (Replace with a secure method in production)
USERS = {
    "admin": "password123"
}

# Load the pre-trained emotion detection model
emotion_model_path = "models/emotion_model.h5"
emotion_model = tf.keras.models.load_model(emotion_model_path)

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# File paths
csv_file = "engagement_log.csv"
excel_file= "engagement_log.xlsx"
register_excel_file = "registered_users.xlsx"

last_face_position = None  # Stores last known face position
person_id = 1
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
            df = new_data  # If the file doesn't exist, create a new one

        df.to_excel(excel_file, index=False, engine='openpyxl')  # Ensure proper saving
        print(f"✅ Logged: {timestamp}, Person {person_id}, {emotion_label}, Confidence: {confidence:.2f}")
    except Exception as e:
        print(f"❌ Error saving to Excel: {e}")


# Function to generate frames
def generate_frames():
    global last_face_position
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


        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # Encode frame to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame in correct format for MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        # Append user data to Excel
        user_data = pd.DataFrame([[name, email, username, password]],
                                 columns=["Name", "Email", "Username", "Password"])

        try:
            # Load existing data or create new
            try:
                existing_data = pd.read_excel(register_excel_file)
                all_data = pd.concat([existing_data, user_data], ignore_index=True)
            except FileNotFoundError:
                all_data = user_data

            all_data.to_excel(register_excel_file, index=False, engine='openpyxl')
            print("✅ User registered and saved to Excel!")

        except Exception as e:
            print(f"❌ Error saving registration: {e}")

        return redirect(url_for('login'))

    return render_template('register.html')


# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in USERS and USERS[username] == password:
            session['user'] = username
            return redirect(url_for('index'))  # ✅ This should go to index.html
        return "Invalid credentials, try again!"
    return render_template('login.html')



# Logout route
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


# Protected route
@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')



@app.route('/video_feed')
def video_feed():
    if 'user' not in session:
        return redirect(url_for('login'))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
