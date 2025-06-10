from face_detection import detect_faces
from emotion_analysis import classify_engagement
import cv2
import numpy as np

# Start the webcam and process frames for emotion detection
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read a frame from the webcam.")
        break

    # Detect faces using the face_detection function (it will process the frame)
    face_detected = detect_faces(frame)  # Ensure that the detect_faces function returns a valid face

    if face_detected is not None:  # Only proceed if a valid face is detected
        # Convert the frame to grayscale, resize, and normalize the face image before prediction
        face_gray = cv2.cvtColor(face_detected, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        face_resized = cv2.resize(face_gray, (48, 48)) / 255.0  # Resize and normalize
        face_input = face_resized.reshape(1, 48, 48, 1)  # Reshape for the model

        # Predict the emotion
        predictions = emotion_model.predict(face_input)
        emotion = emotions[np.argmax(predictions)]  # Get the predicted emotion

        # Classify engagement based on emotion
        engagement = classify_engagement(emotion)

        # Draw bounding box and label
        # You can display the emotion and engagement status on the frame
        cv2.putText(frame, f"{engagement} ({emotion})", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)

    # Show the result frame
    cv2.imshow("Student Engagement Detection", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()