import pickle
import cv2
import numpy as np
import face_recognition
from tensorflow.python.keras.models import load_model

# Path to encoding file
ENCODINGS_FILE = 'encodings.pkl'
LIVENESS_MODEL_PATH = 'liveness.model'  # Replace with actual path to your liveness detection model

# Loading encodings and setting up video capture
with open(ENCODINGS_FILE, 'rb') as f:
    data = pickle.load(f)

known_encodings = data['encodings']
known_names = data['names']

# Load the liveness detection model
liveness_model = load_model(LIVENESS_MODEL_PATH)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame from video source")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    # Locate and encode faces in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Process each face found
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # Find the best match for the recognized face
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if face_distances.size > 0 else -1
        if best_match_index >= 0 and matches[best_match_index]:
            name = known_names[best_match_index]

        # Liveness detection
        face = small_frame[top:bottom, left:right]
        face = cv2.resize(face, (32, 32))  # Resize to match the input size of the liveness model
        face = face.astype("float") / 255.0
        face = np.expand_dims(face, axis=0)

        # Predict liveness
        liveness_pred = liveness_model.predict(face)[0][0]
        if liveness_pred > 0.5:
            liveness_label = "Real"
        else:
            liveness_label = "Fake"

        # Scale back up face locations since the frame we processed was scaled down
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Draw a label with a name and liveness result below the face
        label = f"{name} ({liveness_label})"
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition with Liveness Detection', frame)

    # Quit the window with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()
