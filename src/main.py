import pickle
import cv2
import numpy as np
import face_recognition
from tensorflow.keras.models import load_model

ENCODINGS_FILE = 'data/encodings.pkl'
LIVENESS_MODEL_PATH = 'models/liveness_model.h5'

with open(ENCODINGS_FILE, 'rb') as f:
    data = pickle.load(f)

known_encodings = data['encodings']
known_names = data['names']

liveness_model = load_model(LIVENESS_MODEL_PATH)

video_capture = cv2.VideoCapture(0)

def preprocess_face(face, target_size=(64, 64)):
    face = cv2.resize(face, target_size)
    face = face.astype("float") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame from video source")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if face_distances.size > 0 else -1
        if best_match_index >= 0 and matches[best_match_index]:
            name = known_names[best_match_index]

        face = small_frame[top:bottom, left:right]
        if face.size > 0:
            face_preprocessed = preprocess_face(face, target_size=(64, 64))

            liveness_pred = liveness_model.predict(face_preprocessed)[0][0]
            liveness_label = "Real" if liveness_pred > 0.5 else "Fake"

            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{name} ({liveness_label})"
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
