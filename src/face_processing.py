import face_recognition
import numpy as np
import cv2
import requests

def encode_faces(photo_data):
    face_encodings = {}
    for photo in photo_data:
        user_id = photo["user_id"]
        photo_path = photo["photo_path"]
        print(f"[INFO] Processing photo for user {user_id}: {photo_path}")

        image = load_image_from_url(photo_path)
        if image is None:
            print(f"[WARNING] Skipping photo for user {user_id}.")
            continue

        face_locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, face_locations)

        if encodings:
            if user_id not in face_encodings:
                face_encodings[user_id] = []
            face_encodings[user_id].extend(encodings)
        else:
            print(f"[WARNING] No face found in photo for user {user_id}.")

    return face_encodings



def load_image_from_url(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image = np.asarray(bytearray(response.content), dtype="uint8")
        return cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        raise Exception(f"Failed to load image: {url}")
