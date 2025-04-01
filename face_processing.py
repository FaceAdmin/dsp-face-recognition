import face_recognition
import numpy as np
import cv2
import requests

def load_image_from_url(url):
    resp = requests.get(url, stream=True)
    if not resp.ok:
        raise Exception(f"Failed to load image: {url}")
    arr = np.asarray(bytearray(resp.content), dtype="uint8")
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def encode_faces(photo_data):
    face_encodings = {}
    for item in photo_data:
        user_id = item["user"]
        photo_path = item["photo_path"]
        print(f"[INFO] Processing photo for user {user_id}: {photo_path}")

        image = load_image_from_url(photo_path)
        if image is None:
            print(f"[WARNING] Skipping photo for user {user_id}.")
            continue

        locs = face_recognition.face_locations(image)
        encs = face_recognition.face_encodings(image, locs)

        if encs:
            if user_id not in face_encodings:
                face_encodings[user_id] = []
            face_encodings[user_id].extend(encs)
        else:
            print(f"[WARNING] No face found in photo for user {user_id}.")

    return face_encodings
