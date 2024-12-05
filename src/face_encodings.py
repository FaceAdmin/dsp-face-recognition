import os
import face_recognition
import pickle

KNOWN_FACES_DIR = 'dataset'
ENCODINGS_FILE = 'encodings.pkl'

known_encodings = []
known_names = []

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_path = os.path.join(KNOWN_FACES_DIR, person_name)
    if os.path.isdir(person_path):
        for filename in os.listdir(person_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_path, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                else:
                    print(f"no face on the image {filename} in folder {person_name}")

with open(ENCODINGS_FILE, 'wb') as f:
    pickle.dump({'encodings': known_encodings, 'names': known_names}, f)

print("encodings are saved")