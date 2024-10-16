import os
import face_recognition
import pickle

KNOWN_FACES_DIR = 'dataset'
ENCODINGS_FILE = 'encodings.pkl'

known_encodings = []
known_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            known_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_names.append(name)
        else:
            print(f"There is no face on the image {filename}")

with open(ENCODINGS_FILE, 'wb') as f:
    pickle.dump({'encodings': known_encodings, 'names': known_names}, f)

print("Faces encodings are saved")
