import cv2
from api_client import APIClient
from face_processing import encode_faces, face_recognition

def main():
    api_client = APIClient()
    
    print("[INFO] Fetching photos from API...")
    photos = api_client.fetch_user_photos()
    face_encodings = encode_faces(photos)
    
    print("[INFO] Starting camera...")
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        current_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for encoding in current_encodings:
            matches = face_recognition.compare_faces(list(face_encodings.values()), encoding)
            if True in matches:
                matched_user_id = list(face_encodings.keys())[matches.index(True)]
                print(f"[INFO] Access Granted for User ID: {matched_user_id}")
                api_client.record_attendance(user_id=matched_user_id, check_in=True)
            else:
                print("[WARNING] Access Denied!")

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
