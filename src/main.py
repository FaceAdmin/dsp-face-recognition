import cv2
from api_client import APIClient
from face_processing import encode_faces, face_recognition

def main():
    api_client = APIClient()
    
    print("[INFO] Fetching photos from API...")
    try:
        photos = api_client.fetch_user_photos()
        print(f"[INFO] Retrieved {len(photos)} photos.")
    except Exception as e:
        print(f"[ERROR] Failed to fetch photos: {e}")
        return

    try:
        face_encodings = encode_faces(photos)
        print(f"[INFO] Encoded faces for {len(face_encodings)} users.")
    except Exception as e:
        print(f"[ERROR] Failed to encode faces: {e}")
        return
    
    print("[INFO] Starting camera...")
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] Camera could not be opened.")
        return

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("[ERROR] Failed to capture frame. Exiting...")
                break

            try:
                print("[INFO] Capturing frame...")
                rgb_frame = frame[:, :, ::-1]
                face_locations = face_recognition.face_locations(rgb_frame)
                print(f"[INFO] Detected {len(face_locations)} face(s).")

                if len(face_locations) > 0:
                    current_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                    for encoding in current_encodings:
                        matches = face_recognition.compare_faces(list(face_encodings.values()), encoding)
                        if True in matches:
                            matched_user_id = list(face_encodings.keys())[matches.index(True)]
                            print(f"[INFO] Access Granted for User ID: {matched_user_id}")
                            api_client.record_attendance(user_id=matched_user_id, check_in=True)
                        else:
                            print("[WARNING] Access Denied!")
            except Exception as e:
                print(f"[ERROR] Error during face recognition: {e}")
                continue

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Exiting program...")
                break

    except KeyboardInterrupt:
        print("[INFO] Program interrupted by user.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera closed.")

if __name__ == "__main__":
    main()
