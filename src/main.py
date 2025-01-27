import datetime
import cv2
import numpy as np
import face_recognition
import time
from api_client import APIClient
from face_processing import encode_faces

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
    
    print("[INFO] Starting web camera")
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("[ERROR] Camera could not be opened")
        return

    cooldown_time = 10
    last_recognition_time = {}

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("[ERROR] Failed to capture frame")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings_in_frame = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings_in_frame):
                matches = face_recognition.compare_faces(list(face_encodings.values()), face_encoding)

                face_distances = face_recognition.face_distance(list(face_encodings.values()), face_encoding)
                best_match_index = np.argmin(face_distances) if face_distances.size > 0 else -1

                if best_match_index >= 0 and best_match_index < len(matches) and np.any(matches[best_match_index]):
                    matched_user_id = list(face_encodings.keys())[best_match_index]

                    current_time = time.time()
                    if matched_user_id not in last_recognition_time or \
                    (current_time - last_recognition_time[matched_user_id] > cooldown_time):
                        print(f"[INFO] Access Granted for User ID: {matched_user_id}")
                        check_in_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
                        api_client.record_attendance(user_id=matched_user_id, check_in_time=check_in_time)
                        last_recognition_time[matched_user_id] = current_time

                else:
                    print(f"[INFO] No match found for current face")

                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Exiting program")
                break

    except KeyboardInterrupt:
        print("[INFO] Program interrupted by user")
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera closed")

if __name__ == "__main__":
    main()
