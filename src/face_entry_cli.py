import cv2
import numpy as np
import face_recognition
import time
from datetime import datetime
from api_client import APIClient
from face_processing import encode_faces

def main():
    api = APIClient()
    print("[INFO] Fetching photos from API...")
    try:
        photos = api.fetch_user_photos()
        print(f"[INFO] Retrieved {len(photos)} photos.")
    except Exception as e:
        print(f"[ERROR] Failed to fetch photos: {e}")
        return

    try:
        encodings_dict = encode_faces(photos)
        print(f"[INFO] Encoded faces for {len(encodings_dict)} users.")
    except Exception as e:
        print(f"[ERROR] Failed to encode faces: {e}")
        return

    known_encodings = []
    known_user_ids = []
    for user_id, enc_list in encodings_dict.items():
        for enc in enc_list:
            known_encodings.append(enc)
            known_user_ids.append(user_id)

    print("[INFO] Starting camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return

    frame_skip = 5
    frame_count = 0
    cooldown_time = 10
    last_time_recorded = {}
    tolerance = 0.45

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to capture frame.")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            rgb_small_frame = np.ascontiguousarray(rgb_small_frame)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for loc, enc in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, enc, tolerance=tolerance)
                face_distances = face_recognition.face_distance(known_encodings, enc)
                best_idx = np.argmin(face_distances) if len(face_distances) > 0 else -1

                if best_idx != -1 and matches[best_idx]:
                    user_id = known_user_ids[best_idx]
                    current_time = time.time()
                    if user_id not in last_time_recorded or (current_time - last_time_recorded[user_id]) > cooldown_time:
                        try:
                            api.record_attendance(user_id)
                            user_details = api.get_user(user_id)
                            time = datetime.now().strftime("%d/%m/%Y, %H:%M")
                            full_name = f"{user_details.get('fname', 'Unknown')} {user_details.get('lname', '')}".strip()
                            print(f"Time: {time}\nUser: {full_name}")
                        except Exception as e:
                            print(f"[ERROR] {e}")
                        last_time_recorded[user_id] = current_time
                else:
                    print("[INFO] Unknown face detected.")

                top, right, bottom, left = loc
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Exiting program")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera closed.")

if __name__ == "__main__":
    main()
