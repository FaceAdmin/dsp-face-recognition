import time
import cv2
import face_recognition
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from facetools.liveness_detection import LivenessDetector
from config import OULU_MODEL_PATH, LIVENESS_THRESHOLD
from ui.dialogs import OTPDialog
from ui.overlay import draw_overlay

class FaceRecognitionWindow(QtWidgets.QMainWindow):
    UNKNOWN_TIMEOUT = 7

    def __init__(self, parent=None, api_client=None):
        super().__init__(parent)
        self.api = api_client
        self.setWindowTitle("Face Recognition Entry")
        self.resize(800, 600)
        self.video_label = QtWidgets.QLabel()
        self.video_label.setScaledContents(True)
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_label)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.cap = cv2.VideoCapture(0)
        # self.log = open("TEST_spoof.csv", "w", encoding="utf-8")
        # self.log.write("score\n")


        if not self.cap.isOpened():
            print("[ERROR] Could not open camera.")
            return

        self.known_encodings, self.known_user_ids = self.load_encodings_from_api()
        self.tolerance = 0.45
        self.cooldown_time = 10
        self.process_every = 7
        self.frame_counter = 0
        self.last_face_rectangles = []
        self.last_time_recorded = {}
        self.unknown_face_start = None
        self.otp_dialog_shown = False

        self.liveness_detector = LivenessDetector(OULU_MODEL_PATH)

        self.video_timer = QtCore.QTimer()
        self.video_timer.timeout.connect(self.update_frame)
        self.video_timer.start(30)

        self.overlay_text = ""
        self.overlay_color = (0, 255, 0)
        self.overlay_until = 0

    def load_encodings_from_api(self):
        known_encodings = []
        known_user_ids = []
        try:
            aggregated_encodings = self.api.fetch_aggregated_encodings()
            for user_id_str, enc in aggregated_encodings.items():
                known_encodings.append(np.array(enc))
                known_user_ids.append(int(user_id_str))
        except Exception as e:
            print("[ERROR load_encodings]", e)
        return known_encodings, known_user_ids

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_counter += 1

        if self.frame_counter % self.process_every != 0:
            for rect, color in self.last_face_rectangles:
                (left_orig, top_orig, right_orig, bottom_orig) = rect
                cv2.rectangle(frame, (left_orig, top_orig), (right_orig, bottom_orig), color, 2)
            if time.time() < self.overlay_until and self.overlay_text:
                draw_overlay(frame, self.overlay_text, self.overlay_color)
            self.display_frame(frame)
            return

        self.last_face_rectangles = []

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = np.ascontiguousarray(rgb_small_frame)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        recognized_anyone = False

        if face_locations:
            if self.unknown_face_start is None:
                self.unknown_face_start = time.time()
        else:
            self.unknown_face_start = None
            self.otp_dialog_shown = False

        for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
            top_orig    = top * 4
            right_orig  = right * 4
            bottom_orig = bottom * 4
            left_orig   = left * 4

            face_bgr = frame[top_orig:bottom_orig, left_orig:right_orig]
            if face_bgr.size == 0:
                continue

            liveness_score = self.liveness_detector.get_liveness_score(face_bgr)
            print(f"[LIVENESS] score = {liveness_score:.4f}")
            # self.log.write(f"{liveness_score:.4f}\n")

            if liveness_score < LIVENESS_THRESHOLD:
                self.last_face_rectangles.append(((left_orig, top_orig, right_orig, bottom_orig), (0, 0, 255)))
                continue

            matches = face_recognition.compare_faces(self.known_encodings, face_enc, self.tolerance)
            face_distances = face_recognition.face_distance(self.known_encodings, face_enc)
            best_idx = np.argmin(face_distances) if len(face_distances) > 0 else -1

            if best_idx != -1 and matches[best_idx]:
                recognized_anyone = True
                user_id = self.known_user_ids[best_idx]
                current_time = time.time()
                if user_id not in self.last_time_recorded or (current_time - self.last_time_recorded[user_id]) > self.cooldown_time:
                    try:
                        self.api.record_attendance(user_id)
                        user_details = self.api.get_user(user_id)
                        full_name = f"{user_details.get('first_name', 'Unknown')} {user_details.get('last_name', '')}".strip()
                        self.overlay_text = f"Access Granted"
                        self.overlay_color = (0, 255, 0)
                        self.overlay_until = time.time() + 3
                    except Exception as e:
                        print("[ERROR record_attendance]", e)
                    self.last_time_recorded[user_id] = current_time
                self.last_face_rectangles.append(((left_orig, top_orig, right_orig, bottom_orig), (0, 255, 0)))
            else:
                self.last_face_rectangles.append(((left_orig, top_orig, right_orig, bottom_orig), (255, 255, 0)))

        if not recognized_anyone and self.unknown_face_start is not None:
            elapsed = time.time() - self.unknown_face_start
            if elapsed >= self.UNKNOWN_TIMEOUT and not self.otp_dialog_shown:
                self.show_otp_dialog()

        if time.time() < self.overlay_until and self.overlay_text:
            draw_overlay(frame, self.overlay_text, self.overlay_color)

        for rect, color in self.last_face_rectangles:
            (left_orig, top_orig, right_orig, bottom_orig) = rect
            cv2.rectangle(frame, (left_orig, top_orig), (right_orig, bottom_orig), color, 2)

        self.display_frame(frame)

    def display_frame(self, frame):
        h, w, ch = frame.shape
        qimg = QtGui.QImage(frame.data, w, h, ch * w, QtGui.QImage.Format_BGR888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix)

    def show_otp_dialog(self):
        self.otp_dialog_shown = True
        dialog = OTPDialog(self)
        result = dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            email, code = dialog.get_inputs()
            try:
                verify_resp = self.api.verify_otp(email, code)
                user_id = verify_resp.get("user_id")
                if user_id:
                    self.api.record_attendance(user_id)
                    self.overlay_text = "Access Granted"
                    self.overlay_color = (0, 255, 0)
                    self.overlay_until = time.time() + 3
                else:
                    self.overlay_text = "Access Denied"
                    self.overlay_color = (0, 0, 255)
                    self.overlay_until = time.time() + 3
            except Exception as e:
                print("[ERROR verify_otp]", e)
                self.overlay_text = "Access Denied"
                self.overlay_color = (0, 0, 255)
                self.overlay_until = time.time() + 3
        else:
            self.overlay_text = ""
        self.unknown_face_start = None

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)
