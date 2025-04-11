import time
import cv2
import face_recognition
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from api_client import APIClient
from face_processing import encode_faces
from facetools.liveness_detection import LivenessDetector
from config import OULU_MODEL_PATH, LIVENESS_THRESHOLD
from ui.dialogs import OTPDialog
from ui.overlay import draw_overlay

class FaceRecognitionWindow(QtWidgets.QMainWindow):
    UNKNOWN_TIMEOUT = 7

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Face Recognition Entry")
        self.resize(1600, 1200)
        self.video_label = QtWidgets.QLabel()
        self.video_label.setScaledContents(True)
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_label)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.api = APIClient()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ERROR] Could not open camera.")
            return

        self.known_encodings, self.known_user_ids = self.load_encodings_from_api()
        self.tolerance = 0.45
        self.cooldown_time = 10
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
            photos = self.api.fetch_user_photos()
            enc_dict = encode_faces(photos)
            for user_id, enc_list in enc_dict.items():
                for enc in enc_list:
                    known_encodings.append(enc)
                    known_user_ids.append(user_id)
        except Exception as e:
            print("[ERROR load_encodings]", e)
        return known_encodings, known_user_ids

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = np.ascontiguousarray(rgb_small_frame)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        recognized_anyone = False

        for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
            top_orig    = top * 4
            right_orig  = right * 4
            bottom_orig = bottom * 4
            left_orig   = left * 4

            face_bgr = frame[top_orig:bottom_orig, left_orig:right_orig]
            if face_bgr.size == 0:
                continue

            liveness_score = self.liveness_detector.get_liveness_score(face_bgr)
            if liveness_score < LIVENESS_THRESHOLD:
                cv2.rectangle(frame, (left_orig, top_orig), (right_orig, bottom_orig), (0, 0, 255), 2)
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
                        self.overlay_text = f"Access Granted: {full_name}"
                        self.overlay_color = (0, 255, 0)
                        self.overlay_until = time.time() + 3
                    except Exception as e:
                        print("[ERROR record_attendance]", e)
                    self.last_time_recorded[user_id] = current_time
                cv2.rectangle(frame, (left_orig, top_orig), (right_orig, bottom_orig), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (left_orig, top_orig), (right_orig, bottom_orig), (255, 255, 0), 2)

        if recognized_anyone:
            self.unknown_face_start = None
            self.otp_dialog_shown = False
        else:
            if face_locations:
                if self.unknown_face_start is None:
                    self.unknown_face_start = time.time()
                else:
                    elapsed = time.time() - self.unknown_face_start
                    if elapsed >= self.UNKNOWN_TIMEOUT and not self.otp_dialog_shown:
                        self.show_otp_dialog()
            else:
                self.unknown_face_start = None
                self.otp_dialog_shown = False

        if time.time() < self.overlay_until and self.overlay_text:
            draw_overlay(frame, self.overlay_text, self.overlay_color)

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
                    self.overlay_text = "Access Granted (OTP)"
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
