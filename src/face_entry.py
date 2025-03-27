import sys
import time
import cv2
import face_recognition
import numpy as np

from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets

from api_client import APIClient
from face_processing import encode_faces

def draw_overlay(frame, text, color=(0, 255, 0)):
    overlay = frame.copy()
    h, w, _ = frame.shape
    cv2.rectangle(overlay, (0, h - 80), (w, h), color, -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

class OTPDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter OTP Code")

        self.email_label = QtWidgets.QLabel("Email:")
        self.email_edit = QtWidgets.QLineEdit()
        self.email_edit.setPlaceholderText("user@example.com")

        self.code_label = QtWidgets.QLabel("One-time code:")
        self.code_edit = QtWidgets.QLineEdit()
        self.code_edit.setPlaceholderText("123456")

        self.button_ok = QtWidgets.QPushButton("OK")
        self.button_cancel = QtWidgets.QPushButton("Cancel")
        self.button_ok.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow(self.email_label, self.email_edit)
        form_layout.addRow(self.code_label, self.code_edit)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.button_ok)
        btn_layout.addWidget(self.button_cancel)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def get_inputs(self):
        return self.email_edit.text().strip(), self.code_edit.text().strip()

class FaceRecognitionWindow(QtWidgets.QMainWindow):
    UNKNOWN_TIMEOUT = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Face Recognition Entry")
        self.resize(1280, 720)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setScaledContents(True)

        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
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
                        full_name = f"{user_details.get('fname', 'Unknown')} {user_details.get('lname', '')}".strip()
                        self.overlay_text = f"Access Granted: {full_name}"
                        self.overlay_color = (0, 255, 0)
                        self.overlay_until = time.time() + 3
                    except Exception as e:
                        print("[ERROR record_attendance]", e)
                    self.last_time_recorded[user_id] = current_time

                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            else:
                recognized_anyone = recognized_anyone or False
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)

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

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FaceRecognitionWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
