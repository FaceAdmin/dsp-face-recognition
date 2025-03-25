import sys
import time
import cv2
import face_recognition
import numpy as np

from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets

from api_client import APIClient
from face_processing import encode_faces


# ===========================
# 1. Диалоговое окно для OTP
# ===========================
class OTPDialog(QtWidgets.QDialog):
    """
    Диалоговое окно для ввода OTP-кода.
    Пока просто закрывается на OK, без реальной проверки.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter OTP Code")

        self.label = QtWidgets.QLabel("Please enter your 6-digit code:")
        self.otp_edit = QtWidgets.QLineEdit()
        self.otp_edit.setPlaceholderText("123456")

        # Кнопки OK / Cancel
        self.button_ok = QtWidgets.QPushButton("OK")
        self.button_cancel = QtWidgets.QPushButton("Cancel")
        self.button_ok.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.otp_edit)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.button_ok)
        btn_layout.addWidget(self.button_cancel)

        layout.addLayout(btn_layout)
        self.setLayout(layout)


# ==============================
# 2. Главное окно FaceRecognition
# ==============================
class FaceRecognitionWindow(QtWidgets.QMainWindow):
    UNKNOWN_TIMEOUT = 5  # Секунды ожидания, прежде чем показать форму OTP

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Face Recognition Entry (PyQt)")
        self.resize(900, 900)

        # 2.1 Виджет для видео
        self.video_label = QtWidgets.QLabel()
        self.video_label.setScaledContents(True)

        # 2.2 Статусная строка (Access Granted / Unknown face и т.д.)
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("font-size: 18px; color: green;")

        # Компоновка
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # 2.3 Инициализация камеры
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.setText("Cannot open camera!")
            return

        # 2.4 Загрузка энкодингов с бэкенда
        self.api = APIClient()
        self.known_encodings, self.known_user_ids = self.load_encodings_from_api()

        # 2.5 Прочие переменные
        self.tolerance = 0.45
        self.last_time_recorded = {}
        self.cooldown_time = 10  # (сек) между повторными check-in/outs одного пользователя

        # Для отслеживания неизвестного лица
        self.unknown_face_start = None
        self.otp_dialog_shown = False

        # 2.6 Запуск таймера на обновление кадров
        self.video_timer = QtCore.QTimer()
        self.video_timer.timeout.connect(self.update_frame)
        self.video_timer.start(30)  # ~33 FPS

    def load_encodings_from_api(self):
        """
        1) Получаем фото пользователей
        2) Энкодим их через face_processing.encode_faces
        3) Собираем списки encodings и user_ids
        """
        known_encodings = []
        known_user_ids = []

        try:
            print("[INFO] Fetching photos from API...")
            photos = self.api.fetch_user_photos()
            print(f"[INFO] Retrieved {len(photos)} photos.")

            encodings_dict = encode_faces(photos)
            print(f"[INFO] Encoded faces for {len(encodings_dict)} users.")

            for user_id, enc_list in encodings_dict.items():
                for enc in enc_list:
                    known_encodings.append(enc)
                    known_user_ids.append(user_id)
        except Exception as e:
            print("[ERROR] Failed to load/encode photos:", e)

        return known_encodings, known_user_ids

    def update_frame(self):
        """Считываем кадр, делаем распознавание, отображаем результат в video_label."""
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
                # Вызываем record_attendance, но с проверкой cooldown
                current_time = time.time()
                if user_id not in self.last_time_recorded or (current_time - self.last_time_recorded[user_id]) > self.cooldown_time:
                    try:
                        self.api.record_attendance(user_id)
                        user_details = self.api.get_user(user_id)
                        full_name = f"{user_details.get('fname', 'Unknown')} {user_details.get('lname', '')}".strip()
                        dt_str = datetime.now().strftime("%d/%m/%Y, %H:%M")
                        self.status_label.setText(f"Access Granted: {full_name} ({dt_str})")
                    except Exception as e:
                        print("[ERROR]", e)
                    self.last_time_recorded[user_id] = current_time

                # Рисуем зелёный bounding box
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
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        if recognized_anyone:
            # Сброс таймера для неизвестных
            self.unknown_face_start = None
            self.otp_dialog_shown = False
        else:
            # Если есть лицо, но оно не узнано
            if face_locations:
                self.status_label.setText("Unknown face detected...")
                if self.unknown_face_start is None:
                    self.unknown_face_start = time.time()
                else:
                    elapsed = time.time() - self.unknown_face_start
                    if elapsed >= self.UNKNOWN_TIMEOUT and not self.otp_dialog_shown:
                        self.show_otp_dialog()
            else:
                # Если вообще лиц нет, сброс
                self.status_label.setText("")
                self.unknown_face_start = None
                self.otp_dialog_shown = False

        # Конвертируем кадр BGR -> QImage -> QPixmap
        h, w, ch = frame.shape
        qimg = QtGui.QImage(frame.data, w, h, ch * w, QtGui.QImage.Format_BGR888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix)

    def show_otp_dialog(self):
        """Показываем форму ввода OTP (пока без проверки)."""
        self.otp_dialog_shown = True
        dialog = OTPDialog(self)
        result = dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            # Пользователь нажал OK — в будущем здесь будет проверка OTP
            self.status_label.setText("OTP entered (dummy).")
        else:
            self.status_label.setText("OTP cancelled.")
        self.unknown_face_start = None

    def closeEvent(self, event):
        """Закрываем камеру при выходе из приложения."""
        if self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)


# ===========================
# 3. Точка входа (main)
# ===========================
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FaceRecognitionWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
