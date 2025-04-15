import sys
from PyQt5 import QtWidgets
from ui.login_window import LoginWindow
from ui.face_recognition_window import FaceRecognitionWindow
from api_client import APIClient

def main():
    app = QtWidgets.QApplication(sys.argv)
    api_client = APIClient()
    login_dialog = LoginWindow(api=api_client)

    if login_dialog.exec_() == QtWidgets.QDialog.Accepted:
        face_window = FaceRecognitionWindow(api_client=api_client)
        face_window.show()
        sys.exit(app.exec_())
    else:
        sys.exit()

if __name__ == "__main__":
    main()
