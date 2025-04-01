import sys
from PyQt5 import QtWidgets
from ui.face_recognition_window import FaceRecognitionWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FaceRecognitionWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
