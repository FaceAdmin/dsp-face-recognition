import sys
from PyQt5 import QtWidgets, QtCore
from config import API_BASE_URL

class LoginWindow(QtWidgets.QDialog):
    def __init__(self, parent=None, api=None):
        super(LoginWindow, self).__init__(parent)
        self.api = api
        self.setWindowTitle("FaceAdmin - Login")
        self.resize(800, 600)

        layout = QtWidgets.QVBoxLayout(self)

        title_label = QtWidgets.QLabel("FaceAdmin", self)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 32px; font-weight: bold;")
        
        login_label = QtWidgets.QLabel("Login", self)
        login_label.setAlignment(QtCore.Qt.AlignCenter)
        login_label.setStyleSheet("font-size: 24px;")

        self.email_edit = QtWidgets.QLineEdit(self)
        self.email_edit.setPlaceholderText("Email")
        self.password_edit = QtWidgets.QLineEdit(self)
        self.password_edit.setPlaceholderText("Password")
        self.password_edit.setEchoMode(QtWidgets.QLineEdit.Password)

        self.login_button = QtWidgets.QPushButton("Login", self)
        self.login_button.clicked.connect(self.handle_login)

        layout.addWidget(title_label)
        layout.addWidget(login_label)
        layout.addSpacing(30)
        layout.addWidget(self.email_edit)
        layout.addWidget(self.password_edit)
        layout.addWidget(self.login_button)
        layout.addStretch()

    def handle_login(self):
        email = self.email_edit.text().strip()
        password = self.password_edit.text().strip()

        if not email or not password:
            QtWidgets.QMessageBox.warning(self, "Error", "Please enter both email and password.")
            return

        login_url = f"{API_BASE_URL}/auth/login/"
        try:
            response = self.api.session.post(login_url, json={"email": email, "password": password})
            if response.status_code == 200:
                self.accept()
            else:
                QtWidgets.QMessageBox.warning(self, "Login Failed", "Invalid email or password.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred: {e}")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    login = LoginWindow()
    if login.exec_() == QtWidgets.QDialog.Accepted:
        print("Login succeeded")
    sys.exit(app.exec_())
