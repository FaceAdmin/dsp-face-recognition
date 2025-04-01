from PyQt5 import QtWidgets

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
