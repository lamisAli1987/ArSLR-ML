# Main.py
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QDesktopWidget, QMessageBox
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QFrame, QStackedWidget
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QPixmap, QFont
from data_collection_widget import DataCollectionWidget
from training_widget import TrainingWidget
from testing_widget import TestingWidget
import cv2
import sys

class Dashboard(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        # Top space
        top_space = QFrame(self)
        top_space.setFixedHeight(10)
        top_space.setStyleSheet("background-color: #f0f0f0;")
        main_layout.addWidget(top_space)

        # Top frame
        top_frame = QFrame(self)
        top_frame.setFixedHeight(150)
        top_frame.setStyleSheet("background-color: white;")
        top_layout = QHBoxLayout(top_frame)
        top_layout.setContentsMargins(80, 0, 80, 0)
        top_layout.setSpacing(10)
        main_layout.addWidget(top_frame)

        # University logo
        university_logo = QLabel(top_frame)
        pixmap = QPixmap("images/university_logo.png").scaled(125, 150, Qt.KeepAspectRatio)
        university_logo.setPixmap(pixmap)
        top_layout.addWidget(university_logo, alignment=Qt.AlignLeft)

        # Details text using QLabel
        details_text = ("Northern Technical University\n"
                        "Technical Engineering College\n"
                        "Computer Engineering Department")
        details_label = QLabel(details_text, top_frame)
        details_label.setFont(QFont("georgia", 14, QFont.Bold))
        details_label.setAlignment(Qt.AlignCenter)
        details_label.setStyleSheet("background-color: white; color: darkblue;")
        top_layout.addWidget(details_label)

        # College logo
        college_logo = QLabel(top_frame)
        pixmap = QPixmap("images/college_logo.png").scaled(125, 150, Qt.KeepAspectRatio)
        college_logo.setPixmap(pixmap)
        top_layout.addWidget(college_logo, alignment=Qt.AlignRight)

        # Middle frame
        middle_frame = QFrame(self)
        middle_frame.setStyleSheet("background-color: #f0f0f0;")
        middle_layout = QGridLayout(middle_frame)
        middle_layout.setSpacing(30)
        main_layout.addWidget(middle_frame)

        # Title
        title = QLabel("Arabic Sign Language Communication", middle_frame)
        title.setFont(QFont("Georgia", 16, QFont.Bold))
        font = title.font()
        font.setItalic(True)
        title.setFont(font)
        title.setStyleSheet("color: darkblue; background-color: #f0f0f0;")
        middle_layout.addWidget(title, 0, 0, 2, 2, alignment=Qt.AlignHCenter | Qt.AlignTop)

        # Buttons
        button1 = QPushButton("Sign Language to Voice", middle_frame)
        button1.setFont(QFont("georgia", 12, QFont.Bold))
        button1.setFixedSize(400, 70)
        button1.setStyleSheet("QPushButton { border: 3px solid darkblue; border-radius: 10px; background-color: darkblue; color: white; } QPushButton:hover { background-color: #CCCCCC; color: darkblue; } ")
        button1.clicked.connect(lambda: self.parent.stacked_widget.setCurrentIndex(1))  # Go to GTOS page
        middle_layout.addWidget(button1, 1, 0, alignment=Qt.AlignCenter | Qt.AlignTop)

        exit_button = QPushButton("Exit", middle_frame)
        exit_button.setFont(QFont("georgia", 12, QFont.Bold))
        exit_button.setFixedSize(400, 60)
        exit_button.setStyleSheet("QPushButton { border: 3px solid darkblue; border-radius: 10px; background-color: darkblue; color: white; } QPushButton:hover { background-color: #CCCCCC; color: darkblue; } ")
        exit_button.clicked.connect(self.on_exit)
        middle_layout.addWidget(exit_button, 3, 0, alignment=Qt.AlignCenter)

        # System image
        system_image = QLabel(middle_frame)
        pixmap = QPixmap("images/Voice_to_Sign.jpg").scaled(600, 800, Qt.KeepAspectRatio)
        system_image.setPixmap(pixmap)
        middle_layout.addWidget(system_image, 0, 1, 5, 1, alignment=Qt.AlignCenter)

        # Bottom frame
        bottom_frame = QFrame(self)
        bottom_frame.setStyleSheet("background-color: white;")
        bottom_frame.setFixedHeight(60)
        bottom_layout = QVBoxLayout(bottom_frame)
        main_layout.addWidget(bottom_frame)

        # Dark blue line
        canvas = QLabel(bottom_frame)
        canvas.setFixedSize(1000, 3)
        canvas.setStyleSheet("background-color: darkblue;")
        bottom_layout.addWidget(canvas, alignment=Qt.AlignCenter)

        # Copyright text
        year = QLabel("Copyright © Lamis Ali Hussein 2024", bottom_frame)
        year.setFont(QFont("Arial", 9, QFont.Bold))
        year.setStyleSheet("color: darkblue; background-color: white;")
        year.setAlignment(Qt.AlignCenter)
        bottom_layout.addWidget(year)

    def on_exit(self):
        reply = QMessageBox.question(self, 'Exit Application', "Would you like to exit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            QApplication.instance().quit()


class GTOS(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.stacked_widget = QStackedWidget()

        # Main layout
        main_layout = QVBoxLayout()

        self.label = QLabel(' Sign Language Recognition ', self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: darkblue; background-color: #f0f0f0;")
        self.label.setFixedHeight(70)
        font = QFont("georgia", 16, QFont.Bold)
        font.setItalic(True)
        self.label.setFont(font)
        main_layout.addWidget(self.label)

        row_buttons_layout = QHBoxLayout()                # Horizontal layout for the buttons

        self.button1 = QPushButton('Data Collection', self)
        self.button1.setIcon(QIcon('images/Data_Collection.png'))
        self.button1.setIconSize(QSize(80, 60))
        self.button1.setFont(QFont("Arial", 12, QFont.Bold))
        self.button1.setCursor(Qt.PointingHandCursor)
        self.button1.setStyleSheet("QPushButton { border: 3px solid darkblue; border-radius: 10px; background-color: lightblue; color: black; } QPushButton:hover { border: 3px solid darkorange; background-color: #CCCCCC; color: black; }")
        self.button1.clicked.connect(self.show_data_collection)
        row_buttons_layout.addWidget(self.button1)

        self.button3 = QPushButton('Training', self)
        self.button3.setIcon(QIcon('images/Train_Model.png'))
        self.button3.setIconSize(QSize(80, 60))
        self.button3.setFont(QFont("Arial", 12, QFont.Bold))
        self.button3.setCursor(Qt.PointingHandCursor)
        self.button3.setStyleSheet("QPushButton { border: 3px solid darkblue; border-radius: 10px; background-color: lightblue; color: black; } QPushButton:hover { border: 3px solid darkorange; background-color: #CCCCCC; color: black; }")
        self.button3.clicked.connect(self.show_training )
        row_buttons_layout.addWidget(self.button3)

        self.button4 = QPushButton('Testing', self)
        self.button4.setIcon(QIcon('images/Test_Model.png'))
        self.button4.setIconSize(QSize(80, 60))
        self.button4.setFont(QFont("Arial", 12, QFont.Bold))
        self.button4.setCursor(Qt.PointingHandCursor)
        self.button4.setStyleSheet("QPushButton { border: 3px solid darkblue; border-radius: 10px; background-color: lightblue; color: black; } QPushButton:hover { border: 3px solid darkorange; background-color: #CCCCCC; color: black; }")
        self.button4.clicked.connect(self.show_test_model)
        row_buttons_layout.addWidget(self.button4)
        main_layout.addLayout(row_buttons_layout)

        # Add stacked widget to main layout
        main_layout.addWidget(self.stacked_widget)
        main_layout.setContentsMargins(60, 10, 60, 10)
        # Add DataCollectionWidget to stacked widget
        self.data_collection_widget = DataCollectionWidget(self)
        self.stacked_widget.addWidget(self.data_collection_widget)
        # Add TrainingWidget to stacked widget
        self.training_widget = TrainingWidget(self)
        self.stacked_widget.addWidget(self.training_widget)
        # Add TestingWidget to stacked widget
        self.testing_widget = TestingWidget(self)
        self.stacked_widget.addWidget(self.testing_widget)

        self.setLayout(main_layout)

        # Bottom frame
        frame3 = QFrame(self)
        frame3.setStyleSheet("background-color: white;")
        frame3.setFixedHeight(60)
        frame3_layout = QHBoxLayout(frame3)

        # زر العودة
        self.back_button = QPushButton(' Back ', frame3)
        self.back_button.setCursor(Qt.PointingHandCursor)
        self.back_button.setFixedSize(100, 40)
        self.back_button.setFont(QFont("georgia", 10, QFont.Bold))
        self.back_button.setStyleSheet("QPushButton { border: 3px solid darkblue; border-radius: 10px; background-color: darkblue; color: white; }" "QPushButton:hover { background-color: #CCCCCC; color: darkblue; }")
        self.back_button.clicked.connect(self.back)
        frame3_layout.addWidget(self.back_button)

        # زر الخروج
        self.exit_button = QPushButton("Exit", frame3)
        self.exit_button.setFixedSize(100, 40)
        self.exit_button.setFont(QFont("georgia", 10, QFont.Bold))
        self.exit_button.setStyleSheet("QPushButton { border: 3px solid darkblue; border-radius: 10px; background-color: darkblue; color: white; }" "QPushButton:hover { background-color: #CCCCCC; color: darkblue; }")
        self.exit_button.clicked.connect(self.on_exit)
        frame3_layout.addWidget(self.exit_button)

        main_layout.addWidget(frame3)
        self.setLayout(main_layout)

    def back(self):
        self.parentWidget().setCurrentIndex(0)
        self.data_collection_widget.hide_elements()
        self.training_widget.hide_elements()
        self.testing_widget.hide_elements()

    def on_exit(self):
        reply = QMessageBox.question(self, 'Exit Application', "Would you like to exit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.image_window_open:
                cv2.destroyAllWindows()
            QApplication.instance().quit()

    def show_data_collection(self):
        self.stacked_widget.setCurrentWidget(self.data_collection_widget)
        self.data_collection_widget.show_elements()
        self.training_widget.hide_elements()
        self.testing_widget.hide_elements()

    def show_training(self):
        self.stacked_widget.setCurrentWidget(self.training_widget)
        self.training_widget.show_elements()
        self.data_collection_widget.hide_elements()
        self.testing_widget.hide_elements()

    def show_test_model(self):
        self.stacked_widget.setCurrentWidget(self.testing_widget)
        self.testing_widget.show_elements()
        self.data_collection_widget.hide_elements()
        self.training_widget.hide_elements()




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Two Way Arabic Sign Language Communication')
        self.setFixedSize(1200, 800)
        self.setWindowIcon(QIcon("icons/windowLogo.png"))

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.dashboard = Dashboard(self)
        self.gtos = GTOS(self)

        self.stacked_widget.addWidget(self.dashboard)
        self.stacked_widget.addWidget(self.gtos)

        self.center()

    def center(self):
        screen = QDesktopWidget().availableGeometry().center()
        size = self.frameGeometry()
        size.moveCenter(screen)
        self.move(size.topLeft())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


