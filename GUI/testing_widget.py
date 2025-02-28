# testing_Widget.py
import sys
import cv2
import mediapipe as mp
import numpy as np
import pickle
import ctypes
import datetime
from PyQt5.QtWidgets import QApplication, QSizePolicy, QSpacerItem, QMessageBox, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtGui import QFont, QIcon, QCursor, QImage, QPixmap
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QTimer
from PIL import Image, ImageDraw, ImageFont
from Testing_Letter_sound import labels_dict

class TestingWidget(QWidget):
    testing_requested = pyqtSignal()  # Define the signal here

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_window_open = False
        self.testing_running = False
        self.init_ui()
        self.load_models()  # Load models during initialization
        self.init_mediapipe()

    def init_ui(self):
        layout = QVBoxLayout()  # Main layout

        # First row layout (title and show image button)
        first_row_layout = QHBoxLayout()
        first_row_layout.setContentsMargins(120, 0, 80, 0)

        self.label = QLabel("Scan Gestures", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("georgia", 12, QFont.Bold))
        first_row_layout.addWidget(self.label)

        first_row_layout.addStretch()

        self.show_image_button = QPushButton('', self)
        self.show_image_button.setIcon(QIcon('images/Image_button.png'))
        self.show_image_button.setIconSize(QSize(70, 50))
        self.show_image_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.show_image_button.setStyleSheet("QPushButton { border: 3px solid white; border-radius: 10px; background-color: white; color: black; } QPushButton:hover { border: 3px solid white; background-color: white; color: black; }")
        self.show_image_button.clicked.connect(self.opening)
        first_row_layout.addWidget(self.show_image_button)

        layout.addLayout(first_row_layout)
        layout.addSpacing(10)

        # Second row layout (video and buttons)
        second_row_layout = QHBoxLayout()
        second_row_layout.setContentsMargins(70, 0, 20, 0)

        right_column_layout = QVBoxLayout()

        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setFixedSize(1150, 550)
        self.video_label.setAlignment(Qt.AlignTop)
        right_column_layout.addWidget(self.video_label)

        second_row_layout.addLayout(right_column_layout)

        left_column_layout = QVBoxLayout()
        left_column_layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)

        self.test_model_button = QPushButton('Testing', self)
        self.test_model_button.setFixedSize(200, 50)
        self.test_model_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.test_model_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.test_model_button.clicked.connect(self.start_video)
        left_column_layout.addWidget(self.test_model_button)

        # Adding the new button for "Testing Number"
        self.test_number_button = QPushButton('Testing Number', self)
        self.test_number_button.setFixedSize(200, 50)
        self.test_number_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.test_number_button.setStyleSheet( "QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        left_column_layout.addWidget(self.test_number_button)

        self.stop_test_button = QPushButton('Stop', self)
        self.stop_test_button.setFixedSize(200, 50)
        self.stop_test_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.stop_test_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.stop_test_button.clicked.connect(self.stop_video)
        left_column_layout.addWidget(self.stop_test_button)

        self.clock_label = QLabel(self)
        self.clock_label.setFont(QFont("Arial", 11))
        self.clock_label.setStyleSheet("border: 3px solid black; border-radius: 10px; background-color: white; color: black; padding: 5px;")
        self.clock_label.setAlignment(Qt.AlignCenter)
        left_column_layout.addWidget(self.clock_label)

        self.calendar_label = QLabel(self)
        self.calendar_label.setFont(QFont("Arial", 11))
        self.calendar_label.setStyleSheet("border: 3px solid black; border-radius: 10px; background-color: white; color: black; padding: 5px;")
        self.calendar_label.setAlignment(Qt.AlignCenter)
        left_column_layout.addWidget(self.calendar_label)
        left_column_layout.addStretch(1)

        second_row_layout.addLayout(left_column_layout)
        layout.addLayout(second_row_layout)
        self.setLayout(layout)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.update_clock()

    def load_models(self):
        # Load only the SVM model
        with open('svm_nonlinear_letter_classifier.p', 'rb') as f:
            self.svm_letters = pickle.load(f)['model']

        # Updated labels for numbers only
        self.labels_dict = {0: 'أ', 1: 'ب', 2: 'ة', 3: 'ت', 4: 'ث', 5: 'ج', 6: 'ح', 7: 'خ', 8: 'د', 9: 'ذ', 10: 'ر', 11: 'ز',
                       12: 'س', 13: 'ش', 14: 'ص', 15: 'ض', 16: 'ط', 17: 'ظ', 18: 'ع', 19: 'غ', 20: 'ف', 21: 'ق',
                       22: 'ك', 23: 'ل', 24: 'لا', 25: 'م', 26: 'ن', 27: 'هـ', 28: 'و', 29: 'ي'}

        #self.labels_dict = { 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10'}

    def init_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def start_video(self):
        self.testing_running = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open camera.")
            return
        self.timer.start(30)

    def hide_elements(self):
        self.label.hide()
        self.show_image_button.hide()
        self.test_model_button.hide()
        self.stop_test_button.hide()

    def show_elements(self):
        self.label.show()
        self.show_image_button.show()
        self.test_model_button.show()
        self.stop_test_button.show()

    def show_test_model(self):
        self.testing_requested.emit()

    def opening(self):
        if not self.image_window_open:
            cv2.namedWindow("Arabic Sign Language", cv2.WINDOW_NORMAL)
            self.remove_close_button("Arabic Sign Language")
            image = cv2.imread('template.jpg')
            cv2.imshow("Arabic Sign Language", image)
            cv2.moveWindow("Arabic Sign Language", 1500, 120)
            self.image_window_open = True
        else:
            cv2.destroyWindow("Arabic Sign Language")
            self.image_window_open = False

    def remove_close_button(self, window_name):
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        if hwnd:
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -16)
            style &= ~0x80000
            ctypes.windll.user32.SetWindowLongW(hwnd, -16, style)
            self.update_window(hwnd)

    def update_window(self, hwnd):
        ctypes.windll.user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, 0x0002 | 0x0001 | 0x0020)

    def update_clock(self):
        now = datetime.datetime.now()
        self.clock_label.setText(now.strftime("%H:%M:%S"))
        self.calendar_label.setText(now.strftime("%B %d, %Y"))
        QTimer.singleShot(1000, self.update_clock)

    def convert_to_arabic_numerals(self, number):
        # Create a mapping for Arabic numerals
        english_to_arabic = {'0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
                             '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'}
        # Convert number to string and replace each digit with the corresponding Arabic numeral
        return ''.join(english_to_arabic[digit] if digit in english_to_arabic else digit for digit in str(number))

    def draw_text_with_pil(image, text, position, font_path='arial.ttf', font_size=90):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.truetype(font_path, font_size)
        arabic_char = labels_dict[int(text)] if isinstance(text, (int, np.integer)) else text
        draw.text(position, arabic_char, font=font, fill=(255, 255, 255, 255))  # White color
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def stop_video(self):
        self.testing_running = False
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def update_frame(self):
        if self.testing_running:
            ret, frame = self.cap.read()
            if not ret:
                QMessageBox.critical(self, "Error", "Could not read frame from camera.")
                return
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                # Processing for prediction
                hand_data = []
                for hand_landmark in results.multi_hand_landmarks:
                    landmarks = hand_landmark.landmark
                    hand_data.append([landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks])
                hand_data = np.array(hand_data)

                # Make predictions using only the SVM model
                if len(hand_data) > 0:
                    svm_prediction = self.svm_letters.predict(hand_data)
                    predicted_label = svm_prediction[0]

                    # Displaying the results in Arabic numerals
                    if predicted_label in self.labels_dict:
                        text_to_display = self.labels_dict[predicted_label]
                        self.label.setText(text_to_display)
                        print(f"Predicted: {text_to_display}")
                    else:
                        print("Unknown prediction")

            # Display the video feed
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            h, w, ch = frame_bgr.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(frame_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888)
            self.video_label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    testing_widget = TestingWidget()
    testing_widget.show()
    sys.exit(app.exec_())
