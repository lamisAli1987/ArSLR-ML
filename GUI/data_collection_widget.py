# data_collection_widget.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QLineEdit, QFrame, QSizePolicy, QSpacerItem, QMessageBox
from PyQt5.QtGui import QFont, QIcon, QCursor
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from arabic_labels_dict import arabic_labels_dict  # استيراد القاموس
import os
import cv2
import subprocess  #  دالة تستخدم في فتح مجلد البيانات
import pickle
import mediapipe as mp
from PIL import Image
import numpy as np
import warnings
import shutil
import tempfile
import glob


# إخفاء تحذيرات Mediapipe و Protobuf
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='mediapipe')
warnings.filterwarnings("ignore", message="Feedback manager requires a model with a single signature inference.")


class DataCollectionWidget(QWidget):
    # Define the signal here
    data_collection_requested = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.number_of_classes = 0
        self.dataset_size = 0
        self.image_window_open = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        horizontal_layout = QHBoxLayout()    # إنشاء الطبقة الأفقية

        # إضافة label1 إلى الطبقة الأفقية
        self.label1 = QLabel('Number of classes required ....', self)
        self.label1.setFont(QFont("Arial", 11, QFont.Bold))
        horizontal_layout.addWidget(self.label1)
        # إنشاء مساحة مرنة للفراغ بين label1 والزر
        horizontal_layout.addStretch()

        # إضافة الزر إلى الطبقة الأفقية
        self.show_image_button = QPushButton('', self)
        self.show_image_button.setIcon(QIcon('images/Image_button.png'))
        self.show_image_button.setIconSize(QSize(60, 45))
        self.show_image_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.show_image_button.setStyleSheet("QPushButton { border: 3px solid white; border-radius: 10px; background-color: white; color: black; } QPushButton:hover { border: 3px solid white; background-color: white; color: black; }")
        self.show_image_button.clicked.connect(self.opening)
        horizontal_layout.addWidget(self.show_image_button)

        layout.addLayout(horizontal_layout)    # إضافة الطبقة الأفقية إلى الطبقة العمودية

        layout.addSpacing(8)    # إضافة المسافة هنا

        self.entry1 = QLineEdit(self)
        self.entry1.setFixedHeight(40)
        self.entry1.setFixedWidth(200)
        self.entry1.setFont(QFont("Arial", 11, QFont.Bold))
        self.entry1.setAlignment(Qt.AlignCenter)  # تحديد محاذاة النص إلى الوسط
        layout.addWidget(self.entry1)

        layout.addSpacing(8)  # إضافة المسافة هنا

        self.label2 = QLabel('Number of data entered for each class ....', self)
        self.label2.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(self.label2)

        self.entry2 = QLineEdit(self)
        self.entry2.setFixedHeight(40)
        self.entry2.setFixedWidth(200)
        self.entry2.setFont(QFont("Arial", 11, QFont.Bold))
        self.entry2.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.entry2)

        layout.addSpacing(8)

        self.label3 = QLabel('Class Name', self)
        self.label3.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(self.label3)

        self.entry3 = QLineEdit(self)
        self.entry3.setFixedHeight(40)
        self.entry3.setFixedWidth(200)
        self.entry3.setFont(QFont("Arial", 11, QFont.Bold))
        self.entry3.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.entry3)

        layout.addSpacing(10)

        self.label4 = QLabel(' ', self)
        self.label4.setFont(QFont("Arial", 12, QFont.Bold))
        self.label4.setFixedSize(600, 70)
        self.label4.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.label4.setAlignment(Qt.AlignCenter)
        self.label4.setStyleSheet("border: 1px solid black; background-color: white; padding: 5px;")
        layout.addWidget(self.label4, alignment=Qt.AlignCenter)

        layout.addSpacing(20)

        buttons_layout = QHBoxLayout()    # Horizontal layout for the buttons
        self.open_camera_button = QPushButton('Start Camera', self)
        self.open_camera_button.setFixedSize(220, 60)
        self.open_camera_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.open_camera_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } "  "QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.open_camera_button.clicked.connect(self.open_data_collection)
        buttons_layout.addWidget(self.open_camera_button)

        buttons_layout.addSpacerItem(QSpacerItem(25, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))   # Adding a fixed spacer between the buttons

        self.data_modification_button = QPushButton('Data Modification', self)
        self.data_modification_button.setFixedSize(230, 60)
        self.data_modification_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.data_modification_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; }"  "QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.data_modification_button.clicked.connect(self.open_collect_images_in_specific_folder)
        buttons_layout.addWidget(self.data_modification_button)

        buttons_layout.addSpacerItem(QSpacerItem(25, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))   # Adding a fixed spacer between the buttons

        self.directory_button = QPushButton('Open Directory', self)
        self.directory_button.setFixedSize(230, 60)
        self.directory_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.directory_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; }"  "QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.directory_button.clicked.connect(self.open_data_directory)  # Connect the button to open_directory
        buttons_layout.addWidget(self.directory_button)

        buttons_layout.addSpacerItem( QSpacerItem(25, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))  # Adding a fixed spacer between the buttons

        self.create_dataset_button = QPushButton('Create Dataset', self)
        self.create_dataset_button.setIcon(QIcon('images/Create_Dataset.png'))
        self.create_dataset_button.setIconSize(QSize(70, 50))
        self.create_dataset_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.create_dataset_button.setStyleSheet("QPushButton { border: 3px solid darkblue; border-radius: 10px; background-color: lightblue; color: black; } QPushButton:hover { border: 3px solid darkorange; background-color: #CCCCCC; color: black; }")
        self.create_dataset_button.clicked.connect(self.create_dataset)  # Connect the button to open_directory
        buttons_layout.addWidget(self.create_dataset_button)

        buttons_layout.setAlignment(Qt.AlignCenter)  # Center-align the buttons layout
        layout.addLayout(buttons_layout)  # Add buttons layout to the main layout

        self.setLayout(layout)
        self.hide_elements()

    def hide_elements(self):
        self.label1.hide()
        self.entry1.hide()
        self.label2.hide()
        self.entry2.hide()
        self.label3.hide()
        self.entry3.hide()
        self.label4.hide()
        self.show_image_button.hide()
        self.open_camera_button.hide()
        self.data_modification_button.hide()
        self.directory_button.hide()
        self.create_dataset_button.hide()

    def show_elements(self):
        self.label1.show()
        self.entry1.show()
        self.label2.show()
        self.entry2.show()
        self.label3.show()
        self.entry3.show()
        self.label4.show()
        self.show_image_button.show()
        self.open_camera_button.show()
        self.data_modification_button.show()
        self.directory_button.show()
        self.create_dataset_button.show()

    def show_data_collection(self):  #  show_data_collection():
        self.data_collection_requested.emit()


    def opening(self):
        if not self.image_window_open:
            cv2.namedWindow("Arabic Sign Language", cv2.WINDOW_NORMAL)
            image = cv2.imread('template.jpg')
            cv2.imshow("Arabic Sign Language", image)
            cv2.moveWindow("Arabic Sign Language", 1500, 120)
            self.image_window_open = True  # تعيين حالة النافذة الفرعية إلى مفتوحة
        else:
            # إغلاق النافذة الفرعية
            cv2.destroyWindow("Arabic Sign Language")
            self.image_window_open = False  # تعيين حالة النافذة الفرعية إلى مغلقة


    # function that opens the folder containing the captured images
    def open_data_directory(self):
        data_directory = './My_data'
        if os.path.exists(data_directory):
            try:
                subprocess.Popen(f'explorer {os.path.realpath(data_directory)}')
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open directory: {e}")
        else:
            QMessageBox.critical(self, "Error", "Data directory does not exist!")

    def open_dataset_directory(self):  # A function that opens the folder containing the Dataset file.
        program_directory = os.path.dirname(os.path.realpath(__file__))
        dataset_file = 'dataset.pickle'
        dataset_full_path = os.path.join(program_directory, dataset_file)
        if os.path.exists(dataset_full_path):
            try:
                subprocess.Popen(f'explorer {os.path.realpath(program_directory)}')
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open directory: {e}")
        else:
            QMessageBox.critical(self, "Error", "Dataset file does not exist!")

    def update_number_of_classes(self):
        try:
            self.number_of_classes = int(self.entry1.text())
        except ValueError:
            self.number_of_classes = 0

    def update_dataset_size(self):
        try:
            self.dataset_size = int(self.entry2.text())
        except ValueError:
            self.dataset_size = 0

    def open_data_collection(self):
        self.entry3.setReadOnly(True)  # Disable writing in self.entry3
        self.update_number_of_classes()
        self.update_dataset_size()

        DATA_DIR = os.path.abspath('./My_data')
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        if self.number_of_classes <= 0:
            QMessageBox.warning(self, "Warning", "Please enter a valid number of classes.")
            return

        if self.dataset_size <= 0:
            QMessageBox.warning(self, "Warning", "Please enter a valid dataset size.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open image capture.")
            return
        cv2.namedWindow('Take Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Take Image', 800, 600)
        cv2.moveWindow('Take Image', 600, 10)

        for j in range(self.number_of_classes):
            if j >= len(arabic_labels_dict):
                QMessageBox.critical(self, "Error", f"Not enough labels in the dictionary for class {j}.")
                return
            class_label = arabic_labels_dict[j]
            class_dir = os.path.join(DATA_DIR, class_label)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            self.label4.setText(f'Collecting data for class {class_label}')
            self.label4.repaint()

            while True:
                ret, frame = cap.read()
                if not ret:
                    QMessageBox.critical(self, "Error", "Failed to capture image.")
                    return
                frame_resized = cv2.resize(frame, (900, 700))
                cv2.putText(frame_resized, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                            (100, 255, 0), 3, cv2.LINE_AA)
                cv2.imshow('Take Image', frame_resized)
                if cv2.waitKey(1) == ord('q'):
                    break
            counter = 0
            while counter < self.dataset_size:
                ret, frame = cap.read()
                if not ret:
                    QMessageBox.critical(self, "Error", "Failed to capture image.")
                    return
                frame_resized = cv2.resize(frame, (900, 700))
                cv2.imshow('Take Image', frame_resized)
                key = cv2.waitKey(1)
                if key == 27:  # Check for ESC key to stop the process
                    return
                temp_img_path = os.path.join(DATA_DIR, 'temp.jpg')
                try:
                    print(f"Saving image to {temp_img_path}")  # Message to print the path of the saved image
                    cv2.imwrite(temp_img_path, frame_resized)
                    final_img_path = os.path.join(class_dir, '{}.jpg'.format(counter))
                    # Ensure the directory exists before moving the file
                    if not os.path.exists(class_dir):
                        os.makedirs(class_dir)
                    if os.path.exists(temp_img_path):
                        os.rename(temp_img_path, final_img_path)
                        print(f"Moved image to {final_img_path}")
                    else:
                        raise IOError(f"Image file not found at {temp_img_path}")
                except Exception as e:
                    print(f"Failed to save image {counter} to {final_img_path}: {e}")
                counter += 1
        cap.release()
        cv2.destroyAllWindows()

        self.entry3.setReadOnly(False)  # Re-enable writing in self.entry3
        self.label4.setText(f'Data collection complete for {self.number_of_classes} classes.')

    def open_collect_images_in_specific_folder(self):
        self.entry1.setReadOnly(True)  # Disable writing in self.entry3
        self.update_dataset_size()  # Update dataset_size

        DATA_DIR = './My_data'

        folder_name = self.entry3.text()
        folder_path = os.path.join(DATA_DIR, folder_name)

        if self.dataset_size <= 0:
            QMessageBox.warning(self, "Warning", "Please enter a valid dataset size.")
            return

        if not folder_name:
            QMessageBox.warning(self, "Warning", "Please enter a valid folder name.")
            return

        if not os.path.exists(folder_path):
            QMessageBox.warning(self, "Warning", f"Folder '{folder_name}' does not exist.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open image capture.")
            return

        # Set the camera resolution to 1280x720
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        cv2.namedWindow('Take Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Take Image', 800, 600)
        cv2.moveWindow('Take Image', 600, 10)

        # Temporary directory to store images before moving them to the final folder
        with tempfile.TemporaryDirectory() as temp_dir:
            # Display ready message and wait for 'Q' to start capturing images
            while True:
                ret, frame = cap.read()
                if not ret:
                    QMessageBox.critical(self, "Error", "Failed to capture image.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                # No need to resize, as the frame is already captured at 1280x720
                cv2.putText(frame, 'Ready? Press "q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                            (100, 255, 100), 3, cv2.LINE_AA)
                cv2.imshow('Take Image', frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            counter = 0  # Start capturing images
            while counter < self.dataset_size:
                ret, frame = cap.read()
                if not ret:
                    QMessageBox.critical(self, "Error", "Failed to capture image.")
                    break

                # No need to resize, frame is already 1280x720
                cv2.imshow('Take Image', frame)
                key = cv2.waitKey(1)
                if key == 27:  # Check for ESC key
                    break

                # Save the captured frame to the temporary folder
                temp_image_path = os.path.join(temp_dir, '{}.jpg'.format(counter))
                cv2.imwrite(temp_image_path, frame)  # Save the original frame as is
                print(f"Saving image to {temp_image_path}")
                counter += 1

            # Move images from the temporary folder to the final folder
            for temp_image_name in os.listdir(temp_dir):
                temp_image_path = os.path.join(temp_dir, temp_image_name)
                final_image_path = os.path.join(folder_path, temp_image_name)
                shutil.move(temp_image_path, final_image_path)
                print(f"Moved image to {final_image_path}")

        cap.release()
        cv2.destroyAllWindows()
        self.label4.setText(f' Data Modification complete for folder : {folder_name}')
        self.entry1.setReadOnly(False)  # Re-enable writing in self.entry3

    def process_image(self, img_path, hands):
        data_aux = []
        x_ = []
        y_ = []

        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Error reading image: {img_path}")
            print(e)
            return None

        img_rgb = img.convert("RGB")
        img_np = np.array(img_rgb)

        results = hands.process(img_np)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

        return data_aux

    def create_dataset(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        DATA_DIR = './My_data'
        data = []
        labels = []
        image_files = glob.glob(os.path.join(DATA_DIR, '**/*.jpg'), recursive=True)
        for img_path in image_files:
            result = self.process_image(img_path, hands)
            if result:
                label = os.path.basename(os.path.dirname(img_path))
                data.append(result)
                labels.append(label)
        with open('dataset.pickle', 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)

        hands.close()
        QMessageBox.information(None, "Success", "Dataset created successfully!")
        self.open_dataset_directory()