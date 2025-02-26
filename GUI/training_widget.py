# training_Widget.py
from PyQt5.QtWidgets import QSizePolicy, QSpacerItem, QMessageBox, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QLineEdit, QFrame, QWIDGETSIZE_MAX
from PyQt5.QtGui import QFont, QIcon, QCursor
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QUrl
from PyQt5.QtGui import QDesktopServices   # To open the classification report file
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
import cv2
import subprocess  # Function used to open data folder
import pickle
import ctypes
import mediapipe as mp



class TrainingWidget(QWidget):
    # Define the signal here
    training_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.first_click = True # Variable to track when the button was first pressed (prints the time taken to train the model)
        self.image_window_open = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # First horizontal layout
        horizontal_layout1 = QHBoxLayout()

        # Label "Training the model"
        self.label = QLabel("  Training the Model ", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Arial", 12, QFont.Bold))
        horizontal_layout1.addWidget(self.label)

        # Stretch to add space
        horizontal_layout1.addStretch()

        # Create the button
        self.show_image_button = QPushButton('', self)
        self.show_image_button.setIcon(QIcon('images/Image_button.png'))
        self.show_image_button.setIconSize(QSize(70, 50))
        self.show_image_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.show_image_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.show_image_button.setStyleSheet(  "QPushButton { border: 3px solid white; border-radius: 10px; background-color: white; color: black; } QPushButton:hover { border: 3px solid white; background-color: white; color: black; }")
        self.show_image_button.clicked.connect(self.opening)
        horizontal_layout1.addWidget(self.show_image_button)

        layout.addLayout(horizontal_layout1)   # Add the first horizontal layout to the main vertical layout
        layout.addSpacing(20)   # Add space

        horizontal_layout2 = QHBoxLayout()            # Second horizontal layout

        # Result label
        self.result_label = QLabel('', self)
        self.result_label.setAlignment(Qt.AlignLeft)
        self.result_label.setFont(QFont("Arial", 12))
        self.result_label.setWordWrap(True)
        self.result_label.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.result_label.setMinimumHeight(200)
        self.result_label.setMinimumWidth(280)
        self.result_label.setStyleSheet("border: 1px solid black; background-color: white; padding: 5px;")
        horizontal_layout2.addWidget(self.result_label)

        # Fixed spacer between the buttons
        horizontal_layout2.addSpacerItem(QSpacerItem(50, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))

        # Metrics label
        self.metrics_label = QLabel('', self)
        self.metrics_label.setAlignment(Qt.AlignLeft)
        self.metrics_label.setFont(QFont("Arial", 12))
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.metrics_label.setMinimumHeight(200)
        self.metrics_label.setMinimumWidth(280)
        self.metrics_label.setStyleSheet("border: 1px solid black; background-color: white; padding: 5px;")
        horizontal_layout2.addWidget(self.metrics_label)

        layout.addLayout(horizontal_layout2)           # Add the second horizontal layout to the main vertical layout
        layout.addSpacing(20)   # Add space

        buttons_layout = QHBoxLayout()           # Third horizontal layout for buttons

        self.open_dataset_file_button = QPushButton('Dataset File', self)
        self.open_dataset_file_button.setFixedSize(180, 60)
        self.open_dataset_file_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.open_dataset_file_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.open_dataset_file_button.clicked.connect(self.open_dataset_directory)
        buttons_layout.addWidget(self.open_dataset_file_button)

        buttons_layout.addSpacerItem(QSpacerItem(50, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))   # Fixed spacer between the buttons

        self.train_button = QPushButton("Train Model", self)
        self.train_button.setFixedSize(180, 60)
        self.train_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.train_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.train_button.clicked.connect(self.train_model)
        buttons_layout.addWidget(self.train_button)

        buttons_layout.addSpacerItem(QSpacerItem(50, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))   # Fixed spacer between the buttons

        self.classification_report_button = QPushButton("Classification Report", self)
        self.classification_report_button.setFixedSize(260, 60)
        self.classification_report_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.classification_report_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.classification_report_button.clicked.connect(self.classification_report)
        buttons_layout.addWidget(self.classification_report_button)

        buttons_layout.addSpacerItem(            QSpacerItem(50, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))  # Fixed spacer between the buttons

        self.classification_report_button = QPushButton("Confusion Matrix", self)
        self.classification_report_button.setFixedSize(260, 60)
        self.classification_report_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.classification_report_button.setStyleSheet(
            "QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.classification_report_button.clicked.connect(self.classification_report)
        buttons_layout.addWidget(self.classification_report_button)

        buttons_layout.setAlignment(Qt.AlignCenter)     # Align buttons horizontally
        layout.addLayout(buttons_layout)             # Add the buttons layout to the main vertical layout

        self.setLayout(layout)            # Set the final layout for the window
        self.hide_elements()             # Hide elements initially


    def hide_elements(self):
        self.label.hide()
        self.show_image_button.hide()
        self.result_label.hide()
        self.metrics_label.hide()
        self.open_dataset_file_button.hide()
        self.train_button.hide()
        self.classification_report_button.hide()

    def show_elements(self):
        self.label.show()
        self.show_image_button.show()
        self.result_label.show()
        self.metrics_label.show()
        self.open_dataset_file_button.show()
        self.train_button.show()
        self.classification_report_button.show()

    def show_training(self):
        self.training_requested.emit()

    def opening(self):
        if not self.image_window_open:
            # Create a sub-window without buttons
            cv2.namedWindow("Arabic Sign Language", cv2.WINDOW_NORMAL)
            self.remove_close_button("Arabic Sign Language")
            image = cv2.imread('template.jpg')
            cv2.imshow("Arabic Sign Language", image)
            cv2.moveWindow("Arabic Sign Language", 1500, 120)
            self.image_window_open = True  # Set the subwindow state to open.
        else:
            cv2.destroyWindow("Arabic Sign Language")
            self.image_window_open = False  # Set the subwindow state to close.

    def remove_close_button(self, window_name): # A function used to remove the close mark from the sub-window that displays Arabic sign language gestures.
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        if hwnd:
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -16)
            style &= ~0x80000  # Remove WS_SYSMENU property
            ctypes.windll.user32.SetWindowLongW(hwnd, -16, style)
            self.update_window(hwnd)

    def update_window(self, hwnd):
        ctypes.windll.user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, 0x0002 | 0x0001 | 0x0020)

    def update_label(self, new_text):
        current_text = self.result_label.text()
        updated_text = current_text + "\n" + new_text
        self.result_label.setText(updated_text)

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

    def train_model(self):
        try:
            dataset_path = './dataset.pickle'

            # Update user interface
            self.result_label.setText('')  # مسح المحتوى السابق
            self.update_label("Loading dataset...")

            # Download group data
            data_dict = pickle.load(open(dataset_path, 'rb'))
            self.update_label("Dataset loaded.")

            data = data_dict['data']
            labels = data_dict['labels']
            total_samples = len(data)
            self.update_label(f"Total number of samples: {total_samples}")

            # Ensure data consistency
            data_lengths = [len(d) for d in data]
            unique_lengths = set(data_lengths)

            if len(unique_lengths) > 1:
                max_length = max(data_lengths)
                for i in range(len(data)):
                    if len(data[i]) < max_length:
                        data[i] = np.pad(data[i], (0, max_length - len(data[i])), 'constant')

            # Convert data to NumPy arrays
            data = np.asarray(data)
            labels = np.asarray(labels)

            self.update_label("Splitting into training and testing datasets...")
            x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, shuffle=True,
                                                              stratify=labels)
            self.update_label("Training and testing datasets created.")

            # Train classifier using Grid Search for best parameters
            parameters = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']}
            classifier = SVC(probability=True)
            grid_search = GridSearchCV(classifier, parameters, cv=10, n_jobs=-1, refit=True, verbose=2)

            # Measure the time spent training the model
            start_time = time.time()
            grid_search.fit(x_train, y_train)
            end_time = time.time()

            # Get the best parameters and score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            # Train classifier with best parameters
            best_svm_classifier = grid_search.best_estimator_
            best_svm_classifier.fit(x_train, y_train)

            with open('svm_letters_classifier.p', 'wb') as f:
                pickle.dump({'model': best_svm_classifier}, f)

            training_time = end_time - start_time
            self.update_label(f"Model trained in: {training_time:.2f} seconds")
            self.first_click = False

            # Model evaluation
            y_predict = best_svm_classifier.predict(x_val)
            acc = accuracy_score(y_val, y_predict)
            prec = precision_score(y_val, y_predict, average="weighted")
            rec = recall_score(y_val, y_predict, average="weighted")
            f1 = f1_score(y_val, y_predict, average="weighted")

            metrics_text = f"""
            Validation metrics:
            Accuracy: {acc * 100:.2f}%
            Precision: {prec:.2f}
            Recall: {rec:.2f}
            F1-Score: {f1:.2f}
            """
            self.metrics_label.setText(metrics_text)

            # Calculate confusion matrix and classification report
            conf_matrix = confusion_matrix(y_val, y_predict)
            class_report = classification_report(y_val, y_predict)

            with open('classification_report.txt', 'w') as f:
                f.write(f'Best parameters: {best_params}\n\n')
                f.write(f'Best cross-validation score: {best_score}\n\n')
                f.write(f'Training time: {training_time:.2f} seconds\n\n')
                f.write(f'Validation Accuracy: {acc:.2f}\n\n')
                f.write(f'Confusion Matrix:\n{conf_matrix}\n\n')
                f.write(f'Classification Report:\n{class_report}\n')

            # Save results to self for later viewing
            self.y_val = y_val
            self.y_predict = y_predict
            self.x_test = x_val

        except Exception as e:
            self.update_label(f"Error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to train model: {e}")
            print(f"Error: {e}")

    def classification_report(self):
        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile('classification_report.txt'))

            labels = ['أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ',
                      'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'هـ', 'و', 'ي', 'لا', 'ة']

            cm = confusion_matrix(self.y_val, self.y_predict)
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix - Arabic Letters - SVM')
            plt.savefig('confusion_matrix_svm.png')
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open classification report: {e}")

