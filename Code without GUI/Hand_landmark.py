import os
import pickle
import mediapipe as mp
import cv2
from sklearnex import patch_sklearn
patch_sklearn()
from collections import defaultdict

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Directory containing data
DATA_DIR = './My_data'


# Initialize lists to store data and labels
data = []
labels = []

# Count number of images per class
class_count = defaultdict(int)

# Iterate over each directory in DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Couldn't read image {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process image with MediaPipe Hands
            with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) as hands:
                results = hands.process(img_rgb)
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

                    data.append(data_aux)
                    labels.append(dir_)  # Assuming dir_ is the label for images in each directory
                    class_count[dir_] += 1

# Determine the maximum number of images per class
max_images_per_class = max(class_count.values())

# Perform padding to balance classes
for class_label in class_count:
    while class_count[class_label] < max_images_per_class:
        # Choose a random image from the same class and add to data and labels
        for idx in range(len(labels)):
            if labels[idx] == class_label:
                data.append(data[idx])
                labels.append(class_label)
                class_count[class_label] += 1
                break

# Save data and labels to a pickle file
with open('dataset.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)