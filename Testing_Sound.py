import os
import pickle
import mediapipe as mp
import cv2
from sklearnex import patch_sklearn
import pygame # To add audio support
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Initialize pygame mixer
pygame.mixer.init()

patch_sklearn()

# Load trained model SVM
with open('svm_newdata_classifier.p', 'rb') as f:
    svm_letters = pickle.load(f)['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)  # Initialize video capture
labels_dict = {0: 'أ', 1: 'ب', 2: 'ة', 3: 'ت', 4: 'ث', 5: 'ج', 6: 'ح', 7: 'خ', 8: 'د', 9: 'ذ', 10: 'ر', 11: 'ز',
               12: 'س', 13: 'ش', 14: 'ص', 15: 'ض', 16: 'ط', 17: 'ظ', 18: 'ع', 19: 'غ', 20: 'ف', 21: 'ق', 22: 'ك',
               23: 'ل', 24: 'لا', 25: 'م', 26: 'ن', 27: 'هـ', 28: 'و', 29: 'ي', 30: 'ال'}

# Function to draw text with PIL for Arabic support
def draw_text_with_pil(image, text, position, font_path='arial.ttf', font_size=90):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, font_size)
    arabic_char = labels_dict[int(text)] if isinstance(text, (int, np.integer)) else text
    draw.text(position, arabic_char, font=font, fill=(255, 255, 255, 255))  # White color
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

#To avoid playing the sound multiple times for the same letter
previous_predicted_char = None
current_display_char = None  # To store the character currently displayed on the screen

# Function to analyze the frame using SVM
def svm_predict(frame):
    # Convert frame to RGB and process with MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

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

            # letter prediction using SVM
            prediction_svm = svm_letters.predict([data_aux])
            return labels_dict[int(prediction_svm[0])], results
    return None, None

# Read frames from the video capture
while True:
    ret, frame = cap.read()  # Read a frame
    if not ret:
        break

    # Extract the letter using SVM and plot the points.
    predicted_character_svm, results = svm_predict(frame)

    # If a new character is predicted, the displayed character is updated and the sound is played.
    if predicted_character_svm is not None and predicted_character_svm != previous_predicted_char:
        previous_predicted_char = predicted_character_svm  # Update previous letter
        current_display_char = predicted_character_svm  # Update the currently displayed letter.

        # Load and play audio based on the prediction
        audio_file_path = f"C:/Users/hp/PycharmProjects/ArSL_New_Program/Sound/{predicted_character_svm}.mp3"

        if os.path.exists(audio_file_path):
            pygame.mixer.music.load(audio_file_path)
            pygame.mixer.music.play()

    # If there are MediaPipe results, draw the dots on the hand.
    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # If a letter is currently displayed, keep it displayed on the screen.
    if current_display_char is not None:
        frame = draw_text_with_pil(frame, current_display_char, (80, 80), font_size=90)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Stop the mixer when done
pygame.mixer.music.stop()
