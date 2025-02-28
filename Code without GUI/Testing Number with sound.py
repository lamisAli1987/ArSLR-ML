import os
import pickle
import mediapipe as mp
import cv2
import pygame
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearnex import patch_sklearn
patch_sklearn()
# Initialize pygame mixer
pygame.mixer.init()


with open('svm_number_classifier.p', 'rb') as f:
    svm_letters = pickle.load(f)['model']

# Initialize MediaPipe Hands for video capture
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)  # Initialize video capture
labels_dict = { 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10'}


def convert_to_arabic_numerals(self, number):
    english_to_arabic = {'0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤', '5': '٥', '6': '٦', '7': '٧', '8': '٨',
                         '9': '٩'}
    return ''.join(english_to_arabic[digit] if digit in english_to_arabic else digit for digit in str(number))
# Function to draw text with PIL for Arabic support
def draw_text_with_pil(image, text, position, font_path='arial.ttf', font_size=50, accuracy=None):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, font_size)
    arabic_char = labels_dict[int(text)] if isinstance(text, (int, np.integer)) else text
    arabic_position = position
    draw.text(arabic_position, arabic_char, font=font, fill=(255, 0, 0, 255))  # Red color for Arabic character
    accuracy_position = (position[0] + font_size + 5, position[1])  # Adjust position as needed
    if accuracy is not None:
        draw.text(accuracy_position, f'{accuracy:.2f}', font=font, fill=(255, 0, 0, 255))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# To avoid playing the sound multiple times for the same letter
previous_predicted_char = None

# Read frames from the video capture
while True:
    ret, frame = cap.read()  # Read a frame
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

            # Predict using trained SVM classifier
            prediction_svm = svm_letters.predict([data_aux])
            predicted_character_svm = labels_dict[int(prediction_svm[0])]
            predicted_character_svm_arabic = convert_to_arabic_numerals(predicted_character_svm)

            if predicted_character_svm != previous_predicted_char:
                previous_predicted_char = predicted_character_svm
                # Load and play audio based on the prediction
                audio_file_path = f"C:/Users/hp/PycharmProjects/ArSL_New_Program/Sound/{predicted_character_svm}.mp3"

                if os.path.exists(audio_file_path):
                    pygame.mixer.music.load(audio_file_path)
                    pygame.mixer.music.play()

            # Display predicted character on frame using draw_text_with_pil function
            frame = draw_text_with_pil(frame, predicted_character_svm, (10, 30), font_size=50,
                                       accuracy=svm_letters.predict_proba([data_aux])[0][int(prediction_svm[0])])

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Stop the mixer when done
pygame.mixer.music.stop()
