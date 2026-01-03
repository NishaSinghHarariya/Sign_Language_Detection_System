import os
import cv2
import pickle
import mediapipe as mp

# Path to your dataset
DATA_DIR = './data'  # this should contain folders A, B, C, ... Z

# Limit images per class for faster testing
MAX_IMAGES_PER_CLASS = 200  # change this number for more/less images

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

data = []
labels = []

for dir_ in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_path):
        continue

    print(f"Processing class: {dir_}")

    image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))]
    count = 0

    for idx, img_file in enumerate(image_files):
        if count >= MAX_IMAGES_PER_CLASS:
            break

        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            # Extract landmarks
            landmarks = []
            for hand_landmarks in result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            data.append(landmarks)
            labels.append(dir_)
            count += 1

        if idx % 500 == 0:
            print(f"Processed {idx} images in class {dir_}")

print(f"Total samples: {len(data)}")

# Save to pickle
with open('data.pickle', 'wb') as f:
    pickle.dump((data, labels), f)

print("Dataset creation complete!")
