import cv2
import pickle
import numpy as np
import mediapipe as mp

# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

# -------------------------------
# MEDIAPIPE HANDS SETUP
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# -------------------------------
# OPEN WEBCAM
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Webcam not accessible")
    exit()

# -------------------------------
# REAL-TIME INFERENCE LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        data_aux = []

        # Extract all 21 landmarks → 63 features
        for lm in hand_landmarks.landmark:
            data_aux.extend([lm.x, lm.y, lm.z])

        # ONLY predict if full hand is detected
        if len(data_aux) == 63:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_label = prediction[0]  # ← THIS IS 'A', 'B', 'C'

            # Confidence score
            if hasattr(model, "predict_proba"):
                confidence = np.max(model.predict_proba([np.asarray(data_aux)])) * 100
                text = f"{predicted_label} ({confidence:.1f}%)"
            else:
                text = predicted_label

            # Draw result
            cv2.putText(frame, text, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
        else:
            cv2.putText(frame, "Hand not fully detected", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Sign Language Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()
