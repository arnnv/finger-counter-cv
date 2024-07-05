import time

import cv2
import mediapipe as mp
from mediapipe import solutions

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)
pTime = 0

while True:
    success, img = cap.read()

    if not success:
        break

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    result = hands.process(img)

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS,
                                   solutions.drawing_styles.get_default_hand_landmarks_style(),
                                   solutions.drawing_styles.get_default_hand_connections_style())

    cv2.imshow("Image", img)

    cv2.waitKey(1)
