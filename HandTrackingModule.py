import time

import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_img = static_image_mode
        self.num_hands = max_num_hands
        self.complexity = model_complexity
        self.detection_confidence = min_detection_confidence
        self.tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_img, num_hands, complexity, detection_confidence, tracking_confidence)

    def getHands(self):
        print(self.hands)


def main():
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
        cv2.imshow("Image", img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
