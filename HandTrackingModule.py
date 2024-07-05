import time

import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, static_img=False,
                 num_hands=2,
                 complexity=1,
                 detection_confidence=0.5,
                 tracking_confidence=0.5):
        self.static_img = static_img
        self.num_hands = num_hands
        self.complexity = complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(self.static_img, self.num_hands, self.complexity, self.detection_confidence,
                                         self.tracking_confidence)
        self.result = None

    def getHands(self, img):
        self.result = self.hands.process(img)

        if self.result.multi_hand_landmarks:
            for hand_lms in self.result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                                            mp.solutions.drawing_styles.get_default_hand_connections_style())
        return img

    def getLandmarks(self, img, hand_num=0):
        lm_list = []

        if self.result.multi_hand_landmarks:
            hand = self.result.multi_hand_landmarks[hand_num]
            for i, lm in enumerate(hand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([cx, cy])

        return lm_list


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0

    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.getHands(img)

        lm_list = detector.getLandmarks(img)
        if len(lm_list) != 0:
            print(lm_list)

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
