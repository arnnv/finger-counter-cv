import time
import cv2
from modules.hand_tracking_module import HandDetector


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0

    detector = HandDetector()

    while True:
        success, img = cap.read()

        if not success:
            break

        img = detector.getHands(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Image", img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
