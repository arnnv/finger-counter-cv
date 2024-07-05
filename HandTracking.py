import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0

    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = detector.detect(imgRGB)
        print(result)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Image", img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
