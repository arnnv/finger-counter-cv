import os
import time
import cv2
from modules.hand_tracking_module import HandDetector


def findFingerCount(img, lm_list):
    return img


IMG_FOLDER = "./img/"
IMAGE_NAME = os.listdir(IMG_FOLDER)
FINGER_IMAGES = []
for img_name in IMAGE_NAME:
    img_path = os.path.join(IMG_FOLDER, img_name)
    finger_img = cv2.imread(img_path)
    finger_img = cv2.resize(finger_img, (120, 120))
    FINGER_IMAGES.append(finger_img)

cap = cv2.VideoCapture(0)
pTime = 0

detector = HandDetector(detection_confidence=0.7)

while True:
    success, img = cap.read()
    if not success:
        break

    h, w, c = FINGER_IMAGES[0].shape
    img[0:h, 0:w] = FINGER_IMAGES[0]

    img = detector.getHands(img)
    lm_list = detector.getLandmarks(img)

    img = findFingerCount(img, lm_list)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (500, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Image", img)

    cv2.waitKey(1)
