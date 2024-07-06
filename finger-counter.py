import os
import time
import cv2
from modules.hand_tracking_module import HandDetector


def findFingerCount(lm_list):
    finger_tips = [4, 8, 12, 16, 20]

    open_map = []

    for tip in finger_tips:
        if tip == 4:
            if lm_list[tip][0] < lm_list[tip - 1][0]:

                open_map.append(1)
            else:
                open_map.append(0)
            continue

        if lm_list[tip][1] < lm_list[tip - 2][1]:
            open_map.append(1)
        else:
            open_map.append(0)

    return open_map


def displayFinger(img, finger_count, finger_images):
    # cv2.putText(img, f"Finger Count: {finger_count}", (120, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.rectangle(img, (0, 120), (120, 120+120), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f"{finger_count}", (0, 245), cv2.FONT_HERSHEY_PLAIN, 11.5, (255, 0, 0), 7)

    h, w, c = finger_images[finger_count].shape
    img[0:h, 0:w] = finger_images[finger_count]

    return img


def main():
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

        img = cv2.flip(img, 1)
        img = detector.getHands(img)
        lm_list = detector.getLandmarks(img)

        finger_count = 0

        if len(lm_list) != 0:
            fingers_open = findFingerCount(lm_list)
            finger_count = sum(fingers_open)

        img = displayFinger(img, finger_count, FINGER_IMAGES)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (500, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
