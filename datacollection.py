import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Start camera
capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder= 'Data/B'
counter=0

while True:
    success, img = capture.read()
    if not success:
        break

    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(img.shape[1], x + w + offset)
        y2 = min(img.shape[0], y + h + offset)

        imgCrop = img[y1:y2, x1:x2]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hGap + hCal, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        except Exception as e:
            print("Resize error:", e)
            continue

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) 
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
        print("Image saved!")

    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
