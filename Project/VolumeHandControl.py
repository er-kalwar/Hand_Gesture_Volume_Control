import cv2
import mediapipe
import numpy as np
import HandTrackingModule as htm
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


pTime = 0
cTime = 0
camWidth, camHeight = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0, None)
minVolume = volRange[0]
maxVolume = volRange[1]



while True:
    ret, image = cap.read()
    image = detector.findHands(image)
    lmlist = detector.findPosition(image, draw=False)
    if len(lmlist) != 0:
        #print(lmlist[4], lmlist[8])
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cv2.circle(image, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(image, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(image, (x1, y1), (x2, y2), (258, 0, 0), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot((x2-x1),(y2-y1))
        print(length)


        #Hand range - 50, 350
        #Vol range = -65 -0

        vol = np.interp(length,[50,300], [minVolume, maxVolume])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(image, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(image, str(int(fps)), (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow("Webcam", image)
    cv2.waitKey(1)
