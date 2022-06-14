import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands  # Whenever we use this module, this needs to be written
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils  # draw lines between 21 hand_points detected...provided by mediapipe

current_time = 0
previous_time = 0
while True:
    ret, image = cap.read()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    # print(results.multi_hand_landmarks).... prints value if hand is detected
    if results.multi_hand_landmarks:
        for eachhandlmks in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(image, eachhandlmks, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(eachhandlmks.landmark):
                # print(id, lm)
                height, width, channel = image.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                print(id, cx, cy)


    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 145, 78), 3)

    cv2.imshow("Image", image)
    cv2.waitKey(1)
