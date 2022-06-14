import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, modelC=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands  # Whenever we use this module, this needs to be written
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelC, self.detectionCon, self.trackCon)
        self.mpdraw = mp.solutions.drawing_utils

    def findHands(self, image, draw=True):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_image)
        if self.results.multi_hand_landmarks:
            for eachhandlmks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(image, eachhandlmks, self.mpHands.HAND_CONNECTIONS)

        return image

    def findPosition(self, image, handNo=0, draw=True):

        landmarklist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                height, width, channel = image.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                #print(id, cx, cy)
                landmarklist.append([id, cx, cy])

                if draw:
                    cv2.circle(image, (cx, cy), 5, (0, 188, 255), cv2.FILLED)

        return landmarklist


def main():
    current_time = 0
    previous_time = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        ret, image = cap.read()
        image = detector.findHands(image)
        landmarklist = detector.findPosition(image)
        if len(landmarklist) != 0:
            print(landmarklist[4])

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 145, 78), 3)

        cv2.imshow("Webcam", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
