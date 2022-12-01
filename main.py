import math
import cv2 as cv
import mediapipe as mp
import numpy as np
import time


class armDetector:
    def __init__ (self,
                  static_image_mode = False,
                  model_complexity = 1,
                  smooth_landmarks = True,
                  enable_segmentation = False,
                  smooth_segmentation = True,
                  min_detection_confidence = 0.5,
                  min_tracking_confidence = 0.5
                  ):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpdraw = mp.solutions.drawing_utils
        self.mppose = mp.solutions.pose
        self.pose = self.mppose.Pose (self.static_image_mode,
                                      self.model_complexity,
                                      self.smooth_landmarks,
                                      self.enable_segmentation,
                                      self.smooth_segmentation,
                                      self.min_detection_confidence,
                                      self.min_tracking_confidence
                                      )

    def findbody (self, img, draws = True):
        imgRGB = cv.cvtColor (img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process (imgRGB)

        if self.results.pose_landmarks:
            self.mpdraw.draw_landmarks (img, self.results.pose_landmarks, self.mppose.POSE_CONNECTIONS)

        return img

    def getposition (self, img, draws = True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate (self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int (lm.x * w), int (lm.y * h)
                lmList.append ([id, cx, cy])
                if draws:
                    cv.circle (img, (cx, cy), 10, (255, 0, 0), cv.FILLED)

        return lmList


class handDetector:

    def __init__ (self, mode = False, maxHands = 2, model_complexity = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands (self.mode, self.maxHands, self.model_complexity,
                                         self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findhands (self, img, draws = True):
        imgRGB = cv.cvtColor (img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process (imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draws:
                    self.mpDraw.draw_landmarks (img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def getposition (self, img, hand_no = 0, draws = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]

            for id, lm in enumerate (myHand.landmark):
                h, w, c = img.shape
                cx, cy = int (lm.x * w), int (lm.y * h)
                lmList.append ([id, cx, cy])
                if draws:
                    cv.circle (img, (cx, cy), 10, (255, 0, 0), cv.FILLED)

        return lmList


class Biceps:

    def getangle (self, l1, l2, l3):

        d1 = self.finddistance (l2[1], l3[1], l2[2], l3[2])
        d2 = self.finddistance (l1[1], l2[1], l1[2], l2[2])
        d3 = self.finddistance (l3[1], l1[1], l3[2], l1[2])
        #
        return self.findangle (d1, d2, d3)

    def finddistance (self, x1, x2, y1, y2):
        return math.sqrt (math.pow (x2 - x1, 2) + math.pow (y2 - y1, 2))

    def findangle (self, d1, d2, d3):
        p1 = d1 * d1
        p2 = d2 * d2
        p3 = d3 * d3
        if (2 * d1 * d2) == 0:
            return 0
        if ((p1 + p2 - p3) / (2 * d1 * d2)) > 1:
            return 0
        return math.degrees (math.acos ((p1 + p2 - p3) / (2 * d1 * d2)))


def main ():
    pTime = 0
    cTime = time.time ()

    src = cv.VideoCapture (0)
    draw = handDetector ()
    body = armDetector ()
    count = 0
    up = False
    while True:
        p, img = src.read ()
        img = body.findbody (img)
        lmList = body.getposition (img)
        if len (lmList) != 0:
            biceps = Biceps ()
            a = biceps.getangle (lmList[16], lmList[14], lmList[12])
            if a == 0:
                print ("[Error detecting hand]")
            elif a < 100 and not up:
                count += 1
                up = True
                print (count)

            elif a > 150 and up:
                up = False

        # img = draw.findhands(img)
        # lmList = draw.getposition(img, 0, False)
        # if len(lmList) != 0:
        #     biceps = Biceps()
        #     a = biceps.getangle(lmList[4], lmList[3], lmList[2])
        #     if a == 0:
        #         print("[Error detecting hand]")
        #     elif a < 100 and not up:
        #         count += 1
        #         up = True
        #         print(count)
        #
        #     elif a > 150 and up:
        #         up = False

        # This code detects fps:
        # fps = 1 / (cTime - pTime)
        # pTime = cTime
        # cTime = time.time ()
        # cv.putText (img, str (int (fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv.imshow ("img", img)

        if cv.waitKey (1) == ord ("e"):
            break


if __name__ == "__main__":
    main ()
