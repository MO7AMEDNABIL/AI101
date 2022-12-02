import math
import cv2 as cv
import mediapipe as mp
import numpy as np
import time
from tkinter import messagebox

__all__ = [mp, cv]


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
        imgRGB = img
        self.results = self.pose.process (imgRGB)

        if self.results.pose_landmarks:
            self.mpdraw.draw_landmarks (imgRGB, self.results.pose_landmarks, self.mppose.POSE_CONNECTIONS)

        return imgRGB

    def getposition (self, img, draws = True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate (self.results.pose_landmarks.landmark):
                h, w, c  = img.shape
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
        imgRGB = img
        self.results = self.hands.process (imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draws:
                    self.mpDraw.draw_landmarks (imgRGB, handLms, self.mpHands.HAND_CONNECTIONS)

        return imgRGB

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


class FaceDetection:
    def __init__ (self,
                  static_image_mode = False,
                  model_complexity = 1,
                  smooth_landmarks = True,
                  enable_segmentation = False,
                  smooth_segmentation = True,
                  refine_face_landmarks = False,
                  min_detection_confidence = 0.5,
                  min_tracking_confidence = 0.5
                  ):
        # # -----------------------utility-------------------------------
        # self.color1 = (0, 0, 0)
        # self.color2 = (0, 0, 0)
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.refine_face_landmarks = refine_face_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # self.color_detection = (245, 117, 66)
        # self.color_detection2 = (245, 66, 230)
        # self.circle_radius = 1

        self.mp_drawing = mp.solutions.drawing_utils  # drawing utility to help us gathering data
        self.mp_holistic = mp.solutions.holistic
        self.hollistic = self.mp_holistic.Holistic(
            self.static_image_mode,
            self.model_complexity,
            self.smooth_landmarks,
            self.enable_segmentation,
            self.smooth_segmentation,
            self.refine_face_landmarks,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )

    def findface (self, img, draws = True):


        imgRGB = img
        self.results = self.hollistic.process (imgRGB)

        if self.results.face_landmarks:
            if draws:
                self.mp_drawing.draw_landmarks (imgRGB, self.results.face_landmarks,self.mp_holistic.FACEMESH_TESSELATION,
                                                self.mp_drawing.DrawingSpec (None, thickness = 1, circle_radius = 1),
                                                self.mp_drawing.DrawingSpec (None, thickness = 1, circle_radius = 1)
                                                )

        return imgRGB
        # OLD PROCEDURAL CODE

        # self.boolean, frame = src_camera.read ()
        # # Changing the color to rgb
        # self.image = cv.cvtColor (frame, cv.COLOR_BGR2RGB)
        # self.image.flags.writeable = False
        #
        # # let's detect
        # self.result = holistic.process (self.image)
        #
        # # detecting face gizmos
        # self.image.flags.writeable = True
        # self.image = cv.cvtColor (self.image, cv.COLOR_RGB2BGR)  # rendering!
        #
        # self.mp_drawing.draw_landmarks (self.image, self.result.face_landmarks,
        #                                 self.mp_holistic.FACEMESH_TESSELATION,
        #                                 self.mp_drawing.DrawingSpec (None, thickness = 1,
        #                                                              circle_radius = self.circle_radius),
        #                                 self.mp_drawing.DrawingSpec (None, thickness = 1,
        #                                                              circle_radius = self.circle_radius)
        #                                 )

        # mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        # mp_drawing.DrawingSpec(color=color_detection, thickness=2, circle_radius=4),
        # mp_drawing.DrawingSpec(color=color_detection2, thickness=2, circle_radius=2))

        # print (self.result.face_landmarks)

        # btl3 el data and saving it
        # cv.imshow ("camera of mobile", self.image)
        #
        # landmarks = ['class']
        # for self.value in range (1, 34):
        #     landmarks += [f"x{self.result.face_landmarks}", f"y{self.value}", f"z{self.value}",
        #                   f"v{self.value}"]
        #
        # # opening file to save the data of landmarks
        # with open ("coords.csv", mode = "w", newline = "") as file:
        #     csv_writer = csv.writer (file, delimiter = ",", quotechar = '"',
        #                              quoting = csv.QUOTE_MINIMAL)  # creating data in excel sheet
        #     csv_writer.writerow (landmarks)  # typing the data

        # src_camera.release ()
        # cv.destroyAllWindows ()


def main (goal = 5, arm = "right"):
    pTime = 0
    cTime = time.time ()

# noinspection PyUnresolvedReferences

    src = cv.VideoCapture (0)
    face = FaceDetection()
    draw = handDetector ()
    body = armDetector ()
    count = 0
    up = False

    while True:
        p, img = src.read ()
        # img = cv.cvtColor (img, cv.COLOR_BGR2RGB)
        # img = face.findface(img)
        img = body.findbody(img)
        # img = draw.findhands(img)
        lmList = body.getposition (img, False)
        if len (lmList) != 0:
            biceps = Biceps ()
            if arm == "left":
                a = biceps.getangle (lmList[15], lmList[13], lmList[11])
                b = biceps.getangle (lmList[23], lmList[11], lmList[15])
            else:
                a = biceps.getangle (lmList[16], lmList[14], lmList[12])
                b = biceps.getangle (lmList[24], lmList[12], lmList[14])

            if a == 0:
                print ("[Error detecting hand]")
            elif a < 100 and not up and b < 50:
                count += 1
                up = True

            elif a > 150 and up:
                up = False

        # OLD PROCEDURAL CODE

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
        cv.putText (img, f"Count = {count}", (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv.LINE_AA)
        cv.imshow ("img", img)

        if cv.waitKey (1) == ord ("e") or goal == count:
            cv.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
