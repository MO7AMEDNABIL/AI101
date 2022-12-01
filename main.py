import cv2 as cv
import numpy as np
import mediapipe as mp
import time


while True: 

    cam = np.zeros((500,800,3),dtype='uint8') 

    cv.imshow('cam',cam)

    cam[:]= 0,0,255

    cv.imshow('cam',cam)

    cv.putText(cam,"welcome to AI101!",(200,200), cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,255),2 )

    cv.imshow('cam',cam)

    if cv.waitKey(1) == ord('q'):
        break

                    
                    
src = cv.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


while True:

    p, img = src.read ()
    img = cv.flip (img, 1)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv.imshow ("img", img)

    if cv.waitKey (1) == ord ("e"):
        break







     