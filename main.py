import cv2 as cv
import numpy as np



cam = np.zeros((500,800,3),dtype='uint8') 

cv.imshow('cam',cam)

cam[:]= 0,0,255

cv.imshow('cam',cam)

cv.putText(cam,"welcome to AI101!",(200,200), cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,255),2 )

cv.imshow('cam',cam)





cv.waitKey(0) 

     