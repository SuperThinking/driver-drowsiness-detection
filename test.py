import cv2
import numpy as np
from matplotlib import pyplot as plt

capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
s = 255
lower_blue = np.array([100,0,255-s])
upper_blue = np.array([200,s,255])
# cv2.inRange(hsv, lower_blue, upper_blue)

while(True):
    ret, frame = capture.read()
    mod_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # mod_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    face_frame = face_cascade.detectMultiScale(frame, 1.4, 5)
    # mask = cv2.inRange(frame, lower_blue, upper_blue)
    # frame = cv2.bitwise_and(frame, frame, mask=mask)
    if(len(face_frame)):
        l, m, n, o = face_frame[0]
        # frame = frame[m:m+o, l:l+n]
        eye_frame = eye_cascade.detectMultiScale(frame)
        if(len(eye_frame)==2):
            a, b, c, d = eye_frame[0]
            w, x, y, z = eye_frame[1]
            cv2.rectangle(mod_frame, (a, b), (a+c, b+d), (255, 0, 255), 2)
            cv2.rectangle(mod_frame, (w, x), (w+y, x+z), (255, 255, 255), 2)

    cv2.imshow('frame', mod_frame)

    if(cv2.waitKey(1)==27):
        break
        
capture.release()
cv2.destroyAllWindows()