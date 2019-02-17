import cv2
import numpy as np
from matplotlib import pyplot as plt

capture = cv2.VideoCapture(0)
face_cascade = [cv2.CascadeClassifier('haarcascade_frontalface_default.xml'), cv2.CascadeClassifier('haarcascade_frontalface_alt.xml'), cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')]
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
s = 255
lower_blue = np.array([100,0,255-s])
upper_blue = np.array([200,s,255])
# cv2.inRange(hsv, lower_blue, upper_blue)

while(True):
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for cascade in face_cascade:
        face_frame = cascade.detectMultiScale(gray, 1.4, 5)
        if(len(face_frame)):
            a, b, c, d = face_frame[0]
            cv2.rectangle(gray, (a, b), (a+c, b+d), (0, 0, 0), 3)
            for_eye = gray[b:b+c, a:a+c]
            eyes = eye_cascade.detectMultiScale(for_eye)
            if(len(eyes)==2): #2 eyes
                print(eyes)
                x1, y1, x2, y2 = eyes[0]
                x3, y3, x4, y4 = eyes[1]
                cv2.rectangle(for_eye, (x1, y1), (x1+x2, y1+y2), (0, 0, 0), 3)
                cv2.rectangle(for_eye, (x3, y3), (x3+x4, y3+y4), (0, 0, 0), 3)
                cv2.imwrite('left-eye.jpg', for_eye[y1:y1+y2, x1:x1+x2])
                cv2.imwrite('right-eye.jpg', for_eye[y3:y3+y4, x3:x3+x4])
            break
    cv2.imshow('Driver', gray)
    if(cv2.waitKey(1)==27):
        break
        
capture.release()
cv2.destroyAllWindows()