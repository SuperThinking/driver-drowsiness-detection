import math

import numpy as np
from matplotlib import pyplot as plt
import cv2
import dlib
import imutils
import vlc
from imutils import face_utils

def euc(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2-x1, 2)+math.pow(y2-y1, 2))

def insertObj(frame, glasses, l, r):
    m = -math.atan((r[0][1]/1.0-l[3][1]/1.0)/(l[3][0]/1.0-r[0][0]/1.0))*57.298
    by = 4.35*math.pow(10, -3)*euc(r[0][0],r[0][1],r[3][0],r[3][1])
    print(euc(r[0][0],r[0][1],r[3][0],r[3][1]))
    x_offset = (r[0][0]-int(by*122.0))
    y_offset = (r[0][1]-int(by*195.0))
    glasses = imutils.rotate_bound(glasses, m)
    glasses = cv2.resize(glasses, (0,0), fx=by, fy=by)
    y1, y2 = y_offset, y_offset + glasses.shape[0]
    x1, x2 = x_offset, x_offset + glasses.shape[1]

    alpha_s = glasses[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        frame[y1:y2, x1:x2, c] = (alpha_s * glasses[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
    return frame

capture = cv2.VideoCapture(0)

glasses = cv2.imread('./glasses.png', cv2.IMREAD_UNCHANGED)

detector = dlib.get_frontal_face_detector()
face_shape = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
left_eye = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
right_eye = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

while(True):
    ret, frame = capture.read()
    faces = detector(frame, 0)
    # cv2.rectangle(frame, (face[0][0][0], face[0][0][1]), (face[0][0][0]+face[0][0][0]))
    for face in faces:
        shape = face_utils.shape_to_np(face_shape(frame, face))
        leCoord = shape[left_eye[0]:left_eye[1]]
        reCoord = shape[right_eye[0]:right_eye[1]]
        try:
            frame = insertObj(frame, glasses, leCoord, reCoord)
        except:
            print('Face out of frame')
        break
    
    cv2.imshow('fun', frame)
    if(cv2.waitKey(1)==27):
        break
capture.release()
cv2.destroyAllWindows()
