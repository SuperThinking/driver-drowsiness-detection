import cv2
import math
import numpy as np
import dlib
import imutils
from imutils import face_utils

def euclideanDist(a, b):
    return (math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2)))
def ear(eye):
    return ((euclideanDist(eye[1], eye[5])+euclideanDist(eye[2], eye[4]))/(2*euclideanDist(eye[0], eye[3])))

def getAvg():
    capture = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    (leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    count = 0.0
    sum = 0
    while(True):
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        if(len(rects)):
            shape = face_utils.shape_to_np(predictor(gray, rects[0]))
            leftEye = shape[leStart:leEnd]
            rightEye = shape[reStart:reEnd]
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            leftEAR = ear(leftEye) #Get the left eye aspect ratio
            rightEAR = ear(rightEye) #Get the right eye aspect ratio
            sum+=(leftEAR+rightEAR)/2.0
            count+=1.0
            cv2.drawContours(gray, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(gray, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.imshow('Train', gray)
        if(cv2.waitKey(1)==27):
            break
    
    capture.release()
    cv2.destroyAllWindows()
    return sum/count