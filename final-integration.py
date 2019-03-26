import cv2
import math
import numpy as np
import dlib
import imutils
from imutils import face_utils
from matplotlib import pyplot as plt
import vlc
import train as train

def getFaceDirection(shape, size):
    image_points = np.array([
                                shape[33], #(359, 391),     # Nose tip
                                shape[8],# (399, 561),     # Chin
                                shape[45],# (337, 297),     # Left eye left corner
                                shape[36],# (513, 301),     # Right eye right corne
                                shape[54],# (345, 465),     # Left Mouth corner
                                shape[48]# (453, 469)      # Right mouth corner
                            ], dtype="double")
    
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            
                            ])
    
    # Camera internals
    
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
        
        # print "Camera Matrix :\n {0}".format(camera_matrix)
        
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        # print "Rotation Vector:\n {0}".format(rotation_vector)
        # print "Translation Vector:\n {0}".format(translation_vector)
    # print(translation_vector[1])
    return(translation_vector[1][0])
        
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
 
    # (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        
    # for p in image_points:
    #     cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
        
        
    # p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    # p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        
    # cv2.line(im, p1, p2, (255,0,0), 2)
def euclideanDist(a, b):
    return (math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2)))
#EAR -> Eye Aspect ratio
def ear(eye):
    return ((euclideanDist(eye[1], eye[5])+euclideanDist(eye[2], eye[4]))/(2*euclideanDist(eye[0], eye[3])))
def writeEyes(a, b, img):
    y1 = max(a[1][1], a[2][1])
    y2 = min(a[4][1], a[5][1])
    x1 = a[0][0]
    x2 = a[3][0]
    cv2.imwrite('left-eye.jpg', img[y1:y2, x1:x2])
    y1 = max(b[1][1], b[2][1])
    y2 = min(b[4][1], b[5][1])
    x1 = b[0][0]
    x2 = b[3][0]
    cv2.imwrite('right-eye.jpg', img[y1:y2, x1:x2])
# open_avg = train.getAvg()
# close_avg = train.getAvg()

alert = vlc.MediaPlayer('alert-sound.mp3')
frame_thresh = 15
close_thresh = 0.3#(close_avg+open_avg)/2.0
flag = 0

print(close_thresh)

capture = cv2.VideoCapture(0)
avgEAR = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while(True):
    ret, frame = capture.read()
    size = frame.shape
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame
    rects = detector(gray, 0)
    if(len(rects)):
        shape = face_utils.shape_to_np(predictor(gray, rects[0]))
        leftEye = shape[leStart:leEnd]
        rightEye = shape[reStart:reEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        leftEAR = ear(leftEye) #Get the left eye aspect ratio
        rightEAR = ear(rightEye) #Get the right eye aspect ratio
        avgEAR = (leftEAR+rightEAR)/2.0
        eyeContourColor = (255, 255, 255)
        if(avgEAR<close_thresh):
            flag+=1
            eyeContourColor = (34,34,178)
            print(flag)
            if(flag>=frame_thresh and getFaceDirection(shape, size)<0):
                eyeContourColor = (0, 0, 255)
                alert.play()
        elif(avgEAR>close_thresh and flag):
            print("Flag reseted to 0")
            alert.stop()
            flag=0
        cv2.drawContours(gray, [leftEyeHull], -1, eyeContourColor, 2)
        cv2.drawContours(gray, [rightEyeHull], -1, eyeContourColor, 2)
        writeEyes(leftEye, rightEye, frame)
    if(avgEAR>close_thresh):
        alert.stop()
    cv2.imshow('Driver', gray)
    if(cv2.waitKey(1)==27):
        break
        
capture.release()
cv2.destroyAllWindows()