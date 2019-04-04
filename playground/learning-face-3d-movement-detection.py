import cv2
import numpy as np
import dlib
from imutils import face_utils

#Take Image
capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
while(True):
    ret, im = capture.read()
    # im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = im.shape
    rects = detector(im, 0)
    try:
        shape = face_utils.shape_to_np(predictor(im, rects[0]))
        #2D image points. If you change the image, you need to change vector
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
        
        print "Rotation Vector:\n {0}".format(rotation_vector)
        print "Translation Vector:\n {0}".format(translation_vector)
        
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        
        
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        
        for p in image_points:
            cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
        
        
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        
        cv2.line(im, p1, p2, (255,0,0), 2)
    except:
        print('Face not detected');
    # Display image
    cv2.imshow("Output", im)
    if(cv2.waitKey(1)==27):
            break

# Read Image
# im = cv2.imread("headpose.jpg")
# size = im.shape

# detector = dlib.get_frontal_face_detector()
# rects = detector(im, 0)
# predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
# (leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# shape = face_utils.shape_to_np(predictor(im, rects[0]))
# leftEye = shape[leStart:leEnd]
# rightEye = shape[reStart:reEnd]
# leftEyeHull = cv2.convexHull(leftEye)
# rightEyeHull = cv2.convexHull(rightEye)
# cv2.drawContours(im, [leftEyeHull], -1, (255, 255, 255), 1)
# # cv2.drawContours(im, [rightEyeHull], -1, (255, 255, 255), 1)
# print(rightEye, shape[3])
     
# #2D image points. If you change the image, you need to change vector
# image_points = np.array([
#                             shape[33], #(359, 391),     # Nose tip
#                             shape[8],# (399, 561),     # Chin
#                             shape[45],# (337, 297),     # Left eye left corner
#                             shape[36],# (513, 301),     # Right eye right corne
#                             shape[54],# (345, 465),     # Left Mouth corner
#                             shape[48]# (453, 469)      # Right mouth corner
#                         ], dtype="double")
 
# # 3D model points.
# model_points = np.array([
#                             (0.0, 0.0, 0.0),             # Nose tip
#                             (0.0, -330.0, -65.0),        # Chin
#                             (-225.0, 170.0, -135.0),     # Left eye left corner
#                             (225.0, 170.0, -135.0),      # Right eye right corne
#                             (-150.0, -150.0, -125.0),    # Left Mouth corner
#                             (150.0, -150.0, -125.0)      # Right mouth corner
                         
#                         ])
 
 
# # Camera internals
 
# focal_length = size[1]
# center = (size[1]/2, size[0]/2)
# camera_matrix = np.array(
#                          [[focal_length, 0, center[0]],
#                          [0, focal_length, center[1]],
#                          [0, 0, 1]], dtype = "double"
#                          )
 
# print "Camera Matrix :\n {0}".format(camera_matrix)
 
# dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
# (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
 
# print "Rotation Vector:\n {0}".format(rotation_vector)
# print "Translation Vector:\n {0}".format(translation_vector)
 
 
# # Project a 3D point (0, 0, 1000.0) onto the image plane.
# # We use this to draw a line sticking out of the nose
 
 
# (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
 
# for p in image_points:
#     cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
 
 
# p1 = ( int(image_points[0][0]), int(image_points[0][1]))
# p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
 
# cv2.line(im, p1, p2, (255,0,0), 2)
 
# # Display image
# cv2.imshow("Output", im)
# cv2.waitKey(0)