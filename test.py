import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

img = cv2.imread('left-eye.jpg')

cv2.imshow('lala', img)

if(cv2.waitKey(0)==27):
    cv2.destroyAllWindows()