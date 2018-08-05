import numpy as np
import cv2


photo = cv2.imread("red_box.jpg")
print(photo.shape)
print(photo[...,3])
