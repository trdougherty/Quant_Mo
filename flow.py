import cv2
import numpy as np

class Flow:
    def __init__(self, img):
        self.prev = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    def calc(self, img):
        current = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.flow = cv2.calcOpticalFlowFarneback(self.prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev = current
        return self.flow
