from __future__ import print_function
# Better samples/python2/opt_flow.py
# for Raspberry Pi

## reference
# - http://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/
# - http://stackoverflow.com/questions/2601194/displaying-a-webcam-feed-using-opencv-and-python
# - http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

import time
import cv2
from OpticalFlowShowcase import *
import numpy as np
#from imutils.video import VideoStream
#import imutils
import argparse
import io
import sys
import datetime
from matplotlib import pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False,
	help="path to output video file")
ap.add_argument("-i", "--input", required=False,
        help="path to input file")
args = vars(ap.parse_args())

def makeResult(self, grayFrame, flow):
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    self.hsv[...,0] = ang*180/np.pi/2
    self. hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

def main():
    t0 = time.time()
    name = args["input"]
    photo = cv2.imread(name)

def process(file):
    numObj = np.load(file)
    [date, iter, ar1, ar2] = numObj
    recompose = np.zeros([ar1.shape[0],ar1.shape[1],2])
    recompose[:,:,0] = ar1
    recompose[:,:,1] = ar2
    return [date, iter, recompose]

def localize(point, x, y, mv = 0.08):
    Z = 1/np.power(np.power(point[0]-x,2)+np.power(point[1]-y,2),0.5)
    Z[ Z > max_val ] = mv
    return Z*(1/max_val)

def divergence(arr):


def humanDate(ts):
    return datetime.datetime.fromtimestamp(ts)

def fileImport():
    imported = sys.stdin.read()
    return imported.rstrip().split('\n')

if __name__ == '__main__':
    files = fileImport()
    [sumIter, sumArray] = process(files[0])[1:] #This gives us our first iterations and sum array
    for i in files[1:]:
        [iter, arr] = process(i)[1:]
        sumIter += iter
        sumArray += arr

    x_dist = np.arange(0,sumArray.shape[0])
    y_dist = np.arange(0,sumArray.shape[1])

    xx, yy = np.meshgrid(x_dist, y_dist)
    print(xx)
    # print(distance(3,10,sumArray))
    #distArr = np.zeros(sumArray.shape[0],sumArray.shape[1])
    #for i in sumArray.shape[0]:
    #    for j in sumArray.shape[1]:
    #        distArr[i][j] =

    #finalArr = sumArray / sumIter #This will give the final equivalent motion vector of the system
