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

usage_text = '''
[INFO] Processing the divergence of the compressed motion photo...
'''

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False,
	help="path to output video file")
ap.add_argument("-i", "--input", required=False,
        help="path to input file")
args = vars(ap.parse_args())

''' This is the process of converting motion to cartesian coordinates

class DenseOpticalFlowByHSV(DenseOpticalFlow):
    def makeResult(self, grayFrame, flow):
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        self.hsv[...,0] = ang*180/np.pi/2
        self. hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
        '''

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
    
    finalArr = sumArray / sumIter #This will give the final equivalent motion vector of the system
    print(finalArr)    
